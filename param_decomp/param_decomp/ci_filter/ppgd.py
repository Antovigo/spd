"""The decomposition's merged stochastic-subset persistent-PGD reconstruction (SPEC S34, S13′,
S14′, S24), scored by the CI filter's objective (`MergedPPGDRecon`).

Built from core's own kernels — `init_persistent_sources`, `sources_adam_ascend_project`,
`UniformKRouting`, `source_masking`, `mixed_persistent_stochastic_masking` — with the filter's
frozen components (only the CI fn trains). The persistent sources hold one row per batch slot
(the axis after each stack's stack axis); microbatches read consecutive row slices, and every
Adam ascent updates the whole bundle once from the concatenated slice gradients, so the
trajectory does not depend on the microbatch split beyond float reassociation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int, PRNGKeyArray

from param_decomp.ci_filter.config import CIFilterConfig, MergedPPGDRecon
from param_decomp.ci_filter.objective import objective_rows
from param_decomp.ci_filter.step import (
    CIConstraints,
    FilterCIFn,
    MicroCall,
    Prepared,
    Trainable,
    batch_gradients,
    count_above,
    gather_batch,
    last_logits,
    masked_output,
    output_ci,
    trainable,
)
from param_decomp.core.adversary import (
    SourcesAdamState,
    SourceStacks,
    init_persistent_sources,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.core.losses import importance_minimality_terms
from param_decomp.core.masking import (
    materialize_masking,
    mixed_persistent_stochastic_masking,
    source_masking,
)
from param_decomp.core.recon import UniformKRouting
from param_decomp.core.runtime_schedule import scheduled_value_at
from param_decomp.experiments.lm.load_run import PlacedTransformer

ROW_AXIS = 1
"""The batch-slot axis of every source stack leaf (`[stack, batch, ...]`)."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Adversary:
    sources: SourceStacks
    opt_state: SourcesAdamState


def _ppgd(config: CIFilterConfig) -> MergedPPGDRecon:
    assert isinstance(config.recon, MergedPPGDRecon), config.recon
    return config.recon


def init_adversary(
    placed: PlacedTransformer, batch_size: int, seq_len: int, key: PRNGKeyArray
) -> Adversary:
    """U[0,1] fp32 component sources `(batch, position, C)` and delta sources
    `(batch, position)` per site, stacked per group and replicated (the microbatch row slices
    are host-side indexing; a batch-sharded bundle would make each slice a cross-device
    gather)."""
    sources = init_persistent_sources(placed.model.sites, (batch_size, seq_len), jnp.float32, key)
    if not jax.sharding.get_abstract_mesh().empty:
        sources = jax.tree.map(lambda x: jax.sharding.reshard(x, P()), sources)
    return Adversary(sources=sources, opt_state=init_sources_adam_state(sources))


def _rows(sources: SourceStacks, start: int, stop: int) -> SourceStacks:
    return jax.tree.map(lambda x: jax.lax.slice_in_dim(x, start, stop, axis=ROW_AXIS), sources)


def _concat_rows(parts: list[SourceStacks]) -> SourceStacks:
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=ROW_AXIS), *parts)


type WarmupGrads = Callable[
    [
        PlacedTransformer,
        Prepared,
        FilterCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
        SourceStacks,
    ],
    SourceStacks,
]


def make_warmup_grads(config: CIFilterConfig) -> WarmupGrads:
    """`d(objective)/d(sources)` of one microbatch slice under the ALL-adversarial, route-all
    forward with the CI detached (S24), scaled by `1 / n_microbatches` so the slices sum to the
    batch mean's gradient."""
    objective = config.objective
    scale = 1.0 / config.n_microbatches

    @eqx.filter_jit
    def warmup_grads(
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
        sources: SourceStacks,
    ) -> SourceStacks:
        target_idx = targets[idx]
        clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        clean_last = jax.lax.stop_gradient(last_logits(clean.output))
        lower = jax.lax.stop_gradient(
            output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
        )

        def score(sources: SourceStacks) -> Array:
            masking = materialize_masking(source_masking(lower, sources.per_site()))
            masked_last = last_logits(
                masked_output(placed, prepared, clean.conditioning, masking, remat=config.remat)
            )
            return scale * jnp.mean(
                objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
            )

        return jax.grad(score)(sources)

    return warmup_grads


type PPGDMicroGrads = Callable[
    [
        PlacedTransformer,
        Prepared,
        FilterCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
        Array,
        SourceStacks,
        PRNGKeyArray,
    ],
    tuple[Trainable, SourceStacks, dict[str, Array], dict[str, Array]],
]


def make_ppgd_micro_grads(config: CIFilterConfig) -> PPGDMicroGrads:
    """One microbatch of the merged term plus imp-min: the CI fn's gradient, and the persistent
    sources' gradient of the UNSCALED recon term (S14′: the source path never carries `coeff`),
    both `1 / n_microbatches`-scaled; metrics likewise."""
    objective = config.objective
    imp = config.imp_min
    recon = _ppgd(config)
    scale = 1.0 / config.n_microbatches

    @eqx.filter_jit
    def micro_grads(
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
        train_frac: Array,
        sources: SourceStacks,
        key: PRNGKeyArray,
    ) -> tuple[Trainable, SourceStacks, dict[str, Array], dict[str, Array]]:
        target_idx = targets[idx]
        clean = jax.lax.stop_gradient(
            placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        )
        clean_last = last_logits(clean.output)
        leading = clean.leading_shape
        gamma = scheduled_value_at(train_frac, imp.gamma)
        impmin_coeff = scheduled_value_at(train_frac, imp.coeff)
        freq_coeff = (
            jnp.zeros((), jnp.float32)
            if imp.frequency_coeff is None
            else scheduled_value_at(train_frac, imp.frequency_coeff)
        )
        adv_fraction = scheduled_value_at(train_frac, recon.adv_fraction)
        route_key, mask_key = random.split(key)
        routes = UniformKRouting(placed.model.site_names)(route_key, leading)

        def loss_fn(params: Trainable, sources: SourceStacks) -> tuple[Array, dict[str, Array]]:
            ci = output_ci(
                placed,
                prepared,
                cast(FilterCIFn, eqx.combine(params, ci_fn)),
                constraints,
                clean,
                config.remat,
            )
            masking, merged_routes = mixed_persistent_stochastic_masking(
                mask_key, ci.lower, sources.per_site(), leading, adv_fraction, routes
            )
            masked_last = last_logits(
                masked_output(
                    placed,
                    prepared,
                    clean.conditioning,
                    materialize_masking(masking),
                    remat=config.remat,
                    routes=merged_routes,
                )
            )
            recon_value = jnp.mean(
                objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
            )
            activity, freq = importance_minimality_terms(
                ci.upper,
                gamma,
                None if imp.frequency_coeff is None else imp.reference_datapoint_count,
                normalize_at_one=imp.normalize_at_one,
            )
            total = recon.coeff * recon_value + impmin_coeff * activity + freq_coeff * freq
            metrics = {
                "loss": total,
                "recon": recon_value,
                "imp_activity": activity,
                "imp_freq": freq,
                "l0": count_above(ci.lower, 0.0),
                "l0_alive": count_above(ci.lower, config.alive_threshold),
            }
            return scale * total, metrics

        (param_grads, source_grads), metrics = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(
            trainable(ci_fn), sources
        )
        source_grads = jax.tree.map(lambda g: g / recon.coeff, source_grads)
        metrics = {k: scale * v for k, v in metrics.items()}
        schedules = {
            "gamma": gamma,
            "impmin_coeff": impmin_coeff,
            "freq_coeff": freq_coeff,
            "adv_fraction": adv_fraction,
            "source_lr": scheduled_value_at(train_frac, recon.optimizer.lr_schedule),
        }
        return cast(Trainable, param_grads), source_grads, metrics, schedules

    return micro_grads


def make_ascend(config: CIFilterConfig) -> Callable[[Adversary, SourceStacks, Array], Adversary]:
    """One Adam ascent-and-project of the whole bundle (S13′/S15)."""
    optimizer = _ppgd(config).optimizer

    @eqx.filter_jit
    def ascend(adversary: Adversary, grads: SourceStacks, train_frac: Array) -> Adversary:
        lr = scheduled_value_at(train_frac, optimizer.lr_schedule)
        sources, opt_state = sources_adam_ascend_project(
            adversary.sources, grads, adversary.opt_state, lr, optimizer
        )
        return Adversary(sources=sources, opt_state=opt_state)

    return ascend


@dataclass(frozen=True)
class PPGDStep:
    warmup_grads: WarmupGrads
    micro_grads: PPGDMicroGrads
    ascend: Callable[[Adversary, SourceStacks, Array], Adversary]
    n_warmup: int

    @staticmethod
    def build(config: CIFilterConfig) -> "PPGDStep":
        return PPGDStep(
            warmup_grads=make_warmup_grads(config),
            micro_grads=make_ppgd_micro_grads(config),
            ascend=make_ascend(config),
            n_warmup=_ppgd(config).n_warmup_steps,
        )

    def __call__(
        self,
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: np.ndarray,
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
        train_frac: Array,
        adversary: Adversary,
        key: PRNGKeyArray,
        microbatch_size: int,
        on_host: bool,
    ) -> tuple[Trainable, dict[str, Array], dict[str, Array], Adversary]:
        """Warmup ascents, the main gradient over the microbatches, then the final ascent from
        that same backward's source gradients."""
        starts = list(range(0, len(idx), microbatch_size))
        for _ in range(self.n_warmup):
            parts = [
                self.warmup_grads(
                    placed,
                    prepared,
                    ci_fn,
                    constraints,
                    tokens_all,
                    jnp.asarray(idx[start : start + microbatch_size]),
                    answer_ids,
                    targets,
                    _rows(adversary.sources, start, start + microbatch_size),
                )
                for start in starts
            ]
            adversary = self.ascend(adversary, _concat_rows(parts), train_frac)

        source_parts: list[SourceStacks] = []

        def micro_call(
            k: int, micro_idx: np.ndarray
        ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
            start = starts[k]
            g, source_g, m, schedules = self.micro_grads(
                placed,
                prepared,
                ci_fn,
                constraints,
                tokens_all,
                jnp.asarray(micro_idx),
                answer_ids,
                targets,
                train_frac,
                _rows(adversary.sources, start, start + microbatch_size),
                random.fold_in(key, k),
            )
            source_parts.append(source_g)
            return g, m, schedules

        call: MicroCall = micro_call
        grads, metrics, schedules = batch_gradients(call, idx, microbatch_size, on_host)
        adversary = self.ascend(adversary, _concat_rows(source_parts), train_frac)
        return grads, metrics, schedules, adversary
