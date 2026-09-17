"""The decomposition's merged stochastic-subset persistent-PGD reconstruction (SPEC S34, S13′,
S14′, S24), scored by the CI filter's objective (`MergedPPGDRecon`).

Built from core's own kernels — `init_persistent_sources`, `sources_adam_ascend_project`,
`uniform_k_subset_routes`, `masks_from_sources`, `mixed_persistent_stochastic_masks` — with the
filter's frozen components (only the CI fn trains). The persistent sources hold one row per
batch slot; microbatches read consecutive row slices, and every Adam ascent updates the whole
bundle once from the concatenated slice gradients, so the trajectory does not depend on the
microbatch split beyond float reassociation."""

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
    Ceilings,
    MicroCall,
    Prepared,
    Trainable,
    batch_gradients,
    count_above,
    gather_rows,
    output_ci,
    trainable,
)
from param_decomp.core.adversary import (
    Sources,
    SourcesAdamState,
    init_persistent_sources,
    init_sources_adam_state,
    sources_adam_ascend_project,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.losses import importance_minimality_terms, scheduled_value_at
from param_decomp.core.masking import masks_from_sources, mixed_persistent_stochastic_masks
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.recon import uniform_k_subset_routes


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Adversary:
    sources: Sources
    opt_state: SourcesAdamState


def _ppgd(config: CIFilterConfig) -> MergedPPGDRecon:
    assert isinstance(config.recon, MergedPPGDRecon), config.recon
    return config.recon


def init_adversary(
    placed: PlacedModel, batch_size: int, seq_len: int, key: PRNGKeyArray
) -> Adversary:
    """U[0,1] fp32 sources of shape `(batch, position, C + 1)` per site, replicated (the
    microbatch row slices are host-side indexing; a batch-sharded bundle would make each slice
    a cross-device gather)."""
    sources = init_persistent_sources(
        placed.site_names,
        tuple(site.C for site in placed.sites),
        (batch_size, seq_len),
        jnp.float32,
        key,
    )
    if not jax.sharding.get_abstract_mesh().empty:
        sources = jax.tree.map(lambda x: jax.sharding.reshard(x, P()), sources)
    return Adversary(sources=sources, opt_state=init_sources_adam_state(sources))


def _rows(sources: Sources, start: int, stop: int) -> Sources:
    return jax.tree.map(lambda x: x[start:stop], sources)


def _concat_rows(parts: list[Sources]) -> Sources:
    return parts[0] if len(parts) == 1 else jax.tree.map(lambda *xs: jnp.concatenate(xs), *parts)


type WarmupGrads = Callable[
    [
        PlacedModel,
        Prepared,
        PlacedCIFn,
        Ceilings,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Sources,
    ],
    Sources,
]


def make_warmup_grads(config: CIFilterConfig) -> WarmupGrads:
    """`d(objective)/d(sources)` of one microbatch slice under the ALL-adversarial, route-all
    forward with the CI detached (S24), scaled by `1 / n_microbatches` so the slices sum to the
    batch mean's gradient."""
    objective = config.objective
    scale = 1.0 / config.n_microbatches

    @eqx.filter_jit
    def warmup_grads(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        ceilings: Ceilings,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        sources: Sources,
    ) -> Sources:
        tokens = gather_rows(tokens_all, idx)
        clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
        clean_last = jax.lax.stop_gradient(clean.output[:, -1, :])
        lower = jax.lax.stop_gradient(
            output_ci(placed, ci_fn, ceilings, clean.captures, remat=False).lower
        )

        def score(sources: Sources) -> Array:
            masks, delta_masks = masks_from_sources(lower, sources)
            masked_last = placed.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(component_masks=masks, weight_delta_masks=delta_masks),
                remat=config.remat,
            ).output[:, -1, :]
            return scale * jnp.mean(objective_rows(objective, masked_last, clean_last, answer_ids))

        return jax.grad(score)(sources)

    return warmup_grads


type PPGDMicroGrads = Callable[
    [
        PlacedModel,
        Prepared,
        PlacedCIFn,
        Ceilings,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Array,
        Sources,
        PRNGKeyArray,
    ],
    tuple[Trainable, Sources, dict[str, Array], dict[str, Array]],
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
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        ceilings: Ceilings,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        train_frac: Array,
        sources: Sources,
        key: PRNGKeyArray,
    ) -> tuple[Trainable, Sources, dict[str, Array], dict[str, Array]]:
        tokens = gather_rows(tokens_all, idx)
        clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
        clean_last = jax.lax.stop_gradient(clean.output[:, -1, :])
        captures = jax.lax.stop_gradient(clean.captures)
        gamma = scheduled_value_at(train_frac, imp.gamma)
        impmin_coeff = scheduled_value_at(train_frac, imp.coeff)
        freq_coeff = (
            jnp.zeros((), jnp.float32)
            if imp.frequency_coeff is None
            else scheduled_value_at(train_frac, imp.frequency_coeff)
        )
        adv_fraction = scheduled_value_at(train_frac, recon.adv_fraction)
        route_key, mask_key = random.split(key)
        routes = uniform_k_subset_routes(route_key, placed.site_names, tokens.shape)

        def loss_fn(params: Trainable, sources: Sources) -> tuple[Array, dict[str, Array]]:
            ci = output_ci(
                placed,
                cast(PlacedCIFn, eqx.combine(params, ci_fn)),
                ceilings,
                captures,
                config.remat,
            )
            masks, delta_masks, merged_routes = mixed_persistent_stochastic_masks(
                mask_key, ci.lower, sources, tokens.shape, adv_fraction, routes
            )
            masked_last = placed.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(
                    component_masks=masks, weight_delta_masks=delta_masks, routes=merged_routes
                ),
                remat=config.remat,
            ).output[:, -1, :]
            recon_value = jnp.mean(objective_rows(objective, masked_last, clean_last, answer_ids))
            activity, freq = importance_minimality_terms(
                ci.upper,
                gamma,
                None if imp.frequency_coeff is None else imp.reference_datapoint_count,
                imp.normalize_at_one,
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


def make_ascend(config: CIFilterConfig) -> Callable[[Adversary, Sources, Array], Adversary]:
    """One Adam ascent-and-project of the whole bundle (S13′/S15)."""
    optimizer = _ppgd(config).optimizer

    @eqx.filter_jit
    def ascend(adversary: Adversary, grads: Sources, train_frac: Array) -> Adversary:
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
    ascend: Callable[[Adversary, Sources, Array], Adversary]
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
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        ceilings: Ceilings,
        tokens_all: Int[Array, "N T"],
        idx: np.ndarray,
        answer_ids: Int[Array, " K"],
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
                    ceilings,
                    tokens_all,
                    jnp.asarray(idx[start : start + microbatch_size]),
                    answer_ids,
                    _rows(adversary.sources, start, start + microbatch_size),
                )
                for start in starts
            ]
            adversary = self.ascend(adversary, _concat_rows(parts), train_frac)

        source_parts: list[Sources] = []

        def micro_call(
            k: int, micro_idx: np.ndarray
        ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
            start = starts[k]
            g, source_g, m, schedules = self.micro_grads(
                placed,
                prepared,
                ci_fn,
                ceilings,
                tokens_all,
                jnp.asarray(micro_idx),
                answer_ids,
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
