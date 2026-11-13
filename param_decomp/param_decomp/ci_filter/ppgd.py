"""The decomposition's merged stochastic-subset persistent-PGD reconstruction (SPEC S34, S13′,
S14′, S24), scored by the CI filter's objective (`MergedPPGDRecon`).

Built from core's own kernels — `init_persistent_sources`, `sources_adam_ascend_project`,
`UniformKRouting`, `source_masking`, `mixed_persistent_stochastic_masking` — with the filter's
frozen components (only the CI fn trains). The persistent sources hold `R` rows (the axis after
each stack's stack axis; `R` an integer >= 1): batch element `j` of a step (0-based, in batch
order) reads row `j mod R`, so with `R` = the batch size every element owns one row (the
decomposition's own `bsc` layout) and with fewer rows each row serves every `R`-th element.
Microbatches read their elements' rows, and every Adam ascent updates the whole bundle once
from the slice gradients summed per row (concatenated when the slices tile `0..R-1` in order),
so the trajectory does not depend on the microbatch split beyond float reassociation.
`MergedPPGDRecon.weight_delta` sets whether routed sites add the masked weight delta."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

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
from param_decomp.core.model import MaterializedMasking, SourceMasking
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


def n_rows(sources: SourceStacks) -> int:
    """`R`, the bundle's number of adversary rows."""
    (rows,) = {leaf.shape[ROW_AXIS] for leaf in jax.tree.leaves(sources)}
    return int(rows)


def row_ids(start: int, size: int, rows: int) -> np.ndarray:
    """The adversary rows of batch elements `start .. start + size - 1`: element `j` reads row
    `j mod rows`."""
    return (start + np.arange(size)) % rows


def _rows(sources: SourceStacks, ids: np.ndarray) -> SourceStacks:
    """The bundle's rows `ids`, in order (a slice when they are consecutive)."""
    lo = int(ids[0])
    if np.array_equal(ids, np.arange(lo, lo + ids.size)):
        return jax.tree.map(
            lambda x: jax.lax.slice_in_dim(x, lo, lo + ids.size, axis=ROW_AXIS), sources
        )
    return jax.tree.map(lambda x: jnp.take(x, jnp.asarray(ids), axis=ROW_AXIS), sources)


def _concat_rows(parts: list[SourceStacks]) -> SourceStacks:
    if len(parts) == 1:
        return parts[0]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=ROW_AXIS), *parts)


@eqx.filter_jit
def _add_folded(total: SourceStacks, part: SourceStacks, offset: int, rows: int) -> SourceStacks:
    """`total` plus `part`'s rows folded onto the bundle: `part`'s row `i` (`0 <= i < M`) adds
    to bundle row `(offset + i) mod rows`. A zero pad and a reshaped sum, no scatter (XLA's
    deterministic scatter serializes); `part` replicated first."""

    def add(t: Array, g: Array) -> Array:
        if not jax.sharding.get_abstract_mesh().empty:
            g = jax.sharding.reshard(g, P())
        m = g.shape[ROW_AXIS]
        n = -(-(offset + m) // rows)
        pad = [(0, 0)] * g.ndim
        pad[ROW_AXIS] = (offset, n * rows - offset - m)
        g = jnp.pad(g.astype(t.dtype), pad)
        folded = g.reshape(*g.shape[:ROW_AXIS], n, rows, *g.shape[ROW_AXIS + 1 :])
        return t + folded.sum(axis=ROW_AXIS)

    return jax.tree.map(add, total, part)


def sum_rows(parts: list[SourceStacks], ids: list[np.ndarray], like: SourceStacks) -> SourceStacks:
    """The per-row sum of slice gradients `parts` (slice `k` holding rows `ids[k]`, each
    `row_ids` of a consecutive run of batch elements), shaped like the bundle `like`; the plain
    concatenation when the slices tile `0..R-1` in order."""
    rows = n_rows(like)
    if np.array_equal(np.concatenate(ids), np.arange(rows)):
        return _concat_rows(parts)
    total = jax.tree.map(jnp.zeros_like, like)
    for part, part_ids in zip(parts, ids, strict=True):
        offset = int(part_ids[0])
        assert np.array_equal(part_ids, row_ids(offset, part_ids.size, rows)), part_ids
        total = _add_folded(total, part, offset, rows)
    return total


def _masking(masking: SourceMasking, weight_delta: Literal["on", "off"]) -> MaterializedMasking:
    """The materialized masks, with the weight-delta masks dropped (`d = 0`) when `off`."""
    materialized = materialize_masking(masking)
    match weight_delta:
        case "on":
            return materialized
        case "off":
            return MaterializedMasking(
                component_masks=materialized.component_masks, weight_delta_masks=None
            )


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
    weight_delta = _ppgd(config).weight_delta

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
            masking = _masking(source_masking(lower, sources.per_site()), weight_delta)
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
                    _masking(masking, recon.weight_delta),
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
        rows = n_rows(adversary.sources)
        ids = [row_ids(start, len(idx[start : start + microbatch_size]), rows) for start in starts]
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
                    _rows(adversary.sources, part_ids),
                )
                for start, part_ids in zip(starts, ids, strict=True)
            ]
            adversary = self.ascend(adversary, sum_rows(parts, ids, adversary.sources), train_frac)

        source_parts: list[SourceStacks] = []

        def micro_call(
            k: int, micro_idx: np.ndarray
        ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
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
                _rows(adversary.sources, ids[k]),
                random.fold_in(key, k),
            )
            source_parts.append(source_g)
            return g, m, schedules

        call: MicroCall = micro_call
        grads, metrics, schedules = batch_gradients(call, idx, microbatch_size, on_host)
        adversary = self.ascend(
            adversary, sum_rows(source_parts, ids, adversary.sources), train_frac
        )
        return grads, metrics, schedules, adversary
