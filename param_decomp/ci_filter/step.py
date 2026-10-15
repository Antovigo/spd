"""Jitted CI-fn fine-tuning step, pool evaluation, and batch indexing.

Every jit takes the placed model, its prepared component weights and the CI fn as TRACED
args (the HLO-baking rule in `core/CLAUDE.md`); only static site names and config scalars
are read off the closure. Pool arrays are replicated and gathered in-jit over the batch axes."""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray

from param_decomp.ci_filter.config import CIFilterConfig, CIMaskedRecon
from param_decomp.ci_filter.objective import objective_rows, row_scores
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.ci_fn.runtime import evaluate_ci_role_from_captures
from param_decomp.core.components import ComponentStacks, require_full_emission
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.losses import importance_minimality_terms
from param_decomp.core.model import ForwardResult, MaterializedMasking, SiteRoutes, select_captures
from param_decomp.core.placement import batch_axes
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.recon_eval import fresh_pgd_masking
from param_decomp.core.run_state import clip_by_global_norm_with_eps
from param_decomp.core.runtime_schedule import scheduled_value_at, scheduled_value_traced
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput, StreamedLinearOutput
from param_decomp.targets.transformer import TransformerPreparedWeights

type Prepared = TransformerPreparedWeights
type FilterCIFn = CIFn[LMBatchWithDocuments]
"""The decomposition's CI fn — fp32 masters for the trained one, compute precision for a
ceiling (`CIConstraints`). Every read prepares it (`output_ci`)."""
type CleanForward = ForwardResult[LMOutput, LMBatchWithDocuments]
type Trainable = optax.Params
"""The CI fn's floating arrays with `None` elsewhere: the optimized leaves and their grads."""


def gather_rows(tokens_all: Int[Array, "N T"], idx: Int[Array, " B"]) -> Int[Array, "B T"]:
    """A batch of the replicated pool, sharded over the batch axes. In-jit only."""
    axes = batch_axes(jax.sharding.get_abstract_mesh())
    return tokens_all.at[idx].get(out_sharding=P(axes, None))


def lm_batch(tokens: Int[Array, "B T"]) -> LMBatchWithDocuments:
    """Prompts as the transformer's input: one document per row, no internal boundaries."""
    return LMBatchWithDocuments.from_unsegmented_sequences(tokens)


def gather_batch(tokens_all: Int[Array, "N T"], idx: Int[Array, " B"]) -> LMBatchWithDocuments:
    return lm_batch(gather_rows(tokens_all, idx))


def all_logits(output: LMOutput) -> Float[Array, "B T vocab"]:
    """Every position's logits: the materialized edge's own array, or a streamed output's
    head applied with fp32 accumulation."""
    match output:
        case StreamedLinearOutput(activations=activations, head=head):
            return jnp.dot(activations, head.T, preferred_element_type=jnp.float32)
        case jax.Array():
            return output


def last_logits(output: LMOutput) -> Float[Array, "B vocab"]:
    """`all_logits` at the last position only."""
    match output:
        case StreamedLinearOutput(activations=activations, head=head):
            return jnp.dot(activations[:, -1, :], head.T, preferred_element_type=jnp.float32)
        case jax.Array():
            return output[:, -1, :]


def masked_output(
    placed: PlacedTransformer,
    prepared: Prepared,
    conditioning: LMBatchWithDocuments,
    masking: MaterializedMasking,
    *,
    remat: bool,
    routes: SiteRoutes | None = None,
) -> LMOutput:
    """The masked forward's output under concrete masks, reproducing the clean forward that
    returned `conditioning` (or running the whole model from a plain batch)."""
    return placed.masked_forward(
        prepared,
        conditioning,
        masking=placed.model.prepare_masking(masking),
        routes=routes,
        remat=remat,
    ).output


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseCI:
    """One CI head's bundle over full-emission sites, keyed by output site: the
    preactivations and both squashings (`core.ci_fn.interface.CI`, narrowed to arrays)."""

    preactivations: dict[str, Array]
    lower: dict[str, Array]
    upper: dict[str, Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CIConstraints:
    """What bounds the trained CI fn's output CI, applied at every CI read (`output_ci`)."""

    ceilings: tuple[FilterCIFn, ...]
    """Frozen compute-precision CI fns whose output CI caps the trained one's entrywise
    (a filter's ceilings, `CIFilterConfig`); `()` leaves the CI uncapped."""
    kept: dict[str, Array] | None
    """`{site: (C,)}` 1.0 for a kept component, 0.0 for a removed one, whose CI is forced to 0
    (a filter's pruning, `CIFilterConfig`; its U and V are zeroed by `remove_components`); `None` keeps
    every component."""


UNCONSTRAINED = CIConstraints(ceilings=(), kept=None)


@jax.custom_vjp
def _capped(x: Array, cap: Array) -> Array:
    return jnp.minimum(x, cap)


def _capped_fwd(x: Array, cap: Array) -> tuple[Array, tuple[Array, Array]]:
    return jnp.minimum(x, cap), (x, cap)


def _capped_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    """At or under the cap the gradient passes; over it, only a gradient that LOWERS `x`
    (`g > 0`) does, so an entry pushed over the cap by one term can still be pulled back by
    another (imp-min) instead of freezing there. The cap is frozen."""
    x, cap = res
    return jnp.where((x <= cap) | (g > 0), g, jnp.zeros_like(g)), jnp.zeros_like(cap)


_capped.defvjp(_capped_fwd, _capped_bwd)


def _head_ci(prepared: Prepared, ci_fn: FilterCIFn, forward: CleanForward, remat: bool) -> DenseCI:
    """`ci_fn`'s output head on the CI taps of `forward` (a clean forward, or a masked one
    carrying the taps among other captures), in compute precision."""
    ci = evaluate_ci_role_from_captures(
        ci_fn.prepare(),
        select_captures(forward.captures, ci_fn.capture_keys),
        forward.conditioning,
        prepared,
        sequence=forward.sequence,
        remat=remat,
        role="output",
    )
    return DenseCI(
        preactivations={k: require_full_emission(v) for k, v in ci.preactivations.items()},
        lower={k: require_full_emission(v) for k, v in ci.lower.items()},
        upper={k: require_full_emission(v) for k, v in ci.upper.items()},
    )


def output_ci(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    forward: CleanForward,
    remat: bool,
) -> DenseCI:
    """The output head's CI bundle, capped entrywise by every ceiling's output CI (the
    squashings are monotone, so capping the preactivation, lower and upper CI alike is capping
    the preactivation), zero on removed components, pinned to the placement's
    component-activation layout."""
    ci = _head_ci(prepared, ci_fn, forward, remat)
    for ceiling in constraints.ceilings:
        cap = jax.lax.stop_gradient(_head_ci(prepared, ceiling, forward, remat=False))
        ci = DenseCI(
            preactivations={
                k: _capped(v, cap.preactivations[k]) for k, v in ci.preactivations.items()
            },
            lower={k: _capped(v, cap.lower[k]) for k, v in ci.lower.items()},
            upper={k: _capped(v, cap.upper[k]) for k, v in ci.upper.items()},
        )
    if constraints.kept is not None:
        kept = constraints.kept
        ci = DenseCI(
            preactivations={
                k: jnp.where(kept[k] > 0, v, jnp.minimum(v, 0.0))
                for k, v in ci.preactivations.items()
            },
            lower={k: v * kept[k].astype(v.dtype) for k, v in ci.lower.items()},
            upper={k: v * kept[k].astype(v.dtype) for k, v in ci.upper.items()},
        )
    rule = None if placed.placement is None else placed.placement.activations.component

    def pin(v: Array) -> Array:
        return require_full_emission(constrain_component_activation(v, rule))

    return DenseCI(
        preactivations=ci.preactivations,
        lower={k: pin(v) for k, v in ci.lower.items()},
        upper={k: pin(v) for k, v in ci.upper.items()},
    )


@eqx.filter_jit
def remove_components(components: ComponentStacks, kept: dict[str, Array]) -> ComponentStacks:
    """Zero every removed component's V column and U row (`CIConstraints.kept`): it then
    contributes nothing under any mask, and with the weight delta on its weight lies in the
    delta `W - UV`, exactly as if the component had been deleted. A stack's persist pads are
    all-zero already and stay so."""
    stacks = dict(components.stacks)
    for group, (Vs, Us) in components.stacks.items():
        slots = _group_slots(components, group)
        pad = components.pad_of(group)
        assert len(slots) + pad == Vs.shape[0], (group, slots, pad, Vs.shape)
        keep = jnp.stack([kept[name] for name in slots]).astype(Vs.dtype)
        keep = jnp.pad(keep, ((0, pad), (0, 0)))
        stacks[group] = (Vs * keep[:, None, :], Us * keep[:, :, None])
    return ComponentStacks(
        stacks=stacks,
        site_stack_indices=components.site_stack_indices,
        stack_pads=components.stack_pads,
    )


def _group_slots(components: ComponentStacks, group: str) -> list[str]:
    """The group's site names in stack-index order."""
    by_index = {index: name for name, g, index in components.site_stack_indices if g == group}
    return [by_index[i] for i in range(len(by_index))]


def count_above(ci_lower: dict[str, Array], threshold: float | Array) -> Array:
    """Components above `threshold` per token, summed over sites, averaged over `(B, T)`."""
    return sum(
        (jnp.mean(jnp.sum(v > threshold, axis=-1, dtype=jnp.float32)) for v in ci_lower.values()),
        start=jnp.zeros((), jnp.float32),
    )


@eqx.filter_jit
def prepare_components(
    placed: PlacedTransformer, components: ComponentStacks
) -> tuple[Prepared, dict[str, Array]]:
    """The components in compute form and each component's `||V_c||` (for the grid's
    normalized inner activations), replicated."""
    v_norms = {
        site: jax.sharding.reshard(
            jnp.linalg.norm(components.site(site).V.astype(jnp.float32), axis=0), P()
        )
        for site in placed.model.site_names
    }
    return placed.prepare_compute_weights(components), v_norms


def trainable(ci_fn: FilterCIFn) -> Trainable:
    return cast(Trainable, eqx.filter(ci_fn, eqx.is_inexact_array))


def lr_schedule(config: CIFilterConfig) -> Callable[[ArrayLike], Array]:
    """The CI fn's learning rate at an update count, held at its final value past the run."""

    def schedule(count: ArrayLike) -> Array:
        return scheduled_value_traced(
            jnp.asarray(count, jnp.float32), config.steps, config.lr_schedule
        )

    return schedule


def make_optimizer(config: CIFilterConfig) -> optax.GradientTransformation:
    adam = optax.adamw(
        lr_schedule(config),
        b1=config.betas[0],
        b2=config.betas[1],
        eps=1e-8,
        weight_decay=0.0,
    )
    if config.grad_clip_norm is None:
        return adam
    return optax.chain(clip_by_global_norm_with_eps(config.grad_clip_norm, eps=1e-6), adam)


type MicroGrads = Callable[
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
    ],
    tuple[Trainable, dict[str, Array], dict[str, Array]],
]


def make_micro_grads(config: CIFilterConfig) -> MicroGrads:
    """`(model, prepared, ci_fn, constraints, pool, idx, answer_ids, targets, train_frac) -> (grads, metrics,
    schedules)` for one microbatch. The loss and metrics are scaled by `1 / n_microbatches`, so
    summing them over the microbatches gives batch means."""
    objective = config.objective
    imp = config.imp_min
    scale = 1.0 / config.n_microbatches
    assert isinstance(config.recon, CIMaskedRecon), config.recon
    recon_coeff = config.recon.coeff

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
    ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
        target_idx = targets[idx]
        clean = jax.lax.stop_gradient(
            placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        )
        clean_last = last_logits(clean.output)
        gamma = scheduled_value_at(train_frac, imp.gamma)
        impmin_coeff = scheduled_value_at(train_frac, imp.coeff)
        freq_coeff = (
            jnp.zeros((), jnp.float32)
            if imp.frequency_coeff is None
            else scheduled_value_at(train_frac, imp.frequency_coeff)
        )

        def loss_fn(params: Trainable) -> tuple[Array, dict[str, Array]]:
            ci = output_ci(
                placed,
                prepared,
                cast(FilterCIFn, eqx.combine(params, ci_fn)),
                constraints,
                clean,
                config.remat,
            )
            masked_last = last_logits(
                masked_output(
                    placed,
                    prepared,
                    clean.conditioning,
                    MaterializedMasking(component_masks=ci.lower),
                    remat=config.remat,
                )
            )
            recon = jnp.mean(
                objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
            )
            activity, freq = importance_minimality_terms(
                ci.upper,
                gamma,
                None if imp.frequency_coeff is None else imp.reference_datapoint_count,
                normalize_at_one=imp.normalize_at_one,
            )
            total = recon_coeff * recon + impmin_coeff * activity + freq_coeff * freq
            metrics = {
                "loss": total,
                "recon": recon,
                "imp_activity": activity,
                "imp_freq": freq,
                "l0": count_above(ci.lower, 0.0),
                "l0_alive": count_above(ci.lower, config.alive_threshold),
            }
            return scale * total, metrics

        grads, metrics = eqx.filter_grad(loss_fn, has_aux=True)(trainable(ci_fn))
        metrics = {k: scale * v for k, v in metrics.items()}
        schedules = {"gamma": gamma, "impmin_coeff": impmin_coeff, "freq_coeff": freq_coeff}
        return cast(Trainable, grads), metrics, schedules

    return micro_grads


@eqx.filter_jit(donate="all")
def accumulate(acc: Trainable, grads: Trainable) -> Trainable:
    return jax.tree.map(lambda a, g: a + g, acc, grads)


type MicroCall = Callable[[int, np.ndarray], tuple[Trainable, dict[str, Array], dict[str, Array]]]
"""`(microbatch index, prompt indices) -> (grads, 1/n-scaled metrics, schedules)`."""


def ci_masked_micro_call(
    micro_grads: MicroGrads,
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    tokens_all: Int[Array, "N T"],
    answer_ids: Int[Array, " K"],
    targets: Int[Array, " N"],
    train_frac: Array,
) -> MicroCall:
    """The deterministic CI-masked step's microbatch call for `batch_gradients`."""

    def call(
        _k: int, micro_idx: np.ndarray
    ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
        return micro_grads(
            placed,
            prepared,
            ci_fn,
            constraints,
            tokens_all,
            jnp.asarray(micro_idx),
            answer_ids,
            targets,
            train_frac,
        )

    return call


def batch_gradients(
    micro_call: MicroCall, idx: np.ndarray, microbatch_size: int, on_host: bool
) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
    """One step's summed gradient over consecutive microbatches of `idx`, with the summed
    (already `1 / n`-scaled) metrics and the step's schedule values. `on_host` keeps the running
    sum of every microbatch but the last in host memory (`CIFilterConfig.accumulate_on_host`)."""
    host_sum = None
    grads = None
    metrics: dict[str, Array] = {}
    schedules: dict[str, Array] = {}
    starts = range(0, len(idx), microbatch_size)
    for k, start in enumerate(starts):
        g, m, schedules = micro_call(k, idx[start : start + microbatch_size])
        metrics = {name: metrics[name] + v if metrics else v for name, v in m.items()}
        if on_host and start != starts[-1]:
            host_g = jax.tree.map(np.array, g)  # writable host copies
            del g
            host_sum = (
                host_g
                if host_sum is None
                else jax.tree.map(lambda a, b: np.add(a, b, out=a), host_sum, host_g)
            )
        else:
            grads = g if grads is None else accumulate(grads, g)
    assert grads is not None
    if host_sum is not None:
        device_sum = jax.tree.map(lambda h, like: jax.device_put(h, like.sharding), host_sum, grads)
        grads = accumulate(grads, device_sum)
    return grads, metrics, schedules


def make_apply_update(
    optimizer: optax.GradientTransformation,
) -> Callable[[FilterCIFn, optax.OptState, Trainable], tuple[FilterCIFn, optax.OptState, Array]]:
    @eqx.filter_jit(donate="all-except-first")
    def apply_update(
        ci_fn: FilterCIFn, opt_state: optax.OptState, grads: Trainable
    ) -> tuple[FilterCIFn, optax.OptState, Array]:
        params = trainable(ci_fn)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(ci_fn, updates), opt_state, optax.tree.norm(grads)

    return apply_update


# ----------------------------------- evaluation -----------------------------------


type RowArrays = dict[str, Array]


def _replicated_rows(
    objective_value: Float[Array, " B"],
    masked_last: Float[Array, "B vocab"],
    clean_last: Float[Array, "B vocab"],
    answer_ids: Int[Array, " K"],
    target_idx: Int[Array, " B"],
) -> RowArrays:
    scores = row_scores(masked_last, clean_last, answer_ids, target_idx)
    rows = {**scores._asdict(), "objective": objective_value}
    return {k: jax.sharding.reshard(v, P()) for k, v in rows.items()}


type EvalBatch = Callable[
    [
        PlacedTransformer,
        Prepared,
        FilterCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
    ],
    tuple[dict[str, RowArrays], dict[str, Array], dict[str, Array]],
]


def make_eval_batch(config: CIFilterConfig) -> EvalBatch:
    """`-> ({masking: per-row scores}, {site: max CI over rows x positions}, sums)` for the
    continuous-CI and rounded-CI maskings. Pad rows (repeats of row 0) cannot move a max; the
    caller trims the per-row arrays and so the sums are over REAL rows via `n_real`."""
    objective = config.objective
    threshold = config.alive_threshold

    @eqx.filter_jit
    def eval_batch(
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
    ) -> tuple[dict[str, RowArrays], dict[str, Array], dict[str, Array]]:
        target_idx = targets[idx]
        clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        clean_last = last_logits(clean.output)
        ci = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False)
        maskings = {
            "ci": ci.lower,
            "rounded": {k: (v > threshold).astype(COMPUTE_DT) for k, v in ci.lower.items()},
        }
        rows: dict[str, RowArrays] = {}
        for name, masks in maskings.items():
            masked_last = last_logits(
                masked_output(
                    placed,
                    prepared,
                    clean.conditioning,
                    MaterializedMasking(component_masks=masks),
                    remat=False,
                )
            )
            value = objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
            rows[name] = _replicated_rows(value, masked_last, clean_last, answer_ids, target_idx)
        site_max = {
            k: jax.sharding.reshard(jnp.max(v.astype(jnp.float32), axis=(0, 1)), P())
            for k, v in ci.lower.items()
        }
        above = functools.reduce(
            jnp.add,
            [jnp.sum(v > threshold, axis=-1, dtype=jnp.float32) for v in ci.lower.values()],
        )
        l0_rows = {
            "l0_per_token": jax.sharding.reshard(jnp.mean(above, axis=1), P()),
            "l0_last": jax.sharding.reshard(above[:, -1], P()),
        }
        return rows, site_max, l0_rows

    return eval_batch


type ScoreFixedMasks = Callable[
    [
        PlacedTransformer,
        Prepared,
        dict[str, Array],
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
    ],
    RowArrays,
]


def make_score_fixed_masks(config: CIFilterConfig) -> ScoreFixedMasks:
    """Per-row scores of ONE `(C,)` mask per site shared by every prompt and position."""
    objective = config.objective

    @eqx.filter_jit
    def score(
        placed: PlacedTransformer,
        prepared: Prepared,
        masks: dict[str, Array],
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
    ) -> RowArrays:
        target_idx = targets[idx]
        clean = placed.clean_forward(gather_batch(tokens_all, idx))
        clean_last = last_logits(clean.output)
        broadcast = {k: v[None, None, :].astype(COMPUTE_DT) for k, v in masks.items()}
        masked_last = last_logits(
            masked_output(
                placed,
                prepared,
                clean.conditioning,
                MaterializedMasking(component_masks=broadcast),
                remat=False,
            )
        )
        value = objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
        return _replicated_rows(value, masked_last, clean_last, answer_ids, target_idx)

    return score


type PGDEval = Callable[
    [
        PlacedTransformer,
        Prepared,
        FilterCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
        PRNGKeyArray,
    ],
    Array,
]


def make_pgd_eval(config: CIFilterConfig) -> PGDEval:
    """One batch of the decomposition's fresh-PGD reconstruction eval (`PGDEvalConfig`),
    on the prepared component weights and scored by the filter's objective at the LAST
    position: the kernel is `core.recon_eval.fresh_pgd_masking`, delta channel attacked."""
    pgd = config.pgd_eval
    assert pgd is not None
    objective = config.objective

    @eqx.filter_jit
    def pgd_eval(
        placed: PlacedTransformer,
        prepared: Prepared,
        ci_fn: FilterCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        targets: Int[Array, " N"],
        key: PRNGKeyArray,
    ) -> Array:
        target_idx = targets[idx]
        clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
        clean_last = last_logits(clean.output)
        ci_lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower

        def loss_at_masking(masking: MaterializedMasking) -> Array:
            masked_last = last_logits(
                masked_output(placed, prepared, clean.conditioning, masking, remat=config.remat)
            )
            return jnp.mean(
                objective_rows(objective, masked_last, clean_last, answer_ids, target_idx)
            )

        masking = fresh_pgd_masking(
            placed.model.sites,
            ci_lower,
            clean.leading_shape,
            key,
            pgd.step_size,
            pgd.n_steps,
            loss_at_masking,
        )
        return loss_at_masking(masking)

    return pgd_eval


def pgd_eval_batches(n_prompts: int, config: CIFilterConfig) -> list[np.ndarray]:
    """The fixed, disjoint prompt draws the PGD eval reads, identical before and after."""
    assert config.pgd_eval is not None
    rng = np.random.default_rng(np.random.SeedSequence((config.seed, 1)))
    total = config.pgd_eval.n_batches * config.pgd_eval.batch_size
    return np.split(
        rng.choice(n_prompts, size=total, replace=False).astype(np.int32), config.pgd_eval.n_batches
    )


def index_batches(n: int, batch_size: int) -> list[tuple[np.ndarray, int]]:
    """Fixed-size index batches covering `0..n-1`; the last is padded with row 0 and `real`
    counts its genuine leading rows."""
    out = []
    for start in range(0, n, batch_size):
        real = min(batch_size, n - start)
        idx = np.concatenate(
            [np.arange(start, start + real), np.zeros(batch_size - real, np.int64)]
        )
        out.append((idx.astype(np.int32), real))
    return out


def sample_batch(n: int, batch_size: int, seed: int, step: int) -> np.ndarray:
    """`batch_size` distinct prompts, a pure function of `(seed, step)`."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, step)))
    return rng.choice(n, size=batch_size, replace=False).astype(np.int32)
