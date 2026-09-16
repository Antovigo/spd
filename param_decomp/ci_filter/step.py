"""Jitted CI-fn fine-tuning step, pool evaluation, and batch indexing.

Every jit takes the placed model, its prepared component weights and the CI fn as TRACED
args (the HLO-baking rule in `core/CLAUDE.md`); only static site names and config scalars
are read off the closure. Pool arrays are replicated and gathered in-jit over the batch axes."""

import functools
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.config import CIFilterConfig
from param_decomp.ci_filter.objective import objective_rows, row_scores
from param_decomp.core.ci_fn import CI, PlacedCIFn, ci_for_role, evaluate_ci
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.losses import importance_minimality_terms, scheduled_value_at
from param_decomp.core.model import BATCH_AXES, MaterializedMasking, PlacedModel
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.run_state import clip_by_global_norm_with_eps, optax_schedule

type Prepared = dict[str, dict[str, Array]]
type Trainable = optax.Params
"""The CI fn's floating arrays with `None` elsewhere: the optimized leaves and their grads."""


def gather_rows(tokens_all: Int[Array, "N T"], idx: Int[Array, " B"]) -> Int[Array, "B T"]:
    """A batch of the replicated pool, sharded over the batch axes. In-jit only."""
    return tokens_all.at[idx].get(out_sharding=P(BATCH_AXES, None))


def output_ci(
    placed: PlacedModel, ci_fn: PlacedCIFn, captures: dict[str, Array], remat: bool
) -> CI:
    """The output head's CI bundle, pinned to the placement's component-activation layout."""
    ci = ci_for_role(evaluate_ci(ci_fn, captures, remat=remat), "output")
    return CI(
        preactivations=ci.preactivations,
        lower={k: constrain_component_activation(v, placed.placement) for k, v in ci.lower.items()},
        upper={k: constrain_component_activation(v, placed.placement) for k, v in ci.upper.items()},
    )


def _count_above(ci_lower: dict[str, Array], threshold: float | Array) -> Array:
    """Components above `threshold` per token, summed over sites, averaged over `(B, T)`."""
    return sum(
        (jnp.mean(jnp.sum(v > threshold, axis=-1, dtype=jnp.float32)) for v in ci_lower.values()),
        start=jnp.zeros((), jnp.float32),
    )


def trainable(ci_fn: PlacedCIFn) -> Trainable:
    return cast(Trainable, eqx.filter(ci_fn, eqx.is_inexact_array))


def make_optimizer(config: CIFilterConfig) -> optax.GradientTransformation:
    adam = optax.adamw(
        optax_schedule(config.lr_schedule, config.steps),
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
        PlacedModel,
        Prepared,
        PlacedCIFn,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Array,
    ],
    tuple[Trainable, dict[str, Array], dict[str, Array]],
]


def make_micro_grads(config: CIFilterConfig) -> MicroGrads:
    """`(model, prepared, ci_fn, pool, idx, answer_ids, train_frac) -> (grads, metrics,
    schedules)` for one microbatch. The loss and metrics are scaled by `1 / n_microbatches`, so
    summing them over the microbatches gives batch means."""
    objective = config.objective
    imp = config.imp_min
    scale = 1.0 / config.n_microbatches

    @eqx.filter_jit
    def micro_grads(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        train_frac: Array,
    ) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
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

        def loss_fn(params: Trainable) -> tuple[Array, dict[str, Array]]:
            ci = output_ci(
                placed, cast(PlacedCIFn, eqx.combine(params, ci_fn)), captures, config.remat
            )
            masked_last = placed.masked_forward(
                prepared,
                tokens,
                masking=MaterializedMasking(component_masks=ci.lower),
                remat=config.remat,
            ).output[:, -1, :]
            recon = jnp.mean(objective_rows(objective, masked_last, clean_last, answer_ids))
            activity, freq = importance_minimality_terms(
                ci.upper,
                gamma,
                None if imp.frequency_coeff is None else imp.reference_datapoint_count,
                imp.normalize_at_one,
            )
            total = config.recon_coeff * recon + impmin_coeff * activity + freq_coeff * freq
            metrics = {
                "loss": total,
                "recon": recon,
                "imp_activity": activity,
                "imp_freq": freq,
                "l0": _count_above(ci.lower, 0.0),
                "l0_alive": _count_above(ci.lower, config.alive_threshold),
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


def batch_gradients(
    micro_grads: MicroGrads,
    placed: PlacedModel,
    prepared: Prepared,
    ci_fn: PlacedCIFn,
    tokens_all: Int[Array, "N T"],
    idx: np.ndarray,
    answer_ids: Int[Array, " K"],
    train_frac: Array,
    microbatch_size: int,
    on_host: bool,
) -> tuple[Trainable, dict[str, Array], dict[str, Array]]:
    """One step's summed gradient over consecutive microbatches of `idx`, with the summed
    (already `1 / n`-scaled) metrics and the step's schedule values. `on_host` keeps the running
    sum of every microbatch but the last in host memory (`CIFilterConfig.accumulate_on_host`)."""
    host_sum = None
    grads = None
    metrics: dict[str, Array] = {}
    schedules: dict[str, Array] = {}
    starts = range(0, len(idx), microbatch_size)
    for start in starts:
        g, m, schedules = micro_grads(
            placed,
            prepared,
            ci_fn,
            tokens_all,
            jnp.asarray(idx[start : start + microbatch_size]),
            answer_ids,
            train_frac,
        )
        metrics = {k: metrics[k] + v if metrics else v for k, v in m.items()}
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
) -> Callable[[PlacedCIFn, optax.OptState, Trainable], tuple[PlacedCIFn, optax.OptState, Array]]:
    @eqx.filter_jit(donate="all-except-first")
    def apply_update(
        ci_fn: PlacedCIFn, opt_state: optax.OptState, grads: Trainable
    ) -> tuple[PlacedCIFn, optax.OptState, Array]:
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
) -> RowArrays:
    scores = row_scores(masked_last, clean_last, answer_ids)
    rows = {**scores._asdict(), "objective": objective_value}
    return {k: jax.sharding.reshard(v, P()) for k, v in rows.items()}


type EvalBatch = Callable[
    [PlacedModel, Prepared, PlacedCIFn, Int[Array, "N T"], Int[Array, " B"], Int[Array, " K"]],
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
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
    ) -> tuple[dict[str, RowArrays], dict[str, Array], dict[str, Array]]:
        tokens = gather_rows(tokens_all, idx)
        clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
        clean_last = clean.output[:, -1, :]
        ci = output_ci(placed, ci_fn, clean.captures, remat=False)
        maskings = {
            "ci": ci.lower,
            "rounded": {k: (v > threshold).astype(COMPUTE_DT) for k, v in ci.lower.items()},
        }
        rows: dict[str, RowArrays] = {}
        for name, masks in maskings.items():
            masked_last = placed.masked_forward(
                prepared, tokens, masking=MaterializedMasking(component_masks=masks), remat=False
            ).output[:, -1, :]
            value = objective_rows(objective, masked_last, clean_last, answer_ids)
            rows[name] = _replicated_rows(value, masked_last, clean_last, answer_ids)
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
        PlacedModel,
        Prepared,
        dict[str, Array],
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
    ],
    RowArrays,
]


def make_score_fixed_masks(config: CIFilterConfig) -> ScoreFixedMasks:
    """Per-row scores of ONE `(C,)` mask per site shared by every prompt and position."""
    objective = config.objective

    @eqx.filter_jit
    def score(
        placed: PlacedModel,
        prepared: Prepared,
        masks: dict[str, Array],
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
    ) -> RowArrays:
        tokens = gather_rows(tokens_all, idx)
        clean_last = placed.clean_forward(tokens).output[:, -1, :]
        broadcast = {k: v[None, None, :].astype(COMPUTE_DT) for k, v in masks.items()}
        masked_last = placed.masked_forward(
            prepared, tokens, masking=MaterializedMasking(component_masks=broadcast), remat=False
        ).output[:, -1, :]
        value = objective_rows(objective, masked_last, clean_last, answer_ids)
        return _replicated_rows(value, masked_last, clean_last, answer_ids)

    return score


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
