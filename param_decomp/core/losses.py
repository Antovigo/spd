"""The pure loss terms (SPEC §2) and their schedules — fp32 reductions, no state."""

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from param_decomp.core.configs import (
    AnyImportanceMinimalityLossConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    SmoothL0ImportanceMinimalityLossConfig,
)
from param_decomp.core.recon import (
    ForwardObservations,
    OutputAndHiddenActsReconstruction,
    OutputOnlyReconstruction,
    ReconstructionSpec,
)
from param_decomp.core.schedule import Knot, ScheduleConfig


def _interval_frac_traced(prev: Knot, knot: Knot, t: Array) -> Array:
    u = (t - prev.at) / (knot.at - prev.at)
    match knot.interp:
        case "linear":
            return prev.frac + (knot.frac - prev.frac) * u
        case "cosine":
            return prev.frac + (knot.frac - prev.frac) * 0.5 * (1 - jnp.cos(jnp.pi * u))
        case "hold":
            return jnp.where(u >= 1.0, knot.frac, prev.frac)


def scheduled_value_traced(step_f32: Array, total_steps: int, config: ScheduleConfig) -> Array:
    """jnp twin of `schedule.get_scheduled_value` for a traced `step_f32` (inside the
    jitted step, or as an optax schedule over the update count). Same values pointwise
    (`test_schedule.py` pins the pair); the knot structure is static (from config), only
    `t` is traced, so interval selection is a `jnp.where` chain. Lives here rather than
    next to its host twin so the config schema stays jax-free.

    The `total_steps - 1` denominator in `t` is canonical-torch: the `at = 1.0` knot is
    reached AT `step = total_steps - 1` (SPEC S20). Plain `optax.cosine_decay_schedule`
    divides by `steps` and gets there one update later — a genuine ~O(1/steps) per-step
    divergence this fn must avoid."""
    assert total_steps > 0, f"total_steps must be positive, got {total_steps}"

    if total_steps == 1:
        t = jnp.zeros((), jnp.float32)
    else:
        t = jnp.minimum(step_f32 / (total_steps - 1), 1.0)

    points = config.points
    frac = _interval_frac_traced(points[0], points[1], t)
    for prev, knot in zip(points[1:], points[2:], strict=False):
        frac = jnp.where(t >= prev.at, _interval_frac_traced(prev, knot, t), frac)
    return jnp.asarray(config.max_val * frac, jnp.float32)


def coeff_at(step_f32: Array, total_steps: int, coeff: LossCoeff) -> Float[Array, ""] | float:
    """A loss coefficient's value at this step: a bare float IS the constant; a
    `ScheduleConfig` is evaluated like `pnorm` (traced, so the jit signature stays
    stable across the run)."""
    match coeff:
        case ScheduleConfig():
            return scheduled_value_traced(step_f32, total_steps, coeff)
        case float() | int():  # int: pyright's numeric tower widens the declared float
            return coeff


def reconstruction_spec_at(
    hidden_acts_reconstruction: HiddenActsReconstruction | None,
    step_f32: Array,
    total_steps: int,
) -> ReconstructionSpec:
    """The value-level reconstruction spec at this step — the S35 rider's
    possibly-scheduled coeff resolved to a scalar, so schedule objects never enter the
    loss math. The static twin for step-less contexts is `recon.resolve_reconstruction_spec`."""
    if hidden_acts_reconstruction is None:
        return OutputOnlyReconstruction()
    return OutputAndHiddenActsReconstruction(
        coeff_at(step_f32, total_steps, hidden_acts_reconstruction.coeff),
        hidden_acts_reconstruction.points,
    )


@jaxtyped(typechecker=beartype)
def relative_squared_error(
    masked: Float[Array, "*leading d"],
    clean: Float[Array, "*leading d"],
    *,
    valid_row_mask: Float[Array, " batch"] | None = None,
) -> Float[Array, ""]:
    """`Σ(masked−clean)² / Σ(clean²)` at ONE measurement point, in fp32 (SPEC S35).

    Per point, not over a stacked point axis: points need not share a width, and each
    divides by its own clean scale. Callers stack the resulting scalars, never the
    activations."""
    masked_f32 = masked.astype(jnp.float32)
    clean_f32 = clean.astype(jnp.float32)
    squared_error = (masked_f32 - clean_f32) ** 2
    squared_clean = clean_f32**2
    if valid_row_mask is not None:
        mask = valid_row_mask.reshape(valid_row_mask.shape[0], *((1,) * (clean.ndim - 1)))
        squared_error = squared_error * mask
        squared_clean = squared_clean * mask
    return jnp.sum(squared_error) / jnp.sum(squared_clean)


class OutputOnlyReconstructionLoss(NamedTuple):
    total: Array


class OutputAndHiddenActsReconstructionLoss(NamedTuple):
    total: Array
    output: Array
    hidden_acts_by_point: dict[str, Array]


type ReconstructionLoss = OutputOnlyReconstructionLoss | OutputAndHiddenActsReconstructionLoss


def reconstruction_loss(
    recon_loss_fn: Callable[[Any, Any], Array],
    *,
    masked: ForwardObservations,
    clean: ForwardObservations,
    reconstruction: ReconstructionSpec,
    valid_row_mask: Array | None = None,
) -> ReconstructionLoss:
    """The closed forms of one recon comparison (SPEC S35)."""
    output_loss = recon_loss_fn(masked.output, clean.output)
    match reconstruction:
        case OutputOnlyReconstruction():
            return OutputOnlyReconstructionLoss(output_loss)
        case OutputAndHiddenActsReconstruction(coeff=coeff, points=points):
            per_point = {
                point: relative_squared_error(
                    masked.hidden_acts_by_point[point],
                    clean.hidden_acts_by_point[point],
                    valid_row_mask=valid_row_mask,
                )
                for point in points
            }
            aggregate = jnp.mean(jnp.stack(tuple(per_point.values())))
            return OutputAndHiddenActsReconstructionLoss(
                output_loss + coeff * aggregate, output_loss, per_point
            )


def reconstruction_loss_metrics(loss: ReconstructionLoss) -> dict[str, Array]:
    """Metric suffixes contributed by one reconstruction-loss result."""
    match loss:
        case OutputOnlyReconstructionLoss():
            return {}
        case OutputAndHiddenActsReconstructionLoss(
            output=output, hidden_acts_by_point=hidden_acts_by_point
        ):
            return {
                "e2e": output,
                "hidden_acts_reconstruction": jnp.mean(
                    jnp.stack(tuple(hidden_acts_by_point.values()))
                ),
                **{
                    f"hidden_acts_reconstruction/{point}": value
                    for point, value in hidden_acts_by_point.items()
                },
            }


def mean_reconstruction_losses[T](values: tuple[T, ...]) -> T:
    """Mean every fp32 scalar leaf across structurally identical pytrees."""

    def scalar_mean(*scalars: Array) -> Array:
        return sum(scalars, start=jnp.zeros((), jnp.float32)) / len(scalars)

    return jax.tree.map(scalar_mean, *values)


@jaxtyped(typechecker=beartype)
def faithfulness_loss(weight_deltas: dict[str, Float[Array, "_ _"]]) -> Float[Array, ""]:
    """`Σ_s ‖Δ_s‖² / Σ_s numel` over fp32 deltas (SPEC S17). Each `Δ_s` is `(d_out, d_in)`;
    dims are per-site (anonymous, not bound across sites)."""
    numerator = sum(
        ((delta.astype(jnp.float32) ** 2).sum() for delta in weight_deltas.values()),
        start=jnp.zeros((), jnp.float32),
    )
    # float, not int: the full-model param total (Σ d_in·d_out ≈ 7e9) overflows the int32
    # that jax materializes a Python int into under jit. A float normalizer is exact here.
    denominator = float(sum(delta.size for delta in weight_deltas.values()))
    return numerator / denominator


def _imp_min_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    per_value_penalty: Callable[[Float[Array, "*leading _"]], Float[Array, "*leading _"]],
    reference_datapoint_count: int | Literal["auto"] | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`(lp, freq)` for any per-value penalty `psi`, with per-site grouping (SPEC S7/S8):

    - `lp = Σ_s Σ_c f_c`, the bare per-component mean firing rate `f_c = (Σ_{b,t} psi(c)) / B·T`.
    - `freq = Σ_s Σ_c f_c · log2(1 + a' · f_c)`, the batch-invariant frequency penalty with
      `a' = reference_datapoint_count`; `0.0` when `reference_datapoint_count is None`.

    `"auto"` resolves `a'` to this call's own `B·T`, making `a' · f_c` the raw firing COUNT
    — the torch oracle's `log2(1 + layer_sums * world_size)` exactly, with no constant to
    keep in sync. It is the only spelling that is correct on BOTH tPD passes at once: a
    literal `a'` is right for whichever stream's `B·T` it names and off by that stream's
    ratio on the other (SPEC T6 shares one frequency block across both passes), which puts
    the other stream's penalty knee at `k = B·T / a'` firing tokens instead of 1.

    The two imp-min penalties (`L_p`, smooth-L0) differ ONLY in `psi`. Under GSPMD the
    `*leading` axes are the global batch, so `jnp.sum` IS the exact global per-component
    sum — XLA reduces across shards inside the graph, so `f_c` is the true full-batch
    frequency inside the convex `log2` (a per-shard `f_c` would give a Jensen bias)."""
    lp = jnp.zeros((), jnp.float32)
    freq = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (*leading, C)
        leading_axes = tuple(range(ci.ndim - 1))
        n_positions = math.prod(ci.shape[:-1])
        per_component_sums = jnp.sum(per_value_penalty(ci), axis=leading_axes)  # (C,)
        per_component_means = per_component_sums / n_positions  # f_c
        lp = lp + jnp.sum(per_component_means)
        if reference_datapoint_count is not None:
            a_prime = (
                n_positions if reference_datapoint_count == "auto" else reference_datapoint_count
            )
            freq = freq + jnp.sum(
                per_component_means * jnp.log2(1.0 + a_prime * per_component_means)
            )
    return lp, freq


@jaxtyped(typechecker=beartype)
def importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    pnorm: Float[Array, ""],
    eps: float,
    reference_datapoint_count: int | Literal["auto"] | None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """`L_p` imp-min terms: per-value penalty `(c + eps)^pnorm`, singular at `c=0` for
    `pnorm < 1` (the `eps` floor caps the gradient there)."""
    return _imp_min_terms(ci_upper, lambda ci: (ci + eps) ** pnorm, reference_datapoint_count)


@jaxtyped(typechecker=beartype)
def smooth_l0_importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    gamma: Float[Array, ""],
    reference_datapoint_count: int | Literal["auto"] | None,
    normalize_at_one: bool,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Geman–McClure smooth-L0 imp-min terms: per-value penalty `c^2 / (c^2 + gamma^2)`.
    Flat at the origin (`phi'(0)=0`) and bounded (`|phi'| <= 0.65/gamma`) — no singularity,
    no `eps` floor. Approaches the true `L_0` count as `gamma -> 0`.

    `normalize_at_one` rescales by `(1 + gamma^2)` so a fully-active component (`c = 1`)
    contributes exactly 1 at every gamma, removing the implicit ~2x coefficient ramp the
    bare form applies across a `1.0 -> 0.01` gamma anneal."""
    gamma_sq = gamma * gamma
    scale = 1.0 + gamma_sq if normalize_at_one else 1.0
    return _imp_min_terms(
        ci_upper, lambda ci: scale * ci**2 / (ci**2 + gamma_sq), reference_datapoint_count
    )


def annealed_imp_min_param(
    step_f32: Array, total_steps: int, cfg: AnyImportanceMinimalityLossConfig
) -> Array:
    """The scheduled per-value-penalty parameter at this step (`p` for `L_p`, `gamma` for
    smooth-L0; SPEC S9/S9′). Pure in the step, so the train step hoists it out of the
    loss `grad`."""
    match cfg:
        case ImportanceMinimalityLossConfig():
            schedule = cfg.pnorm
        case SmoothL0ImportanceMinimalityLossConfig():
            schedule = cfg.gamma
    return scheduled_value_traced(step_f32, total_steps, schedule)


def imp_min_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]],
    cfg: AnyImportanceMinimalityLossConfig,
    annealed_param: Array,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Dispatch `(lp, freq)` on the imp-min penalty kind, given its annealed parameter; the
    `freq` term is `0.0` unless `cfg.frequency` is configured."""
    ref = cfg.frequency.reference_datapoint_count if cfg.frequency is not None else None
    match cfg:
        case ImportanceMinimalityLossConfig():
            return importance_minimality_terms(ci_upper, annealed_param, cfg.eps, ref)
        case SmoothL0ImportanceMinimalityLossConfig():
            return smooth_l0_importance_minimality_terms(
                ci_upper, annealed_param, ref, cfg.normalize_at_one
            )
