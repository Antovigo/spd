"""Jitted passes over the factored model: the shared-sample cache, single-component flips on it,
per-component `H_c` batches, full-`P` scoring and the mask gradient.

Sign convention: a flip changes one component (`Flips.sign = -1` removes an on one, `+1` adds an
off one). Every returned difference is `δf = f(without c) − f(with c)`, i.e. `-sign · (f_flip −
f_base)`, so `δ` of `ce` is positive when the component helps, whichever direction was flipped.
`Δq` follows the same convention (`q_without − q_with`)."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from param_decomp.pruning.model import (
    Arch,
    DenseWeights,
    Factors,
    Flips,
    Mask,
    Precision,
    Shared,
    block_inputs,
    dense_logits,
    factored_logits,
    last_logits,
    run_blocks,
)
from param_decomp.pruning.stats import (
    N_QUANT,
    N_REL,
    Rows,
    log_softmax32,
    power_sums,
    rows,
    shift_sums,
)


class SampleCache(eqx.Module):
    """One chunk of the shared sample `R` under the current global mask."""

    inputs: Float[Array, "L B T d"]
    """The residual entering every block."""
    logp_base: Float[Array, "B vocab"]
    logp_target: Float[Array, "B vocab"]
    base: Rows
    y: Int[Array, " B"]
    r: Int[Array, " B"]


@eqx.filter_jit
def make_sample_cache(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    logp_target: Float[Array, "B vocab"],
    tokens: Int[Array, "B T"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    site_precision: Precision,
    head_precision: Precision,
) -> SampleCache:
    inputs, final = block_inputs(arch, shared, factors, mask, tokens, site_precision)
    logp_base = log_softmax32(last_logits(arch, shared, final, head_precision))
    return SampleCache(
        inputs=inputs,
        logp_base=logp_base,
        logp_target=logp_target,
        base=rows(logp_base, logp_target, logp_base, y, r, int_ids),
        y=y,
        r=r,
    )


@eqx.filter_jit
def target_logp(
    arch: Arch,
    shared: Shared,
    dense: DenseWeights,
    tokens: Int[Array, "B T"],
    site_precision: Precision,
    head_precision: Precision,
) -> Float[Array, "B vocab"]:
    return log_softmax32(dense_logits(arch, shared, dense, tokens, site_precision, head_precision))


class FlipResult(eqx.Module):
    """One component flipped over a batch of prompts."""

    without: Float[Array, "Q B"]
    """`QUANTITIES` of the model WITHOUT the component."""
    delta: Float[Array, "Q B"]
    """`f(without) − f(with)` per prompt."""
    dq_x: Float[Array, " 1000"]
    """`Σ_b Δq_b(x)` over the batch."""
    dq_rel: Float[Array, " 1999"]
    """The same indexed by `d = x − r_b` (entry `d + 999`)."""


def _flip_rows(
    logp_flip: Array,
    logp_base: Array,
    logp_target: Array,
    base: Rows,
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    sign: Float[Array, ""] | Float[Array, " B"],
) -> tuple[Rows, Array, Array]:
    """The without-c rows and the δ (scalars and `Δq`) of a flip; `sign` is a scalar or one per
    row. For an addition the without-c model is the base, whose `kl_base` is 0."""
    sign = jnp.asarray(sign, jnp.float32)
    per_row = sign if sign.ndim == 1 else jnp.broadcast_to(sign, y.shape)
    flipped = rows(logp_flip, logp_target, logp_base, y, r, int_ids)
    delta = -per_row[None, :] * (flipped.scalars - base.scalars)
    dq = -per_row[:, None] * (flipped.q - base.q)
    without = jnp.where(per_row[None, :] < 0, flipped.scalars, base.scalars)
    return flipped._replace(scalars=without), delta, dq


@eqx.filter_jit
def flip_on_sample(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    cache: SampleCache,
    kind: Int[Array, ""],
    layer: Int[Array, ""],
    slot: Int[Array, ""],
    sign: Float[Array, ""],
    int_ids: Int[Array, " 1000"],
    site_precision: Precision,
    head_precision: Precision,
) -> FlipResult:
    """Flip one component on every prompt of a sample chunk, rerunning blocks `layer..L-1`."""
    batch = cache.y.shape[0]
    flips = Flips.same(batch, kind, layer, slot, sign)
    start = jax.lax.dynamic_index_in_dim(cache.inputs, layer, keepdims=False)
    resid = run_blocks(arch, shared, factors, mask, flips, start, layer, site_precision)
    logp = log_softmax32(last_logits(arch, shared, resid, head_precision))
    without, delta, dq = _flip_rows(
        logp, cache.logp_base, cache.logp_target, cache.base, cache.y, cache.r, int_ids, sign
    )
    dq_x, dq_rel = shift_sums(dq, cache.r, jnp.ones((batch,), jnp.float32))
    return FlipResult(without=without.scalars, delta=delta, dq_x=dq_x, dq_rel=dq_rel)


class GroupResult(eqx.Module):
    """`K` components, each flipped on its own `n` prompts (`H_c`)."""

    without: Float[Array, "K Q n"]
    delta: Float[Array, "K Q n"]
    sums: Float[Array, "K 2 2 Q 3"]
    """Power sums of (without, delta) over (all of H_c, H_c ∩ R)."""
    dq_x: Float[Array, "K 2 1000"]
    dq_rel: Float[Array, "K 2 1999"]


@eqx.filter_jit
def flip_on_groups(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    dense: DenseWeights | None,
    mask: Mask,
    tokens: Int[Array, "K n T"],
    y: Int[Array, "K n"],
    r: Int[Array, "K n"],
    in_sample: Float[Array, "K n"],
    kind: Int[Array, " K"],
    layer: Int[Array, " K"],
    slot: Int[Array, " K"],
    sign: Float[Array, " K"],
    int_ids: Int[Array, " 1000"],
    site_precision: Precision,
    head_precision: Precision,
    target_precision: Precision,
) -> GroupResult:
    """Each of `K` components flipped on its own prompts: full forwards of the base, the flipped
    model and (unless `dense` is None, then the base stands in) the target."""
    k, n, t = tokens.shape
    flat = tokens.reshape(k * n, t)
    yf, rf = y.reshape(-1), r.reshape(-1)
    rep = lambda a: jnp.repeat(a, n)  # noqa: E731
    flips = Flips(kind=rep(kind), layer=rep(layer), slot=rep(slot), sign=rep(sign))
    none = Flips.none(k * n)
    logp_base = log_softmax32(
        factored_logits(arch, shared, factors, mask, none, flat, site_precision, head_precision)
    )
    logp_flip = log_softmax32(
        factored_logits(arch, shared, factors, mask, flips, flat, site_precision, head_precision)
    )
    logp_tgt = (
        logp_base
        if dense is None
        else log_softmax32(
            dense_logits(arch, shared, dense, flat, target_precision, target_precision)
        )
    )
    base = rows(logp_base, logp_tgt, logp_base, yf, rf, int_ids)
    without, delta, dq = _flip_rows(
        logp_flip, logp_base, logp_tgt, base, yf, rf, int_ids, flips.sign
    )
    without_k = without.scalars.reshape(N_QUANT, k, n).transpose(1, 0, 2)
    delta_k = delta.reshape(N_QUANT, k, n).transpose(1, 0, 2)
    dq_k = dq.reshape(k, n, -1)
    weights = jnp.stack([jnp.ones_like(in_sample), in_sample], axis=1)  # (K, 2, n)

    def group(w_both: Array, wo: Array, de: Array, dqg: Array, rg: Array):
        sums = jnp.stack(
            [
                jnp.stack([power_sums(wo, w), power_sums(de, w)])  # (2, Q, 3)
                for w in (w_both[0], w_both[1])
            ]
        )
        x_and_rel = [shift_sums(dqg, rg, w) for w in (w_both[0], w_both[1])]
        return (
            sums,
            jnp.stack([x for x, _ in x_and_rel]),
            jnp.stack([rel for _, rel in x_and_rel]),
        )

    sums, dq_x, dq_rel = jax.vmap(group)(weights, without_k, delta_k, dq_k, r)
    assert dq_rel.shape[-1] == N_REL
    return GroupResult(without=without_k, delta=delta_k, sums=sums, dq_x=dq_x, dq_rel=dq_rel)


@eqx.filter_jit
def score_prompts(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    dense: DenseWeights | None,
    tokens: Int[Array, "B T"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
    target_precision: Precision,
) -> Float[Array, "Q B"]:
    """`QUANTITIES` of `M_S` per prompt (`kl_base` = 0; `kl_target` = 0 without `dense`)."""
    none = Flips.none(tokens.shape[0])
    logp = log_softmax32(
        factored_logits(arch, shared, factors, mask, none, tokens, precision, precision)
    )
    logp_tgt = (
        logp
        if dense is None
        else log_softmax32(
            dense_logits(arch, shared, dense, tokens, target_precision, target_precision)
        )
    )
    return rows(logp, logp_tgt, logp, y, r, int_ids).scalars


@eqx.filter_jit
def score_target(
    arch: Arch,
    shared: Shared,
    dense: DenseWeights,
    tokens: Int[Array, "B T"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
) -> Float[Array, "Q B"]:
    """`QUANTITIES` of the target model itself (both KLs 0)."""
    logp = log_softmax32(dense_logits(arch, shared, dense, tokens, precision, precision))
    return rows(logp, logp, logp, y, r, int_ids).scalars


type GradObjective = Literal["ce", "ce_int", "kl_int"]


def int_log_q(
    logp: Float[Array, "B vocab"], int_ids: Int[Array, " 1000"]
) -> Float[Array, "B 1000"]:
    """`log q`: the next-token distribution renormalized over the integer tokens."""
    lp_int = jnp.take(logp, int_ids, axis=-1)
    return lp_int - jax.nn.logsumexp(lp_int, axis=-1, keepdims=True)


def kl_int(log_q_ref: Float[Array, "B 1000"], log_q: Float[Array, "B 1000"]) -> Float[Array, " B"]:
    """`KL(q_ref ‖ q)` over the integer tokens, per prompt."""
    return jnp.sum(jnp.exp(log_q_ref) * (log_q_ref - log_q), axis=-1)


@eqx.filter_jit
def kl_int_prompts(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    reference: Mask,
    tokens: Int[Array, "B T"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
) -> Float[Array, " B"]:
    """`KL(q_reference ‖ q_mask)` over the integer tokens, per prompt."""
    none = Flips.none(tokens.shape[0])

    def log_q(m: Mask) -> Array:
        logits = factored_logits(arch, shared, factors, m, none, tokens, precision, precision)
        return int_log_q(log_softmax32(logits), int_ids)

    return kl_int(log_q(reference), log_q(mask))


@eqx.filter_jit
def factored_logp(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    precision: Precision,
) -> Float[Array, "B vocab"]:
    none = Flips.none(tokens.shape[0])
    return log_softmax32(
        factored_logits(arch, shared, factors, mask, none, tokens, precision, precision)
    )


@eqx.filter_jit
def kl_four(
    logp: Float[Array, "B vocab"],
    logp_s0: Float[Array, "B vocab"],
    logp_llama: Float[Array, "B vocab"],
    int_ids: Int[Array, " 1000"],
) -> Float[Array, "4 B"]:
    """Per prompt: `KL(ref ‖ p)` over the integer tokens and over the full vocabulary, for
    `ref` = `S₀` then Llama, in that order."""
    lq = int_log_q(logp, int_ids)
    return jnp.stack(
        [
            kl_int(int_log_q(logp_s0, int_ids), lq),
            jnp.sum(jnp.exp(logp_s0) * (logp_s0 - logp), axis=-1),
            kl_int(int_log_q(logp_llama, int_ids), lq),
            jnp.sum(jnp.exp(logp_llama) * (logp_llama - logp), axis=-1),
        ]
    )


@eqx.filter_jit
def mask_gradient(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    reference: Mask,
    tokens: Int[Array, "B T"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    weight: Float[Array, " B"],
    precision: Precision,
    objective: GradObjective,
) -> Mask:
    """`∂/∂m Σ_b weight_b · f_b` (`weight` zero on padding rows), `f` the full-vocabulary `CE`
    of the answer (`ce`), the `CE` among the integer tokens (`ce_int`, `-log q(r)`), or
    `KL(q_reference ‖ q_m)` over the integer tokens (`kl_int`; `reference` is read only by
    it)."""
    none = Flips.none(tokens.shape[0])

    def logp_of(m: Mask) -> Array:
        return log_softmax32(
            factored_logits(arch, shared, factors, m, none, tokens, precision, precision)
        )

    def loss(m: Mask) -> Array:
        logp = logp_of(m)
        match objective:
            case "ce":
                f = -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0]
            case "ce_int":
                f = -jnp.take_along_axis(int_log_q(logp, int_ids), r[:, None], axis=1)[:, 0]
            case "kl_int":
                ref = jax.lax.stop_gradient(int_log_q(logp_of(reference), int_ids))
                f = kl_int(ref, int_log_q(logp, int_ids))
        return jnp.sum(weight * f)

    return jax.grad(loss)(mask)


class TemperatureFit(eqx.Module):
    """How much of one component's ablation a pure temperature change explains, per prompt."""

    kl: Float[Array, " B"]
    """`KL(p_base ‖ p_ablated)` at the last position."""
    r2: Float[Array, " B"]
    """Temperature R²: the fit `log p_base − log p_ablated ≈ c + (1 − β) z_base` over the
    vocabulary, weighted by `(p_base + p_ablated) / 2`, as an explained fraction of the
    weighted variance (`notes/rmsnorm/entropy-components.md`, regression version)."""
    beta: Float[Array, " B"]
    """The fitted inverse-temperature factor `β` (> 1 sharpens)."""


@eqx.filter_jit
def flip_temperature(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    cache: SampleCache,
    kind: Int[Array, ""],
    layer: Int[Array, ""],
    slot: Int[Array, ""],
    sign: Float[Array, ""],
    site_precision: Precision,
    head_precision: Precision,
) -> TemperatureFit:
    """Flip one component on a sample chunk and fit a temperature change to its effect. The base
    logits enter as `logp_base` (a per-prompt constant shift, absorbed by `c`)."""
    batch = cache.y.shape[0]
    flips = Flips.same(batch, kind, layer, slot, sign)
    start = jax.lax.dynamic_index_in_dim(cache.inputs, layer, keepdims=False)
    resid = run_blocks(arch, shared, factors, mask, flips, start, layer, site_precision)
    logp_a = log_softmax32(last_logits(arch, shared, resid, head_precision))
    return temperature_fit(cache.logp_base, logp_a)


def temperature_fit(
    logp_b: Float[Array, "B vocab"], logp_a: Float[Array, "B vocab"]
) -> TemperatureFit:
    """`TemperatureFit` of the change from `logp_b` (base) to `logp_a` (ablated)."""
    w = 0.5 * (jnp.exp(logp_b) + jnp.exp(logp_a))
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    d = logp_b - logp_a
    mz = jnp.sum(w * logp_b, axis=-1, keepdims=True)
    md = jnp.sum(w * d, axis=-1, keepdims=True)
    cov = jnp.sum(w * (logp_b - mz) * (d - md), axis=-1)
    var_z = jnp.sum(w * (logp_b - mz) ** 2, axis=-1)
    var_d = jnp.sum(w * (d - md) ** 2, axis=-1)
    return TemperatureFit(
        kl=jnp.sum(jnp.exp(logp_b) * d, axis=-1),
        r2=cov**2 / jnp.maximum(var_z * var_d, 1e-30),
        beta=1.0 - cov / jnp.maximum(var_z, 1e-30),
    )
