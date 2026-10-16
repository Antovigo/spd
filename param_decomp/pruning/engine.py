"""Jitted passes over the factored model: the shared-sample cache, single-component flips on it,
per-component `H_c` batches, full-`P` scoring and the mask gradient.

Sign convention: a flip changes one component (`Flips.sign = -1` removes an on one, `+1` adds an
off one). Every returned difference is `δf = f(without c) − f(with c)`, i.e. `-sign · (f_flip −
f_base)`, so `δ` of `ce` is positive when the component helps, whichever direction was flipped.
`Δq` follows the same convention (`q_without − q_with`)."""

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


@eqx.filter_jit
def mask_gradient(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    weight: Float[Array, " B"],
    precision: Precision,
    integer: bool,
) -> Mask:
    """`∂/∂m Σ_b weight_b · CE_b` (`weight` zero on padding rows); with `integer`, the `CE`
    among the integer tokens (`-log q(r)`)."""
    none = Flips.none(tokens.shape[0])

    def loss(m: Mask) -> Array:
        logp = log_softmax32(
            factored_logits(arch, shared, factors, m, none, tokens, precision, precision)
        )
        if integer:
            lp_int = jnp.take(logp, int_ids, axis=-1)
            log_q = lp_int - jax.nn.logsumexp(lp_int, axis=-1, keepdims=True)
            ce = -jnp.take_along_axis(log_q, r[:, None], axis=1)[:, 0]
        else:
            ce = -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0]
        return jnp.sum(weight * ce)

    return jax.grad(loss)(mask)
