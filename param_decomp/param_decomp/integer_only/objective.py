"""Last-position scores over the integer tokens, and the jitted passes that compute them.

Notation (as `pool.py`): `V_int` is the 1000 tokens of the integers 0..999 in integer order,
`r` (integer in 0..999) a prompt's true answer and `y` its token. For a next-token distribution
`p` over the vocabulary, `q(x) = p(x) / Σ_{x' ∈ V_int} p(x')` is its renormalization over
`V_int`, indexed by the integer `x` (0..999). `KL_int(p_ref ‖ p) = Σ_x q_ref(x) (log q_ref(x) −
log q(x))` (nats). The reference `p_ref` is always the ORIGINAL target (Llama)."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from param_decomp.integer_only.config import Precision
from param_decomp.integer_only.model import (
    Arch,
    DenseWeights,
    Factors,
    Mask,
    Shared,
    dense_logits,
    factored_logits,
)

QUANTITIES = ("kl_int", "kl_full", "ce", "ce_int", "p_int", "entropy_int", "correct")
"""Per-prompt scores of a model `p` against the target `p_ref`, in this order: `kl_int` =
`KL_int(p_ref ‖ p)`; `kl_full` = `KL(p_ref ‖ p)` over the full vocabulary; `ce` = `−log p(y)`;
`ce_int` = `−log q(r)`; `p_int` = `Σ_{V_int} p`; `entropy_int` = entropy of `q` (nats);
`correct` = 1.0 iff `argmax_x q(x) = r` (the accuracy among the integers)."""


def log_softmax32(logits: Array) -> Float[Array, "B vocab"]:
    return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)


def int_log_q(
    logp: Float[Array, "B vocab"], int_ids: Int[Array, " 1000"]
) -> Float[Array, "B 1000"]:
    """`log q`: the next-token distribution renormalized over the integer tokens."""
    lp_int = jnp.take(logp, int_ids, axis=-1)
    return lp_int - jax.nn.logsumexp(lp_int, axis=-1, keepdims=True)


def kl_int(log_q_ref: Float[Array, "B 1000"], log_q: Float[Array, "B 1000"]) -> Float[Array, " B"]:
    """`KL(q_ref ‖ q)` over the integer tokens, per prompt."""
    return jnp.sum(jnp.exp(log_q_ref) * (log_q_ref - log_q), axis=-1)


def scores(
    logp: Float[Array, "B vocab"],
    logp_ref: Float[Array, "B vocab"],
    r: Int[Array, " B"],
    y: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
) -> Float[Array, "Q B"]:
    """`QUANTITIES` per prompt."""
    lp_int = jnp.take(logp, int_ids, axis=-1)
    lse = jax.nn.logsumexp(lp_int, axis=-1)
    log_q = lp_int - lse[:, None]
    return jnp.stack(
        [
            kl_int(int_log_q(logp_ref, int_ids), log_q),
            jnp.sum(jnp.exp(logp_ref) * (logp_ref - logp), axis=-1),
            -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0],
            -jnp.take_along_axis(log_q, r[:, None], axis=1)[:, 0],
            jnp.exp(lse),
            -jnp.sum(jnp.exp(log_q) * log_q, axis=-1),
            (jnp.argmax(log_q, axis=-1) == r).astype(jnp.float32),
        ]
    )


@eqx.filter_jit
def factored_logp(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    precision: Precision,
) -> Float[Array, "B vocab"]:
    return log_softmax32(factored_logits(arch, shared, factors, mask, tokens, precision, precision))


@eqx.filter_jit
def target_logp(
    arch: Arch,
    shared: Shared,
    dense: DenseWeights,
    tokens: Int[Array, "B T"],
    precision: Precision,
) -> Float[Array, "B vocab"]:
    return log_softmax32(dense_logits(arch, shared, dense, tokens, precision, precision))


@eqx.filter_jit
def target_log_q(
    arch: Arch,
    shared: Shared,
    dense: DenseWeights,
    tokens: Int[Array, "B T"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
) -> Float[Array, "B 1000"]:
    """The target's `log q` at the last position: the reference of every `KL_int`."""
    logp = log_softmax32(dense_logits(arch, shared, dense, tokens, precision, precision))
    return int_log_q(logp, int_ids)


@eqx.filter_jit
def probe_scores(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    log_q_ref: Float[Array, "B 1000"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
) -> Float[Array, "2 B"]:
    """Per prompt, `KL_int` to the target (from its cached `log q`) and `ce_int`."""
    logp = log_softmax32(factored_logits(arch, shared, factors, mask, tokens, precision, precision))
    log_q = int_log_q(logp, int_ids)
    ce_int = -jnp.take_along_axis(log_q, r[:, None], axis=1)[:, 0]
    return jnp.stack([kl_int(log_q_ref, log_q), ce_int])


@eqx.filter_jit
def mask_gradient(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    log_q_ref: Float[Array, "B 1000"],
    weight: Float[Array, " B"],
    int_ids: Int[Array, " 1000"],
    precision: Precision,
) -> Mask:
    """`∂/∂m Σ_b weight_b · KL_int(p_target ‖ p_m)` for prompts `b` of the batch (`weight` zero
    on padding rows), with the target's `log q` given (`log_q_ref`)."""

    def loss(m: Mask) -> Array:
        logits = factored_logits(arch, shared, factors, m, tokens, precision, precision)
        log_q = int_log_q(log_softmax32(logits), int_ids)
        return jnp.sum(weight * kl_int(log_q_ref, log_q))

    return jax.grad(loss)(mask)
