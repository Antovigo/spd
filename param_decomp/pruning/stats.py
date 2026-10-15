"""Per-prompt last-position quantities, the stratified estimator and its moment sums.

Notation (plan §1): `y` is the answer token, `r` the true answer (an integer in 0..999),
`V_int` the 1,000 tokens "0".."999" in integer order, so the integer-conditional distribution
`q(x) = p(x) / Σ_{x' ∈ V_int} p(x')` is indexed by the integer `x` itself."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

N_INT = 1000
"""`|V_int|`: the integers 0..999."""
N_REL = 2 * N_INT - 1
"""Answer-relative offsets `d = x - r` in -999..999."""

QUANTITIES = (
    "ce",
    "ce_int",
    "p_int",
    "entropy",
    "entropy_int",
    "kl_target",
    "kl_base",
    "correct",
)
"""Per-prompt scalars, in this order along the quantity axis:
`ce` = -log p(y) over the full vocabulary; `ce_int` = -log q(r); `p_int` = Σ_{V_int} p;
`entropy` / `entropy_int` = entropy of p / of q (nats); `kl_target` = KL(p_target ‖ p);
`kl_base` = KL(p_base ‖ p) (0 for the base itself); `correct` = 1.0 iff argmax_x q(x) = r."""
N_QUANT = len(QUANTITIES)
POWERS = 3
"""Moment sums Σ f, Σ f², Σ f³."""


class Rows(NamedTuple):
    scalars: Float[Array, "Q B"]
    """`QUANTITIES` per prompt."""
    q: Float[Array, "B 1000"]
    """The integer-conditional distribution."""


def log_softmax32(logits: Array) -> Float[Array, "B vocab"]:
    return jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)


def kl(logp_ref: Array, logp: Array) -> Float[Array, " B"]:
    """`KL(p_ref ‖ p)` per row."""
    return jnp.sum(jnp.exp(logp_ref) * (logp_ref - logp), axis=-1)


def rows(
    logp: Float[Array, "B vocab"],
    logp_target: Float[Array, "B vocab"],
    logp_base: Float[Array, "B vocab"],
    y: Int[Array, " B"],
    r: Int[Array, " B"],
    int_ids: Int[Array, " 1000"],
) -> Rows:
    lp_int = jnp.take(logp, int_ids, axis=-1)
    lse = jax.nn.logsumexp(lp_int, axis=-1)
    log_q = lp_int - lse[:, None]
    q = jnp.exp(log_q)
    p = jnp.exp(logp)
    scalars = jnp.stack(
        [
            -jnp.take_along_axis(logp, y[:, None], axis=1)[:, 0],
            -jnp.take_along_axis(log_q, r[:, None], axis=1)[:, 0],
            jnp.exp(lse),
            -jnp.sum(p * logp, axis=-1),
            -jnp.sum(q * log_q, axis=-1),
            kl(logp_target, logp),
            kl(logp_base, logp),
            (jnp.argmax(log_q, axis=-1) == r).astype(jnp.float32),
        ]
    )
    return Rows(scalars=scalars, q=q)


def power_sums(values: Float[Array, "... B"], weight: Float[Array, " B"]) -> Float[Array, "... 3"]:
    """`Σ_b w_b v_b^k` for `k = 1, 2, 3`."""
    return jnp.stack([jnp.sum(weight * values**k, axis=-1) for k in (1, 2, 3)], axis=-1)


def shift_sums(
    dq: Float[Array, "B 1000"], r: Int[Array, " B"], weight: Float[Array, " B"]
) -> tuple[Float[Array, " 1000"], Float[Array, " 1999"]]:
    """`Σ_b w_b Δq_b(x)` per integer `x`, and the same re-indexed by `d = x - r_b` (offset by
    999 so `d = -999` is entry 0)."""
    by_x = jnp.sum(weight[:, None] * dq, axis=0)
    rel_index = jnp.arange(N_INT)[None, :] - r[:, None] + (N_INT - 1)
    by_rel = jnp.zeros((N_REL,), jnp.float32).at[rel_index].add(weight[:, None] * dq)
    return by_x, by_rel


# ----------------------------------- the stratified estimator -----------------------------------


def stratified(
    n_total: int,
    sum_h: np.ndarray,
    sum_r: np.ndarray,
    sum_rh: np.ndarray,
    n_h: int,
    n_r_minus_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The mean over the population and its standard error, from power sums `(..., 3)`:

        F̂ = (1/N) [ Σ_H f + w Σ_{R∖H} f ],   w = (N − n_H) / |R ∖ H|
        SE = ((N − n_H)/N) · sd_{R∖H}(f) / √|R ∖ H|

    `sum_h` over the stratum H (scored exactly), `sum_r` over the shared sample R, `sum_rh` over
    `R ∩ H` (so `Σ_{R∖H} = sum_r − sum_rh`); `n_r_minus_h` broadcasts against the leading axes."""
    rest = sum_r - sum_rh
    m = n_r_minus_h.astype(np.float64)
    w = (n_total - n_h) / m
    mean = (sum_h[..., 0] + w * rest[..., 0]) / n_total
    rest_mean = rest[..., 0] / m
    rest_var = np.maximum(rest[..., 1] / m - rest_mean**2, 0.0) * m / np.maximum(m - 1.0, 1.0)
    se = (n_total - n_h) / n_total * np.sqrt(rest_var) / np.sqrt(m)
    return mean, se


def weighted_moments(
    n_total: int,
    sum_h: np.ndarray,
    sum_r: np.ndarray,
    sum_rh: np.ndarray,
    n_h: int,
    n_r_minus_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population mean, sd and skewness estimated with the stratum weights (1 on H, `w` on
    `R ∖ H`) from the same power sums."""
    rest = sum_r - sum_rh
    m = n_r_minus_h.astype(np.float64)
    w = (n_total - n_h) / m
    raw = [(sum_h[..., k] + w * rest[..., k]) / n_total for k in range(3)]
    mean = raw[0]
    var = np.maximum(raw[1] - mean**2, 0.0)
    third = raw[2] - 3 * mean * raw[1] + 2 * mean**3
    sd = np.sqrt(var)
    skew = np.where(sd > 0, third / np.maximum(sd, 1e-30) ** 3, 0.0)
    return mean, sd, skew


def weighted_quantiles(
    values_h: np.ndarray, values_rest: np.ndarray, w: float, qs: tuple[float, ...]
) -> np.ndarray:
    """Quantiles of the population estimated from H (weight 1) and `R ∖ H` (weight `w`)."""
    values = np.concatenate([values_h, values_rest])
    weights = np.concatenate([np.ones(values_h.size), np.full(values_rest.size, w)])
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    return np.asarray([values[order][min(np.searchsorted(cdf, q), values.size - 1)] for q in qs])
