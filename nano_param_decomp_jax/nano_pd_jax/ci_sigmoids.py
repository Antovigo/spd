"""Sigmoid variants for CI. v1 has only `lower_leaky_hard_sigmoid`.

Forward: clamp(x, 0, 1). Backward: pass-through inside (0, 1); zero above 1;
below 0 leak `alpha * g` ONLY when upstream `g < 0` (so dead components can be
resurrected by a 'turn me on' gradient signal, but not pushed further negative).
Matches `param_decomp/ci_sigmoids.py:49`.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.clip(x, 0.0, 1.0)


def _fwd(x):
    return jnp.clip(x, 0.0, 1.0), x


def _bwd(x, g):
    alpha = 0.01
    leak = jnp.where(g < 0, alpha * g, 0.0)
    grad = jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0))
    return (grad,)


lower_leaky_hard_sigmoid.defvjp(_fwd, _bwd)
