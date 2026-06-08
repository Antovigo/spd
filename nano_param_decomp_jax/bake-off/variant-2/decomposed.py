"""The `Decomposed` pytree leaf and the polymorphic `linop`.

Variant 2's central architectural idea: decomposition is represented by a typed leaf
in the params pytree. The user's forward is written ONCE in terms of `linop(leaf, x,
mask)`. When `leaf` is a raw `jax.Array`, `linop` does the plain `x @ W` matmul.
When it's a `Decomposed`, it dispatches to the masked rank-decomposed forward.

Two pytrees with identical structure:
    target_params      : {"W1": Array, "W2": Array, ...}
    decomposed_params  : {"W1": Decomposed(V, U, W_delta), "W2": Decomposed(...), ...}
"""

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class Decomposed(eqx.Module):
    """V/U rank-C decomposition of a `[d_in, d_out]` weight plus a frozen residual.

    Forward: `y = ((x @ V) * mask) @ U + x @ W_delta`. `W_delta` is initialised to
    `W_target - V @ U` at decomposition time and never trained — the trainer routes
    only V/U into the main optimizer.
    """

    V: Float[Array, "d_in C"]
    U: Float[Array, "C d_out"]
    W_delta: Float[Array, "d_in d_out"]

    @property
    def C(self) -> int:
        return self.V.shape[1]

    @property
    def d_in(self) -> int:
        return self.V.shape[0]

    @property
    def d_out(self) -> int:
        return self.U.shape[1]


def init_decomposed(
    key: PRNGKeyArray, W_target: Float[Array, "d_in d_out"], C: int
) -> Decomposed:
    d_in, d_out = W_target.shape
    kV, kU = jax.random.split(key)
    V = jax.random.normal(kV, (d_in, C)) / jnp.sqrt(d_in)
    U = jax.random.normal(kU, (C, d_out)) / jnp.sqrt(C)
    W_delta = W_target - V @ U
    return Decomposed(V=V, U=U, W_delta=W_delta)


# A param-tree leaf is either a raw weight (target mode) or a Decomposed struct.
Leaf: TypeAlias = Float[Array, "..."] | Decomposed


def linop(
    leaf: Leaf,
    x: Float[Array, "... d_in"],
    mask: Float[Array, "... C"] | None = None,
) -> Float[Array, "... d_out"]:
    """Polymorphic linear op. Dispatches on leaf type.

    - `Decomposed` leaf: requires a mask. Returns `((x @ V) * mask) @ U + x @ W_delta`.
    - Plain array leaf: target mode. Returns `x @ leaf`.
    """
    if isinstance(leaf, Decomposed):
        assert mask is not None, "mask is required for Decomposed leaf"
        comp_acts = x @ leaf.V
        comp_out = (comp_acts * mask) @ leaf.U
        delta_out = x @ leaf.W_delta
        return comp_out + delta_out
    assert mask is None, "mask must be None for non-Decomposed leaf"
    return x @ leaf


def is_decomposed(x) -> bool:
    return isinstance(x, Decomposed)


def decomposed_sites(params) -> dict[str, Decomposed]:
    """Pull out every Decomposed leaf in the (flat-dict) params pytree, keyed by name."""
    return {k: v for k, v in params.items() if isinstance(v, Decomposed)}
