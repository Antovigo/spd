"""The CI ceiling of a fine-tune: a frozen CI fn whose CI the trained fn's may not exceed.

`cap_ci` takes the entrywise minimum of the trained CI and the ceiling's, on every leaf of the
bundle (preactivations, lower, upper; both heads of a dual fn). Its gradient passes at or
under the cap; over it, only a gradient that LOWERS the trained value does, so an entry pushed
over the cap by one term can still be pulled back by another (imp-min) instead of freezing
there. The ceiling never receives a gradient.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from param_decomp.core.ci_fn.interface import AnyCI


@jax.custom_vjp
def _capped(x: Array, cap: Array) -> Array:
    return jnp.minimum(x, cap)


def _capped_fwd(x: Array, cap: Array) -> tuple[Array, tuple[Array, Array]]:
    return jnp.minimum(x, cap), (x, cap)


def _capped_bwd(res: tuple[Array, Array], g: Array) -> tuple[Array, Array]:
    x, cap = res
    return jnp.where((x <= cap) | (g > 0), g, jnp.zeros_like(g)), jnp.zeros_like(cap)


_capped.defvjp(_capped_fwd, _capped_bwd)


def cap_ci[C: AnyCI](ci: C, ceiling: C) -> C:
    """`ci` capped entrywise by `ceiling` (same bundle structure, full-emission sites only:
    a block-selected site's picks need not index the same blocks in both)."""
    for leaf in jax.tree.leaves(ci):
        assert isinstance(leaf, jax.Array), type(leaf)
    return jax.tree.map(
        lambda x, cap: _capped(x, jax.lax.stop_gradient(cap).astype(x.dtype)), ci, ceiling
    )
