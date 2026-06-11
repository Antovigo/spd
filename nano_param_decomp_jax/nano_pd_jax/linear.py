"""Mask-tolerant Linear shim.

Lets user models call `self.layer(x, mask)` uniformly across decomposed and
undecomposed sites — no isinstance dispatch needed. The undecomposed shim
asserts `mask is None` and returns `(out, x_in)`. After substitution, the same
call shape gives the decomposed forward (see decomposed.py).
"""

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class Linear(eqx.Module):
    inner: eqx.nn.Linear

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        self.inner = eqx.nn.Linear(d_in, d_out, use_bias=use_bias, key=key)

    def __call__(
        self,
        x: Float[Array, " d_in"],
        mask: Float[Array, " C"] | None = None,
    ) -> tuple[Float[Array, " d_out"], Float[Array, " d_in"]]:
        assert mask is None, "use DecomposedLinear at decomposed sites"
        return self.inner(x), x
