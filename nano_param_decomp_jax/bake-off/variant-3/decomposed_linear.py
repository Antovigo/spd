"""DecomposedLinear: an eqx.Module that replaces an eqx.nn.Linear at a decomposed site.

The decomposed forward is `y = ((x @ V) * m) @ U + x @ W_delta + bias`, where
`W_delta = W_target - V_init @ U_init` is frozen at init. We store `W_target`
explicitly as a frozen array so the faithfulness loss can compute the live residual
`W_target - V @ U`.

Masks are threaded in through __call__ — the user's model must accept a
`masks: dict | None` arg and route the per-site mask in.
"""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class DecomposedLinear(eqx.Module):
    """Rank-C decomposition substituted in place of an eqx.nn.Linear.

    V: [d_in, C], U: [C, d_out] — trainable.
    W_target: [d_in, d_out] — frozen, the original target weight (transposed from
              eqx.nn.Linear's [d_out, d_in] storage).
    W_delta: [d_in, d_out] — frozen residual `W_target - V_init @ U_init`.
    bias: [d_out] or None — frozen, copied from target.

    __call__(x, mask) takes a single example x of shape [d_in] (Equinox's Linear
    convention — vmap for batches) and a mask of shape [C]. If mask is None the layer
    reverts to the target forward (used for target-output collection and pre-weight
    activation gathering).
    """

    V: Float[Array, "d_in C"]
    U: Float[Array, "C d_out"]
    W_target: Float[Array, "d_in d_out"]
    W_delta: Float[Array, "d_in d_out"]
    bias: Float[Array, " d_out"] | None
    d_in: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    C: int = eqx.field(static=True)

    def __init__(self, linear: eqx.nn.Linear, C: int, *, key: PRNGKeyArray):
        d_out, d_in = linear.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.C = C
        key_v, key_u = jax.random.split(key)
        self.V = jax.random.normal(key_v, (d_in, C)) / jnp.sqrt(d_in)
        self.U = jax.random.normal(key_u, (C, d_out)) / jnp.sqrt(C)
        self.W_target = linear.weight.T
        self.W_delta = self.W_target - self.V @ self.U
        self.bias = cast(Float[Array, " d_out"] | None, linear.bias)

    def __call__(
        self, x: Float[Array, " d_in"], mask: Float[Array, " C"] | None = None
    ) -> Float[Array, " d_out"]:
        if mask is None:
            y = x @ self.W_target
        else:
            y = ((x @ self.V) * mask) @ self.U + x @ self.W_delta
        if self.bias is not None:
            y = y + self.bias
        return y


def substitute_decomposed(
    model: eqx.Module,
    site_paths: dict[str, int],
    *,
    key: PRNGKeyArray,
) -> eqx.Module:
    """Replace each `eqx.nn.Linear` at the named attribute path with a DecomposedLinear.

    `site_paths` maps a dotted attribute path (e.g. "layer1" or "layer2.up") to the
    component count C for that site. Uses `eqx.tree_at` for the substitution.
    """
    keys = jax.random.split(key, len(site_paths))
    for k, (path, C) in zip(keys, site_paths.items()):
        target = _get_by_path(model, path)
        assert isinstance(target, eqx.nn.Linear), f"{path} is not eqx.nn.Linear: {type(target)}"
        replacement = DecomposedLinear(target, C, key=k)
        model = eqx.tree_at(_path_getter(path), model, replacement)
    return model


def _get_by_path(obj: object, path: str) -> object:
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _path_getter(path: str):
    def getter(m: object) -> object:
        return _get_by_path(m, path)

    return getter
