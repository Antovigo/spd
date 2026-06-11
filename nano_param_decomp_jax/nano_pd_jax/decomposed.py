"""DecomposedLinear + substitution machinery.

Same `__call__(x, mask=None) -> (out, x_in)` signature as the Linear shim.
- mask=None: target forward `x @ W_target + bias`
- mask=m  : decomposed forward `((x @ V) * m) @ U + x @ W_delta + bias`,
            with `W_delta = W_target - V @ U` recomputed each call (matches
            nano reference `ComponentLinear.weight_delta`).

Substitution is structural — `eqx.tree_at` swaps a `Linear` for a
`DecomposedLinear` at the named attribute path. The user model's forward is
unchanged.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nano_pd_jax.linear import Linear


class DecomposedLinear(eqx.Module):
    V: Float[Array, "d_in C"]
    U: Float[Array, "C d_out"]
    W_target: Float[Array, "d_in d_out"]
    bias: Float[Array, " d_out"] | None
    d_in: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    C: int = eqx.field(static=True)

    def __init__(self, linear: Linear, C: int, *, key: PRNGKeyArray):
        d_out, d_in = linear.inner.weight.shape
        self.d_in = d_in
        self.d_out = d_out
        self.C = C
        kV, kU = jax.random.split(key)
        self.V = jax.random.normal(kV, (d_in, C)) / jnp.sqrt(d_in)
        self.U = jax.random.normal(kU, (C, d_out)) / jnp.sqrt(C)
        self.W_target = linear.inner.weight.T
        self.bias = linear.inner.bias

    def __call__(
        self,
        x: Float[Array, " d_in"],
        mask: Float[Array, " C"] | None = None,
    ) -> tuple[Float[Array, " d_out"], Float[Array, " d_in"]]:
        if mask is None:
            y = x @ self.W_target
        else:
            W_delta = self.W_target - self.V @ self.U
            y = ((x @ self.V) * mask) @ self.U + x @ W_delta
        if self.bias is not None:
            y = y + self.bias
        return y, x


def substitute_decomposed(
    model: eqx.Module,
    site_paths: dict[str, int],
    *,
    key: PRNGKeyArray,
) -> eqx.Module:
    keys = jax.random.split(key, len(site_paths))
    for k, (path, C) in zip(keys, site_paths.items(), strict=True):
        target = get_by_path(model, path)
        assert isinstance(target, Linear), f"{path} is not a nano_pd_jax.Linear: {type(target)}"
        replacement = DecomposedLinear(target, C, key=k)
        model = eqx.tree_at(_path_getter(path), model, replacement)
    return model


def collect_site_paths(model: eqx.Module) -> list[str]:
    """Dotted paths to every DecomposedLinear. Tuples/lists use integer indices."""
    paths: list[str] = []

    def visit(prefix: str, node: object) -> None:
        if isinstance(node, DecomposedLinear):
            paths.append(prefix)
            return
        if isinstance(node, eqx.Module):
            for f in node.__class__.__dataclass_fields__:
                visit(f"{prefix}.{f}" if prefix else f, getattr(node, f))
            return
        if isinstance(node, (tuple, list)):
            for i, child in enumerate(node):
                visit(f"{prefix}.{i}" if prefix else str(i), child)

    visit("", model)
    return sorted(paths)


def get_by_path(obj: object, path: str) -> object:
    for attr in path.split("."):
        obj = obj[int(attr)] if isinstance(obj, tuple | list) else getattr(obj, attr)
    return obj


def _path_getter(path: str):
    def getter(m: object) -> object:
        return get_by_path(m, path)

    return getter
