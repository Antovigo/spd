"""CI (causal-importance) functions: one small MLP per decomposed site.

Each site has its own `eqx.nn.MLP` mapping pre-weight activations of shape
`[..., d_in]` to logits of shape `[..., C]`. The logits are squashed via
`lower_leaky_hard_sigmoid` to produce ci in [0, 1].
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@dataclass(frozen=True)
class SiteSpec:
    """Static metadata about one decomposed site."""

    d_in: int
    d_out: int
    C: int


# A CIFn is just a dict of eqx.nn.MLP per site. Each MLP takes [d_in] -> [C]
# We vmap over leading batch dims when applying it.
CIFn = dict[str, eqx.nn.MLP]


def make_ci_fn(
    sites: dict[str, SiteSpec], hidden_size: int, key: PRNGKeyArray
) -> CIFn:
    keys = jax.random.split(key, len(sites))
    return {
        name: eqx.nn.MLP(
            in_size=spec.d_in,
            out_size=spec.C,
            width_size=hidden_size,
            depth=1,
            activation=jax.nn.gelu,
            key=k,
        )
        for (name, spec), k in zip(sorted(sites.items()), keys, strict=True)
    }


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Forward: clamp(x, 0, 1). Backward: pass-through on (0, 1); below 0 leaks
    only when grad_out < 0; above 1 zero. Matches the PyTorch reference."""
    return jnp.clip(x, 0.0, 1.0)


def _lower_leaky_fwd(x: Float[Array, "..."]) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    return jnp.clip(x, 0.0, 1.0), x


def _lower_leaky_bwd(
    res: Float[Array, "..."], g: Float[Array, "..."]
) -> tuple[Float[Array, "..."]]:
    x = res
    alpha = 0.01
    grad = jnp.where(
        x <= 0,
        jnp.where(g < 0, alpha * g, 0.0),
        jnp.where(x <= 1, g, 0.0),
    )
    return (grad,)


lower_leaky_hard_sigmoid.defvjp(_lower_leaky_fwd, _lower_leaky_bwd)


def apply_ci_fn(
    ci_fn: CIFn, pre_acts: dict[str, Float[Array, "... d_in"]]
) -> dict[str, Float[Array, "... C"]]:
    """Apply each per-site MLP over all leading dims, then sigmoid-squash to [0, 1]."""
    out: dict[str, Float[Array, "... C"]] = {}
    for name, mlp in ci_fn.items():
        x = pre_acts[name]
        # eqx.nn.MLP expects a single example; vmap over leading dims.
        flat = x.reshape(-1, x.shape[-1])
        logits_flat = jax.vmap(mlp)(flat)
        logits = logits_flat.reshape(*x.shape[:-1], -1)
        out[name] = lower_leaky_hard_sigmoid(logits)
    return out
