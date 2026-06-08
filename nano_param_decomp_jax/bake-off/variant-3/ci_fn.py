"""CI function: per-site MLP mapping pre-weight activations to ci ∈ [0, 1]^C.

Each site has its own small MLP. The CI fn pytree mirrors the site keys so
optimizer state lines up trivially. The sigmoid is `lower_leaky_hard_sigmoid`:
forward is `clamp(x, 0, 1)`, backward leaks linearly below zero when the upstream
grad is negative (so dead components can be resurrected). JAX's `custom_vjp`
expresses this directly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.clip(x, 0.0, 1.0)


def _lhs_fwd(x: Float[Array, "..."]):
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_bwd(x: Float[Array, "..."], g: Float[Array, "..."]):
    alpha = 0.01
    leak = jnp.where(g < 0, alpha * g, 0.0)
    grad = jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0))
    return (grad,)


lower_leaky_hard_sigmoid.defvjp(_lhs_fwd, _lhs_bwd)


class SiteCI(eqx.Module):
    """Per-site CI MLP: d_in -> hidden -> C, GELU, then lower_leaky_hard_sigmoid."""

    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear

    def __init__(self, d_in: int, hidden: int, C: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.proj_in = eqx.nn.Linear(d_in, hidden, key=k1)
        self.proj_out = eqx.nn.Linear(hidden, C, key=k2)

    def __call__(self, x: Float[Array, " d_in"]) -> Float[Array, " C"]:
        h = jax.nn.gelu(self.proj_in(x))
        logits = self.proj_out(h)
        return lower_leaky_hard_sigmoid(logits)


class CIFn(eqx.Module):
    """Bundle of per-site CI MLPs, keyed by site path."""

    sites: dict[str, SiteCI]

    def __init__(self, d_in_per_site: dict[str, int], C_per_site: dict[str, int], hidden: int, *, key: PRNGKeyArray):
        keys = jax.random.split(key, len(d_in_per_site))
        self.sites = {
            name: SiteCI(d_in_per_site[name], hidden, C_per_site[name], key=k)
            for k, name in zip(keys, sorted(d_in_per_site))
        }

    def __call__(self, acts: dict[str, Float[Array, "... d_in"]]) -> dict[str, Float[Array, "... C"]]:
        """acts maps site -> [..., d_in]; vmaps the per-example SiteCI over leading dims."""
        out: dict[str, Float[Array, "... C"]] = {}
        for name, site in self.sites.items():
            a = acts[name]
            leading = a.shape[:-1]
            flat = a.reshape(-1, a.shape[-1])
            ci_flat = jax.vmap(site)(flat)
            out[name] = ci_flat.reshape(*leading, ci_flat.shape[-1])
        return out
