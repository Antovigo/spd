"""Per-site CI functions: small MLPs mapping pre-weight activations -> CI in [0, 1]^C.

Each decomposed site gets its own MLP. We bundle them in a dict keyed by site name —
matching the params pytree's keying — so the trainer can build masks site-by-site.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from decomposed import Decomposed
from jaxtyping import Array, Float, PRNGKeyArray

CI_HIDDEN = 64
LEAKY_ALPHA = 0.01


def _lower_leaky_hard_sigmoid_fwd(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.clip(x, 0.0, 1.0)


def _lower_leaky_hard_sigmoid_bwd(
    alpha: float, x: Float[Array, "..."], g: Float[Array, "..."]
) -> tuple[Float[Array, "..."]]:
    leak = jnp.where(g < 0, alpha * g, 0.0)
    in_range = jnp.where(x <= 1, g, 0.0)
    grad = jnp.where(x <= 0, leak, in_range)
    return (grad,)


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Forward: clip(x, 0, 1). Backward: pass-through inside [0,1]; alpha*g below 0
    only when g<0; zero above 1. Matches `_LowerLeakyHardSigmoid` in nano reference."""
    return _lower_leaky_hard_sigmoid_fwd(x)


def _llhs_fwd(x):
    return _lower_leaky_hard_sigmoid_fwd(x), x


def _llhs_bwd(res, g):
    x = res
    return _lower_leaky_hard_sigmoid_bwd(LEAKY_ALPHA, x, g)


lower_leaky_hard_sigmoid.defvjp(_llhs_fwd, _llhs_bwd)


class SiteCI(eqx.Module):
    """Small MLP: pre-weight acts [..., d_in] -> CI [..., C] in [0, 1] via leaky sigmoid."""

    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(self, d_in: int, C: int, hidden: int, key: PRNGKeyArray) -> None:
        k1, k2 = jax.random.split(key)
        self.linear_in = eqx.nn.Linear(d_in, hidden, key=k1)
        self.linear_out = eqx.nn.Linear(hidden, C, key=k2)

    def __call__(self, x: Float[Array, "... d_in"]) -> Float[Array, "... C"]:
        # eqx.nn.Linear is single-vector; vmap over leading dims.
        def per_token(v: Float[Array, " d_in"]) -> Float[Array, " C"]:
            h = jax.nn.gelu(self.linear_in(v))
            return self.linear_out(h)

        flat = x.reshape(-1, x.shape[-1])
        out = jax.vmap(per_token)(flat).reshape(*x.shape[:-1], -1)
        return lower_leaky_hard_sigmoid(out)


def init_ci_fns(
    key: PRNGKeyArray, decomposed: dict[str, Decomposed], hidden: int = CI_HIDDEN
) -> dict[str, SiteCI]:
    """One SiteCI per Decomposed site, each sized (d_in_site -> C_site)."""
    names = sorted(decomposed.keys())
    keys = jax.random.split(key, len(names))
    return {
        name: SiteCI(d_in=decomposed[name].d_in, C=decomposed[name].C, hidden=hidden, key=k)
        for name, k in zip(names, keys)
    }


def compute_ci(
    ci_fns: dict[str, SiteCI], pre_acts: dict[str, Float[Array, "... d_in"]]
) -> dict[str, Float[Array, "... C"]]:
    """Run every site's CI fn on its cached pre-weight activations."""
    return {name: ci_fns[name](pre_acts[name]) for name in ci_fns}
