"""Per-site CI MLP.

Each decomposed site gets its own `d_in → hidden → C` MLP with GELU and
lower-leaky-hard sigmoid. CIFn is a dict-of-MLPs keyed by site path; calling
it on an acts dict produces a CI dict with matching keys.

SiteCI operates on single d_in vectors; CIFn flattens arbitrary leading dims
(batch, seq, ...) before vmapping the per-site MLP, then restores them. The
trainer therefore calls `ci(acts)` directly — no outer vmap.
"""

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from nano_pd_jax.ci_sigmoids import lower_leaky_hard_sigmoid


class SiteCI(eqx.Module):
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear

    def __init__(self, d_in: int, hidden: int, C: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.proj_in = eqx.nn.Linear(d_in, hidden, key=k1)
        self.proj_out = eqx.nn.Linear(hidden, C, key=k2)

    def __call__(self, x: Float[Array, " d_in"]) -> Float[Array, " C"]:
        h = jax.nn.gelu(self.proj_in(x))
        return lower_leaky_hard_sigmoid(self.proj_out(h))


class CIFn(eqx.Module):
    sites: dict[str, SiteCI]

    def __init__(
        self,
        d_in_per_site: dict[str, int],
        C_per_site: dict[str, int],
        hidden: int,
        *,
        key: PRNGKeyArray,
    ):
        names = sorted(d_in_per_site)
        keys = jax.random.split(key, len(names))
        self.sites = {
            name: SiteCI(d_in_per_site[name], hidden, C_per_site[name], key=k)
            for k, name in zip(keys, names, strict=True)
        }

    def __call__(
        self,
        acts: dict[str, Float[Array, "... d_in"]],
    ) -> dict[str, Float[Array, "... C"]]:
        out: dict[str, Float[Array, "... C"]] = {}
        for name, site in self.sites.items():
            a = acts[name]
            leading = a.shape[:-1]
            flat = a.reshape(-1, a.shape[-1])
            ci_flat = jax.vmap(site)(flat)
            out[name] = ci_flat.reshape(*leading, ci_flat.shape[-1])
        return out
