"""Stochastic mask sampling: m = ci + (1 - ci) * u with u ~ U[0, 1].

Splits the PRNG key per site so each site gets independent noise.
"""

import jax
import jax.random
from jaxtyping import Array, Float, PRNGKeyArray


def sample_masks(
    key: PRNGKeyArray,
    ci: dict[str, Float[Array, "... C"]],
) -> dict[str, Float[Array, "... C"]]:
    names = sorted(ci)
    keys = jax.random.split(key, len(names))
    return {
        name: ci[name] + (1.0 - ci[name]) * jax.random.uniform(k, ci[name].shape)
        for k, name in zip(keys, names, strict=True)
    }
