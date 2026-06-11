"""The three SPD losses.

- faithfulness: mean squared `W_target - V @ U` across all decomposed sites
- importance_minimality: mean(ci^p) across sites (p ~ 0.9 for sparsity)
- stochastic_recon: MSE between decomposed-forward output and target output
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from nano_pd_jax.decomposed import DecomposedLinear, get_by_path


def faithfulness_loss(model: eqx.Module, site_paths: list[str]) -> Float[Array, ""]:
    sum_sq = jnp.array(0.0)
    numel = 0
    for path in site_paths:
        site = get_by_path(model, path)
        assert isinstance(site, DecomposedLinear)
        residual = site.W_target - site.V @ site.U
        sum_sq = sum_sq + (residual**2).sum()
        numel += residual.size
    return sum_sq / numel


def importance_minimality_loss(
    ci: dict[str, Float[Array, "... C"]], p: float = 0.9
) -> Float[Array, ""]:
    per_site = [jnp.mean(jnp.clip(v, 0.0, 1.0) ** p) for v in ci.values()]
    return jnp.mean(jnp.stack(per_site))


def stochastic_recon_loss(
    decomp_out: Float[Array, "B d_out"],
    target_out: Float[Array, "B d_out"],
) -> Float[Array, ""]:
    return jnp.mean((decomp_out - target_out) ** 2)
