"""The three losses for stochastic parameter decomposition."""

import equinox as eqx
import jax
import jax.numpy as jnp
from decomposed_linear import DecomposedLinear
from jaxtyping import Array, Float, PRNGKeyArray


def faithfulness_loss(model: eqx.Module, site_paths: list[str]) -> Float[Array, ""]:
    """Mean squared element of `W_target - V @ U` across all decomposed sites.

    Element-weighted (total squared error / total elements), matching the reference impl.
    """
    sum_sq = jnp.array(0.0)
    numel = 0
    for path in site_paths:
        site = _get_by_path(model, path)
        assert isinstance(site, DecomposedLinear)
        residual = site.W_target - site.V @ site.U
        sum_sq = sum_sq + (residual ** 2).sum()
        numel += residual.size
    return sum_sq / numel


def importance_minimality_loss(ci: dict[str, Float[Array, "... C"]], p: float) -> Float[Array, ""]:
    """mean(ci^p) averaged across all sites. Simple sparsity penalty (spec value: p=0.9)."""
    terms = [jnp.mean(jnp.clip(v, 0.0, 1.0) ** p) for v in ci.values()]
    return jnp.mean(jnp.stack(terms))


def stochastic_recon_loss(
    decomposed_model: eqx.Module,
    ci: dict[str, Float[Array, "... C"]],
    x: Float[Array, "B d_in"],
    target_out: Float[Array, "B d_out"],
    *,
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    """MSE between decomposed-forward output (with stochastic mask) and target output.

    `mask = ci + (1 - ci) * u`, fresh u ~ U[0,1] per (b, c) per site.
    """
    masks: dict[str, Float[Array, "... C"]] = {}
    keys = jax.random.split(key, len(ci))
    for k, (name, c) in zip(keys, ci.items()):
        u = jax.random.uniform(k, c.shape)
        masks[name] = c + (1.0 - c) * u
    pred = jax.vmap(decomposed_model, in_axes=(0, None))(x, masks)
    return jnp.mean((pred - target_out) ** 2)


def _get_by_path(obj: object, path: str) -> object:
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj
