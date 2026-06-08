"""Three losses from SHARED_SPEC: faithfulness, importance minimality, stochastic recon."""

import jax
import jax.numpy as jnp
from decomposed import decomposed_sites
from jaxtyping import Array, Float, PRNGKeyArray

IMP_P = 0.9


def faithfulness_loss(decomposed_params: dict) -> Float[Array, ""]:
    """Mean squared (W_target - V @ U) across all decomposed sites.

    `W_target` is recovered as `W_delta + V @ U` since `W_delta` was frozen at init to
    `W_target - V @ U`. So the residual we're shrinking is exactly `W_target - V @ U =
    W_delta + V @ U - V @ U = W_delta`... wait — that's only true at init. After V/U
    move, `W_target` is no longer `W_delta + V @ U`; we'd lose track of `W_target`.

    Fix: faithfulness shrinks `||W_target - V @ U||^2`. `W_target` is *not* in the
    pytree (we don't want grads on it). We pass `target_weights` in separately.
    """
    raise NotImplementedError("use faithfulness_loss_against_targets")


def faithfulness_loss_against_targets(
    decomposed_params: dict, target_weights: dict[str, Float[Array, "d_in d_out"]]
) -> Float[Array, ""]:
    """mean((W_target - V @ U)^2) averaged over all decomposed sites' elements."""
    sites = decomposed_sites(decomposed_params)
    total_sq = jnp.zeros(())
    total_n = 0
    for name, d in sites.items():
        W_target = target_weights[name]
        residual = W_target - d.V @ d.U
        total_sq = total_sq + jnp.sum(residual**2)
        total_n += residual.size
    return total_sq / total_n


def importance_minimality_loss(
    ci: dict[str, Float[Array, "... C"]], p: float = IMP_P
) -> Float[Array, ""]:
    """Simple sparsity penalty: mean(ci^p) averaged across all sites' elements."""
    total = jnp.zeros(())
    total_n = 0
    for v in ci.values():
        total = total + jnp.sum(v**p)
        total_n += v.size
    return total / total_n


def sample_masks(
    key: PRNGKeyArray, ci: dict[str, Float[Array, "... C"]]
) -> dict[str, Float[Array, "... C"]]:
    """mask = ci + (1 - ci) * U[0, 1] per (..., C). Fresh sample every call."""
    names = sorted(ci.keys())
    keys = jax.random.split(key, len(names))
    masks = {}
    for name, k in zip(names, keys):
        c = ci[name]
        u = jax.random.uniform(k, c.shape, dtype=c.dtype)
        masks[name] = c + (1 - c) * u
    return masks


def stochastic_recon_loss(
    y_decomposed: Float[Array, "..."], y_target: Float[Array, "..."]
) -> Float[Array, ""]:
    """MSE between decomposed forward output and target forward output."""
    return jnp.mean((y_decomposed - y_target) ** 2)
