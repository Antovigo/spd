"""Masked decomposed forward, layerwise (per-site) reconstruction.

The single-pool VPD recon losses in the production LM config are *layerwise*:
each decomposed site reconstructs its own output `y = W_target @ x` from the
masked components, independently of the rest of the model. That makes the recon
math site-local and model-agnostic — the step never threads masks back through
the frozen target's attention/MLP, it only needs each site's pre-weight acts
`x` and the site's `(V, U, W_target)`.

This module is the site-local masked forward. The trainer stacks sites along a
leading `S` axis (heterogeneous `d_in`/`d_out` are padded or, in the homogeneous
LM-MLP case, naturally equal) and vmaps these over `(S, batch...)`.

mask convention (matches `param_decomp/masks.py`):
  y_dec = ((x @ V) * component_mask) @ U + (x @ W_delta) * delta_mask
  W_delta = W_target - V @ U
`delta_mask is None` applies the residual unmasked (no-weight-delta path).
`y_tgt = x @ W_target`. Layerwise recon is `mean((y_dec - y_tgt) ** 2)`.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


def site_decomposed_out(
    x: Float[Array, "... d_in"],
    V: Float[Array, "d_in C"],
    U: Float[Array, "C d_out"],
    W_target: Float[Array, "d_in d_out"],
    component_mask: Float[Array, "... C"],
    delta_mask: Float[Array, "..."] | None,
) -> Float[Array, "... d_out"]:
    W_delta = W_target - V @ U
    y = ((x @ V) * component_mask) @ U
    residual = x @ W_delta
    if delta_mask is None:
        return y + residual
    return y + residual * delta_mask[..., None]


def site_target_out(
    x: Float[Array, "... d_in"],
    W_target: Float[Array, "d_in d_out"],
) -> Float[Array, "... d_out"]:
    return x @ W_target


def faithfulness_residual_sq(
    V: Float[Array, "d_in C"],
    U: Float[Array, "C d_out"],
    W_target: Float[Array, "d_in d_out"],
) -> Float[Array, ""]:
    resid = W_target - V @ U
    return jnp.mean(resid**2)
