"""The four VPD losses over a stacked-site decomposition.

The state is stacked along a leading site axis `S` (homogeneous sites — the
production LM target decomposes a fixed set of equal-shape MLP matrices; the
einsums assume equal `d_in`/`d_out` across sites). A `Decomposition` bundles
`(V, U, W_target)` stacked `[S, ...]`; a `CIParams` bundles the per-site CI
linear `(w, b)`. All four losses are pure.

  faithfulness          mean over sites of mean((W_target - V@U)^2)
  importance_minimality mean(ci^p)
  stochastic_recon      layerwise MSE under a fresh stochastic mask
  ppgd_recon            layerwise MSE under the adversarial (worst-case) mask

mask = ci + (1 - ci) * source; the weight-delta source channel (when present)
gates the residual per position. See `forward.py` and `scopes.py`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from nano_pd_jax.ci_sigmoids import lower_leaky_hard_sigmoid

from jax_single_pool.forward import (
    faithfulness_residual_sq,
    site_decomposed_out,
    site_target_out,
)


class Decomposition(NamedTuple):
    V: Float[Array, "S d_in C"]
    U: Float[Array, "S C d_out"]
    W_target: Float[Array, "S d_in d_out"]  # frozen


class CIParams(NamedTuple):
    w: Float[Array, "S d_in C"]
    b: Float[Array, "S C"]


def ci_envelope(ci_params: CIParams, x: Float[Array, "S ... d_in"]) -> Float[Array, "S ... C"]:
    """Per-site CI in [0,1]. `x` is the stacked pre-weight acts (one per site)."""
    logits = jnp.einsum("s...i,sic->s...c", x, ci_params.w) + ci_params.b.reshape(
        ci_params.b.shape[0], *([1] * (x.ndim - 2)), ci_params.b.shape[1]
    )
    return lower_leaky_hard_sigmoid(logits)


def faithfulness_loss(decomp: Decomposition) -> Float[Array, ""]:
    per_site = jax.vmap(faithfulness_residual_sq)(decomp.V, decomp.U, decomp.W_target)
    return jnp.mean(per_site)


def importance_minimality_loss(ci: Float[Array, "S ... C"], p: float) -> Float[Array, ""]:
    return jnp.mean(jnp.clip(ci, 0.0, 1.0) ** p)


def _split_delta_channel(
    mask: Float[Array, "S ... source_c"], use_delta_component: bool
) -> tuple[Float[Array, "S ... C"], Float[Array, "S ..."] | None]:
    if not use_delta_component:
        return mask, None
    return mask[..., :-1], mask[..., -1]


def layerwise_recon_loss(
    decomp: Decomposition,
    x: Float[Array, "S ... d_in"],
    mask: Float[Array, "S ... source_c"],
    use_delta_component: bool,
) -> Float[Array, ""]:
    """Per-site MSE of the masked decomposed output vs the frozen target output.

    `mask` already encodes `ci + (1-ci)*source` (component channels) and, when
    `use_delta_component`, a trailing weight-delta channel.
    """
    component_mask, delta_mask = _split_delta_channel(mask, use_delta_component)

    def per_site(
        V: Array, U: Array, W_target: Array, x_s: Array, cmask_s: Array, dmask_s: Array | None
    ) -> Float[Array, ""]:
        y_dec = site_decomposed_out(x_s, V, U, W_target, cmask_s, dmask_s)
        y_tgt = jax.lax.stop_gradient(site_target_out(x_s, W_target))
        return jnp.mean((y_dec - y_tgt) ** 2)

    if delta_mask is None:
        per = jax.vmap(lambda V, U, W, xs, cm: per_site(V, U, W, xs, cm, None))(
            decomp.V, decomp.U, decomp.W_target, x, component_mask
        )
    else:
        per = jax.vmap(per_site)(decomp.V, decomp.U, decomp.W_target, x, component_mask, delta_mask)
    return jnp.mean(per)


def interpolate_mask(
    ci: Float[Array, "S ... C"],
    source: Float[Array, "S ... source_c"],
    use_delta_component: bool,
) -> Float[Array, "S ... source_c"]:
    """mask = ci + (1 - ci) * source, with the delta channel passed through raw.

    Matches torch: component channels interpolate with ci; the weight-delta
    channel is the source value directly (no ci interpolation — ci has no delta
    component).
    """
    if not use_delta_component:
        return ci + (1.0 - ci) * source
    comp = ci + (1.0 - ci) * source[..., :-1]
    return jnp.concatenate([comp, source[..., -1:]], axis=-1)


def sample_stochastic_source(
    key: PRNGKeyArray, shape: tuple[int, ...]
) -> Float[Array, "S ... source_c"]:
    return jax.random.uniform(key, shape)
