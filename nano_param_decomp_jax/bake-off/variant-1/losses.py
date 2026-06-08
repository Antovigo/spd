"""The three losses from the SHARED_SPEC: faithfulness, importance-minimality,
stochastic-recon. All operate on the dict-of-site structures the trainer threads
through the forward functions."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

Components = dict[str, dict[str, Float[Array, "..."]]]  # {site: {V, U, W_delta}}
Masks = dict[str, Float[Array, "... C"]]
CIs = dict[str, Float[Array, "... C"]]
TargetWeights = dict[str, Float[Array, "d_in d_out"]]


def faithfulness_loss(components: Components, target_weights: TargetWeights) -> Float[Array, ""]:
    """mean over all sites of (W_target - V@U)^2."""
    sum_sq = jnp.array(0.0)
    numel = 0
    for name, w_target in target_weights.items():
        c = components[name]
        delta = w_target - c["V"] @ c["U"]
        sum_sq = sum_sq + jnp.sum(delta**2)
        numel += delta.size
    return sum_sq / numel


def importance_minimality_loss(cis: CIs, p: float = 0.9) -> Float[Array, ""]:
    """mean(ci^p) averaged across all sites (simple sparsity penalty per the spec)."""
    parts = [jnp.mean(ci**p) for ci in cis.values()]
    return jnp.mean(jnp.stack(parts))


def stochastic_recon_loss(
    y_decomposed: Float[Array, "..."], y_target: Float[Array, "..."]
) -> Float[Array, ""]:
    """Plain MSE on outputs."""
    return jnp.mean((y_decomposed - y_target) ** 2)


def sample_masks(cis: CIs, key: PRNGKeyArray) -> Masks:
    """mask = ci + (1 - ci) * U[0, 1], independent draw per (b, t, c)."""
    keys = jax.random.split(key, len(cis))
    masks: Masks = {}
    for k, (name, ci) in zip(keys, sorted(cis.items()), strict=True):
        u = jax.random.uniform(k, ci.shape, dtype=ci.dtype)
        masks[name] = ci + (1.0 - ci) * u
    return masks


def init_components(
    target_weights: TargetWeights, c_per_site: dict[str, int], key: PRNGKeyArray
) -> Components:
    """V ~ N(0, 1/sqrt(d_in)), U ~ N(0, 1/sqrt(C)).

    Note: per the nano reference (`ComponentLinear.weight_delta()`), W_delta is
    *computed fresh* each forward as `W_target - V @ U` — not frozen at init.
    The SHARED_SPEC's "frozen at init" reading was flagged ambiguous; we follow
    nano because the faithfulness loss then drives W_delta -> 0 in tandem with V@U
    converging on W_target, which is the algorithm's intended dynamic.

    We therefore don't store W_delta here — the decomposed forward recomputes it
    from V/U and W_target. We only return V/U as the trainable per-site state.
    """
    components: Components = {}
    names = sorted(target_weights.keys())
    keys = jax.random.split(key, len(names) * 2).reshape(len(names), 2, -1)
    for (name, kv_pair) in zip(names, keys, strict=True):
        w = target_weights[name]
        d_in, d_out = w.shape
        C = c_per_site[name]
        v_key, u_key = kv_pair
        V = jax.random.normal(v_key, (d_in, C)) / jnp.sqrt(d_in)
        U = jax.random.normal(u_key, (C, d_out)) / jnp.sqrt(C)
        components[name] = {"V": V, "U": U}
    return components
