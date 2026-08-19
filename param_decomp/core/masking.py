"""Mask materialization shared by reconstruction objectives and model targets."""

import jax.numpy as jnp
from jax import random
from jax.typing import DTypeLike
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.components import SiteSpec
from param_decomp.core.model import Masking, MaterializedMasking, StochasticMasking


def all_live_masking_no_delta(
    sites: tuple[SiteSpec, ...], *, leading_shape: tuple[int, ...], dtype: DTypeLike
) -> MaterializedMasking:
    """Turn every component on while disabling frozen-weight delta corrections."""
    return MaterializedMasking(
        component_masks={site.name: jnp.ones((*leading_shape, site.C), dtype) for site in sites}
    )


def _sample_stochastic_masks(
    ci_lower: dict[str, Array], live_sites: tuple[str, ...], draw_key: Array
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Draw fresh component and weight-delta masks for every live site."""
    mask_key, delta_key = random.split(draw_key)
    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, site in enumerate(live_sites):
        ci = ci_lower[site]
        masks[site] = ci + (1.0 - ci) * random.uniform(
            random.fold_in(mask_key, site_idx), ci.shape, ci.dtype
        )
        delta_masks[site] = random.uniform(
            random.fold_in(delta_key, site_idx), ci.shape[:-1], ci.dtype
        )
    return masks, delta_masks


def materialize_masking(masking: Masking) -> MaterializedMasking:
    """Return concrete mask arrays for a target that cannot sample inside its blocks.

    Materialized inputs — including adversarial masks — pass through unchanged. Stochastic
    inputs are sampled eagerly here. Such targets implement ``stack_ci`` as identity, so
    ``ci_stacked`` remains the ordinary per-site ``ci_lower`` dictionary; the scan target
    consumes its stacked recipe directly and never calls this helper.
    """
    match masking:
        case MaterializedMasking():
            return masking
        case StochasticMasking(
            ci_stacked=ci_lower,
            draw_key=draw_key,
            live_sites=live_sites,
            routes=routes,
        ):
            assert isinstance(ci_lower, dict), (
                f"materialize_masking requires identity stack_ci; got {type(ci_lower).__name__}"
            )
            component_masks, weight_delta_masks = _sample_stochastic_masks(
                ci_lower, live_sites, draw_key
            )
            return MaterializedMasking(
                component_masks=component_masks,
                weight_delta_masks=weight_delta_masks,
                routes=routes,
            )


def stochastic_delta_pinned_masks(
    ci_lower: dict[str, Array], live_sites: tuple[str, ...], draw_key: Array
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Stochastic component masks with every weight-delta mask pinned to 1.0 — the tPD
    non-target pass (SPEC T4), where `components + Δ` must reconstruct the frozen output.

    Pre-built (`MaterializedMasking`) rather than the in-target `StochasticMasking`
    rebuild, which draws its own `U[0,1]` delta inside each block and cannot pin it. The
    key split mirrors `_sample_stochastic_masks` (source half used, delta half discarded —
    the delta is deterministic here)."""
    mask_key, _ = random.split(draw_key)
    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for site_idx, site in enumerate(live_sites):
        ci = ci_lower[site]
        masks[site] = ci + (1.0 - ci) * random.uniform(
            random.fold_in(mask_key, site_idx), ci.shape, ci.dtype
        )
        delta_masks[site] = jnp.ones(ci.shape[:-1], ci.dtype)
    return masks, delta_masks


def constant_delta_pinned_masks(
    value: float, ci_lower: dict[str, Array], live_sites: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Constant component masks (`ci + (1-ci)·value`) with every weight-delta mask pinned
    to 1.0 — the tPD non-target pass's constant-source arm (SPEC T4). The plain objective's
    constant arm carries NO delta path at all; here the delta must be fully on."""
    masks = {site: ci_lower[site] + (1.0 - ci_lower[site]) * value for site in live_sites}
    delta_masks = {
        site: jnp.ones(ci_lower[site].shape[:-1], ci_lower[site].dtype) for site in live_sites
    }
    return masks, delta_masks


def unmasked_no_delta_masks(
    ci_lower: dict[str, Array], live_sites: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Every component mask `1.0` with every weight-delta mask pinned to `0.0` — the tPD
    non-target pass's one delta-OFF arm (SPEC T4's enumerated exception): the FULL
    component sum alone must reconstruct the frozen output, so components that never
    activate cannot hide behind the delta. Deterministic — no sources are drawn;
    `ci_lower` supplies only shapes and dtypes."""
    masks = {site: jnp.ones_like(ci_lower[site]) for site in live_sites}
    delta_masks = {
        site: jnp.zeros(ci_lower[site].shape[:-1], ci_lower[site].dtype) for site in live_sites
    }
    return masks, delta_masks


def persistent_delta_pinned_masks(
    ci_lower: dict[str, Array], sources: dict[str, Array], live_sites: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Component masks from a persistent bundle's COMPONENT channels with every
    weight-delta mask pinned to 1.0 — the tPD non-target adversarial arm (SPEC T4, T5/T7
    amended 2026-08-19). The source's trailing delta channel is ignored: with the delta
    pinned fully on off-target, the adversary attacks components only, so the probed
    worst case is a partial ablation of components on data they should be inactive on."""
    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for site in live_sites:
        ci = ci_lower[site]
        source = sources[site].astype(ci.dtype)
        masks[site] = ci + (1.0 - ci) * source[..., :-1]
        delta_masks[site] = jnp.ones(ci.shape[:-1], ci.dtype)
    return masks, delta_masks


def mixed_persistent_stochastic_delta_pinned_masks(
    key: PRNGKeyArray,
    ci_lower: dict[str, Array],
    persistent_sources: dict[str, Array],
    live_sites: tuple[str, ...],
    components_per_site: dict[str, int],
    leading: tuple[int, ...],
    adv_fraction: Array,
    stochastic_routes: dict[str, Array] | None,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array] | None]:
    """`mixed_persistent_stochastic_masks` with BOTH families' weight-delta masks pinned
    to 1.0 — the S34 merge on the tPD non-target pass (T4; T5/T7 amended 2026-08-19).
    Component masks and routing compose exactly as the plain merge; only the delta
    polarity differs, and it rides this constructor rather than a flag."""
    assignment_key, uniform_key = random.split(key)
    adversarial = _per_sample_adversarial_assignment(assignment_key, adv_fraction, leading)
    fresh_uniform_components = {
        site: random.uniform(
            random.fold_in(uniform_key, site_idx),
            (*leading, components_per_site[site]),
        )
        for site_idx, site in enumerate(live_sites)
    }
    adv_masks, _ = persistent_delta_pinned_masks(ci_lower, persistent_sources, live_sites)
    masks: dict[str, Array] = {}
    delta_masks: dict[str, Array] = {}
    for site in live_sites:
        ci = ci_lower[site]
        stoch = ci + (1.0 - ci) * fresh_uniform_components[site].astype(ci.dtype)
        masks[site] = jnp.where(adversarial[..., None], adv_masks[site], stoch)
        delta_masks[site] = jnp.ones(ci.shape[:-1], ci.dtype)
    routes = (
        None
        if stochastic_routes is None
        else {site: jnp.logical_or(adversarial, stochastic_routes[site]) for site in live_sites}
    )
    return masks, delta_masks, routes


def masks_from_sources(
    ci_lower: dict[str, Array], sources: dict[str, Array], site_names: tuple[str, ...]
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Build component and weight-delta masks from per-site sources (SPEC S1).

    Sources broadcast over the leading dimensions left singleton by their source shape.
    Casting the fp32 source state to the CI dtype here matches torch under autocast while
    preserving the source gradient through the cast.
    """
    masks = {}
    delta_masks = {}
    for site in site_names:
        source = sources[site].astype(ci_lower[site].dtype)
        masks[site] = ci_lower[site] + (1.0 - ci_lower[site]) * source[..., :-1]
        delta_masks[site] = source[..., -1]
    return masks, delta_masks


def _per_sample_adversarial_assignment(
    key: PRNGKeyArray, adv_fraction: Array, leading: tuple[int, ...]
) -> Array:
    """Draw one Bernoulli selector per sample, broadcast across position axes (SPEC S34)."""
    one_flag_per_sample = (leading[0], *(1,) * (len(leading) - 1))
    return random.bernoulli(key, adv_fraction, one_flag_per_sample)


def mixed_persistent_stochastic_masks(
    key: PRNGKeyArray,
    ci_lower: dict[str, Array],
    persistent_sources: dict[str, Array],
    live_sites: tuple[str, ...],
    components_per_site: dict[str, int],
    leading: tuple[int, ...],
    adv_fraction: Array,
    stochastic_routes: dict[str, Array] | None,
) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array] | None]:
    """Build the merged stochastic+PPGD term's forward inputs (SPEC S34).

    Adversarial samples use the persistent bundle and route every live site; the rest use
    fresh uniform sources and the stochastic routes. The persistent sources remain graph
    leaves, so the per-sample selection gates their gradient to adversarial samples.
    """
    assignment_key, uniform_key = random.split(key)
    adversarial = _per_sample_adversarial_assignment(assignment_key, adv_fraction, leading)
    fresh_uniform_sources = {
        site: random.uniform(
            random.fold_in(uniform_key, site_idx),
            (*leading, components_per_site[site] + 1),
        )
        for site_idx, site in enumerate(live_sites)
    }
    adv_masks, adv_deltas = masks_from_sources(ci_lower, persistent_sources, live_sites)
    stoch_masks, stoch_deltas = masks_from_sources(ci_lower, fresh_uniform_sources, live_sites)
    masks = {
        site: jnp.where(adversarial[..., None], adv_masks[site], stoch_masks[site])
        for site in live_sites
    }
    delta_masks = {
        site: jnp.where(adversarial, adv_deltas[site], stoch_deltas[site]) for site in live_sites
    }
    routes = (
        None
        if stochastic_routes is None
        else {site: jnp.logical_or(adversarial, stochastic_routes[site]) for site in live_sites}
    )
    return masks, delta_masks, routes
