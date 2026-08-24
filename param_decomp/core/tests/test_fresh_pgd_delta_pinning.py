"""The fresh-PGD probe's delta polarity (SPEC T4, amended 2026-08-20).

Unpinned (plain runs, target stream), the probe's sources carry a live delta channel the
ascent can drive to 0. Delta-pinned (a targeted run's non-target stream), every delta
mask composes as exactly 1.0 and the sources' delta channel receives no gradient — the
probe measures the component-only worst case the T4 training forwards defend.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from param_decomp.core.components import SiteC
from param_decomp.core.masking import masks_from_sources, persistent_delta_pinned_masks
from param_decomp.core.recon_eval import FreshPGDReconEval, fresh_pgd_recon_sources
from param_decomp.targets.tms import TMSConfig, site_specs

SITES = site_specs(TMSConfig(n_features=5, n_hidden=2), (SiteC("linear1", 8), SiteC("linear2", 6)))
PROBE = FreshPGDReconEval(n_steps=10, step_size=0.2)


def _ci_lower() -> dict[str, Array]:
    # Mid-window CI everywhere: the adversary has a (1 - ci) gap to exploit at every site.
    return {site.name: jnp.full((4, site.C), 0.3) for site in SITES}


def _ablation_reward(masks: dict[str, Array], delta_masks: dict[str, Array]) -> Array:
    """A stand-in worst case that grows as component AND delta masks fall."""
    component = sum((jnp.sum(1.0 - m) for m in masks.values()), start=jnp.zeros(()))
    delta = sum((jnp.sum(1.0 - dm) for dm in delta_masks.values()), start=jnp.zeros(()))
    return component + delta


def _ascended(delta_pinned: bool) -> tuple[dict[str, Array], dict[str, Array]]:
    ci = _ci_lower()
    sources = fresh_pgd_recon_sources(
        SITES, ci, (4,), jax.random.PRNGKey(0), PROBE, _ablation_reward, delta_pinned
    )
    mask_fn = persistent_delta_pinned_masks if delta_pinned else masks_from_sources
    return mask_fn(ci, sources)


def test_unpinned_probe_attacks_the_delta_channel():
    masks, delta_masks = _ascended(delta_pinned=False)
    for site in delta_masks:
        assert float(jnp.max(delta_masks[site])) == 0.0, "ascent should drive delta to 0"
        # Components ablate down to the CI floor, never below (S1).
        np.testing.assert_allclose(np.asarray(masks[site]), 0.3, rtol=1e-6)


def test_pinned_probe_leaves_delta_fully_on_and_still_ascends_components():
    masks, delta_masks = _ascended(delta_pinned=True)
    for site in delta_masks:
        np.testing.assert_array_equal(np.asarray(delta_masks[site]), 1.0)
        np.testing.assert_allclose(np.asarray(masks[site]), 0.3, rtol=1e-6)
