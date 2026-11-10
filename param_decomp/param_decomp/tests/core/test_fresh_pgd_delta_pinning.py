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
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.recon_eval import FreshPGDAttack, fresh_pgd_read_outs
from param_decomp.targets.tms import TMSConfig, site_specs

SITES = site_specs(TMSConfig(n_features=5, n_hidden=2), (SiteC("linear1", 8), SiteC("linear2", 6)))
ATTACK = FreshPGDAttack(step_size=0.2, read_out_steps=(10,))


def _ci_lower() -> dict[str, Array]:
    # Mid-window CI everywhere: the adversary has a (1 - ci) gap to exploit at every site.
    return {site.name: jnp.full((4, site.C), 0.3) for site in SITES}


def _masks_of(masking: MaterializedMasking) -> tuple[dict[str, Array], dict[str, Array]]:
    assert masking.weight_delta_masks is not None
    masks = {site: jnp.asarray(m) for site, m in masking.component_masks.items()}
    return masks, dict(masking.weight_delta_masks)


def _ablation_reward(masking: MaterializedMasking) -> Array:
    """A stand-in worst case that grows as component AND delta masks fall."""
    masks, delta_masks = _masks_of(masking)
    component = sum((jnp.sum(1.0 - m) for m in masks.values()), start=jnp.zeros(()))
    delta = sum((jnp.sum(1.0 - dm) for dm in delta_masks.values()), start=jnp.zeros(()))
    return component + delta


def _ascended(delta_pinned: bool) -> tuple[dict[str, Array], dict[str, Array]]:
    """The production probe's ascent (`fresh_pgd_read_outs`, as `make_fresh_pgd_eval_step`
    drives it), read out as the composed masks after the last step."""
    (masks,) = fresh_pgd_read_outs(
        SITES,
        _ci_lower(),
        (4,),
        jax.random.PRNGKey(0),
        ATTACK,
        _ablation_reward,
        _masks_of,
        delta_pinned=delta_pinned,
    )
    return masks


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
