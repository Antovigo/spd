"""The target-coupled V/U inits (`pd.weight_init`: `coupled` / `zero_u`).

The coupling claim is checked against each site's own frozen `W`, read back through
`weight_deltas` — the same seam the init itself uses.
"""

import jax
import jax.numpy as jnp

from param_decomp.core.components import (
    init_component_stacks_coupled,
    zero_component_stacks,
)
from param_decomp.targets.glu_transformer import glu_site_specs, mlp_family_site_cs
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm


def _tiny_model_and_weights():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(0, 1, 4))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    return model, sites, model.weight_deltas(zero_component_stacks(sites))


def test_zero_components_read_back_the_frozen_weights():
    """`weight_deltas` is `W - (V@U)^T`, so the zero stacks recover `W` — the premise the
    coupled inits stand on."""
    model, sites, weights = _tiny_model_and_weights()
    assert set(weights) == {spec.name for spec in sites}
    for spec in sites:
        assert weights[spec.name].shape == (spec.d_out, spec.d_in)
        assert jnp.all(jnp.isfinite(weights[spec.name]))
    # A nonzero V/U must move the delta away from W, or the read-back is vacuous.
    nonzero = zero_component_stacks(sites)
    shape, (Vs, Us) = next(iter(nonzero.stacks.items()))
    perturbed = nonzero.stacks | {shape: (Vs + 1.0, Us + 1.0)}
    moved = model.weight_deltas(type(nonzero)(stacks=perturbed, site_slots=nonzero.site_slots))
    assert any(not jnp.allclose(moved[s.name], weights[s.name]) for s in sites)


def test_coupled_init_couples_the_wide_side_to_w():
    _model, sites, weights = _tiny_model_and_weights()
    vu = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1), zero_u=False)
    for spec in sites:
        site = vu.site(spec.name)
        W = weights[spec.name]
        if spec.d_in <= spec.d_out:
            # V is the unit-norm seed; U is its raw W-image.
            assert jnp.allclose(jnp.linalg.norm(site.V, axis=0), 1.0, atol=1e-5), spec.name
            assert jnp.allclose(site.U, (W @ site.V).T, atol=1e-4), spec.name
        else:
            assert jnp.allclose(jnp.linalg.norm(site.U, axis=1), 1.0, atol=1e-5), spec.name
            assert jnp.allclose(site.V, W.T @ site.U.T, atol=1e-4), spec.name


def test_zero_u_silences_every_component_but_keeps_v_live():
    """The component sum is exactly zero at init, so the delta carries all of `W`; `V` is
    untouched so `x @ V` still feeds the CI nets."""
    model, sites, weights = _tiny_model_and_weights()
    vu = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1), zero_u=True)
    coupled = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1), zero_u=False)
    for spec in sites:
        assert jnp.all(vu.site(spec.name).U == 0.0), spec.name
        assert jnp.any(vu.site(spec.name).V != 0.0), spec.name
        assert jnp.array_equal(vu.site(spec.name).V, coupled.site(spec.name).V), spec.name
    deltas = model.weight_deltas(vu)
    for spec in sites:
        assert jnp.allclose(deltas[spec.name], weights[spec.name]), spec.name


def test_placed_init_matches_the_eager_values():
    """The jitted, sharding-placed path reads `W` inside the graph; same values up to fp32
    reassociation in the `W`-image matmul (XLA picks its own layout)."""
    from param_decomp.core.init_placed import init_component_stacks_coupled_placed
    from param_decomp.core.placement import from_config_for_consumer
    from param_decomp.core.sharding import single_device_mesh

    model, sites, weights = _tiny_model_and_weights()
    mesh = single_device_mesh()
    rules = from_config_for_consumer("ddp", mesh, sites)
    for zero_u in (False, True):
        placed = init_component_stacks_coupled_placed(
            model, jax.random.PRNGKey(1), rules, zero_u=zero_u
        )
        eager = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1), zero_u=zero_u)
        for spec in sites:
            placed_site, eager_site = placed.site(spec.name), eager.site(spec.name)
            assert jnp.allclose(placed_site.V, eager_site.V, atol=1e-5), spec.name
            assert jnp.allclose(placed_site.U, eager_site.U, atol=1e-5), spec.name
