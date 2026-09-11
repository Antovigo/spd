"""The `zero_u` initializer (ported onto `ComponentInitializer` for #1001).

What `zero_u` claims, and what each test here pins:
  - the component sum is EXACTLY zero at init, so the delta carries all of `W`;
  - `V` is still live, so the CI nets see a real signal from step 0;
  - `V` is the COUPLED seed against each site's own frozen `W`, drawn with the same key
    discipline the retired `pd.weight_init: coupled`/`zero_u` arms used — which is what
    lets a run on this branch be compared against the pre-#1001 zero_u runs at one seed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    _coupled_site_vu,
    init_component_stacks_zero_u,
    zero_component_stacks,
)
from param_decomp.core.init_placed import (
    init_model_component_stacks_placed,
    zero_u_component_initializer,
)
from param_decomp.core.model import PlacedModel, site_weight_delta
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import single_device_mesh
from param_decomp.targets.glu_transformer import canonical_site_cs, glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

KEY = jax.random.PRNGKey(1)


def _model():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            (
                SiteC(site_name(0, "q"), 3),  # d_in > d_out and the reverse are different
                SiteC(site_name(0, "o"), 3),  # branches of the coupled seed
                SiteC(site_name(0, "gate"), 4),
                SiteC(site_name(0, "down"), 5),
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    zero = zero_component_stacks(sites)
    stacked = model.weight_deltas(zero)
    weights = {s.name: np.asarray(site_weight_delta(stacked, zero, s.name)) for s in sites}
    return model, sites, weights


def _site_arrays(vu: ComponentStacks, name: str) -> tuple[np.ndarray, np.ndarray]:
    site = dict(vu.sites_items())[name]
    return np.asarray(site.V), np.asarray(site.U)


def test_zero_components_read_back_the_frozen_weights():
    """The premise the whole init stands on: `weight_deltas` of zero stacks IS `W`."""
    model, sites, weights = _model()
    assert set(weights) == {s.name for s in sites}
    for spec in sites:
        assert weights[spec.name].shape == (spec.d_out, spec.d_in)
        assert np.all(np.isfinite(weights[spec.name]))
    # A nonzero V/U must move the delta, or the read-back above is vacuous.
    zero = zero_component_stacks(sites)
    group, (Vs, Us) = next(iter(zero.stacks.items()))
    perturbed = type(zero)(
        stacks=zero.stacks | {group: (Vs + 1.0, Us + 1.0)}, site_slots=zero.site_slots
    )
    moved = model.weight_deltas(perturbed)
    assert any(
        not np.allclose(np.asarray(site_weight_delta(moved, perturbed, s.name)), weights[s.name])
        for s in sites
    )


def test_the_component_sum_is_exactly_zero_and_the_delta_carries_all_of_w():
    model, sites, weights = _model()
    vu = zero_u_component_initializer(model, KEY)
    deltas = model.weight_deltas(vu)
    for spec in sites:
        V, U = _site_arrays(vu, spec.name)
        assert np.array_equal(U, np.zeros_like(U)), spec.name
        assert np.array_equal((V @ U).T, np.zeros((spec.d_out, spec.d_in), np.float32)), spec.name
        np.testing.assert_allclose(
            np.asarray(site_weight_delta(deltas, vu, spec.name)), weights[spec.name], atol=1e-5
        )


def test_v_stays_live_so_the_ci_nets_see_a_signal():
    """`U` is silenced, `V` is not — a dead `V` would starve the CI fn at step 0."""
    model, sites, _w = _model()
    vu = zero_u_component_initializer(model, KEY)
    for spec in sites:
        V, _U = _site_arrays(vu, spec.name)
        assert np.all(np.linalg.norm(V, axis=0) > 0), spec.name


def test_v_is_the_coupled_seed_with_the_pre_1001_key_discipline():
    """Bit-exactness against the retired arm: per-site keys are `split(key, len(sites))`
    indexed by site position, and V is `_coupled_site_vu`'s V. This is what makes a
    pre-#1001 zero_u run a valid control for one on this branch at the same seed."""
    _m, sites, weights = _model()
    vu = init_component_stacks_zero_u(sites, {k: jnp.asarray(v) for k, v in weights.items()}, KEY)
    keys = jax.random.split(KEY, len(sites))
    for idx, spec in enumerate(sites):
        expected_V, _expected_U = _coupled_site_vu(
            jnp.asarray(weights[spec.name]), keys[idx], spec.C
        )
        V, _U = _site_arrays(vu, spec.name)
        np.testing.assert_array_equal(V, np.asarray(expected_V))


def test_the_seed_actually_matters():
    model, sites, _w = _model()
    a = zero_u_component_initializer(model, jax.random.PRNGKey(1))
    b = zero_u_component_initializer(model, jax.random.PRNGKey(2))
    assert any(
        not np.array_equal(_site_arrays(a, s.name)[0], _site_arrays(b, s.name)[0]) for s in sites
    )


def test_placed_init_matches_the_eager_values():
    model, sites, _w = _model()
    eager = zero_u_component_initializer(model, KEY)
    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    with jax.set_mesh(mesh):
        placed = init_model_component_stacks_placed(
            PlacedModel(model=model, placement=rules), KEY, rules, zero_u_component_initializer
        )
    for spec in sites:
        for x, y in zip(
            _site_arrays(placed, spec.name), _site_arrays(eager, spec.name), strict=True
        ):
            np.testing.assert_allclose(x, y, atol=1e-6)


@pytest.mark.parametrize("init", ["random", "zero_u", "neuron_aligned"])
def test_the_lm_dispatch_resolves_every_data_free_arm(init: str):
    """`zero_u` is selectable exactly like the other data-free arms — no artifact, no
    capacity bound (unlike the aligned inits, which cap C at the coordinate count)."""
    from param_decomp.experiments.lm.load_run import component_initializer_for
    from param_decomp.experiments.lm.resolved import TargetConfig

    target = TargetConfig(
        model_name="meta-llama/Llama-3.1-8B",
        sites=(),
        weights_dtype="bfloat16",
        attention_implementation="xla",
        component_initialization=init,  # pyright: ignore[reportArgumentType]
    )
    assert callable(component_initializer_for(target))
