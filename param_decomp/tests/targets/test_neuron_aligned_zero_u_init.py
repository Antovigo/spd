"""The `neuron_aligned_zero_u` init: `neuron_aligned_targeted`'s `V` with `U` zeroed.

What this init claims, and what each test here pins:

  * `V` is bit-identical to `neuron_aligned_targeted`'s — the two inits select the SAME
    coordinates from the same ranking, so a run of each at one seed differs only in `U`.
  * `U` is exactly zero, so the component sum is exactly zero and the delta carries all of
    `W` (`zero_u`'s discipline, which is the point of the arm).
  * It consumes no randomness, and the placed path matches the eager values.
"""

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteSpec,
    component_stacks_from_site_arrays,
)
from param_decomp.core.init_placed import init_model_component_stacks_placed
from param_decomp.core.model import PlacedModel, site_weight_delta
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import single_device_mesh
from param_decomp.targets.glu_transformer import (
    GLU_ANATOMY,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.neuron_alignment import (
    NeuronAlignment,
    neuron_aligned_targeted_component_initializer,
    neuron_aligned_zero_u_component_initializer,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

RANKING = [5, 2, 7, 0, 3, 6, 1, 4]  # the same hand ranking the targeted init's tests use


def _model():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            (
                SiteC(site_name(0, "q"), 3),
                SiteC(site_name(0, "k"), 2),
                SiteC(site_name(0, "v"), 2),
                SiteC(site_name(0, "o"), 3),
                SiteC(site_name(0, "gate"), 4),
                SiteC(site_name(0, "up"), 4),
                SiteC(site_name(0, "down"), 6),
            )
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    zero = component_stacks_from_site_arrays(
        sites,
        {
            s.name: (jnp.zeros((s.d_in, s.C), jnp.float32), jnp.zeros((s.C, s.d_out), jnp.float32))
            for s in sites
        },
    )
    stacked = model.weight_deltas(zero)
    weights = {s.name: np.asarray(site_weight_delta(stacked, zero, s.name)) for s in sites}
    return model, sites, weights


def _hand_alignment(sites: tuple[SiteSpec, ...]) -> NeuronAlignment:
    assert all(len(RANKING) >= spec.C for spec in sites)
    return {spec.name: np.asarray(RANKING[: spec.C], dtype=np.int32) for spec in sites}


def _site_arrays(vu: ComponentStacks, name: str) -> tuple[np.ndarray, np.ndarray]:
    site = dict(vu.sites_items())[name]
    return np.asarray(site.V), np.asarray(site.U)


def test_u_is_exactly_zero_and_the_delta_carries_all_of_w():
    """The defining property: nothing is reconstructed at init, so a subcomponent the
    reconstruction losses never ask for stays at exactly zero."""
    model, sites, weights = _model()
    vu = neuron_aligned_zero_u_component_initializer(_hand_alignment(sites))(
        model, jax.random.PRNGKey(1)
    )
    deltas = model.weight_deltas(vu)
    for spec in sites:
        V, U = _site_arrays(vu, spec.name)
        assert U.shape == (spec.C, spec.d_out)
        assert np.array_equal(U, np.zeros_like(U)), spec.name
        np.testing.assert_array_equal((V @ U).T, np.zeros((spec.d_out, spec.d_in), np.float32))
        np.testing.assert_allclose(
            np.asarray(site_weight_delta(deltas, vu, spec.name)), weights[spec.name], atol=1e-5
        )


def test_v_is_bit_identical_to_the_targeted_init():
    """The two aligned inits differ ONLY in `U`. This is what makes a `neuron_aligned_
    targeted` run a valid control for a `neuron_aligned_zero_u` run at the same seed."""
    model, sites, _weights = _model()
    alignment = _hand_alignment(sites)
    aligned = neuron_aligned_targeted_component_initializer(alignment)(model, jax.random.PRNGKey(1))
    zeroed = neuron_aligned_zero_u_component_initializer(alignment)(model, jax.random.PRNGKey(1))
    for spec in sites:
        np.testing.assert_array_equal(
            _site_arrays(aligned, spec.name)[0], _site_arrays(zeroed, spec.name)[0], spec.name
        )


def test_v_still_reads_the_aligned_coordinates():
    """`x @ V` must stay a live, aligned signal — zeroing `U` is not allowed to blank `V`.

    Note the asymmetry inherited from `selected_unit_factors`: on d_out sites (q/k/v,
    gate/up) `V` holds the coordinate's own weight vector; on d_in sites (o, down) it is the
    one-hot selecting that coordinate, because there the weights live in `U`.
    """
    model, sites, weights = _model()
    alignment = _hand_alignment(sites)
    vu = neuron_aligned_zero_u_component_initializer(alignment)(model, jax.random.PRNGKey(1))
    for spec in sites:
        V, _ = _site_arrays(vu, spec.name)
        units = alignment[spec.name]
        _, kind = GLU_ANATOMY.family.parse(spec.name)
        assert np.linalg.norm(V) > 0, spec.name
        if kind in GLU_ANATOMY.row_kinds:  # o, down: V is one-hot on the aligned columns
            expected = np.zeros((spec.d_in, spec.C), np.float32)
            expected[units, np.arange(spec.C)] = 1.0
            np.testing.assert_allclose(V, expected, atol=1e-6)
        else:  # q, k, v, gate, up: V is the coordinate's own weight row
            np.testing.assert_allclose(V, weights[spec.name][units, :].T, atol=1e-5)


def test_the_init_consumes_no_randomness():
    model, sites, _weights = _model()
    initializer = neuron_aligned_zero_u_component_initializer(_hand_alignment(sites))
    a = initializer(model, jax.random.PRNGKey(1))
    b = initializer(model, jax.random.PRNGKey(2))
    for spec in sites:
        for x, y in zip(_site_arrays(a, spec.name), _site_arrays(b, spec.name), strict=True):
            np.testing.assert_array_equal(x, y, spec.name)


def test_the_placed_path_matches_the_eager_values():
    model, sites, _weights = _model()
    initializer = neuron_aligned_zero_u_component_initializer(_hand_alignment(sites))
    eager = initializer(model, jax.random.PRNGKey(1))
    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    with jax.set_mesh(mesh):
        placed = init_model_component_stacks_placed(
            PlacedModel(model=model, placement=rules), jax.random.PRNGKey(1), rules, initializer
        )
    for spec in sites:
        for x, y in zip(
            _site_arrays(eager, spec.name), _site_arrays(placed, spec.name), strict=True
        ):
            np.testing.assert_allclose(x, y, atol=1e-6, err_msg=spec.name)
