"""The `neuron_aligned_targeted` init (SPEC T13) as a `ComponentInitializer`: every
decomposed site — attention included — reads back exactly its aligned coordinates, the
delta carries the rest, and the placed path matches the eager values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    validate_neuron_alignment,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

RANKING = [5, 2, 7, 0, 3, 6, 1, 4]  # a hand ranking of 8 coordinates


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


def test_every_site_reads_back_its_aligned_coordinates_and_leaves_the_rest_to_the_delta():
    model, sites, weights = _model()
    alignment = _hand_alignment(sites)
    validate_neuron_alignment(sites, GLU_ANATOMY, alignment)
    vu = neuron_aligned_targeted_component_initializer(alignment)(model, jax.random.PRNGKey(1))
    deltas = model.weight_deltas(vu)
    for spec in sites:
        W = weights[spec.name]
        units = alignment[spec.name]
        _, kind = GLU_ANATOMY.family.parse(spec.name)
        expected = np.zeros_like(W)
        if kind in GLU_ANATOMY.row_kinds:  # o, down: units on d_in (columns)
            expected[:, units] = W[:, units]
        else:  # q, k, v, gate, up: units on d_out (rows)
            expected[units, :] = W[units, :]
        V, U = _site_arrays(vu, spec.name)
        np.testing.assert_allclose((V @ U).T, expected, atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(site_weight_delta(deltas, vu, spec.name)), W - expected, atol=1e-5
        )
        # Each subcomponent IS one coordinate: its norm is that coordinate's weight norm.
        norms = np.linalg.norm(V, axis=0) * np.linalg.norm(U, axis=1)
        own = (
            np.linalg.norm(W[:, units], axis=0)
            if kind in GLU_ANATOMY.row_kinds
            else np.linalg.norm(W[units, :], axis=1)
        )
        np.testing.assert_allclose(norms, own, rtol=1e-5)


def test_the_init_consumes_no_randomness():
    model, sites, _weights = _model()
    initializer = neuron_aligned_targeted_component_initializer(_hand_alignment(sites))
    a = initializer(model, jax.random.PRNGKey(1))
    b = initializer(model, jax.random.PRNGKey(2))
    for spec in sites:
        for x, y in zip(_site_arrays(a, spec.name), _site_arrays(b, spec.name), strict=True):
            assert np.array_equal(x, y)


def test_validation_refuses_bad_alignments():
    _model_, sites, _weights = _model()
    good = _hand_alignment(sites)
    validate_neuron_alignment(sites, GLU_ANATOMY, good)
    name = site_name(0, "gate")
    with pytest.raises(AssertionError, match="repeated"):
        validate_neuron_alignment(sites, GLU_ANATOMY, {**good, name: np.asarray([1, 1, 2, 3])})
    with pytest.raises(AssertionError):  # C=4, three given
        validate_neuron_alignment(sites, GLU_ANATOMY, {**good, name: np.asarray([1, 2, 3])})
    n = next(s for s in sites if s.name == name).d_out
    with pytest.raises(AssertionError):  # out of range
        validate_neuron_alignment(sites, GLU_ANATOMY, {**good, name: np.asarray([0, 1, 2, n])})
    with pytest.raises(AssertionError):  # a site missing / an undecomposed site named
        validate_neuron_alignment(sites, GLU_ANATOMY, {k: v for k, v in good.items() if k != name})
    with pytest.raises(AssertionError):
        validate_neuron_alignment(
            sites, GLU_ANATOMY, {**good, "layers.5.mlp.gate_proj": good[name]}
        )


def test_placed_init_matches_the_eager_values():
    model, sites, _weights = _model()
    initializer = neuron_aligned_targeted_component_initializer(_hand_alignment(sites))
    eager = initializer(model, jax.random.PRNGKey(1))
    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    with jax.set_mesh(mesh):
        placed = init_model_component_stacks_placed(
            PlacedModel(model=model, placement=rules), jax.random.PRNGKey(1), rules, initializer
        )
    for spec in sites:
        for x, y in zip(
            _site_arrays(placed, spec.name), _site_arrays(eager, spec.name), strict=True
        ):
            np.testing.assert_allclose(x, y, atol=1e-6)
