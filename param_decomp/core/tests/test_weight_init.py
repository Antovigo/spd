"""The target-coupled V/U inits (`pd.weight_init`: `coupled` / `zero_u` /
`neuron_aligned_targeted`).

The coupling claim is checked against each site's own frozen `W`, read back through
`weight_deltas` — the same seam the init itself uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import (
    NeuronAlignment,
    SiteNeuronAlignment,
    init_component_stacks_coupled,
    init_component_stacks_neuron_aligned,
    validate_neuron_alignment,
    with_silenced_u,
    zero_component_stacks,
)
from param_decomp.core.model import PlacedModel, site_weight_delta
from param_decomp.targets.glu_transformer import (
    GLU_ANATOMY,
    glu_site_specs,
    mlp_family_site_cs,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm


def _tiny_model_and_weights():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, mlp_family_site_cs(0, 1, 4))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    # `weight_deltas` is per persistence STACK since #1000; these tests reason per SITE.
    zero = zero_component_stacks(sites)
    stacked = model.weight_deltas(zero)
    return model, sites, {s.name: site_weight_delta(stacked, zero, s.name) for s in sites}


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
    group, (Vs, Us) = next(iter(nonzero.stacks.items()))
    perturbed = nonzero.stacks | {group: (Vs + 1.0, Us + 1.0)}
    bumped = type(nonzero)(stacks=perturbed, site_slots=nonzero.site_slots)
    moved = model.weight_deltas(bumped)
    assert any(
        not jnp.allclose(site_weight_delta(moved, bumped, s.name), weights[s.name]) for s in sites
    )


def test_coupled_init_couples_the_wide_side_to_w():
    _model, sites, weights = _tiny_model_and_weights()
    vu = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1))
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
    coupled = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1))
    vu = with_silenced_u(coupled)
    for spec in sites:
        assert jnp.all(vu.site(spec.name).U == 0.0), spec.name
        assert jnp.any(vu.site(spec.name).V != 0.0), spec.name
        assert jnp.array_equal(vu.site(spec.name).V, coupled.site(spec.name).V), spec.name
    deltas = model.weight_deltas(vu)
    for spec in sites:
        assert jnp.allclose(site_weight_delta(deltas, vu, spec.name), weights[spec.name]), spec.name


def test_placed_init_matches_the_eager_values():
    """The jitted, sharding-placed path reads `W` inside the graph; same values up to fp32
    reassociation in the `W`-image matmul (XLA picks its own layout)."""
    from param_decomp.core.init_placed import init_component_stacks_coupled_placed
    from param_decomp.core.placement import from_config
    from param_decomp.core.sharding import single_device_mesh

    model, sites, weights = _tiny_model_and_weights()
    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    for zero_u in (False, True):
        placed = init_component_stacks_coupled_placed(
            PlacedModel(model=model, placement=rules), jax.random.PRNGKey(1), rules, zero_u=zero_u
        )
        coupled = init_component_stacks_coupled(sites, weights, jax.random.PRNGKey(1))
        eager = with_silenced_u(coupled) if zero_u else coupled
        for spec in sites:
            placed_site, eager_site = placed.site(spec.name), eager.site(spec.name)
            assert jnp.allclose(placed_site.V, eager_site.V, atol=1e-5), spec.name
            assert jnp.allclose(placed_site.U, eager_site.U, atol=1e-5), spec.name


# ----------------------------- neuron_aligned_targeted (SPEC T13) -----------------------------


def _mixed_sites():
    """Two blocks: block 0 fully decomposed (q/k/v/o + gate/up/down, gate/up narrower than
    down), block 1 attention-only — the non-MLP sites must come out as `zero_u`."""
    from param_decomp.core.components import SiteC
    from param_decomp.targets.glu_transformer import canonical_site_cs, site_name

    cfg = tiny_glu_cfg()
    cs = {"q": 3, "k": 3, "v": 3, "o": 5, "gate": 4, "up": 4, "down": 6}
    site_cs = tuple(SiteC(site_name(0, kind), c) for kind in cs for c in (cs[kind],))
    # A kind is one persistence stack across layers, so block 1 repeats block 0's Cs.
    site_cs += tuple(SiteC(site_name(1, kind), cs[kind]) for kind in ("q", "k", "v", "o"))
    sites = glu_site_specs(cfg, canonical_site_cs(site_cs))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    zero = zero_component_stacks(sites)
    stacked = model.weight_deltas(zero)
    return model, sites, {s.name: site_weight_delta(stacked, zero, s.name) for s in sites}


RANKING = np.array([5, 0, 9, 2, 7, 1, 8, 3], dtype=np.int32)
"""A hand ranking over the first 10 of the tiny model's 64 neurons."""


def _hand_alignment() -> NeuronAlignment:
    """One ranking shared by the block's three MLP sites; each takes its prefix."""
    from param_decomp.targets.glu_transformer import site_name

    return {
        site_name(0, "gate"): SiteNeuronAlignment(jnp.asarray(RANKING[:4]), "d_out"),
        site_name(0, "up"): SiteNeuronAlignment(jnp.asarray(RANKING[:4]), "d_out"),
        site_name(0, "down"): SiteNeuronAlignment(jnp.asarray(RANKING[:6]), "d_in"),
    }


def test_neuron_aligned_init_reads_back_the_selected_neurons_and_leaves_the_rest_to_the_delta():
    model, sites, weights = _mixed_sites()
    alignment = _hand_alignment()
    validate_neuron_alignment(sites, alignment)
    vu = init_component_stacks_neuron_aligned(sites, weights, alignment, jax.random.PRNGKey(1))
    deltas = model.weight_deltas(vu)
    for name, site_alignment in alignment.items():
        spec = next(s for s in sites if s.name == name)
        W = np.asarray(weights[name])
        S = np.asarray(site_alignment.neurons)
        site = vu.site(name)
        composed = np.asarray((site.V @ site.U).T)
        expected = np.zeros_like(W)
        if site_alignment.neuron_axis == "d_out":
            expected[S, :] = W[S, :]
            # x @ V is the selected neurons' pre-activation; U is one-hot rows.
            assert np.array_equal(np.asarray(site.V), W[S, :].T)
            assert np.array_equal(np.asarray(site.U), np.asarray(jax.nn.one_hot(S, spec.d_out)))
        else:
            expected[:, S] = W[:, S]
            assert np.array_equal(np.asarray(site.V), np.asarray(jax.nn.one_hot(S, spec.d_in)).T)
            assert np.array_equal(np.asarray(site.U), W[:, S].T)
        assert np.allclose(composed, expected, atol=1e-6), name
        assert np.allclose(
            np.asarray(site_weight_delta(deltas, vu, name)), W - expected, atol=1e-5
        ), name
        # Each subcomponent carries exactly its neuron's weight norm.
        norms = np.linalg.norm(np.asarray(site.V), axis=0) * np.linalg.norm(
            np.asarray(site.U), axis=1
        )
        selected = W[S, :] if site_alignment.neuron_axis == "d_out" else W[:, S].T
        assert np.allclose(norms, np.linalg.norm(selected, axis=1), atol=1e-5), name


def test_neuron_aligned_init_gives_every_other_site_the_zero_u_values_bit_for_bit():
    _model, sites, weights = _mixed_sites()
    alignment = _hand_alignment()
    key = jax.random.PRNGKey(7)
    aligned = init_component_stacks_neuron_aligned(sites, weights, alignment, key)
    zero_u = with_silenced_u(init_component_stacks_coupled(sites, weights, key))
    for spec in sites:
        if spec.name in alignment:
            continue
        assert jnp.array_equal(aligned.site(spec.name).V, zero_u.site(spec.name).V), spec.name
        assert jnp.array_equal(aligned.site(spec.name).U, zero_u.site(spec.name).U), spec.name
        assert jnp.all(aligned.site(spec.name).U == 0.0), spec.name


def test_neuron_aligned_prefixes_nest_across_the_block_sites():
    """gate/up take the top-4, down the top-6 of ONE ranking: the writers' neurons are a
    prefix of the reader's, in the same slot order."""
    alignment = _hand_alignment()
    gate = np.asarray(alignment["layers.0.mlp.gate_proj"].neurons)
    down = np.asarray(alignment["layers.0.mlp.down_proj"].neurons)
    assert np.array_equal(down[: len(gate)], gate)
    assert set(gate) < set(down)


def test_neuron_alignment_validation_refuses_bad_indices():
    _model, sites, _weights = _mixed_sites()
    good = _hand_alignment()
    name = "layers.0.mlp.gate_proj"
    validate_neuron_alignment(sites, good)
    with pytest.raises(AssertionError, match="repeated"):
        bad = {**good, name: SiteNeuronAlignment(jnp.asarray([1, 1, 2, 3]), "d_out")}
        validate_neuron_alignment(sites, bad)
    with pytest.raises(AssertionError):
        bad = {**good, name: SiteNeuronAlignment(jnp.asarray([1, 2, 3]), "d_out")}  # C=4
        validate_neuron_alignment(sites, bad)
    with pytest.raises(AssertionError):
        n = tiny_glu_cfg().n_intermediate
        bad = {**good, name: SiteNeuronAlignment(jnp.asarray([0, 1, 2, n]), "d_out")}
        validate_neuron_alignment(sites, bad)
    with pytest.raises(AssertionError, match="undecomposed"):
        validate_neuron_alignment(sites, {"layers.5.mlp.gate_proj": good[name]})


def test_neuron_aligned_placed_init_matches_the_eager_values():
    from param_decomp.core.init_placed import init_component_stacks_neuron_aligned_placed
    from param_decomp.core.placement import from_config
    from param_decomp.core.sharding import single_device_mesh

    model, sites, weights = _mixed_sites()
    alignment = _hand_alignment()
    mesh = single_device_mesh()
    rules = from_config("ddp", mesh, sites)
    placed = init_component_stacks_neuron_aligned_placed(
        PlacedModel(model=model, placement=rules), jax.random.PRNGKey(1), rules, alignment
    )
    eager = init_component_stacks_neuron_aligned(sites, weights, alignment, jax.random.PRNGKey(1))
    for spec in sites:
        assert jnp.allclose(placed.site(spec.name).V, eager.site(spec.name).V, atol=1e-5), spec.name
        assert jnp.allclose(placed.site(spec.name).U, eager.site(spec.name).U, atol=1e-5), spec.name
    assert GLU_ANATOMY.mlp.hidden == ("gate", "up")
