"""SPEC S11': per-layer Bernoulli routing with a scheduled probability, and single-site routing."""

import jax
import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter

from param_decomp.core.configs import LayerProbabilityRoutingConfig, SubsetRoutingType
from param_decomp.core.objective import routing_sampler_from_config
from param_decomp.core.recon import site_layer_index

SITES = (
    "layers.0.self_attn.q_proj",
    "layers.0.mlp.down_proj",
    "layers.1.self_attn.o_proj",
    "layers.1.mlp.gate_proj",
    "layers.1.mlp.up_proj",
    "layers.2.mlp.down_proj",
)

RAMP = {
    "type": "layer_probability",
    "p": {"max_val": 0.9, "points": [{"at": 0.0, "frac": 0.1}, {"at": 1.0, "frac": 1.0}]},
}


def test_site_layer_index_reads_the_first_numbered_segment():
    assert site_layer_index("layers.30.mlp.up_proj") == 30
    assert site_layer_index("h.2.attn.c_attn") == 2
    with pytest.raises(AssertionError, match="names no numbered block"):
        site_layer_index("lm_head")


def test_config_parses_bare_float_and_refuses_p_above_one():
    adapter = TypeAdapter(SubsetRoutingType)
    constant = adapter.validate_python({"type": "layer_probability", "p": 0.25})
    assert isinstance(constant, LayerProbabilityRoutingConfig)
    assert constant.p.max_val == 0.25
    with pytest.raises(ValueError, match="peaks at 1.5"):
        adapter.validate_python({"type": "layer_probability", "p": 1.5})


def test_sites_of_one_layer_share_their_draw_and_layers_are_independent():
    routing = TypeAdapter(SubsetRoutingType).validate_python(RAMP)
    sample = routing_sampler_from_config(routing, SITES, n_draws=2)
    draws = sample(jax.random.PRNGKey(0), (64, 32), jnp.asarray(0.5, jnp.float32))
    assert len(draws) == 2
    for routes in draws:
        assert routes is not None
        assert set(routes) == set(SITES)
        assert jnp.array_equal(routes[SITES[0]], routes[SITES[1]])
        assert jnp.array_equal(routes[SITES[2]], routes[SITES[3]])
        assert jnp.array_equal(routes[SITES[2]], routes[SITES[4]])
        assert not jnp.array_equal(routes[SITES[0]], routes[SITES[2]])
        assert not jnp.array_equal(routes[SITES[2]], routes[SITES[5]])
    first, second = draws
    assert first is not None and second is not None
    assert not jnp.array_equal(first[SITES[0]], second[SITES[0]])


def test_routing_rate_follows_the_schedule_under_jit():
    routing = TypeAdapter(SubsetRoutingType).validate_python(RAMP)
    sample = routing_sampler_from_config(routing, SITES, n_draws=1)

    def rate(train_frac: jax.Array) -> jax.Array:
        (routes,) = sample(jax.random.PRNGKey(1), (256, 256), train_frac)
        assert routes is not None
        return jnp.mean(jnp.stack([routes[s] for s in SITES]).astype(jnp.float32))

    # p(t) = 0.9 * (0.1 + 0.9 t): 0.09 at t = 0, 0.495 at t = 0.5, 0.9 at t = 1.
    for train_frac, expected in ((0.0, 0.09), (0.5, 0.495), (1.0, 0.9)):
        observed = float(jax.jit(rate)(jnp.asarray(train_frac, jnp.float32)))
        assert abs(observed - expected) < 0.01, (train_frac, observed)


def test_single_site_routes_exactly_one_site_per_sequence():
    routing = TypeAdapter(SubsetRoutingType).validate_python({"type": "single_site"})
    sample = routing_sampler_from_config(routing, SITES, n_draws=2)
    leading = (512, 7)
    for routes in sample(jax.random.PRNGKey(2), leading, jnp.zeros(())):
        assert routes is not None
        stacked = jnp.stack([routes[s] for s in SITES]).astype(jnp.int32)
        assert stacked.shape == (len(SITES), *leading)
        assert jnp.all(stacked.sum(0) == 1)
        assert jnp.all(stacked == stacked[:, :, :1])
        per_site = stacked[:, :, 0].mean(1)
        assert jnp.all(jnp.abs(per_site - 1 / len(SITES)) < 0.06), per_site
