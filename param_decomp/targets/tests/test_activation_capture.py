"""Typed selective capture: point parity, strict 1:1 binding, and empty-plan compilation."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import BATCH_AXES, MaterializedMasking
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.targets.glu_transformer import (
    GLUDecomposedModel,
    GLULayer,
    _capture_source_for_point,
    _clean_mlp_out,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
    post_attention_tap_key,
    site_output_tap_key,
)
from param_decomp.vendored_jax.llama import rms_norm


def _model():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(
                SiteC(site_name(block, kind), 4)
                for block in (2, 3)
                for kind in ("q", "k", "v", "o", "gate", "up", "down")
            )
        ),
    )
    return cfg, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _all_point_classes(block: int, n_layer: int) -> tuple[str, ...]:
    sites = tuple(site_name(block, kind) for kind in ("q", "k", "v", "o", "gate", "up", "down"))
    return (
        "resid.0",
        f"resid.{block + 1}",
        f"resid.{n_layer}",
        post_attention_tap_key(block),
        attention_input_tap_key(block),
        attention_output_tap_key(block),
        mlp_input_tap_key(block),
        mlp_hidden_tap_key(block),
        *(site_output_tap_key(site) for site in sites),
    )


def test_clean_and_frozen_masked_paths_agree_at_every_declared_point_class():
    cfg, model = _model()
    keys = frozenset(_all_point_classes(2, cfg.n_layer))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)
    clean_forward_result = model.clean_forward(tokens, keys)
    clean_captures = clean_forward_result.captures

    components = init_component_stacks(model.sites, jax.random.PRNGKey(2))
    masked_forward_result = model.masked_forward(
        model.prepare_compute_weights(components),
        tokens,
        masking=MaterializedMasking(component_masks={}),
        capture_keys=keys,
        remat=True,
    )
    masked_captures = masked_forward_result.captures

    assert set(clean_captures) == set(masked_captures) == keys
    for key in keys:
        assert jnp.array_equal(masked_captures[key], clean_captures[key]), key
        assert clean_captures[key].shape[-1] == model._capture_grammar().width_of(key)
    assert jnp.array_equal(masked_forward_result.output, clean_forward_result.output)


def test_new_residual_point_classes_match_direct_block_algebra():
    cfg, model = _model()
    block = 2
    keys = frozenset(("resid.0", post_attention_tap_key(block), f"resid.{cfg.n_layer}"))
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 8), 0, cfg.vocab_size)
    captures = model.clean_forward(tokens, keys).captures

    residual = model.embed_tokens(tokens)
    assert jnp.array_equal(captures["resid.0"], residual)
    expected_post_attention = None
    for index, layer in enumerate(model.layers):
        residual = residual + layer.attn(rms_norm(residual, layer.ln1, model.eps), model.inv_freq)
        if index == block:
            expected_post_attention = residual
        residual = residual + _clean_mlp_out(layer, rms_norm(residual, layer.ln2, model.eps))

    assert expected_post_attention is not None
    assert jnp.allclose(
        captures[post_attention_tap_key(block)], expected_post_attention, rtol=1e-5, atol=1e-5
    )
    assert jnp.allclose(captures[f"resid.{cfg.n_layer}"], residual, rtol=1e-5, atol=1e-5)


def test_forward_result_pytree_reconstruction_is_inert():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)
    clean_forward_result = model.clean_forward(tokens, frozenset({"resid.2"}))

    erased = jax.tree.map(lambda _value: None, clean_forward_result)
    assert erased.output is None
    assert erased.captures == dict.fromkeys(clean_forward_result.captures)

    shaped = jax.eval_shape(lambda tree: tree, clean_forward_result)
    assert all(isinstance(value, jax.ShapeDtypeStruct) for value in shaped.captures.values())


def test_shared_input_has_one_canonical_capture_key():
    _cfg, model = _model()
    qkv = tuple(site_name(2, kind) for kind in ("q", "k", "v"))
    tokens = jnp.ones((1, 4), jnp.int32)
    for site in qkv:
        with pytest.raises(AssertionError, match="unknown transformer activation"):
            model.clean_forward(tokens, frozenset({site}))

    clean_forward_result = model.clean_forward(tokens, frozenset({attention_input_tap_key(2)}))
    assert clean_forward_result.captures.keys() == {attention_input_tap_key(2)}


@pytest.mark.multidevice
def test_capture_values_are_batch_sharded_at_the_producer():
    cfg, model = _model()
    capture_keys = frozenset({"resid.2", attention_input_tap_key(2)})
    mesh = hsdp_mesh()
    tokens = jax.random.randint(
        jax.random.PRNGKey(3), (2 * mesh.devices.size, 8), 0, cfg.vocab_size
    )

    components = model.prepare_compute_weights(
        init_component_stacks(model.sites, jax.random.PRNGKey(5))
    )
    with jax.set_mesh(mesh):
        clean_forward_result = jax.jit(lambda m, x: m.clean_forward(x, capture_keys))(model, tokens)
        masked_forward_result = jax.jit(
            lambda m, prepared, x: m.masked_forward(
                prepared,
                x,
                masking=MaterializedMasking(component_masks={}),
                capture_keys=capture_keys,
                remat=True,
            )
        )(model, components, tokens)

    expected = NamedSharding(mesh, P(BATCH_AXES, None, None))
    for forward_result in (clean_forward_result, masked_forward_result):
        assert forward_result.captures
        assert all(
            value.sharding.is_equivalent_to(expected, value.ndim)
            for value in forward_result.captures.values()
        )


def test_capture_sources_are_unique_and_request_aligned():
    _cfg, model = _model()
    keys = (
        site_output_tap_key(site_name(3, "down")),
        "resid.1",
        attention_input_tap_key(2),
    )
    sources = model._capture_grammar().resolve(keys, _capture_source_for_point)
    assert len(keys) == len(sources) == len(set(sources))


def test_no_capture_wrapper_lowers_to_the_compact_clean_graph():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)

    def compact_clean(m: GLUDecomposedModel, x: Array) -> Array:
        def block(residual: Array, layer: GLULayer) -> tuple[Array, None]:
            residual = residual + layer.attn(rms_norm(residual, layer.ln1, m.eps), m.inv_freq)
            residual = residual + _clean_mlp_out(layer, rms_norm(residual, layer.ln2, m.eps))
            return residual, None

        residual = m.embed_tokens(x)
        residual, _ = jax.lax.scan(block, residual, m.stacked)
        residual = rms_norm(residual, m.norm, m.eps)
        return residual @ m.lm_head.T

    direct = jax.jit(lambda m, x: compact_clean(m, x)).lower(model, tokens).as_text()
    public = jax.jit(lambda m, x: m.clean_forward(x).output).lower(model, tokens).as_text()
    assert public == direct


def test_resolution_fails_at_first_trace_for_unknown_points():
    _cfg, model = _model()
    tokens = jnp.ones((1, 4), jnp.int32)
    with pytest.raises(AssertionError, match="unknown transformer activation"):
        jax.jit(lambda m, x: m.clean_forward(x, frozenset({"python_local_variable"})).output).lower(
            model, tokens
        )
