"""Prefix reuse (`ResidualStart`) must be a pure performance change.

A target whose decomposed sites all sit past block `split_layer` runs that frozen lead once
per step and hands every forward the resulting activation. The lead is mask-independent, so
resuming from it must reproduce the token path EXACTLY — not within tolerance: it is the same
ops on the same values, so any difference is a bug, not reassociation.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import (
    ForwardResult,
    MaterializedMasking,
    ResidualStart,
    site_weight_delta,
)
from param_decomp.targets.glu_transformer import (
    GLUDecomposedModel,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    mlp_hidden_tap_key,
    site_output_tap_key,
)
from param_decomp.vendored_jax.llama import LlamaConfig

KINDS = ("q", "k", "v", "o", "gate", "up", "down")


def _model(blocks: tuple[int, ...]) -> tuple[LlamaConfig, GLUDecomposedModel]:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(SiteC(site_name(block, kind), 4) for block in blocks for kind in KINDS)
        ),
    )
    return cfg, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _tokens(cfg: LlamaConfig) -> Array:
    return jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, cfg.vocab_size)


def test_split_layer_is_the_first_decomposed_block():
    cfg, model = _model((3, 4))
    assert model.split_layer == 3
    assert model.stacked_prefix is not None
    assert jax.tree_util.tree_leaves(model.stacked_prefix)[0].shape[0] == 3
    # `stacked` is the DECOMPOSED SPAN (blocks 3-4 here), not everything above the prefix:
    # the frozen tail is its own field so no forward slices it out of a shared stack.
    assert model.tail_layer == 5
    assert jax.tree_util.tree_leaves(model.stacked)[0].shape[0] == 2
    assert model.stacked_tail is not None
    assert jax.tree_util.tree_leaves(model.stacked_tail)[0].shape[0] == cfg.n_layer - 5
    # The per-layer view still spans the whole model, in global order.
    assert len(model.layers) == cfg.n_layer


def test_no_prefix_when_block_zero_is_decomposed():
    _cfg, model = _model((0, 1))
    assert model.split_layer == 0
    assert model.stacked_prefix is None
    with pytest.raises(AssertionError, match="no frozen prefix"):
        model.prefix_residual(_tokens(_cfg), None)


def test_clean_forward_from_residual_start_matches_tokens():
    cfg, model = _model((3, 4))
    tokens = _tokens(cfg)
    start = ResidualStart(model.prefix_residual(tokens, None))
    assert jnp.array_equal(
        model.clean_forward(start, placement=None).output,
        model.clean_forward(tokens, placement=None).output,
    )


def test_masked_forward_and_captures_from_residual_start_match_tokens():
    cfg, model = _model((3, 4))
    tokens = _tokens(cfg)
    keys = frozenset(
        (
            f"resid.{model.split_layer}",  # the prefix boundary — served from the start
            f"resid.{cfg.n_layer}",
            attention_input_tap_key(3),
            mlp_hidden_tap_key(4),
            *(site_output_tap_key(site) for site in model.site_names),
        )
    )
    components = init_component_stacks(model.sites, jax.random.PRNGKey(2))
    prepared = model.prepare_compute_weights(components, None)
    start = ResidualStart(model.prefix_residual(tokens, None))

    def run(inputs: Array | ResidualStart) -> ForwardResult:
        return model.masked_forward(
            prepared,
            inputs,
            masking=MaterializedMasking(
                component_masks={site: jnp.ones((2, 8, 4)) for site in model.site_names}
            ),
            capture_keys=keys,
            remat=True,
            placement=None,
        )

    from_tokens, from_start = run(tokens), run(start)
    assert jnp.array_equal(from_start.output, from_tokens.output)
    assert set(from_start.captures) == set(from_tokens.captures) == keys
    for key in keys:
        assert jnp.array_equal(from_start.captures[key], from_tokens.captures[key]), key


def test_prefix_boundary_capture_is_the_start_activation():
    cfg, model = _model((3, 4))
    tokens = _tokens(cfg)
    prefix = model.prefix_residual(tokens, None)
    captured = model.clean_forward(
        tokens, frozenset((f"resid.{model.split_layer}",)), placement=None
    ).captures
    assert jnp.array_equal(captured[f"resid.{model.split_layer}"], prefix)


def test_tokens_still_capture_inside_the_prefix():
    """Splitting the stack must not cost capability: from token inputs the prefix runs
    through the same capture machinery, so every point an unsplit model answered still
    resolves — that path just never runs in training, where no capture point lives below the
    first decomposed block."""
    cfg, model = _model((3, 4))
    _cfg, unsplit = _model((0, 1))  # same PRNGKey, so identical frozen weights
    assert unsplit.split_layer == 0
    tokens = _tokens(cfg)

    wanted = frozenset(f"resid.{i}" for i in range(cfg.n_layer + 1))
    captures = model.clean_forward(tokens, wanted, placement=None).captures
    reference = unsplit.clean_forward(tokens, wanted, placement=None).captures
    assert set(captures) == wanted
    for key in wanted:
        assert jnp.array_equal(captures[key], reference[key]), key


def test_capture_below_the_prefix_is_refused_from_a_residual_start():
    """The one real restriction: a `ResidualStart` consumed the prefix, so its internals are
    genuinely unrecoverable."""
    _cfg, model = _model((3, 4))
    start = ResidualStart(model.prefix_residual(_tokens(_cfg), None))
    with pytest.raises(AssertionError, match="below block 3 from a ResidualStart"):
        model.clean_forward(start, frozenset(("resid.1",)), placement=None)


def test_a_multi_block_span_above_the_split_matches_tokens():
    """The segment offsets: with a span of blocks 3-5 inside an 8-block model there is a
    prefix (0-2) AND a tail (6-7), so a `ResidualStart` has to line the span up against both.
    (This replaces a pre-dechunk test of a partially-live chunk — masked forwards are total
    over the model's sites now, so a chunk cannot be partly live.)"""
    cfg, model = _model((3, 4, 5))
    assert model.split_layer == 3 and model.tail_layer == 6
    assert model.stacked_prefix is not None and model.stacked_tail is not None
    tokens = _tokens(cfg)
    components = init_component_stacks(model.sites, jax.random.PRNGKey(4))
    prepared = model.prepare_compute_weights(components, None)
    masking = MaterializedMasking(
        component_masks={site: jnp.ones((2, 8, 4)) for site in model.site_names}
    )
    keys = frozenset(
        (f"resid.{cfg.n_layer}", *(site_output_tap_key(site) for site in model.site_names))
    )
    start = ResidualStart(model.prefix_residual(tokens, None))

    from_tokens = model.masked_forward(
        prepared, tokens, masking=masking, capture_keys=keys, remat=True, placement=None
    )
    from_start = model.masked_forward(
        prepared, start, masking=masking, capture_keys=keys, remat=True, placement=None
    )
    assert jnp.array_equal(from_start.output, from_tokens.output)
    assert set(from_start.captures) == set(from_tokens.captures) == keys
    for key in keys:
        assert jnp.array_equal(from_start.captures[key], from_tokens.captures[key]), key


def test_weight_deltas_index_the_suffix_stack():
    _cfg, model = _model((3, 4))
    components = init_component_stacks(model.sites, jax.random.PRNGKey(3))
    # `weight_deltas` is per persistence STACK since #1000; the claim here is that the
    # per-site read lands on the SPAN stack, not on a block the split moved out of it.
    stacked_deltas = model.weight_deltas(components)
    for site in model.site_names:
        delta = site_weight_delta(stacked_deltas, components, site)
        assert jnp.all(jnp.isfinite(delta)), site
