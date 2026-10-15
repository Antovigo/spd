"""Prefix reuse (`PrefixResumedBatch`) must be a pure performance change.

A target whose decomposed sites all sit past block `split_layer` runs that frozen lead once
per step — in the clean forward, whose conditioning carries the resulting activation — and
every masked forward handed that conditioning resumes from it. The lead is mask-independent,
so resuming from it must reproduce the token path EXACTLY — not within tolerance: it is the
same ops on the same values, so any difference is a bug, not reassociation.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp

from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.model import ForwardResult, MaterializedMasking, site_weight_delta
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.target_ports.llama import LlamaConfig
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import materialized_logits, tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    PrefixResumedBatch,
    TransformerDecomposedModel,
    TransformerPreparedWeights,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.transformer_taps import (
    attention_input_tap_key,
    mlp_hidden_tap_key,
    site_output_tap_key,
)

KINDS = ("q", "k", "v", "o", "gate", "up", "down")


def _model(blocks: tuple[int, ...]) -> tuple[LlamaConfig, TransformerDecomposedModel]:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        canonical_site_cs(
            tuple(SiteC(site_name(block, kind), 4) for block in blocks for kind in KINDS)
        ),
    )
    return cfg, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _batch(cfg: LlamaConfig, seed: int = 1) -> LMBatchWithDocuments:
    tokens = jax.random.randint(jax.random.PRNGKey(seed), (2, 8), 0, cfg.vocab_size)
    return LMBatchWithDocuments.from_unsegmented_sequences(tokens)


def _resumed(model: TransformerDecomposedModel, batch: LMBatchWithDocuments) -> PrefixResumedBatch:
    """The conditioning a clean forward hands its masked forwards."""
    start = model.clean_forward(batch, placement=None).conditioning
    assert isinstance(start, PrefixResumedBatch), type(start)
    return start


def _on_other_tokens(start: PrefixResumedBatch, cfg: LlamaConfig) -> PrefixResumedBatch:
    """The same start activation riding DIFFERENT tokens. A masked forward that matches the
    token path from this batch can only have resumed from `prefix_residual`: re-running the
    prefix from the batch's tokens would see the wrong sequence. (The pre-merge
    `ResidualStart` carried no tokens at all, so this is what keeps the pin as strong.)"""
    other = _batch(cfg, seed=99).batch.token_ids
    assert not jnp.array_equal(other, start.batch.token_ids)
    return replace(start, batch=LMBatch(other))


def _masked(
    model: TransformerDecomposedModel,
    prepared: TransformerPreparedWeights,
    inputs: LMBatchWithDocuments,
    keys: frozenset[str],
    *,
    remat: bool,
) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
    masking = MaterializedMasking(
        component_masks={site: jnp.ones((2, 8, 4)) for site in model.site_names}
    )
    return model.masked_forward(
        prepared,
        inputs,
        masking=model.prepare_masking(masking),
        routes=None,
        capture_keys=keys,
        remat=remat,
        placement=None,
    )


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
    """With no frozen lead there is nothing to resume from: the clean forward hands back
    the plain batch, from both the compact and the capturing clean path."""
    cfg, model = _model((0, 1))
    assert model.split_layer == 0
    assert model.stacked_prefix is None
    batch = _batch(cfg)
    assert model.clean_forward(batch, placement=None).conditioning is batch
    assert model.clean_forward(batch, frozenset(("resid.1",)), placement=None).conditioning is batch


def test_prefix_boundary_capture_is_the_start_activation():
    """The conditioning's `prefix_residual` IS the residual entering `split_layer`, and the
    compact (no-capture) and capturing clean paths hand back the same one."""
    cfg, model = _model((3, 4))
    batch = _batch(cfg)
    compact = _resumed(model, batch)
    boundary = f"resid.{model.split_layer}"
    capturing = model.clean_forward(batch, frozenset((boundary,)), placement=None)
    assert isinstance(capturing.conditioning, PrefixResumedBatch)
    assert jnp.array_equal(capturing.captures[boundary], compact.prefix_residual)
    assert jnp.array_equal(capturing.conditioning.prefix_residual, compact.prefix_residual)
    assert jnp.array_equal(compact.batch.token_ids, batch.batch.token_ids)


def test_masked_forward_and_captures_from_the_resumed_batch_match_tokens():
    cfg, model = _model((3, 4))
    batch = _batch(cfg)
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
    start = _resumed(model, batch)

    from_tokens = _masked(model, prepared, batch, keys, remat=True)
    for resumed in (start, _on_other_tokens(start, cfg)):
        from_start = _masked(model, prepared, resumed, keys, remat=True)
        assert jnp.array_equal(
            materialized_logits(from_start.output), materialized_logits(from_tokens.output)
        )
        assert set(from_start.captures) == set(from_tokens.captures) == keys
        for key in keys:
            assert jnp.array_equal(from_start.captures[key], from_tokens.captures[key]), key


def test_tokens_still_capture_inside_the_prefix():
    """Splitting the stack must not cost capability: from token inputs the prefix runs
    through the same capture machinery, so every point an unsplit model answered still
    resolves — that path just never runs in training, where no capture point lives below the
    first decomposed block."""
    cfg, model = _model((3, 4))
    _cfg, unsplit = _model((0, 1))  # same PRNGKey, so identical frozen weights
    assert unsplit.split_layer == 0
    batch = _batch(cfg)

    wanted = frozenset(f"resid.{i}" for i in range(cfg.n_layer + 1))
    captures = model.clean_forward(batch, wanted, placement=None).captures
    reference = unsplit.clean_forward(batch, wanted, placement=None).captures
    assert set(captures) == wanted
    for key in wanted:
        assert jnp.array_equal(captures[key], reference[key]), key


def test_capture_inside_the_prefix_from_a_resumed_batch_runs_the_prefix_from_its_tokens():
    """A resumed batch still carries its tokens, so a masked forward asked for a capture
    INSIDE the prefix (here the embedding residual and block 1's output) runs the prefix
    from them instead of resuming — the upstream contract that a masked forward handed the
    clean conditioning can capture any point (eval's fresh-PGD hidden-acts points and the
    activation-capture agreement test both rely on it). It must match the token path
    exactly, and it must NOT have resumed: on a start riding other tokens it reads THOSE."""
    cfg, model = _model((3, 4))
    batch = _batch(cfg)
    start = _resumed(model, batch)
    prepared = model.prepare_compute_weights(
        init_component_stacks(model.sites, jax.random.PRNGKey(2)), None
    )
    keys = frozenset(("resid.0", "resid.2", "resid.3", "resid.5"))
    from_tokens = _masked(model, prepared, batch, keys, remat=False)
    from_start = _masked(model, prepared, start, keys, remat=False)
    assert jnp.array_equal(
        materialized_logits(from_start.output), materialized_logits(from_tokens.output)
    )
    assert set(from_start.captures) == set(from_tokens.captures) == keys
    for key in keys:
        assert jnp.array_equal(from_start.captures[key], from_tokens.captures[key]), key
    other = _on_other_tokens(start, cfg)
    from_other = _masked(model, prepared, other, keys, remat=False)
    assert not jnp.array_equal(from_other.captures["resid.0"], from_tokens.captures["resid.0"])


def test_a_multi_block_span_above_the_split_matches_tokens():
    """The segment offsets: with a span of blocks 3-5 inside an 8-block model there is a
    prefix (0-2) AND a tail (6-7), so a resumed batch has to line the span up against both.
    (This replaces a pre-dechunk test of a partially-live chunk — masked forwards are total
    over the model's sites now, so a chunk cannot be partly live.)"""
    cfg, model = _model((3, 4, 5))
    assert model.split_layer == 3 and model.tail_layer == 6
    assert model.stacked_prefix is not None and model.stacked_tail is not None
    batch = _batch(cfg)
    components = init_component_stacks(model.sites, jax.random.PRNGKey(4))
    prepared = model.prepare_compute_weights(components, None)
    keys = frozenset(
        (f"resid.{cfg.n_layer}", *(site_output_tap_key(site) for site in model.site_names))
    )
    start = _resumed(model, batch)

    from_tokens = _masked(model, prepared, batch, keys, remat=True)
    for resumed in (start, _on_other_tokens(start, cfg)):
        from_start = _masked(model, prepared, resumed, keys, remat=True)
        assert jnp.array_equal(
            materialized_logits(from_start.output), materialized_logits(from_tokens.output)
        )
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
