"""The ranking harvest and artifact behind `initialization: neuron_aligned_targeted`
(SPEC T13): every ranking family — MLP neurons and the q/k/v/o attention channels — from
the sweep's moments, the artifact round trip with its provenance refusals, and the
rankings → per-site alignment mapping."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import SiteC
from param_decomp.core.model import PlacedModel
from param_decomp.targets.glu_transformer import (
    GLU_ANATOMY,
    canonical_site_cs,
    glu_site_specs,
    site_name,
)
from param_decomp.targets.neuron_alignment import (
    UNIT_KINDS,
    NeuronRanks,
    NeuronRanksMeta,
    UnitKind,
    accumulate_neuron_moments,
    alignment_coverage,
    assert_neuron_ranks_provenance,
    block_scores,
    block_tap_keys,
    capture_groups,
    frozen_mlp_down_weight,
    make_moments_step,
    neuron_alignment_from_ranks,
    pool_slices,
    pool_tokens_sha256,
    rank_units,
    ranked_blocks,
    read_neuron_ranks,
    tap_widths,
    unit_counts,
    unit_kind_of,
    write_neuron_ranks,
)
from param_decomp.targets.testing import capture_clean, tiny_glu_cfg, tiny_glu_decomposed_lm


def _model(site_cs: tuple[SiteC, ...]):
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, canonical_site_cs(site_cs))
    return cfg, sites, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _pool(n_prompts: int, n_positions: int, vocab: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, vocab, size=(n_prompts, n_positions), dtype=np.int32)


def _meta(n_units: dict[UnitKind, int], layers: list[int], tokens: np.ndarray) -> NeuronRanksMeta:
    return NeuronRanksMeta(
        target="tiny",
        tokens_sha256=pool_tokens_sha256(tokens),
        n_prompts=tokens.shape[0],
        prompt_len=tokens.shape[1],
        statistic="unit_energy",
        layers=layers,
        n_units=n_units,
    )


def _tiny_counts() -> dict[UnitKind, int]:
    cfg = tiny_glu_cfg()
    return {
        "mlp": cfg.n_intermediate,
        "q": cfg.n_head * cfg.head_dim,
        "k": cfg.n_kv_head * cfg.head_dim,
        "v": cfg.n_kv_head * cfg.head_dim,
        "o": cfg.n_head * cfg.head_dim,
    }


def test_every_matrix_kind_maps_to_its_ranking_family():
    assert unit_kind_of(GLU_ANATOMY, "gate") == unit_kind_of(GLU_ANATOMY, "down") == "mlp"
    assert [unit_kind_of(GLU_ANATOMY, k) for k in ("q", "k", "v", "o")] == ["q", "k", "v", "o"]
    with pytest.raises(AssertionError):
        unit_kind_of(GLU_ANATOMY, "embed")


def test_unit_counts_and_ranked_blocks():
    _cfg, sites, model = _model(
        (
            SiteC(site_name(0, "q"), 2),
            SiteC(site_name(3, "gate"), 2),
            SiteC(site_name(3, "down"), 2),
        )
    )
    assert unit_counts(model) == _tiny_counts()
    assert ranked_blocks(sites, GLU_ANATOMY) == (0, 3)


def test_frozen_down_weight_is_read_for_undecomposed_blocks_too():
    cfg, _sites, model = _model((SiteC(site_name(3, "gate"), 2),))
    for block in (0, 3, cfg.n_layer - 1):  # prefix, span, tail
        assert frozen_mlp_down_weight(model, block).shape == (cfg.n_embd, cfg.n_intermediate)


def test_pool_slices_visit_every_row_exactly_once():
    tokens = np.arange(23 * 3, dtype=np.int32).reshape(23, 3)  # every row unique
    seen: list[np.ndarray] = []
    for rows, mask in pool_slices(tokens, 5):
        assert rows.shape == (5, 3) and mask.shape == (5,)
        seen.append(rows[mask])
    assert np.array_equal(np.concatenate(seen), tokens)
    assert sum(1 for _ in pool_slices(tokens, 5)) == 5  # 23 = 4 full + 1 padded


def test_scores_match_a_hand_computation_from_the_taps():
    cfg, _sites, model = _model(
        (
            SiteC(site_name(2, "gate"), 2),
            SiteC(site_name(2, "up"), 2),
            SiteC(site_name(4, "o"), 2),
        )
    )
    blocks = (2, 4)
    tokens = _pool(7, 4, cfg.vocab_size)
    widths = tap_widths(unit_counts(model))
    step = make_moments_step(PlacedModel(model=model, placement=None), blocks)
    slices = ((jnp.asarray(rows), jnp.asarray(mask)) for rows, mask in pool_slices(tokens, 3))
    totals = accumulate_neuron_moments(step, blocks, widths, slices)

    keys = [key for b in blocks for key in block_tap_keys(GLU_ANATOMY, b).values()]
    captures = capture_clean(model, jnp.asarray(tokens), keys)
    for block in blocks:
        taps = block_tap_keys(GLU_ANATOMY, block)
        acts = {tap: np.asarray(captures[key], dtype=np.float64) for tap, key in taps.items()}
        assert totals[block].n_tokens == 7 * 4
        for tap, x in acts.items():
            assert x.shape[-1] == widths[tap], (tap, x.shape)
            assert np.allclose(
                totals[block].sum_sq[tap], (x * x).sum(axis=(0, 1)), rtol=1e-4, atol=1e-5
            )
        scores = block_scores(model, block, totals[block])
        layer = model.frozen_block(block)
        Wd = np.asarray(frozen_mlp_down_weight(model, block), dtype=np.float64)
        Wo = np.asarray(layer.attn.wo, dtype=np.float64)
        mean_sq = {tap: (x * x).mean(axis=(0, 1)) for tap, x in acts.items()}
        assert np.allclose(
            scores["mlp"], mean_sq["mlp"] * np.sum(Wd**2, axis=0), rtol=1e-4, atol=1e-6
        )
        assert np.allclose(scores["q"], mean_sq["q"], rtol=1e-4, atol=1e-6)
        assert np.allclose(scores["k"], mean_sq["k"], rtol=1e-4, atol=1e-6)
        o_expected = mean_sq["z"] * np.sum(Wo**2, axis=0)
        assert np.allclose(scores["o"], o_expected, rtol=1e-4, atol=1e-6)
        # v channel (g, d) sums the write energy of attention channel (h, d) over its group.
        n_rep = cfg.n_head // cfg.n_kv_head
        v_expected = o_expected.reshape(cfg.n_kv_head, n_rep, cfg.head_dim).sum(axis=1).reshape(-1)
        assert np.allclose(scores["v"], v_expected, rtol=1e-4, atol=1e-6)


def test_rank_units_is_descending_with_index_tie_break():
    score = np.array([0.5, 2.0, 0.5, 2.0, 1.0])
    assert rank_units(score).tolist() == [1, 3, 4, 0, 2]


def test_capture_groups_respect_the_byte_budget():
    blocks = tuple(range(7))
    budget = 4 * 5 * 10 * 4 * 3
    assert capture_groups(blocks, 4, 5, 10, budget) == ((0, 1, 2), (3, 4, 5), (6,))
    assert capture_groups(blocks, 4, 5, 10, budget_bytes=1) == tuple((b,) for b in blocks)


def _random_ranks(
    counts: dict[UnitKind, int], layers: list[int], seed: int
) -> tuple[dict[UnitKind, dict[int, np.ndarray]], dict[UnitKind, dict[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    rank: dict[UnitKind, dict[int, np.ndarray]] = {
        kind: {b: rng.permutation(counts[kind]).astype(np.int32) for b in layers}
        for kind in UNIT_KINDS
    }
    score: dict[UnitKind, dict[int, np.ndarray]] = {
        kind: {b: np.sort(rng.random(counts[kind]).astype(np.float32))[::-1] for b in layers}
        for kind in UNIT_KINDS
    }
    return rank, score


def test_artifact_round_trip_and_provenance_refusals(tmp_path: Path):
    tokens = _pool(5, 3, 50)
    counts, layers = _tiny_counts(), [1, 3]
    rank, score = _random_ranks(counts, layers, 1)
    out = tmp_path / "ranks"
    write_neuron_ranks(out, _meta(counts, layers, tokens), rank, score)
    ranks = read_neuron_ranks(out)
    assert ranks.meta.layers == [1, 3] and set(ranks.rank["q"]) == {1, 3}
    for kind in UNIT_KINDS:
        assert np.array_equal(ranks.rank[kind][1], rank[kind][1])
        assert np.isclose(ranks.coverage(kind, 1, counts[kind]), 1.0)
        assert 0 < ranks.coverage(kind, 1, 4) < 1
    assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens, (1, 3))
    with pytest.raises(AssertionError, match="harvested on"):
        assert_neuron_ranks_provenance(ranks.meta, "other-model", tokens, (1,))
    with pytest.raises(AssertionError, match="different prompt pool"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens[:-1], (1,))
    with pytest.raises(AssertionError, match="different prompt pool"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens + 1, (1,))
    with pytest.raises(AssertionError, match="decomposes"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens, (1, 2))
    # A ranking that is not a permutation (here a negative index) is refused at read.
    rank["k"][1][0] = -1
    write_neuron_ranks(tmp_path / "bad", _meta(counts, layers, tokens), rank, score)
    with pytest.raises(AssertionError, match="permutation"):
        read_neuron_ranks(tmp_path / "bad")
    # An artifact of another statistic (the MLP-only harvest) refuses at read.
    stale = _meta(counts, layers, tokens).model_copy(update={"statistic": "write_energy"})
    write_neuron_ranks(tmp_path / "stale", stale, rank, score)
    with pytest.raises(AssertionError, match="re-harvest"):
        read_neuron_ranks(tmp_path / "stale")


def test_alignment_from_ranks_takes_prefixes_per_site_kind():
    _cfg, sites, _model_ = _model(
        (
            SiteC(site_name(1, "q"), 3),
            SiteC(site_name(1, "v"), 2),
            SiteC(site_name(1, "o"), 5),
            SiteC(site_name(1, "gate"), 4),
            SiteC(site_name(1, "up"), 4),
            SiteC(site_name(1, "down"), 6),
            SiteC(site_name(3, "down"), 5),
        )
    )
    counts = _tiny_counts()
    rank, score = _random_ranks(counts, [1, 3], 2)
    ranks = NeuronRanks(_meta(counts, [1, 3], _pool(1, 2, 8)), rank, score)
    alignment = neuron_alignment_from_ranks(ranks, sites, GLU_ANATOMY)
    assert set(alignment) == {spec.name for spec in sites}
    assert np.array_equal(alignment[site_name(1, "q")], rank["q"][1][:3])
    assert np.array_equal(alignment[site_name(1, "v")], rank["v"][1][:2])
    assert np.array_equal(alignment[site_name(1, "o")], rank["o"][1][:5])
    gate, down = alignment[site_name(1, "gate")], alignment[site_name(1, "down")]
    assert np.array_equal(gate, rank["mlp"][1][:4])
    assert np.array_equal(down, rank["mlp"][1][:6])  # nested prefix of the same ranking
    assert np.array_equal(alignment[site_name(3, "down")], rank["mlp"][3][:5])
    coverage = alignment_coverage(ranks, sites, GLU_ANATOMY)
    assert coverage[site_name(1, "down")] > coverage[site_name(1, "gate")]
    assert coverage[site_name(1, "o")] > coverage[site_name(1, "q")]


def test_alignment_refuses_more_components_than_coordinates():
    too_many = tiny_glu_cfg().n_kv_head * tiny_glu_cfg().head_dim + 1
    _cfg, sites, _model_ = _model((SiteC(site_name(1, "k"), too_many),))
    counts = _tiny_counts()
    rank, score = _random_ranks(counts, [1], 3)
    ranks = NeuronRanks(_meta(counts, [1], _pool(1, 2, 8)), rank, score)
    with pytest.raises(AssertionError, match="C <="):
        neuron_alignment_from_ranks(ranks, sites, GLU_ANATOMY)
