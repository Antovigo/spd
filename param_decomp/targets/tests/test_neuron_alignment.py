"""The neuron-ranking harvest and artifact behind `weight_init: neuron_aligned_targeted`
(SPEC T13): the score against a hand computation from the `mlp_hidden` tap, the
exhaustive sweep's row coverage, the ranking's determinism, the artifact round-trip and
its provenance refusals, and the ranking -> alignment mapping on partial blocks."""

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
    BosPolicy,
    NeuronRanks,
    NeuronRanksMeta,
    accumulate_neuron_moments,
    alignment_coverage,
    assert_neuron_ranks_provenance,
    capture_groups,
    down_column_sq_norms,
    frozen_mlp_down_weight,
    make_moments_step,
    mlp_blocks_of,
    neuron_alignment_from_ranks,
    pool_slices,
    pool_tokens_sha256,
    rank_neurons,
    read_neuron_ranks,
    write_energy,
    write_neuron_ranks,
)
from param_decomp.targets.testing import capture_clean, tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import mlp_hidden_tap_key


def _model(site_cs: tuple[SiteC, ...]):
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, canonical_site_cs(site_cs))
    return cfg, sites, tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))


def _pool(n_prompts: int, n_positions: int, vocab: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, vocab, size=(n_prompts, n_positions), dtype=np.int32)


def test_mlp_blocks_of_partial_blocks():
    """Attention-only blocks are absent; writer-only, reader-only and full blocks each get
    their sides."""
    _cfg, sites, _model_ = _model(
        (
            SiteC(site_name(0, "q"), 2),
            SiteC(site_name(1, "gate"), 2),
            SiteC(site_name(1, "up"), 2),
            SiteC(site_name(2, "down"), 2),
            SiteC(site_name(3, "gate"), 2),
            SiteC(site_name(3, "up"), 2),
            SiteC(site_name(3, "down"), 3),
        )
    )
    blocks = mlp_blocks_of(sites, GLU_ANATOMY)
    assert list(blocks) == [1, 2, 3]
    assert (
        blocks[1].writers == (site_name(1, "gate"), site_name(1, "up")) and blocks[1].reader is None
    )
    assert blocks[2].writers == () and blocks[2].reader == site_name(2, "down")
    assert blocks[3].names == (site_name(3, "gate"), site_name(3, "up"), site_name(3, "down"))


def test_frozen_down_weight_is_read_for_undecomposed_blocks_too():
    cfg, _sites, model = _model((SiteC(site_name(3, "gate"), 2),))
    for block in (0, 3, cfg.n_layer - 1):  # prefix, span, tail
        Wd = frozen_mlp_down_weight(model, block)
        assert Wd.shape == (cfg.n_embd, cfg.n_intermediate)
        assert np.allclose(down_column_sq_norms(model, block), np.sum(np.asarray(Wd) ** 2, axis=0))


def test_pool_slices_visit_every_row_exactly_once():
    tokens = np.arange(23 * 3, dtype=np.int32).reshape(23, 3)  # every row unique
    seen: list[np.ndarray] = []
    for rows, mask in pool_slices(tokens, 5):
        assert rows.shape == (5, 3) and mask.shape == (5,)
        seen.append(rows[mask])
    assert np.array_equal(np.concatenate(seen), tokens)
    n_slices = sum(1 for _ in pool_slices(tokens, 5))
    assert n_slices == 5  # 23 = 4 full + 1 padded


@pytest.mark.parametrize("bos", ["exclude", "include"])
def test_moments_match_a_hand_computation_from_the_tap(bos: BosPolicy):
    cfg, _sites, model = _model(
        (
            SiteC(site_name(2, "gate"), 2),
            SiteC(site_name(2, "up"), 2),
            SiteC(site_name(4, "down"), 2),
        )
    )
    blocks = (2, 4)
    tokens = _pool(7, 4, cfg.vocab_size)
    step = make_moments_step(PlacedModel(model=model, placement=None), blocks, bos)
    slices = ((jnp.asarray(rows), jnp.asarray(mask)) for rows, mask in pool_slices(tokens, 3))
    totals = accumulate_neuron_moments(step, blocks, cfg.n_intermediate, slices)

    first = 1 if bos == "exclude" else 0
    captures = capture_clean(model, jnp.asarray(tokens), [mlp_hidden_tap_key(b) for b in blocks])
    for block in blocks:
        h = np.asarray(captures[mlp_hidden_tap_key(block)], dtype=np.float64)[:, first:, :]
        assert totals[block].n_tokens == h.shape[0] * h.shape[1]
        assert np.allclose(totals[block].sum_h, h.sum(axis=(0, 1)), rtol=1e-4, atol=1e-5)
        assert np.allclose(totals[block].sum_h2, (h * h).sum(axis=(0, 1)), rtol=1e-4, atol=1e-5)
        score = write_energy(totals[block], down_column_sq_norms(model, block))
        expected = (h * h).mean(axis=(0, 1)) * np.sum(
            np.asarray(frozen_mlp_down_weight(model, block)) ** 2, axis=0
        )
        assert np.allclose(score, expected, rtol=1e-4, atol=1e-6)


def test_rank_neurons_is_descending_with_index_tie_break():
    score = np.array([0.5, 2.0, 0.5, 2.0, 1.0])
    assert rank_neurons(score).tolist() == [1, 3, 4, 0, 2]


def test_capture_groups_respect_the_byte_budget():
    blocks = tuple(range(7))
    groups = capture_groups(
        blocks, batch=4, n_positions=5, n_neurons=10, budget_bytes=4 * 5 * 10 * 4 * 3
    )
    assert groups == ((0, 1, 2), (3, 4, 5), (6,))
    assert capture_groups(blocks, 4, 5, 10, budget_bytes=1) == tuple((b,) for b in blocks)


def _artifact(tmp_path: Path, n: int, layers: list[int], tokens: np.ndarray) -> Path:
    rng = np.random.default_rng(1)
    rank = {b: rng.permutation(n).astype(np.int32) for b in layers}
    score = {b: np.sort(rng.random(n).astype(np.float32))[::-1] for b in layers}
    meta = NeuronRanksMeta(
        target="tiny",
        tokens_sha256=pool_tokens_sha256(tokens),
        n_prompts=tokens.shape[0],
        prompt_len=tokens.shape[1],
        statistic="write_energy",
        bos="exclude",
        layers=layers,
        n_neurons=n,
        positions_counted=tokens.shape[1] - 1,
    )
    out = tmp_path / "ranks"
    write_neuron_ranks(out, meta, rank, score)
    return out


def test_artifact_round_trip_and_provenance_refusals(tmp_path: Path):
    tokens = _pool(5, 3, 50)
    out = _artifact(tmp_path, n=16, layers=[1, 3], tokens=tokens)
    ranks = read_neuron_ranks(out)
    assert ranks.meta.layers == [1, 3] and set(ranks.rank) == {1, 3}
    assert np.isclose(ranks.coverage(1, 16), 1.0) and 0 < ranks.coverage(1, 4) < 1
    assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens, (1, 3))
    with pytest.raises(AssertionError, match="harvested on"):
        assert_neuron_ranks_provenance(ranks.meta, "other-model", tokens, (1,))
    with pytest.raises(AssertionError, match="different prompt pool"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens[:-1], (1,))
    with pytest.raises(AssertionError, match="different prompt pool"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens + 1, (1,))
    with pytest.raises(AssertionError, match="decomposes"):
        assert_neuron_ranks_provenance(ranks.meta, "tiny", tokens, (1, 2))


def test_alignment_from_ranks_takes_prefixes_per_site_axis():
    cfg, sites, _model_ = _model(
        (
            SiteC(site_name(1, "gate"), 4),
            SiteC(site_name(1, "up"), 4),
            SiteC(site_name(1, "down"), 6),
            SiteC(site_name(2, "q"), 3),
            SiteC(site_name(3, "down"), 5),
        )
    )
    n = cfg.n_intermediate
    rank = {1: np.arange(n)[::-1].astype(np.int32), 3: np.arange(n, dtype=np.int32)}
    score = {b: np.linspace(1.0, 0.0, n, dtype=np.float32) for b in rank}
    meta = NeuronRanksMeta(
        target="tiny", tokens_sha256="0" * 64, n_prompts=1, prompt_len=2, statistic="write_energy",
        bos="exclude", layers=[1, 3], n_neurons=n, positions_counted=1,
    )  # fmt: skip
    alignment = neuron_alignment_from_ranks(NeuronRanks(meta, rank, score), sites, GLU_ANATOMY)
    assert set(alignment) == {
        site_name(1, "gate"),
        site_name(1, "up"),
        site_name(1, "down"),
        site_name(3, "down"),
    }
    gate, down = alignment[site_name(1, "gate")], alignment[site_name(1, "down")]
    assert gate.neuron_axis == "d_out" and down.neuron_axis == "d_in"
    assert np.array_equal(np.asarray(gate.neurons), rank[1][:4])
    assert np.array_equal(np.asarray(down.neurons), rank[1][:6])
    assert np.array_equal(np.asarray(alignment[site_name(3, "down")].neurons), rank[3][:5])
    coverage = alignment_coverage(NeuronRanks(meta, rank, score), sites, GLU_ANATOMY)
    assert coverage[site_name(1, "down")] > coverage[site_name(1, "gate")]
