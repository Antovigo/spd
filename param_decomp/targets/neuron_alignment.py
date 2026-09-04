"""Neuron-aligned targeted init support for the transformer targets (SPEC T13).

`initialization: neuron_aligned_targeted` starts every subcomponent of every decomposed
site as ONE architectural coordinate of its matrix — the top-C by activity on the target
prompt pool. Which coordinate a site's components align to is the matrix's structural
role (`unit_kind_of`), and each role has its own ranking per block:

    mlp (gate/up | fc, down):  score_i = E[h_i²] · ‖W_down[:, i]‖²   write energy of neuron i
    q:                         score_j = E[q_j²]                      the projection's own energy
    k:                         score_j = E[k_j²]
    o:                         score_j = E[z_j²] · ‖W_o[:, j]‖²      write energy of attention channel j
    v:                         score_(g,d) = Σ_{h ∈ group g} score_o[(h, d)]

with `h` the post-nonlinearity hidden activation the down projection consumes (`mlp_hidden`
tap), `q`/`k` the projections' raw outputs (their `.out` taps, pre-RoPE), and `z` the
attention-core output the o projection consumes (`attention_output` tap). Attention mixes
across positions, never across channels, so `q_j` and `k_j` interact only inside their
head's dot product and rank independently; `v` channel `(g, d)` IS attention-output channel
`(h, d)` for every query head `h` of kv group `g`, so v and o share the write-energy ranking
(v's summed over its group). Expectations run over every position of every pool prompt
exactly once (an exhaustive sweep, not the trainer's with-replacement sampler). Uncentred
on purpose: a coordinate that fires the same on every prompt of a narrow pool is exactly
one the decomposition must reconstruct. One ranking per (block, kind) serves every site of
that kind, so a site's top-C is a PREFIX of one ranking.

The rankings depend on the target model and the pool only — never on the decomposition —
so they are harvested once (`experiments.lm.harvest_neuron_ranks`) into an artifact this
module reads and writes, and a run's provenance is checked against it at load. The init
itself is a `ComponentInitializer` (`neuron_aligned_targeted_component_initializer`), the
same seam the data-free `neuron_aligned` init uses.
"""

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray
from pydantic import Field

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.components import (
    ComponentStacks,
    SiteSpec,
    component_stacks_from_site_arrays,
)
from param_decomp.core.configs import NamedNeuronRanks, NeuronRanksDir, NeuronRanksRef
from param_decomp.core.init_placed import ComponentInitializer
from param_decomp.core.model import DecomposedModel, PlacedModel
from param_decomp.targets.glu_transformer import (
    Anatomy,
    GatedMLP,
    GLUDecomposedModel,
    PlainMLP,
    _frozen_site_weight,
    neuron_aligned_component_count,
    selected_unit_factors,
)
from param_decomp.targets.transformer_taps import (
    attention_output_tap_key,
    mlp_hidden_tap_key,
    site_output_tap_key,
)

STATISTIC = "unit_energy"
NEURON_RANKS_FILENAME = "neuron_ranks.npz"
NEURON_RANKS_META_FILENAME = "meta.json"

UnitKind = Literal["mlp", "q", "k", "v", "o"]
"""The ranking families: one per structural role a decomposed matrix can have."""
UNIT_KINDS: tuple[UnitKind, ...] = get_args(UnitKind)

NeuronAlignment = dict[str, np.ndarray]
"""Site name -> `int32[C]` distinct unit indices in rank order (slot `i` = the `i`-th
ranked coordinate of the site's kind), for EVERY decomposed site."""


# ----------------------------- sites ↔ rankings -----------------------------


def unit_kind_of(anatomy: Anatomy, kind: str) -> UnitKind:
    """Which ranking a matrix kind aligns to."""
    if kind in anatomy.mlp.hidden or kind == anatomy.mlp.down:
        return "mlp"
    if kind == anatomy.q:
        return "q"
    if kind == anatomy.k:
        return "k"
    if kind == anatomy.v:
        return "v"
    if kind == anatomy.o:
        return "o"
    raise AssertionError(f"{kind!r} is not a matrix kind of {anatomy.family}")


def site_unit_kinds(
    sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> dict[str, tuple[int, UnitKind]]:
    """Site name -> (block, ranking family)."""
    out: dict[str, tuple[int, UnitKind]] = {}
    for spec in sites:
        block, kind = anatomy.family.parse(spec.name)
        out[spec.name] = (block, unit_kind_of(anatomy, kind))
    return out


def ranked_blocks(sites: tuple[SiteSpec, ...], anatomy: Anatomy) -> tuple[int, ...]:
    """The blocks holding at least one decomposed site, ascending."""
    return tuple(sorted({block for block, _ in site_unit_kinds(sites, anatomy).values()}))


def frozen_mlp_down_weight(model: GLUDecomposedModel, block: int) -> Float[Array, "d n"]:
    """The block's frozen down projection `[d_model, n_neurons]`, decomposed or not — the
    target-owned read the score needs for every ranked block."""
    match model.frozen_block(block).mlp:
        case GatedMLP(Wd=Wd):
            return Wd
        case PlainMLP(Wdown=Wd):
            return Wd


def _column_sq_norms(weight: Array) -> np.ndarray:
    """`‖W[:, j]‖²` per column, fp32."""
    w = weight.astype(jnp.float32)
    return np.asarray(jnp.sum(w * w, axis=0), dtype=np.float32)


def unit_counts(model: GLUDecomposedModel) -> dict[UnitKind, int]:
    """Coordinates per ranking family — the same for every block of one architecture."""
    attn = model.frozen_block(0).attn
    return {
        "mlp": int(frozen_mlp_down_weight(model, 0).shape[1]),
        "q": attn.n_head * attn.head_dim,
        "k": attn.n_kv_head * attn.head_dim,
        "v": attn.n_kv_head * attn.head_dim,
        "o": attn.n_head * attn.head_dim,
    }


# ----------------------------- the sweep -----------------------------


def pool_slices(tokens: np.ndarray, batch: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Row-contiguous `[batch, T]` slices of the whole pool, each with its row mask; only
    the last slice is padded (by repeating its last row, masked out). Every pool row is
    yielded exactly once — the exhaustive sweep, deliberately NOT `pool_batch`, which
    samples with replacement."""
    n_rows = tokens.shape[0]
    assert batch >= 1 and n_rows >= 1, (batch, n_rows)
    for start in range(0, n_rows, batch):
        rows = tokens[start : start + batch]
        kept = rows.shape[0]
        mask = np.zeros((batch,), dtype=bool)
        mask[:kept] = True
        if kept < batch:
            rows = np.concatenate([rows, np.repeat(rows[-1:], batch - kept, axis=0)], axis=0)
        yield rows, mask


_MOMENT_TAPS = ("mlp", "q", "k", "z")
"""The captured activations one block's rankings need: the MLP hidden, the q and k
projection outputs, and the attention-core output `z` (which serves BOTH v and o)."""


class BlockMoments(eqx.Module):
    """Per-coordinate `Σ x²` for each of one block's captured taps, and the token count
    they sum over — fp32 on device."""

    sum_sq: dict[str, Float[Array, " n"]]
    n_tokens: Float[Array, ""]


MomentsStep = Callable[[Array, Array], dict[int, BlockMoments]]


def block_tap_keys(anatomy: Anatomy, block: int) -> dict[str, str]:
    """`_MOMENT_TAPS` name -> capture key for one block."""
    return {
        "mlp": mlp_hidden_tap_key(block),
        "q": site_output_tap_key(anatomy.family.name_of(block, anatomy.q)),
        "k": site_output_tap_key(anatomy.family.name_of(block, anatomy.k)),
        "z": attention_output_tap_key(block),
    }


def make_moments_step(model: PlacedModel, blocks: tuple[int, ...]) -> MomentsStep:
    """One jitted `(tokens [B, T], row_mask [B]) -> {block: moments}` over the frozen
    forward with the blocks' taps captured, every position counted. The outputs are
    declared FULLY REPLICATED, so the batch reduction's collective happens inside the
    graph and every device (and process) holds the complete vectors."""
    assert len(set(blocks)) == len(blocks) and blocks, blocks
    glu = model.model
    assert isinstance(glu, GLUDecomposedModel), type(glu)
    keys_by_block = {block: block_tap_keys(glu.anatomy, block) for block in blocks}
    keys = frozenset(key for taps in keys_by_block.values() for key in taps.values())
    model_arrays, model_static = eqx.partition(model, eqx.is_array)

    def step(arrays: PlacedModel, tokens: Array, row_mask: Array) -> dict[int, BlockMoments]:
        placed: PlacedModel = eqx.combine(arrays, model_static)
        captures = placed.clean_forward(tokens, keys).captures
        weight = jnp.broadcast_to(row_mask[:, None], tokens.shape).astype(jnp.float32)
        moments: dict[int, BlockMoments] = {}
        for block in blocks:
            sum_sq: dict[str, Array] = {}
            for tap, key in keys_by_block[block].items():
                x = captures[key].astype(jnp.float32)
                sum_sq[tap] = jnp.einsum("bt,btn->n", weight, x * x)
            moments[block] = BlockMoments(sum_sq=sum_sq, n_tokens=jnp.sum(weight))
        return moments

    if model.placement is None:
        jitted = jax.jit(step)
    else:
        replicated = NamedSharding(model.placement.mesh, P())
        jitted = jax.jit(step, out_shardings=replicated)
    return lambda tokens, row_mask: jitted(model_arrays, tokens, row_mask)


@dataclass
class HostMoments:
    """The float64 host accumulators one block's sweep lands in."""

    sum_sq: dict[str, np.ndarray]
    n_tokens: float

    @classmethod
    def zeros(cls, widths: dict[str, int]) -> "HostMoments":
        return cls({tap: np.zeros(n, np.float64) for tap, n in widths.items()}, 0.0)

    def add(self, device: BlockMoments) -> None:
        assert set(device.sum_sq) == set(self.sum_sq), (set(device.sum_sq), set(self.sum_sq))
        for tap, value in device.sum_sq.items():
            self.sum_sq[tap] += np.asarray(value, dtype=np.float64)
        self.n_tokens += float(device.n_tokens)


def tap_widths(counts: dict[UnitKind, int]) -> dict[str, int]:
    """Captured width per `_MOMENT_TAPS` entry, from the ranking-family unit counts."""
    return {"mlp": counts["mlp"], "q": counts["q"], "k": counts["k"], "z": counts["o"]}


def accumulate_neuron_moments(
    step: MomentsStep,
    blocks: tuple[int, ...],
    widths: dict[str, int],
    slices: Iterator[tuple[Array, Array]],
) -> dict[int, HostMoments]:
    """Drive one `make_moments_step` over already-placed `(tokens, row_mask)` slices."""
    totals = {block: HostMoments.zeros(widths) for block in blocks}
    for tokens, row_mask in slices:
        for block, moments in step(tokens, row_mask).items():
            totals[block].add(moments)
    return totals


def capture_groups(
    blocks: tuple[int, ...], batch: int, n_positions: int, captured_width: int, budget_bytes: int
) -> tuple[tuple[int, ...], ...]:
    """Blocks to capture per forward: the forward retains one fp32 `[B, T, width]` slot per
    REQUESTED block (`captured_width` = the sum of its tap widths), so a many-block harvest
    is cut into groups under a byte budget."""
    per_block = batch * n_positions * captured_width * 4
    per_group = max(1, budget_bytes // per_block)
    return tuple(blocks[i : i + per_group] for i in range(0, len(blocks), per_group))


# ----------------------------- moments → scores → rankings -----------------------------


def block_scores(
    model: GLUDecomposedModel, block: int, moments: HostMoments
) -> dict[UnitKind, np.ndarray]:
    """Every ranking family's per-coordinate score for one block, float64 (module docstring)."""
    assert moments.n_tokens > 0, "the sweep counted no tokens"
    mean_sq = {tap: value / moments.n_tokens for tap, value in moments.sum_sq.items()}
    attn = model.frozen_block(block).attn
    o_score = mean_sq["z"] * _column_sq_norms(attn.wo).astype(np.float64)
    n_rep = attn.n_head // attn.n_kv_head
    v_score = o_score.reshape(attn.n_kv_head, n_rep, attn.head_dim).sum(axis=1).reshape(-1)
    return {
        "mlp": mean_sq["mlp"]
        * _column_sq_norms(frozen_mlp_down_weight(model, block)).astype(np.float64),
        "q": mean_sq["q"],
        "k": mean_sq["k"],
        "v": v_score,
        "o": o_score,
    }


def rank_units(score: np.ndarray) -> np.ndarray:
    """Coordinate indices by DESCENDING score, ties broken by ascending index —
    deterministic on every host."""
    assert score.ndim == 1 and np.all(np.isfinite(score)), score.shape
    return np.lexsort((np.arange(score.shape[0]), -score)).astype(np.int32)


# ----------------------------- the artifact -----------------------------


class NeuronRanksMeta(BaseConfig):
    """The artifact's provenance — what a run is checked against before it may start from
    the rankings. `tokens_sha256` fingerprints the pool's token matrix itself, so two pool
    specs that tokenize identically share an artifact and any drift refuses."""

    target: str = Field(min_length=1)
    tokens_sha256: str = Field(min_length=64, max_length=64)
    n_prompts: int
    prompt_len: int
    statistic: str
    layers: list[int]
    n_units: dict[UnitKind, int]


@dataclass(frozen=True)
class NeuronRanks:
    meta: NeuronRanksMeta
    rank: dict[UnitKind, dict[int, np.ndarray]]
    """kind -> block -> `int32[n]` coordinate indices by descending score."""
    score: dict[UnitKind, dict[int, np.ndarray]]
    """kind -> block -> `float32[n]` scores in rank order."""

    def coverage(self, kind: UnitKind, block: int, C: int) -> float:
        """The fraction of the (block, kind) total score its top-C coordinates carry."""
        score = self.score[kind][block].astype(np.float64)
        total = float(score.sum())
        return float(score[:C].sum() / total) if total > 0 else 0.0


def pool_tokens_sha256(tokens: np.ndarray) -> str:
    """Fingerprint of the pool: shape + row-major int32 bytes."""
    tokens = np.ascontiguousarray(tokens, dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(np.asarray(tokens.shape, dtype=np.int64).tobytes())
    digest.update(tokens.tobytes())
    return digest.hexdigest()


def neuron_ranks_dir(data_root: Path, name: str) -> Path:
    """The store layout: a named artifact lives at `<data_root>/neuron_ranks/<name>`."""
    return data_root / "neuron_ranks" / name


def resolve_neuron_ranks_ref(ref: NeuronRanksRef, data_root: Path) -> Path:
    match ref:
        case NamedNeuronRanks(name=name):
            return neuron_ranks_dir(data_root, name)
        case NeuronRanksDir(dir=dir):
            return dir


def _array_key(prefix: str, kind: UnitKind, block: int) -> str:
    return f"{prefix}_{kind}_{block}"


def write_neuron_ranks(
    out_dir: Path,
    meta: NeuronRanksMeta,
    rank: dict[UnitKind, dict[int, np.ndarray]],
    score: dict[UnitKind, dict[int, np.ndarray]],
) -> None:
    assert set(rank) == set(score) == set(UNIT_KINDS), (set(rank), set(score))
    assert set(meta.n_units) == set(UNIT_KINDS), meta.n_units
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for kind in UNIT_KINDS:
        assert set(rank[kind]) == set(score[kind]) == set(meta.layers), (kind, meta.layers)
        for block in meta.layers:
            n = meta.n_units[kind]
            assert rank[kind][block].shape == score[kind][block].shape == (n,), (kind, block)
            arrays[_array_key("rank", kind, block)] = rank[kind][block].astype(np.int32)
            arrays[_array_key("score", kind, block)] = score[kind][block].astype(np.float32)
    np.savez(out_dir / NEURON_RANKS_FILENAME, **arrays)  # pyright: ignore[reportArgumentType]
    (out_dir / NEURON_RANKS_META_FILENAME).write_text(meta.model_dump_json(indent=2) + "\n")


def read_neuron_ranks(artifact_dir: Path) -> NeuronRanks:
    meta_path = artifact_dir / NEURON_RANKS_META_FILENAME
    assert meta_path.exists(), f"no neuron-ranks artifact at {artifact_dir}: {meta_path} missing"
    meta = NeuronRanksMeta.model_validate(json.loads(meta_path.read_text()))
    assert meta.statistic == STATISTIC, (
        f"neuron ranks at {artifact_dir} carry statistic {meta.statistic!r}; this code reads "
        f"{STATISTIC!r} (every ranking family, attention included) — re-harvest"
    )
    rank: dict[UnitKind, dict[int, np.ndarray]] = {kind: {} for kind in UNIT_KINDS}
    score: dict[UnitKind, dict[int, np.ndarray]] = {kind: {} for kind in UNIT_KINDS}
    with np.load(artifact_dir / NEURON_RANKS_FILENAME) as arrays:
        for kind in UNIT_KINDS:
            n = meta.n_units[kind]
            for block in meta.layers:
                r = np.asarray(arrays[_array_key("rank", kind, block)])
                s = np.asarray(arrays[_array_key("score", kind, block)])
                assert r.shape == s.shape == (n,), (kind, block, r.shape, s.shape, n)
                # A permutation of 0..n-1, checked rather than trusted: a negative index
                # would silently fancy-index from the end.
                assert np.array_equal(np.sort(r), np.arange(n)), (
                    f"{kind} block {block}: rank is not a permutation of 0..{n - 1}"
                )
                rank[kind][block], score[kind][block] = r, s
    return NeuronRanks(meta, rank, score)


def assert_neuron_ranks_provenance(
    meta: NeuronRanksMeta, target: str, tokens: np.ndarray, blocks: tuple[int, ...]
) -> None:
    """Fail closed on any drift between the artifact's world and the run's: same target
    model, byte-identical prompt pool, every block the run decomposes ranked."""
    assert meta.target == target, (
        f"neuron ranks were harvested on {meta.target!r}, run targets {target!r}"
    )
    assert meta.statistic == STATISTIC, meta.statistic
    digest = pool_tokens_sha256(tokens)
    assert meta.tokens_sha256 == digest, (
        "neuron ranks were harvested on a different prompt pool: "
        f"artifact {meta.tokens_sha256[:12]}… ({meta.n_prompts} x {meta.prompt_len}), "
        f"run {digest[:12]}… ({tokens.shape[0]} x {tokens.shape[1]})"
    )
    missing = sorted(set(blocks) - set(meta.layers))
    assert not missing, f"neuron ranks cover layers {meta.layers}; the run decomposes {missing} too"


# ----------------------------- rankings → alignment → init -----------------------------


def neuron_alignment_from_ranks(
    ranks: NeuronRanks, sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> NeuronAlignment:
    """Every decomposed site's top-C of its (block, kind) ranking: slot `i` is the `i`-th
    ranked coordinate, so two sites of one kind in one block take nested prefixes."""
    alignment: NeuronAlignment = {}
    for spec in sites:
        block, kind = anatomy.family.parse(spec.name)
        rank = ranks.rank[unit_kind_of(anatomy, kind)][block]
        n = neuron_aligned_component_count(anatomy, spec)
        assert n == rank.shape[0], (spec.name, n, rank.shape)
        assert n >= spec.C, (
            f"{spec.name}: neuron_aligned_targeted needs C <= {n} distinct coordinates, got C={spec.C}"
        )
        alignment[spec.name] = rank[: spec.C].astype(np.int32)
    return alignment


def alignment_coverage(
    ranks: NeuronRanks, sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> dict[str, float]:
    """Per site, the score fraction its C covers — the step-0 number that says whether C
    is enough for the pool."""
    return {
        spec.name: ranks.coverage(kind, block, spec.C)
        for spec in sites
        for block, kind in (site_unit_kinds((spec,), anatomy)[spec.name],)
    }


def validate_neuron_alignment(
    sites: tuple[SiteSpec, ...], anatomy: Anatomy, alignment: NeuronAlignment
) -> None:
    """Host-side checks the traced init cannot make: every decomposed site is aligned, with
    exactly `C` DISTINCT indices inside its unit axis."""
    assert set(alignment) == {spec.name for spec in sites}, sorted(
        set(alignment) ^ {spec.name for spec in sites}
    )
    for spec in sites:
        units = np.asarray(alignment[spec.name])
        n = neuron_aligned_component_count(anatomy, spec)
        assert units.shape == (spec.C,) and units.dtype.kind == "i", (
            spec.name,
            units.shape,
            spec.C,
        )
        assert len(set(units.tolist())) == spec.C, f"{spec.name}: repeated unit indices"
        assert units.min() >= 0 and units.max() < n, (spec.name, units.min(), units.max(), n)


def neuron_aligned_targeted_component_initializer(
    alignment: NeuronAlignment,
) -> ComponentInitializer:
    """The `neuron_aligned_targeted` init (SPEC T13) as a `ComponentInitializer`: every
    site's subcomponents ARE its aligned coordinates (`selected_unit_factors`), read from
    the frozen weights inside the init graph. Consumes no randomness."""

    def initialize(model: DecomposedModel, key: PRNGKeyArray) -> ComponentStacks:
        del key
        assert isinstance(model, GLUDecomposedModel), (
            f"neuron_aligned_targeted needs a transformer target, got {type(model)}"
        )
        validate_neuron_alignment(model.sites, model.anatomy, alignment)
        site_arrays: dict[str, tuple[Array, Array]] = {}
        for spec in model.sites:
            layer, kind = model.anatomy.family.parse(spec.name)
            weight = _frozen_site_weight(model.anatomy, model.frozen_block(layer), kind)
            site_arrays[spec.name] = selected_unit_factors(
                weight,
                spec,
                kind in model.anatomy.row_kinds,
                jnp.asarray(alignment[spec.name], dtype=jnp.int32),
            )
        return component_stacks_from_site_arrays(model.sites, site_arrays)

    return initialize
