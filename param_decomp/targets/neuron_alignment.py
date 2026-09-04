"""Neuron-aligned init support for the transformer targets (SPEC T13).

The `neuron_aligned_targeted` init starts every MLP subcomponent as one neuron of its
block — the top-C by **write energy** on the target prompt pool:

    score_i = E_tokens[ h_i² ] · ‖W_down[:, i]‖²

with `h` the post-nonlinearity hidden activation the block's down projection consumes
(`mlp_hidden` tap: `silu(gate)·up` on the gated anatomy, the post-GELU `fc` output on the
plain one) and the expectation over every position of every pool prompt exactly once (an
exhaustive sweep, not the trainer's with-replacement sampler). Uncentred on purpose: a
neuron that fires the same on every prompt of a narrow pool is exactly one the
decomposition must reconstruct. The down column norm turns hidden-unit scale into
residual-stream effect, and one block score serves the block's writers and reader alike, so
a site's top-C is a PREFIX of one ranking.

The ranking depends on the target model and the pool only — never on the decomposition —
so it is harvested once (`experiments.lm.harvest_neuron_ranks`) into an artifact this
module reads and writes, and a run's provenance is checked against it at load.

Core stays anatomy-blind (`core.components.SiteNeuronAlignment` carries the neuron axis
and the indices); everything anatomy-aware lives here. Targets import core only.
"""

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float
from pydantic import Field

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.components import NeuronAlignment, SiteNeuronAlignment, SiteSpec
from param_decomp.core.configs import NamedNeuronRanks, NeuronRanksDir, NeuronRanksRef
from param_decomp.core.model import PlacedModel
from param_decomp.targets.glu_transformer import (
    Anatomy,
    GatedMLP,
    GLUDecomposedModel,
    PlainMLP,
)
from param_decomp.targets.transformer_taps import mlp_hidden_tap_key

STATISTIC = "write_energy"
NEURON_RANKS_FILENAME = "neuron_ranks.npz"
NEURON_RANKS_META_FILENAME = "meta.json"


# ----------------------------- sites ↔ blocks -----------------------------


@dataclass(frozen=True)
class MLPBlockSites:
    """One block's decomposed MLP sites: the hidden writers (gate/up | fc) and the reader
    (down). Either side may be absent — a writer-only or reader-only decomposition still
    ranks, since the score reads the frozen down matrix, not the site."""

    block: int
    writers: tuple[str, ...]
    reader: str | None

    @property
    def names(self) -> tuple[str, ...]:
        return self.writers + ((self.reader,) if self.reader is not None else ())


def mlp_blocks_of(sites: tuple[SiteSpec, ...], anatomy: Anatomy) -> dict[int, MLPBlockSites]:
    """The blocks holding at least one decomposed MLP site, by block; attention-only
    blocks have no entry."""
    writers: dict[int, list[str]] = {}
    readers: dict[int, str] = {}
    for spec in sites:
        block, kind = anatomy.family.parse(spec.name)
        if kind in anatomy.mlp.hidden:
            writers.setdefault(block, []).append(spec.name)
        elif kind == anatomy.mlp.down:
            assert block not in readers, (block, readers[block], spec.name)
            readers[block] = spec.name
    return {
        block: MLPBlockSites(block, tuple(writers.get(block, ())), readers.get(block))
        for block in sorted(set(writers) | set(readers))
    }


def frozen_mlp_down_weight(model: GLUDecomposedModel, block: int) -> Float[Array, "d n"]:
    """The block's frozen down projection `[d_model, n_neurons]`, decomposed or not — the
    target-owned read the score needs for every ranked block."""
    match model.frozen_block(block).mlp:
        case GatedMLP(Wd=Wd):
            return Wd
        case PlainMLP(Wdown=Wd):
            return Wd


def down_column_sq_norms(model: GLUDecomposedModel, block: int) -> np.ndarray:
    """`‖W_down[:, i]‖²` per neuron, fp32."""
    Wd = frozen_mlp_down_weight(model, block).astype(jnp.float32)
    return np.asarray(jnp.sum(Wd * Wd, axis=0), dtype=np.float32)


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


class NeuronMoments(eqx.Module):
    """Per-neuron `Σh²` and the token count it sums over, fp32 on device."""

    sum_h2: Float[Array, " n"]
    n_tokens: Float[Array, ""]


MomentsStep = Callable[[Array, Array], dict[int, NeuronMoments]]


def make_moments_step(model: PlacedModel, blocks: tuple[int, ...]) -> MomentsStep:
    """One jitted `(tokens [B, T], row_mask [B]) -> {block: moments}` over the frozen
    forward with the blocks' `mlp_hidden` taps captured, every position counted. The
    outputs are declared FULLY REPLICATED, so the batch reduction's collective happens
    inside the graph and every device (and process) holds the complete `[n]` vectors."""
    assert len(set(blocks)) == len(blocks) and blocks, blocks
    keys = frozenset(mlp_hidden_tap_key(block) for block in blocks)
    model_arrays, model_static = eqx.partition(model, eqx.is_array)

    def step(arrays: PlacedModel, tokens: Array, row_mask: Array) -> dict[int, NeuronMoments]:
        placed: PlacedModel = eqx.combine(arrays, model_static)
        captures = placed.clean_forward(tokens, keys).captures
        weight = jnp.broadcast_to(row_mask[:, None], tokens.shape).astype(jnp.float32)
        moments: dict[int, NeuronMoments] = {}
        for block in blocks:
            h = captures[mlp_hidden_tap_key(block)].astype(jnp.float32)
            moments[block] = NeuronMoments(
                sum_h2=jnp.einsum("bt,btn->n", weight, h * h), n_tokens=jnp.sum(weight)
            )
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

    sum_h2: np.ndarray
    n_tokens: float

    @classmethod
    def zeros(cls, n_neurons: int) -> "HostMoments":
        return cls(np.zeros(n_neurons, np.float64), 0.0)

    def add(self, device: NeuronMoments) -> None:
        self.sum_h2 += np.asarray(device.sum_h2, dtype=np.float64)
        self.n_tokens += float(device.n_tokens)


def accumulate_neuron_moments(
    step: MomentsStep,
    blocks: tuple[int, ...],
    n_neurons: int,
    slices: Iterator[tuple[Array, Array]],
) -> dict[int, HostMoments]:
    """Drive one `make_moments_step` over already-placed `(tokens, row_mask)` slices."""
    totals = {block: HostMoments.zeros(n_neurons) for block in blocks}
    for tokens, row_mask in slices:
        for block, moments in step(tokens, row_mask).items():
            totals[block].add(moments)
    return totals


def capture_groups(
    blocks: tuple[int, ...], batch: int, n_positions: int, n_neurons: int, budget_bytes: int
) -> tuple[tuple[int, ...], ...]:
    """Blocks to capture per forward: the forward retains one fp32 `[B, T, n]` slot per
    REQUESTED block, so a many-block harvest is cut into groups under a byte budget."""
    per_block = batch * n_positions * n_neurons * 4
    per_group = max(1, budget_bytes // per_block)
    return tuple(blocks[i : i + per_group] for i in range(0, len(blocks), per_group))


# ----------------------------- score → ranking -----------------------------


def write_energy(moments: HostMoments, down_sq_norms: np.ndarray) -> np.ndarray:
    """`E[h²] · ‖W_down[:, i]‖²` per neuron, float64."""
    assert moments.n_tokens > 0, "the sweep counted no tokens"
    return moments.sum_h2 / moments.n_tokens * down_sq_norms.astype(np.float64)


def rank_neurons(score: np.ndarray) -> np.ndarray:
    """Neuron indices by DESCENDING score, ties broken by ascending index — deterministic
    on every host."""
    assert score.ndim == 1 and np.all(np.isfinite(score)), score.shape
    return np.lexsort((np.arange(score.shape[0]), -score)).astype(np.int32)


# ----------------------------- the artifact -----------------------------


class NeuronRanksMeta(BaseConfig):
    """The artifact's provenance — what a run is checked against before it may start from
    the ranking. `tokens_sha256` fingerprints the pool's token matrix itself, so two pool
    specs that tokenize identically share an artifact and any drift refuses."""

    target: str = Field(min_length=1)
    tokens_sha256: str = Field(min_length=64, max_length=64)
    n_prompts: int
    prompt_len: int
    statistic: str
    layers: list[int]
    n_neurons: int


@dataclass(frozen=True)
class NeuronRanks:
    meta: NeuronRanksMeta
    rank: dict[int, np.ndarray]
    """block -> `int32[n]` neuron indices by descending score."""
    score: dict[int, np.ndarray]
    """block -> `float32[n]` scores in rank order."""

    def coverage(self, block: int, C: int) -> float:
        """The fraction of the block's total write energy its top-C neurons carry."""
        score = self.score[block].astype(np.float64)
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


def write_neuron_ranks(
    out_dir: Path,
    meta: NeuronRanksMeta,
    rank: dict[int, np.ndarray],
    score: dict[int, np.ndarray],
) -> None:
    assert set(rank) == set(score) == set(meta.layers), (set(rank), set(score), meta.layers)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for block in meta.layers:
        assert rank[block].shape == score[block].shape == (meta.n_neurons,), block
        arrays[f"rank_{block}"] = rank[block].astype(np.int32)
        arrays[f"score_{block}"] = score[block].astype(np.float32)
    np.savez(out_dir / NEURON_RANKS_FILENAME, **arrays)  # pyright: ignore[reportArgumentType]
    (out_dir / NEURON_RANKS_META_FILENAME).write_text(meta.model_dump_json(indent=2) + "\n")


def read_neuron_ranks(artifact_dir: Path) -> NeuronRanks:
    meta_path = artifact_dir / NEURON_RANKS_META_FILENAME
    assert meta_path.exists(), f"no neuron-ranks artifact at {artifact_dir}: {meta_path} missing"
    meta = NeuronRanksMeta.model_validate(json.loads(meta_path.read_text()))
    with np.load(artifact_dir / NEURON_RANKS_FILENAME) as arrays:
        rank = {block: np.asarray(arrays[f"rank_{block}"]) for block in meta.layers}
        score = {block: np.asarray(arrays[f"score_{block}"]) for block in meta.layers}
    for block in meta.layers:
        assert rank[block].shape == score[block].shape == (meta.n_neurons,), block
        # A permutation of 0..n-1, checked rather than trusted: a negative index would
        # silently fancy-index from the end.
        assert np.array_equal(np.sort(rank[block]), np.arange(meta.n_neurons)), (
            f"block {block}: rank is not a permutation of 0..{meta.n_neurons - 1}"
        )
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


# ----------------------------- ranking → alignment -----------------------------


def neuron_alignment_from_ranks(
    ranks: NeuronRanks, sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> NeuronAlignment:
    """Every MLP site's top-C: writers select on `d_out`, the reader on `d_in`; slot `i`
    is the block's `i`-th ranked neuron. Attention sites are absent (they take `zero_u`)."""
    by_name = {spec.name: spec for spec in sites}
    alignment: NeuronAlignment = {}
    for block, block_sites in mlp_blocks_of(sites, anatomy).items():
        rank = ranks.rank[block]
        for name in block_sites.names:
            spec = by_name[name]
            axis: Literal["d_out", "d_in"] = "d_out" if name in block_sites.writers else "d_in"
            n = spec.d_out if axis == "d_out" else spec.d_in
            assert n == rank.shape[0], (name, n, rank.shape)
            assert n >= spec.C, (name, spec.C, n)
            alignment[name] = SiteNeuronAlignment(
                neurons=jnp.asarray(rank[: spec.C], dtype=jnp.int32), neuron_axis=axis
            )
    return alignment


def alignment_coverage(
    ranks: NeuronRanks, sites: tuple[SiteSpec, ...], anatomy: Anatomy
) -> dict[str, float]:
    """Per aligned site, the write-energy fraction its C covers — the step-0 number that
    says whether C is enough for the pool."""
    return {
        name: ranks.coverage(block, spec.C)
        for block, block_sites in mlp_blocks_of(sites, anatomy).items()
        for name in block_sites.names
        for spec in (next(s for s in sites if s.name == name),)
    }
