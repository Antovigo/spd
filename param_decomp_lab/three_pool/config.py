"""Serializable, normalized topology for 3-pool training.

``ThreePoolTopology`` authors only the genuine degrees of freedom — the per-rank
batch for each pool, plus how the decomposed sites split into chunks — and DERIVES
every rank integer via a canonical assignment. Authoring rank ids directly made a
whole class of invalid states representable (overlap, dup, gaps, sum≠world,
non-uniform DP, the rank-0 convention); deriving them makes those unrepresentable.

The three pools:

  * **CI pool** — replicated CI fn, DP across batch. ``n_ci = batch / ci.per_rank_batch``.
  * **Chunkwise pool** — V/U sharded into chunks of sites; each chunk replicated across
    ``chunk_dp = batch / chunkwise.per_rank_batch`` ranks (DDP within chunk).
  * **PPGD pool** — stateless full V/U replica, DP across batch.
    ``n_ppgd = batch / ppgd.per_rank_batch``.

Everything else (batch size, loss coefficients, optimizer LRs, PPGD config,
faithfulness warmup, LR schedules, autocast, ci_config, sigmoid_type,
decomposition_targets) lives on the regular ``PDConfig`` / ``RuntimeConfig`` the
3-pool path consumes.

Validation splits by what it needs:

  * **Parse-time** (no batch, no model): cross-divisibility of the three per-rank
    batches (see ``ThreePoolTopology.validate_cross_divisibility``).
  * **batch-dependent** (needs ``pd.batch_size``): each per-rank batch divides the
    global batch — on ``ThreePoolLMExperimentConfig`` (``three_pool_run.py``).
  * **expansion-dependent** (needs the loaded model): every resolved chunk site is a
    decomposed site — in ``optimize._build_runtime``.

``resolve(ordered_sites, batch_size)`` returns the pure ``ResolvedLayout``;
``optimize.py`` builds the runtime ``Chunk`` objects from it. ``config.py`` does NOT
import ``layout.py`` (avoids a cycle) — it returns plain data.
"""

from dataclasses import dataclass
from typing import Self

from pydantic import PositiveInt, model_validator

from param_decomp.base_config import BaseConfig


class PoolSpec(BaseConfig):
    per_rank_batch: PositiveInt


class ChunkwiseSpec(BaseConfig):
    per_rank_batch: PositiveInt
    sites_per_chunk: PositiveInt | None = None
    """How many decomposed sites each chunk owns. ``None`` puts all sites in one chunk."""


@dataclass(frozen=True)
class ResolvedLayout:
    """Pure canonical rank assignment derived from a ``ThreePoolTopology`` + the
    expanded site list + the global batch. ``chunks`` is per-chunk ``(ranks, sites)``.

    Canonical order: chunks first (rank 0 = chunk-0 leader), then CI, then PPGD.
    """

    ci_ranks: tuple[int, ...]
    ppgd_ranks: tuple[int, ...]
    chunks: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]
    world_size: int


class ThreePoolTopology(BaseConfig):
    """Normalized 3-pool topology. Pairs with a regular ``PDConfig``.

    Authors per-rank batch (the memory-meaningful number) and the site→chunk split;
    every rank id is derived by ``resolve`` in canonical order. The cross-divisibility
    constraint (CI↔chunk, CI↔PPGD per-rank batches each cross-divide) keeps every
    cross-pool batch overlap a whole, aligned sub-slice.
    """

    ci: PoolSpec
    ppgd: PoolSpec
    chunkwise: ChunkwiseSpec
    use_fused_kl: bool = True

    @model_validator(mode="after")
    def validate_cross_divisibility(self) -> Self:
        bl_ci = self.ci.per_rank_batch
        bl_ppgd = self.ppgd.per_rank_batch
        bl_chunk = self.chunkwise.per_rank_batch
        # n_ci / chunk_dp / n_ppgd are batch/bl; the n's cross-divide iff the bl's do.
        # Cross-divisible per-rank batches ⇒ every CI↔chunk and CI↔PPGD batch overlap is
        # a whole, aligned sub-slice (one-to-K fan-out in either direction). Ragged pairs
        # where neither divides the other are rejected.
        assert bl_chunk % bl_ci == 0 or bl_ci % bl_chunk == 0, (
            f"ci.per_rank_batch ({bl_ci}) and chunkwise.per_rank_batch ({bl_chunk}) must "
            f"cross-divide (one divides the other) so each CI↔chunk batch overlap is a whole "
            f"sub-slice. Tip: make one a multiple of the other."
        )
        assert bl_ppgd % bl_ci == 0 or bl_ci % bl_ppgd == 0, (
            f"ci.per_rank_batch ({bl_ci}) and ppgd.per_rank_batch ({bl_ppgd}) must "
            f"cross-divide (one divides the other) so each CI↔PPGD batch overlap is a whole "
            f"sub-slice. Tip: make one a multiple of the other."
        )
        return self

    def resolve(self, ordered_sites: list[str], batch_size: int) -> ResolvedLayout:
        """Derive the canonical rank assignment for ``ordered_sites`` + ``batch_size``.

        Canonical order: chunk-0 ranks (rank 0 = chunk-0 leader), …, chunk-N ranks,
        CI ranks, PPGD ranks. Each per-rank batch must divide ``batch_size`` (asserted
        upstream on the experiment config; re-asserted here for the standalone call).
        """
        assert batch_size % self.ci.per_rank_batch == 0
        assert batch_size % self.ppgd.per_rank_batch == 0
        assert batch_size % self.chunkwise.per_rank_batch == 0
        assert ordered_sites, "resolve needs at least one decomposed site"

        n_ci = batch_size // self.ci.per_rank_batch
        n_ppgd = batch_size // self.ppgd.per_rank_batch
        chunk_dp = batch_size // self.chunkwise.per_rank_batch
        spc = self.chunkwise.sites_per_chunk or len(ordered_sites)
        site_chunks = [ordered_sites[i : i + spc] for i in range(0, len(ordered_sites), spc)]

        r = 0
        chunks: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        for sites in site_chunks:
            chunks.append((tuple(range(r, r + chunk_dp)), tuple(sites)))
            r += chunk_dp
        ci_ranks = tuple(range(r, r + n_ci))
        r += n_ci
        ppgd_ranks = tuple(range(r, r + n_ppgd))
        r += n_ppgd
        return ResolvedLayout(
            ci_ranks=ci_ranks,
            ppgd_ranks=ppgd_ranks,
            chunks=tuple(chunks),
            world_size=r,
        )
