"""Deterministic batch schedule over local pre-tokenized Parquet shards (SPEC S18).

The schedule is a pure function of `(seed, epoch)`: shard and row order are seeded
permutations, and batches are consecutive row windows. `locate(step)` maps directly
to `(epoch, shard, batch-in-shard)`, so resume needs no replay. Each shard tail
(`rows % global_batch`) is dropped to preserve exact addressability.

Every process computes the same global schedule and serves its contiguous slice of
each batch; the trainer assembles the global device array.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _perm(seed_parts: tuple[int, ...], n: int) -> np.ndarray:
    return np.random.default_rng(seed_parts).permutation(n)


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    n_rows: int


def scan_shards(data_dir: Path) -> tuple[ShardInfo, ...]:
    files = sorted(data_dir.glob("*.parquet"))
    assert files, f"no *.parquet under {data_dir}"
    return tuple(ShardInfo(f, pq.ParquetFile(f).metadata.num_rows) for f in files)


@dataclass(frozen=True)
class BatchLocation:
    epoch: int
    file_idx: int  # index into the sorted file list
    batch_in_shard: int


class BatchSchedule:
    """The pure schedule: step -> which rows of which shard form the global batch."""

    def __init__(self, shards: tuple[ShardInfo, ...], global_batch: int, seed: int):
        assert global_batch > 0
        self.shards = shards
        self.global_batch = global_batch
        self.seed = seed
        self._batches_per_shard = np.array([s.n_rows // global_batch for s in shards])
        assert (self._batches_per_shard > 0).all(), (
            f"global_batch={global_batch} larger than some shard"
        )
        self.steps_per_epoch = int(self._batches_per_shard.sum())

    def _shard_order(self, epoch: int) -> np.ndarray:
        return _perm((self.seed, epoch, 0xD5), len(self.shards))

    def locate(self, step: int) -> BatchLocation:
        epoch, pos = divmod(step, self.steps_per_epoch)
        order = self._shard_order(epoch)
        cum = np.cumsum(self._batches_per_shard[order])
        shard_idx = int(np.searchsorted(cum, pos, side="right"))
        prev = 0 if shard_idx == 0 else int(cum[shard_idx - 1])
        return BatchLocation(
            epoch=epoch,
            file_idx=int(order[shard_idx]),
            batch_in_shard=pos - prev,
        )

    def row_perm(self, loc: BatchLocation) -> np.ndarray:
        return _perm((self.seed, loc.epoch, 0xA0, loc.file_idx), self.shards[loc.file_idx].n_rows)

    def batch_rows(self, step: int) -> tuple[BatchLocation, np.ndarray]:
        """Global-batch row indices into the located shard, in batch order."""
        loc = self.locate(step)
        start = loc.batch_in_shard * self.global_batch
        return loc, self.row_perm(loc)[start : start + self.global_batch]


class ShardServer:
    """Per-process I/O layer over a `BatchSchedule`: loads one shard's token matrix at
    a time, serves this process's slice of each global batch."""

    def __init__(
        self,
        schedule: BatchSchedule,
        seq_len: int,
        process_index: int,
        process_count: int,
    ):
        assert schedule.global_batch % process_count == 0, (
            f"global_batch={schedule.global_batch} not divisible by {process_count} processes"
        )
        self.schedule = schedule
        self.seq_len = seq_len
        self.per_process = schedule.global_batch // process_count
        self.process_index = process_index
        self._loaded_file_idx: int | None = None
        self._tokens: np.ndarray | None = None

    def _load_shard(self, file_idx: int) -> np.ndarray:
        if self._loaded_file_idx != file_idx:
            shard = self.schedule.shards[file_idx]
            table = pq.read_table(shard.path, columns=["input_ids"])
            ids = table.column("input_ids")
            flat = ids.combine_chunks().flatten().to_numpy(zero_copy_only=False)
            tokens = flat.reshape(shard.n_rows, -1)
            # Rows may carry one trailing extra token (next-token-target staging);
            # truncate to the leading seq_len.
            assert tokens.shape[1] in (self.seq_len, self.seq_len + 1), (
                f"{shard.path} rows have seq {tokens.shape[1]}, config says {self.seq_len}"
            )
            self._tokens = tokens[:, : self.seq_len]
            self._loaded_file_idx = file_idx
        assert self._tokens is not None
        return self._tokens

    def local_batch(self, step: int) -> np.ndarray:
        """This process's `[per_process, seq_len]` int32 slice of the step's batch."""
        loc, rows = self.schedule.batch_rows(step)
        tokens = self._load_shard(loc.file_idx)
        lo = self.process_index * self.per_process
        return np.ascontiguousarray(tokens[rows[lo : lo + self.per_process]], dtype=np.int32)
