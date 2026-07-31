"""Pre-stage + pre-tokenize a portion of a HF text dataset to local int32 parquet shards.

One-shot offline tool so training never streams or tokenizes at run time: the trainer's
deterministic schedule reads a fixed local shard set, startup keeps third-party services
out of the N-rank collective's critical path, and no rank pays tokenization CPU.

Processes source parquet files one at a time (download -> tokenize with `num_proc` ->
write one int32 output shard -> delete the raw file) so peak disk stays ~one raw file plus
the growing output, and is resumable: an output shard that already exists is skipped, so a
requeued job continues where it left off.

Output: `<out_dir>/shard_<NNNNN>.parquet`, each row an `input_ids` list of length `seq_len`
(int32), plus the dataset's `meta.json` (`infra.dataset_store.DatasetMeta`: seq_len + tokenizer —
the dir is self-describing). Publish the dir into the dataset store under a new
immutable name (a deployment step) and reference it as `data: {kind: name, name: <name>}`,
or point `data: {kind: dir, dir: <out_dir>}` at it directly.

Run: `python -m param_decomp.experiments.lm.prestage_tokenized --out_dir <abs> [...]`
"""

from pathlib import Path

import fire
from datasets import Dataset, Sequence, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from param_decomp.core.log import logger
from param_decomp.experiments.lm.data import tokenize_and_concatenate
from param_decomp.infra.dataset_store import DatasetMeta, write_dataset_meta


def _shard_token_count(path: Path, seq_len: int) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows * seq_len


def prestage(
    *,
    out_dir: str,
    num_files: int,
    skip_files: int,
    task_id: int,
    num_tasks: int,
    dataset_repo: str,
    subdir: str,
    revision: str,
    tokenizer_name: str,
    seq_len: int,
    column_name: str,
    num_proc: int,
) -> None:
    """Tokenize `num_files` source parquet files, starting at `skip_files`, into int32
    shards. A disjoint eval split is `skip_files` set past the training split's file
    range, into the same `dataset_repo`.

    Fan-out: task `task_id` of `num_tasks` processes the strided slice
    `range(task_id, num_files, num_tasks)`; shards are named by GLOBAL file index so
    tasks never collide. For scale: 366 files of fineweb `sample/350BT` ≈ 256B tokens ≈ 512GB on disk
    (int32 with ~2x parquet compression).
    Interruption-safe (scavenge): writes are atomic (`.tmp` + rename) and resume skips
    any already-complete shard, so a preempted+requeued task continues cleanly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_dataset_meta(out, DatasetMeta(seq_len=seq_len, tokenizer_name=tokenizer_name))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(dataset_repo, repo_type="dataset", revision=revision)
        if f.startswith(f"{subdir}/") and f.endswith(".parquet")
    )
    assert files, f"no parquet files under {subdir} in {dataset_repo}@{revision}"
    assert skip_files < len(files), f"skip_files={skip_files} >= {len(files)} available"
    num_files = min(num_files, len(files) - skip_files)
    my_indices = list(range(skip_files + task_id, skip_files + num_files, num_tasks))
    logger.info(
        f"task {task_id}/{num_tasks}: {len(files)} files available, processing "
        f"{len(my_indices)} of files [{skip_files}, {skip_files + num_files}) "
        f"(indices {my_indices[:3]}...)"
    )

    for i in my_indices:
        shard = out / f"shard_{i:05d}.parquet"
        if shard.exists():
            logger.info(f"shard_{i:05d} exists; skip")
            continue
        tmp = out / f"shard_{i:05d}.parquet.tmp"  # atomic: write tmp, rename on success

        local = Path(
            hf_hub_download(dataset_repo, files[i], repo_type="dataset", revision=revision)
        )
        raw = load_dataset("parquet", data_files=str(local), split="train")
        assert isinstance(raw, Dataset)
        tokenized = tokenize_and_concatenate(
            raw, tokenizer, column_name=column_name, max_length=seq_len, num_proc=num_proc
        )
        assert isinstance(tokenized, Dataset)
        tokenized = tokenized.cast_column("input_ids", Sequence(Value("int32")))  # pyright: ignore[reportArgumentType]
        tokenized.to_parquet(str(tmp))
        tmp.rename(shard)

        n = len(tokenized) * seq_len
        logger.info(f"[{i}] {files[i]}: {len(tokenized)} seqs / {n / 1e9:.2f}B tok -> {shard.name}")
        local.unlink(missing_ok=True)  # bound peak disk to ~one raw file + the output

    staged = sum(_shard_token_count(p, seq_len) for p in sorted(out.glob("shard_*.parquet")))
    logger.info(
        f"task {task_id} DONE; total staged across all tasks so far: {staged / 1e9:.1f}B tokens"
    )


def cli() -> None:
    fire.Fire(prestage)


if __name__ == "__main__":
    cli()
