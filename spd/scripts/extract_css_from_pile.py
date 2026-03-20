"""Extract a CSS-only subset of the Pile and upload to HuggingFace Hub.

Downloads only the CSS parquet files from andstor/the_pile_github (the Pile's GitHub subset
with language metadata), strips comments, tokenizes, and chunks to match the Pile's sequence
length. Output chunks are written to parquet shards on disk as they are produced, then shuffled
and uploaded.

Usage:
    uv run python -m spd.scripts.extract_css_from_pile --hf_repo <username>/pile-css-no-comments
"""

import argparse
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

PILE_GITHUB_REPO = "andstor/the_pile_github"
PILE_TOKENIZED_REPO = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
SHARD_SIZE = 10_000  # chunks per output parquet shard


def get_pile_seq_len() -> int:
    """Check the sequence length of the tokenized Pile dataset."""
    ds = load_dataset(PILE_TOKENIZED_REPO, split="train", streaming=True)
    seq_len = len(next(iter(ds))["input_ids"])
    print(f"Pile sequence length: {seq_len}")
    return seq_len


def list_parquet_files(split: str) -> list[str]:
    """List CSS parquet files in the Pile GitHub repo for the given split."""
    api = HfApi()
    all_files = api.list_repo_files(PILE_GITHUB_REPO, repo_type="dataset")
    files = sorted(f for f in all_files if f.startswith(f"data/{split}/CSS/"))
    print(f"Found {len(files)} CSS parquet file(s) for '{split}' split")
    return files


def iter_css_texts(parquet_files: list[str]) -> Iterator[str]:
    """Download parquet files one by one and yield comment-stripped CSS texts.

    Reads each parquet in batches to avoid loading all rows into memory at once.
    """
    for path in parquet_files:
        print(f"  Downloading {path}...")
        local = hf_hub_download(PILE_GITHUB_REPO, path, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            for text in batch.column("text").to_pylist():
                stripped = CSS_COMMENT_RE.sub("", text)
                if len(stripped.strip()) >= 20:
                    yield stripped


def process_and_write_shards(
    docs: Iterator[str],
    seq_len: int,
    out_dir: Path,
    max_docs: int | None,
) -> int:
    """Tokenize docs, chunk the token stream, and write parquet shards to out_dir.

    Returns the total number of chunks written.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    out_dir.mkdir(parents=True, exist_ok=True)

    buf: list[int] = []
    shard_buf: list[list[int]] = []
    n_docs = 0
    n_chunks = 0
    shard_idx = 0

    def flush_shard() -> None:
        nonlocal shard_buf, shard_idx
        if not shard_buf:
            return
        table = pa.table({"input_ids": shard_buf}, schema=schema)
        pq.write_table(table, out_dir / f"shard-{shard_idx:05d}.parquet")
        shard_idx += 1
        shard_buf = []

    for text in docs:
        tokens: list[int] = tokenizer.encode(text)  # pyright: ignore[reportAssignmentType]
        buf.extend(tokens)
        while len(buf) >= seq_len:
            shard_buf.append(buf[:seq_len])
            buf = buf[seq_len:]
            n_chunks += 1
            if len(shard_buf) >= SHARD_SIZE:
                flush_shard()

        n_docs += 1
        if n_docs % 5000 == 0:
            print(f"  {n_docs:,} docs → {n_chunks:,} chunks ({shard_idx} shards written)")
        if max_docs is not None and n_docs >= max_docs:
            break

    flush_shard()
    print(f"  {n_docs:,} docs → {n_chunks:,} chunks in {shard_idx} shards")
    return n_chunks


def upload_shards(shard_dir: Path, hf_repo: str, split: str) -> None:
    """Upload all parquet shards in a directory to HF Hub under the given split."""
    api = HfApi()
    shards = sorted(shard_dir.glob("*.parquet"))
    assert shards, f"No parquet files found in {shard_dir}"
    for shard in shards:
        path_in_repo = f"data/{split}/{shard.name}"
        print(f"  Uploading {shard.name} → {path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(shard),
            path_in_repo=path_in_repo,
            repo_id=hf_repo,
            repo_type="dataset",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_repo", required=True, help="HF Hub repo (e.g. user/pile-css)")
    parser.add_argument("--work_dir", default="pile-css-work", help="Temp working directory")
    parser.add_argument("--max_docs", type=int, default=None, help="Cap on CSS documents")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    seq_len = get_pile_seq_len()
    work_dir = Path(args.work_dir)

    # Find CSS parquet files via Hub API (avoids resolving the full 50+ language dataset)
    print("\nListing CSS parquet files...")
    train_files = list_parquet_files("train")
    val_files = list_parquet_files("validation")
    assert train_files, "No CSS train parquet files found"

    # Process train split → shards on disk
    train_dir = work_dir / "train"
    print("\nProcessing train split...")
    n_train = process_and_write_shards(
        iter_css_texts(train_files), seq_len, train_dir, args.max_docs
    )
    assert n_train > 0, "No train chunks produced"

    # Process val split → shards on disk
    val_dir = work_dir / "val"
    print("\nProcessing validation split...")
    n_val = process_and_write_shards(
        iter_css_texts(val_files), seq_len, val_dir, max_docs=None
    )
    print(f"\nTotal: {n_train:,} train chunks, {n_val:,} val chunks")

    # Verify a sample
    first_shard = sorted(train_dir.glob("*.parquet"))[0]
    sample_ids = pq.read_table(first_shard).column("input_ids")[0].as_py()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    sample: str = tokenizer.decode(sample_ids)  # pyright: ignore[reportAttributeAccessIssue]
    print(f"\nSample (first 300 chars):\n{sample[:300]}")

    # Create HF repo and upload shards
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", exist_ok=True)

    print(f"\nUploading train shards to {args.hf_repo}...")
    upload_shards(train_dir, args.hf_repo, "train")

    if n_val > 0:
        print(f"Uploading val shards to {args.hf_repo}...")
        upload_shards(val_dir, args.hf_repo, "val")

    print("Done!")


if __name__ == "__main__":
    main()
