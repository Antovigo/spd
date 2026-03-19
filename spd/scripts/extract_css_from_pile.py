"""Extract CSS code from The Pile's GitHub subset, strip comments, tokenize, and push to HF Hub.

Matches the Pile's processing pipeline: concatenate documents → tokenize → chunk → shuffle.

Usage:
    uv run python -m spd.scripts.extract_css_from_pile --hf_repo <your-username>/pile-css-chunks
"""

import argparse
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

PILE_GITHUB_DATASET = "andstor/the_pile_github"
PILE_TOKENIZED_DATASET = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"

CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_css_comments(text: str) -> str:
    return CSS_COMMENT_PATTERN.sub("", text)


def get_pile_seq_len() -> int:
    """Get the sequence length used in the tokenized Pile dataset."""
    ds = load_dataset(PILE_TOKENIZED_DATASET, split="train", streaming=True)
    sample = next(iter(ds))
    seq_len = len(sample["input_ids"])
    print(f"Pile tokenized sequence length: {seq_len}")
    return seq_len


def extract_css_texts(max_docs: int | None = None) -> list[str]:
    """Stream the Pile GitHub subset and extract CSS documents.

    Args:
        max_docs: Max number of CSS documents to collect (None = all).
    """
    ds = load_dataset(PILE_GITHUB_DATASET, split="train", streaming=True)
    # Language is nested under meta.language in this dataset
    ds = ds.filter(lambda x: x.get("meta", {}).get("language") == "CSS")

    # Verify column structure on first example
    first = next(iter(ds))
    assert "text" in first, f"Expected 'text' column, got columns: {list(first.keys())}"

    texts: list[str] = []
    for example in ds:
        text = example["text"]
        stripped = strip_css_comments(text)
        if len(stripped.strip()) < 20:
            continue
        texts.append(stripped)
        if len(texts) % 1000 == 0:
            print(f"  Collected {len(texts):,} CSS documents...")
        if max_docs is not None and len(texts) >= max_docs:
            break

    print(f"Collected {len(texts):,} CSS documents total")
    return texts


def tokenize_and_chunk(texts: list[str], seq_len: int) -> list[list[int]]:
    """Concatenate texts, tokenize in batches, and chunk into fixed-length sequences."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    total_chars = sum(len(t) for t in texts)
    print(f"Total text: {total_chars:,} characters across {len(texts):,} documents")

    # Tokenize document-by-document and concatenate token IDs.
    # This avoids building a single massive string and is equivalent to concatenating
    # texts then tokenizing (up to BOS/EOS tokens which we strip).
    all_tokens: list[int] = []
    for i, text in enumerate(texts):
        tokens: list[int] = tokenizer.encode(text)  # pyright: ignore[reportAssignmentType]
        all_tokens.extend(tokens)
        if (i + 1) % 5000 == 0:
            print(f"  Tokenized {i + 1:,}/{len(texts):,} documents ({len(all_tokens):,} tokens)")

    print(f"Total tokens: {len(all_tokens):,}")

    # Chunk into seq_len sequences
    n_chunks = len(all_tokens) // seq_len
    chunks = [all_tokens[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)]
    print(f"Created {n_chunks:,} chunks of {seq_len} tokens each")

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CSS from Pile GitHub subset")
    parser.add_argument(
        "--hf_repo",
        type=str,
        required=True,
        help="HuggingFace Hub repo to push to (e.g., 'username/pile-css-chunks')",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="pile-css-chunks",
        help="Local directory to save the dataset before uploading (default: pile-css-chunks)",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Max CSS documents to collect (None = all)",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    # Step 1: Get Pile sequence length
    seq_len = get_pile_seq_len()

    # Step 2: Extract CSS texts
    print("\nExtracting CSS from Pile GitHub subset...")
    texts = extract_css_texts(max_docs=args.max_docs)

    assert len(texts) > 0, "No CSS documents found"

    # Step 3: Tokenize and chunk
    print("\nTokenizing and chunking...")
    chunks = tokenize_and_chunk(texts, seq_len)

    assert len(chunks) > 0, "No chunks created — not enough CSS tokens"

    # Step 4: Shuffle
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(chunks))
    chunks = [chunks[i] for i in indices]

    # Step 5: Split train/val
    n_val = max(1, int(len(chunks) * args.val_frac))
    n_train = len(chunks) - n_val
    print(f"\nSplit: {n_train:,} train, {n_val:,} val")

    train_ds = Dataset.from_dict({"input_ids": chunks[:n_train]})
    val_ds = Dataset.from_dict({"input_ids": chunks[n_train:]})
    dataset_dict = DatasetDict({"train": train_ds, "val": val_ds})

    # Step 6: Verify a sample
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    sample_tokens = chunks[0]
    sample_text: str = tokenizer.decode(sample_tokens)  # pyright: ignore[reportAttributeAccessIssue]
    print(f"\nSample decoded text (first 200 chars):\n{sample_text[:200]}")

    # Step 7: Save locally
    local_path = Path(args.local_dir)
    print(f"\nSaving dataset locally to: {local_path}")
    dataset_dict.save_to_disk(str(local_path))
    print(f"Saved to {local_path}")

    # Step 8: Push to HF Hub
    print(f"\nPushing to HuggingFace Hub: {args.hf_repo}")
    dataset_dict.push_to_hub(args.hf_repo)
    print("Done!")


if __name__ == "__main__":
    main()
