"""Static, file-backed LM target loader for targeted decomposition runs."""

from collections.abc import Iterator
from typing import Any, override

import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from param_decomp_lab.experiments.lm.data import LMDataConfig


def load_prompts_dataset(
    prompts_file: str,
    tokenizer: PreTrainedTokenizer,
) -> Int[Tensor, "n_prompts seq_len"]:
    """Tokenize one prompt per non-empty line into a `[n_prompts, seq_len]` tensor.

    No padding: every prompt must tokenize to the same length, so the answer sits at a
    constant position across the pool (relied on by last-position reconstruction). Raises
    if the lengths differ.
    """
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    assert prompts, f"no prompts found in {prompts_file}"

    rows = [tokenizer.encode(prompt) for prompt in prompts]
    seq_len = len(rows[0])
    for prompt, token_ids in zip(prompts, rows, strict=True):
        assert len(token_ids) == seq_len, (
            f"prompt tokenizes to {len(token_ids)} tokens but expected {seq_len}; all prompts "
            f"must share one length (padding is disabled): {prompt[:80]!r}"
        )
    return torch.tensor(rows, dtype=torch.long)


class StaticBatchLoader(IterableDataset[Tensor]):
    """Infinite iterable over a fixed in-memory prompt pool.

    Each iteration randomly samples `min(batch_size, n_prompts)` rows from the pool
    without replacement within a batch (the whole pool, reshuffled, when
    `batch_size >= n_prompts`). Seeded for reproducibility.
    """

    def __init__(self, pool: Int[Tensor, "n_prompts seq"], batch_size: int, seed: int):
        self.pool = pool
        self.rows_per_batch = min(batch_size, pool.shape[0])
        self.generator = torch.Generator().manual_seed(seed)

    @override
    def __iter__(self) -> Iterator[Tensor]:
        while True:
            perm = torch.randperm(self.pool.shape[0], generator=self.generator)
            yield self.pool[perm[: self.rows_per_batch]]


def create_prompts_data_loader(
    cfg: LMDataConfig,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[Any], PreTrainedTokenizer]:
    """Build a static prompts loader from `cfg.prompts_file`, mirroring
    `create_lm_data_loader`'s return shape."""
    assert cfg.prompts_file is not None, "create_prompts_data_loader requires prompts_file"
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    pool = load_prompts_dataset(cfg.prompts_file, tokenizer)
    dataset = StaticBatchLoader(pool, batch_size=batch_size, seed=seed)
    return DataLoader(dataset, batch_size=None), tokenizer
