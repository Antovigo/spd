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
    max_seq_len: int,
) -> Int[Tensor, "n_prompts max_seq_len"]:
    """Tokenize one prompt per non-empty line, padded to `max_seq_len`.

    Raises if any tokenized prompt exceeds `max_seq_len` (no silent truncation).
    """
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    assert prompts, f"no prompts found in {prompts_file}"

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    assert isinstance(pad_token_id, int), "tokenizer has neither pad_token_id nor eos_token_id"

    rows = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        assert len(token_ids) <= max_seq_len, (
            f"prompt tokenizes to {len(token_ids)} tokens, exceeding max_seq_len={max_seq_len}: "
            f"{prompt[:80]!r}"
        )
        rows.append(token_ids + [pad_token_id] * (max_seq_len - len(token_ids)))
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
    pool = load_prompts_dataset(cfg.prompts_file, tokenizer, cfg.max_seq_len)
    dataset = StaticBatchLoader(pool, batch_size=batch_size, seed=seed)
    return DataLoader(dataset, batch_size=None), tokenizer
