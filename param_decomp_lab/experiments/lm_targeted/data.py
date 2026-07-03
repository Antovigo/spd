"""Fixed-prompt loader for the tPD TARGET stream.

The broad NON-TARGET stream reuses the LM parquet path (`ShardServer`); the TARGET stream
is a small fixed pool of prompts read from a text file, tokenized ONCE at build time (no
per-step tokenization — the 80-rank thunderherd lesson still applies, though the pool is
tiny). This mirrors the torch `prompts_dataset.py` / `StaticBatchLoader`.

See `notes/targeted_jax_plan.md` Phase 3.
"""

from pathlib import Path

import jax
import numpy as np


def load_prompt_tokens(prompts_file: Path, tokenizer_name: str, max_seq_len: int) -> np.ndarray:
    """Read one prompt per non-empty line, tokenize each, and return a `[n_prompts,
    max_seq_len]` int32 array. RAISE on any prompt longer than `max_seq_len` (no silent
    truncation); short prompts are padded.

    TODO(tPD): implement per plan Phase 3. Build the tokenizer
    (`AutoTokenizer.from_pretrained(tokenizer_name)`), tokenize each line, assert
    `len <= max_seq_len`, pad, stack. Decide padding token + whether the target recon
    scores only the final position (the paper reconstructs the last-token completion).
    """
    _ = (prompts_file, tokenizer_name, max_seq_len)  # pending implementation
    raise NotImplementedError("tPD prompt tokenization — see targeted_jax_plan.md Phase 3")


class TargetPromptServer:
    """A fixed in-memory pool of tokenized target prompts, yielding a per-step batch sharded
    over the mesh (mirrors `param_decomp.data.ShardServer`'s per-step contract so the engine
    `sample_batch` seam is identical for both streams).

    TODO(tPD): implement per plan Phase 3. If `batch_size >= n_prompts`, yield the whole
    pool each step (optionally reshuffled); else sample `batch_size` rows without
    replacement, seeded by step, then shard across the mesh like `_global_token_batch`.
    """

    def __init__(self, tokens: np.ndarray, global_batch: int, mesh: jax.sharding.Mesh, seed: int):
        self.tokens = tokens
        self.global_batch = global_batch
        self.mesh = mesh
        self.seed = seed

    def local_batch(self, step: int) -> jax.Array:
        _ = step  # pending implementation
        raise NotImplementedError("tPD target-prompt batching — see targeted_jax_plan.md Phase 3")
