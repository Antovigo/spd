"""Fixed-prompt target stream for targeted PD (SPEC S34): a small newline-separated prompt
pool tokenized once at build time, sampled (with replacement) into per-step token batches.
The non-target stream stays the normal parquet path (`experiments.lm` loaders)."""

from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer


def load_prompt_tokens(prompts_file: str, tokenizer_name: str, max_seq_len: int) -> np.ndarray:
    """Tokenize each non-empty line of `prompts_file` to a fixed `[n_prompts, max_seq_len]`
    int32 array. Every prompt must tokenize to EXACTLY `max_seq_len` (no pad / no truncate) —
    tPD targets are short constant-length prompts (e.g. "import numpy as"), and silently
    padding or truncating would change which positions the recon KL scores."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    lines = [ln for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    assert lines, f"no prompts in {prompts_file}"
    rows = []
    for ln in lines:
        ids = tokenizer.encode(ln, add_special_tokens=False)
        assert len(ids) == max_seq_len, (
            f"prompt {ln!r} tokenizes to {len(ids)} tokens, expected max_seq_len={max_seq_len}"
        )
        rows.append(ids)
    return np.asarray(rows, dtype=np.int32)


def make_prompt_sample_batch(
    tokens: np.ndarray, global_batch: int, mesh: Mesh, seed: int
) -> Callable[[int], jax.Array]:
    """Per-step target batch: sample `global_batch` prompts (with replacement) from the fixed
    pool and shard over the mesh — a pure function of `step`, like the parquet `sample_batch`."""
    pool = jax.numpy.asarray(tokens)
    n_prompts = int(tokens.shape[0])
    base_key = random.PRNGKey(seed)

    def sample_batch(step: int) -> jax.Array:
        idx = random.randint(random.fold_in(base_key, step), (global_batch,), 0, n_prompts)
        batch = pool[idx]
        return jax.lax.with_sharding_constraint(
            batch, NamedSharding(mesh, P(("replicate", "fsdp")))
        )

    return sample_batch
