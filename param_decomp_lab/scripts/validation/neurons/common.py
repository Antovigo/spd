"""Shared helpers for the L18 neuron census: 0..200 prompt grids, answer tokens, output dir."""

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import op_symbol

NEURONS_DIR: Path = PARAM_DECOMP_OUT_DIR / "runs" / "neurons"
GRID_MAX = 200  # operands run 0..GRID_MAX inclusive
VALUES: NDArray[np.int32] = np.arange(GRID_MAX + 1, dtype=np.int32)
N_VALUES = len(VALUES)
PROMPT_LEN = 5  # <BOS> a op b =  — every 0..200 operand is a single Llama-3 token
LAST_POS = PROMPT_LEN - 1
PERIODS = (2, 5, 10, 20, 25, 33, 50, 100)
NEURON_OPS = ("add", "sub")
D_INT = 14336  # Llama-3.1-8B MLP intermediate width

# Answer offsets whose logprob the ablation sweep records: does ablating a period-p neuron
# move probability mass to `answer ± p`?
OFFSETS = (-100, -50, -25, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 25, 50, 100)


def grid_prompts(op: str) -> list[str]:
    """Every `a<op>b=` prompt, a-outer / b-inner (row-major in the `[a, b]` grid)."""
    sym = op_symbol(op)
    return [f"{a}{sym}{b}=" for a in VALUES for b in VALUES]


def correct_answer_grid(op: str) -> NDArray[np.int64]:
    a = VALUES[:, None].astype(np.int64)
    b = VALUES[None, :].astype(np.int64)
    match op:
        case "add":
            return a + b
        case "sub":
            return a - b
        case _:
            raise AssertionError(f"unsupported op for the neuron census: {op}")


def correct_first_token_grid(tokenizer: PreTrainedTokenizerBase, op: str) -> NDArray[np.int32]:
    """First token id of the true answer string — what a correct model emits at `=`.

    Sums are single tokens (≤ 400); negative differences start with a `-` token.
    """
    answers = correct_answer_grid(op)
    first = {
        v: tokenizer.encode(str(v), add_special_tokens=False)[0]
        for v in np.unique(answers).tolist()
    }
    return np.vectorize(first.__getitem__)(answers).astype(np.int32)


def offset_first_token_grid(tokenizer: PreTrainedTokenizerBase, op: str) -> NDArray[np.int32]:
    """First token id of `str(answer + δ)` per offset δ — `[N_VALUES, N_VALUES, len(OFFSETS)]`.

    Offsets crossing zero degenerate to the bare `-` token for several δ at once; downstream
    analysis masks those by comparing token ids, not by assuming distinct targets.
    """
    shifted = correct_answer_grid(op)[..., None] + np.array(OFFSETS, dtype=np.int64)
    first = {
        v: tokenizer.encode(str(v), add_special_tokens=False)[0]
        for v in np.unique(shifted).tolist()
    }
    return np.vectorize(first.__getitem__)(shifted).astype(np.int32)


def token_value_map(
    tokenizer: PreTrainedTokenizerBase, token_ids: NDArray[np.int32]
) -> dict[int, int]:
    """`token id -> integer value` for the ids whose string parses as an int (e.g. `400`, not `-`)."""
    out: dict[int, int] = {}
    for tid in np.unique(token_ids).tolist():
        s = tokenizer.decode([tid]).strip()
        if s.lstrip("-").isdigit() and s != "-":
            out[tid] = int(s)
    return out


def tokenize_grid(tokenizer: PreTrainedTokenizerBase, op: str) -> torch.Tensor:
    """All prompts tokenized, `[N_VALUES**2, PROMPT_LEN]` — uniform length, no padding."""
    ids = tokenizer(grid_prompts(op), return_tensors="pt").input_ids
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (N_VALUES**2, PROMPT_LEN), ids.shape
    return ids
