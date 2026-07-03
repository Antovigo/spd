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


def translation_lags() -> NDArray[np.int32]:
    """The `(Δa, Δb)` lag set for periodicity scoring: pure-a `(p, 0)`, pure-b `(0, p)`, and
    every mixed `(p, q)` / `(p, -q)` combination of `PERIODS` (checkerboards / diagonals).

    Signed mixed lags matter: correlation at `(Δa, Δb)` equals `(-Δa, -Δb)`, so `(p, q)` and
    `(p, -q)` are the two genuinely distinct diagonal families (a pattern in `a+b mod p` is
    invariant along `(k, -k)`; one in `a-b mod p` along `(k, k)`).
    """
    lags = [(p, 0) for p in PERIODS] + [(0, p) for p in PERIODS]
    lags += [(p, sq) for p in PERIODS for q in PERIODS for sq in (q, -q)]
    return np.array(lags, dtype=np.int32)


def translation_scores(
    grids: NDArray[np.floating], lags: NDArray[np.int32], neuron_chunk: int = 2048
) -> NDArray[np.float32]:
    """Translation-invariance score per grid per lag: `[n, n_lags]`.

    For each `[N, N]` grid, the Pearson correlation between the grid and itself shifted by
    `(Δa, Δb)`, over the overlapping region (mean-centred there, so DC never counts as
    periodic). Each grid is first **planar-detrended** (best-fit `c0 + c1·a + c2·b` removed):
    a shifted linear trend correlates perfectly with itself, so without this every
    magnitude-trending neuron would score ~1 at every lag. 1 = exactly translation-invariant
    at that lag; no sinusoid assumption. A true period-p pattern also scores high at multiples
    of p — read profiles, not argmaxes. Near-constant / pure-trend grids (residual std ~ 0)
    score 0.
    """
    n, n_side, _ = grids.shape
    assert grids.shape[2] == n_side
    coords = np.linspace(-1.0, 1.0, n_side, dtype=np.float32)
    design = np.stack(
        [
            np.ones((n_side, n_side), dtype=np.float32),
            np.broadcast_to(coords[:, None], (n_side, n_side)),
            np.broadcast_to(coords[None, :], (n_side, n_side)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    pinv = np.linalg.pinv(design)  # [3, N*N]
    out = np.zeros((n, len(lags)), dtype=np.float32)
    for start in range(0, n, neuron_chunk):
        chunk = grids[start : start + neuron_chunk].astype(np.float32)
        n_chunk = chunk.shape[0]
        flat = chunk.reshape(n_chunk, -1)
        coeffs = flat @ pinv.T  # [chunk, 3]
        g = (flat - coeffs @ design.T).reshape(n_chunk, n_side, n_side)
        for li, (da, db) in enumerate(lags.tolist()):
            assert 0 <= da < n_side and abs(db) < n_side
            if db >= 0:
                x = g[:, : n_side - da, : n_side - db]
                y = g[:, da:, db:]
            else:
                x = g[:, : n_side - da, -db:]
                y = g[:, da:, : n_side + db]
            xc = x.reshape(n_chunk, -1)
            yc = y.reshape(n_chunk, -1)
            xc = xc - xc.mean(axis=1, keepdims=True)
            yc = yc - yc.mean(axis=1, keepdims=True)
            denom = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = (xc * yc).sum(axis=1) / denom
            out[start : start + n_chunk, li] = np.where(denom > 1e-6, corr, 0.0)
    return out


def silu_combine(gate: NDArray[np.floating], up: NDArray[np.floating]) -> NDArray[np.float32]:
    """Post-SwiGLU neuron activation `silu(gate) * up`, computed in fp32."""
    g = gate.astype(np.float32)
    return ((g / (1.0 + np.exp(-g))) * up.astype(np.float32)).astype(np.float32)


def tokenize_grid(tokenizer: PreTrainedTokenizerBase, op: str) -> torch.Tensor:
    """All prompts tokenized, `[N_VALUES**2, PROMPT_LEN]` — uniform length, no padding."""
    ids = tokenizer(grid_prompts(op), return_tensors="pt").input_ids
    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (N_VALUES**2, PROMPT_LEN), ids.shape
    return ids
