"""Synthetic input generation for InterpBench (Tracr-derived) decomposition.

PD's reconstruction loss is self-supervised against the target model, so the loader only
needs in-distribution `input_ids` — no labels. Inputs are short token sequences over each
case's tiny task vocabulary (`BOS` + content + `PAD` padding to `n_ctx`).

The token<->id mapping is NOT in the HF artifacts (no tokenizer; it lives in the
tracr-compiled encoder). Rather than depend on tracr/JAX, we vendor `circuits_benchmark`'s
vocab + sampling convention and pin each case's encoding in `CASE_SPECS`, recovered
empirically (the model only behaves correctly under its true training encoding). See
`notes/interpbench/notebook.md`. Adding a new case requires its maps.

Vendored from FlyingPumba/circuits_benchmark (MIT):
`circuits_benchmark/benchmark/vocabs.py` (`TRACR_BOS`, `TRACR_PAD`, ascii-letters vocab)
and the `[BOS] + sample + PAD-padding` sampling convention from `tracr_benchmark_case.py`.
"""

import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

TRACR_BOS = "BOS"
TRACR_PAD = "PAD"


def ascii_letters_vocab(count: int) -> list[str]:
    return list(string.ascii_lowercase[:count])


@dataclass(frozen=True)
class CaseDataSpec:
    """Per-case vocabulary, encoding, and ground-truth task for input generation + tests.

    `input_token_to_id` covers content tokens plus `TRACR_BOS`/`TRACR_PAD`;
    `output_id_to_token` decodes output classes back to content tokens. `task` is the
    reference function (content tokens -> per-position output, `None` where a position is
    dropped) used only to validate the pipeline.
    """

    content_tokens: list[str]
    input_token_to_id: dict[str, int]
    output_id_to_token: dict[int, str]
    n_ctx: int
    min_content_len: int
    max_content_len: int
    task: Callable[[list[str]], list[str | None]]

    def __post_init__(self):
        assert set(self.content_tokens) | {TRACR_BOS, TRACR_PAD} == set(self.input_token_to_id), (
            "content_tokens must be exactly the input vocab minus BOS/PAD"
        )
        assert 1 <= self.min_content_len <= self.max_content_len <= self.n_ctx - 1


def _dedup_consecutive(tokens: list[str]) -> list[str | None]:
    """Case 19: drop a token when it equals the previous content token (BOS as sentinel)."""
    prev = TRACR_BOS
    out: list[str | None] = []
    for tok in tokens:
        out.append(None if tok == prev else tok)
        prev = tok
    return out


CASE_SPECS: dict[str, CaseDataSpec] = {
    # Case 19 "remove consecutive duplicate tokens". Maps recovered empirically and
    # validated to 100% on interior content positions (see notebook).
    "19": CaseDataSpec(
        content_tokens=ascii_letters_vocab(3),
        input_token_to_id={TRACR_PAD: 0, TRACR_BOS: 1, "b": 2, "a": 3, "c": 4},
        output_id_to_token={0: "b", 1: "a", 2: "c"},
        n_ctx=15,
        min_content_len=3,
        max_content_len=14,
        task=_dedup_consecutive,
    ),
}


def encode_tokens(tokens: Sequence[str], spec: CaseDataSpec) -> list[int]:
    """`[BOS] + content + PAD-padding` -> input ids of length `n_ctx`."""
    seq = [TRACR_BOS, *tokens]
    seq += [TRACR_PAD] * (spec.n_ctx - len(seq))
    assert len(seq) == spec.n_ctx, f"content too long: {len(tokens)} > n_ctx-1"
    return [spec.input_token_to_id[t] for t in seq]


def sample_content(spec: CaseDataSpec, rng: np.random.Generator) -> list[str]:
    length = int(rng.integers(spec.min_content_len, spec.max_content_len + 1))
    return [spec.content_tokens[i] for i in rng.integers(0, len(spec.content_tokens), length)]


class InterpBenchInputs(Dataset[Tensor]):
    """`n_samples` seeded `input_ids` rows `[n_ctx]`; looped by the trainer."""

    def __init__(self, case_id: str, n_samples: int, seed: int):
        assert case_id in CASE_SPECS, f"no data spec for case {case_id!r}"
        spec = CASE_SPECS[case_id]
        rng = np.random.default_rng(seed)
        rows = [encode_tokens(sample_content(spec, rng), spec) for _ in range(n_samples)]
        self.input_ids = torch.tensor(rows, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.input_ids)

    @override
    def __getitem__(self, idx: int) -> Tensor:
        return self.input_ids[idx]


def make_interpbench_loader(
    case_id: str, *, batch_size: int, n_samples: int, seed: int, shuffle: bool
) -> DataLoader[Tensor]:
    return DataLoader(
        InterpBenchInputs(case_id, n_samples, seed),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
