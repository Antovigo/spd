"""The arithmetic prompt pool and the answer-token set of the integer objective."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from param_decomp.ci_filter.config import ArithmeticPoolConfig, Operation
from param_decomp.experiments.lm.arithmetic_eval import ArithmeticGrid

SYMBOLS: dict[Operation, str] = {"add": "+", "sub": "-"}


class Tokenizer(Protocol):
    """The slice of an HF tokenizer the pool needs."""

    def encode(self, text: str, /, *, add_special_tokens: bool) -> Sequence[int] | np.ndarray: ...

    def decode(self, token_ids: list[int], /) -> str: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class OperationBlock:
    """One operation's contiguous rows of the pool."""

    operation: Operation
    grid: ArithmeticGrid
    start: int

    @property
    def stop(self) -> int:
        return self.start + self.grid.n_a * self.grid.n_b


@dataclass(frozen=True)
class ArithmeticPool:
    tokens: np.ndarray
    """`(n_prompts, T)` int32: the operation blocks in config order, each row-major `(a, b)`."""
    blocks: tuple[OperationBlock, ...]
    position_labels: tuple[str, ...]
    """What each position holds, read off the first prompt (e.g. `<BOS>`, `a`, `+`, `b`, `=`)."""

    @property
    def n_prompts(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])


def build_pool(config: ArithmeticPoolConfig, tokenizer: Tokenizer) -> ArithmeticPool:
    """Tokenize the pool; every prompt must share one token length, so positions are aligned
    across prompts (Llama-3 encodes every integer up to 999 as a single token)."""
    a_values = tuple(range(config.a_range[0], config.a_range[1] + 1))
    b_values = tuple(range(config.b_range[0], config.b_range[1] + 1))
    rows: list[list[int]] = []
    blocks: list[OperationBlock] = []
    for operation in config.operations:
        symbol = SYMBOLS[operation]
        blocks.append(
            OperationBlock(operation, ArithmeticGrid(a_values, b_values, symbol), len(rows))
        )
        rows.extend(
            [int(t) for t in tokenizer.encode(f"{a}{symbol}{b}=", add_special_tokens=True)]
            for a in a_values
            for b in b_values
        )
    lengths = sorted({len(r) for r in rows})
    assert len(lengths) == 1, f"prompts must tokenize to ONE shared length, got {lengths}"
    tokens = np.asarray(rows, dtype=np.int32)
    labels = _position_labels(tokens[0], tokenizer)
    return ArithmeticPool(tokens=tokens, blocks=tuple(blocks), position_labels=labels)


def _position_labels(first_prompt: np.ndarray, tokenizer: Tokenizer) -> tuple[str, ...]:
    """Role names for a `<BOS> a op b =` prompt, else the decoded tokens."""
    decoded = [tokenizer.decode([int(t)]) for t in first_prompt]
    if len(decoded) == 5:
        return ("<BOS>", "a", "op", "b", "=")
    return tuple(decoded)


def answer_token_ids(tokenizer: Tokenizer, include_minus: bool) -> np.ndarray:
    """Every token whose text is all ASCII digits, plus (optionally) the bare `-` token a
    negative answer starts with. Ascending int32."""
    digits = [i for i in range(len(tokenizer)) if re.fullmatch(r"[0-9]+", tokenizer.decode([i]))]
    assert digits, "the tokenizer has no all-digit tokens"
    ids = set(digits)
    if include_minus:
        minus = [int(t) for t in tokenizer.encode("-", add_special_tokens=False)]
        assert len(minus) == 1, f"'-' is not a single token: {minus}"
        ids.add(minus[0])
    return np.asarray(sorted(ids), dtype=np.int32)
