"""The prompt pool read from a prompts file, the problem `P` and the integer token table.

Each line of the prompts file is one of the config's templates with two integer operands `a`,
`b` filled in; the line's operation is the template it matches and its true answer `r`
(integer) is `a + b` or `a - b`. Every line must tokenize (with the BOS token) to one shared
length `T`, so position `t` (integer, `0 <= t < T`) holds the same role in every prompt. `P` is
the prompts with `r >= 0`; `V_int` is the 1000 single tokens of the integers 0..999, in integer
order, and `r` must be one of them."""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.pool import Tokenizer
from param_decomp.integer_only.config import Operation

N_INT = 1000
"""`|V_int|`: the integers 0..999."""
FIELDS = ("a", "b", "op")


@dataclass(frozen=True)
class PromptPool:
    tokens: np.ndarray
    """`(N, T)` int32, row `i` = line `i` of the prompts file (0-based, nonempty lines)."""
    ops: tuple[Operation, ...]
    """The operations, in the config's template order."""
    op: np.ndarray
    """`(N,)` int8: index into `ops`."""
    a: np.ndarray
    b: np.ndarray
    r: np.ndarray
    """`(N,)` int64 operands and true answer."""
    position_labels: tuple[str, ...]
    """`(T,)`: `a`, `b` or `op` where the token is a function of that field (and varies), else
    the decoded constant token with spaces shown as `␣`."""
    sha256: str
    """Of the prompts file."""

    @property
    def n(self) -> int:
        return int(self.r.size)

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    def problem_rows(self) -> np.ndarray:
        """Ascending rows of `P`: true answer `r >= 0`, each answer an integer of `V_int`."""
        rows = np.flatnonzero(self.r >= 0)
        assert rows.size and int(self.r[rows].max()) < N_INT, "answers must lie in 0..999"
        return rows

    def save(self, path: Path) -> None:
        np.savez(
            path, tokens=self.tokens, ops=np.asarray(self.ops), op=self.op, a=self.a, b=self.b,
            r=self.r, position_labels=np.asarray(self.position_labels),
            sha256=np.asarray(self.sha256),
        )  # fmt: skip

    @staticmethod
    def load(path: Path) -> "PromptPool":
        with np.load(path) as z:
            return PromptPool(
                tokens=z["tokens"], ops=tuple(str(o) for o in z["ops"]),  # pyright: ignore[reportArgumentType]
                op=z["op"], a=z["a"], b=z["b"], r=z["r"],
                position_labels=tuple(str(x) for x in z["position_labels"]),
                sha256=str(z["sha256"]),
            )  # fmt: skip


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template_pattern(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    for field in ("a", "b"):
        escaped = escaped.replace(re.escape("{" + field + "}"), rf"(?P<{field}>-?[0-9]+)")
    return re.compile(f"^{escaped}$")


def result(op: Operation, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    match op:
        case "add":
            return a + b
        case "sub":
            return a - b


def parse_lines(
    lines: list[str], templates: Mapping[Operation, str]
) -> tuple[tuple[Operation, ...], np.ndarray, np.ndarray, np.ndarray]:
    """`(ops, op, a, b)` of every line; each line must match exactly one template."""
    ops = tuple(templates)
    patterns = [_template_pattern(templates[o]) for o in ops]
    op = np.empty(len(lines), np.int8)
    a = np.empty(len(lines), np.int64)
    b = np.empty(len(lines), np.int64)
    for i, line in enumerate(lines):
        hits = [(k, m) for k, p in enumerate(patterns) if (m := p.match(line)) is not None]
        assert len(hits) == 1, f"line {i} {line!r} matches {len(hits)} templates"
        (k, m), = hits  # fmt: skip
        op[i], a[i], b[i] = k, int(m["a"]), int(m["b"])
    return ops, op, a, b


def position_labels(
    tokens: np.ndarray, fields: Mapping[str, np.ndarray], tokenizer: Tokenizer
) -> tuple[str, ...]:
    """Name each position by the field its token is a function of (and varies with), else by
    its constant decoded token."""
    labels = []
    for t in range(tokens.shape[1]):
        column = tokens[:, t]
        if np.all(column == column[0]):
            labels.append(tokenizer.decode([int(column[0])]).replace(" ", "␣"))
            continue
        named = [f for f, v in fields.items() if _is_function_of(column, v)]
        assert len(named) == 1, f"position {t} depends on {named or 'several fields'}"
        labels.append(named[0])
    return tuple(labels)


def _is_function_of(column: np.ndarray, field: np.ndarray) -> bool:
    """Whether `column[i]` is determined by `field[i]` for every row."""
    pairs = np.unique(np.stack([field, column], axis=1), axis=0)
    return np.unique(pairs[:, 0]).size == pairs.shape[0]


def build_pool(
    prompts: Path, templates: Mapping[Operation, str], tokenizer: Tokenizer
) -> PromptPool:
    """Tokenize the prompts file; checks one shared length and the template parse."""
    lines = [line for line in prompts.read_text().splitlines() if line.strip()]
    assert lines, f"no prompts in {prompts}"
    ops, op, a, b = parse_lines(lines, templates)
    rows = [[int(t) for t in tokenizer.encode(line, add_special_tokens=True)] for line in lines]
    lengths = sorted({len(r) for r in rows})
    assert len(lengths) == 1, f"prompts must tokenize to ONE shared length, got {lengths}"
    tokens = np.asarray(rows, np.int32)
    r = np.empty(op.size, np.int64)
    for k, o in enumerate(ops):
        sel = op == k
        r[sel] = result(o, a[sel], b[sel])
    labels = position_labels(tokens, {"a": a, "b": b, "op": op}, tokenizer)
    return PromptPool(
        tokens=tokens, ops=ops, op=op, a=a, b=b, r=r, position_labels=labels,
        sha256=file_sha256(prompts),
    )  # fmt: skip


def integer_token_ids(tokenizer: Tokenizer) -> np.ndarray:
    """`(1000,)` int32: the single token of each integer 0..999 (asserted single, distinct)."""
    ids = []
    for x in range(N_INT):
        encoded = [int(t) for t in tokenizer.encode(str(x), add_special_tokens=False)]
        assert len(encoded) == 1, f"{x} is not a single token: {encoded}"
        ids.append(encoded[0])
    assert len(set(ids)) == N_INT
    return np.asarray(ids, np.int32)


def check_answer_tokens(
    prompts: Path, pool: PromptPool, rows: np.ndarray, int_ids: np.ndarray, tokenizer: Tokenizer
) -> None:
    """The prompt text followed by its answer's digits tokenizes to the prompt's tokens plus
    exactly the answer's `V_int` token (e.g. a trailing space stays its own token)."""
    lines = [line for line in prompts.read_text().splitlines() if line.strip()]
    for i in rows:
        got = [int(t) for t in tokenizer.encode(lines[i] + str(pool.r[i]), add_special_tokens=True)]
        want = [*pool.tokens[i].tolist(), int(int_ids[pool.r[i]])]
        assert got == want, f"row {i} {lines[i]!r}: answer tokenizes as {got[pool.seq_len :]}"


def draw_check_set(rows: np.ndarray, size: int, seed: int) -> np.ndarray:
    """`Q`: `size` distinct rows of `P`, in the order drawn (a uniform random order, so any
    leading slice is itself a uniform random subset)."""
    assert size <= rows.size, (size, rows.size)
    rng = np.random.default_rng(np.random.SeedSequence((seed, 17)))
    return rng.choice(rows, size, replace=False).astype(np.int64)
