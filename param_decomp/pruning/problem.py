"""The prompt set `P` (true answer ≥ 1), its answer tokens and the integer token table."""

from dataclasses import dataclass

import numpy as np

from param_decomp.ci_filter.attribution import VALUE_OF
from param_decomp.ci_filter.config import ArithmeticPoolConfig
from param_decomp.ci_filter.pool import Tokenizer, build_pool
from param_decomp.pruning.stats import N_INT


@dataclass(frozen=True)
class Problem:
    tokens: np.ndarray
    """`(N, T)` int32 prompts of `P`, in pool order."""
    pool_rows: np.ndarray
    """`(N,)` int: each prompt's row in the full pool (the CI filter's row order)."""
    op: np.ndarray
    """`(N,)` int: 0 = add, 1 = sub, 2 = mul (`ArithmeticPoolConfig.operations` order mapped
    by name)."""
    a: np.ndarray
    b: np.ndarray
    r: np.ndarray
    """`(N,)` int: the true answer, an integer in 1..999."""
    y: np.ndarray
    """`(N,)` int32: the answer's token (the integer's single token)."""
    int_ids: np.ndarray
    """`(1000,)` int32: the token of each integer 0..999."""

    @property
    def n(self) -> int:
        return int(self.r.size)


OP_ID = {"add": 0, "sub": 1, "mul": 2}


def integer_token_ids(tokenizer: Tokenizer) -> np.ndarray:
    """The single token of every integer 0..999 (asserted single)."""
    ids = []
    for x in range(N_INT):
        encoded = [int(t) for t in tokenizer.encode(str(x), add_special_tokens=False)]
        assert len(encoded) == 1, f"{x} is not a single token: {encoded}"
        ids.append(encoded[0])
    assert len(set(ids)) == N_INT
    return np.asarray(ids, np.int32)


def build_problem(config: ArithmeticPoolConfig, tokenizer: Tokenizer) -> Problem:
    pool = build_pool(config, tokenizer)
    a_all, b_all, op_all, r_all = [], [], [], []
    for block in pool.blocks:
        value_of = VALUE_OF[block.operation]
        for a in block.grid.a_values:
            for b in block.grid.b_values:
                a_all.append(a)
                b_all.append(b)
                op_all.append(OP_ID[block.operation])
                r_all.append(value_of(a, b))
    r_full = np.asarray(r_all)
    keep = np.nonzero(r_full >= 1)[0]
    r = r_full[keep]
    assert r.max() < N_INT, f"answers up to {r.max()} do not fit V_int"
    int_ids = integer_token_ids(tokenizer)
    return Problem(
        tokens=pool.tokens[keep],
        pool_rows=keep,
        op=np.asarray(op_all)[keep],
        a=np.asarray(a_all)[keep],
        b=np.asarray(b_all)[keep],
        r=r,
        y=int_ids[r],
        int_ids=int_ids,
    )
