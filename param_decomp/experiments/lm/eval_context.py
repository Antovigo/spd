"""Runtime inputs shared by bound LM evaluation operations."""

from dataclasses import dataclass

import jax

from param_decomp.core.run import EvalInvocation


@dataclass(frozen=True)
class LMEvalContext(EvalInvocation):
    pass_index: int
    batches: tuple[jax.Array, ...]
