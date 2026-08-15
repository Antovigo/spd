"""Runtime inputs shared by bound LM evaluation operations."""

from dataclasses import dataclass

import jax

from param_decomp.core.run import EvalInvocation


@dataclass(frozen=True)
class LMEvalContext(EvalInvocation):
    pass_index: int
    batches: tuple[jax.Array, ...]
    target_batches: tuple[jax.Array, ...] | None = None
    """A tPD run's prompt-pool draws, which `data.eval` cannot supply — the pool has no
    held-out split, so the targeted root draws them exactly as training does. `None` on a
    plain run, which HAS no second stream; that `None` is also what tells every log key
    which run kind it is in (`scalar_eval_operations.stream_log_prefix`)."""
