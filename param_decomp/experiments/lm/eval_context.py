"""Runtime inputs shared by bound LM evaluation operations."""

from dataclasses import dataclass

import jax

from param_decomp.core.run import EvalInvocation


@dataclass(frozen=True)
class LMEvalContext(EvalInvocation):
    pass_index: int
    batches: tuple[jax.Array, ...]
    """The broad `data.eval` stream — for a tPD run that is the NON-TARGET distribution."""
    target_batches: tuple[jax.Array, ...] | None
    """The tPD target prompt pool, drawn like a training target batch. `None` on a plain
    run, which has no target stream; the target-stream metrics refuse that at bind time."""
