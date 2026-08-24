"""The stream namespace rule: the data a run optimizes for is unlabelled.

A plain run has one stream and keeps `eval/`; a tPD run adds a second, so its target pool
takes the bare namespace and the broad corpus moves under `eval/nontarget_data/`. The plain
case is pinned here because it is a COMPATIBILITY GUARANTEE — those keys are shared with
every non-targeted run and both toys, and adding a second stream must not move them.
"""

from typing import cast

import jax.numpy as jnp

from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.scalar_eval_operations import Stream, stream_log_prefix

TARGET_BATCHES = (jnp.zeros((1, 4), jnp.int32),)


def context(target_batches: tuple[jnp.ndarray, ...] | None) -> LMEvalContext:
    """A context carrying only what the namespace rule reads; `state` is never touched."""
    return LMEvalContext(
        state=cast(TrainState, cast(object, None)),
        now_step=0,
        placed_ci_fn=cast(PlacedCIFn, cast(object, None)),
        pass_index=0,
        batches=(),
        shared_ci_reductions=lambda: {},
        target_batches=target_batches,
    )


def test_plain_run_keys_are_unchanged_by_the_two_stream_rule():
    assert stream_log_prefix("nontarget", context(None)) == "eval/"


def test_targeted_run_labels_the_broad_stream_and_leaves_the_target_bare():
    targeted = context(TARGET_BATCHES)
    assert stream_log_prefix("target", targeted) == "eval/"
    assert stream_log_prefix("nontarget", targeted) == "eval/nontarget_data/"


def test_the_bare_namespace_is_always_the_data_of_interest():
    """What makes the run kinds comparable AS OBJECTIVES — and the trap that comes with it:
    the bare namespace is not the same DATA in both."""
    plain, targeted = context(None), context(TARGET_BATCHES)
    assert stream_log_prefix("nontarget", plain) == stream_log_prefix("target", targeted)
    assert stream_log_prefix("nontarget", targeted) != stream_log_prefix("nontarget", plain)


def test_every_stream_resolves_under_both_run_kinds():
    for stream in cast(tuple[Stream, ...], ("nontarget", "target")):
        for target_batches in (None, TARGET_BATCHES):
            prefix = stream_log_prefix(stream, context(target_batches))
            assert prefix.startswith("eval/") and prefix.endswith("/")
