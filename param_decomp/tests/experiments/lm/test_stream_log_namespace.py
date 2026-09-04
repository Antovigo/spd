"""The stream namespace rule: the data a run optimizes for is unlabelled.

A plain run has one stream and keeps `eval/`; a tPD run adds a second, so its target pool
takes the bare namespace and the broad corpus moves under `eval/nontarget_data/`. The plain
case is pinned here because it is a COMPATIBILITY GUARANTEE — those keys are shared with
every non-targeted run and both toys, and adding a second stream must not move them.
"""

from typing import cast

import jax.numpy as jnp
import pytest

from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.eval_context import (
    LMEvalPass,
    Stream,
    stream_batches,
    stream_log_prefix,
)

TARGET_BATCHES = (jnp.zeros((1, 4), jnp.int32),)


def eval_pass(target_batches: tuple[jnp.ndarray, ...] | None) -> LMEvalPass:
    """A pass carrying only what the namespace rule reads; `state` is never touched."""
    return LMEvalPass(
        state=cast(TrainState, cast(object, None)),
        now_step=0,
        placed_ci_fn=cast(PlacedCIFn, cast(object, None)),
        pass_index=0,
        batches=(),
        target_batches=target_batches,
    )


def test_plain_run_keys_are_unchanged_by_the_two_stream_rule():
    assert stream_log_prefix("nontarget", targeted=False) == "eval/"


def test_targeted_run_labels_the_broad_stream_and_leaves_the_target_bare():
    assert stream_log_prefix("target", targeted=True) == "eval/"
    assert stream_log_prefix("nontarget", targeted=True) == "eval/nontarget_data/"


def test_the_bare_namespace_is_always_the_data_of_interest():
    """What makes the run kinds comparable AS OBJECTIVES — and the trap that comes with it:
    the bare namespace is not the same DATA in both."""
    assert stream_log_prefix("nontarget", targeted=False) == stream_log_prefix(
        "target", targeted=True
    )
    assert stream_log_prefix("nontarget", targeted=True) != stream_log_prefix(
        "nontarget", targeted=False
    )


def test_the_hidden_role_takes_its_own_segment_under_either_stream():
    assert stream_log_prefix("target", targeted=True, role="hidden") == "eval/hidden_ci/"
    assert (
        stream_log_prefix("nontarget", targeted=True, role="hidden")
        == "eval/nontarget_data/hidden_ci/"
    )


def test_a_plain_run_cannot_spell_the_target_stream():
    with pytest.raises(AssertionError):
        stream_log_prefix("target", targeted=False)
    with pytest.raises(AssertionError):
        stream_batches("target", eval_pass(None))


def test_the_pass_reports_its_run_kind_from_the_target_batches():
    assert not eval_pass(None).targeted
    assert eval_pass(TARGET_BATCHES).targeted
    for stream in cast(tuple[Stream, ...], ("nontarget", "target")):
        assert stream_batches(stream, eval_pass(TARGET_BATCHES)) is not None
