from dataclasses import dataclass
from typing import cast

import pytest

from param_decomp.core.eval_schedule import Every, FirstThenEvery
from param_decomp.core.run import (
    EvalOperation,
    Evaluation,
    _run_due_evaluation,
)
from param_decomp.core.train import TrainState


@dataclass(frozen=True)
class Context:
    step: int


def test_core_schedules_operations_and_builds_one_context():
    contexts: list[int] = []

    def make_context(_state: TrainState, step: int) -> Context:
        contexts.append(step)
        return Context(step)

    evaluation = Evaluation(
        operations=(
            EvalOperation(schedule=Every(4), run=lambda ctx: {"four": ctx.step}),
            EvalOperation(schedule=Every(6), run=lambda ctx: {"six": ctx.step}),
        ),
        make_context=make_context,
    )
    state = cast(TrainState, object())
    assert _run_due_evaluation(evaluation, state, 2) is None
    assert contexts == []
    assert _run_due_evaluation(evaluation, state, 4) == {"four": 4}
    assert contexts == [4]
    assert _run_due_evaluation(evaluation, state, 12) == {"four": 12, "six": 12}
    assert contexts == [4, 12]


def test_core_rejects_eval_output_collisions():
    evaluation = Evaluation(
        operations=(
            EvalOperation(schedule=Every(1), run=lambda _ctx: {"same": 1}),
            EvalOperation(schedule=Every(1), run=lambda _ctx: {"same": 2}),
        ),
        make_context=lambda _state, step: Context(step),
    )
    with pytest.raises(AssertionError, match="colliding keys"):
        _run_due_evaluation(evaluation, cast(TrainState, object()), 1)


def test_first_then_every_schedule_is_explicit():
    evaluation = Evaluation(
        operations=(
            EvalOperation(
                schedule=FirstThenEvery(first=2, steps=10),
                run=lambda ctx: {"slow": ctx.step},
            ),
        ),
        make_context=lambda _state, step: Context(step),
    )
    state = cast(TrainState, object())
    assert _run_due_evaluation(evaluation, state, 2) == {"slow": 2}
    assert _run_due_evaluation(evaluation, state, 4) is None
    assert _run_due_evaluation(evaluation, state, 10) == {"slow": 10}
