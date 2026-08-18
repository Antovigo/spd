"""When an eval operation fires: the cadence vocabulary the engine dispatches on.

Pure value types, deliberately free of jax and of the engine that consumes them, so the
config layer can name a schedule (`experiments.eval_config.schedule_for`) without dragging
the trainer — and matplotlib, and optax — into every config parse.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Every:
    """Every `steps` steps, NOT at step 0 — the baseline pass belongs to whatever asks
    for it by name (`FirstThenEvery(0, ...)`)."""

    steps: int


@dataclasses.dataclass(frozen=True)
class FirstThenEvery:
    first: int
    steps: int


@dataclasses.dataclass(frozen=True)
class EveryAfterFirst:
    """`Every`, minus the `first` step — for an operation whose output at an untrained
    decomposition is not worth what it costs, whatever the callback's `slow_on_first_step`
    says."""

    first: int
    steps: int


type EvalSchedule = Every | FirstThenEvery | EveryAfterFirst


def eval_due(schedule: EvalSchedule, step: int) -> bool:
    match schedule:
        case Every(steps):
            return step > 0 and step % steps == 0
        case FirstThenEvery(first, steps):
            return step == first or step % steps == 0
        case EveryAfterFirst(first, steps):
            return step != first and step % steps == 0
