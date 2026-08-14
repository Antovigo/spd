"""The stream namespace rule: the data a run optimizes for is unlabelled.

A plain run has one stream and keeps `eval/`; a tPD run adds a second, so its target pool
takes the bare namespace and the broad corpus moves under `eval/nontarget_data/`. The plain
case is pinned here because it is a COMPATIBILITY GUARANTEE — those keys are shared with
every non-targeted run and both toys, and adding a second stream must not move them.
"""

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
from jaxtyping import Array

from param_decomp.core.configs import (
    CIMaskedReconLossConfig,
    UnmaskedNoDeltaReconLossConfig,
)
from param_decomp.core.eval_schedule import Every
from param_decomp.core.run import EvalOperation
from param_decomp.core.train import TrainState
from param_decomp.experiments.lm.eval import CE_KL_VARIANTS
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.scalar_eval_operations import (
    Stream,
    _make_scalar_operation,
    stream_log_prefix,
)

TARGET_BATCHES = (jnp.zeros((1, 4), jnp.int32),)


def context(target_batches: tuple[jnp.ndarray, ...] | None) -> LMEvalContext:
    """A context carrying only what the namespace rule reads; `state` is never touched."""
    return LMEvalContext(
        state=cast(TrainState, cast(object, None)),
        now_step=0,
        pass_index=0,
        batches=(),
        target_batches=target_batches,
    )


def test_plain_run_keys_are_unchanged_by_the_two_stream_rule():
    assert stream_log_prefix("broad", context(None)) == "eval/"


def test_targeted_run_labels_the_broad_stream_and_leaves_the_target_bare():
    targeted = context(TARGET_BATCHES)
    assert stream_log_prefix("target_data", targeted) == "eval/"
    assert stream_log_prefix("broad", targeted) == "eval/nontarget_data/"


def test_the_bare_namespace_is_always_the_data_of_interest():
    """What makes the run kinds comparable AS OBJECTIVES — and the trap that comes with it:
    the bare namespace is not the same DATA in both."""
    plain, targeted = context(None), context(TARGET_BATCHES)
    assert stream_log_prefix("broad", plain) == stream_log_prefix("target_data", targeted)
    assert stream_log_prefix("broad", targeted) != stream_log_prefix("broad", plain)


def test_every_stream_resolves_under_both_run_kinds():
    for stream in cast(tuple[Stream, ...], ("broad", "target_data")):
        for target_batches in (None, TARGET_BATCHES):
            prefix = stream_log_prefix(stream, context(target_batches))
            assert prefix.startswith("eval/") and prefix.endswith("/")


def test_one_operation_per_stream_reads_that_stream_and_labels_it():
    """The end-to-end shape the binder relies on: bind the SAME step twice, once per
    stream, and the two operations read different batches and land under different keys —
    which is what lets a metric be authored once and measured on both."""

    def step(
        _model: Any, _components: Any, _ci_fn: Any, value: Array, _key: Any
    ) -> dict[str, Array]:
        return {"l0/0.0_site": value}

    def operation_for(stream: Stream) -> EvalOperation[LMEvalContext]:
        return _make_scalar_operation(
            Every(1),
            step,
            ("l0/",),
            cast(Any, object()),
            jnp.array([0, 0], dtype=jnp.uint32),
            train_steps=0,
            eval_steps=1,
            stream=stream,
        )

    targeted = LMEvalContext(
        state=cast(
            TrainState,
            cast(
                object, SimpleNamespace(decomposition=SimpleNamespace(components=None, ci_fn=None))
            ),
        ),
        now_step=0,
        pass_index=0,
        batches=(jnp.asarray(3.0),),
        target_batches=(jnp.asarray(7.0),),
    )
    broad = operation_for("broad").run(targeted)
    target = operation_for("target_data").run(targeted)

    assert broad == {"eval/nontarget_data/l0/0.0_site": 3.0}
    assert target == {"eval/l0/0.0_site": 7.0}
    assert not broad.keys() & target.keys()


def test_narrow_kl_evals_name_arms_the_full_ce_kl_evaluator_also_emits():
    """The narrow metrics exist to cut CE/KL's clutter, NOT to respell its numbers: each
    selects one arm of the same evaluator, so `eval/ce_kl/kl_<arm>` means the same thing
    whether a run authored `CEandKLLosses` or the single-arm config. A new arm name here
    that the full evaluator doesn't emit would silently give one quantity two names."""
    assert {"ci_masked", "unmasked"} <= set(CE_KL_VARIANTS)


def test_the_two_recon_configs_are_authorable_as_evals():
    """`coeff` is what separates the loss role from the eval role (`LossMetricConfig`), so
    both must validate with none — this is the `PGDReconLoss` dual-role pattern."""
    for config in (CIMaskedReconLossConfig(), UnmaskedNoDeltaReconLossConfig()):
        assert config.coeff is None
        assert config.slow is False
