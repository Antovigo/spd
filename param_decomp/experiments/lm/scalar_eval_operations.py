"""Independent CE/KL, causal-L0, masked-KL and fresh-PGD LM operations over the shared batch
context.

Every operation is bound to ONE `(stream, CI role)` by its caller: it reads its role's head
of the shared CI envelope and reports under `log_prefix`, that stream's and role's
namespace (`eval_context.stream_log_prefix`); `eval_operations` feeds it only its stream's
contexts. A targeted run's non-target stream composes the recon scorers delta-pinned.
"""

import jax
import numpy as np
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray
from numpy.typing import NDArray

from param_decomp.core.ci_fn.interface import CIRole
from param_decomp.core.configs import (
    AnyPGDEvalConfig,
    CI_L0Config,
    EvalPGDReconLossConfig,
    SlowPGDReconLossConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ComponentActivations,
    PlacedModel,
)
from param_decomp.core.recon_eval import fresh_pgd_probe
from param_decomp.core.run import (
    SharedForwardOperation,
    SharedForwardOperationPlan,
    shared_forward_operation,
)
from param_decomp.experiments.lm.eval import (
    MaskingArm,
    PreparedLMBatch,
    ScalarScorer,
    ScalarStep,
    make_ce_kl_scorer,
    make_ce_kl_step,
    make_ci_l0_scorer,
    make_ci_l0_step,
    make_fresh_pgd_scorer,
    make_fresh_pgd_step,
    make_masked_kl_scorer,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    prepared_batch_from_context,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput

type AnyScalarMetricConfig = CEandKLLossesConfig | CI_L0Config | AnyPGDEvalConfig


def _ci_l0_groups(metric: CI_L0Config) -> dict[str, tuple[str, ...]] | None:
    return (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )


def scalar_step_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyScalarMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    role: CIRole = "output",
) -> ScalarStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """The scalar tier's STANDALONE step (own clean forward) — what the AOT eval fit check
    compiles; the operations below score the pass's shared context via `scalar_scorer_for`."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_step(model, ci_capture_keys, metric.rounding_threshold, mesh)
        case CI_L0Config():
            return make_ci_l0_step(
                model,
                ci_capture_keys,
                metric.ci_alive_threshold,
                _ci_l0_groups(metric),
                mesh,
                role=role,
            )
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return make_fresh_pgd_step(
                model, ci_capture_keys, fresh_pgd_probe(metric), mesh, role=role
            )


def scalar_scorer_for[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyScalarMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    mesh: Mesh,
    *,
    delta_pinned: bool = False,
) -> ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """THE config→scorer binding for the scalar tier: the pure scorer each operation jits
    over the pass's shared batch context, and what the trace gate lowers. `delta_pinned`
    reaches the recon families (CE/KL, fresh PGD); CI L0 reads the envelope alone."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_scorer(
                model, metric.rounding_threshold, mesh, delta_pinned=delta_pinned
            )
        case CI_L0Config():
            return make_ci_l0_scorer(model, metric.ci_alive_threshold, _ci_l0_groups(metric))
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return make_fresh_pgd_scorer(
                model, fresh_pgd_probe(metric), mesh, delta_pinned=delta_pinned
            )


def _make_scalar_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    schedule: EvalSchedule,
    scorer: ScalarScorer[TargetIn, PreparedT, Conditioning, PreparedMaskingT],
    prefixes: tuple[str, ...],
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    compiler_options: dict[str, bool | int | str],
    reconstruction_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
    *,
    role: CIRole = "output",
    log_prefix: str = "eval/",
    max_batches: int | None = None,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    """Fold `scorer` over the contexts it is fed (the first `max_batches` when capped) and
    average each scalar it emits. The RNG key stays keyed off the full `eval_steps` stride,
    so a cap changes which batches run, never which key a batch draws."""
    n_batches = eval_steps if max_batches is None else min(max_batches, eval_steps)

    def prepare(
        _example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        def score(
            model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
            batch: PreparedLMBatch[TargetIn, PreparedT, Conditioning],
            pass_index: NDArray[np.uint32],
            batch_index: NDArray[np.uint32],
        ) -> dict[str, Array]:
            key = random.fold_in(
                run_key, EvalKeyStream.SCALARS * train_steps + pass_index * eval_steps + batch_index
            )
            return scorer(model, batch, key)

        score_step = (
            jax.jit(score, compiler_options=compiler_options)
            .lower(
                model,
                prepared_batch_from_context(example_context, reconstruction_capture_keys, role),
                np.zeros((), np.uint32),
                np.zeros((), np.uint32),
            )
            .compile()
        )

        def init() -> dict[str, np.ndarray]:
            return {}

        def update(
            sums: dict[str, np.ndarray],
            context: LMBatchContext[TargetIn, PreparedT, Conditioning],
        ) -> dict[str, np.ndarray]:
            if context.batch_index >= n_batches:
                return sums
            values = score_step(
                model,
                prepared_batch_from_context(context, reconstruction_capture_keys, role),
                np.asarray(context.pass_index, np.uint32),
                np.asarray(context.batch_index, np.uint32),
            )
            folded = dict(sums)
            for name, value in values.items():
                if name.startswith(prefixes):
                    folded[name] = folded.get(name, np.zeros((), np.float32)) + np.asarray(value)
            return folded

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], sums: dict[str, np.ndarray]
        ) -> LogRecord:
            del eval_pass
            return {f"{log_prefix}{name}": float(value) / n_batches for name, value in sums.items()}

        return shared_forward_operation(schedule, init, update, finish)

    return SharedForwardOperationPlan(prepare)


def make_ce_kl_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    *,
    log_prefix: str = "eval/",
    delta_pinned: bool = False,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    """CE/KL reads the OUTPUT head's CI; it has no hidden-role form."""
    return _make_scalar_operation(
        schedule,
        scalar_scorer_for(metric, model, mesh, delta_pinned=delta_pinned),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        log_prefix=log_prefix,
    )


def make_masked_kl_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    arm: MaskingArm,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    *,
    role: CIRole = "output",
    log_prefix: str = "eval/",
    delta_pinned: bool = False,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    """ONE masking arm, authored under the loss config that names the same construction —
    `CIMaskedReconLoss` / `UnmaskedNoDeltaReconLoss`."""
    return _make_scalar_operation(
        schedule,
        make_masked_kl_scorer(model, arm, mesh, delta_pinned=delta_pinned),
        (f"ce_kl/kl_{arm}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        role=role,
        log_prefix=log_prefix,
    )


def make_ci_l0_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: CI_L0Config,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    *,
    role: CIRole = "output",
    log_prefix: str = "eval/",
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    def prepare(
        example_pass: LMEvalPass[TargetIn, Conditioning],
        example_context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    ) -> SharedForwardOperation[
        LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
    ]:
        scalars = _make_scalar_operation(
            schedule,
            scalar_scorer_for(metric, model, mesh),
            ("l0/",),
            model,
            run_key,
            train_steps,
            eval_steps,
            compiler_options,
            role=role,
            log_prefix=log_prefix,
        )

        prepared_scalars = scalars.prepare(example_pass, example_context)

        def finish(
            eval_pass: LMEvalPass[TargetIn, Conditioning], sums: dict[str, np.ndarray]
        ) -> LogRecord:
            record = dict(prepared_scalars.finish(eval_pass, sums))
            prefix = f"{log_prefix}l0/{metric.ci_alive_threshold}_"
            record[f"{log_prefix}l0/bar_chart"] = BarChart(
                rows=tuple(
                    (name.removeprefix(prefix), value)
                    for name, value in record.items()
                    if name.startswith(prefix) and isinstance(value, float)
                ),
                x_label="layer",
                y_label="l0",
                title=f"L0_{metric.ci_alive_threshold}",
            )
            return record

        return SharedForwardOperation(
            schedule, prepared_scalars.init, prepared_scalars.update, finish
        )

    return SharedForwardOperationPlan(prepare)


def make_fresh_pgd_operation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyPGDEvalConfig,
    schedule: EvalSchedule,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    *,
    role: CIRole = "output",
    log_prefix: str = "eval/",
    delta_pinned: bool = False,
) -> SharedForwardOperationPlan[
    LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
]:
    """Unpinned (plain run / target stream), the probe's ascent owns a live delta channel it
    can drive to 0 — pinned, it measures the component-only worst case."""
    probe = fresh_pgd_probe(metric)
    return _make_scalar_operation(
        schedule,
        scalar_scorer_for(metric, model, mesh, delta_pinned=delta_pinned),
        tuple(f"loss/{name}" for name in probe.read_out_names),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        reconstruction_capture_keys=probe.reconstruction_capture_keys,
        role=role,
        log_prefix=log_prefix,
        max_batches=metric.n_batches,
    )
