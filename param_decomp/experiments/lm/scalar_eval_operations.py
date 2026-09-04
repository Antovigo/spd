"""Independent CE/KL, causal-L0, and fresh-PGD LM operations over the shared batch context.

Every operation is bound to ONE `(stream, CI role)`: it folds only the pass's contexts on
its stream, reads its role's head of the shared CI envelope, and reports under that
stream's namespace (`stream_log_prefix`). A targeted run's non-target stream composes
its recon scorers delta-pinned (`nontarget_delta_pinned`, SPEC T4).
"""

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.ci_fn import CIRole
from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.metrics import BarChart, LogRecord
from param_decomp.core.model import EMPTY_CAPTURE_KEYS, CaptureKeys, PlacedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import BatchedOperation, batched_operation
from param_decomp.experiments.lm.eval import (
    MaskingArm,
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
    Stream,
    nontarget_delta_pinned,
    prepared_batch_from_context,
    role_log_segment,
    stream_batches,
    stream_log_prefix,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream

__all__ = [
    "Stream",
    "fresh_pgd_probe",
    "make_ce_kl_operation",
    "make_ci_l0_operation",
    "make_fresh_pgd_operation",
    "make_masked_kl_operation",
    "nontarget_delta_pinned",
    "role_log_segment",
    "scalar_step_for",
    "stream_batches",
    "stream_log_prefix",
]

type AnyScalarMetricConfig = CEandKLLossesConfig | CI_L0Config | PGDReconLossConfig


def fresh_pgd_probe(metric: PGDReconLossConfig) -> FreshPGDReconEval:
    assert metric.init == "random" and metric.source_shape == "c", metric
    return FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_reconstruction_spec(metric.hidden_acts_reconstruction),
    )


def scalar_step_for(
    metric: AnyScalarMetricConfig,
    model: PlacedModel,
    ci_capture_keys: CaptureKeys,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str] | None,
    role: CIRole = "output",
    delta_pinned: bool = False,
) -> ScalarStep:
    """THE config→kernel binding for the fused scalar steps — the AOT eval fit check
    compiles the same kernels the batched operations below score with, from one spelling.

    `delta_pinned` (SPEC T4) reaches only the CE/KL family; the fit check leaves it at
    the default, which is also the memory-conservative arm — pinning REPLACES the
    stochastic delta draw with a constant, so the unpinned step is the larger one."""
    match metric:
        case CEandKLLossesConfig():
            return make_ce_kl_step(
                model,
                ci_capture_keys,
                metric.rounding_threshold,
                mesh,
                compiler_options,
                delta_pinned=delta_pinned,
            )
        case CI_L0Config():
            return make_ci_l0_step(
                model,
                ci_capture_keys,
                metric.ci_alive_threshold,
                _l0_groups(metric),
                mesh,
                compiler_options,
                role=role,
            )
        case PGDReconLossConfig():
            return make_fresh_pgd_step(
                model, ci_capture_keys, fresh_pgd_probe(metric), mesh, compiler_options, role=role
            )


def _l0_groups(metric: CI_L0Config) -> dict[str, tuple[str, ...]] | None:
    return (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )


def _make_scalar_operation(
    schedule: EvalSchedule,
    scorer: ScalarScorer,
    prefixes: tuple[str, ...],
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    compiler_options: dict[str, bool | int | str],
    stream: Stream,
    role: CIRole,
    hidden_acts_capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
    max_batches: int | None = None,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    """Fold `scorer` over the pass's contexts on `stream` (the first `max_batches` when
    capped) and average each scalar it emits. The RNG key stays keyed off the full
    `eval_steps` stride, so a cap changes which batches run, never which key a batch draws."""
    score_step = filter_jit(scorer, compiler_options=compiler_options)
    n_batches = eval_steps if max_batches is None else min(max_batches, eval_steps)

    def init() -> dict[str, Array]:
        return {}

    def update(sums: dict[str, Array], context: LMBatchContext) -> dict[str, Array]:
        if context.stream != stream or context.batch_index >= n_batches:
            return sums
        key = random.fold_in(
            run_key,
            EvalKeyStream.SCALARS * train_steps
            + context.pass_index * eval_steps
            + context.batch_index,
        )
        values = score_step(
            model, prepared_batch_from_context(context, role, hidden_acts_capture_keys), key
        )
        folded = dict(sums)
        for name, value in values.items():
            if name.startswith(prefixes):
                folded[name] = folded.get(name, jnp.zeros(())) + value
        return folded

    def finish(eval_pass: LMEvalPass, sums: dict[str, Array]) -> LogRecord:
        log_prefix = stream_log_prefix(stream, eval_pass.targeted, role)
        return {f"{log_prefix}{name}": float(value) / n_batches for name, value in sums.items()}

    return batched_operation(schedule, init, update, finish)


def make_ce_kl_operation(
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    stream: Stream,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    role: CIRole,
    *,
    targeted: bool,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    assert role == "output", "CE/KL's step reads the output head's CI; it has no hidden-role form"
    return _make_scalar_operation(
        schedule,
        make_ce_kl_scorer(
            model,
            metric.rounding_threshold,
            mesh,
            delta_pinned=nontarget_delta_pinned(targeted=targeted, stream=stream),
        ),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        stream,
        role,
    )


def make_masked_kl_operation(
    arm: MaskingArm,
    schedule: EvalSchedule,
    stream: Stream,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    role: CIRole,
    *,
    targeted: bool,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    """ONE masking arm, authored as the loss config that names the same construction —
    `CIMaskedReconLoss` / `UnmaskedNoDeltaReconLoss`, as `PGDReconLoss` already is."""
    return _make_scalar_operation(
        schedule,
        make_masked_kl_scorer(
            model, arm, mesh, delta_pinned=nontarget_delta_pinned(targeted=targeted, stream=stream)
        ),
        (f"ce_kl/kl_{arm}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        stream,
        role,
    )


def make_ci_l0_operation(
    metric: CI_L0Config,
    schedule: EvalSchedule,
    stream: Stream,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    role: CIRole,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    del mesh  # CI L0 is a reduction of the shared envelope; nothing to shard
    scalars = _make_scalar_operation(
        schedule,
        make_ci_l0_scorer(model, metric.ci_alive_threshold, _l0_groups(metric)),
        ("l0/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        stream,
        role,
    )

    def finish(eval_pass: LMEvalPass, sums: dict[str, Array]) -> LogRecord:
        record = dict(scalars.finish(eval_pass, sums))
        log_prefix = stream_log_prefix(stream, eval_pass.targeted, role)
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

    return BatchedOperation(schedule, scalars.init, scalars.update, finish)


def make_fresh_pgd_operation(
    metric: PGDReconLossConfig,
    schedule: EvalSchedule,
    stream: Stream,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
    role: CIRole,
    *,
    targeted: bool,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    probe = fresh_pgd_probe(metric)
    # Unpinned (plain run / target stream), the probe's ascent owns a live delta channel
    # it can drive to 0 — pinned, it measures the component-only worst case.
    delta_pinned = nontarget_delta_pinned(targeted=targeted, stream=stream)
    return _make_scalar_operation(
        schedule,
        make_fresh_pgd_scorer(model, probe, mesh, delta_pinned=delta_pinned),
        (f"loss/{probe.name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        compiler_options,
        stream,
        role,
        hidden_acts_capture_keys=probe.hidden_acts_capture_keys,
        max_batches=metric.n_batches,
    )
