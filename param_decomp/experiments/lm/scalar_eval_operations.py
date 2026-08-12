"""Independent CE/KL, causal-L0, and fresh-PGD LM operations."""

from collections.abc import Callable

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.configs import (
    CI_L0Config,
    PGDReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import BarChart, LogRecord, PNGImage
from param_decomp.core.model import CaptureKeys, DecomposedModel
from param_decomp.core.recon import resolve_reconstruction_spec
from param_decomp.core.recon_eval import FreshPGDReconEval
from param_decomp.core.run import EvalOperation
from param_decomp.experiments.lm.eval import (
    ScalarStep,
    make_ce_kl_step,
    make_ci_l0_step,
    make_fresh_pgd_step,
    make_unmasked_recon_step,
)
from param_decomp.experiments.lm.eval_config import (
    CEandKLLossesConfig,
    TargetPoolScalarsConfig,
)
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.eval_keys import EvalKeyStream


def broad_stream_batches(context: LMEvalContext) -> tuple[Array, ...]:
    return context.batches


def target_pool_batches(context: LMEvalContext) -> tuple[Array, ...]:
    assert context.target_batches is not None, (
        "target-stream metrics need a tPD run's prompt pool; a plain run has no target stream"
    )
    return context.target_batches


def _make_scalar_operation(
    schedule: EvalSchedule,
    step: ScalarStep,
    prefixes: tuple[str, ...],
    model: DecomposedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    batches_of: Callable[[LMEvalContext], tuple[Array, ...]],
    log_prefix: str,
) -> EvalOperation[LMEvalContext]:
    def run(context: LMEvalContext) -> LogRecord:
        sums: dict[str, Array] = {}
        for batch_index, tokens in enumerate(batches_of(context)):
            key = random.fold_in(
                run_key,
                EvalKeyStream.SCALARS * train_steps + context.pass_index * eval_steps + batch_index,
            )
            values = step(
                model,
                context.state.decomposition.components,
                context.state.decomposition.ci_fn,
                tokens,
                key,
            )
            for name, value in values.items():
                if name.startswith(prefixes):
                    sums[name] = sums.get(name, jnp.zeros(())) + value
        return {f"{log_prefix}{name}": float(value) / eval_steps for name, value in sums.items()}

    return EvalOperation(schedule, run)


def make_ce_kl_operation(
    metric: CEandKLLossesConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    scalars = _make_scalar_operation(
        schedule,
        make_ce_kl_step(model, ci_capture_keys, metric.rounding_threshold, mesh, compiler_options),
        ("ce_kl/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        broad_stream_batches,
        "eval/",
    )

    return scalars


def make_ci_l0_operation(
    metric: CI_L0Config,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    groups = (
        {name: tuple(patterns) for name, patterns in metric.groups.items()}
        if metric.groups is not None
        else None
    )
    scalars = _make_scalar_operation(
        schedule,
        make_ci_l0_step(
            model, ci_capture_keys, metric.ci_alive_threshold, groups, mesh, compiler_options
        ),
        ("l0/",),
        model,
        run_key,
        train_steps,
        eval_steps,
        broad_stream_batches,
        "eval/",
    )

    def run(context: LMEvalContext) -> LogRecord:
        record = dict(scalars.run(context))
        prefix = f"eval/l0/{metric.ci_alive_threshold}_"
        record["eval/l0/bar_chart"] = BarChart(
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

    return EvalOperation(schedule, run)


def make_unmasked_recon_operation(
    metric: UnmaskedReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    assert metric.hidden_acts_reconstruction is None, (
        "the S35 rider belongs on a TRAINING term; the unmasked probe reports e2e recon only"
    )
    return _make_scalar_operation(
        schedule,
        make_unmasked_recon_step(model, ci_capture_keys, mesh, compiler_options),
        ("UnmaskedReconLoss",),
        model,
        run_key,
        train_steps,
        eval_steps,
        broad_stream_batches,
        "eval/",
    )


def make_target_pool_scalars_operation(
    metric: TargetPoolScalarsConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    """CI-L0 (+ optional fresh PGD) over the tPD prompt pool, logged under
    `eval/target_pool/` so the broad-stream twins stay readable side by side."""
    groups = (
        {name: tuple(patterns) for name, patterns in metric.ci_l0_groups.items()}
        if metric.ci_l0_groups is not None
        else None
    )
    parts = [
        _make_scalar_operation(
            schedule,
            make_ci_l0_step(
                model, ci_capture_keys, metric.ci_alive_threshold, groups, mesh, compiler_options
            ),
            ("l0/",),
            model,
            run_key,
            train_steps,
            eval_steps,
            target_pool_batches,
            "eval/target_pool/",
        )
    ]
    if metric.fresh_pgd is not None:
        probe = FreshPGDReconEval(
            name="PGDReconLoss",
            n_steps=metric.fresh_pgd.n_steps,
            step_size=metric.fresh_pgd.step_size,
            reconstruction=resolve_reconstruction_spec(None),
        )
        parts.append(
            _make_scalar_operation(
                schedule,
                make_fresh_pgd_step(model, ci_capture_keys, probe, mesh, compiler_options),
                (f"loss/{probe.name}",),
                model,
                run_key,
                train_steps,
                eval_steps,
                target_pool_batches,
                "eval/target_pool/",
            )
        )

    def run(context: LMEvalContext) -> LogRecord:
        record: dict[str, float | BarChart | PNGImage] = {}
        for part in parts:
            record.update(part.run(context))
        return record

    return EvalOperation(schedule, run)


def make_fresh_pgd_operation(
    metric: PGDReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    eval_steps: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    assert metric.init == "random" and metric.source_shape == "c", metric
    probe = FreshPGDReconEval(
        name=metric.name or metric.type,
        n_steps=metric.n_steps,
        step_size=metric.step_size,
        reconstruction=resolve_reconstruction_spec(metric.hidden_acts_reconstruction),
    )
    return _make_scalar_operation(
        schedule,
        make_fresh_pgd_step(model, ci_capture_keys, probe, mesh, compiler_options),
        (f"loss/{probe.name}",),
        model,
        run_key,
        train_steps,
        eval_steps,
        broad_stream_batches,
        "eval/",
    )
