"""Binding of authored evaluation operations for toy targets.

The fast-tier scalars come from the shared target-generic binder
(`experiments/fast_eval_operations.py`); only the UV figures are toy-owned, because they
read the toy's single-feature CI probe.
"""

from collections.abc import Callable

from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.built_run import BuiltRun, TargetSites
from param_decomp.core.configs import CI_L0Config, PGDReconLossConfig, UVPlotsConfig
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import DecomposedModel
from param_decomp.core.run import EvalInvocation, EvalOperation
from param_decomp.core.train import TrainState
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.eval_config import EvalConfig, schedule_for
from param_decomp.experiments.fast_eval_operations import (
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig

type ToyRun[TargetT: TargetSites] = BuiltRun[None, TargetT]
type ProbeCI = Callable[[TrainState], dict[str, Array]]


def _make_uv_plots_operation(
    metric: UVPlotsConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    probe_ci: ProbeCI,
    wandb_configured: bool,
) -> EvalOperation[EvalInvocation]:
    assert wandb_configured, "UVPlots requires a configured wandb transport"
    spec = toy_uv_eval.toy_uv_spec(model, metric)

    def run(context: EvalInvocation) -> LogRecord:
        return toy_uv_eval.render_uv_metric(
            spec,
            dict(context.state.decomposition.components.sites_items()),
            probe_ci(context.state),
        )

    return EvalOperation(schedule, run)


def make_toy_evaluation_operations(
    eval_config: EvalConfig,
    seed: int,
    compiler_options: dict[str, bool | int | str],
    model: DecomposedModel,
    mesh: Mesh,
    sample_eval_batch: Callable[[int], Array],
    probe_ci: ProbeCI,
    wandb_configured: bool,
) -> tuple[EvalOperation[EvalInvocation], ...]:
    """Exhaustively bind each authored toy metric to one executable operation."""
    operations: list[EvalOperation[EvalInvocation]] = []
    for metric in eval_config.metrics:
        schedule = schedule_for(metric, eval_config)
        match metric:
            case PGDReconLossConfig():
                operation = make_fresh_pgd_operation(
                    metric,
                    eval_config,
                    schedule,
                    seed,
                    compiler_options,
                    model,
                    mesh,
                    sample_eval_batch,
                )
            case CI_L0Config():
                operation = make_ci_l0_operation(
                    metric,
                    eval_config,
                    schedule,
                    seed,
                    compiler_options,
                    model,
                    mesh,
                    sample_eval_batch,
                )
            case UVPlotsConfig():
                operation = _make_uv_plots_operation(
                    metric, schedule, model, probe_ci, wandb_configured
                )
            case CEandKLLossesConfig():
                raise AssertionError(
                    "CEandKLLosses scores next-token cross-entropy and KL over a categorical "
                    "output distribution; a toy target emits neither tokens nor logits"
                )
            case _:
                raise AssertionError(f"eval metric {metric.type!r} has no toy binding")
        operations.append(operation)
    return tuple(operations)
