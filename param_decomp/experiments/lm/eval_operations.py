"""LM evaluation operation binding and execution."""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.configs import (
    CI_L0Config,
    CIHiddenActsReconLossConfig,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    PGDReconLossConfig,
    StochasticHiddenActsReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.model import DecomposedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalOperation,
    Evaluation,
    MetricsSink,
)
from param_decomp.core.train import TrainState
from param_decomp.experiments.eval_config import AnyEvalMetricConfig, EvalConfig, schedule_for
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_hidden_acts_operation,
    make_permutation_operation,
    make_site_figures_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.resolved import LMRun
from param_decomp.experiments.lm.scalar_eval_operations import (
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


def global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replicate", "fsdp")))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def make_lm_evaluation(
    built: LMRun,
    eval: EvalConfig,
    model: DecomposedModel,
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
) -> Evaluation[LMEvalContext]:
    """Construct one executable operation for every authored LM metric."""
    pd = built.pd
    data = built.data
    schedule = BatchSchedule(scan_shards(data.eval_dir), eval.batch_size, pd.seed + 1)
    server = ShardServer(
        schedule, read_dataset_meta(data.eval_dir).seq_len, jax.process_index(), n_proc
    )
    assert server.per_process % jax.local_device_count() == 0
    renderer = BackgroundRenderer(sink)

    def batches(pass_index: int) -> list[jax.Array]:
        return [
            global_token_batch(
                server.local_batch(pass_index * eval.n_steps + j), mesh, eval.batch_size
            )
            for j in range(eval.n_steps)
        ]

    def make_operation(metric: AnyEvalMetricConfig) -> EvalOperation[LMEvalContext]:
        schedule = schedule_for(metric, eval)
        match metric:
            case CEandKLLossesConfig():
                return make_ce_kl_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )
            case CI_L0Config():
                return make_ci_l0_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )
            case PGDReconLossConfig():
                return make_fresh_pgd_operation(
                    metric,
                    schedule,
                    model,
                    run_key,
                    pd.steps,
                    eval.n_steps,
                    mesh,
                    compiler_options,
                )

            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return make_attention_operation(
                    metric, schedule, model, run_key, pd.steps, compiler_options
                )
            case CIHiddenActsReconLossConfig() | StochasticHiddenActsReconLossConfig():
                return make_hidden_acts_operation(
                    metric, schedule, model, run_key, pd.steps, compiler_options
                )
            case (
                CIHistogramsConfig()
                | ComponentActivationDensityConfig()
                | CIMeanPerComponentConfig()
            ):
                return make_site_figures_operation(
                    metric, schedule, model, compiler_options, renderer
                )
            case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                return make_permutation_operation(
                    metric, schedule, model, compiler_options, renderer
                )

            case ArithmeticCIGridConfig():
                return make_arithmetic_operation(
                    metric,
                    schedule,
                    built.target,
                    model,
                    mesh,
                    n_proc,
                    sink,
                    run_key,
                    pd.steps,
                    compiler_options,
                )

    operations = tuple(make_operation(metric) for metric in eval.metrics)

    def make_context(state: TrainState, now_step: int) -> LMEvalContext:
        pass_index = now_step // eval.every
        return LMEvalContext(
            state=state,
            now_step=now_step,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
        )

    return Evaluation(operations, make_context)
