"""LM evaluation operation binding and execution."""

from collections.abc import Callable

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.configs import (
    CI_L0Config,
    CIHiddenActsReconLossConfig,
    CIHistogramsConfig,
    CIMaskedReconLossConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    PGDReconLossConfig,
    StochasticHiddenActsReconLossConfig,
    UnmaskedNoDeltaReconLossConfig,
    UVPlotsConfig,
    WeightMagnitudeConfig,
    WellTemperednessConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.model import DecomposedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalOperation,
    Evaluation,
    MetricsSink,
)
from param_decomp.core.train import TrainState
from param_decomp.core.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.experiments.eval_config import AnyEvalMetricConfig, EvalConfig, schedule_for
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_hidden_acts_operation,
    make_permutation_operation,
    make_site_figures_operation,
    make_two_stream_ci_mean_operation,
    make_weight_magnitude_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
    TwoStreamCIMeanPerComponentConfig,
)
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.resolved import LMAnyRun
from param_decomp.experiments.lm.scalar_eval_operations import (
    Stream,
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
    make_masked_kl_operation,
    stream_batches,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


def global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(("replicate", "fsdp")))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def make_lm_evaluation(
    built: LMAnyRun,
    eval: EvalConfig,
    model: DecomposedModel,
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
    target_pool_batches_for: Callable[[int], list[jax.Array]] | None = None,
) -> Evaluation[LMEvalContext]:
    """Construct the executable operations for every authored LM metric — one PER STREAM
    the metric measures.

    `target_pool_batches_for(pass_index)` supplies the tPD target stream, which `data.eval`
    cannot: the prompt pool has no held-out split, so the targeted root draws pool batches
    the same way training does. `None` on a plain run, and that is what makes a plain run's
    metric set — and every one of its log keys — exactly what it was before targeted runs
    existed: `data_streams` collapses to the single nontarget stream."""
    pd = built.pd
    capture_inputs = built.ci_fn.capture_keys
    data = built.data
    schedule = BatchSchedule(scan_shards(data.eval_dir), eval.batch_size, pd.seed + 1)
    seq_len = read_dataset_meta(data.eval_dir).seq_len
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0
    renderer = BackgroundRenderer(sink)

    def batches(pass_index: int) -> list[jax.Array]:
        return [
            global_token_batch(
                server.local_batch(pass_index * eval.n_steps + j), mesh, eval.batch_size
            )
            for j in range(eval.n_steps)
        ]

    targeted = target_pool_batches_for is not None
    # Every stream a metric authored for BOTH measures; a plain run has exactly one.
    data_streams: tuple[Stream, ...] = ("nontarget", "target") if targeted else ("nontarget",)
    # The stream the run optimizes for, and the DEFAULT for any single-stream eval: a
    # diagnostic you read once should describe the data you are interpreting. On a plain run
    # this IS the nontarget stream, so every such metric keeps the keys and data it had.
    optimized_stream: Stream = "target" if targeted else "nontarget"

    def well_temperedness_inputs(
        context: LMEvalContext,
    ) -> tuple[jax.Array, PRNGKeyArray]:
        return stream_batches(optimized_stream, context)[0], jax.random.fold_in(
            run_key, EvalKeyStream.WELL_TEMPEREDNESS * pd.steps + context.pass_index
        )

    def per_stream(
        maker: Callable[..., EvalOperation[LMEvalContext]],
        metric: object,
        schedule: EvalSchedule,
        streams: tuple[Stream, ...],
    ) -> tuple[EvalOperation[LMEvalContext], ...]:
        """One operation per stream. The scalar makers share a signature exactly so the
        stream can be the only thing that varies between a metric's two readouts."""
        return tuple(
            maker(
                metric,
                schedule,
                stream,
                model,
                capture_inputs,
                run_key,
                pd.steps,
                eval.n_steps,
                mesh,
                compiler_options,
            )
            for stream in streams
        )

    def make_operations(metric: AnyEvalMetricConfig) -> tuple[EvalOperation[LMEvalContext], ...]:
        schedule = schedule_for(metric, eval)
        match metric:
            case CEandKLLossesConfig():
                return per_stream(make_ce_kl_operation, metric, schedule, data_streams)
            case CIMaskedReconLossConfig():
                return per_stream(make_masked_kl_operation, "ci_masked", schedule, data_streams)
            case UnmaskedNoDeltaReconLossConfig():
                # The non-target pass's OWN training term, already reported as a train loss
                # there; measuring it again off-target would restate the objective.
                return per_stream(
                    make_masked_kl_operation, "unmasked", schedule, (optimized_stream,)
                )
            case CI_L0Config():
                return per_stream(make_ci_l0_operation, metric, schedule, data_streams)
            case PGDReconLossConfig():
                return per_stream(make_fresh_pgd_operation, metric, schedule, data_streams)

            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return (
                    make_attention_operation(
                        metric,
                        schedule,
                        model,
                        capture_inputs,
                        run_key,
                        pd.steps,
                        compiler_options,
                        optimized_stream,
                    ),
                )
            case CIHiddenActsReconLossConfig() | StochasticHiddenActsReconLossConfig():
                return (
                    make_hidden_acts_operation(
                        metric,
                        schedule,
                        model,
                        capture_inputs,
                        run_key,
                        pd.steps,
                        compiler_options,
                        optimized_stream,
                    ),
                )
            case (
                CIHistogramsConfig()
                | ComponentActivationDensityConfig()
                | CIMeanPerComponentConfig()
            ):
                return (
                    make_site_figures_operation(
                        metric,
                        schedule,
                        model,
                        capture_inputs,
                        compiler_options,
                        renderer,
                        optimized_stream,
                    ),
                )
            case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                return (
                    make_permutation_operation(
                        metric,
                        schedule,
                        model,
                        capture_inputs,
                        compiler_options,
                        renderer,
                        optimized_stream,
                    ),
                )

            case WellTemperednessConfig():
                return (
                    make_well_temperedness_operation(
                        metric,
                        schedule,
                        model,
                        capture_inputs,
                        mesh,
                        compiler_options,
                        inputs_for_context=well_temperedness_inputs,
                        figure_rendering=renderer if sink.accepts_deferred_media else None,
                    ),
                )

            case ArithmeticCIGridConfig():
                return (
                    make_arithmetic_operation(
                        metric,
                        schedule,
                        built.target,
                        model,
                        capture_inputs,
                        mesh,
                        n_proc,
                        sink,
                        run_key,
                        pd.steps,
                        compiler_options,
                    ),
                )
            case TwoStreamCIMeanPerComponentConfig():
                return (
                    make_two_stream_ci_mean_operation(
                        schedule, model, capture_inputs, compiler_options, renderer
                    ),
                )
            case WeightMagnitudeConfig():
                return (make_weight_magnitude_operation(schedule, renderer),)

    needs_target_stream = tuple(
        metric.type
        for metric in eval.metrics
        if isinstance(metric, TwoStreamCIMeanPerComponentConfig)
    )
    assert not needs_target_stream or targeted, (
        f"{needs_target_stream} measure the tPD target stream; a plain run has no prompt pool"
    )
    authored = {type(metric) for metric in eval.metrics}
    assert not {TwoStreamCIMeanPerComponentConfig, CIMeanPerComponentConfig} <= authored, (
        "TwoStreamCIMeanPerComponent already computes the nontarget-stream reduction "
        "CIMeanPerComponent does, so authoring both pays for that pass twice"
    )
    # The single-arm evals ARE arms of `CEandKLLosses`, under the same keys. Authoring both
    # is a duplicate measurement that `_run_due_evaluation`'s collision assert would catch
    # at the first eval pass — hours into a run. Catch it while reading the config.
    single_arm = tuple(
        metric.type
        for metric in eval.metrics
        if isinstance(metric, CIMaskedReconLossConfig | UnmaskedNoDeltaReconLossConfig)
    )
    assert not (single_arm and any(isinstance(m, CEandKLLossesConfig) for m in eval.metrics)), (
        f"{single_arm} emit arms CEandKLLosses already emits, under the same `ce_kl/kl_*` "
        "keys: author the narrow metrics OR CEandKLLosses, not both"
    )
    operations = tuple(
        operation for metric in eval.metrics for operation in make_operations(metric)
    )

    def make_context(state: TrainState, now_step: int) -> LMEvalContext:
        pass_index = now_step // eval.every
        return LMEvalContext(
            state=state,
            now_step=now_step,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
            target_batches=(
                None
                if target_pool_batches_for is None
                else tuple(target_pool_batches_for(pass_index))
            ),
        )

    return Evaluation(operations, make_context)
