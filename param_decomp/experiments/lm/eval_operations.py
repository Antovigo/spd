"""LM evaluation operation binding and execution.

Binding is two closed passes over the authored metrics: first each metric declares its
clean-capture demand on the shared batch context (`_clean_capture_demand`), then each
binds its operations — one PER STREAM the metric measures, and per CI role on a dual
run. The pass's one context step captures the union of those demands, so every batched
operation reads one clean forward + CI envelope per batch of each stream.
"""

from collections.abc import Callable
from functools import partial

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn import DUAL_CI_ROLES, CIRole
from param_decomp.core.components import nonlinearity_partitions
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMaskedReconLossConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    PGDReconLossConfig,
    UnmaskedNoDeltaReconLossConfig,
    UVPlotsConfig,
    WeightMagnitudeConfig,
    WellTemperednessConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.model import BATCH_AXES, EMPTY_CAPTURE_KEYS, CaptureKeys, PlacedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalInvocation,
    EvalOperation,
    Evaluation,
    MetricsSink,
)
from param_decomp.core.sharding import local_data_parallel_size
from param_decomp.core.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.experiments.eval_config import (
    AnyEvalMetricConfig,
    EvalConfig,
    schedule_for,
    slow_schedule,
)
from param_decomp.experiments.lm.ab_grid_operation import make_ab_grid_operation
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.attn_patterns_eval import attn_output_key_by_site
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_nonlinearity_operation,
    make_permutation_operation,
    make_site_figures_operation,
    make_two_stream_ci_mean_operation,
    make_weight_magnitude_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ABGridDatasetConfig,
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
    TwoStreamCIMeanPerComponentConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    Stream,
    make_lm_batch_context_step,
    make_lm_batch_contexts,
    stream_batches,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream
from param_decomp.experiments.lm.resolved import LMAnyRun
from param_decomp.experiments.lm.scalar_eval_operations import (
    fresh_pgd_probe,
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
    make_masked_kl_operation,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards

type LMOperation = EvalOperation[LMEvalPass, LMBatchContext]


def global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P(BATCH_AXES))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def make_lm_evaluation(
    built: LMAnyRun,
    eval: EvalConfig,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
    target_pool_batches_for: Callable[[int], list[jax.Array]] | None = None,
) -> Evaluation[LMEvalPass, LMBatchContext]:
    """Construct the executable operations for every authored LM metric — one PER STREAM
    the metric measures.

    `target_pool_batches_for(pass_index)` supplies the tPD target stream, which `data.eval`
    cannot. `None` on a plain run, which collapses every metric to the one stream it has."""
    pd = built.pd
    capture_inputs = built.ci_fn.capture_keys
    data = built.data
    schedule = BatchSchedule(scan_shards(data.eval_dir), eval.batch_size, pd.seed + 1)
    seq_len = read_dataset_meta(data.eval_dir).seq_len
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    assert server.per_process % local_data_parallel_size(mesh) == 0
    renderer = BackgroundRenderer(sink)

    def batches(pass_index: int) -> list[jax.Array]:
        return [
            global_token_batch(
                server.local_batch(pass_index * eval.n_steps + j), mesh, eval.batch_size
            )
            for j in range(eval.n_steps)
        ]

    targeted = target_pool_batches_for is not None
    data_streams: tuple[Stream, ...] = ("nontarget", "target") if targeted else ("nontarget",)
    optimized_stream: Stream = "target" if targeted else "nontarget"
    # A dual CI fn (SPEC S37) answers every CI-reading metric TWICE — once per readout
    # head — because the two heads are the experiment: a component important for the
    # output objective and not the hidden one (or the reverse) is exactly what a dual run
    # is looking for, and reporting one head would hide it. A single-role run keeps one
    # readout under unchanged keys.
    ci_roles: tuple[CIRole, ...] = DUAL_CI_ROLES if built.ci_fn.dual else ("output",)
    ce_kl_authored = any(isinstance(m, CEandKLLossesConfig) for m in eval.metrics)

    def well_temperedness_inputs(
        eval_pass: LMEvalPass,
    ) -> tuple[jax.Array, PRNGKeyArray]:
        return stream_batches(optimized_stream, eval_pass)[0], jax.random.fold_in(
            run_key, EvalKeyStream.WELL_TEMPEREDNESS * pd.steps + eval_pass.pass_index
        )

    def _clean_capture_demand(metric: AnyEvalMetricConfig) -> CaptureKeys:
        """What this metric reads off the shared clean forward beyond the CI taps."""
        match metric:
            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return frozenset(attn_output_key_by_site(model).values())
            case PGDReconLossConfig():
                return fresh_pgd_probe(metric).hidden_acts_capture_keys
            case _:
                return EMPTY_CAPTURE_KEYS

    def per_stream(
        maker: Callable[..., LMOperation],
        metric: object,
        schedule: EvalSchedule,
        streams: tuple[Stream, ...],
        roles: tuple[CIRole, ...] = ("output",),
    ) -> tuple[LMOperation, ...]:
        """One operation per (stream, CI role); the scalar makers share a signature so those
        two are the only things that vary between a metric's readouts."""
        return tuple(
            maker(
                metric,
                schedule,
                stream,
                model,
                run_key,
                pd.steps,
                eval.n_steps,
                mesh,
                compiler_options,
                role,
            )
            for stream in streams
            for role in roles
        )

    def make_operations(metric: AnyEvalMetricConfig) -> tuple[LMOperation, ...]:
        schedule = schedule_for(metric, eval)
        match metric:
            case CEandKLLossesConfig():
                return per_stream(
                    partial(make_ce_kl_operation, targeted=targeted), metric, schedule, data_streams
                )
            case CIMaskedReconLossConfig():
                # CEandKLLosses already emits the output-role arm under the same
                # `ce_kl/kl_ci_masked` key; keep only the roles it cannot measure.
                masked_roles: tuple[CIRole, ...] = (
                    tuple(role for role in ci_roles if role != "output")
                    if ce_kl_authored
                    else ci_roles
                )
                return per_stream(
                    partial(make_masked_kl_operation, targeted=targeted),
                    "ci_masked",
                    schedule,
                    data_streams,
                    masked_roles,
                )
            case UnmaskedNoDeltaReconLossConfig():
                # The nontarget pass's own training term, already reported there as a loss.
                return per_stream(
                    partial(make_masked_kl_operation, targeted=targeted),
                    "unmasked",
                    schedule,
                    (optimized_stream,),
                )
            case CI_L0Config():
                return per_stream(make_ci_l0_operation, metric, schedule, data_streams, ci_roles)
            case PGDReconLossConfig():
                return per_stream(
                    partial(make_fresh_pgd_operation, targeted=targeted),
                    metric,
                    schedule,
                    data_streams,
                    ci_roles,
                )

            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return (
                    make_attention_operation(
                        metric,
                        schedule,
                        model,
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
                return tuple(
                    make_site_figures_operation(
                        metric, schedule, compiler_options, renderer, optimized_stream, role
                    )
                    for role in ci_roles
                )
            case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                return (
                    make_permutation_operation(
                        metric, schedule, model, compiler_options, renderer, optimized_stream
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
            case ABGridDatasetConfig():
                return (
                    make_ab_grid_operation(
                        metric,
                        schedule,
                        built.target,
                        model,
                        capture_inputs,
                        mesh,
                        n_proc,
                        built.run.run_dir,
                        # ONE operation, every role: the grids come off a single frozen
                        # forward, so the second head costs its readout and nothing else.
                        ci_roles,
                    ),
                )
            case TwoStreamCIMeanPerComponentConfig():
                return (make_two_stream_ci_mean_operation(schedule, compiler_options, renderer),)
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
    if ce_kl_authored:
        assert not any(isinstance(m, UnmaskedNoDeltaReconLossConfig) for m in eval.metrics), (
            "UnmaskedNoDeltaReconLoss emits the arm CEandKLLosses already emits, under the "
            "same `ce_kl/kl_unmasked` key: author it OR CEandKLLosses, not both"
        )
        assert not (
            any(isinstance(m, CIMaskedReconLossConfig) for m in eval.metrics)
            and ci_roles == ("output",)
        ), (
            "in a single-role run CIMaskedReconLoss emits only the arm CEandKLLosses already "
            "emits, under the same `ce_kl/kl_ci_masked` key: author it OR CEandKLLosses, not "
            "both (a dual run keeps it for the hidden-role arm)"
        )
    partitions = nonlinearity_partitions(model.sites)
    standing_operations = (
        (make_nonlinearity_operation(slow_schedule(eval), partitions, compiler_options),)
        if partitions
        else ()
    )
    operations = (
        tuple(operation for metric in eval.metrics for operation in make_operations(metric))
        + standing_operations
    )

    operation_capture_keys = frozenset().union(
        *(_clean_capture_demand(metric) for metric in eval.metrics), EMPTY_CAPTURE_KEYS
    )
    context_step = make_lm_batch_context_step(
        model, capture_inputs, operation_capture_keys, mesh, compiler_options
    )

    def make_pass(invocation: EvalInvocation) -> LMEvalPass:
        pass_index = invocation.now_step // eval.every
        return LMEvalPass(
            state=invocation.state,
            now_step=invocation.now_step,
            placed_ci_fn=invocation.placed_ci_fn,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
            target_batches=(
                None
                if target_pool_batches_for is None
                else tuple(target_pool_batches_for(pass_index))
            ),
        )

    return Evaluation(operations, make_pass, make_lm_batch_contexts(context_step, model))
