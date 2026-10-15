"""LM evaluation operation binding and execution.

Binding is two closed passes over the authored metrics: first each metric declares its
clean-capture demand on the shared batch context (`clean_capture_demand`), then each
binds its operations — one PER STREAM the metric measures, and per CI role on a dual run.
The pass's context step captures the union of those demands, so every shared-forward
operation reads one clean forward + CI envelope per batch of each stream.
"""

from collections.abc import Callable, Iterable
from typing import Any

import jax
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn.interface import CIRole, roles_for
from param_decomp.core.components import nonlinearity_partitions
from param_decomp.core.configs import (
    CI_L0Config,
    CIHistogramsConfig,
    CIMaskedReconEvalConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    EvalPGDReconLossConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    SlowPGDReconLossConfig,
    UnmaskedNoDeltaReconEvalConfig,
    UVPlotsConfig,
    WeightMagnitudeConfig,
)
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ComponentActivations,
    PlacedModel,
)
from param_decomp.core.recon_eval import fresh_pgd_probe
from param_decomp.core.run import (
    BackgroundRenderer,
    EvalInvocation,
    EvalOperation,
    EvalOperationPlan,
    Evaluation,
    EvaluationPlan,
    MetricsSink,
    SharedForwardOperation,
    SharedForwardOperationPlan,
    StandaloneOperationPlan,
    no_batch_contexts,
)
from param_decomp.core.slow_eval import component_group_counts
from param_decomp.experiments.eval_config import (
    AnyEvalMetricConfig,
    EvalConfig,
    schedule_for,
    slow_schedule,
)
from param_decomp.experiments.lm.ab_grid_operation import make_ab_grid_operation
from param_decomp.experiments.lm.arithmetic_eval_operation import make_arithmetic_operation
from param_decomp.experiments.lm.attn_patterns_eval import attn_output_key_by_site
from param_decomp.experiments.lm.ci_position_eval import make_ci_position_counts_operation
from param_decomp.experiments.lm.diagnostic_eval_operations import (
    make_attention_operation,
    make_nonlinearity_operation,
    make_permutation_operation,
    make_router_divergence_operation,
    make_site_figures_operation,
    make_two_stream_ci_mean_operation,
    make_weight_magnitude_operation,
)
from param_decomp.experiments.lm.eval_config import (
    ABGridDatasetConfig,
    ArithmeticCIGridConfig,
    CEandKLLossesConfig,
    CIActiveCountsPerPositionConfig,
    CIMaskedAttnPatternsReconLossConfig,
    RouterDivergenceConfig,
    StochasticAttnPatternsReconLossConfig,
    TwoStreamCIMeanPerComponentConfig,
    WellTemperednessConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    MultiStreamOperationPlan,
    Stream,
    make_lm_batch_context_step,
    make_lm_batch_contexts,
    nontarget_delta_pinned,
    stream_batches,
    stream_log_prefix,
)
from param_decomp.experiments.lm.resolved import LMAnyRun
from param_decomp.experiments.lm.router_divergence_eval import router_probs_capture_keys
from param_decomp.experiments.lm.scalar_eval_operations import (
    make_ce_kl_operation,
    make_ci_l0_operation,
    make_fresh_pgd_operation,
    make_masked_kl_operation,
)
from param_decomp.experiments.lm.well_temperedness_eval import make_well_temperedness_operation
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput


def clean_capture_demand[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    metric: AnyEvalMetricConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> CaptureKeys:
    """What this metric reads off the shared clean forward beyond the CI taps."""
    match metric:
        case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
            return frozenset(attn_output_key_by_site(model).values())
        case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
            return fresh_pgd_probe(metric).reconstruction_capture_keys
        case RouterDivergenceConfig():
            return frozenset(router_probs_capture_keys(model))
        case (
            CEandKLLossesConfig()
            | CI_L0Config()
            | CIActiveCountsPerPositionConfig()
            | CIHistogramsConfig()
            | ComponentActivationDensityConfig()
            | CIMeanPerComponentConfig()
            | PermutedCIPlotsConfig()
            | UVPlotsConfig()
            | IdentityCIErrorConfig()
            | WellTemperednessConfig()
            | ArithmeticCIGridConfig()
            | ABGridDatasetConfig()
            | CIMaskedReconEvalConfig()
            | UnmaskedNoDeltaReconEvalConfig()
            | TwoStreamCIMeanPerComponentConfig()
            | WeightMagnitudeConfig()
        ):
            return EMPTY_CAPTURE_KEYS


def _stream_filtered[PassT, ContextT: LMBatchContext[Any, Any, Any]](
    operation: SharedForwardOperation[PassT, ContextT], stream: Stream
) -> SharedForwardOperation[PassT, ContextT]:
    """The operation folding only its own stream's contexts: the pass yields every stream's,
    and a metric bound to one stream must not average the other's batches into its own."""

    def update(state: object, context: ContextT) -> object:
        return state if context.stream != stream else operation.update(state, context)

    return SharedForwardOperation(operation.schedule, operation.init, update, operation.finish)


type _BoundPlan[PassT, ContextT] = tuple[
    EvalOperationPlan[PassT, ContextT] | MultiStreamOperationPlan[PassT, ContextT], Stream | None
]
"""An operation plan and the one stream whose example context its kernels compile against
(`None` for a standalone operation or one that folds several streams)."""


def make_lm_evaluation[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    built: LMAnyRun,
    eval: EvalConfig,
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    sink: MetricsSink,
    compiler_options: dict[str, bool | int | str],
    *,
    sample_batch: Callable[[int], TargetIn],
    sample_target_batch: Callable[[int], TargetIn] | None = None,
) -> EvaluationPlan[
    Conditioning,
    LMEvalPass[TargetIn, Conditioning],
    LMBatchContext[TargetIn, PreparedT, Conditioning],
]:
    """Describe the kernels each authored metric needs before any evaluation runs — one
    operation PER STREAM the metric measures, and per CI role on a dual run.

    `sample_target_batch` supplies a tPD run's target stream, which `data.eval` cannot;
    `None` on a plain run, which collapses every metric to the one stream it has."""
    pd = built.pd
    capture_inputs = built.ci_fn.capture_keys
    renderer = BackgroundRenderer(sink)
    targeted = sample_target_batch is not None
    data_streams: tuple[Stream, ...] = ("nontarget", "target") if targeted else ("nontarget",)
    optimized_stream: Stream = "target" if targeted else "nontarget"
    # A dual CI fn answers every CI-reading metric TWICE — once per readout head — because
    # the two heads are the experiment: a component important for the output objective and
    # not the hidden one (or the reverse) is exactly what a dual run is looking for, and
    # reporting one head would hide it. A single-role run keeps one readout under unchanged
    # keys.
    ci_roles: tuple[CIRole, ...] = roles_for(bool(getattr(built.ci_fn, "dual", False)))
    authored = {type(metric) for metric in eval.metrics}
    ce_kl_authored = CEandKLLossesConfig in authored

    assert TwoStreamCIMeanPerComponentConfig not in authored or targeted, (
        "TwoStreamCIMeanPerComponent measures the tPD target stream; a plain run has no prompt pool"
    )
    assert not {TwoStreamCIMeanPerComponentConfig, CIMeanPerComponentConfig} <= authored, (
        "TwoStreamCIMeanPerComponent already computes the nontarget-stream reduction "
        "CIMeanPerComponent does, so authoring both pays for that pass twice"
    )
    if ce_kl_authored:
        assert UnmaskedNoDeltaReconEvalConfig not in authored, (
            "UnmaskedNoDeltaReconLoss emits the arm CEandKLLosses already emits, under the "
            "same `ce_kl/kl_unmasked` key: author it OR CEandKLLosses, not both"
        )
        assert not (CIMaskedReconEvalConfig in authored and ci_roles == ("output",)), (
            "in a single-role run CIMaskedReconLoss emits only the arm CEandKLLosses already "
            "emits, under the same `ce_kl/kl_ci_masked` key: author it OR CEandKLLosses, not "
            "both (a dual run keeps it for the hidden-role arm)"
        )

    def batches(pass_index: int) -> list[TargetIn]:
        return [sample_batch(pass_index * eval.n_steps + j) for j in range(eval.n_steps)]

    def target_batches(pass_index: int) -> list[TargetIn] | None:
        if sample_target_batch is None:
            return None
        return [sample_target_batch(pass_index * eval.n_steps + j) for j in range(eval.n_steps)]

    def make_operations(
        metric: AnyEvalMetricConfig,
    ) -> list[
        _BoundPlan[
            LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
        ]
    ]:
        schedule = schedule_for(metric, eval)
        match metric:
            case CEandKLLossesConfig():
                return [
                    (
                        make_ce_kl_operation(
                            metric,
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            eval.n_steps,
                            mesh,
                            compiler_options,
                            log_prefix=stream_log_prefix(stream, targeted),
                            delta_pinned=nontarget_delta_pinned(targeted=targeted, stream=stream),
                        ),
                        stream,
                    )
                    for stream in data_streams
                ]
            case CIMaskedReconEvalConfig():
                # CEandKLLosses already emits the output-role arm under the same
                # `ce_kl/kl_ci_masked` key; keep only the roles it cannot measure.
                masked_roles: tuple[CIRole, ...] = (
                    tuple(role for role in ci_roles if role != "output")
                    if ce_kl_authored
                    else ci_roles
                )
                return [
                    (
                        make_masked_kl_operation(
                            "ci_masked",
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            eval.n_steps,
                            mesh,
                            compiler_options,
                            role=role,
                            log_prefix=stream_log_prefix(stream, targeted, role),
                            delta_pinned=nontarget_delta_pinned(targeted=targeted, stream=stream),
                        ),
                        stream,
                    )
                    for stream in data_streams
                    for role in masked_roles
                ]
            case UnmaskedNoDeltaReconEvalConfig():
                return [
                    (
                        make_masked_kl_operation(
                            "unmasked",
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            eval.n_steps,
                            mesh,
                            compiler_options,
                            log_prefix=stream_log_prefix(optimized_stream, targeted),
                        ),
                        optimized_stream,
                    )
                ]
            case CIActiveCountsPerPositionConfig():
                return [
                    (
                        make_ci_position_counts_operation(schedule, compiler_options),
                        optimized_stream,
                    )
                ]
            case CI_L0Config():
                return [
                    (
                        make_ci_l0_operation(
                            metric,
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            eval.n_steps,
                            mesh,
                            compiler_options,
                            role=role,
                            log_prefix=stream_log_prefix(stream, targeted, role),
                        ),
                        stream,
                    )
                    for stream in data_streams
                    for role in ci_roles
                ]
            case EvalPGDReconLossConfig() | SlowPGDReconLossConfig():
                return [
                    (
                        make_fresh_pgd_operation(
                            metric,
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            eval.n_steps,
                            mesh,
                            compiler_options,
                            role=role,
                            log_prefix=stream_log_prefix(stream, targeted, role),
                            delta_pinned=nontarget_delta_pinned(targeted=targeted, stream=stream),
                        ),
                        stream,
                    )
                    for stream in data_streams
                    for role in ci_roles
                ]

            case CIMaskedAttnPatternsReconLossConfig() | StochasticAttnPatternsReconLossConfig():
                return [
                    (
                        make_attention_operation(
                            metric,
                            schedule,
                            model,
                            run_key,
                            pd.steps,
                            compiler_options,
                            log_prefix=stream_log_prefix(optimized_stream, targeted),
                        ),
                        optimized_stream,
                    )
                ]
            case RouterDivergenceConfig():
                return [
                    (
                        make_router_divergence_operation(
                            metric, schedule, model, run_key, pd.steps, mesh, compiler_options
                        ),
                        optimized_stream,
                    )
                ]
            case (
                CIHistogramsConfig()
                | ComponentActivationDensityConfig()
                | CIMeanPerComponentConfig()
            ):
                return [
                    (
                        make_site_figures_operation(
                            metric,
                            schedule,
                            component_group_counts(model.model.sites),
                            compiler_options,
                            renderer,
                            role=role,
                        ),
                        optimized_stream,
                    )
                    for role in ci_roles
                ]
            case PermutedCIPlotsConfig() | UVPlotsConfig() | IdentityCIErrorConfig():
                return [
                    (
                        make_permutation_operation(
                            metric,
                            schedule,
                            model,
                            compiler_options,
                            renderer,
                            log_prefix=stream_log_prefix(optimized_stream, targeted),
                        ),
                        optimized_stream,
                    )
                ]

            case WellTemperednessConfig():
                return [
                    (
                        make_well_temperedness_operation(
                            metric,
                            schedule,
                            model,
                            capture_inputs,
                            mesh,
                            compiler_options,
                            run_key=run_key,
                            train_steps=pd.steps,
                            figure_rendering=renderer if sink.accepts_deferred_media else None,
                            stream=optimized_stream,
                        ),
                        None,
                    )
                ]

            case ArithmeticCIGridConfig():
                return [
                    (
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
                        None,
                    )
                ]
            case ABGridDatasetConfig():
                return [
                    (
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
                        None,
                    )
                ]
            case TwoStreamCIMeanPerComponentConfig():
                return [
                    (make_two_stream_ci_mean_operation(schedule, compiler_options, renderer), None)
                ]
            case WeightMagnitudeConfig():
                return [(make_weight_magnitude_operation(schedule, renderer), None)]

    standing: list[
        _BoundPlan[
            LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
        ]
    ] = (
        [
            (
                make_nonlinearity_operation(
                    slow_schedule(eval), model.model.sites, compiler_options, mesh
                ),
                "nontarget",
            )
        ]
        if nonlinearity_partitions(model.model.sites)
        else []
    )
    plans: list[
        _BoundPlan[
            LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
        ]
    ] = [bound for metric in eval.metrics for bound in make_operations(metric)] + standing

    operation_capture_keys = frozenset().union(
        *(clean_capture_demand(metric, model) for metric in eval.metrics), EMPTY_CAPTURE_KEYS
    )
    context_step = make_lm_batch_context_step(model, capture_inputs, operation_capture_keys, mesh)

    def streams_read(
        bound: _BoundPlan[
            LMEvalPass[TargetIn, Conditioning], LMBatchContext[TargetIn, PreparedT, Conditioning]
        ],
    ) -> tuple[Stream, ...]:
        match bound:
            case (MultiStreamOperationPlan() as plan, _):
                return plan.streams
            case (SharedForwardOperationPlan(), stream):
                assert stream is not None
                return (stream,)
            case (StandaloneOperationPlan(), _):
                return ()

    # Broad stream first, then the target pool: the order the pass yields contexts in.
    stream_order: tuple[Stream, ...] = ("nontarget", "target")
    needed_streams: tuple[Stream, ...] = tuple(
        stream for stream in stream_order if any(stream in streams_read(bound) for bound in plans)
    )

    def make_pass(invocation: EvalInvocation[Conditioning]) -> LMEvalPass[TargetIn, Conditioning]:
        pass_index = invocation.now_step // eval.every
        target = target_batches(pass_index)
        return LMEvalPass(
            decomposition=invocation.decomposition,
            persistent_sources=invocation.persistent_sources,
            now_step=invocation.now_step,
            pass_index=pass_index,
            batches=tuple(batches(pass_index)),
            target_batches=None if target is None else tuple(target),
        )

    def prepare(
        invocation: EvalInvocation[Conditioning],
    ) -> Evaluation[
        Conditioning,
        LMEvalPass[TargetIn, Conditioning],
        LMBatchContext[TargetIn, PreparedT, Conditioning],
    ]:
        example_pass = make_pass(invocation)
        example_contexts: dict[Stream, LMBatchContext[TargetIn, PreparedT, Conditioning]] = {}
        batch_contexts: Callable[
            [LMEvalPass[TargetIn, Conditioning]],
            Iterable[LMBatchContext[TargetIn, PreparedT, Conditioning]],
        ] = no_batch_contexts
        if needed_streams:
            # One compiled context step per stream: the two streams' batches differ in shape.
            compiled_contexts = {}
            for stream in needed_streams:
                compiled = (
                    jax.jit(context_step, compiler_options=compiler_options)
                    .lower(
                        model,
                        invocation.decomposition.components,
                        invocation.decomposition.ci_fn,
                        stream_batches(stream, example_pass)[0],
                    )
                    .compile()
                )
                compiled_contexts[stream] = compiled
                example_contexts[stream] = LMBatchContext(
                    pass_index=example_pass.pass_index,
                    batch_index=0,
                    forward=compiled.out_info,
                    persistent_sources=invocation.persistent_sources,
                    stream=stream,
                )
            batch_contexts = make_lm_batch_contexts(compiled_contexts, model)
        prepared_operations: list[
            EvalOperation[
                LMEvalPass[TargetIn, Conditioning],
                LMBatchContext[TargetIn, PreparedT, Conditioning],
            ]
        ] = []
        for plan, stream in plans:
            match plan:
                case StandaloneOperationPlan():
                    prepared_operations.append(plan.prepare(example_pass))
                case SharedForwardOperationPlan():
                    assert stream is not None
                    prepared_operations.append(
                        _stream_filtered(
                            plan.prepare(example_pass, example_contexts[stream]), stream
                        )
                    )
                case MultiStreamOperationPlan():
                    prepared_operations.append(
                        plan.prepare(
                            example_pass,
                            {stream: example_contexts[stream] for stream in plan.streams},
                        )
                    )
        return Evaluation(tuple(prepared_operations), make_pass, batch_contexts)

    return EvaluationPlan(prepare)
