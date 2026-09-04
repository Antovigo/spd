"""Attention and causal-importance diagnostic operations.

Every batched operation here folds over the pass's shared batch contexts ON ITS STREAM:
the CI-reduction family is a cheap on-device reduction of the context's CI envelope (one
head of it, `role`), and the masked-forward metrics (attention patterns) run only their
masked side — the clean side comes from the context. The weight-magnitude figure reads
the trained V/U alone and is a pass-level operation.
"""

from collections.abc import Callable, Mapping
from functools import partial

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from param_decomp.core.ci_fn import CIRole
from param_decomp.core.configs import (
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import PlacedModel
from param_decomp.core.nonlinearity import NonlinearityPartition
from param_decomp.core.nonlinearity_eval import (
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
)
from param_decomp.core.run import (
    BackgroundRenderer,
    BatchedOperation,
    DeferredMediaRecord,
    PassOperation,
    batched_operation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    VALUE_HISTOGRAM_N_BINS,
    CIReductionStep,
    PermutationMetricSpec,
    PositionCI,
    PositionCIAccumulation,
    SiteReduction,
    SiteReductionAccumulation,
    compute_identity_ci_errors,
    empty_position_ci_accumulation,
    empty_site_reduction_accumulation,
    fold_position_ci,
    fold_site_reduction,
    make_ci_reduction_step,
    make_position_ci_step,
    mean_cis,
    plot_mean_component_cis_two_streams,
    plot_weight_magnitudes,
    position_ci,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
    site_reductions,
    weight_magnitudes,
)
from param_decomp.experiments.lm.attn_patterns_eval import (
    LayerKLReduction,
    attn_output_key_by_site,
    attn_patterns_log_entries,
    fold_layer_kl,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval_config import (
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    LMEvalPass,
    Stream,
    role_log_segment,
    stream_log_prefix,
)
from param_decomp.experiments.lm.eval_keys import EvalKeyStream


def _figure_record(now_step: int, media: dict[str, bytes]) -> DeferredMediaRecord:
    """A slow-tier figure batch on the dedicated figure-step axis (SPEC S28)."""
    return DeferredMediaRecord(step_key="slow_eval/figure_step", step=now_step, media=media)


def _render_selected_figures(
    reductions: dict[str, SiteReduction], wanted: set[str], now_step: int, role_segment: str = ""
) -> DeferredMediaRecord:
    figures = render_slow_eval_figures(reductions)
    return _figure_record(
        now_step, {f"slow_eval/{role_segment}{name}": figures[name] for name in wanted}
    )


def _render_permutation(
    spec: PermutationMetricSpec,
    position_ci_by_site: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    include_ci_heatmaps: bool,
    now_step: int,
) -> DeferredMediaRecord:
    figures = render_permutation_figures(spec, position_ci_by_site, components)
    if not include_ci_heatmaps:
        figures = {key: value for key, value in figures.items() if key == "figures/uv_matrices"}
    return _figure_record(now_step, {f"slow_eval/{name}": value for name, value in figures.items()})


def _fold_stream_reduction(
    stream: Stream, role: CIRole, reduction_step: CIReductionStep
) -> Callable[[SiteReductionAccumulation, LMBatchContext], SiteReductionAccumulation]:
    """The one fold every CI-reduction operation shares: skip the other stream's contexts,
    reduce this role's fp32 preactivations otherwise."""

    def update(
        accumulation: SiteReductionAccumulation, context: LMBatchContext
    ) -> SiteReductionAccumulation:
        if context.stream != stream:
            return accumulation
        return fold_site_reduction(
            accumulation, reduction_step(context.ci_for(role).preactivations)
        )

    return update


def make_nonlinearity_operation(
    schedule: EvalSchedule,
    partitions: Mapping[str, NonlinearityPartition],
    compiler_options: dict[str, bool | int | str],
    stream: Stream = "nontarget",
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    """The standing nonlinearity eval (SPEC S36), stratified by the output head's mean
    CI over the broad stream."""
    reduction_step = make_ci_reduction_step(0.0, None, None, compiler_options)
    nonlinearity_step = make_nonlinearity_eval_step(partitions, compiler_options)
    update = _fold_stream_reduction(stream, "output", reduction_step)

    def finish(eval_pass: LMEvalPass, accumulation: SiteReductionAccumulation) -> LogRecord:
        reductions = site_reductions(accumulation)
        ci_means = {name: value.ci_sums / value.n_positions for name, value in reductions.items()}
        return nonlinearity_log_entries(
            nonlinearity_step(eval_pass.state.decomposition.components), ci_means, partitions
        )

    return batched_operation(schedule, empty_site_reduction_accumulation, update, finish)


def make_attention_operation(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
    stream: Stream,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            step = make_ci_attn_patterns_step(model, compiler_options)
        case StochasticAttnPatternsReconLossConfig():
            step = make_stochastic_attn_patterns_step(
                model, metric.n_mask_samples, compiler_options
            )
    output_key_by_site = attn_output_key_by_site(model)

    def init() -> dict[str, LayerKLReduction]:
        return {}

    def update(
        reductions: dict[str, LayerKLReduction], context: LMBatchContext
    ) -> dict[str, LayerKLReduction]:
        if context.stream != stream:
            return reductions
        base_key = jax.random.fold_in(
            run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + context.pass_index
        )
        # Output role explicitly: the attention-pattern probe measures how the OUTPUT-
        # reconstruction mask perturbs attention, matching the mask the recon grid uses.
        batch_sum, batch_n = step(
            model,
            context.prepared_weights,
            context.tokens,
            context.ci_for("output").lower,
            {site: context.captures[key] for site, key in output_key_by_site.items()},
            jax.random.fold_in(base_key, context.batch_index),
        )
        return fold_layer_kl(reductions, batch_sum, batch_n)

    def finish(eval_pass: LMEvalPass, reductions: dict[str, LayerKLReduction]) -> LogRecord:
        prefix = stream_log_prefix(stream, eval_pass.targeted)
        return {
            f"{prefix}loss/{name}": value
            for name, value in attn_patterns_log_entries(metric.type, reductions).items()
        }

    return batched_operation(schedule, init, update, finish)


def _render_weight_magnitudes(
    magnitudes: dict[str, np.ndarray], now_step: int
) -> DeferredMediaRecord:
    return _figure_record(
        now_step, {"slow_eval/figures/weight_magnitude": plot_weight_magnitudes(magnitudes)}
    )


def make_weight_magnitude_operation(
    schedule: EvalSchedule, renderer: BackgroundRenderer
) -> PassOperation[LMEvalPass]:
    """`‖V_c‖·‖U_c‖` per site. Reads the trained V/U only — no model, no batch, no step."""

    def run(eval_pass: LMEvalPass) -> LogRecord:
        magnitudes = weight_magnitudes(eval_pass.state.decomposition.components)
        renderer.submit(partial(_render_weight_magnitudes, magnitudes, eval_pass.now_step))
        return {}

    return PassOperation(schedule, run)


def _render_two_stream_ci_means(
    target: dict[str, np.ndarray],
    nontarget: dict[str, np.ndarray],
    now_step: int,
) -> DeferredMediaRecord:
    linear, log = plot_mean_component_cis_two_streams(target, nontarget)
    return _figure_record(
        now_step,
        {
            "slow_eval/figures/ci_mean_per_component_two_streams": linear,
            "slow_eval/figures/ci_mean_per_component_two_streams_log": log,
        },
    )


def make_two_stream_ci_mean_operation(
    schedule: EvalSchedule,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    """Both streams' output-head mean CI per component in one figure, ordered by the
    target mean. One fold keeps an accumulation per stream."""
    # `value_histogram_n_bins=None`: this operation reads the `(C,)` means only.
    reduction_step = make_ci_reduction_step(0.0, None, None, compiler_options)

    def init() -> dict[Stream, SiteReductionAccumulation]:
        return {
            "target": empty_site_reduction_accumulation(),
            "nontarget": empty_site_reduction_accumulation(),
        }

    def update(
        accumulations: dict[Stream, SiteReductionAccumulation], context: LMBatchContext
    ) -> dict[Stream, SiteReductionAccumulation]:
        return {
            **accumulations,
            context.stream: fold_site_reduction(
                accumulations[context.stream],
                reduction_step(context.ci_for("output").preactivations),
            ),
        }

    def finish(
        eval_pass: LMEvalPass, accumulations: dict[Stream, SiteReductionAccumulation]
    ) -> LogRecord:
        assert eval_pass.targeted, "TwoStreamCIMeanPerComponent measures a tPD run's two streams"
        renderer.submit(
            partial(
                _render_two_stream_ci_means,
                mean_cis(site_reductions(accumulations["target"])),
                mean_cis(site_reductions(accumulations["nontarget"])),
                eval_pass.now_step,
            )
        )
        return {}

    return batched_operation(schedule, init, update, finish)


def make_site_figures_operation(
    metric: CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig,
    schedule: EvalSchedule,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
    stream: Stream,
    role: CIRole,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    match metric:
        case CIHistogramsConfig():
            assert metric.n_batches_accum in (None, 1), (
                "CIHistograms bins its values exactly over one eval batch (the counts from "
                f"different batches sit on different edges), so n_batches_accum="
                f"{metric.n_batches_accum} cannot be honoured"
            )
            bins = metric.density_heatmap_n_bins
            wanted = {
                "figures/causal_importance_values",
                "figures/causal_importance_values_pre_sigmoid",
                *({"figures/ci_density_heatmap"} if bins is not None else set()),
            }
            reduction_step = make_ci_reduction_step(
                0.0, bins, VALUE_HISTOGRAM_N_BINS, compiler_options
            )
        case ComponentActivationDensityConfig():
            wanted = {"figures/component_activation_density"}
            reduction_step = make_ci_reduction_step(
                metric.ci_alive_threshold, None, None, compiler_options
            )
        case CIMeanPerComponentConfig():
            wanted = {
                "figures/ci_mean_per_component",
                "figures/ci_mean_per_component_log",
            }
            reduction_step = make_ci_reduction_step(0.0, None, None, compiler_options)

    update = _fold_stream_reduction(stream, role, reduction_step)

    def finish(eval_pass: LMEvalPass, accumulation: SiteReductionAccumulation) -> LogRecord:
        renderer.submit(
            partial(
                _render_selected_figures,
                site_reductions(accumulation),
                wanted,
                eval_pass.now_step,
                role_log_segment(role),
            )
        )
        return {}

    return batched_operation(schedule, empty_site_reduction_accumulation, update, finish)


def make_permutation_operation(
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: PlacedModel,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
    stream: Stream,
) -> BatchedOperation[LMEvalPass, LMBatchContext]:
    spec = resolve_permutation_metrics(model.site_names, [metric])
    position_step = make_position_ci_step(compiler_options)

    def update(
        accumulation: PositionCIAccumulation, context: LMBatchContext
    ) -> PositionCIAccumulation:
        if context.stream != stream:
            return accumulation
        return fold_position_ci(
            accumulation, position_step(context.ci_for("output").preactivations)
        )

    def finish(eval_pass: LMEvalPass, accumulation: PositionCIAccumulation) -> LogRecord:
        position_ci_by_site = position_ci(accumulation)
        match metric:
            case IdentityCIErrorConfig():
                errors = compute_identity_ci_errors(
                    spec, position_ci_by_site, IDENTITY_CI_ERROR_TOLERANCE
                )
                prefix = stream_log_prefix(stream, eval_pass.targeted)
                return {f"{prefix}slow/{name}": value for name, value in errors.items()}
            case UVPlotsConfig():
                include_ci_heatmaps = False
                components = {
                    name: (np.asarray(site_components.V), np.asarray(site_components.U))
                    for name, site_components in (
                        eval_pass.state.decomposition.components.sites_items()
                    )
                }
            case PermutedCIPlotsConfig():
                include_ci_heatmaps = True
                components = None
        renderer.submit(
            partial(
                _render_permutation,
                spec,
                position_ci_by_site,
                components,
                include_ci_heatmaps,
                eval_pass.now_step,
            )
        )
        return {}

    return batched_operation(schedule, empty_position_ci_accumulation, update, finish)
