"""Attention, hidden-activation, and causal-importance diagnostic operations."""

from functools import partial

import jax
import numpy as np
from jaxtyping import PRNGKeyArray

from param_decomp.core.configs import (
    CIHiddenActsReconLossConfig,
    CIHistogramsConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    IdentityCIErrorConfig,
    PermutedCIPlotsConfig,
    StochasticHiddenActsReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.core.eval_schedule import EvalSchedule
from param_decomp.core.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from param_decomp.core.metrics import LogRecord
from param_decomp.core.model import CaptureKeys, DecomposedModel
from param_decomp.core.run import (
    BackgroundRenderer,
    DeferredMediaRecord,
    EvalOperation,
)
from param_decomp.core.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    PermutationMetricSpec,
    PositionCI,
    SiteReduction,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_identity_ci_errors,
    make_position_ci_step,
    make_slow_eval_step,
    render_permutation_figures,
    render_slow_eval_figures,
    resolve_permutation_metrics,
)
from param_decomp.experiments.lm.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.eval_config import (
    CIMaskedAttnPatternsReconLossConfig,
    StochasticAttnPatternsReconLossConfig,
)
from param_decomp.experiments.lm.eval_context import LMEvalContext
from param_decomp.experiments.lm.eval_keys import EvalKeyStream


def _render_selected_figures(
    reductions: dict[str, SiteReduction], wanted: set[str], now_step: int
) -> DeferredMediaRecord:
    figures = render_slow_eval_figures(reductions)
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={f"slow_eval/{name}": figures[name] for name in wanted},
    )


def _render_permutation(
    spec: PermutationMetricSpec,
    position_ci: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    include_ci_heatmaps: bool,
    now_step: int,
) -> DeferredMediaRecord:
    figures = render_permutation_figures(spec, position_ci, components)
    if not include_ci_heatmaps:
        figures = {key: value for key, value in figures.items() if key == "figures/uv_matrices"}
    return DeferredMediaRecord(
        step_key="slow_eval/figure_step",
        step=now_step,
        media={f"slow_eval/{name}": value for name, value in figures.items()},
    )


def make_attention_operation(
    metric: CIMaskedAttnPatternsReconLossConfig | StochasticAttnPatternsReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIMaskedAttnPatternsReconLossConfig():
            step = make_ci_attn_patterns_step(model, ci_capture_keys, compiler_options)
        case StochasticAttnPatternsReconLossConfig():
            step = make_stochastic_attn_patterns_step(
                model, ci_capture_keys, metric.n_mask_samples, compiler_options
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_attn_patterns(
            step,
            model,
            context.state.decomposition.components,
            context.state.decomposition.ci_fn,
            list(context.batches),
            jax.random.fold_in(
                run_key, EvalKeyStream.ATTENTION_PATTERNS * train_steps + context.pass_index
            ),
        )
        return {
            f"eval/loss/{name}": value
            for name, value in attn_patterns_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def make_hidden_acts_operation(
    metric: CIHiddenActsReconLossConfig | StochasticHiddenActsReconLossConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    run_key: PRNGKeyArray,
    train_steps: int,
    compiler_options: dict[str, bool | int | str],
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIHiddenActsReconLossConfig():
            step = make_ci_hidden_acts_step(model, ci_capture_keys, compiler_options)
        case StochasticHiddenActsReconLossConfig():
            step = make_stochastic_hidden_acts_step(
                model, ci_capture_keys, metric.n_mask_samples, compiler_options
            )

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_hidden_acts(
            step,
            model,
            context.state.decomposition.components,
            context.state.decomposition.ci_fn,
            list(context.batches),
            jax.random.fold_in(
                run_key, EvalKeyStream.HIDDEN_ACTS * train_steps + context.pass_index
            ),
        )
        return {
            f"eval/slow/loss/{name}": value
            for name, value in hidden_acts_log_entries(metric.type, reductions).items()
        }

    return EvalOperation(schedule, run)


def make_site_figures_operation(
    metric: CIHistogramsConfig | ComponentActivationDensityConfig | CIMeanPerComponentConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> EvalOperation[LMEvalContext]:
    match metric:
        case CIHistogramsConfig():
            threshold = 0.0
            bins = metric.density_heatmap_n_bins
            limit = metric.n_batches_accum
            wanted = {
                "figures/causal_importance_values",
                "figures/causal_importance_values_pre_sigmoid",
                *({"figures/ci_density_heatmap"} if bins is not None else set()),
            }
        case ComponentActivationDensityConfig():
            threshold = metric.ci_alive_threshold
            bins = None
            limit = None
            wanted = {"figures/component_activation_density"}
        case CIMeanPerComponentConfig():
            threshold = 0.0
            bins = None
            limit = None
            wanted = {
                "figures/ci_mean_per_component",
                "figures/ci_mean_per_component_log",
            }
    step = make_slow_eval_step(model, ci_capture_keys, threshold, bins, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        reductions = accumulate_site_reductions(
            step,
            model,
            context.state.decomposition.ci_fn,
            list(context.batches),
            limit,
        )
        renderer.submit(partial(_render_selected_figures, reductions, wanted, context.now_step))
        return {}

    return EvalOperation(schedule, run)


def make_permutation_operation(
    metric: PermutedCIPlotsConfig | UVPlotsConfig | IdentityCIErrorConfig,
    schedule: EvalSchedule,
    model: DecomposedModel,
    ci_capture_keys: CaptureKeys,
    compiler_options: dict[str, bool | int | str],
    renderer: BackgroundRenderer,
) -> EvalOperation[LMEvalContext]:
    spec = resolve_permutation_metrics(model.site_names, [metric])
    position_step = make_position_ci_step(model, ci_capture_keys, compiler_options)

    def run(context: LMEvalContext) -> LogRecord:
        position_ci = accumulate_position_ci(
            position_step,
            model,
            context.state.decomposition.ci_fn,
            list(context.batches),
        )
        match metric:
            case IdentityCIErrorConfig():
                errors = compute_identity_ci_errors(spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE)
                return {f"eval/slow/{name}": value for name, value in errors.items()}
            case UVPlotsConfig():
                include_ci_heatmaps = False
                components = {
                    name: (np.asarray(site_components.V), np.asarray(site_components.U))
                    for name, site_components in context.state.decomposition.components.sites_items()
                }
            case PermutedCIPlotsConfig():
                include_ci_heatmaps = True
                components = None
        renderer.submit(
            partial(
                _render_permutation,
                spec,
                position_ci,
                components,
                include_ci_heatmaps,
                context.now_step,
            )
        )
        return {}

    return EvalOperation(schedule, run)
