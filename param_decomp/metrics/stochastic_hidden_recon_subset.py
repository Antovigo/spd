"""Hidden-activation counterpart of `StochasticReconSubsetLoss`."""

from typing import Annotated, Literal, override

from pydantic import Field
from torch import Tensor

from param_decomp.ci_fns import CIRole
from param_decomp.component_model import ComponentModel
from param_decomp.masks import (
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
    calc_stochastic_component_mask_info,
    get_subset_router,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    SiteErrors,
    SiteInputs,
    add_site_errors,
    clean_site_outputs,
    detached_site_errors,
    masked_site_outputs,
    mean_relative_error,
    reduced_relative_errors,
    resolve_measured_sites,
    site_squared_errors,
)


class StochasticHiddenReconSubsetLossConfig(LossMetricConfig):
    """Config for the stochastic per-site hidden-activation reconstruction loss.

    `site_patterns` restricts which sites the error is *measured* at (fnmatch, e.g.
    `["*.mlp.down_proj", "*.self_attn.o_proj"]` for the residual-stream writes only);
    `None` measures every site in `ComponentModel.measurement_sites` — every decomposed
    site plus any `pd.hidden_readout_sites`. Masking always covers every decomposed site
    regardless — only measurement is filtered.

    `site_inputs` selects whether each site is judged on the chained masked forward's input
    or on its own clean one; see `SiteInputs`.
    """

    type: Literal["StochasticHiddenReconSubsetLoss"] = "StochasticHiddenReconSubsetLoss"
    ci_role: CIRole = "hidden"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
    site_patterns: list[str] | None = None
    site_inputs: SiteInputs = "masked_forward"


class StochasticHiddenReconSubsetLoss(Metric[StochasticHiddenReconSubsetLossConfig]):
    """Relative error of the decomposed sites' activations under stochastic subset ablation.

    Same masks and same subset routing as `StochasticReconSubsetLoss`, but the error is
    read at the decomposed sites against the frozen model's own site outputs instead of at
    the logits — so each layer gets signal from immediately downstream rather than
    backpropagated from the output.

    Under `site_inputs="masked_forward"` the forward stops after the last decomposed site
    (`ComponentModel.site_outputs`), so nothing past that point is computed or retained for
    backward. Under `site_inputs="clean"` there is no forward at all.
    """

    log_namespace = "loss"
    short_name = "StochHiddenReconSub"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.router = get_subset_router(self.cfg.routing, device)
        self.measured_sites = resolve_measured_sites(
            model, self.cfg.site_patterns, self.cfg.site_inputs
        )

    @override
    def reset(self) -> None:
        self._accum: SiteErrors = {}

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        ci = ctx.ci_for(self.cfg.ci_role).lower_leaky
        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        targets = clean_site_outputs(self.model, ctx.pre_weight_acts, self.measured_sites)

        batch_errors: SiteErrors = {}
        for _ in range(ctx.n_mask_samples):
            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=ci,
                component_mask_sampling=ctx.sampling,
                weight_deltas=weight_deltas,
                router=self.router,
            )
            site_outputs = masked_site_outputs(
                model=self.model,
                batch=ctx.batch,
                pre_weight_acts=ctx.pre_weight_acts,
                mask_infos=mask_infos,
                sites=self.measured_sites,
                site_inputs=self.cfg.site_inputs,
            )
            add_site_errors(batch_errors, site_squared_errors(site_outputs, targets, mask_infos))

        if ctx.is_eval:  # `compute()` is eval-only, and each eval pass `reset()`s first
            add_site_errors(self._accum, detached_site_errors(batch_errors))
        return mean_relative_error(batch_errors)

    @override
    def compute(self) -> MetricResult:
        return reduced_relative_errors(self._accum, self.instance_key)
