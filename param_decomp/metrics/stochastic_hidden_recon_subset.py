"""Hidden-activation counterpart of `StochasticReconSubsetLoss`."""

from typing import Annotated, Literal, override

from pydantic import Field
from torch import Tensor

from param_decomp.ci_fns import CIRole
from param_decomp.component_model import ComponentModel
from param_decomp.masks import (
    RoutingType,
    UniformKSubsetRoutingConfig,
    calc_stochastic_component_mask_info,
    get_router,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    HiddenActsSitesConfig,
    SiteErrors,
    add_site_errors,
    clean_site_outputs,
    detached_site_errors,
    masked_site_outputs,
    mean_relative_error,
    reduced_relative_errors,
    resolve_measured_sites,
    site_squared_errors,
)


class StochasticHiddenReconSubsetLossConfig(LossMetricConfig, HiddenActsSitesConfig):
    """Config for the stochastic per-site hidden-activation reconstruction loss."""

    type: Literal["StochasticHiddenReconSubsetLoss"] = "StochasticHiddenReconSubsetLoss"
    ci_role: CIRole = "hidden"
    routing: Annotated[
        RoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticHiddenReconSubsetLoss(Metric[StochasticHiddenReconSubsetLossConfig]):
    """Relative error of the decomposed sites' activations under stochastic subset ablation.

    Same masks and same subset routing as `StochasticReconSubsetLoss`, but the error is
    read at the decomposed sites against the frozen model's own site outputs instead of at
    the logits — so each layer gets signal from immediately downstream rather than
    backpropagated from the output.

    Under `site_inputs="masked_forward"` the forward stops after the last decomposed site
    (`ComponentModel.site_outputs`), so nothing past that point is computed or retained for
    backward. Under `site_inputs="clean"` there is no forward at all.

    `routing` decides how far ablation damage travels: under a subsetting router a site is
    scored only where it is itself routed, and each upstream site is frozen at a good
    fraction of those positions, so the compounding is heavily diluted. `{type: all}`
    replaces every site everywhere, which is what the adversarial sibling
    (`PersistentPGDHiddenActsReconLoss`) has always done.
    """

    log_namespace = "loss"
    short_name = "StochHiddenReconSub"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.router = get_router(self.cfg.routing, device)
        self.measured_sites = resolve_measured_sites(model, self.cfg)

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
                ctx, mask_infos, self.measured_sites, self.cfg.site_inputs
            )
            add_site_errors(batch_errors, site_squared_errors(site_outputs, targets, mask_infos))

        if ctx.is_eval:  # `compute()` is eval-only, and each eval pass `reset()`s first
            add_site_errors(self._accum, detached_site_errors(batch_errors))
        return mean_relative_error(batch_errors)

    @override
    def compute(self) -> MetricResult:
        return reduced_relative_errors(self._accum, self.instance_key)
