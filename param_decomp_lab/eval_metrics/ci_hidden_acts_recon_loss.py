from typing import Literal, override

from param_decomp.ci_fns import CIRole
from param_decomp.component_model import ComponentModel
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.base import (
    EvalCadenceConfig,
    Metric,
    MetricResult,
    NamedMetricConfig,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    HiddenActsSitesConfig,
    SiteErrors,
    add_site_errors,
    clean_site_outputs,
    detached_site_errors,
    masked_site_outputs,
    reduced_relative_errors,
    resolve_measured_sites,
    site_squared_errors,
)


class CIHiddenActsReconLossConfig(NamedMetricConfig, EvalCadenceConfig, HiddenActsSitesConfig):
    """`ci_role` picks which CI net supplies the mask."""

    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"
    ci_role: CIRole = "output"


class CIHiddenActsReconLoss(Metric[CIHiddenActsReconLossConfig]):
    """Relative per-site activation error with components masked by CI directly.

    The no-attack member of the hidden-acts family: same relative error as
    `StochasticHiddenReconSubsetLoss` and `PGDHiddenActsReconLoss`, but with the mask set to
    CI itself rather than sampled or adversarially optimised. Run it once per `ci_role` on a
    dual-CI run to read off how much hidden-activation error each net's CI assignment leaves.
    """

    log_namespace = "loss"
    # One truncated forward per eval batch under `site_inputs="masked_forward"` — cheaper
    # than `CEandKLLosses`, which is not slow either — and none at all under `"clean"`.
    slow = False
    short_name = "CIHiddenActRecon"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.measured_sites = resolve_measured_sites(model, self.cfg)

    @override
    def reset(self) -> None:
        self._accum: SiteErrors = {}

    @override
    def update(self, ctx: MetricContext) -> None:
        targets = clean_site_outputs(self.model, ctx.pre_weight_acts, self.measured_sites)
        mask_infos = make_mask_infos(
            ctx.ci_for(self.cfg.ci_role).lower_leaky, weight_deltas_and_masks=None
        )
        site_outputs = masked_site_outputs(
            ctx, mask_infos, self.measured_sites, self.cfg.site_inputs
        )
        add_site_errors(
            self._accum,
            detached_site_errors(site_squared_errors(site_outputs, targets, mask_infos)),
        )
        return None

    @override
    def compute(self) -> MetricResult:
        return reduced_relative_errors(self._accum, self.instance_key)
