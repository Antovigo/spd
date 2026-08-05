from typing import ClassVar, Literal, override

from param_decomp.ci_fns import CIRole
from param_decomp.component_model import ComponentModel
from param_decomp.masks import make_mask_infos, pinned_delta_masks
from param_decomp.metrics.base import (
    EvalCadenceConfig,
    Metric,
    MetricResult,
    NamedMetricConfig,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    SiteErrors,
    add_site_errors,
    clean_site_outputs,
    detached_site_errors,
    reduced_relative_errors,
    select_sites,
    site_squared_errors,
)


class _CIHiddenActsReconLossConfigBase(NamedMetricConfig, EvalCadenceConfig):
    """`ci_role` picks which CI net supplies the mask; `site_patterns` filters the sites."""

    ci_role: CIRole = "output"
    site_patterns: list[str] | None = None


class CIHiddenActsReconLossConfig(_CIHiddenActsReconLossConfigBase):
    type: Literal["CIHiddenActsReconLoss"] = "CIHiddenActsReconLoss"


class NontargetCIHiddenActsReconLossConfig(_CIHiddenActsReconLossConfigBase):
    type: Literal["NontargetCIHiddenActsReconLoss"] = "NontargetCIHiddenActsReconLoss"


class _CIHiddenActsReconLossBase[TConfig: _CIHiddenActsReconLossConfigBase](Metric[TConfig]):
    """Relative per-site activation error with components masked by CI directly.

    The no-attack member of the hidden-acts family: same relative error as
    `StochasticHiddenReconSubsetLoss` and `PGDHiddenActsReconLoss`, but with the mask set to
    CI itself rather than sampled or adversarially optimised. Run it once per `ci_role` on a
    dual-CI run to read off how much hidden-activation error each net's CI assignment leaves.
    """

    log_namespace = "loss"
    # One truncated forward per eval batch since it moved to `site_outputs` — cheaper than
    # `CEandKLLosses`, which is not slow either. It was slow when it ran two full forwards.
    slow = False
    delta_value: ClassVar[float | None] = None
    """Weight-delta mask for this distribution, as on `_TargetedReconLossBase`; `None`
    ablates the delta entirely."""

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.measured_sites = select_sites(model.measurement_sites, self.cfg.site_patterns)

    @override
    def reset(self) -> None:
        self._accum: SiteErrors = {}

    @override
    def update(self, ctx: MetricContext) -> None:
        targets = clean_site_outputs(self.model, ctx.pre_weight_acts, self.measured_sites)
        ci = ctx.ci_for(self.cfg.ci_role).lower_leaky
        mask_infos = make_mask_infos(
            ci, weight_deltas_and_masks=pinned_delta_masks(ctx.weight_deltas, ci, self.delta_value)
        )
        site_outputs = self.model.site_outputs(ctx.batch, mask_infos)
        add_site_errors(
            self._accum,
            detached_site_errors(site_squared_errors(site_outputs, targets, mask_infos)),
        )
        return None

    @override
    def compute(self) -> MetricResult:
        return reduced_relative_errors(self._accum, self.instance_key)


class CIHiddenActsReconLoss(_CIHiddenActsReconLossBase[CIHiddenActsReconLossConfig]):
    short_name = "CIHiddenActRecon"


class NontargetCIHiddenActsReconLoss(
    _CIHiddenActsReconLossBase[NontargetCIHiddenActsReconLossConfig]
):
    """Per-site activation error under the CI mask on the *nontarget* distribution.

    The no-attack floor `NontargetPGDHiddenActsReconLoss` is read against: same relative
    per-site error, but with the delta pinned fully on rather than ablated, which is what
    makes the number mean "how much the components perturb data they should ignore".
    """

    eval_distribution = "nontarget"
    short_name = "NontargetCIHiddenActRecon"
    delta_value = 1.0
