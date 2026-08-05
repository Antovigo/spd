"""Worst-case per-site (hidden-activation) reconstruction error under PGD-optimised masks.

Eval-only counterpart of `PGDReconLoss`: it reports how much hidden-activation error a CI
assignment *permits* in the worst case, which is the probe that separates "these CI values
happen to work under stochastic sampling" from "these CI values really are safe to mask".
"""

from typing import Literal, override

from jaxtyping import Float
from torch import Tensor

from param_decomp.ci_fns import CIRole
from param_decomp.component_model import ComponentModel
from param_decomp.masks import AllLayersRouter, ComponentsMaskInfo
from param_decomp.metrics.base import EvalCadenceConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    SiteErrors,
    add_site_errors,
    clean_site_outputs,
    detached_site_errors,
    mean_relative_error,
    reduced_relative_errors,
    select_sites,
    site_squared_errors,
)
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_objective_update


class PGDHiddenActsReconLossConfig(PGDConfig, EvalCadenceConfig):
    """Config for the PGD-attacked per-site hidden-activation error.

    `site_patterns` filters which sites the error is measured at, as on
    `StochasticHiddenReconSubsetLoss`. Masks always cover every decomposed site. Cost is set
    by `n_steps` (each step is one truncated forward plus a backward to the sources), so
    `slow` is left to the config: a few-step probe is cheap enough to run every eval, a
    20-step one is better placed on the slow cadence.
    """

    type: Literal["PGDHiddenActsReconLoss"] = "PGDHiddenActsReconLoss"
    ci_role: CIRole = "hidden"
    site_patterns: list[str] | None = None


class PGDHiddenActsReconLoss(Metric[PGDHiddenActsReconLossConfig]):
    """Relative per-site activation error under adversarially-optimised masks.

    Runs `cfg.n_steps` of sign-PGD on fresh adversarial sources each batch (no cross-batch
    persistence), maximising the relative site error rather than the output recon loss, and
    reports the error at the final sources. Uses the truncated `site_outputs` forward, so an
    `n_steps`-step attack costs `n_steps + 1` partial forwards rather than full ones.
    """

    log_namespace = "loss"
    # Fast by default: knowing early whether the adversary finds much more error than
    # sampled masks is the point of having this probe. Set `slow: true` in the config for a
    # high-`n_steps` instance.
    slow = False
    short_name = "PGDHiddenRecon"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.measured_sites = select_sites(model.measurement_sites, self.cfg.site_patterns)

    @override
    def reset(self) -> None:
        self._accum: SiteErrors = {}

    @override
    def update(self, ctx: MetricContext) -> None:
        ci = ctx.ci_for(self.cfg.ci_role).lower_leaky
        weight_deltas = ctx.weight_deltas if ctx.use_delta_component else None
        targets = clean_site_outputs(self.model, ctx.pre_weight_acts, self.measured_sites)

        # The PGD driver only hands back the scalar objective, but accumulating an exact
        # ratio across batches needs the per-site numerators and denominators from the
        # *final* sources — so the objective stashes them on the way past.
        final_errors: SiteErrors = {}

        def site_error_objective(
            mask_infos: dict[str, ComponentsMaskInfo],
        ) -> tuple[Float[Tensor, ""], int]:
            nonlocal final_errors
            site_outputs = self.model.site_outputs(ctx.batch, mask_infos)
            final_errors = site_squared_errors(site_outputs, targets, mask_infos)
            return mean_relative_error(final_errors), 1

        pgd_masked_objective_update(
            model=self.model,
            ci=ci,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
            pgd_config=self.cfg,
            objective=site_error_objective,
        )
        add_site_errors(self._accum, detached_site_errors(final_errors))
        return None

    @override
    def compute(self) -> MetricResult:
        return reduced_relative_errors(self._accum, self.instance_key)
