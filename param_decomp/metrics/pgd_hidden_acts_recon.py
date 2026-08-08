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
    HiddenActsSitesConfig,
    SiteErrors,
    add_site_errors,
    assert_sources_reach_every_site,
    clean_site_outputs,
    detached_site_errors,
    masked_site_outputs,
    mean_relative_error,
    reduced_relative_errors,
    resolve_measured_sites,
    site_squared_errors,
)
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_objective_update


class PGDHiddenActsReconLossConfig(PGDConfig, EvalCadenceConfig, HiddenActsSitesConfig):
    """Config for the PGD-attacked per-site hidden-activation error.

    Cost under `site_inputs="masked_forward"` is set by `n_steps` (each step is one
    truncated forward plus a backward to the sources), so `slow` is left to the config: a
    few-step probe is cheap enough to run every eval, a 20-step one is better placed on the
    slow cadence. Under `"clean"` each step is a matmul per site instead of a forward.

    `"clean"` additionally requires that `site_patterns` leave no decomposed site
    unmeasured — see `assert_sources_reach_every_site`.
    """

    type: Literal["PGDHiddenActsReconLoss"] = "PGDHiddenActsReconLoss"
    ci_role: CIRole = "hidden"


class PGDHiddenActsReconLoss(Metric[PGDHiddenActsReconLossConfig]):
    """Relative per-site activation error under adversarially-optimised masks.

    Runs `cfg.n_steps` of sign-PGD on fresh adversarial sources each batch (no cross-batch
    persistence), maximising the relative site error rather than the output recon loss, and
    reports the error at the final sources. Under `site_inputs="masked_forward"` it uses the
    truncated `site_outputs` forward, so an `n_steps`-step attack costs `n_steps + 1` partial
    forwards rather than full ones; under `"clean"` no forward runs at all.
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
        self.measured_sites = resolve_measured_sites(model, self.cfg)
        if self.cfg.site_inputs == "clean":
            assert_sources_reach_every_site(model, self.measured_sites, self.instance_key)

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
            site_outputs = masked_site_outputs(
                ctx, mask_infos, self.measured_sites, self.cfg.site_inputs
            )
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
