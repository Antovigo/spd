from collections import defaultdict
from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor

from param_decomp.ci_fns import CIRole
from param_decomp.distributed import gather_all_tensors
from param_decomp.metrics.base import Metric, MetricResult, NamedMetricConfig
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_ci_values_histograms


class CIHistogramsConfig(NamedMetricConfig):
    """`n_batches_accum=None` accumulates every batch in the eval pass.

    `ci_role` picks which CI net to observe; give each instance a distinct `name` so their
    log keys do not collide.
    """

    type: Literal["CIHistograms"] = "CIHistograms"
    ci_role: CIRole = "output"
    n_batches_accum: int | None


class CIHistograms(Metric[CIHistogramsConfig]):
    """Per-layer histograms of CI values (lower-leaky and pre-sigmoid)."""

    log_namespace = "figures"
    slow = True
    short_name = "CIHist"

    @override
    def reset(self) -> None:
        self.batches_seen = 0
        self.lower_leaky_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)
        self.pre_sigmoid_causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.cfg.n_batches_accum is not None and self.batches_seen >= self.cfg.n_batches_accum:
            return None
        self.batches_seen += 1
        ci = ctx.ci_for(self.cfg.ci_role)
        for k, v in ci.lower_leaky.items():
            self.lower_leaky_causal_importances[k].append(v.detach())
        for k, v in ci.pre_sigmoid.items():
            self.pre_sigmoid_causal_importances[k].append(v.detach())
        return None

    @override
    def compute(self) -> MetricResult:
        if self.batches_seen == 0:
            raise RuntimeError("No batches seen yet")
        lower_leaky_cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name, ci_list in self.lower_leaky_causal_importances.items():
            lower_leaky_cis[module_name] = torch.cat(
                gather_all_tensors(torch.cat(ci_list, dim=0)), dim=0
            )
        pre_sigmoid_cis: dict[str, Float[Tensor, "... C"]] = {}
        for module_name, ci_list in self.pre_sigmoid_causal_importances.items():
            pre_sigmoid_cis[module_name] = torch.cat(
                gather_all_tensors(torch.cat(ci_list, dim=0)), dim=0
            )
        lower_leaky_fig = plot_ci_values_histograms(causal_importances=lower_leaky_cis)
        pre_sigmoid_fig = plot_ci_values_histograms(causal_importances=pre_sigmoid_cis)
        return {
            f"{self.key_prefix}causal_importance_values": lower_leaky_fig,
            f"{self.key_prefix}causal_importance_values_pre_sigmoid": pre_sigmoid_fig,
        }
