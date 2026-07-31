from typing import Literal, override

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_weight_magnitude


class WeightMagnitudeConfig(BaseConfig):
    type: Literal["WeightMagnitude"] = "WeightMagnitude"


class WeightMagnitude(Metric[WeightMagnitudeConfig]):
    """Per-layer plot of `‖V_c‖·‖U_c‖` per component, points coloured by mean CI over the batch (log scale)."""

    log_namespace = "figures"
    slow = True
    short_name = "WeightMag"

    @override
    def reset(self) -> None:
        self.ci_sum_per_component: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }
        self.examples_seen: dict[str, Tensor] = {
            module_name: torch.zeros((), device=self.device, dtype=torch.long)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        for module_name, ci_vals in ctx.ci.lower_leaky.items():
            n_leading_dims = ci_vals.ndim - 1
            self.examples_seen[module_name] += ci_vals.shape[:n_leading_dims].numel()
            self.ci_sum_per_component[module_name] += ci_vals.detach().sum(
                dim=tuple(range(n_leading_dims))
            )
        return None

    @override
    def compute(self) -> MetricResult:
        weight_magnitudes = {
            name: torch.linalg.norm(comp.V, dim=0) * torch.linalg.norm(comp.U, dim=1)
            for name, comp in self.model.components.items()
        }
        mean_ci_per_component = {
            module_name: all_reduce(ci_sum, op=ReduceOp.SUM)
            / all_reduce(self.examples_seen[module_name], op=ReduceOp.SUM)
            for module_name, ci_sum in self.ci_sum_per_component.items()
        }
        return {"weight_magnitude": plot_weight_magnitude(weight_magnitudes, mean_ci_per_component)}
