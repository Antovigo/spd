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
    """Per-layer plot of `‖V_c‖·‖U_c‖` per component, points coloured by max CI over the batch."""

    log_namespace = "figures"
    slow = True
    short_name = "WeightMag"

    @override
    def reset(self) -> None:
        self.max_ci_per_component: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        for module_name, ci_vals in ctx.ci.lower_leaky.items():
            batch_max = ci_vals.detach().amax(dim=tuple(range(ci_vals.ndim - 1)))
            self.max_ci_per_component[module_name] = torch.maximum(
                self.max_ci_per_component[module_name], batch_max
            )
        return None

    @override
    def compute(self) -> MetricResult:
        weight_magnitudes = {
            name: torch.linalg.norm(comp.V, dim=0) * torch.linalg.norm(comp.U, dim=1)
            for name, comp in self.model.components.items()
        }
        max_ci_per_component = {
            module_name: all_reduce(max_ci, op=ReduceOp.MAX)
            for module_name, max_ci in self.max_ci_per_component.items()
        }
        return {"weight_magnitude": plot_weight_magnitude(weight_magnitudes, max_ci_per_component)}
