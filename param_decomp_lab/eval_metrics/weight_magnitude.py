from typing import Literal, override

import torch

from param_decomp.base_config import BaseConfig
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_weight_magnitude


class WeightMagnitudeConfig(BaseConfig):
    type: Literal["WeightMagnitude"] = "WeightMagnitude"


class WeightMagnitude(Metric[WeightMagnitudeConfig]):
    """Per-layer plot of `‖V_c‖·‖U_c‖` per component (pure weights-derived, no ctx use)."""

    log_namespace = "figures"
    slow = True
    short_name = "WeightMag"

    @override
    def reset(self) -> None:
        pass

    @override
    def update(self, ctx: MetricContext) -> None:
        del ctx
        return None

    @override
    def compute(self) -> MetricResult:
        weight_magnitudes = {
            name: torch.linalg.norm(comp.V, dim=0) * torch.linalg.norm(comp.U, dim=1)
            for name, comp in self.model.components.items()
        }
        return {"weight_magnitude": plot_weight_magnitude(weight_magnitudes)}
