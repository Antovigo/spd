from typing import Literal, override

import torch
import wandb.plot
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class CIAnomalyRateConfig(BaseConfig):
    type: Literal["CIAnomalyRate"] = "CIAnomalyRate"
    ci_threshold: float


class CIAnomalyRate(Metric[CIAnomalyRateConfig]):
    """Fraction of (position, component) pairs important for output but not hidden recon.

    "Important" is a hard threshold on each net's CI: `ci_output > ci_threshold` and
    `ci_hidden <= ci_threshold`. Only meaningful on a `pd.dual_hidden_ci` run — asserts via
    `ctx.ci_for("hidden")` otherwise.
    """

    log_namespace = "ci_anomaly"
    short_name = "CIAnomaly"

    @override
    def reset(self) -> None:
        self.anomaly_sum: dict[str, Tensor] = {
            module_name: torch.zeros((), device=self.device)
            for module_name in self.model.components
        }
        self.total_count: dict[str, Tensor] = {
            module_name: torch.zeros((), device=self.device, dtype=torch.long)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        ci_output = ctx.ci_for("output").lower_leaky
        ci_hidden = ctx.ci_for("hidden").lower_leaky
        threshold = self.cfg.ci_threshold
        for module_name in self.model.components:
            anomaly = (ci_output[module_name] > threshold) & (ci_hidden[module_name] <= threshold)
            self.anomaly_sum[module_name] += anomaly.float().sum().detach()
            self.total_count[module_name] += anomaly.numel()
        return None

    @override
    def compute(self) -> MetricResult:
        out: dict[str, float | wandb.plot.CustomChart] = {}
        table_data: list[tuple[str, float]] = []
        total_anomaly = torch.zeros((), device=self.device)
        total_count = torch.zeros((), device=self.device, dtype=torch.long)
        for module_name in self.model.components:
            anomaly = all_reduce(self.anomaly_sum[module_name], op=ReduceOp.SUM)
            count = all_reduce(self.total_count[module_name], op=ReduceOp.SUM)
            rate = (anomaly / count).item()
            out[module_name] = rate
            table_data.append((module_name, rate))
            total_anomaly += anomaly
            total_count += count
        out["total"] = (total_anomaly / total_count).item()
        out["bar_chart"] = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "anomaly_rate"], data=table_data),
            label="layer",
            value="anomaly_rate",
            title=f"P(ci_output>{self.cfg.ci_threshold} and ci_hidden<={self.cfg.ci_threshold})",
        )
        return out
