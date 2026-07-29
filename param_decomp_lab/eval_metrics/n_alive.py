from typing import Literal, override

import torch
import wandb.plot
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.ci_fns import CIRole
from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import Metric, MetricResult, NamedMetricConfig
from param_decomp.metrics.context import MetricContext


class NAliveConfig(NamedMetricConfig):
    """`ci_role` picks which CI net to observe; give each instance a distinct `name` so their
    log keys do not collide.
    """

    type: Literal["NAlive"] = "NAlive"
    ci_role: CIRole = "output"
    ci_alive_threshold: float = 0.1


class NAlive(Metric[NAliveConfig]):
    """Per-matrix count of components active (CI > threshold) at least once over the eval batch.

    A component is "alive" if its causal importance exceeds `ci_alive_threshold` at any
    position of any example seen since the last `reset()`. Accumulates a per-component
    running max across `update()` calls (and across DDP ranks via an all-reduce MAX in
    `compute()`), so the count reflects the whole eval batch.
    """

    log_namespace = "n_alive"
    short_name = "NAlive"

    @override
    def reset(self) -> None:
        self.component_ci_max: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }

    @override
    def update(self, ctx: MetricContext) -> None:
        for module_name, ci_vals in ctx.ci_for(self.cfg.ci_role).lower_leaky.items():
            leading_dim_idxs = tuple(range(ci_vals.ndim - 1))
            batch_max = ci_vals.detach().float().amax(dim=leading_dim_idxs)
            self.component_ci_max[module_name] = torch.maximum(
                self.component_ci_max[module_name], batch_max
            )
        return None

    @override
    def compute(self) -> MetricResult:
        threshold = self.cfg.ci_alive_threshold
        out: dict[str, float | wandb.plot.CustomChart] = {}
        table_data: list[tuple[str, float]] = []
        total = 0.0
        for module_name in self.model.components:
            global_max = all_reduce(self.component_ci_max[module_name], op=ReduceOp.MAX)
            n_alive = float((global_max > threshold).sum().item())
            out[f"{self.key_prefix}{module_name}"] = n_alive
            total += n_alive
            table_data.append((module_name, n_alive))
        out[f"{self.key_prefix}total"] = total
        out[f"{self.key_prefix}bar_chart"] = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "n_alive"], data=table_data),
            label="layer",
            value="n_alive",
            title=f"n_alive (ci>{threshold}) [{self.cfg.ci_role}]",
        )
        return out
