"""Targeted L0 metric for CI values on target vs nontarget data."""

import re
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, ClassVar, override

import torch
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.component_utils import calc_ci_l_zero
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import extract_batch_data


class TargetedCI_L0(Metric):
    """L0 metric for CI values comparing target and nontarget data.

    Computes L0 sparsity for both target data (from the standard eval iterator)
    and nontarget data (from the nontarget eval iterator), enabling analysis of
    how well the decomposition separates target-specific vs general mechanisms.

    NOTE: Assumes all batches and sequences are the same size.
    """

    metric_section: ClassVar[str] = "targeted_l0"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        ci_alive_threshold: float,
        groups: dict[str, list[str]] | None,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ],
        sampling: SamplingType,
    ) -> None:
        self.model = model
        self.l0_threshold = ci_alive_threshold
        self.groups = groups
        self.device = device
        self.nontarget_eval_iterator = nontarget_eval_iterator
        self.sampling: SamplingType = sampling

        self.target_l0_values = defaultdict[str, list[float]](list)
        self.n_target_batches = 0

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        """Accumulate L0 values from target data."""
        group_sums = defaultdict(float) if self.groups else {}
        for layer_name, layer_ci in ci.lower_leaky.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.target_l0_values[layer_name].append(l0_val)

            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break

        for group_name, group_sum in group_sums.items():
            self.target_l0_values[group_name].append(group_sum)

        self.n_target_batches += 1

    @override
    def compute(self) -> dict[str, float | wandb.plot.CustomChart]:
        """Compute L0 for both target and nontarget data."""
        out: dict[str, float | wandb.plot.CustomChart] = {}
        table_data = []

        # Compute target L0 averages
        for key, l0s in self.target_l0_values.items():
            global_sum = all_reduce(torch.tensor(l0s, device=self.device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=self.device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
            out[f"target/{self.l0_threshold}_{key}"] = avg_l0
            table_data.append((f"target/{key}", avg_l0))

        # Compute nontarget L0 by processing matching number of batches
        nontarget_l0_values = defaultdict[str, list[float]](list)

        for _ in range(self.n_target_batches):
            batch_raw = next(self.nontarget_eval_iterator)
            batch = extract_batch_data(batch_raw).to(self.device)

            with torch.no_grad():
                pre_weight_acts = self.model(batch, cache_type="input").cache
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    detach_inputs=False,
                    sampling=self.sampling,
                )

            group_sums = defaultdict(float) if self.groups else {}
            for layer_name, layer_ci in ci.lower_leaky.items():
                l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
                nontarget_l0_values[layer_name].append(l0_val)

                if self.groups:
                    for group_name, patterns in self.groups.items():
                        for pattern in patterns:
                            if re.match(pattern.replace("*", ".*"), layer_name):
                                group_sums[group_name] += l0_val
                                break

            for group_name, group_sum in group_sums.items():
                nontarget_l0_values[group_name].append(group_sum)

        # Compute nontarget L0 averages
        for key, l0s in nontarget_l0_values.items():
            global_sum = all_reduce(torch.tensor(l0s, device=self.device).sum(), op=ReduceOp.SUM)
            global_count = all_reduce(torch.tensor(len(l0s), device=self.device), op=ReduceOp.SUM)
            avg_l0 = (global_sum / global_count).item()
            out[f"nontarget/{self.l0_threshold}_{key}"] = avg_l0
            table_data.append((f"nontarget/{key}", avg_l0))

        bar_chart = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"Targeted_L0_{self.l0_threshold}",
        )
        out["bar_chart"] = bar_chart
        return out
