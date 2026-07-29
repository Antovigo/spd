"""Per-step state passed to every metric's `update()`.

Built once per training step (after the DDP forward + CI calc) and once per eval batch.
"""

from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.ci_fns import CIRole
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.masks import SamplingType


@dataclass(frozen=True)
class MetricContext:
    """Per-step bundle handed to every `Metric.update(ctx)`.

    Built once per training step (after the DDP forward + CI calc) and once per eval
    batch. `ci` is the output CI net's; `ci_hidden` is the hidden-activation CI net's, and
    is `None` unless the run sets `pd.dual_hidden_ci`. Metrics that can read either reach
    them through `ci_for`.
    """

    model: ComponentModel
    batch: Any
    target_out: Tensor
    pre_weight_acts: dict[str, Float[Tensor, "..."]]
    ci: CIOutputs
    ci_hidden: CIOutputs | None
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]]
    step: int
    total_steps: int
    use_delta_component: bool
    sampling: SamplingType
    n_mask_samples: int
    reconstruction_loss: ReconstructionLoss
    is_eval: bool

    @property
    def current_frac_of_training(self) -> float:
        return self.step / self.total_steps if self.total_steps > 0 else 1.0

    def ci_for(self, role: CIRole) -> CIOutputs:
        match role:
            case "output":
                return self.ci
            case "hidden":
                assert self.ci_hidden is not None, (
                    "metric asked for the hidden CI net, but this run has none — "
                    "set pd.dual_hidden_ci to build it"
                )
                return self.ci_hidden
