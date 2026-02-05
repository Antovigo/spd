"""Weight magnitude metric: components sorted by weight magnitude, colored by max CI."""

from typing import ClassVar, override

from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.metrics.ci_vs_weight_magnitude import CIvsWeightMagnitude
from spd.plotting import plot_weight_magnitude

C = None  # jaxtyping placeholder


class WeightMagnitude(CIvsWeightMagnitude):
    """Weight magnitude per component, sorted by mean CI.

    For each layer:
    - X-axis: Components sorted by mean CI (descending)
    - Y-axis: Weight magnitude (||V|| * ||U||) per component
    - Color: Max CI over target inputs per component

    Reuses CIvsWeightMagnitude's data generation (target CIs + weight magnitudes).
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    @override
    def compute(self) -> dict[str, Image.Image]:
        target_cis = self._compute_target_cis()
        weight_magnitudes = self._compute_weight_magnitudes()

        max_cis: dict[str, Float[Tensor, C]] = {}
        mean_cis: dict[str, Float[Tensor, C]] = {}

        for layer_name, ci_tensor in target_cis.items():
            ci_flat = ci_tensor.reshape(-1, ci_tensor.shape[-1])  # [N, C]
            max_cis[layer_name] = ci_flat.max(dim=0).values  # [C]
            mean_cis[layer_name] = ci_flat.mean(dim=0)  # [C]

        img = plot_weight_magnitude(
            weight_magnitudes=weight_magnitudes,
            max_cis=max_cis,
            mean_cis=mean_cis,
        )
        return {"weight_magnitude": img}
