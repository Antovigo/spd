from typing import Any, ClassVar, override

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import _render_figure
from spd.utils.general_utils import get_obj_device


class CompletenessCIPlots(Metric):
    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        sampling: SamplingType,
        vocab_size: int,
        eq_token: int,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.vocab_size = vocab_size
        self.eq_token = eq_token

    @override
    def update(self, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, Image.Image]:
        device = get_obj_device(self.model)

        # Generate all valid inputs: [X, eq_token] for X in 1..vocab_size-1
        token_values = torch.arange(1, self.vocab_size, device=device)
        eq_col = torch.full_like(token_values, self.eq_token)
        batch = torch.stack([token_values, eq_col], dim=1)  # (n_inputs, 2)

        pre_weight_acts = self.model(batch, cache_type="input").cache
        ci = self.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=False,
            sampling=self.sampling,
        )

        # Select position 1 (eq_token position where the model predicts)
        ci_vals = {k: v[:, 1, :] for k, v in ci.upper_leaky.items()}
        img = _plot_completeness_ci(ci_vals, token_values.cpu())
        return {"completeness_ci": img}


def _plot_completeness_ci(
    ci_vals: dict[str, Float[Tensor, "n_inputs C"]],
    token_values: Tensor,
) -> Image.Image:
    n_modules = len(ci_vals)
    fig, axs = plt.subplots(
        1,
        n_modules,
        figsize=(5 * n_modules, 5),
        constrained_layout=True,
        squeeze=False,
        dpi=300,
    )
    axs = np.array(axs)

    images = []
    tick_labels = [str(v.item()) for v in token_values]

    for j, (name, ci) in enumerate(ci_vals.items()):
        data = ci.detach().cpu().numpy()
        ax = axs[0, j]
        im = ax.matshow(data, aspect="auto", cmap="Blues")
        images.append(im)

        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Input token")
        ax.set_yticks(range(len(tick_labels)))
        ax.set_yticklabels(tick_labels)
        ax.set_title(name)

    norm = plt.Normalize(
        vmin=min(v.min().item() for v in ci_vals.values()),
        vmax=max(v.max().item() for v in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    fig.suptitle("Completeness CI per input token")

    img = _render_figure(fig)
    plt.close(fig)
    return img
