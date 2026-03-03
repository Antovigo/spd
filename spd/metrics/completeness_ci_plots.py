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

        # ci.lower_leaky values have shape (n_inputs, seq_len, C)
        img = _plot_completeness_ci(ci.lower_leaky, token_values.cpu())
        return {"completeness_ci": img}


def _plot_completeness_ci(
    ci_vals: dict[str, Float[Tensor, "n_inputs seq_len C"]],
    token_values: Tensor,
) -> Image.Image:
    """Plot CI heatmaps: rows = input tokens, columns = modules.

    Each subplot: x = component index, y = sequence position, color = CI.
    """
    n_inputs = len(token_values)
    n_modules = len(ci_vals)
    module_names = list(ci_vals.keys())

    fig, axs = plt.subplots(
        n_inputs,
        n_modules,
        figsize=(4 * n_modules, 2 * n_inputs),
        constrained_layout=True,
        squeeze=False,
        dpi=200,
    )
    axs = np.array(axs)

    images = []
    for row, token_val in enumerate(token_values):
        for col, name in enumerate(module_names):
            # ci shape: (n_inputs, seq_len, C) -> select this input
            data = ci_vals[name][row].detach().cpu().numpy()  # (seq_len, C)
            ax = axs[row, col]
            im = ax.matshow(data, aspect="auto", cmap="Blues")
            images.append(im)

            ax.xaxis.tick_bottom()
            ax.xaxis.set_label_position("bottom")

            if row == 0:
                ax.set_title(name)
            if row == n_inputs - 1:
                ax.set_xlabel("Component index")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"token {token_val.item()}")
            else:
                ax.set_yticklabels([])

    norm = plt.Normalize(
        vmin=min(v.min().item() for v in ci_vals.values()),
        vmax=max(v.max().item() for v in ci_vals.values()),
    )
    for im in images:
        im.set_norm(norm)
    fig.colorbar(images[0], ax=axs.ravel().tolist())

    fig.suptitle("Completeness CI (per input token × sequence position)")

    img = _render_figure(fig)
    plt.close(fig)
    return img
