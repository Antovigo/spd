from typing import Literal, cast, override

import matplotlib.pyplot as plt
import torch

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import _render_figure
from param_decomp_lab.experiments.toy_model_redundancy.ci_figure import plot_subcomponent_grid
from param_decomp_lab.toy_models.toy_model_redundancy_copy import ToyModelRedundancyCopyTransformer


class ToyRedundancyCIPlotConfig(BaseConfig):
    type: Literal["ToyRedundancyCIPlot"] = "ToyRedundancyCIPlot"
    ci_alive_threshold: Probability = 0.1


class ToyRedundancyCIPlot(Metric[ToyRedundancyCIPlotConfig]):
    """The plot_ci active-subcomponent grid, from one forward over the whole vocab.

    Ignores the eval batches — the toy's input space is enumerable, so `compute()`
    runs every token through the model directly.
    """

    log_namespace = "figures"
    short_name = "ToyRedCIPlot"

    @override
    def reset(self) -> None:
        pass

    @override
    def update(self, ctx: MetricContext) -> None:
        return None

    @override
    def compute(self) -> MetricResult:
        target = cast(ToyModelRedundancyCopyTransformer, self.model.target_model)
        tokens = target.enumerate_inputs().to(self.device)
        with torch.no_grad():
            cached = self.model(tokens, cache_type="input")
            ci = self.model.calc_causal_importances(cached.cache, sampling="continuous")
        cis = {
            module: ci.lower_leaky[module].float().numpy(force=True)
            for module in sorted(ci.lower_leaky)
        }
        fig = plot_subcomponent_grid(cis)
        img = _render_figure(fig)
        plt.close(fig)
        return {"active_subcomponents": img}
