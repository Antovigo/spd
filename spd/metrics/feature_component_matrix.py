from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_active_subcomponent_weights


class FeatureComponentMatrix(Metric):
    """Plot the weight matrices of subcomponents active on each target feature.

    For each target feature:
    1. Generate input with that feature active at a fixed value (e.g., 0.75)
    2. Compute CI to identify which subcomponents are active (CI > threshold)
    3. Plot the weight matrix (V[:, c] @ U[c, :]) for each active subcomponent
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        n_features: int,
        ci_threshold: float = 0.1,
        target_features: list[int] | None = None,
        input_activation: float = 0.75,
    ) -> None:
        """Initialize the feature component matrix metric.

        Args:
            model: The component model to evaluate.
            device: Device to run computations on.
            n_features: Total number of features in the input space.
            ci_threshold: Threshold for considering a component "active".
            target_features: If provided, only compute for these features.
            input_activation: The input value to use for the active feature.
        """
        self.model = model
        self.device = device
        self.n_features = n_features
        self.ci_threshold = ci_threshold
        self.input_activation = input_activation
        self.features_to_eval = (
            target_features if target_features is not None else list(range(n_features))
        )
        self._computed = False
        self._results: dict[str, dict[int, tuple[list[int], list[Tensor]]]] = {}

    def _generate_single_feature_input(
        self, feature_idx: int
    ) -> Float[Tensor, "1 n_features"]:
        """Generate single input where only the specified feature is active."""
        batch = torch.zeros(1, self.n_features, device=self.device)
        batch[0, feature_idx] = self.input_activation
        return batch

    @override
    def update(self, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, Image.Image]:
        if self._computed:
            return self._create_figures()

        # For each layer, store {feature_idx: (active_component_indices, weight_matrices)}
        for module_name in self.model.components:
            self._results[module_name] = {}

        for feature in self.features_to_eval:
            batch = self._generate_single_feature_input(feature)

            with torch.no_grad():
                output = self.model(batch, cache_type="input")
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=output.cache,
                    detach_inputs=True,
                    sampling="continuous",
                )

            for module_name, ci_vals in ci.lower_leaky.items():
                # ci_vals shape: [1, C] - squeeze batch dim
                ci_vec = ci_vals.squeeze(0)
                active_mask = ci_vec > self.ci_threshold
                active_indices = torch.where(active_mask)[0].tolist()

                # Get component weight matrices for active components
                component = self.model.components[module_name]
                V = component.V  # [d_in, C]
                U = component.U  # [C, d_out]

                weight_matrices = []
                for c in active_indices:
                    # W_c = V[:, c:c+1] @ U[c:c+1, :] -> [d_in, d_out]
                    W_c = V[:, c : c + 1] @ U[c : c + 1, :]
                    weight_matrices.append(W_c.cpu())

                self._results[module_name][feature] = (active_indices, weight_matrices)

        self._computed = True
        return self._create_figures()

    def _create_figures(self) -> dict[str, Image.Image]:
        figures = {}
        for module_name, feature_data in self._results.items():
            fig = plot_active_subcomponent_weights(
                feature_data=feature_data,
                module_name=module_name,
                ci_threshold=self.ci_threshold,
                input_activation=self.input_activation,
            )
            figures[f"active_subcomponent_weights/{module_name}"] = fig
        return figures
