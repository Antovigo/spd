from typing import Any, ClassVar, override

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel, SPDRunInfo


class ReferenceDecompositionSimilarity(Metric):
    """Compare learned decomposition against a reference decomposition via cosine similarity.

    For each target feature:
    1. Find the most active subcomponent per layer (highest CI)
    2. Compute rank-one matrix: V[:, c] @ U[c, :]
    3. Compare to reference model's equivalent via cosine similarity
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "target_solution_similarity"

    # Class-level cache to avoid reloading reference model each eval
    _reference_model: ClassVar[ComponentModel | None] = None
    _reference_model_path: ClassVar[str | None] = None

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        reference_model_path: str,
        target_features: list[int] | None,
        n_features: int | None = None,
        input_activation: float = 0.75,
    ) -> None:
        """Initialize the reference decomposition similarity metric.

        Args:
            model: The component model to evaluate.
            device: Device to run computations on.
            reference_model_path: Path to the reference model (wandb: or local path).
            target_features: Feature indices to evaluate. If None, uses all features.
            n_features: Number of input features. If None, inferred from first component
                (may be incorrect for models with varying layer input dimensions).
            input_activation: The input value to use for the active feature.
        """
        self.model = model
        self.device = device
        self.reference_model_path = reference_model_path
        self.input_activation = input_activation

        if n_features is not None:
            self.n_features = n_features
        else:
            # Infer n_features from model's first component input dimension
            # Warning: may be incorrect for models with varying layer input dimensions
            first_module = list(model.components.keys())[0]
            self.n_features = model.components[first_module].V.shape[0]

        # Fall back to all features if not specified
        self.target_features = (
            target_features if target_features is not None else list(range(self.n_features))
        )

        self._computed = False
        self._results: dict[str, float] = {}

    def _load_reference_model_if_needed(self) -> ComponentModel:
        """Lazy load with class-level caching."""
        cls = type(self)
        if cls._reference_model is None or cls._reference_model_path != self.reference_model_path:
            run_info = SPDRunInfo.from_path(self.reference_model_path)
            ref_model = ComponentModel.from_run_info(run_info)
            ref_model.to(self.device).eval().requires_grad_(False)
            cls._reference_model = ref_model
            cls._reference_model_path = self.reference_model_path
        return cls._reference_model

    def _generate_single_feature_input(self, feature_idx: int) -> Float[Tensor, "1 n_features"]:
        """Generate single input where only the specified feature is active."""
        batch = torch.zeros(1, self.n_features, device=self.device)
        batch[0, feature_idx] = self.input_activation
        return batch

    def _get_most_active_component_rank_one(
        self, model: ComponentModel, batch: Float[Tensor, "1 n_features"], module_name: str
    ) -> Float[Tensor, "d_in d_out"]:
        """Get rank-one matrix for the most active component."""
        with torch.no_grad():
            output = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=output.cache,
                detach_inputs=True,
                sampling="continuous",
            )

        # ci.lower_leaky[module_name] shape: [1, C] - squeeze batch dim
        ci_vec = ci.lower_leaky[module_name].squeeze(0)
        most_active_c = ci_vec.argmax().item()

        V = model.components[module_name].V  # [d_in, C]
        U = model.components[module_name].U  # [C, d_out]

        # W_c = V[:, c:c+1] @ U[c:c+1, :] -> [d_in, d_out]
        return V[:, most_active_c : most_active_c + 1] @ U[most_active_c : most_active_c + 1, :]

    def _compute_cosine_similarity(
        self,
        current: Float[Tensor, "d_in d_out"],
        reference: Float[Tensor, "d_in d_out"],
    ) -> float:
        """Absolute cosine similarity between flattened matrices."""
        current_flat = rearrange(current, "d_in d_out -> (d_in d_out)")
        ref_flat = rearrange(reference, "d_in d_out -> (d_in d_out)")

        current_norm = F.normalize(current_flat, p=2, dim=0)
        ref_norm = F.normalize(ref_flat, p=2, dim=0)

        return abs((current_norm * ref_norm).sum().item())

    @override
    def update(self, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, float]:
        if self._computed:
            return self._results

        ref_model = self._load_reference_model_if_needed()
        results: dict[str, float] = {}

        # Per-layer accumulators for computing means
        layer_sums: dict[str, float] = {}
        layer_counts: dict[str, int] = {}

        for feature in self.target_features:
            batch = self._generate_single_feature_input(feature)

            for module_name in self.model.components:
                current = self._get_most_active_component_rank_one(self.model, batch, module_name)
                ref = self._get_most_active_component_rank_one(ref_model, batch, module_name)

                sim = self._compute_cosine_similarity(current, ref)
                results[f"feature_{feature}/{module_name}"] = sim

                # Accumulate for layer means
                if module_name not in layer_sums:
                    layer_sums[module_name] = 0.0
                    layer_counts[module_name] = 0
                layer_sums[module_name] += sim
                layer_counts[module_name] += 1

        # Compute per-layer means
        all_sims: list[float] = []
        for module_name in layer_sums:
            mean_sim = layer_sums[module_name] / layer_counts[module_name]
            results[f"mean/{module_name}"] = mean_sim
            all_sims.append(mean_sim)

        # Compute overall mean
        if all_sims:
            results["mean/all"] = sum(all_sims) / len(all_sims)

        self._computed = True
        self._results = results
        return results
