"""CI vs Weight Magnitude metric for visualizing component importance vs parameter norms."""

from pathlib import Path
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig, ResidMLPTaskConfig, TMSTaskConfig
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import LinearComponents
from spd.plotting import plot_ci_vs_weight_magnitude


class CIvsWeightMagnitude(Metric):
    """Scatter plot of max CI (over target inputs) vs weight magnitude per component.

    For each layer:
    - X-axis: Max CI over target inputs (per component)
    - Y-axis: Weight magnitude (||V|| * ||U||) per component
    - Color: Mean CI per component

    Target data generation follows TargetedCIHeatmap:
    - LM: Uses prompts from prompts_file
    - TMS/ResidMLP: One input per active_index
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        run_config: Config,
        device: str,
    ) -> None:
        self.model = model
        self.run_config = run_config
        self.device = device

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        pass

    @override
    def compute(self) -> dict[str, Image.Image]:
        target_cis = self._compute_target_cis()
        weight_magnitudes = self._compute_weight_magnitudes()

        max_cis: dict[str, Float[Tensor, "C"]] = {}
        mean_cis: dict[str, Float[Tensor, "C"]] = {}

        for layer_name, ci_tensor in target_cis.items():
            # Flatten all dimensions except C, then compute max/mean over samples
            ci_flat = ci_tensor.reshape(-1, ci_tensor.shape[-1])  # [N, C]
            max_cis[layer_name] = ci_flat.max(dim=0).values  # [C]
            mean_cis[layer_name] = ci_flat.mean(dim=0)  # [C]

        img = plot_ci_vs_weight_magnitude(
            max_cis=max_cis,
            weight_magnitudes=weight_magnitudes,
            mean_cis=mean_cis,
        )
        return {"ci_vs_weight_magnitude": img}

    def _compute_weight_magnitudes(self) -> dict[str, Float[Tensor, "C"]]:
        """Compute ||V|| * ||U|| for each component in each layer."""
        weight_magnitudes: dict[str, Float[Tensor, "C"]] = {}

        for layer_name, component in self.model.components.items():
            if not isinstance(component, LinearComponents):
                continue
            # V: (d_in, C), U: (C, d_out)
            v_norms = component.V.norm(dim=0)  # [C]
            u_norms = component.U.norm(dim=1)  # [C]
            weight_magnitudes[layer_name] = (v_norms * u_norms).detach().cpu()

        return weight_magnitudes

    def _compute_cis_from_batch(self, batch: Tensor) -> dict[str, Float[Tensor, "... C"]]:
        """Compute CIs for a batch of inputs."""
        batch = batch.to(self.device)

        with torch.no_grad():
            pre_weight_acts = self.model(batch, cache_type="input").cache
            ci = self.model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                detach_inputs=False,
                sampling=self.run_config.sampling,
            )

        return {name: vals.detach().cpu() for name, vals in ci.lower_leaky.items()}

    def _compute_target_cis(self) -> dict[str, Float[Tensor, "... C"]]:
        """Generate target data and compute CIs."""
        task_config = self.run_config.task_config

        match task_config:
            case LMTaskConfig():
                batch = self._generate_lm_target_batch(task_config)
            case TMSTaskConfig() | ResidMLPTaskConfig():
                batch = self._generate_toy_model_target_batch(task_config)
            case _:
                raise ValueError(f"Unsupported task config type: {type(task_config)}")

        return self._compute_cis_from_batch(batch)

    def _generate_lm_target_batch(self, task_config: LMTaskConfig) -> Tensor:
        """Generate target batch for LM: all prompts from prompts_file."""
        prompts_file = task_config.prompts_file
        assert prompts_file is not None, (
            "LM targeted mode requires prompts_file to be set in task_config"
        )
        prompts_path = Path(prompts_file)
        assert prompts_path.exists(), f"Prompts file not found: {prompts_path}"

        tokenizer = self._get_tokenizer()

        prompts = prompts_path.read_text().strip().split("\n")
        prompts = [p.strip() for p in prompts if p.strip()]

        encoded = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=task_config.max_seq_len,
            return_tensors="pt",
        )
        return encoded["input_ids"]

    def _generate_toy_model_target_batch(
        self,
        task_config: TMSTaskConfig | ResidMLPTaskConfig,
    ) -> Float[Tensor, "batch n_features"]:
        """Generate target batch for TMS/ResidMLP: one input per active_index."""
        n_features = self._get_n_features()
        active_indices = task_config.active_indices
        assert active_indices is not None, (
            "Targeted mode requires active_indices to be set in task_config"
        )

        batch = torch.zeros(len(active_indices), n_features)
        for i, idx in enumerate(active_indices):
            batch[i, idx] = 1.0

        return batch

    def _get_n_features(self) -> int:
        """Get n_features from the target model's config."""
        target_model = self.model.target_model
        model_config = getattr(target_model, "config", None)
        assert model_config is not None, "Target model must have a config attribute"
        n_features = getattr(model_config, "n_features", None)
        assert isinstance(n_features, int), "Target model must have config.n_features as int"
        return n_features

    def _get_tokenizer(self) -> Any:
        """Get tokenizer for LM experiments."""
        assert self.run_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(self.run_config.tokenizer_name)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id  # pyright: ignore[reportAttributeAccessIssue]
        return tokenizer
