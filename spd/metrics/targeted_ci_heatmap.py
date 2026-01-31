"""Targeted CI heatmap metric for visualizing causal importances on target vs nontarget data."""

from pathlib import Path
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig, ResidMLPTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.prompts_dataset import create_prompts_data_loader
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.plotting import plot_targeted_ci_heatmaps


class TargetedCIHeatmap(Metric):
    """Visualize causal importances comparing target vs nontarget data.

    This is a slow eval metric that generates controlled inputs for both target
    and nontarget data, then creates a heatmap visualization comparing the CIs.

    For ResidMLP: Target inputs have one row per active_index (single feature active).
    For LM: Target inputs are the prompts from the prompts_file.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        run_config: Config,
        device: str,
        n_nontarget_examples: int,
    ) -> None:
        self.model = model
        self.run_config = run_config
        self.device = device
        self.n_nontarget_examples = n_nontarget_examples

    @override
    def update(self, *, ci: CIOutputs, **_: Any) -> None:
        # We don't accumulate from eval batches - we generate our own controlled inputs
        pass

    @override
    def compute(self) -> dict[str, Image.Image]:
        target_cis, target_labels = self._compute_target_cis()
        nontarget_cis, nontarget_labels = self._compute_nontarget_cis()

        img = plot_targeted_ci_heatmaps(
            target_cis=target_cis,
            nontarget_cis=nontarget_cis,
            n_nontarget_examples=self.n_nontarget_examples,
            target_labels=target_labels,
            nontarget_labels=nontarget_labels,
        )
        return {"targeted_ci_heatmap": img}

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

    def _compute_target_cis(self) -> tuple[dict[str, Float[Tensor, "... C"]], list[str]]:
        """Generate target data and compute CIs. Returns (cis, labels)."""
        task_config = self.run_config.task_config
        assert isinstance(task_config, ResidMLPTaskConfig | LMTaskConfig), (
            f"TargetedCIHeatmap only supports ResidMLPTaskConfig or LMTaskConfig, "
            f"got {type(task_config)}"
        )

        if isinstance(task_config, ResidMLPTaskConfig):
            batch, labels = self._generate_resid_mlp_target_batch(task_config)
        else:
            batch, labels = self._generate_lm_target_batch(task_config)

        cis = self._compute_cis_from_batch(batch)
        return cis, labels

    def _compute_nontarget_cis(self) -> tuple[dict[str, Float[Tensor, "... C"]], list[str]]:
        """Generate nontarget data and compute CIs. Returns (cis, labels)."""
        nontarget_task_config = self.run_config.nontarget_task_config
        assert nontarget_task_config is not None, (
            "TargetedCIHeatmap requires nontarget_task_config to be set"
        )
        assert isinstance(nontarget_task_config, ResidMLPTaskConfig | LMTaskConfig), (
            f"TargetedCIHeatmap only supports ResidMLPTaskConfig or LMTaskConfig, "
            f"got {type(nontarget_task_config)}"
        )

        if isinstance(nontarget_task_config, ResidMLPTaskConfig):
            batch, labels = self._generate_resid_mlp_nontarget_batch(nontarget_task_config)
        else:
            batch, labels = self._generate_lm_nontarget_batch(nontarget_task_config)

        cis = self._compute_cis_from_batch(batch)
        return cis, labels

    def _get_n_features(self) -> int:
        """Get n_features from the target model's config."""
        target_model = self.model.target_model
        model_config = getattr(target_model, "config", None)
        assert model_config is not None, "ResidMLP target model must have a config attribute"
        n_features = getattr(model_config, "n_features", None)
        assert isinstance(n_features, int), (
            "ResidMLP target model must have config.n_features as int"
        )
        return n_features

    def _generate_resid_mlp_target_batch(
        self,
        task_config: ResidMLPTaskConfig,
    ) -> tuple[Float[Tensor, "batch n_features"], list[str]]:
        """Generate target batch for ResidMLP: one input per active_index.

        Creates inputs where exactly one feature is active, with one row per
        feature in active_indices.

        Returns (batch, labels) where labels are the feature indices.
        """
        n_features = self._get_n_features()
        active_indices = task_config.active_indices
        assert active_indices is not None, (
            "ResidMLP targeted mode requires active_indices to be set in task_config"
        )

        # Create one input per active index with that feature set to 1.0
        batch = torch.zeros(len(active_indices), n_features)
        labels = []
        for i, idx in enumerate(active_indices):
            batch[i, idx] = 1.0
            labels.append(f"feat {idx}")

        return batch, labels

    def _generate_resid_mlp_nontarget_batch(
        self,
        task_config: ResidMLPTaskConfig,
    ) -> tuple[Float[Tensor, "batch n_features"], list[str]]:
        """Generate nontarget batch for ResidMLP experiments.

        Returns (batch, labels) where labels show which features are active in each row.
        """
        n_features = self._get_n_features()

        dataset = ResidMLPDataset(
            n_features=n_features,
            feature_probability=task_config.feature_probability,
            device="cpu",
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=task_config.data_generation_type,
            active_indices=task_config.active_indices,
        )
        batch, _ = dataset.generate_batch(self.n_nontarget_examples)

        # Generate labels showing which features are active in each row
        labels = []
        for row in batch:
            active = (row != 0).nonzero(as_tuple=True)[0].tolist()
            if len(active) == 0:
                labels.append("none")
            elif len(active) <= 3:
                labels.append(",".join(str(i) for i in active))
            else:
                labels.append(f"{active[0]},...,{active[-1]}")

        return batch, labels

    def _get_tokenizer(self) -> Any:
        """Get tokenizer for LM experiments."""
        assert self.run_config.tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(self.run_config.tokenizer_name)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id  # pyright: ignore[reportAttributeAccessIssue]
        return tokenizer

    def _tokens_to_labels(self, tokens: Tensor, tokenizer: Any) -> list[str]:
        """Convert token tensor to list of string labels (one per flattened position)."""
        # tokens shape: (batch, seq_len)
        labels = []
        for batch_idx in range(tokens.shape[0]):
            for pos_idx in range(tokens.shape[1]):
                token_id = int(tokens[batch_idx, pos_idx].item())
                token_str = tokenizer.decode([token_id])
                # Clean up the token string for display
                token_str = token_str.replace("\n", "\\n").replace("\t", "\\t")
                if len(token_str) > 10:
                    token_str = token_str[:8] + ".."
                labels.append(f"{batch_idx}:{token_str}")
        return labels

    def _generate_lm_target_batch(self, task_config: LMTaskConfig) -> tuple[Tensor, list[str]]:
        """Generate target batch for LM: all prompts from prompts_file.

        Returns (batch, labels) where labels are the tokens at each position.
        """
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
        tokens = encoded["input_ids"]
        labels = self._tokens_to_labels(tokens, tokenizer)

        return tokens, labels

    def _generate_lm_nontarget_batch(self, task_config: LMTaskConfig) -> tuple[Tensor, list[str]]:
        """Generate nontarget batch for LM experiments.

        Returns (batch, labels) where labels are the tokens at each position.
        """
        if task_config.prompts_file is not None:
            assert self.run_config.tokenizer_name is not None
            loader, tokenizer = create_prompts_data_loader(
                prompts_file=Path(task_config.prompts_file),
                tokenizer_name=self.run_config.tokenizer_name,
                max_seq_len=task_config.max_seq_len,
                batch_size=self.n_nontarget_examples,
                dist_state=None,
                seed=self.run_config.seed + 42,
            )
            batch_data = next(iter(loader))
            tokens = batch_data["input_ids"]
        else:
            assert task_config.dataset_name is not None, (
                "nontarget_task_config must have either prompts_file or dataset_name set"
            )
            data_config = DatasetConfig(
                name=task_config.dataset_name,
                hf_tokenizer_path=self.run_config.tokenizer_name,
                split=task_config.train_data_split,
                n_ctx=task_config.max_seq_len,
                is_tokenized=task_config.is_tokenized,
                streaming=task_config.streaming,
                column_name=task_config.column_name,
                shuffle_each_epoch=False,
            )
            loader, tokenizer = create_data_loader(
                dataset_config=data_config,
                batch_size=self.n_nontarget_examples,
                buffer_size=task_config.buffer_size,
                global_seed=self.run_config.seed + 42,
            )
            batch_data = next(iter(loader))
            # Handle dict-style batches from HuggingFace datasets
            tokens = (
                batch_data[data_config.column_name] if isinstance(batch_data, dict) else batch_data
            )

        labels = self._tokens_to_labels(tokens, tokenizer)
        return tokens, labels
