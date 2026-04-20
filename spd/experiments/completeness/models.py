from dataclasses import dataclass
from typing import Any, override

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import Dataset

from spd.experiments.completeness.configs import CompletenessModelConfig, CompletenessTrainConfig
from spd.interfaces import LoadableModule, RunInfo
from spd.spd_types import ModelPath


@dataclass
class CompletenessTargetRunInfo(RunInfo[CompletenessTrainConfig]):
    config_class = CompletenessTrainConfig
    config_filename = "completeness_train_config.yaml"
    checkpoint_filename = "completeness.pth"


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention. No MLP, no LayerNorm."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**-0.5

    @override
    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        attn_scores = einops.einsum(q, k, "b sq d, b sk d -> b sq sk") * self.scale

        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        attn_scores = attn_scores + causal_mask

        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = einops.einsum(attn_weights, v, "b sq sk, b sk d -> b sq d")
        return self.W_O(attn_out)


class RedundantCopyTransformer(LoadableModule):
    def __init__(self, config: CompletenessModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [SingleHeadAttention(config.d_model) for _ in range(config.n_layers)]
        )
        self.linear = nn.Linear(config.d_model, config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.layer_dropout_p = 0.0

    @override
    def forward(
        self,
        tokens: Int[Tensor, "batch seq"],
        **_: Any,
    ) -> Float[Tensor, "batch vocab_size"]:
        batch_size = tokens.shape[0]
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        if self.training and self.layer_dropout_p > 0:
            n_layers = len(self.layers)
            drop_mask = torch.rand(n_layers) < self.layer_dropout_p
            if drop_mask.all():
                keep_idx = torch.randint(n_layers, ())
                drop_mask[keep_idx] = False
        else:
            drop_mask = None

        for i, layer in enumerate(self.layers):
            if drop_mask is not None and drop_mask[i]:
                continue
            x = x + layer(x)

        x = self.linear(x)
        logits = self.unembed(x)
        # Return logits at position 1 (the eq_token position where we predict X)
        assert logits.shape == (batch_size, self.config.seq_len, self.config.vocab_size)
        return logits[:, 1, :]

    @classmethod
    @override
    def from_run_info(
        cls, run_info: RunInfo[CompletenessTrainConfig]
    ) -> "RedundantCopyTransformer":
        model = cls(config=run_info.config.completeness_model_config)
        model.load_state_dict(
            torch.load(run_info.checkpoint_path, weights_only=True, map_location="cpu")
        )
        return model

    @classmethod
    @override
    def from_pretrained(cls, path: ModelPath) -> "RedundantCopyTransformer":
        run_info = CompletenessTargetRunInfo.from_path(path)
        return cls.from_run_info(run_info)


class CopyTaskDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, vocab_size: int, eq_token: int, device: str | torch.device):
        self.vocab_size = vocab_size
        self.eq_token = eq_token
        self.device = device

    def __len__(self) -> int:
        return 2**31

    def generate_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        x = torch.randint(1, self.vocab_size, (batch_size,), device=self.device)
        tokens = torch.stack([x, torch.full_like(x, self.eq_token)], dim=1)
        return tokens, x
