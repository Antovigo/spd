"""Pre-RMSNorm logit lens comparing the original model against the rounded circuit."""

from functools import partial
from typing import Literal, override

import torch
from torch import Tensor, nn
from torch.distributed import ReduceOp
from transformers import AutoTokenizer

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import make_mask_infos
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_rounded_logit_lens


class RoundedLogitLensConfig(BaseConfig):
    """Rounds the *output* CI net at `rounding_threshold` (delta off) for the "rounded"
    (circuit-only) forward; `tokens` must each tokenize to a single id under
    `tokenizer_name`."""

    type: Literal["RoundedLogitLens"] = "RoundedLogitLens"
    tokenizer_name: str
    tokens: list[str]
    rounding_threshold: float


class RoundedLogitLens(Metric[RoundedLogitLensConfig]):
    """Per-block pre-`ln_f` projection onto `tokens`, original model vs rounded circuit.

    At every transformer block, captures the residual stream at the last token position
    (before the model's own final RMSNorm) and dots it with the tied unembedding rows for
    `tokens`, for both the frozen target model and the rounded (CI > threshold, delta off)
    circuit-only reconstruction. Assumes `model.target_model` exposes `h` (a `ModuleList` of
    blocks) and `lm_head` (an `nn.Linear`), matching `LlamaSimpleMLP`.
    """

    log_namespace = "logit_lens"
    slow = True
    short_name = "LogitLens"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        token_ids = []
        for token in self.cfg.tokens:
            ids = tokenizer.encode(token)
            assert len(ids) == 1, f"{token!r} does not tokenize to a single token: {ids}"
            token_ids.append(ids[0])
        self.token_ids = token_ids
        h = model.target_model.get_submodule("h")
        assert isinstance(h, nn.ModuleList)
        self.blocks: list[nn.Module] = list(h)
        lm_head = model.target_model.get_submodule("lm_head")
        assert isinstance(lm_head, nn.Linear)
        self.lm_head = lm_head
        super().bind(model=model, device=device)

    @override
    def reset(self) -> None:
        n_layer = len(self.blocks)
        n_tokens = len(self.token_ids)
        self.sum_logits: dict[str, Tensor] = {
            forward_type: torch.zeros(n_layer, n_tokens, device=self.device)
            for forward_type in ("original", "rounded")
        }
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @staticmethod
    def _capture_last_position(
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
        captured: dict[int, Tensor],
        layer_idx: int,
    ) -> None:
        del module, inputs
        captured[layer_idx] = output[:, -1, :].detach()

    @override
    def update(self, ctx: MetricContext) -> None:
        assert ctx.use_delta_component, "RoundedLogitLens requires use_delta_component"
        ci = ctx.ci.lower_leaky
        ci_sample = next(iter(ci.values()))
        leading_dims = ci_sample.shape[:-1]
        rounded_mask_infos = make_mask_infos(
            {k: (v > self.cfg.rounding_threshold).float() for k, v in ci.items()},
            weight_deltas_and_masks={
                layer: (
                    ctx.weight_deltas[layer],
                    torch.full(leading_dims, 0.0, device=ci_sample.device, dtype=ci_sample.dtype),
                )
                for layer in ci
            },
        )

        captured: dict[int, Tensor] = {}
        handles = [
            block.register_forward_hook(
                partial(self._capture_last_position, captured=captured, layer_idx=i)
            )
            for i, block in enumerate(self.blocks)
        ]
        try:
            self.model(ctx.batch, mask_infos=None)
            original_residuals = dict(captured)
            captured.clear()
            self.model(ctx.batch, mask_infos=rounded_mask_infos)
            rounded_residuals = dict(captured)
        finally:
            for handle in handles:
                handle.remove()

        token_ids = torch.tensor(self.token_ids, device=self.device)
        w_u = self.lm_head.weight[token_ids].float()  # [n_tokens, d_model]
        for forward_type, residuals in (
            ("original", original_residuals),
            ("rounded", rounded_residuals),
        ):
            for layer_idx, residual in residuals.items():
                self.sum_logits[forward_type][layer_idx] += (residual.float() @ w_u.T).sum(dim=0)
        self.n_examples += ci_sample.shape[0]
        return None

    @override
    def compute(self) -> MetricResult:
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        mean_logits = {
            forward_type: all_reduce(sums, op=ReduceOp.SUM) / n_examples
            for forward_type, sums in self.sum_logits.items()
        }
        out: dict[str, float] = {}
        for forward_type, layer_token_means in mean_logits.items():
            for layer_idx in range(layer_token_means.shape[0]):
                for token_idx, token in enumerate(self.cfg.tokens):
                    out[f"{forward_type}/layer{layer_idx}/{token.strip()}"] = layer_token_means[
                        layer_idx, token_idx
                    ].item()
        return {**out, "trajectory": plot_rounded_logit_lens(mean_logits, self.cfg.tokens)}
