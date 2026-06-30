"""Plain-`nn.Linear` transformer equivalent to an InterpBench `HookedTransformer`.

InterpBench ships its (Tracr-derived, SIIT-trained) models as `transformer_lens`
`HookedTransformer` weights — folded attention parameters (`W_Q [n_heads, d_model,
d_head]`, …) that are `nn.Parameter`s, not `nn.Linear` submodules, so PD can't find
decomposable matrices. `InterpBenchTransformer` re-expresses the same computation with
`nn.Linear` q/k/v/o + MLP modules (heads kept together in one matrix), copying the
weights in by unfolding. The forward is a faithful transcription of the HookedTransformer
math for the Tracr-derived models: no LayerNorm, additive positional embedding, optional
causal mask, scaled-dot-product attention. Embedding / positional / unembedding stay as
non-decomposed parameters (one-hot inputs, distribution-over-vocab outputs).

The unfolding is exact (float64 round-trip agrees to ~1e-13); see
`test_interpbench.py`.
"""

import pickle
from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int
from torch import Tensor

INTERPBENCH_HF_REPO = "cybershiptrooper/InterpBench"
INTERPBENCH_MODEL_CLASS = "param_decomp_lab.experiments.lm.interpbench.model.InterpBenchTransformer"

ACT_FNS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


class InterpBenchAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, causal: bool, attn_scale: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal = causal
        self.attn_scale = attn_scale
        self.q_proj = nn.Linear(d_model, n_heads * d_head)
        self.k_proj = nn.Linear(d_model, n_heads * d_head)
        self.v_proj = nn.Linear(d_model, n_heads * d_head)
        self.o_proj = nn.Linear(n_heads * d_head, d_model)

    @override
    def forward(self, x: Float[Tensor, "b t d_model"]) -> Float[Tensor, "b t d_model"]:
        split = {"h": self.n_heads, "d": self.d_head}
        q = rearrange(self.q_proj(x), "b t (h d) -> b t h d", **split)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d", **split)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d", **split)
        scores = einsum(q, k, "b q h d, b k h d -> b h q k") / self.attn_scale
        if self.causal:
            t = x.shape[1]
            future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        z = einsum(scores.softmax(dim=-1), v, "b h q k, b k h d -> b q h d")
        return self.o_proj(rearrange(z, "b t h d -> b t (h d)"))


class InterpBenchMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: str):
        super().__init__()
        assert act_fn in ACT_FNS, f"unsupported act_fn {act_fn!r}"
        self.act = ACT_FNS[act_fn]
        self.in_proj = nn.Linear(d_model, d_mlp)
        self.out_proj = nn.Linear(d_mlp, d_model)

    @override
    def forward(self, x: Float[Tensor, "b t d_model"]) -> Float[Tensor, "b t d_model"]:
        return self.out_proj(self.act(self.in_proj(x)))


class InterpBenchBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_mlp: int,
        causal: bool,
        attn_scale: float,
        act_fn: str,
    ):
        super().__init__()
        self.attn = InterpBenchAttention(d_model, n_heads, d_head, causal, attn_scale)
        self.mlp = InterpBenchMLP(d_model, d_mlp, act_fn)

    @override
    def forward(self, x: Float[Tensor, "b t d_model"]) -> Float[Tensor, "b t d_model"]:
        x = x + self.attn(x)
        return x + self.mlp(x)


class InterpBenchTransformer(nn.Module):
    """`nn.Linear`-based equivalent of a Tracr-derived InterpBench HookedTransformer.

    Build via `from_hf`. `forward` returns logits `[b, t, d_vocab_out]`. The per-block
    `attn.{q,k,v,o}_proj` and `mlp.{in,out}_proj` are the decomposable matrices;
    `embed` / `pos` / `unembed` are intentionally left as plain (non-decomposed)
    parameters.
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        assert cfg["normalization_type"] is None, (
            "converter supports only Tracr-derived InterpBench models (no LayerNorm); "
            f"got normalization_type={cfg['normalization_type']!r}"
        )
        assert not cfg["attn_only"], "expected MLP blocks present"
        assert cfg["positional_embedding_type"] == "standard", (
            f"converter assumes additive positional embeddings; got "
            f"{cfg['positional_embedding_type']!r}"
        )
        d_model, n_ctx = cfg["d_model"], cfg["n_ctx"]
        causal = cfg["attention_dir"] == "causal"
        attn_scale = cfg["d_head"] ** 0.5 if cfg["use_attn_scale"] else 1.0
        self.embed = nn.Embedding(cfg["d_vocab"], d_model)
        self.pos = nn.Parameter(torch.zeros(n_ctx, d_model))
        self.blocks = nn.ModuleList(
            InterpBenchBlock(
                d_model,
                cfg["n_heads"],
                cfg["d_head"],
                cfg["d_mlp"],
                causal,
                attn_scale,
                cfg["act_fn"],
            )
            for _ in range(cfg["n_layers"])
        )
        self.unembed = nn.Linear(d_model, cfg["d_vocab_out"])

    @override
    def forward(self, input_ids: Int[Tensor, "b t"]) -> Float[Tensor, "b t d_vocab_out"]:
        t = input_ids.shape[1]
        assert t <= self.pos.shape[0], f"sequence length {t} exceeds n_ctx {self.pos.shape[0]}"
        x = self.embed(input_ids) + self.pos[:t]
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    @classmethod
    def from_hf(
        cls, case_id: str, device: str = "cpu", repo: str = INTERPBENCH_HF_REPO
    ) -> "InterpBenchTransformer":
        """Download an InterpBench case, build the converted model, freeze it, set eval."""
        with open(hf_hub_download(repo, f"{case_id}/ll_model_cfg.pkl"), "rb") as f:
            cfg = pickle.load(f)
        state = torch.load(
            hf_hub_download(repo, f"{case_id}/ll_model.pth"), map_location="cpu", weights_only=False
        )
        model = cls(cfg)
        _load_hooked_weights(model, state)
        model.requires_grad_(False)
        return model.to(device).eval()


def _load_hooked_weights(model: InterpBenchTransformer, state: dict[str, Tensor]) -> None:
    """Copy HookedTransformer folded weights into the `nn.Linear` modules (exact unfolding).

    `W_Q/K/V [n_heads, d_model, d_head]` -> `Linear(d_model, n_heads*d_head)` with heads
    concatenated; `W_O [n_heads, d_head, d_model]` -> `Linear(n_heads*d_head, d_model)`;
    `W_in/W_out` transposed into their Linears. `nn.Linear` stores `[out, in]` and computes
    `x @ Wᵀ + b`, so each `W_*` (stored `[in, out]`) is transposed on copy.
    """
    with torch.no_grad():
        model.embed.weight.copy_(state["embed.W_E"])
        model.pos.copy_(state["pos_embed.W_pos"])
        model.unembed.weight.copy_(state["unembed.W_U"].T)
        model.unembed.bias.copy_(state["unembed.b_U"])
        for i, block in enumerate(model.blocks):
            assert isinstance(block, InterpBenchBlock)
            attn = f"blocks.{i}.attn."
            for proj, w in [
                (block.attn.q_proj, "W_Q"),
                (block.attn.k_proj, "W_K"),
                (block.attn.v_proj, "W_V"),
            ]:
                proj.weight.copy_(rearrange(state[attn + w], "h m d -> (h d) m"))
                proj.bias.copy_(rearrange(state[attn + "b_" + w[-1]], "h d -> (h d)"))
            block.attn.o_proj.weight.copy_(rearrange(state[attn + "W_O"], "h d m -> m (h d)"))
            block.attn.o_proj.bias.copy_(state[attn + "b_O"])
            mlp = f"blocks.{i}.mlp."
            block.mlp.in_proj.weight.copy_(state[mlp + "W_in"].T)
            block.mlp.in_proj.bias.copy_(state[mlp + "b_in"])
            block.mlp.out_proj.weight.copy_(state[mlp + "W_out"].T)
            block.mlp.out_proj.bias.copy_(state[mlp + "b_out"])
