"""The plain Llama-3.1 architecture (RMSNorm, RoPE with the "llama3" frequency scaling,
grouped-query attention, SwiGLU MLP, untied `lm_head`), specialised for decomposition training:
no KV cache, full causal forward with `seq_len <= block_size`. `componentize_llama`
(in `components.py`) turns a frozen `VendoredLlama` into a mask-threading `ComponentLlama`.

Module paths match HF with the `model.` prefix stripped (e.g. `layers.18.mlp.gate_proj`), so
`from_hf_pretrained` is a direct `load_state_dict` after a prefix strip.
"""

import math
from typing import Any, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from param_decomp_lab.distributed import log0
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)

# rotary buffers are register_buffer-initialized; pyright can't model that on a non-abstract class.
# pyright: reportUninitializedInstanceVariable=false


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotary_cos_sin(
    head_dim: int, n_ctx: int, theta: float, scaling: Llama3RopeScaling | None
) -> tuple[Tensor, Tensor]:
    """HF-equivalent RoPE tables: cos/sin of shape [n_ctx, head_dim] (the [freqs, freqs]
    concat convention paired with rotate_half)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if scaling is not None:
        old_ctx = scaling.original_max_position_embeddings
        low_wavelen = old_ctx / scaling.low_freq_factor
        high_wavelen = old_ctx / scaling.high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_wavelen, inv_freq / scaling.factor, inv_freq)
        smooth = (old_ctx / wavelen - scaling.low_freq_factor) / (
            scaling.high_freq_factor - scaling.low_freq_factor
        )
        smoothed = (1 - smooth) * inv_freq_llama / scaling.factor + smooth * inv_freq_llama
        is_medium = ~(wavelen < high_wavelen) * ~(wavelen > low_wavelen)
        inv_freq = torch.where(is_medium, smoothed, inv_freq_llama)
    pos = torch.arange(n_ctx, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [n_ctx, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [n_ctx, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: Tensor) -> Tensor:
    n = x.shape[-1] // 2
    return torch.cat([-x[..., n:], x[..., :n]], dim=-1)


class LlamaAttention(nn.Module):
    # q/k/v are separate projections (k/v narrower under GQA) — HF Llama is already unfused
    # (unlike GPT-2's fused c_attn), so each is an independent decomposition target with no
    # split step needed.
    rotary_cos: Tensor
    rotary_sin: Tensor

    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_key_value_heads
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        cos, sin = _rotary_cos_sin(
            self.head_dim, config.block_size, config.rope_theta, config.rope_scaling
        )
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, T, _ = q.shape
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        cos = self.rotary_cos[:T].to(q.dtype)
        sin = self.rotary_sin[:T].to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)


class LlamaMLP(nn.Module):
    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.n_intermediate, bias=False)
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=False)

    @override
    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    @override
    def forward(self, x: Float[Tensor, "b t d"]) -> Float[Tensor, "b t d"]:
        h = self.input_layernorm(x)
        a = self.self_attn
        x = x + a.o_proj(a._attend(a.q_proj(h), a.k_proj(h), a.v_proj(h)))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VendoredLlama(nn.Module):
    """Plain Llama-3.1 target (untied `lm_head`). `componentize_llama` turns it into a
    `ComponentLlama` with a mask-threading forward."""

    def __init__(self, config: VendoredLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self._layers: list[LlamaBlock] = [LlamaBlock(config) for _ in range(config.n_layer)]
        self.layers = nn.ModuleList(self._layers)
        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._use_activation_checkpointing: bool = False

    def enable_activation_checkpointing(self) -> None:
        self._use_activation_checkpointing = True

    @override
    def forward(self, idx: Int[Tensor, "b t"]) -> Float[Tensor, "b t vocab"]:
        _b, t = idx.size()
        assert t <= self.config.block_size, f"seq len {t} > block size {self.config.block_size}"
        x = self.embed_tokens(idx)
        for block in self._layers:
            x = block(x)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_hf_pretrained(cls, model_name: str, block_size: int = 1024) -> "VendoredLlama":
        from transformers import LlamaForCausalLM

        log0(f"loading HF weights into vendored Llama: {model_name}")
        hf = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        hf_cfg: Any = hf.config  # transformers config attrs are dynamic; type as Any
        assert hf_cfg.model_type == "llama", f"expected a llama config, got {hf_cfg.model_type}"
        rs = hf_cfg.rope_scaling
        scaling = (
            Llama3RopeScaling(
                factor=rs["factor"],
                low_freq_factor=rs["low_freq_factor"],
                high_freq_factor=rs["high_freq_factor"],
                original_max_position_embeddings=rs["original_max_position_embeddings"],
            )
            if rs is not None
            else None
        )
        assert not hf_cfg.tie_word_embeddings, "vendored Llama assumes an untied lm_head"
        config = VendoredLlamaConfig(
            model_type="VendoredLlama",
            block_size=block_size,
            vocab_size=hf_cfg.vocab_size,
            n_layer=hf_cfg.num_hidden_layers,
            n_head=hf_cfg.num_attention_heads,
            n_key_value_heads=hf_cfg.num_key_value_heads,
            n_embd=hf_cfg.hidden_size,
            n_intermediate=hf_cfg.intermediate_size,
            rope_theta=hf_cfg.rope_theta,
            rope_scaling=scaling,
            rms_norm_eps=hf_cfg.rms_norm_eps,
        )
        model = cls(config)
        stripped = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        # persistent=False rotary buffers are absent from both sides; nothing real may be missing.
        assert not missing, f"missing keys loading HF Llama: {missing}"
        assert not unexpected, f"unexpected keys loading HF Llama: {unexpected}"
        del hf
        return model
