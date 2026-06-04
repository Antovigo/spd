"""Component-aware Llama-3.1: a vendored, checkpointable masked forward for decomposition.

A clean reimplementation of the HF `LlamaForCausalLM` architecture (RMSNorm, RoPE with the
"llama3" frequency scaling, grouped-query attention, SwiGLU MLP, untied `lm_head`), specialised
for decomposition training: no KV cache, full causal forward with `seq_len <= block_size`.

Mirrors `vendored/gpt2.py`: `componentize_llama` takes a frozen `VendoredLlama` plus a set of
core `Components`, swaps the decomposition-target leaves for in-tree `ComponentLinear`, and
re-points the mlp / attn / block / model forwards (via `__class__` reassignment) at variants
that thread a path-keyed `mask_infos` dict down to those leaves. Threading masks as a forward
argument — not via hooks — is what makes the masked forward checkpoint / FSDP / compile friendly.

Module paths match HF with the `model.` prefix stripped, e.g. `layers.18.mlp.gate_proj`, so
`from_hf_pretrained` is a direct `load_state_dict` after a prefix strip.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, cast, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from param_decomp.base_config import BaseConfig
from param_decomp.components import Components, EmbeddingComponents, LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.distributed import log0

# Buffers (target_weight / bias / rotary tables) are annotated for downstream typing but
# initialized via `register_buffer`, which pyright's uninitialized-instance check doesn't model
# unless the class is abstract (cf. the ABC-based `component_modules.py`).
# pyright: reportUninitializedInstanceVariable=false

MaskInfos = dict[str, ComponentsMaskInfo]
PreWeightActs = dict[str, Tensor]


class ComponentLinear(nn.Module):
    """In-tree, checkpointable replacement for a target `nn.Linear`: routes between the V/U
    component output and the frozen-target output as a pure function of `(x, mask_info)` (no
    side-channel hooks). `mask_info is None` → behave as the frozen target."""

    target_weight: Float[Tensor, "d_out d_in"]
    bias: Float[Tensor, "... d_out"] | None

    def __init__(self, components: LinearComponents, target_weight: Float[Tensor, "d_out d_in"]):
        super().__init__()
        self.components = components
        self.path = ""  # submodule path, set at swap time; keys this leaf's mask in mask_infos
        assert target_weight.shape == (components.d_out, components.d_in)
        self.register_buffer("target_weight", target_weight)
        self.register_buffer("bias", components.bias)

    def target_forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return F.linear(x, self.target_weight, self.bias)

    @override
    def forward(
        self,
        x: Tensor,
        mask_info: ComponentsMaskInfo | None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Tensor:
        if mask_info is None:
            assert component_acts_cache is None, "component_acts_cache needs an active mask"
            return self.target_forward(x)
        components_out = self.components(
            x,
            mask=mask_info.component_mask,
            weight_delta_and_mask=mask_info.weight_delta_and_mask,
            component_acts_cache=component_acts_cache,
        )
        if mask_info.routing_mask == "all":
            return components_out
        return torch.where(
            mask_info.routing_mask[..., None], components_out, self.target_forward(x)
        )


def _proj(
    module: nn.Module,
    x: Tensor,
    mask_infos: MaskInfos | None,
    collect: PreWeightActs | None,
    collect_outputs: PreWeightActs | None = None,
) -> Tensor:
    """Apply a (possibly component-decomposed) leaf, routing its mask in by path. For component
    leaves, optionally records the leaf's input into `collect` and output into `collect_outputs`."""
    if isinstance(module, ComponentLinear):
        if collect is not None:
            collect[module.path] = x
        mask_info = None if mask_infos is None else mask_infos.get(module.path)
        out = module(x, mask_info)
        if collect_outputs is not None:
            collect_outputs[module.path] = out
        return out
    return module(x)


class Llama3RopeScaling(BaseConfig):
    """The "llama3" RoPE frequency rescaling (Llama-3.1+). Reshapes inv_freq by wavelength:
    low-frequency components divided by `factor`, high-frequency untouched, smooth interpolation
    between. `original_max_position_embeddings` is the pre-scaling context the thresholds are
    defined against, NOT the actual sequence length."""

    factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


class VendoredLlamaConfig(BaseConfig):
    model_type: Literal["VendoredLlama"]
    block_size: int = 1024
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_key_value_heads: int = 8
    n_embd: int = 4096
    n_intermediate: int = 14336
    rope_theta: float = 500000.0
    rope_scaling: Llama3RopeScaling | None = Llama3RopeScaling()
    rms_norm_eps: float = 1e-5


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


# ----------------------------------------------------------------------------- component variants


class ComponentLlamaMLP(LlamaMLP):
    @override
    def forward(
        self,
        x: Float[Tensor, "... dim"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "... dim"]:
        gate = _proj(self.gate_proj, x, mask_infos, collect, collect_outputs)
        up = _proj(self.up_proj, x, mask_infos, collect, collect_outputs)
        return _proj(self.down_proj, F.silu(gate) * up, mask_infos, collect, collect_outputs)


class ComponentLlamaAttention(LlamaAttention):
    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "b t d"]:
        q = _proj(self.q_proj, x, mask_infos, collect, collect_outputs)
        k = _proj(self.k_proj, x, mask_infos, collect, collect_outputs)
        v = _proj(self.v_proj, x, mask_infos, collect, collect_outputs)
        return _proj(self.o_proj, self._attend(q, k, v), mask_infos, collect, collect_outputs)


class ComponentLlamaBlock(LlamaBlock):
    @override
    def forward(
        self,
        x: Float[Tensor, "b t d"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "b t d"]:
        attn: ComponentLlamaAttention = self.self_attn  # pyright: ignore[reportAssignmentType]
        mlp: ComponentLlamaMLP = self.mlp  # pyright: ignore[reportAssignmentType]
        x = x + attn(self.input_layernorm(x), mask_infos, collect, collect_outputs)
        x = x + mlp(self.post_attention_layernorm(x), mask_infos, collect, collect_outputs)
        return x


class ComponentLlama(VendoredLlama):
    """`VendoredLlama` whose forward threads a path-keyed `mask_infos` to in-tree components.

    Adopted via `__class__` reassignment by `componentize_llama` (no `__init__` of its own).
    `forward` returns logits, or the post-final-norm hidden state under `bypass_lm_head()`.
    """

    _bypass_lm_head: bool = False

    @override
    def forward(
        self,
        idx: Int[Tensor, "b t"],
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "b t vocab"] | Float[Tensor, "b t d"]:
        _b, t = idx.size()
        assert t <= self.config.block_size, f"seq len {t} > block size {self.config.block_size}"
        return self.forward_from_residual(
            self.embed_tokens(idx), 0, mask_infos, collect, collect_outputs
        )

    @property
    def decomposition_start_layer(self) -> int:
        """Lowest block index holding a decomposition target. Blocks below it are frozen and
        component-free, so the residual entering this block is identical across masked forwards
        and can be cached once (see `residual_at` / `forward_from_residual`)."""
        return min(int(p.split("layers.")[1].split(".")[0]) for p in self.component_modules)

    @torch.no_grad()
    def residual_at(self, idx: Int[Tensor, "b t"], layer: int) -> Float[Tensor, "b t d"]:
        """Clean residual entering block `layer`, run un-checkpointed under no_grad. Valid only
        for `layer <= decomposition_start_layer` (the prefix is frozen + component-free, so no
        masks/grad are needed and it is constant across this step's masked forwards)."""
        assert layer <= self.decomposition_start_layer, "prefix must be below all component sites"
        x = self.embed_tokens(idx)
        for block in self._layers[:layer]:
            x = block(x)
        return x

    def forward_from_residual(
        self,
        residual: Float[Tensor, "b t d"],
        start_layer: int,
        mask_infos: MaskInfos | None = None,
        collect: PreWeightActs | None = None,
        collect_outputs: PreWeightActs | None = None,
    ) -> Float[Tensor, "b t vocab"] | Float[Tensor, "b t d"]:
        """Run blocks `[start_layer:]` + final norm + head on a cached `residual`, threading
        masks. With `start_layer == 0` and `residual = embed_tokens(idx)` this is the full
        forward; with `start_layer == decomposition_start_layer` and a cached `residual_at` it
        skips the frozen prefix. Bit-identical either way (same ops on the suffix)."""
        x = residual
        blocks: list[ComponentLlamaBlock] = self._layers[start_layer:]  # pyright: ignore[reportAssignmentType]
        if self._use_activation_checkpointing and collect is None and collect_outputs is None:
            for block in blocks:
                x = checkpoint(block, x, mask_infos, use_reentrant=False)
        else:
            for block in blocks:
                x = block(x, mask_infos, collect, collect_outputs)
        x = self.norm(x)
        return x if self._bypass_lm_head else self.lm_head(x)

    def forward_with_pre_weight_acts(
        self, idx: Int[Tensor, "b t"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        collect: PreWeightActs = {}
        out = self(idx, mask_infos, collect)
        return out, collect

    def pre_weight_acts(self, idx: Int[Tensor, "b t"]) -> PreWeightActs:
        return self.forward_with_pre_weight_acts(idx)[1]

    def forward_with_output_acts(
        self, idx: Int[Tensor, "b t"], mask_infos: MaskInfos | None = None
    ) -> tuple[Tensor, PreWeightActs]:
        collect_outputs: PreWeightActs = {}
        out = self(idx, mask_infos, None, collect_outputs)
        return out, collect_outputs

    @contextmanager
    def bypass_lm_head(self) -> Iterator[Float[Tensor, "vocab d_model"]]:
        self._bypass_lm_head = True
        try:
            yield self.lm_head.weight
        finally:
            self._bypass_lm_head = False

    @property
    def component_modules(self) -> dict[str, ComponentLinear]:
        return {m.path: m for m in self.modules() if isinstance(m, ComponentLinear)}

    @property
    def components(self) -> dict[str, Components]:
        return {path: m.components for path, m in self.component_modules.items()}

    @property
    def module_to_c(self) -> dict[str, int]:
        return {path: m.components.C for path, m in self.component_modules.items()}

    @property
    def target_module_paths(self) -> list[str]:
        return list(self.component_modules)

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        return self.component_modules[module_name].target_weight

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        return {
            path: self.target_weight(path) - m.components.weight
            for path, m in self.component_modules.items()
        }

    def drop_components(self) -> None:
        """Free the per-site V/U params — for the CI pool, which holds no V/U.
        Only safe with a global CI fn (asserted: no embedding sites). See
        `ComponentGPT2.drop_components`."""
        assert not any(
            isinstance(m.components, EmbeddingComponents) for m in self.component_modules.values()
        ), "drop_components() needs V to convert token ids to acts for an embedding site"
        for m in self.component_modules.values():
            comp = m.components
            for pname in ("V", "U", "bias"):
                if hasattr(comp, pname) and getattr(comp, pname) is not None:
                    delattr(comp, pname)


def componentize_llama(model: VendoredLlama, components: dict[str, Components]) -> ComponentLlama:
    """In-place: freeze the target, swap decomposition-target leaves for component modules, and
    re-point the mlp / attn / block / model forwards to mask-threading variants.

    `components` is keyed by submodule path (e.g. `layers.18.mlp.gate_proj`)."""
    for param in model.parameters():
        param.requires_grad_(False)

    for path, comp in components.items():
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        target_module = getattr(parent, attr)
        assert isinstance(comp, LinearComponents), (
            f"vendored Llama only decomposes nn.Linear leaves; got {type(comp)} at {path}"
        )
        new = ComponentLinear(comp, target_module.weight.data)
        new.path = path
        setattr(parent, attr, new)

    for block in model._layers:
        block.self_attn.__class__ = ComponentLlamaAttention
        block.mlp.__class__ = ComponentLlamaMLP
        block.__class__ = ComponentLlamaBlock
    model.__class__ = ComponentLlama
    return cast(ComponentLlama, model)
