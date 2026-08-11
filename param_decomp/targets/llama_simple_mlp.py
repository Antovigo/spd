"""`LlamaSimpleMLP` pile-pretrained target — the second `DecomposedModel` implementation.

Torch reference (read-only, ground truth):
`param_decomp/experiments/lm/pretrain/models/llama_simple_mlp.py`, weights from
pretrain run `goodfire/spd/runs/t-9d2b8f02`. Llama-style pre-RMSNorm blocks under a
GPT2-style module tree `h.{i}.`: rotary GQA attention (`rotary_dim == head_dim`,
plain base-`rotary_base` rotate-half RoPE — NOT llama3-rescaled) and a GELU(tanh) MLP
`c_fc -> gelu -> down_proj`. `wte` and `lm_head` are tied; no biases anywhere.

The torch RoPE construction (`freq = base**(i/(rd/2))` tiled `.repeat(2)`,
`rotate_every_two` with `rotary_adjacent_pairs=False`) is exactly the rotate-half RoPE
of `param_decomp.vendored_jax.llama.rope_cos_sin`/`apply_rope` with `inv_freq = base**(-2i/hd)` —
pinned by the torch-fixture equivalence test (`tests/simple_mlp_equivalence/`).

Decomposed sites are torch-module-path named: `h.{i}.attn.{q,k,v,o}_proj`,
`h.{i}.mlp.c_fc`, `h.{i}.mlp.down_proj`, each with its own C. q/k/v sites are
decomposed BEFORE RoPE/SDPA; o after; c_fc before the GELU; down_proj after. The
model is the full frozen network — embedding through every block to the (tied) LM head;
blocks without a decomposed site run the plain frozen path.

Weights load from the torch pretrain cache
(`<data_root>/pretrain_cache/<project>-<run_id>/`), converted once to
safetensors by `tools/convert_llama_simple_mlp_checkpoint.py` (torch venv).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.core import family
from param_decomp.core.components import ComponentStacks, SiteC, SiteDims, SiteSpec, site_out
from param_decomp.core.family import ArchFamily
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
)
from param_decomp.targets.glu_transformer import FrozenAttn
from param_decomp.targets.losses import kl_per_position
from param_decomp.targets.transformer_taps import (
    BlockTap,
    PostAttentionResidual,
    ResidualBoundary,
    SiteOutput,
    TransformerPoint,
    TransformerTapGrammar,
    attention_input_tap_key,
    attention_output_tap_key,
    mlp_hidden_tap_key,
    mlp_input_tap_key,
    site_output_tap_key,
)
from param_decomp.vendored_jax.llama import rms_norm

# Plain-GELU MLP (LlamaSimpleMLP). The family's matrix vocabulary — the authored c-spec
# keys (lab-side) are typed by it, so a c-spec key and a target matrix cannot drift.
SimpleMlpMatrix = Literal["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]

KIND_ORDER: tuple[str, ...] = get_args(SimpleMlpMatrix)
"""Within-layer canonical site order = computation order, DERIVED from the `SimpleMlpMatrix`
vocabulary. The canonical site order (`site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KINDS = ("c_fc", "down_proj")
assert KIND_ORDER == ATTN_KINDS + MLP_KINDS, KIND_ORDER


def _hidden_acts_reconstruction_dependencies(point: TransformerPoint) -> frozenset[str]:
    """Same-block decomposed kinds whose output can influence ``point``."""
    match point:
        case ResidualBoundary():
            return frozenset()
        case PostAttentionResidual():
            return frozenset(ATTN_KINDS)
        case BlockTap(name=name):
            match name:
                case "attn_in":
                    return frozenset()
                case "attn_out":
                    return frozenset(("q_proj", "k_proj", "v_proj"))
                case "mlp_in":
                    return frozenset(ATTN_KINDS)
                case "mlp_hidden":
                    return frozenset((*ATTN_KINDS, "c_fc"))
                case _:
                    raise AssertionError(name)
        case SiteOutput(name=name):
            _block, kind = parse_site_name(name)
            match kind:
                case "q_proj" | "k_proj" | "v_proj":
                    return frozenset((kind,))
                case "o_proj":
                    return frozenset(ATTN_KINDS)
                case "c_fc":
                    return frozenset((*ATTN_KINDS, "c_fc"))
                case "down_proj":
                    return frozenset(KIND_ORDER)
                case _:
                    raise AssertionError(kind)


SITE_NAME_PATTERN = re.compile(
    r"^h\.(\d+)\.(?:attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(c_fc|down_proj))$"
)


@dataclass(frozen=True)
class LlamaSimpleMLPConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    n_intermediate: int
    rotary_base: float
    rms_norm_eps: float
    n_ctx: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_rep(self) -> int:
        return self.n_head // self.n_kv_head


def config_from_model_config_dict(raw: dict[str, object]) -> LlamaSimpleMLPConfig:
    """Parse the pretrain run's `model_config.yaml` dict, refusing every torch-config
    variant this port does not implement (biases, adjacent-pair rotary, merged-qkv
    non-GQA attention). `flash_attention` is ignored: both torch paths compute the
    same causal SDPA math."""
    assert raw["model_type"] == "LlamaSimpleMLP", raw["model_type"]
    assert raw["use_grouped_query_attention"] is True, "merged-qkv (c_attn) unsupported"
    assert raw["attn_bias"] is False and raw["mlp_bias"] is False, "biases unsupported"
    assert raw["rotary_adjacent_pairs"] is False, "adjacent-pair rotary unsupported"
    cfg = LlamaSimpleMLPConfig(
        vocab_size=int(raw["vocab_size"]),  # pyright: ignore[reportArgumentType]
        n_layer=int(raw["n_layer"]),  # pyright: ignore[reportArgumentType]
        n_head=int(raw["n_head"]),  # pyright: ignore[reportArgumentType]
        n_kv_head=int(raw["n_key_value_heads"]),  # pyright: ignore[reportArgumentType]
        n_embd=int(raw["n_embd"]),  # pyright: ignore[reportArgumentType]
        n_intermediate=int(raw["n_intermediate"]),  # pyright: ignore[reportArgumentType]
        rotary_base=float(raw["rotary_base"]),  # pyright: ignore[reportArgumentType]
        rms_norm_eps=float(raw["rms_norm_eps"]),  # pyright: ignore[reportArgumentType]
        n_ctx=int(raw["n_ctx"]),  # pyright: ignore[reportArgumentType]
    )
    assert cfg.n_embd % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0, cfg
    # the torch class forces rotary_dim = head_dim regardless of config; insist the
    # config agrees so nothing is silently overridden
    assert raw["rotary_dim"] == cfg.head_dim, (raw["rotary_dim"], cfg.head_dim)
    assert raw["block_size"] == cfg.n_ctx, (raw["block_size"], cfg.n_ctx)
    return cfg


def plain_rope_inv_freq(cfg: LlamaSimpleMLPConfig) -> Float[Array, " hd2"]:
    """`base**(-2i/head_dim)` fp32 — the torch `calculate_sin_cos_rotary` frequencies
    (`1 / base**(i/(rd/2))`), no llama3 rescaling."""
    exponents = jnp.arange(0, cfg.head_dim, 2, dtype=jnp.float32) / cfg.head_dim
    return 1.0 / (cfg.rotary_base**exponents)


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    submodule = "attn" if kind in ATTN_KINDS else "mlp"
    return f"h.{layer}.{submodule}.{kind}"


def parse_site_name(name: str) -> tuple[int, str]:
    """`h.{i}.{attn,mlp}.{kind}` -> (layer, kind); rejects anything else (including
    kind/submodule mismatches like `attn.c_fc`)."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, (
        f"not a simple_mlp site: {name!r} (sites are h.{{i}}.attn.{{q|k|v|o}}_proj"
        f" / h.{{i}}.mlp.{{c_fc|down_proj}})"
    )
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


FAMILY = ArchFamily("simple_mlp", KIND_ORDER, site_name, parse_site_name)
"""This target's matrix grammar as data — the vocabulary + name renderer the tiled
`simple_mlp` c-specs resolve against."""


def site_dims(cfg: LlamaSimpleMLPConfig, kind: str) -> SiteDims:
    """Dimensions of one per-layer matrix in right-mult orientation."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    match kind:
        case "q_proj":
            return SiteDims(d_in=d, d_out=qd)
        case "k_proj" | "v_proj":
            return SiteDims(d_in=d, d_out=kvd)
        case "o_proj":
            return SiteDims(d_in=qd, d_out=d)
        case "c_fc":
            return SiteDims(d_in=d, d_out=di)
        case "down_proj":
            return SiteDims(d_in=di, d_out=d)
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    return family.canonical_site_cs(FAMILY, site_cs)


def site_specs(cfg: LlamaSimpleMLPConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    return family.site_specs(FAMILY, site_cs, lambda kind: site_dims(cfg, kind), cfg.n_layer)


# ----------------------------- frozen layers -----------------------------


class SimpleMLPLayer(eqx.Module):
    """One layer's frozen weights — norms, rotary GQA attention (`FrozenAttn` is
    target-agnostic: weights + RoPE/SDPA core), GELU MLP. Decomposed sites read their
    frozen target W from here at forward time; weights pass as a runtime arg."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wfc: Float[Array, "di d"]
    Wdown: Float[Array, "d di"]


def _frozen_site_weight(layer: SimpleMLPLayer, kind: str) -> Array:
    match kind:
        case "q_proj":
            return layer.attn.wq
        case "k_proj":
            return layer.attn.wk
        case "v_proj":
            return layer.attn.wv
        case "o_proj":
            return layer.attn.wo
        case "c_fc":
            return layer.Wfc
        case "down_proj":
            return layer.Wdown
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- forwards -----------------------------


def _gelu_tanh(x: Array) -> Array:
    """The torch reference's `NewGELU` (tanh approximation), exactly
    `jax.nn.gelu(approximate=True)` — pinned by the torch-fixture equivalence test."""
    return jax.nn.gelu(x, approximate=True)


def _clean_mlp_out(layer: SimpleMLPLayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return _gelu_tanh(mlp_in @ layer.Wfc.T) @ layer.Wdown.T


def _clean_block(layer: SimpleMLPLayer, x: Array, inv_freq: Array, eps: float) -> Array:
    x = x + layer.attn(rms_norm(x, layer.ln1, eps), inv_freq)
    return x + _clean_mlp_out(layer, rms_norm(x, layer.ln2, eps))


class _SimpleTap(Enum):
    RESIDUAL_IN = "residual_in"
    QKV_INPUT = "qkv_input"
    Q_OUTPUT = "q_output"
    K_OUTPUT = "k_output"
    V_OUTPUT = "v_output"
    ATTENTION_OUTPUT = "attention_output"
    O_OUTPUT = "o_output"
    POST_ATTENTION_RESIDUAL = "post_attention_residual"
    MLP_INPUT = "mlp_input"
    FC_OUTPUT = "fc_output"
    DOWN_INPUT = "down_input"
    DOWN_OUTPUT = "down_output"
    RESIDUAL_OUT = "residual_out"


@dataclass(frozen=True, kw_only=True)
class _SimpleCaptureSource:
    block: int
    tap: _SimpleTap


SimpleCaptureSources = tuple[_SimpleCaptureSource, ...]


@dataclass(frozen=True, kw_only=True)
class _SimpleMLPBlockActivations:
    residual_in: Array
    qkv_input: Array
    q_output: Array
    k_output: Array
    v_output: Array
    attention_output: Array
    o_output: Array
    post_attention_residual: Array
    mlp_input: Array
    fc_output: Array
    down_input: Array
    down_output: Array
    residual_out: Array


def _capture_source_for_point(point: TransformerPoint) -> _SimpleCaptureSource:
    match point:
        case ResidualBoundary(boundary=0):
            return _SimpleCaptureSource(block=0, tap=_SimpleTap.RESIDUAL_IN)
        case ResidualBoundary(boundary=boundary):
            return _SimpleCaptureSource(block=boundary - 1, tap=_SimpleTap.RESIDUAL_OUT)
        case PostAttentionResidual(block=block):
            return _SimpleCaptureSource(block=block, tap=_SimpleTap.POST_ATTENTION_RESIDUAL)
        case BlockTap(name=name, block=block):
            value_by_name = {
                "attn_in": _SimpleTap.QKV_INPUT,
                "attn_out": _SimpleTap.ATTENTION_OUTPUT,
                "mlp_in": _SimpleTap.MLP_INPUT,
                "mlp_hidden": _SimpleTap.DOWN_INPUT,
            }
            return _SimpleCaptureSource(block=block, tap=value_by_name[name])
        case SiteOutput(name=name, block=block):
            _layer, kind = parse_site_name(name)
            match kind:
                case "q_proj":
                    value = _SimpleTap.Q_OUTPUT
                case "k_proj":
                    value = _SimpleTap.K_OUTPUT
                case "v_proj":
                    value = _SimpleTap.V_OUTPUT
                case "o_proj":
                    value = _SimpleTap.O_OUTPUT
                case "c_fc":
                    value = _SimpleTap.FC_OUTPUT
                case "down_proj":
                    value = _SimpleTap.DOWN_OUTPUT
                case _:
                    raise AssertionError(f"unknown simple-MLP site kind {kind!r}")
            return _SimpleCaptureSource(block=block, tap=value)


def _captured_activation(block_activations: _SimpleMLPBlockActivations, tap: _SimpleTap) -> Array:
    match tap:
        case _SimpleTap.RESIDUAL_IN:
            return block_activations.residual_in
        case _SimpleTap.QKV_INPUT:
            return block_activations.qkv_input
        case _SimpleTap.Q_OUTPUT:
            return block_activations.q_output
        case _SimpleTap.K_OUTPUT:
            return block_activations.k_output
        case _SimpleTap.V_OUTPUT:
            return block_activations.v_output
        case _SimpleTap.ATTENTION_OUTPUT:
            return block_activations.attention_output
        case _SimpleTap.O_OUTPUT:
            return block_activations.o_output
        case _SimpleTap.POST_ATTENTION_RESIDUAL:
            return block_activations.post_attention_residual
        case _SimpleTap.MLP_INPUT:
            return block_activations.mlp_input
        case _SimpleTap.FC_OUTPUT:
            return block_activations.fc_output
        case _SimpleTap.DOWN_INPUT:
            return block_activations.down_input
        case _SimpleTap.DOWN_OUTPUT:
            return block_activations.down_output
        case _SimpleTap.RESIDUAL_OUT:
            return block_activations.residual_out


def _record_block_captures(
    block: int,
    block_activations: _SimpleMLPBlockActivations,
    sources: tuple[_SimpleCaptureSource, ...],
    captured: dict[_SimpleCaptureSource, Array],
) -> None:
    for source in sources:
        if source.block == block:
            captured[source] = _captured_activation(block_activations, source.tap)


def _decomposed_or_frozen_site_output(
    components: ComponentStacks,
    site: str,
    frozen_weight: Array,
    site_input: Array,
    component_masks: dict[str, Array],
    weight_delta_masks: dict[str, Array] | None,
    routes: dict[str, Array] | None,
    live_sites: frozenset[str],
) -> Array:
    """One site's masked output; non-live sites take the frozen `x @ W` path."""
    if site not in live_sites:
        return site_input @ frozen_weight.T
    site_components = components.site(site)
    site_output = site_out(
        site_input,
        site_components.V,
        site_components.U,
        frozen_weight,
        component_masks[site],
        None if weight_delta_masks is None else weight_delta_masks[site],
        None if routes is None else routes[site],
    )  # fmt: skip
    return site_output


class SimpleMLPDecomposedModel(eqx.Module):
    """The `LlamaSimpleMLP` `DecomposedModel` (the `model.py` contract; SPEC §1).

    Carries the FROZEN full model (tied embedding, all blocks, final norm, lm_head) as array
    fields — threaded into the jitted step as a pytree arg, weights traced not baked. Forward
    methods take token `inputs` and embed internally; blocks with no decomposed site run the
    plain frozen block. The TRAINABLE V/U (`vu: ComponentStacks`) is an
    explicit method arg, NOT a field (separate lifecycle). `sites` / `has_position_axis` / `n_ctx`
    / `eps` are static config."""

    embed: Float[Array, "vocab d"]
    layers: list[SimpleMLPLayer]
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    n_ctx: int = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, mesh: Mesh) -> "SimpleMLPDecomposedModel":
        repl = NamedSharding(mesh, P())
        return jax.tree.map(lambda _a: repl, self)

    def attention_pattern_from_qk(
        self,
        q_site: str,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
    ) -> Float[Array, "b h t t"]:
        """The target-owned attn-patterns eval recipe (`attn_patterns_eval`'s
        `AttnPatternModel` protocol): that layer's own attention module's pattern."""
        layer = parse_site_name(q_site)[0]
        return self.layers[layer].attn.pattern(q_flat, k_flat, self.inv_freq)

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Array:
        return kl_per_position(masked_output, clean_output)

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        return self.embed[tokens]

    def _capture_grammar(self) -> TransformerTapGrammar:
        def site_dimensions(name: str) -> tuple[int, int]:
            layer, kind = parse_site_name(name)
            W = _frozen_site_weight(self.layers[layer], kind)
            return W.shape[1], W.shape[0]

        return TransformerTapGrammar(
            family=FAMILY,
            n_layer=len(self.layers),
            d_resid=self.embed.shape[1],
            d_attention_output=site_dimensions(FAMILY.name_of(0, "o_proj"))[0],
            d_mlp_hidden=site_dimensions(FAMILY.name_of(0, "down_proj"))[0],
            d_out_of=lambda name: site_dimensions(name)[1],
        )

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(site_output_tap_key(site) for site in sites)

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        self._capture_grammar().assert_hidden_acts_reconstruction_points(
            keys, self.site_names, _hidden_acts_reconstruction_dependencies
        )

    def _clean_output(self, inputs: Int[Array, "b t"]) -> Array:
        """The untouched all-frozen forward used when no captures are requested."""
        assert inputs.shape[1] <= self.n_ctx, (inputs.shape, self.n_ctx)
        x = self.embed_tokens(inputs)
        for layer in self.layers:
            x = _clean_block(layer, x, self.inv_freq, self.eps)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    def clean_forward(
        self, inputs: Int[Array, "b t"], capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        if not capture_keys:
            return ForwardResult.from_producer(
                output=self._clean_output(inputs), capture_keys=(), capture_values=()
            )
        ordered_capture_keys = tuple(sorted(capture_keys))
        capture_sources = self._capture_grammar().resolve(
            ordered_capture_keys, _capture_source_for_point
        )
        assert inputs.shape[1] <= self.n_ctx, (inputs.shape, self.n_ctx)
        captured_by_source: dict[_SimpleCaptureSource, Array] = {}
        x = self.embed_tokens(inputs)
        for block, layer in enumerate(self.layers):
            residual_in = x
            attn = layer.attn
            h1 = rms_norm(residual_in, layer.ln1, self.eps)
            q = h1 @ attn.wq.T
            k = h1 @ attn.wk.T
            v = h1 @ attn.wv.T
            attn_y = attn.core(q, k, v, self.inv_freq)
            o = attn_y @ attn.wo.T
            post_attn = residual_in + o
            mlp_in = rms_norm(post_attn, layer.ln2, self.eps)
            fc = mlp_in @ layer.Wfc.T
            down_in = _gelu_tanh(fc)
            down = down_in @ layer.Wdown.T
            x = post_attn + down
            _record_block_captures(
                block,
                _SimpleMLPBlockActivations(
                    residual_in=residual_in,
                    qkv_input=h1,
                    q_output=q,
                    k_output=k,
                    v_output=v,
                    attention_output=attn_y,
                    o_output=o,
                    post_attention_residual=post_attn,
                    mlp_input=mlp_in,
                    fc_output=fc,
                    down_input=down_in,
                    down_output=down,
                    residual_out=x,
                ),
                capture_sources,
                captured_by_source,
            )
        assert set(captured_by_source) == set(capture_sources), (
            captured_by_source.keys(),
            capture_sources,
        )
        x = rms_norm(x, self.norm, self.eps)
        return ForwardResult.from_producer(
            output=x @ self.lm_head.T,
            capture_keys=ordered_capture_keys,
            capture_values=tuple(captured_by_source[source] for source in capture_sources),
        )

    def _run_masked_forward(
        self,
        vu: ComponentStacks,
        inputs: Int[Array, "b t"],
        component_masks: dict[str, Array],
        weight_delta_masks: dict[str, Array] | None,
        routes: dict[str, Array] | None,
        capture_keys: tuple[str, ...],
    ) -> ForwardResult:
        """Masked forward plus exactly the requested captures.

        Empty keys retain the compact frozen attention/MLP calls; non-empty keys expand
        only the operations whose semantic values must be bound.
        """
        assert inputs.shape[1] <= self.n_ctx, (inputs.shape, self.n_ctx)
        capture_sources = (
            self._capture_grammar().resolve(capture_keys, _capture_source_for_point)
            if capture_keys
            else ()
        )
        live_sites = frozenset(component_masks)
        captured_by_source: dict[_SimpleCaptureSource, Array] = {}
        x = self.embed_tokens(inputs)
        for layer_idx, layer in enumerate(self.layers):
            live_kinds = {kind for kind in KIND_ORDER if site_name(layer_idx, kind) in live_sites}
            residual_in = x
            attn = layer.attn
            h1 = rms_norm(residual_in, layer.ln1, self.eps)
            site_args = (component_masks, weight_delta_masks, routes, live_sites)
            if not capture_keys:
                if not live_kinds & set(ATTN_KINDS):
                    attn_out = attn(h1, self.inv_freq)
                else:
                    q = _decomposed_or_frozen_site_output(
                        vu, site_name(layer_idx, "q_proj"), attn.wq, h1, *site_args
                    )
                    k = _decomposed_or_frozen_site_output(
                        vu, site_name(layer_idx, "k_proj"), attn.wk, h1, *site_args
                    )
                    v = _decomposed_or_frozen_site_output(
                        vu, site_name(layer_idx, "v_proj"), attn.wv, h1, *site_args
                    )
                    attn_y = attn.core(q, k, v, self.inv_freq)
                    attn_out = _decomposed_or_frozen_site_output(
                        vu, site_name(layer_idx, "o_proj"), attn.wo, attn_y, *site_args
                    )
                post_attn = residual_in + attn_out
                mlp_in = rms_norm(post_attn, layer.ln2, self.eps)
                if not live_kinds & set(MLP_KINDS):
                    mlp_out = _clean_mlp_out(layer, mlp_in)
                else:
                    fc = _decomposed_or_frozen_site_output(
                        vu, site_name(layer_idx, "c_fc"), layer.Wfc, mlp_in, *site_args
                    )
                    mlp_out = _decomposed_or_frozen_site_output(
                        vu,
                        site_name(layer_idx, "down_proj"),
                        layer.Wdown,
                        _gelu_tanh(fc),
                        *site_args,
                    )
                x = post_attn + mlp_out
                continue

            q = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "q_proj"), attn.wq, h1, *site_args
            )
            k = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "k_proj"), attn.wk, h1, *site_args
            )
            v = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "v_proj"), attn.wv, h1, *site_args
            )
            attn_y = attn.core(q, k, v, self.inv_freq)
            o = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "o_proj"), attn.wo, attn_y, *site_args
            )
            post_attn = residual_in + o
            mlp_in = rms_norm(post_attn, layer.ln2, self.eps)
            fc = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "c_fc"), layer.Wfc, mlp_in, *site_args
            )
            down_in = _gelu_tanh(fc)
            down = _decomposed_or_frozen_site_output(
                vu, site_name(layer_idx, "down_proj"), layer.Wdown, down_in, *site_args
            )
            x = post_attn + down
            _record_block_captures(
                layer_idx,
                _SimpleMLPBlockActivations(
                    residual_in=residual_in,
                    qkv_input=h1,
                    q_output=q,
                    k_output=k,
                    v_output=v,
                    attention_output=attn_y,
                    o_output=o,
                    post_attention_residual=post_attn,
                    mlp_input=mlp_in,
                    fc_output=fc,
                    down_input=down_in,
                    down_output=down,
                    residual_out=x,
                ),
                capture_sources,
                captured_by_source,
            )
        x = rms_norm(x, self.norm, self.eps)
        if not capture_keys:
            return ForwardResult.from_producer(
                output=x @ self.lm_head.T,
                capture_keys=capture_keys,
                capture_values=(),
            )
        assert set(captured_by_source) == set(capture_sources), (
            captured_by_source.keys(),
            capture_sources,
        )
        return ForwardResult.from_producer(
            output=x @ self.lm_head.T,
            capture_keys=capture_keys,
            capture_values=tuple(captured_by_source[source] for source in capture_sources),
        )

    def prepare_compute_weights(self, vu: ComponentStacks) -> ComponentStacks:
        return vu

    def component_activation_forward(
        self,
        prepared_weights: ComponentStacks,
        inputs: Int[Array, "b t"],
        /,
        *,
        capture_keys: CaptureKeys,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """Run one frozen forward for requested captures and every decomposed site's ``x @ V``."""
        component_input_keys: list[str] = []
        for site in self.site_names:
            layer, kind = parse_site_name(site)
            match kind:
                case "q_proj" | "k_proj" | "v_proj":
                    component_input_keys.append(attention_input_tap_key(layer))
                case "o_proj":
                    component_input_keys.append(attention_output_tap_key(layer))
                case "c_fc":
                    component_input_keys.append(mlp_input_tap_key(layer))
                case "down_proj":
                    component_input_keys.append(mlp_hidden_tap_key(layer))
                case _:
                    raise AssertionError(kind)

        full_forward_result = self.clean_forward(
            inputs,
            capture_keys | frozenset(component_input_keys),
        )
        component_activations: dict[str, Array] = {}
        for site, key in zip(self.site_names, component_input_keys, strict=True):
            V = prepared_weights.site(site).V
            component_activations[site] = full_forward_result.captures[key].astype(V.dtype) @ V
        requested_forward_result = ForwardResult(
            output=full_forward_result.output,
            captures={key: full_forward_result.captures[key] for key in sorted(capture_keys)},
        )
        return requested_forward_result, component_activations

    def stack_ci(self, ci_lower: dict[str, Array]) -> dict[str, Array]:
        return ci_lower

    def masked_forward(
        self,
        prepared_weights: ComponentStacks,
        inputs: Int[Array, "b t"],
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        explicit_masking = materialize_masking(masking)
        ordered_capture_keys = tuple(sorted(capture_keys))

        def forward(
            vu: ComponentStacks,
            inputs: Array,
            component_masks: dict[str, Array],
            weight_delta_masks: dict[str, Array] | None,
            routes: dict[str, Array] | None,
        ) -> ForwardResult:
            return self._run_masked_forward(
                vu, inputs, component_masks, weight_delta_masks, routes, ordered_capture_keys
            )

        forward = jax.checkpoint(forward) if remat else forward
        return forward(
            prepared_weights,
            inputs,
            explicit_masking.component_masks,
            explicit_masking.weight_delta_masks,
            explicit_masking.routes,
        )

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Array]:
        """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
        weight_deltas: dict[str, Array] = {}
        for spec in self.sites:
            layer_idx, kind = parse_site_name(spec.name)
            W = _frozen_site_weight(self.layers[layer_idx], kind)
            site_components = vu.site(spec.name)
            weight_deltas[spec.name] = (
                W.astype(jnp.float32)
                - (site_components.V.astype(jnp.float32) @ site_components.U.astype(jnp.float32)).T
            )
        return weight_deltas


# ----------------------------- weight loading -----------------------------


def load_model_config(cache_dir: Path) -> LlamaSimpleMLPConfig:
    return config_from_model_config_dict(
        yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    )


def checkpoint_safetensors_path(cache_dir: Path) -> Path:
    candidates = sorted(cache_dir.glob("model_step_*.safetensors"))
    assert len(candidates) == 1, (
        f"expected exactly one model_step_*.safetensors under {cache_dir}, found "
        f"{candidates or 'none'} — convert the torch checkpoint once (torch venv; "
        f"converter at git tag `torch-oracle`): {cache_dir}"
    )
    return candidates[0]


WeightGetter = Callable[[str], Array]
"""Checkpoint key -> array, e.g. `h.0.attn.q_proj.weight`. `lm_head.weight` is NOT a
key — the head is tied to `wte.weight`."""


def _checkpoint_weight_getter(cache_dir: Path, dtype: DTypeLike) -> WeightGetter:
    handle = safe_open(str(checkpoint_safetensors_path(cache_dir)), framework="numpy")

    def get(key: str) -> Array:
        host_array = np.asarray(handle.get_tensor(key), dtype=dtype)
        return cast(Array, cast(object, host_array))

    return get


def _layer_from_weights(
    get: WeightGetter, layer_idx: int, cfg: LlamaSimpleMLPConfig
) -> SimpleMLPLayer:
    return SimpleMLPLayer(
        ln1=get(f"h.{layer_idx}.rms_1.weight"),
        ln2=get(f"h.{layer_idx}.rms_2.weight"),
        attn=FrozenAttn(
            wq=get(f"h.{layer_idx}.attn.q_proj.weight"),
            wk=get(f"h.{layer_idx}.attn.k_proj.weight"),
            wv=get(f"h.{layer_idx}.attn.v_proj.weight"),
            wo=get(f"h.{layer_idx}.attn.o_proj.weight"),
            n_head=cfg.n_head,
            n_kv_head=cfg.n_kv_head,
            head_dim=cfg.head_dim,
            n_rep=cfg.n_rep,
        ),
        Wfc=get(f"h.{layer_idx}.mlp.c_fc.weight"),
        Wdown=get(f"h.{layer_idx}.mlp.down_proj.weight"),
    )


def build_decomposed_simple_mlp(
    embed: Array,
    layers: list[SimpleMLPLayer],
    norm: Array,
    lm_head: Array,
    cfg: LlamaSimpleMLPConfig,
    sites: tuple[SiteSpec, ...],
) -> SimpleMLPDecomposedModel:
    """Assemble a `SimpleMLPDecomposedModel` from the frozen full-model arrays + decomposition
    config. `sites` must be canonical-ordered with dims matching `cfg`."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    return SimpleMLPDecomposedModel(
        embed=embed,
        layers=layers,
        norm=norm,
        lm_head=lm_head,
        inv_freq=plain_rope_inv_freq(cfg),
        sites=sites,
        has_position_axis=True,
        eps=cfg.rms_norm_eps,
        n_ctx=cfg.n_ctx,
    )


def all_site_specs(cfg: LlamaSimpleMLPConfig) -> tuple[SiteSpec, ...]:
    """The full decomposable site set (every kind at every layer, C=1) — for a
    forward-only model (e.g. the torch-parity equivalence test) that decomposes nothing."""
    site_cs = tuple(
        SiteC(site_name(layer, kind), 1) for layer in range(cfg.n_layer) for kind in KIND_ORDER
    )
    return site_specs(cfg, canonical_site_cs(site_cs))


def target_from_weights(get: WeightGetter, cfg: LlamaSimpleMLPConfig) -> SimpleMLPDecomposedModel:
    """Build a forward-only decomposed model from checkpoint-keyed weights, with the full
    decomposable site set. Used by the torch-parity equivalence test (it calls the no-capture
    clean forward, which is independent of the site config)."""
    return build_decomposed_simple_mlp(
        embed=get("wte.weight"),
        layers=[_layer_from_weights(get, i, cfg) for i in range(cfg.n_layer)],
        norm=get("ln_f.weight"),
        lm_head=get("wte.weight"),
        cfg=cfg,
        sites=all_site_specs(cfg),
    )


def load_target_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, dtype: DTypeLike
) -> SimpleMLPDecomposedModel:
    """Forward-only decomposed model from the pretrain cache (full site set). For the
    torch-parity equivalence test; the real PD path uses
    `load_decomposed_lm_from_pretrain_cache`."""
    return target_from_weights(_checkpoint_weight_getter(cache_dir, dtype), cfg)


def load_decomposed_lm_from_pretrain_cache(
    cache_dir: Path, cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], dtype: DTypeLike
) -> SimpleMLPDecomposedModel:
    """Load the `SimpleMLP` `DecomposedModel`: the full frozen model (tied embedding, all
    blocks, final norm, lm_head) as fields plus the static decomposition `sites`."""
    get = _checkpoint_weight_getter(cache_dir, dtype)
    return build_decomposed_simple_mlp(
        embed=get("wte.weight"),
        layers=[_layer_from_weights(get, i, cfg) for i in range(cfg.n_layer)],
        norm=get("ln_f.weight"),
        lm_head=get("wte.weight"),
        cfg=cfg,
        sites=sites,
    )
