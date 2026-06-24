"""Llama-3.1-8B vendored target — the first `DecomposedModel` implementation.

The decomposed sites are any per-layer weight matrices (SPEC §1/§3) named torch-style:
`layers.{i}.self_attn.{q,k,v,o}_proj` and `layers.{i}.mlp.{gate,up,down}_proj`, each
with its own C. `LlamaDecomposedModel` (an `eqx.Module`) carries the frozen residual-start
suffix — from the lowest decomposed layer (`first_decomposed_layer`) to the LM head — as
array fields, threaded into the jitted step as a pytree arg; suffix layers without sites
run the plain frozen block.

q/k/v sites are decomposed BEFORE RoPE/SDPA (the masked site output feeds the
attention math); the o site applies to the attention output. V/U masters are fp32
keyed per site (`DecompVU`); frozen weights are stored bf16 (SPEC N1) — the trainer
casts for compute.

Real HF weights load straight from the cached safetensors (no torch dep).
"""

import json
import re
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.components import DecompVU, SiteC, SiteSpec, site_out
from param_decomp.losses import kl_per_position
from vendored_jax.llama import (
    LlamaConfig,
    apply_rope,
    causal_sdpa,
    llama3_inv_freq,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)

DT = jnp.bfloat16

KIND_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
"""Within-layer canonical site order = computation order. The canonical site order
(`llama_site_specs`) is layer-ascending, then this."""
ATTN_KINDS = ("q", "k", "v", "o")
MLP_KINDS = ("gate", "up", "down")

SITE_NAME_PATTERN = re.compile(
    r"^layers\.(\d+)\.(?:self_attn\.(q|k|v|o)|mlp\.(gate|up|down))_proj$"
)


def llama31_8b_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=128256,
        n_layer=32,
        n_head=32,
        n_kv_head=8,
        n_embd=4096,
        n_intermediate=14336,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=8192,
    )


def site_name(layer: int, kind: str) -> str:
    assert kind in KIND_ORDER, kind
    submodule = "self_attn" if kind in ATTN_KINDS else "mlp"
    return f"layers.{layer}.{submodule}.{kind}_proj"


def parse_site_name(name: str) -> tuple[int, str]:
    """`layers.{i}.{self_attn,mlp}.{kind}_proj` -> (layer, kind); rejects anything else
    (including kind/submodule mismatches like `self_attn.gate_proj`)."""
    match = SITE_NAME_PATTERN.match(name)
    assert match is not None, f"unsupported site name {name!r}"
    layer, attn_kind, mlp_kind = match.groups()
    return int(layer), attn_kind if attn_kind is not None else mlp_kind


def site_dims(cfg: LlamaConfig, kind: str) -> tuple[int, int]:
    """(d_in, d_out) of one per-layer matrix, right-mult orientation."""
    d, di = cfg.n_embd, cfg.n_intermediate
    qd = cfg.n_head * cfg.head_dim
    kvd = cfg.n_kv_head * cfg.head_dim
    match kind:
        case "q":
            return d, qd
        case "k" | "v":
            return d, kvd
        case "o":
            return qd, d
        case "gate" | "up":
            return d, di
        case "down":
            return di, d
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


def canonical_site_cs(site_cs: tuple[SiteC, ...]) -> tuple[SiteC, ...]:
    """Canonical site order: layer-ascending, `KIND_ORDER` within a layer. Names must
    parse and be unique."""
    names = [site.name for site in site_cs]
    assert len(set(names)) == len(names), f"duplicate sites in {names}"

    def order_key(site: SiteC) -> tuple[int, int]:
        layer, kind = parse_site_name(site.name)
        return layer, KIND_ORDER.index(kind)

    return tuple(sorted(site_cs, key=order_key))


def mlp_family_site_cs(first_layer: int, last_layer: int, C: int) -> tuple[SiteC, ...]:
    """The gate/up/down sites of a contiguous layer range at one C (the native-config
    target family), in canonical order."""
    assert first_layer <= last_layer, (first_layer, last_layer)
    return tuple(
        SiteC(site_name(layer, kind), C)
        for layer in range(first_layer, last_layer + 1)
        for kind in MLP_KINDS
    )


def llama_site_specs(cfg: LlamaConfig, site_cs: tuple[SiteC, ...]) -> tuple[SiteSpec, ...]:
    """Shape-resolved specs in canonical order (input must already be canonical)."""
    assert site_cs == canonical_site_cs(site_cs), f"sites not in canonical order: {site_cs}"
    specs = []
    for site in site_cs:
        layer, kind = parse_site_name(site.name)
        assert 0 <= layer < cfg.n_layer, (site.name, cfg.n_layer)
        assert site.C >= 1, site
        specs.append(SiteSpec(site.name, *site_dims(cfg, kind), site.C))
    return tuple(specs)


def first_decomposed_layer(site_names: tuple[str, ...]) -> int:
    """The residual-start boundary: the suffix runs from the lowest decomposed layer."""
    assert site_names
    return min(parse_site_name(name)[0] for name in site_names)


# ----------------------------- frozen suffix -----------------------------


class FrozenAttn(eqx.Module):
    wq: Float[Array, "qd d"]
    wk: Float[Array, "kvd d"]
    wv: Float[Array, "kvd d"]
    wo: Float[Array, "d qd"]
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    n_rep: int = eqx.field(static=True)

    def core(
        self,
        q_flat: Float[Array, "b t qd"],
        k_flat: Float[Array, "b t kvd"],
        v_flat: Float[Array, "b t kvd"],
        inv_freq: Array,
    ) -> Float[Array, "b t qd"]:
        """RoPE + causal SDPA between the q/k/v projections and the o projection —
        the seam the decomposed q/k/v site outputs feed into."""
        b, t, _ = q_flat.shape
        assert q_flat.shape[-1] == self.n_head * self.head_dim, q_flat.shape
        assert k_flat.shape[-1] == self.n_kv_head * self.head_dim, k_flat.shape
        assert v_flat.shape[-1] == self.n_kv_head * self.head_dim, v_flat.shape
        q = q_flat.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, q_flat.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        return causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        return self.core(x @ self.wq.T, x @ self.wk.T, x @ self.wv.T, inv_freq) @ self.wo.T


class SuffixLayer(eqx.Module):
    """One suffix layer's frozen weights — norms, attention, MLP. Decomposed sites read
    their frozen target W from here at forward time; layers without sites run the
    plain frozen block from the same fields. Weights pass as a runtime arg — never
    baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]


def _frozen_site_weight(suffix_layer: SuffixLayer, kind: str) -> Array:
    match kind:
        case "q":
            return suffix_layer.attn.wq
        case "k":
            return suffix_layer.attn.wk
        case "v":
            return suffix_layer.attn.wv
        case "o":
            return suffix_layer.attn.wo
        case "gate":
            return suffix_layer.Wg
        case "up":
            return suffix_layer.Wu
        case "down":
            return suffix_layer.Wd
        case _:
            raise AssertionError(f"unknown kind {kind!r}")


# ----------------------------- forwards -----------------------------


def _clean_mlp_out(suffix_layer: SuffixLayer, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live sites carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return (
        jax.nn.silu(mlp_in @ suffix_layer.Wg.T) * (mlp_in @ suffix_layer.Wu.T)
    ) @ suffix_layer.Wd.T


def _tap_layer(key: str) -> int:
    """Global block index a `read_activations` key reads at: the block a `resid.{L}` tap
    enters, or the block a decomposed site lives in."""
    if key.startswith("resid."):
        return int(key.split(".")[1])
    return parse_site_name(key)[0]


def _masked_site_out(
    components: DecompVU,
    site: str,
    W: Array,
    x_in: Array,
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
    has_delta: bool,
    collect: dict[str, Array] | None,
) -> Array:
    """One site's output in the masked forward; if `collect` is given, the per-`live`-site
    decomposed output is recorded there (the hidden-acts recon material, SPEC S31).
    Non-live sites take the frozen `x @ W` path and are NOT collected."""
    if site not in live_set:
        return x_in @ W.T
    V, U = components.site(site)
    out = site_out(
        x_in, V, U, W, masks[site], delta_masks[site] if has_delta else None,
        None if routes is None else routes[site],
    )  # fmt: skip
    if collect is not None:
        collect[site] = out
    return out


class LlamaDecomposedModel(eqx.Module):
    """The Llama-8B `DecomposedModel` (the `lm.py` contract; SPEC §1).

    Carries the FROZEN full model (embedding, all blocks, final norm, lm_head) as array
    fields — so it threads into the jitted step as a pytree arg, its weights traced not
    baked. The TRAINABLE V/U (`vu: DecompVU`) is passed to the forward methods explicitly:
    separate lifecycle (own optimizer + checkpoint, C-sharded while these weights
    replicate), so it is NOT a field here.

    Forward methods take token `inputs` and embed internally; the `*_from_residual`
    variants run the post-embedding computation (the engine calls the token entry, tests
    drive the residual entry). Blocks with no decomposed site run the plain frozen path —
    so a subset decomposition just leaves the rest frozen; there is no prefix/suffix cut.

    `sites` / `leading_axes` are static config."""

    embed: Float[Array, "vocab d"]
    layers: list[SuffixLayer]
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    leading_axes: tuple[str, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    @staticmethod
    def recon_loss_fn(masked_output: Array, clean_output: Array) -> Array:
        return kl_per_position(masked_output, clean_output)

    def embed_tokens(self, tokens: Int[Array, "b t"]) -> Float[Array, "b t d"]:
        return self.embed[tokens]

    def clean_output(self, inputs: Int[Array, "b t"]) -> Array:
        """The all-frozen forward — the recon target (SPEC S3)."""
        x = self.embed_tokens(inputs)
        for block in self.layers:
            x = x + block.attn(rms_norm(x, block.ln1, self.eps), self.inv_freq)
            x = x + _clean_mlp_out(block, rms_norm(x, block.ln2, self.eps))
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    def read_activations(
        self, inputs: Int[Array, "b t"], wanted: tuple[str, ...]
    ) -> dict[str, Array]:
        """Frozen-path activation accessor (CI input side, SPEC S4; harvest's per-site
        matrix inputs).

        `wanted` keys are either `resid.{layer}` (residual stream ENTERING that block — the
        chunkwise CI fn's `input_names`) or a decomposed SITE NAME (the activation entering
        that site's weight on the frozen path: `q/k/v_proj` ← post-LN1 residual, `o_proj` ←
        the attention output, `gate/up_proj` ← post-LN2 residual, `down_proj` ←
        `silu(gate)·up`). The residual is threaded identically to `clean_output`; the
        per-site intermediates come from the same RMSNorm/attn/MLP math. Stops once the last
        requested key's block is fully covered (no wasted block compute past it)."""
        wanted_set = frozenset(wanted)
        last = max(_tap_layer(key) for key in wanted)
        taps: dict[str, Array] = {}
        x = self.embed_tokens(inputs)
        for layer, block in enumerate(self.layers):
            if f"resid.{layer}" in wanted_set:
                taps[f"resid.{layer}"] = x
            attn = block.attn
            h1 = rms_norm(x, block.ln1, self.eps)
            attn_y = attn.core(h1 @ attn.wq.T, h1 @ attn.wk.T, h1 @ attn.wv.T, self.inv_freq)
            post_attn = x + attn_y @ attn.wo.T
            mlp_in = rms_norm(post_attn, block.ln2, self.eps)
            down_in = jax.nn.silu(mlp_in @ block.Wg.T) * (mlp_in @ block.Wu.T)
            for kind, site_input in (
                ("q", h1), ("k", h1), ("v", h1), ("o", attn_y),
                ("gate", mlp_in), ("up", mlp_in), ("down", down_in),
            ):  # fmt: skip
                name = site_name(layer, kind)
                if name in wanted_set:
                    taps[name] = site_input
            x = post_attn + down_in @ block.Wd.T
            if layer == last:
                break
        assert set(taps) == wanted_set, (sorted(taps), sorted(wanted))
        return taps

    def _run_masked_suffix(
        self,
        vu: DecompVU,
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
        collect: dict[str, Array] | None,
    ) -> Array:
        """The masked decomposed forward shared by `masked_output` and
        `masked_site_outputs` (SPEC §1.3, S2): sites in `live` run their decomposed forward
        with `masks[s]` / `delta_masks[s]` / `routes[s]`; every other site — and every site
        absent from the decomposition entirely — runs the frozen `x @ W` path. `live` and
        `has_delta` are static under jit; `has_delta` False skips the `x @ Δ` matmul
        (LOSS_PARITY_DESIGN §4b). A non-None `collect` gathers per-site decomposed
        outputs."""
        live_set = frozenset(live)
        x = self.embed_tokens(inputs)
        for layer, block in enumerate(self.layers):
            live_kinds = {kind for kind in KIND_ORDER if site_name(layer, kind) in live_set}
            attn = block.attn
            site_args = (masks, delta_masks, routes, live_set, has_delta, collect)
            h1 = rms_norm(x, block.ln1, self.eps)
            if not live_kinds & set(ATTN_KINDS):
                attn_out = attn(h1, self.inv_freq)
            else:
                q = _masked_site_out(vu, site_name(layer, "q"), attn.wq, h1, *site_args)
                k = _masked_site_out(vu, site_name(layer, "k"), attn.wk, h1, *site_args)
                v = _masked_site_out(vu, site_name(layer, "v"), attn.wv, h1, *site_args)
                attn_y = attn.core(q, k, v, self.inv_freq)
                attn_out = _masked_site_out(vu, site_name(layer, "o"), attn.wo, attn_y, *site_args)
            post_attn = x + attn_out
            mlp_in = rms_norm(post_attn, block.ln2, self.eps)
            if not live_kinds & set(MLP_KINDS):
                mlp_out = _clean_mlp_out(block, mlp_in)
            else:
                gate = _masked_site_out(vu, site_name(layer, "gate"), block.Wg, mlp_in, *site_args)
                up = _masked_site_out(vu, site_name(layer, "up"), block.Wu, mlp_in, *site_args)
                down_in = jax.nn.silu(gate) * up
                mlp_out = _masked_site_out(
                    vu, site_name(layer, "down"), block.Wd, down_in, *site_args
                )
            x = post_attn + mlp_out
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    def masked_output(
        self,
        vu: DecompVU,
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> Array:
        return self._run_masked_suffix(
            vu, inputs, masks, delta_masks, routes, live, has_delta, None
        )

    def masked_site_outputs(
        self,
        vu: DecompVU,
        inputs: Int[Array, "b t"],
        masks: dict[str, Array],
        delta_masks: dict[str, Array],
        routes: dict[str, Array] | None,
        live: tuple[str, ...],
        has_delta: bool,
    ) -> dict[str, Array]:
        """Per-`live`-site decomposed output of the masked forward (SPEC S31). Runs the
        exact `masked_output` forward, discards the logits, returns the collected outputs."""
        collect: dict[str, Array] = {}
        self._run_masked_suffix(vu, inputs, masks, delta_masks, routes, live, has_delta, collect)
        assert set(collect) == set(live), (sorted(collect), sorted(live))
        return collect

    def weight_deltas(self, vu: DecompVU) -> dict[str, Array]:
        """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
        out: dict[str, Array] = {}
        for spec in self.sites:
            layer, kind = parse_site_name(spec.name)
            W = _frozen_site_weight(self.layers[layer], kind)
            V, U = vu.site(spec.name)
            out[spec.name] = (
                W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
            )
        return out


# ----------------------------- HF weight loading -----------------------------


def _hf_snapshot_dir(model_name: str) -> Path:
    import os

    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    repo = "models--" + model_name.replace("/", "--")
    snaps = sorted((cache / repo / "snapshots").iterdir())
    assert snaps, f"no snapshot for {model_name} under {cache}"
    return snaps[-1]


class _HFWeights:
    """Lazy keyed access to the sharded safetensors of an HF Llama checkpoint."""

    def __init__(self, snapshot: Path):
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        self._key_to_file = index["weight_map"]
        self._snapshot = snapshot
        self._open: dict[str, Any] = {}

    def get(self, key: str) -> Array:
        fname = self._key_to_file[key]
        if fname not in self._open:
            self._open[fname] = safe_open(str(self._snapshot / fname), framework="numpy")
        return jnp.asarray(np.array(self._open[fname].get_tensor(key)), dtype=DT)


def _load_attn(w: _HFWeights, i: int, cfg: LlamaConfig) -> FrozenAttn:
    pre = "model.layers"
    return FrozenAttn(
        wq=w.get(f"{pre}.{i}.self_attn.q_proj.weight"),
        wk=w.get(f"{pre}.{i}.self_attn.k_proj.weight"),
        wv=w.get(f"{pre}.{i}.self_attn.v_proj.weight"),
        wo=w.get(f"{pre}.{i}.self_attn.o_proj.weight"),
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        head_dim=cfg.head_dim,
        n_rep=cfg.n_rep,
    )


def _load_blocks(w: "_HFWeights", cfg: LlamaConfig) -> list[SuffixLayer]:
    pre = "model.layers"
    return [
        SuffixLayer(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=_load_attn(w, i, cfg),
            Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        )
        for i in range(cfg.n_layer)
    ]


def build_decomposed_lm(
    embed: Array,
    layers: list[SuffixLayer],
    norm: Array,
    lm_head: Array,
    inv_freq: Array,
    cfg: LlamaConfig,
    sites: tuple[SiteSpec, ...],
) -> LlamaDecomposedModel:
    """Assemble a `LlamaDecomposedModel` from the frozen full-model arrays + decomposition
    config. `sites` must be canonical-ordered with dims matching `cfg`."""
    site_cs = tuple(SiteC(s.name, s.C) for s in sites)
    assert sites == llama_site_specs(cfg, canonical_site_cs(site_cs)), (
        f"sites are not the canonical specs for this config: {sites}"
    )
    return LlamaDecomposedModel(
        embed=embed,
        layers=layers,
        norm=norm,
        lm_head=lm_head,
        inv_freq=inv_freq,
        sites=sites,
        leading_axes=("sequence",),
        eps=cfg.rms_norm_eps,
    )


def load_decomposed_lm_from_hf(
    model_name: str, cfg: LlamaConfig, sites: tuple[SiteSpec, ...]
) -> LlamaDecomposedModel:
    """Load the Llama-8B `DecomposedModel`: the full frozen model (embedding, all blocks,
    final norm, lm_head) as fields plus the static decomposition config (`sites`). Blocks
    without a decomposed site run the plain frozen path."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    return build_decomposed_lm(
        embed=w.get("model.embed_tokens.weight"),
        layers=_load_blocks(w, cfg),
        norm=w.get("model.norm.weight"),
        lm_head=w.get("lm_head.weight"),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
    )
