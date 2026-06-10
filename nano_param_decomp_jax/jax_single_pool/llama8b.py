"""Llama-3.1-8B vendored target — the first `DecomposedLM` implementation.

The decomposed sites are the MLP matrices (gate/up/down) of a contiguous layer range
(SPEC §1.1/§1.2), named torch-style: `layers.{i}.mlp.{gate,up,down}_proj`. The frozen
residual-start suffix (decomposed layers' attn + LN frozen, then fully-frozen tail
blocks, final norm, lm_head) is a `Target` pytree threaded as a runtime arg.

Internally V/U are stacked over a leading layer axis `L` (`DecompVU`); the
`DecomposedLM` boundary regroups the trainer's flat site-keyed dicts. Frozen weights
are stored bf16; V/U masters are fp32 (SPEC N1) — the trainer casts for compute.

Real HF weights load straight from the cached safetensors (no torch dep).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from safetensors import safe_open
from vendored_jax.llama import (
    LlamaConfig,
    apply_rope,
    causal_sdpa,
    llama3_inv_freq,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)

from jax_single_pool.lm import DecomposedLM, SiteSpec

DT = jnp.bfloat16
KINDS = ("gate", "up", "down")


@dataclass(frozen=True)
class LayerRange:
    """Contiguous inclusive range of layers whose MLP is decomposed."""

    first: int
    last: int

    @property
    def n_layers(self) -> int:
        return self.last - self.first + 1

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(range(self.first, self.last + 1))


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
    return f"layers.{layer}.mlp.{kind}_proj"


def llama_site_specs(cfg: LlamaConfig, rng: LayerRange, C: int) -> tuple[SiteSpec, ...]:
    """Canonical site order: layer-ascending, (gate, up, down) within a layer."""
    d, di = cfg.n_embd, cfg.n_intermediate
    dims = {"gate": (d, di), "up": (d, di), "down": (di, d)}
    return tuple(SiteSpec(site_name(i, k), *dims[k], C) for i in rng.layers for k in KINDS)


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

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        b, t, _ = x.shape
        q = (x @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv.T).reshape(b, t, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        y = causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(b, t, self.n_head * self.head_dim)
        return y @ self.wo.T


class FrozenMLP(eqx.Module):
    wg: Float[Array, "di d"]
    wu: Float[Array, "di d"]
    wd: Float[Array, "d di"]

    def __call__(self, x: Array) -> Array:
        return (jax.nn.silu(x @ self.wg.T) * (x @ self.wu.T)) @ self.wd.T


class FrozenBlock(eqx.Module):
    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    mlp: FrozenMLP
    eps: float = eqx.field(static=True)

    def __call__(self, x: Array, inv_freq: Array) -> Array:
        x = x + self.attn(rms_norm(x, self.ln1, self.eps), inv_freq)
        x = x + self.mlp(rms_norm(x, self.ln2, self.eps))
        return x


class DecompLayerFrozen(eqx.Module):
    """The frozen pieces of a decomposed layer: the two layernorms, the attention, and
    the MLP target weights. The MLP weights live here (frozen) and pass as a runtime
    arg — never baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]


class Target(eqx.Module):
    """Frozen residual-start suffix: decomposed layers (frozen attn/lns + frozen MLP
    target weights), then fully-frozen tail blocks, final norm, lm_head."""

    decomp_layers: list[DecompLayerFrozen]  # one per decomposed layer, in order
    tail: list[FrozenBlock]  # layers above `last` (empty if last == n_layer-1)
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)


# ----------------------------- decomposed V/U (stacked over layers) -----------------------------


class DecompVU(eqx.Module):
    """fp32 master V/U for the decomposed MLP sites, stacked over a leading layer axis
    `L`. Per layer the three sites are: gate (d->di), up (d->di), down (di->d)."""

    Vg: Float[Array, "L d C"]
    Ug: Float[Array, "L C di"]
    Vu: Float[Array, "L d C"]
    Uu: Float[Array, "L C di"]
    Vd: Float[Array, "L di C"]
    Ud: Float[Array, "L C d"]

    def site(self, layer_idx: int, kind: str) -> tuple[Array, Array]:
        match kind:
            case "gate":
                return self.Vg[layer_idx], self.Ug[layer_idx]
            case "up":
                return self.Vu[layer_idx], self.Uu[layer_idx]
            case "down":
                return self.Vd[layer_idx], self.Ud[layer_idx]
            case _:
                raise AssertionError(f"unknown kind {kind!r}")


def init_decomp_vu(cfg: LlamaConfig, C: int, n_layers: int, key: Array) -> DecompVU:
    """Small random fp32 V/U; the weight-delta channel carries the faithfulness
    residual at init (before faithfulness warmup)."""
    d, di = cfg.n_embd, cfg.n_intermediate
    ks = iter(jax.random.split(key, 6))

    def n(shape: tuple[int, ...], s: float) -> Array:
        return jax.random.normal(next(ks), shape) * s

    return DecompVU(
        Vg=n((n_layers, d, C), d**-0.5),
        Ug=n((n_layers, C, di), C**-0.5),
        Vu=n((n_layers, d, C), d**-0.5),
        Uu=n((n_layers, C, di), C**-0.5),
        Vd=n((n_layers, di, C), di**-0.5),
        Ud=n((n_layers, C, d), C**-0.5),
    )


# ----------------------------- forwards -----------------------------


def _site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §1.3): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask`/`route` broadcast over batch; trailing dim added here."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    delta = W - (V @ U).T  # (d_out, d_in)
    out = out + delta_mask[..., None] * (x @ delta.T)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out


def _clean_mlp_out(fl: DecompLayerFrozen, mlp_in: Array) -> Array:
    """Frozen target MLP — exactly `W` applied, not the `V@U + (W−V@U)` identity, so
    non-live layers carry no V/U gradient and no decomposition rounding (SPEC S2/S3)."""
    return (jax.nn.silu(mlp_in @ fl.Wg.T) * (mlp_in @ fl.Wu.T)) @ fl.Wd.T


def decomp_layer_mlp_input(
    fl: DecompLayerFrozen, resid: Float[Array, "b t d"], inv_freq: Array, eps: float
) -> tuple[Array, Array]:
    """Run a decomposed layer's frozen attention; return (post-attn residual, MLP input)."""
    x = resid + fl.attn(rms_norm(resid, fl.ln1, eps), inv_freq)
    return x, rms_norm(x, fl.ln2, eps)


def clean_suffix_logits(tgt: Target, resid: Float[Array, "b t d"]) -> Array:
    """The all-frozen suffix forward — the recon target (SPEC S3)."""
    x = resid
    for fl in tgt.decomp_layers:
        post_attn, mlp_in = decomp_layer_mlp_input(fl, x, tgt.inv_freq, tgt.eps)
        x = post_attn + _clean_mlp_out(fl, mlp_in)
    for blk in tgt.tail:
        x = blk(x, tgt.inv_freq)
    x = rms_norm(x, tgt.norm, tgt.eps)
    return x @ tgt.lm_head.T


def clean_site_inputs(
    tgt: Target, rng: LayerRange, resid: Float[Array, "b t d"]
) -> dict[str, Array]:
    """Clean CI inputs per site (SPEC S4): gate_in = up_in = the post-LN2 residual,
    down_in = silu(gate)·up — all on the frozen path, threaded layer to layer."""
    inputs: dict[str, Array] = {}
    x = resid
    for li, fl in enumerate(tgt.decomp_layers):
        layer = rng.layers[li]
        post_attn, mlp_in = decomp_layer_mlp_input(fl, x, tgt.inv_freq, tgt.eps)
        gate = mlp_in @ fl.Wg.T
        up = mlp_in @ fl.Wu.T
        down_in = jax.nn.silu(gate) * up
        inputs[site_name(layer, "gate")] = mlp_in
        inputs[site_name(layer, "up")] = mlp_in
        inputs[site_name(layer, "down")] = down_in
        x = post_attn + down_in @ fl.Wd.T
    return inputs


def _masked_kind_out(
    vu: DecompVU,
    li: int,
    layer: int,
    kind: str,
    W_kind: Array,
    x_in: Array,
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live_set: frozenset[str],
) -> Array:
    s = site_name(layer, kind)
    if s not in live_set:
        return x_in @ W_kind.T
    V, U = vu.site(li, kind)
    return _site_out(
        x_in, V, U, W_kind, masks[s], delta_masks[s],
        None if routes is None else routes[s],
    )  # fmt: skip


def masked_suffix_logits(
    tgt: Target,
    vu: DecompVU,
    rng: LayerRange,
    resid: Float[Array, "b t d"],
    masks: dict[str, Array],
    delta_masks: dict[str, Array],
    routes: dict[str, Array] | None,
    live: tuple[str, ...],
) -> Array:
    """Masked decomposed suffix forward (SPEC §1.3, S2): sites in `live` run their
    decomposed forward with `masks[s]` / `delta_masks[s]` / `routes[s]`; every other
    site runs the frozen `x @ W` path. `live` is static under jit."""
    live_set = frozenset(live)
    x = resid
    for li, fl in enumerate(tgt.decomp_layers):
        layer = rng.layers[li]
        post_attn, mlp_in = decomp_layer_mlp_input(fl, x, tgt.inv_freq, tgt.eps)
        if not any(site_name(layer, k) in live_set for k in KINDS):
            mlp_out = _clean_mlp_out(fl, mlp_in)
        else:
            args = (masks, delta_masks, routes, live_set)
            gate = _masked_kind_out(vu, li, layer, "gate", fl.Wg, mlp_in, *args)
            up = _masked_kind_out(vu, li, layer, "up", fl.Wu, mlp_in, *args)
            down_in = jax.nn.silu(gate) * up
            mlp_out = _masked_kind_out(vu, li, layer, "down", fl.Wd, down_in, *args)
        x = post_attn + mlp_out
    for blk in tgt.tail:
        x = blk(x, tgt.inv_freq)
    x = rms_norm(x, tgt.norm, tgt.eps)
    return x @ tgt.lm_head.T


def weight_deltas_fp32(tgt: Target, vu: DecompVU, rng: LayerRange) -> dict[str, Array]:
    """fp32 `W − V@U` per site from fp32 masters (SPEC N2; faithfulness input)."""
    out: dict[str, Array] = {}
    for li, fl in enumerate(tgt.decomp_layers):
        layer = rng.layers[li]
        for kind, W in (("gate", fl.Wg), ("up", fl.Wu), ("down", fl.Wd)):
            V, U = vu.site(li, kind)
            out[site_name(layer, kind)] = (
                W.astype(jnp.float32) - (V.astype(jnp.float32) @ U.astype(jnp.float32)).T
            )
    return out


def llama_decomposed_lm(cfg: LlamaConfig, rng: LayerRange, C: int) -> DecomposedLM:
    """The `DecomposedLM` boundary for this target (SPEC §1; `lm.py` contract)."""
    return DecomposedLM(
        sites=llama_site_specs(cfg, rng, C),
        clean_logits=lambda frozen, resid: clean_suffix_logits(frozen, resid),
        site_inputs=lambda frozen, resid: clean_site_inputs(frozen, rng, resid),
        masked_logits=lambda frozen, vu, resid, masks, delta_masks, routes, live: (
            masked_suffix_logits(frozen, vu, rng, resid, masks, delta_masks, routes, live)
        ),
        weight_deltas=lambda frozen, vu: weight_deltas_fp32(frozen, vu, rng),
    )


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
        self._open: dict[str, object] = {}

    def get(self, key: str) -> Array:
        fname = self._key_to_file[key]
        if fname not in self._open:
            self._open[fname] = safe_open(str(self._snapshot / fname), framework="numpy")
        return jnp.asarray(np.array(self._open[fname].get_tensor(key)), dtype=DT)  # type: ignore[union-attr]


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


def _load_block(w: _HFWeights, i: int, cfg: LlamaConfig) -> FrozenBlock:
    pre = "model.layers"
    return FrozenBlock(
        ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
        ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
        attn=_load_attn(w, i, cfg),
        mlp=FrozenMLP(
            wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        ),
        eps=cfg.rms_norm_eps,
    )


def load_target_from_hf(model_name: str, cfg: LlamaConfig, rng: LayerRange) -> Target:
    """Load the frozen residual-start suffix. Only `rng.first..n_layer-1` is
    materialized — the prefix is consumed via `make_real_target_residual`."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    pre = "model.layers"

    decomp_layers = [
        DecompLayerFrozen(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=_load_attn(w, i, cfg),
            Wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
            Wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
            Wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
        )
        for i in rng.layers
    ]
    tail = [_load_block(w, i, cfg) for i in range(rng.last + 1, cfg.n_layer)]
    return Target(
        decomp_layers=decomp_layers,
        tail=tail,
        norm=w.get("model.norm.weight"),
        lm_head=w.get("lm_head.weight"),
        inv_freq=llama3_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
    )


class Prefix(eqx.Module):
    """The frozen L0..first-1 prefix: embedding + blocks. Used only to harvest the
    residual entering the suffix (SPEC §1.1) — never in any gradient graph."""

    embed: Float[Array, "vocab d"]
    blocks: list[FrozenBlock]
    inv_freq: Float[Array, " hd2"]


def load_prefix_from_hf(model_name: str, cfg: LlamaConfig, rng: LayerRange) -> Prefix:
    w = _HFWeights(_hf_snapshot_dir(model_name))
    return Prefix(
        embed=w.get("model.embed_tokens.weight"),
        blocks=[_load_block(w, i, cfg) for i in range(rng.first)],
        inv_freq=llama3_inv_freq(cfg),
    )


def prefix_residual(prefix: Prefix, idx: Array) -> Array:
    """Pure prefix forward: token ids `(b, t)` -> residual entering `first` (b, t, d).
    The trainer jits this with `prefix` as a runtime arg and the batch dp-sharded."""
    x = prefix.embed[idx]
    for blk in prefix.blocks:
        x = blk(x, prefix.inv_freq)
    return x


def make_real_target_residual(
    model_name: str, cfg: LlamaConfig, rng: LayerRange, idx: Array, chunk: int
) -> Array:
    """One-shot eager harvest for the bench: loads the prefix, runs it in micro-batch
    chunks (`chunk`) so peak activation is one chunk's forward, discards the weights.
    The trainer instead keeps a `Prefix` resident and jits `prefix_residual`."""
    prefix = load_prefix_from_hf(model_name, cfg, rng)
    b = idx.shape[0]
    outs = [
        jax.block_until_ready(prefix_residual(prefix, idx[i : i + chunk]))
        for i in range(0, b, chunk)
    ]
    return jnp.concatenate(outs, axis=0)
