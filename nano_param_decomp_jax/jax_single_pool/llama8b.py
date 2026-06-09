"""Llama-3.1-8B target for the single-pool VPD step — full-suffix output-recon.

The `jax_single_pool/` core (`step.py`, `losses.py`) does *layerwise site-local*
recon (`mean((y_dec - y_tgt)^2)` per decomposed weight, no model re-forward). That
is a simplification: the torch reference `StochasticReconLayerwiseLoss` masks ONE
module at a time but reconstructs the **final model logits** via a full masked
re-forward. This module is the full-LM output-recon target the core deferred — the
residual-start suffix with the decomposed-layers' MLPs decomposed, recon on logits.

Generalized 1->N decomposed layers. `DecompLayers` decomposes the MLP
(gate/up/down) of a CONTIGUOUS range of layers `[first_layer, last_layer]`. The
residual-start point is `first_layer`: a frozen L0..first-1 prefix forward
(amortized outside the differentiated step) harvests the residual stream entering
`first_layer`; the differentiated step runs the suffix from `first_layer` onward
(decomposed-layer attn frozen + MLP decomposed; any layers above `last_layer`
fully frozen) then final norm + lm_head, recon on logits.

Mirrors the torch config `llama8b_l18_b512_2pool_lr_mid.yaml` (extended to a layer
range): decompose `layers.{first..last}.mlp.{gate,up,down}_proj`, weight-delta on,
CI fn = `global_shared_transformer` (one shared transformer over ALL 3N sites,
inputs concatenated), bf16 params/compute.

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
    """The frozen (non-decomposed) pieces of a decomposed layer: the two layernorms,
    the attention, and the MLP target weights. The MLP weights live here (frozen) and
    feed the decomposed forward via `weight_deltas` — so they pass as a runtime arg, not
    baked into the HLO as a multi-GB constant."""

    ln1: Float[Array, " d"]
    ln2: Float[Array, " d"]
    attn: FrozenAttn
    Wg: Float[Array, "di d"]
    Wu: Float[Array, "di d"]
    Wd: Float[Array, "d di"]


class Target(eqx.Module):
    """Frozen residual-start suffix: `n_decomp` decomposed layers (frozen attn/lns +
    frozen MLP target weights), then any fully-frozen tail blocks, final norm, lm_head."""

    decomp_layers: list  # DecompLayerFrozen, one per decomposed layer (in order)
    tail: list  # FrozenBlock for layers above `last` (empty if last == n_layer-1)
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)


# ----------------------------- decomposed MLP (stacked over layers) -----------------------------


class DecompVU(eqx.Module):
    """V/U for the decomposed MLP sites, stacked over a leading layer axis `L`.

    Per layer the three sites are: gate (d->di), up (d->di), down (di->d). Each named
    array carries the layer axis first, so `Vg[i]` is layer i's gate V."""

    Vg: Float[Array, "L d C"]
    Ug: Float[Array, "L C di"]
    Vu: Float[Array, "L d C"]
    Uu: Float[Array, "L C di"]
    Vd: Float[Array, "L di C"]
    Ud: Float[Array, "L C d"]


def _proj(x, V, U, W, mask, delta_mask, route):
    """Masked weight-delta forward of one decomposed linear (matches torch LinearComponents).

    `route` (or None for "all"): per-position bool `(..., 1)`; where False the clean target
    output `x @ W.T` is used instead of the masked decomposed output (torch routing_mask)."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    delta = W - (V @ U).T  # (d_out, d_in)
    out = out + delta_mask * (x @ delta.T)
    if route is not None:
        out = jnp.where(route, out, x @ W.T)
    return out


def _decomp_mlp_forward_one(Vg, Ug, Vu, Uu, Vd, Ud, Wg, Wu, Wd, x, masks, delta_masks, routes):
    gate = _proj(x, Vg, Ug, Wg, masks["gate"], delta_masks["gate"], routes["gate"])
    up = _proj(x, Vu, Uu, Wu, masks["up"], delta_masks["up"], routes["up"])
    d_in = jax.nn.silu(gate) * up
    return _proj(d_in, Vd, Ud, Wd, masks["down"], delta_masks["down"], routes["down"])


def weight_deltas(vu: DecompVU, frozen_layers: list) -> dict[str, Array]:
    """Per-kind weight deltas stacked over the layer axis: {kind: (L, d_out, d_in)}.

    `frozen_layers` is `Target.decomp_layers` (carries the per-layer MLP target weights)."""
    Wg = jnp.stack([fl.Wg for fl in frozen_layers])
    Wu = jnp.stack([fl.Wu for fl in frozen_layers])
    Wd = jnp.stack([fl.Wd for fl in frozen_layers])
    vmt = jax.vmap(lambda V, U: (V @ U).T)
    return {
        "gate": Wg - vmt(vu.Vg, vu.Ug),
        "up": Wu - vmt(vu.Vu, vu.Uu),
        "down": Wd - vmt(vu.Vd, vu.Ud),
    }


def decomp_layer_mlp_input(fl: DecompLayerFrozen, resid: Float[Array, "b t d"], inv_freq, eps):
    """Run a decomposed layer's frozen attention, return the post-attn residual and the
    MLP input (rms-normed post-attn residual)."""
    x = resid + fl.attn(rms_norm(resid, fl.ln1, eps), inv_freq)
    return x, rms_norm(x, fl.ln2, eps)


def site_inputs_for_layer(fl: DecompLayerFrozen, mlp_in: Array) -> tuple[Array, Array, Array]:
    """Clean per-site CI inputs for one layer: gate_in=up_in=mlp_in (d), down_in=silu(gate)*up (di)."""
    gate = mlp_in @ fl.Wg.T
    up = mlp_in @ fl.Wu.T
    return mlp_in, mlp_in, jax.nn.silu(gate) * up


def all_site_inputs(tgt: Target, resid: Float[Array, "b t d"]) -> list:
    """Clean CI inputs for ALL sites, in (layer, kind) order: a flat list of length 3N.

    Runs each decomposed layer's frozen attention + MLP to harvest the per-site clean
    inputs. The residual is threaded through the decomposed layers' clean MLP (so layer
    i+1's site inputs see layer i's clean output)."""
    inputs: list[Array] = []
    x = resid
    for fl in tgt.decomp_layers:
        post_attn, mlp_in = decomp_layer_mlp_input(fl, x, tgt.inv_freq, tgt.eps)
        g_in, u_in, d_in = site_inputs_for_layer(fl, mlp_in)
        inputs.extend([g_in, u_in, d_in])
        clean_mlp_out = (jax.nn.silu(mlp_in @ fl.Wg.T) * (mlp_in @ fl.Wu.T)) @ fl.Wd.T
        x = post_attn + clean_mlp_out  # carry the residual forward to the next layer's sites
    return inputs


def suffix_logits(tgt: Target, vu: DecompVU, resid, masks: dict, delta_masks: dict, routes: dict):
    """Masked decomposed forward of the whole suffix -> logits.

    `masks` / `delta_masks` / `routes`: {kind: (L, ...) | None}. Layer i's value for
    `kind` is `[k][i]` (mask None means no component mask; route None means "all"
    positions routed to the decomposed module — the clean/PPGD case)."""
    x = resid
    for i, fl in enumerate(tgt.decomp_layers):
        post_attn = x + fl.attn(rms_norm(x, fl.ln1, tgt.eps), tgt.inv_freq)
        mlp_in = rms_norm(post_attn, fl.ln2, tgt.eps)
        m = {k: (None if masks[k] is None else masks[k][i]) for k in KINDS}
        dm = {k: delta_masks[k][i] for k in KINDS}
        rt = {k: (None if routes[k] is None else routes[k][i]) for k in KINDS}
        mlp_out = _decomp_mlp_forward_one(
            vu.Vg[i], vu.Ug[i], vu.Vu[i], vu.Uu[i], vu.Vd[i], vu.Ud[i],
            fl.Wg, fl.Wu, fl.Wd, mlp_in, m, dm, rt,
        )  # fmt: skip
        x = post_attn + mlp_out
    for blk in tgt.tail:
        x = blk(x, tgt.inv_freq)
    x = rms_norm(x, tgt.norm, tgt.eps)
    return x @ tgt.lm_head.T


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
    """Load the frozen residual-start suffix (decomposed layers + tail + norm + lm_head).

    Only the suffix (`rng.first`..`n_layer-1`) is materialized — the L0..first-1 prefix
    is never loaded (the step consumes the residual entering `rng.first`, harvested by a
    separate prefix forward)."""
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


def init_decomp_vu(cfg: LlamaConfig, C: int, n_layers: int, key) -> DecompVU:
    """Small random V/U stacked over the layer axis; the weight-delta channel carries
    the faithfulness residual at init."""
    d, di = cfg.n_embd, cfg.n_intermediate
    ks = iter(jax.random.split(key, 6))

    def n(shape, s):
        return (jax.random.normal(next(ks), shape) * s).astype(DT)

    return DecompVU(
        Vg=n((n_layers, d, C), d**-0.5),
        Ug=n((n_layers, C, di), C**-0.5),
        Vu=n((n_layers, d, C), d**-0.5),
        Uu=n((n_layers, C, di), C**-0.5),
        Vd=n((n_layers, di, C), di**-0.5),
        Ud=n((n_layers, C, d), C**-0.5),
    )


def make_real_target_residual(
    model_name: str, cfg: LlamaConfig, rng: LayerRange, idx, chunk: int
) -> Array:
    """Harvest the residual stream entering `rng.first` with ONE frozen prefix forward.

    The residual-start amortization: the differentiated step never re-runs the prefix.
    Loads layers L0..first-1 from HF, runs them once, discards them. `idx`: (b, t) tokens.

    Runs the prefix in micro-batch chunks (`chunk`) so peak activation is one chunk's
    prefix forward, not the full (global) batch."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    embed = w.get("model.embed_tokens.weight")
    inv_freq = llama3_inv_freq(cfg)
    blocks = [_load_block(w, i, cfg) for i in range(rng.first)]

    def prefix_chunk(idx_c):
        x = embed[idx_c]
        for blk in blocks:
            x = blk(x, inv_freq)
        return x

    # eager (not jit'd): a one-time harvest where jit's constant-capture + compile of the
    # prefix weights dwarfs the runtime. Eager keeps peak activation to one chunk.
    b = idx.shape[0]
    outs = [jax.block_until_ready(prefix_chunk(idx[i : i + chunk])) for i in range(0, b, chunk)]
    return jnp.concatenate(outs, axis=0)
