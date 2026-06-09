"""Llama-3.1-8B target for the single-pool VPD step — full-suffix output-recon.

The `jax_single_pool/` core (`step.py`, `losses.py`) does *layerwise site-local*
recon (`mean((y_dec - y_tgt)^2)` per decomposed weight, no model re-forward). That
is a simplification: the torch reference `StochasticReconLayerwiseLoss` masks ONE
module at a time but reconstructs the **final model logits** via a full masked
re-forward. This module is the full-LM output-recon target the core deferred — the
residual-start L18->L31 suffix with the L18 MLP decomposed, recon on logits.

Mirrors the torch FSDP config `llama8b_l18_mlp_fsdp.yaml`:
  * decompose `layers.18.mlp.{gate,up,down}_proj`, C=24576, weight-delta on
  * residual-start: the step consumes the residual stream entering L18 (one frozen
    prefix forward is amortized outside the differentiated step) and runs the
    14-block L18->L31 suffix + final norm + lm_head
  * CI fn = `global_shared_transformer` (d_model 4096, 4 bidirectional-RoPE blocks,
    64 heads, mlp 16384), per-site clean inputs concatenated
  * bf16 params/compute

Real HF weights load straight from the cached safetensors (no torch dep). The
config matches `meta-llama/Llama-3.1-8B` exactly.
"""

import json
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
SITES = ("gate", "up", "down")
DECOMPOSED_LAYER = 18


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


class Target(eqx.Module):
    """Frozen residual-start suffix: L18 (frozen attn/lns + frozen MLP weights) then
    L19..L31 frozen blocks, final norm, lm_head. The L18 MLP weights live here (frozen)
    and the decomposed forward consumes them via `weight_deltas` — so the suffix is one
    pytree passed as a runtime arg, not baked into the HLO as a multi-GB constant."""

    l18_ln1: Float[Array, " d"]
    l18_ln2: Float[Array, " d"]
    l18_attn: FrozenAttn
    l18_Wg: Float[Array, "di d"]
    l18_Wu: Float[Array, "di d"]
    l18_Wd: Float[Array, "d di"]
    rest: list  # FrozenBlock for L19..L31
    norm: Float[Array, " d"]
    lm_head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]
    eps: float = eqx.field(static=True)


# ----------------------------- decomposed L18 MLP -----------------------------


class DecompVU(eqx.Module):
    Vg: Float[Array, "d C"]
    Ug: Float[Array, "C di"]
    Vu: Float[Array, "d C"]
    Uu: Float[Array, "C di"]
    Vd: Float[Array, "di C"]
    Ud: Float[Array, "C d"]


def _proj(x, V, U, W, mask, delta_mask):
    """Masked weight-delta forward of one decomposed linear (matches torch LinearComponents)."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if W is not None:
        delta = W - (V @ U).T  # (d_out, d_in)
        out = out + delta_mask * (x @ delta.T)
    return out


def decomp_mlp_forward(vu: DecompVU, Wg, Wu, Wd, x, masks, delta_masks):
    gate = _proj(x, vu.Vg, vu.Ug, Wg, masks["gate"], delta_masks["gate"])
    up = _proj(x, vu.Vu, vu.Uu, Wu, masks["up"], delta_masks["up"])
    d_in = jax.nn.silu(gate) * up
    return _proj(d_in, vu.Vd, vu.Ud, Wd, masks["down"], delta_masks["down"])


def weight_deltas(vu: DecompVU, Wg, Wu, Wd) -> dict[str, Array]:
    return {
        "gate": Wg - (vu.Vg @ vu.Ug).T,
        "up": Wu - (vu.Vu @ vu.Uu).T,
        "down": Wd - (vu.Vd @ vu.Ud).T,
    }


def l18_resid_to_mlp_input(tgt: Target, resid: Float[Array, "b t d"]) -> Array:
    x = resid + tgt.l18_attn(rms_norm(resid, tgt.l18_ln1, tgt.eps), tgt.inv_freq)
    return rms_norm(x, tgt.l18_ln2, tgt.eps)


def mlp_site_inputs(Wg, Wu, mlp_in):
    """Clean per-site CI inputs: gate_in=up_in=mlp_in (d), down_in=silu(gate)*up (di)."""
    gate = mlp_in @ Wg.T
    up = mlp_in @ Wu.T
    return mlp_in, mlp_in, jax.nn.silu(gate) * up


def suffix_logits(tgt: Target, vu: DecompVU, resid, masks, delta_masks):
    x = resid + tgt.l18_attn(rms_norm(resid, tgt.l18_ln1, tgt.eps), tgt.inv_freq)
    mlp_in = rms_norm(x, tgt.l18_ln2, tgt.eps)
    x = x + decomp_mlp_forward(vu, tgt.l18_Wg, tgt.l18_Wu, tgt.l18_Wd, mlp_in, masks, delta_masks)
    for blk in tgt.rest:
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


def load_target_from_hf(model_name: str, cfg: LlamaConfig) -> Target:
    """Load the frozen residual-start suffix (L18..L31 + norm + lm_head) from HF safetensors.

    Only the suffix is materialized — the L0..L17 prefix is never loaded (the step
    consumes the residual stream entering L18, harvested by a separate prefix forward)."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    pre = "model.layers"

    def attn(i: int) -> FrozenAttn:
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

    def block(i: int) -> FrozenBlock:
        return FrozenBlock(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=attn(i),
            mlp=FrozenMLP(
                wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
                wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
                wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
            ),
            eps=cfg.rms_norm_eps,
        )

    L = DECOMPOSED_LAYER
    rest = [block(i) for i in range(L + 1, cfg.n_layer)]
    return Target(
        l18_ln1=w.get(f"{pre}.{L}.input_layernorm.weight"),
        l18_ln2=w.get(f"{pre}.{L}.post_attention_layernorm.weight"),
        l18_attn=attn(L),
        l18_Wg=w.get(f"{pre}.{L}.mlp.gate_proj.weight"),
        l18_Wu=w.get(f"{pre}.{L}.mlp.up_proj.weight"),
        l18_Wd=w.get(f"{pre}.{L}.mlp.down_proj.weight"),
        rest=rest,
        norm=w.get("model.norm.weight"),
        lm_head=w.get("lm_head.weight"),
        inv_freq=llama3_inv_freq(cfg),
        eps=cfg.rms_norm_eps,
    )


def init_decomp_vu(cfg: LlamaConfig, C: int, target: Target, key) -> DecompVU:
    """Initialize V/U with a faithful least-squares-free start: V random, U solved so
    V@U ~ W_target.T is not attempted (overcomplete C); instead small random V/U and the
    weight-delta channel carries the residual (faithfulness loss then closes the gap)."""
    d, di = cfg.n_embd, cfg.n_intermediate
    ks = iter(jax.random.split(key, 12))

    def n(shape, s):
        return (jax.random.normal(next(ks), shape) * s).astype(DT)

    return DecompVU(
        Vg=n((d, C), d**-0.5),
        Ug=n((C, di), C**-0.5),
        Vu=n((d, C), d**-0.5),
        Uu=n((C, di), C**-0.5),
        Vd=n((di, C), di**-0.5),
        Ud=n((C, d), C**-0.5),
    )


def make_real_target_residual(model_name: str, cfg: LlamaConfig, idx, chunk: int) -> Array:
    """Harvest the residual stream entering L18 with ONE frozen prefix forward (L0..L17).

    The residual-start amortization: the differentiated step never re-runs the prefix.
    Loads the prefix from HF, runs it once, discards it. `idx`: (b, t) token ids.

    Runs the prefix in micro-batch chunks (`chunk`) so peak activation is one chunk's
    prefix forward, not the full (global) batch — without this the global-batch prefix
    activations OOM alongside the suffix on each rank."""
    w = _HFWeights(_hf_snapshot_dir(model_name))
    pre = "model.layers"
    embed = w.get("model.embed_tokens.weight")
    inv_freq = llama3_inv_freq(cfg)
    blocks = [
        FrozenBlock(
            ln1=w.get(f"{pre}.{i}.input_layernorm.weight"),
            ln2=w.get(f"{pre}.{i}.post_attention_layernorm.weight"),
            attn=FrozenAttn(
                wq=w.get(f"{pre}.{i}.self_attn.q_proj.weight"),
                wk=w.get(f"{pre}.{i}.self_attn.k_proj.weight"),
                wv=w.get(f"{pre}.{i}.self_attn.v_proj.weight"),
                wo=w.get(f"{pre}.{i}.self_attn.o_proj.weight"),
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                n_rep=cfg.n_rep,
            ),
            mlp=FrozenMLP(
                wg=w.get(f"{pre}.{i}.mlp.gate_proj.weight"),
                wu=w.get(f"{pre}.{i}.mlp.up_proj.weight"),
                wd=w.get(f"{pre}.{i}.mlp.down_proj.weight"),
            ),
            eps=cfg.rms_norm_eps,
        )
        for i in range(DECOMPOSED_LAYER)
    ]

    def prefix_chunk(idx_c):
        x = embed[idx_c]
        for blk in blocks:
            x = blk(x, inv_freq)
        return x

    # eager (not jit'd): a one-time harvest where jit's constant-capture + compile of the
    # 8GB prefix weights dwarfs the runtime. Eager keeps peak activation to one chunk.
    b = idx.shape[0]
    outs = [jax.block_until_ready(prefix_chunk(idx[i : i + chunk])) for i in range(0, b, chunk)]
    return jnp.concatenate(outs, axis=0)
