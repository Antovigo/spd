"""JAX side of the Llama parity check: load the torch dump, build the Equinox ComponentLlama
from the same arrays, and compare clean logits, masked logits, and V/U grads.

Run from the jax_spike dir:  python -m parity.check_llama_jax --ref parity/llama_ref.npz
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from vendored_jax.llama import (
    ComponentLlama,
    LlamaConfig,
    MaskInfo,
    all_target_paths,
    build_from_torch_state,
    rms_norm,
)

jax.config.update("jax_enable_x64", False)  # fp32, match torch dump

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
args = ap.parse_args()

d = np.load(args.ref)


def meta(k):
    return d[f"META/{k}"]


cfg = LlamaConfig(
    vocab_size=int(meta("cfg/vocab_size")),
    n_layer=int(meta("cfg/n_layer")),
    n_head=int(meta("cfg/n_head")),
    n_kv_head=int(meta("cfg/n_key_value_heads")),
    n_embd=int(meta("cfg/n_embd")),
    n_intermediate=int(meta("cfg/n_intermediate")),
    rope_theta=float(meta("cfg/rope_theta")),
    rms_norm_eps=float(meta("cfg/rms_norm_eps")),
    max_position_embeddings=int(meta("cfg/max_position_embeddings")),
    rope_factor=float(meta("cfg/rope_factor")),
    rope_low_freq_factor=float(meta("cfg/rope_low")),
    rope_high_freq_factor=float(meta("cfg/rope_high")),
    rope_original_max_position_embeddings=int(meta("cfg/rope_orig")),
)

sd = {k: jnp.asarray(v) for k, v in d.items() if not k.startswith("META/")}
model = build_from_torch_state(cfg, sd)
paths = all_target_paths(cfg)
idx = jnp.asarray(meta("idx"))
masks = {p: MaskInfo(component_mask=jnp.asarray(d[f"META/mask/{p}"])) for p in paths}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-8)))


# --- intermediate-activation localization (clean path) ---
emb = model.embed_tokens[idx]
h = emb
print("--- stage-by-stage (clean) ---")
print(f"emb           rel err: {rel(emb, meta('dbg/emb')):.3e}")
for li, block in enumerate(model.blocks):
    h = block(h, None)
    print(f"h_layer{li}      rel err: {rel(h, meta(f'dbg/h_layer{li}')):.3e}")
print(f"h_pre_norm    rel err: {rel(h, meta('dbg/h_pre_norm')):.3e}")
h_post = rms_norm(h, model.norm, cfg.rms_norm_eps)
print(f"h_post_norm   rel err: {rel(h_post, meta('dbg/h_post_norm')):.3e}")
print("--- full ---")

logits_clean = model(idx, None)
logits_masked = model(idx, masks)
e_clean = rel(logits_clean, meta("logits_clean"))
e_masked = rel(logits_masked, meta("logits_masked"))


def loss_fn(m: ComponentLlama):
    return jnp.mean(m(idx, masks) ** 2)


gmodel = eqx.filter_grad(loss_fn)(model)


def clin_of(root, path):
    parts = path.split(".")
    blk = root.blocks[int(parts[1])]
    parent = blk.self_attn if parts[2] == "self_attn" else blk.mlp
    return getattr(parent, parts[3])


worst_gV = worst_gU = 0.0
for p in paths:
    c = clin_of(gmodel, p)
    worst_gV = max(worst_gV, rel(c.V, d[f"META/gV/{p}"]))
    worst_gU = max(worst_gU, rel(c.U, d[f"META/gU/{p}"]))

print(f"clean  logits rel err: {e_clean:.3e}")
print(f"masked logits rel err: {e_masked:.3e}")
print(f"grad V worst rel err:  {worst_gV:.3e}")
print(f"grad U worst rel err:  {worst_gU:.3e}")
tol = 2e-3
ok = max(e_clean, e_masked, worst_gV, worst_gU) < tol
print(f"LLAMA PARITY (tol {tol:.0e}):", "PASS" if ok else "FAIL")
