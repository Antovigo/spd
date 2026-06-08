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
    apply_rope,
    build_from_torch_state,
    causal_sdpa,
    repeat_kv,
    rms_norm,
    rope_cos_sin,
)

jax.config.update("jax_enable_x64", False)  # fp32, match torch dump
# JAX defaults to TF32 for fp32 matmuls on GPU; torch ran true fp32 -> force exact fp32
jax.config.update("jax_default_matmul_precision", "highest")

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
# block-0 internal breakdown
b0 = model.blocks[0]
ln1 = rms_norm(emb, b0.input_layernorm, cfg.rms_norm_eps)
a_out = b0.self_attn(ln1, None)
x_mid = emb + a_out
ln2 = rms_norm(x_mid, b0.post_attention_layernorm, cfg.rms_norm_eps)
m_out = b0.mlp(ln2, None)
for nm, arr in [
    ("b0_ln1", ln1),
    ("b0_attn", a_out),
    ("b0_xmid", x_mid),
    ("b0_ln2", ln2),
    ("b0_mlp", m_out),
]:
    print(f"{nm:13s} rel err: {rel(arr, meta('dbg/' + nm)):.3e}")
# attention internals
at = b0.self_attn
bsz, tlen, _ = ln1.shape
hd = cfg.head_dim
qp, kp, vp = at.q_proj(ln1, None), at.k_proj(ln1, None), at.v_proj(ln1, None)
print(f"  b0_qproj    rel err: {rel(qp, meta('dbg/b0_qproj')):.3e}")
cos, sin = rope_cos_sin(at.inv_freq, tlen, ln1.dtype)
print(f"  b0_cos      rel err: {rel(cos, meta('dbg/b0_cos')):.3e}")
print(f"  b0_sin      rel err: {rel(sin, meta('dbg/b0_sin')):.3e}")
q = qp.reshape(bsz, tlen, cfg.n_head, hd).transpose(0, 2, 1, 3)
k = kp.reshape(bsz, tlen, cfg.n_kv_head, hd).transpose(0, 2, 1, 3)
v = vp.reshape(bsz, tlen, cfg.n_kv_head, hd).transpose(0, 2, 1, 3)
q, k = apply_rope(q, k, cos, sin)
k, v = repeat_kv(k, cfg.n_rep), repeat_kv(v, cfg.n_rep)
y = causal_sdpa(q, k, v).transpose(0, 2, 1, 3).reshape(bsz, tlen, cfg.n_head * hd)
print(f"  b0_y_attend rel err: {rel(y, meta('dbg/b0_y_attend')):.3e}")
for li, block in enumerate(model.blocks):
    h = block(h, None)
    print(f"h_layer{li}      rel err: {rel(h, meta(f'dbg/h_layer{li}')):.3e}")
print(f"h_pre_norm    rel err: {rel(h, meta('dbg/h_pre_norm')):.3e}")
h_post = rms_norm(h, model.norm, cfg.rms_norm_eps)
print(f"h_post_norm   rel err: {rel(h_post, meta('dbg/h_post_norm')):.3e}")
print("--- full ---")

logits_clean = model(idx, None)
logits_masked = model(idx, masks)


def rel_l2(a, b):
    """Relative L2 norm — the right cross-framework grad-check metric (max element-wise
    rel err explodes on near-zero elements and is misleading)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def loss_fn(m: ComponentLlama):
    return jnp.mean(m(idx, masks) ** 2)


gmodel = eqx.filter_grad(loss_fn)(model)


def clin_of(root, path):
    parts = path.split(".")
    blk = root.blocks[int(parts[1])]
    parent = blk.self_attn if parts[2] == "self_attn" else blk.mlp
    return getattr(parent, parts[3])


mc = rel_l2(logits_clean, meta("logits_clean"))
mm = rel_l2(logits_masked, meta("logits_masked"))
gV = max(rel_l2(clin_of(gmodel, p).V, d[f"META/gV/{p}"]) for p in paths)
gU = max(rel_l2(clin_of(gmodel, p).U, d[f"META/gU/{p}"]) for p in paths)
# max element-wise rel as a (looser) diagnostic
xc = rel(logits_clean, meta("logits_clean"))
xm = rel(logits_masked, meta("logits_masked"))

print(f"clean  logits  rel-L2 {mc:.3e}   (max-rel {xc:.3e})")
print(f"masked logits  rel-L2 {mm:.3e}   (max-rel {xm:.3e})")
print(f"grad V (worst) rel-L2 {gV:.3e}")
print(f"grad U (worst) rel-L2 {gU:.3e}")
tol = 2e-3
ok = max(mc, mm, gV, gU) < tol
print(f"LLAMA PARITY (rel-L2 tol {tol:.0e}):", "PASS" if ok else "FAIL")
