"""Torch side of the Llama parity check: build the vendored ComponentLlama at a small config,
run clean + masked forwards and one backward, dump everything to npz for the JAX port to match.

Run with the repo torch venv + the feature/multipool vendored code on the path:
  PYTHONPATH=~/pd-mp ~/param-decomp/.venv/bin/python parity/dump_llama_torch.py --out parity/llama_ref.npz
"""

import argparse

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from param_decomp.components import LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import componentize_llama
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--C", type=int, default=8)
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--T", type=int, default=16)
args = ap.parse_args()

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

cfg = VendoredLlamaConfig(
    model_type="VendoredLlama",
    max_position_embeddings=8192,
    vocab_size=256,
    n_layer=2,
    n_head=4,
    n_key_value_heads=2,
    n_embd=64,
    n_intermediate=128,
    rope_theta=500000.0,
    rope_scaling=Llama3RopeScaling(),
    rms_norm_eps=1e-5,
)
model = VendoredLlama(cfg).float()

paths = [
    f"layers.{i}.{leaf}"
    for i in range(cfg.n_layer)
    for leaf in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
]
comps = {}
for path in paths:
    lin = model.get_submodule(path)
    d_out, d_in = lin.weight.shape
    comps[path] = LinearComponents(C=args.C, d_in=d_in, d_out=d_out, bias=None)
cmodel = componentize_llama(model, comps).float()

B, T, C = args.B, args.T, args.C
idx = torch.randint(0, cfg.vocab_size, (B, T))
masks = {p: torch.rand(B, T, C) for p in paths}
mask_infos = {
    p: ComponentsMaskInfo(component_mask=masks[p], routing_mask="all", weight_delta_and_mask=None)
    for p in paths
}

with sdpa_kernel(SDPBackend.MATH):  # match the JAX manual softmax numerics
    logits_clean = cmodel(idx)
    logits_masked = cmodel(idx, mask_infos)
loss = (logits_masked**2).mean()
loss.backward()

out: dict[str, np.ndarray] = {k: v.detach().cpu().numpy() for k, v in cmodel.state_dict().items()}
out["META/idx"] = idx.numpy()
out["META/logits_clean"] = logits_clean.detach().numpy()
out["META/logits_masked"] = logits_masked.detach().numpy()
out["META/loss"] = np.array(loss.item(), dtype=np.float32)
for p in paths:
    out[f"META/mask/{p}"] = masks[p].numpy()
    out[f"META/gV/{p}"] = cmodel.component_modules[p].components.V.grad.detach().numpy()
    out[f"META/gU/{p}"] = cmodel.component_modules[p].components.U.grad.detach().numpy()
for k in (
    "vocab_size",
    "n_layer",
    "n_head",
    "n_key_value_heads",
    "n_embd",
    "n_intermediate",
    "rope_theta",
    "rms_norm_eps",
    "max_position_embeddings",
):
    out[f"META/cfg/{k}"] = np.array(getattr(cfg, k))
rs = cfg.rope_scaling
out["META/cfg/rope_factor"] = np.array(rs.factor)
out["META/cfg/rope_low"] = np.array(rs.low_freq_factor)
out["META/cfg/rope_high"] = np.array(rs.high_freq_factor)
out["META/cfg/rope_orig"] = np.array(rs.original_max_position_embeddings)

# intermediate activations (clean path) to localize any port mismatch
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    emb = cmodel.embed_tokens(idx)
    h = emb
    for li, blk in enumerate(cmodel._layers):
        h = blk(h)  # clean (mask_infos=None)
        out[f"META/dbg/h_layer{li}"] = h.detach().numpy()
    out["META/dbg/emb"] = emb.detach().numpy()
    out["META/dbg/h_pre_norm"] = h.detach().numpy()
    out["META/dbg/h_post_norm"] = cmodel.norm(h).detach().numpy()
    # block-0 internal breakdown (clean)
    b0 = cmodel._layers[0]
    ln1 = b0.input_layernorm(emb)
    a_out = b0.self_attn(ln1)
    x_mid = emb + a_out
    ln2 = b0.post_attention_layernorm(x_mid)
    m_out = b0.mlp(ln2)
    for nm, arr in [("b0_ln1", ln1), ("b0_attn", a_out), ("b0_xmid", x_mid), ("b0_ln2", ln2), ("b0_mlp", m_out)]:
        out[f"META/dbg/{nm}"] = arr.detach().numpy()
    # attention internals (block 0)
    at = b0.self_attn
    hd = cfg.n_embd // cfg.n_head
    aq, ak, av = at.q_proj(ln1), at.k_proj(ln1), at.v_proj(ln1)
    y_attend = at._attend(aq, ak, av)  # pre o_proj
    qr = aq.view(B, T, cfg.n_head, hd).transpose(1, 2)
    cos, sin = at._rope_cos_sin(qr, T)
    out["META/dbg/b0_qproj"] = aq.detach().numpy()
    out["META/dbg/b0_y_attend"] = y_attend.detach().numpy()
    out["META/dbg/b0_cos"] = cos.squeeze(0).detach().numpy()  # (T, hd)
    out["META/dbg/b0_sin"] = sin.squeeze(0).detach().numpy()

np.savez(args.out, **out)
print(
    f"dumped {len(out)} arrays to {args.out} | loss={loss.item():.6f} "
    f"| logits_masked {tuple(logits_masked.shape)}"
)
