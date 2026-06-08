"""Torch side of the GPT-2 parity check. Build the vendored ComponentGPT2 at a small config,
run clean + masked forwards + one backward (GPU, true fp32), dump to npz.

  PYTHONPATH=~/pd-mp ~/param-decomp/.venv/bin/python parity/dump_gpt2_torch.py --out parity/gpt2_ref.npz
"""

import argparse

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from param_decomp.components import LinearComponents
from param_decomp.masks import ComponentsMaskInfo
from param_decomp_lab.experiments.lm.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from param_decomp_lab.experiments.lm.vendored.gpt2 import componentize_gpt2

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--C", type=int, default=8)
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--T", type=int, default=16)
args = ap.parse_args()

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"

cfg = GPT2SimpleConfig(
    model_type="GPT2Simple",
    block_size=64,
    vocab_size=256,
    n_layer=2,
    n_head=4,
    n_embd=64,
    flash_attention=True,  # run under SDPBackend.MATH so it matches the JAX manual softmax
)
model = GPT2Simple(cfg)

paths = [
    f"h.{i}.{leaf}"
    for i in range(cfg.n_layer)
    for leaf in (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    )
]
comps = {}
for path in paths:
    lin = model.get_submodule(path)
    d_out, d_in = lin.weight.shape
    comps[path] = LinearComponents(C=args.C, d_in=d_in, d_out=d_out, bias=lin.bias.data.clone())
cmodel = componentize_gpt2(model, comps).to(DEV)

B, T, C = args.B, args.T, args.C
idx = torch.randint(0, cfg.vocab_size, (B, T), device=DEV)
masks = {p: torch.rand(B, T, C, device=DEV) for p in paths}
mask_infos = {
    p: ComponentsMaskInfo(component_mask=masks[p], routing_mask="all", weight_delta_and_mask=None)
    for p in paths
}

with sdpa_kernel(SDPBackend.MATH):
    logits_clean = cmodel(idx)
    logits_masked = cmodel(idx, mask_infos)
loss = (logits_masked**2).mean()
loss.backward()

out: dict[str, np.ndarray] = {k: v.detach().cpu().numpy() for k, v in cmodel.state_dict().items()}
out["META/idx"] = idx.cpu().numpy()
out["META/logits_clean"] = logits_clean.detach().cpu().numpy()
out["META/logits_masked"] = logits_masked.detach().cpu().numpy()
out["META/loss"] = np.array(loss.item(), dtype=np.float32)
for p in paths:
    out[f"META/mask/{p}"] = masks[p].cpu().numpy()
    out[f"META/gV/{p}"] = cmodel.component_modules[p].components.V.grad.detach().cpu().numpy()
    out[f"META/gU/{p}"] = cmodel.component_modules[p].components.U.grad.detach().cpu().numpy()
for k in ("vocab_size", "n_layer", "n_head", "n_embd", "block_size"):
    out[f"META/cfg/{k}"] = np.array(getattr(cfg, k))

np.savez(args.out, **out)
print(
    f"dumped {len(out)} arrays to {args.out} | loss={loss.item():.6f} | logits {tuple(logits_masked.shape)}"
)
