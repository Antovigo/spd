"""JAX side of the GPT-2 parity check. Load the torch dump, build the Equinox ComponentGPT2
from the same arrays, compare clean/masked logits + V/U grads (relative-L2).

  python -m parity.check_gpt2_jax --ref parity/gpt2_ref.npz
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from vendored_jax.gpt2 import (
    ComponentGPT2,
    GPT2Config,
    MaskInfo,
    all_target_paths,
    build_from_torch_state,
)

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_default_matmul_precision", "highest")  # match torch true-fp32

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
args = ap.parse_args()

d = np.load(args.ref)
meta = lambda k: d[f"META/{k}"]

cfg = GPT2Config(
    vocab_size=int(meta("cfg/vocab_size")),
    n_layer=int(meta("cfg/n_layer")),
    n_head=int(meta("cfg/n_head")),
    n_embd=int(meta("cfg/n_embd")),
    block_size=int(meta("cfg/block_size")),
)
sd = {k: jnp.asarray(v) for k, v in d.items() if not k.startswith("META/")}
model = build_from_torch_state(cfg, sd)
paths = all_target_paths(cfg)
idx = jnp.asarray(meta("idx"))
masks = {p: MaskInfo(component_mask=jnp.asarray(d[f"META/mask/{p}"])) for p in paths}


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


logits_clean = model(idx, None)
logits_masked = model(idx, masks)


def loss_fn(m: ComponentGPT2):
    return jnp.mean(m(idx, masks) ** 2)


gmodel = eqx.filter_grad(loss_fn)(model)


def clin_of(root, path):
    parts = path.split(".")
    blk = root.blocks[int(parts[1])]
    parent = blk.attn if parts[2] == "attn" else blk.mlp
    return getattr(parent, parts[3])


mc = rel_l2(logits_clean, meta("logits_clean"))
mm = rel_l2(logits_masked, meta("logits_masked"))
gV = max(rel_l2(clin_of(gmodel, p).V, d[f"META/gV/{p}"]) for p in paths)
gU = max(rel_l2(clin_of(gmodel, p).U, d[f"META/gU/{p}"]) for p in paths)

print(f"clean  logits  rel-L2 {mc:.3e}")
print(f"masked logits  rel-L2 {mm:.3e}")
print(f"grad V (worst) rel-L2 {gV:.3e}")
print(f"grad U (worst) rel-L2 {gU:.3e}")
tol = 2e-3
ok = max(mc, mm, gV, gU) < tol
print(f"GPT2 PARITY (rel-L2 tol {tol:.0e}):", "PASS" if ok else "FAIL")
