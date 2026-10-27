"""The mod-20 plane of the stream at "=" after the copy, and the same plane before the copy and in the attention's write
(copy of a by block 16, copy of b by block 15; alive-only and original models). Nothing is centred: the cross in every panel
is the activation zero.

Notation (o = the copied operand, integer 1..400; s = o mod 20; copy_build.py's prompt sets: every o, ten values of the other
operand, both operations; 8,000 prompts in the alive-only model, copy_figs.py's 2,000-prompt cache in the original model):
  E        the stream at "=" entering the copying block
  W        the copying block's whole attention output at "=" (all heads, all source positions)
  E'       = E + W, the stream at "=" after the attention
  P'       the frequency-1 plane of o mod 20 of E': span{Re Psi_1, Im Psi_1}, Psi_1 = sum_s m_s e^{-2 pi i s / 20} with m_s the
           mean of E' over the prompts with o mod 20 = s; turned so that the vector from residue 10's mean to residue 0's mean
           points right
  panels   P'^T E', P'^T W and P'^T E for every prompt (E' = E + W, so the first panel is the sum of the other two), coloured by
           the raw value of o; labels: the residue means of s
Writes <out>/mod20_eq_planes.png and <run>/analysis/reader_writer/track_mod20/eq_planes.json (per panel: the mean of the
points, their root-mean-square distance to that mean, and the norm of Psi_1 in the plane).

    python mod20_eq_planes.py --out notebook_mod20/fig
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy_analyze import attention_parts
from copy_build import OUT as COPY
from copy_figs import SUB
from mod10_L0 import HUB
from mod20_copy import orig_parts
from mod20_track import OUT
from track_build import load_blocks

WHICH = {"a": 16, "b": 15}


def means(Z: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    return np.stack([Z[lab == q].mean(0) for q in range(n)])


def plane(Z: np.ndarray, oper: np.ndarray) -> np.ndarray:
    s = oper % 20
    mu = means(Z, s, 20)
    p = np.exp(-2j * np.pi * np.arange(20) / 20) @ mu
    P = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)[0]
    v = (mu[0] - mu[10]) @ P
    ang = np.arctan2(v[1], v[0])
    return P @ np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks = load_blocks()[0]
    data = {}
    for which, L in WHICH.items():
        st = np.load(COPY / f"stream_{which}.npz")
        X = st[f"X{L}"].astype(np.float64)
        parts = attention_parts(blocks[L], X)
        n = len(X)
        z = parts["p"][:, :, :, None] * parts["v"][:, :, parts["rep"]].transpose(
            0, 2, 1, 3
        )  # (n, 32, T, 128)
        W = blocks[L].proj(z.sum(2).reshape(n, -1), "o")
        del parts, z
        data[("alive", which)] = (X[:, 4], W, st[which].astype(int))
        Xo = np.load(COPY / f"orig_cache_{which}.npz")["Xor"].astype(np.float64)
        _, p, v, Wo = orig_parts(snap, wmap, cfg, L, Xo)
        Wor = np.einsum("nht,nthd->nhd", p, v).reshape(len(Xo), -1) @ Wo.reshape(4096, -1).T
        data[("orig", which)] = (Xo[:, 4], Wor, st[which][SUB[which]].astype(int))
        del p, v, X
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 17))
    norm = plt.Normalize(1, 400)
    stats = {}
    for row, (model, which) in enumerate(
        (("alive", "a"), ("alive", "b"), ("orig", "a"), ("orig", "b"))
    ):
        E, W, oper = data[(model, which)]
        P = plane(E + W, oper)
        L = WHICH[which]
        for col, (name, Z) in enumerate(((f'"=" after block {L}\'s attention (E\')', E + W),
                                         (f"block {L}'s attention write (W)", W), ('"=" before the attention (E)', E))):  # fmt: skip
            ax = axes[row, col]
            q = Z @ P
            sc = ax.scatter(
                q[:, 0], q[:, 1], c=oper, cmap="viridis", norm=norm, s=3, alpha=0.6, lw=0
            )
            m = means(q, oper % 20, 20)
            for s in range(20):
                ax.text(*m[s], str(s), fontsize=6.5, ha="center", va="center",
                        bbox={"boxstyle": "circle,pad=0.1", "fc": "white", "ec": "0.4", "lw": 0.5, "alpha": 0.85})  # fmt: skip
            ax.plot([0], [0], "k+", ms=10, mew=1.5)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            mn = {"alive": "components", "orig": "neurons"}[model]
            ax.set_title(f"{mn}, copy of {which}: {name}", fontsize=8.5)
            psi1 = np.linalg.norm(np.exp(-2j * np.pi * np.arange(20) / 20) @ m)
            stats[f"{model}_{which}_{['after', 'write', 'before'][col]}"] = {
                "mean": q.mean(0).tolist(), "rms_spread": float(np.sqrt(((q - q.mean(0)) ** 2).sum(1).mean())),
                "psi1_norm_in_plane": float(psi1)}  # fmt: skip
        # one shared range per row so the three panels are comparable
        lims = np.asarray([[*axes[row, c].get_xlim(), *axes[row, c].get_ylim()] for c in range(3)])
        lo, hi = lims[:, [0, 2]].min(), lims[:, [1, 3]].max()
        for c in range(3):
            axes[row, c].set_xlim(lo, hi)
            axes[row, c].set_ylim(lo, hi)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01, label="value of the copied operand")
    fig.savefig(args.out / "mod20_eq_planes.png", dpi=95, bbox_inches="tight")
    (OUT / "eq_planes.json").write_text(json.dumps(stats, indent=1))
    for k, v_ in stats.items():
        print(
            k,
            {
                kk: (np.round(vv, 3).tolist() if isinstance(vv, list) else round(vv, 3))
                for kk, vv in v_.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
