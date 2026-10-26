"""Figures for section 12 of the notebook page "Mod 10 representations": the copies of a and b at "=" in their k = 1
Fourier plane, coloured by the copying head's attention weight on the source (alive-only and original models), and the
splits of their spread (from p1_magnitude.json and weight_qk_split.json).

Copy = the attention write at "=" (position 4) from the source position (a: block 16, position 1, head 21; b: block 15,
position 3, head 13); P_1 = the k = 1 plane of the copy's residue means; p = the head's weight on the source.
Writes notebook/fig/copyvar_{clouds,split}.png.

    python copyvar_figs.py --out <notebook dir>
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import b_subclouds
import matplotlib.pyplot as plt
import numpy as np
import smear_a
from copy_analyze import attention_parts, copied
from copy_build import OUT
from copy_figs import SUB, planes
from mod10_L0 import HUB
from track_build import load_blocks

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
CASES = {"a": (16, 1, 21), "b": (15, 3, 13)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (snap,) = glob.glob(HUB)
    cfg = json.loads(Path(snap, "config.json").read_text())
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    blocks = load_blocks()[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    for i, (w_, (L, s, H)) in enumerate(CASES.items()):
        st = np.load(OUT / f"stream_{w_}.npz")
        sub = SUB[w_]
        lab = (st[w_][sub] % 10).astype(int)
        for j, name in enumerate(("alive-only", "original")):
            if name == "alive-only":
                parts = attention_parts(blocks[L], st[f"X{L}"][sub].astype(np.float64))
                p, Os = parts["p"][:, H, s], copied(blocks[L], parts, s)
            else:
                X = np.load(OUT / f"orig_cache_{w_}.npz")["Xor"].astype(np.float64)
                if w_ == "a":
                    pp, Osrc, _ = smear_a.orig_parts(snap, wmap, cfg, X)
                    p, Os = pp[:, H, s], Osrc[:, s]
                else:
                    pp, Os, _ = b_subclouds.orig_parts(snap, wmap, cfg, X)
                    p = pp[:, H, s]
            P1 = planes(Os, lab)[0][0]
            z = Os @ P1
            ax = axes[i, j]
            sc = ax.scatter(z[:, 0], z[:, 1], c=p, cmap="viridis", vmin=0, vmax=1, s=3, lw=0)
            for r in range(10):
                m = z[lab == r].mean(0)
                ax.text(
                    *m,
                    str(r),
                    fontsize=10,
                    weight="bold",
                    ha="center",
                    va="center",
                    color="#c0392b",
                )
            ax.plot([0], [0], "k+", ms=10)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f'copy of {w_} at "=" (block {L}), k = 1, {name}\ncolour: head {H}\'s weight on position {s}',
                fontsize=9,
            )
    fig.colorbar(sc, ax=axes, shrink=0.5, label="attention weight on the source")
    fig.savefig(args.out / "fig" / "copyvar_clouds.png", bbox_inches="tight")
    plt.close(fig)

    pm = json.loads((OUT / "p1_magnitude.json").read_text())
    qk = json.loads((OUT / "weight_qk_split.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    names = ("alive-only", "original")
    segs = [("other heads", "share_other_heads", "#bbbbbb"), ("head 21: weight", "head21_weight_p21", "#2a5db0"),
            ("head 21: value", "head21_value_c", "#e6a23c"), ("head 21: product", "head21_product", "#7a7a7a")]  # fmt: skip
    for y, nm in enumerate(names):
        c = pm[nm]["copy_alone"]
        left = 0.0
        for lbl, key, col in segs:
            v = c[key] * (c["share_head21"] if key.startswith("head21_") else 1)
            ax.barh(y, v, left=left, color=col, label=lbl if y == 0 else None)
            left += v
    ax.set_yticks(range(2))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the within-residue radial spread of the copy of a (k = 1)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax = axes[1]
    rows = [(f"{w_}, {nm}", qk[f"{w_}/{nm}"]["share"]) for w_ in ("a", "b") for nm in names]
    cols = {"Q": "#2a5db0", "K_src": "#e6a23c", "K_rest": "#3c9d5d"}
    lbl = {
        "Q": 'query at "="',
        "K_src": "key at the source",
        "K_rest": "keys at the other positions",
    }
    for y, (_nm, sh) in enumerate(rows):
        left = 0.0
        for f in ("Q", "K_src", "K_rest"):
            ax.barh(y, sh[f], left=left, color=cols[f], label=lbl[f] if y == 0 else None)
            ax.text(
                left + sh[f] / 2,
                y,
                f"{sh[f]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
            left += sh[f]
    ax.set_yticks(range(4))
    ax.set_yticklabels(
        [
            f"head 21 on a, {r[0].split(', ')[1]}"
            if r[0][0] == "a"
            else f"head 13 on b, {r[0].split(', ')[1]}"
            for r in rows
        ]
    )
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("share of the within-residue variance of the copying head's weight on the source")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "copyvar_split.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
