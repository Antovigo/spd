"""Figure for section 8: the residue clouds in the frequency-1 plane P_1 of the residual stream, block by block.

Position 1, alive-only model (track_build.py's stream.npz). In each panel the plane is P_1 of that boundary's own
residue means; inside the plane the axes are turned so that residue 0's mean lies on the positive horizontal
axis. Nothing is centred: the cross is the activation zero. Titles give the closest-pair and median separation in
the plane, sep(r, r') = ||c_r - c_r'|| / sqrt((s_r^2 + s_r'^2)/2) with c_r the residue's mean and s_r its RMS spread.

    python track_k1_fig.py --out <notebook dir>
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import A
from track_analyze import basis
from track_build import OUT

R = A % 10
CM = plt.get_cmap("tab10")
SHOW = (0, 2, 4, 6, 8, 11, 12, 13, 14, 15, 16, 17)


def separation(q: np.ndarray) -> tuple[float, float]:
    c = np.stack([q[r == R].mean(0) for r in range(10)])
    s = np.asarray([np.sqrt(((q[r == R] - c[r]) ** 2).sum(1).mean()) for r in range(10)])
    v = [
        np.linalg.norm(c[r] - c[p]) / np.sqrt((s[r] ** 2 + s[p] ** 2) / 2)
        for r in range(10)
        for p in range(r + 1, 10)
    ]
    return float(min(v)), float(np.median(v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    st = np.load(OUT / "stream.npz")
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
    for ax, b in zip(axes.flat, SHOW, strict=True):
        X = st[f"pre_{b}"].astype(np.float64)
        P = basis(X)[0][0]
        q = X @ P
        c0 = q[R == 0].mean(0)
        ang = np.arctan2(c0[1], c0[0])
        rot = np.asarray([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        q = q @ rot.T
        for r in range(10):
            sel = r == R
            ax.scatter(q[sel, 0], q[sel, 1], s=4, color=CM(r), alpha=0.55, lw=0)
        for r in range(10):
            m = q[r == R].mean(0)
            ax.text(m[0], m[1], str(r), fontsize=10, ha="center", va="center", weight="bold",
                    bbox={"boxstyle": "circle,pad=0.15", "fc": "white", "ec": CM(r), "lw": 1})  # fmt: skip
        ax.plot([0], [0], "k+", ms=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        smin, smed = separation(q)
        where = (
            "embedding (entering block 0)"
            if b == 0
            else ("leaving block 16" if b == 17 else f"entering block {b}")
        )
        ax.set_title(f"{where}\nclosest pair {smin:.2f}, median {smed:.2f}", fontsize=9)
    fig.suptitle("Residual stream at position 1, projected onto its frequency-1 plane P_1 (a mod 10); "
                 "dots: the 400 values of a, coloured by r = a mod 10; labels: residue means", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "mod10_track_k1.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
