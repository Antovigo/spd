"""Per-residue view of the a mod 10 representation through blocks 0..LAST (position 1, alive-only model).

Input: track_build.py's stream.npz. Notation as in track_analyze.py; in addition, at a boundary X:
  z(a)       coordinates of X(a) in an orthonormal basis of D^X (9 dims; the basis is the 5 planes P_k,
             orthonormalised in order k = 1..5)
  c_r        mean of z over the 40 a with a mod 10 = r (the residue's position in D^X; ||c_r|| its radius)
  s_r        root-mean-square distance of z(a) to c_r over those 40 a (the cloud's spread)
  sep(r, r') = ||c_r - c_r'|| / sqrt((s_r^2 + s_r'^2) / 2)   separation of two residue clouds
  acc        leave-one-out nearest-class-mean accuracy of r from z(a) (chance 0.1)
For a sublayer S from boundary X to X' = X + S(X):
  w_r        mean of S(X)(a) over a with a mod 10 = r; reported ||Pi_{D^X} w_r|| (writes along the input
             representation) and ||Pi_{D^X'} w_r|| (writes into the output representation), per residue,
             relative to the residue's input radius ||Pi_{D^X} mu_r||
Writes <run>/analysis/reader_writer/track_mod10/residues.json and notebook figures mod10_track_b<l>.png.

    python track_residues.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import A
from track_analyze import basis
from track_build import LAST, OUT

R = A % 10
CM = plt.get_cmap("tab10")


def clouds(X: np.ndarray, D: np.ndarray) -> dict:
    z = X @ D
    c = np.stack([z[r == R].mean(0) for r in range(10)])
    s = np.asarray([np.sqrt(((z[r == R] - c[r]) ** 2).sum(1).mean()) for r in range(10)])
    sep = np.full((10, 10), np.inf)
    for r in range(10):
        for q in range(10):
            if q != r:
                sep[r, q] = np.linalg.norm(c[r] - c[q]) / np.sqrt((s[r] ** 2 + s[q] ** 2) / 2)
    correct = 0
    for i in range(len(z)):
        r = R[i]
        cc = c.copy()
        cc[r] = (c[r] * 40 - z[i]) / 39
        correct += int(np.argmin(((cc - z[i]) ** 2).sum(1)) == r)
    i, j = np.unravel_index(np.argmin(sep), sep.shape)
    return {
        "radius": np.linalg.norm(c, axis=1).tolist(),
        "spread": s.tolist(),
        "sep_min": float(sep.min()),
        "sep_min_pair": [int(i), int(j)],
        "sep_median": float(np.median(sep[np.isfinite(sep)])),
        "sep_nearest_per_residue": sep.min(1).tolist(),
        "acc_loo": correct / len(z),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    st = np.load(OUT / "stream.npz")
    names = [f"{t}_{b}" for b in range(LAST + 1) for t in ("pre", "mid")] + [f"pre_{LAST + 1}"]
    Xs = {n: st[n].astype(np.float64) for n in names}
    bases = {n: basis(Xs[n]) for n in names}
    res: dict = {"boundaries": {n: clouds(Xs[n], bases[n][1]) for n in names}, "sublayers": {}}
    for i in range(len(names) - 1):
        n0, n1 = names[i], names[i + 1]
        X0, X1 = Xs[n0], Xs[n1]
        D0, D1 = bases[n0][1], bases[n1][1]
        mu = np.stack([X0[r == R].mean(0) for r in range(10)])
        w = np.stack([(X1 - X0)[r == R].mean(0) for r in range(10)])
        rad = np.linalg.norm(mu @ D0, axis=1)
        res["sublayers"][n1] = {
            "write_along_input_rel": (np.linalg.norm(w @ D0, axis=1) / rad).tolist(),
            "write_into_output_rel": (np.linalg.norm(w @ D1, axis=1) / rad).tolist(),
            "radius_ratio": (
                np.linalg.norm(np.stack([X1[r == R].mean(0) for r in range(10)]) @ D1, axis=1) / rad
            ).tolist(),
        }
    (OUT / "residues.json").write_text(json.dumps(res, indent=1))
    # figures: per block, the five planes entering and leaving the block, in each boundary's own planes
    for b in range(LAST + 1):
        fig, ax = plt.subplots(2, 5, figsize=(13, 5.4))
        for row, n in enumerate((f"pre_{b}", f"pre_{b + 1}")):
            P = bases[n][0]
            for k in range(5):
                a = ax[row, k]
                q = Xs[n] @ P[k]
                if q.shape[1] == 1:
                    q = np.concatenate([q, A[:, None] / 400 * np.abs(q).max()], 1)
                for r in range(10):
                    sel = r == R
                    a.scatter(q[sel, 0], q[sel, 1], s=3, color=CM(r), alpha=0.5)
                    m = q[sel].mean(0)
                    a.text(m[0], m[1], str(r), fontsize=8, ha="center", va="center", weight="bold")
                a.plot([0], [0], "k+", ms=8)
                a.set_aspect("equal" if k < 4 else "auto")
                a.set_xticks([])
                a.set_yticks([])
                lab = "entering" if row == 0 else "leaving"
                a.set_title(
                    f"block {b} {lab}, k = {k + 1}" + (" (line vs a)" if k == 4 else ""), fontsize=8
                )
        fig.tight_layout()
        fig.savefig(args.out / "fig" / f"mod10_track_b{b}.png", dpi=90)
        plt.close(fig)
    for n in names:
        c = res["boundaries"][n]
        print(n, "acc", round(c["acc_loo"], 3), "sep_min", round(c["sep_min"], 2), c["sep_min_pair"], "median", round(c["sep_median"], 2),
              "spread", round(float(np.mean(c["spread"])), 3), "radius", round(float(np.mean(c["radius"])), 3))  # fmt: skip


if __name__ == "__main__":
    main()
