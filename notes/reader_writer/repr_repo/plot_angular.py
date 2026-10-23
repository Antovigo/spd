"""How much of the sphere the operand a occupies, blocks 0..16, position 1 (the token a), a = 1..400.

Two spaces per block:
  full   x(a) = the block's MLP input (post-attention RMSNorm with its gain) in R^4096, from the float64 rebuild of
         the alive-only model (track_build.py, track_mod10/stream.npz)
  span   x_V(a) = the stored MLP input projected onto the span of the block's kept readers' read vectors
         (plot_gram.block_xv), what the readers see
Directions only: xhat(a) = x(a) / ||x(a)||, read from the origin (nothing is centred). mu_r = mean of x(a) over the
40 operands with a mod 10 = r, r = 0..9; mhat = the unit mean of xhat(a) over a.
Metrics (angles in degrees):
  theta_means   mean over the 45 pairs r < r' of angle(mu_r, mu_r'): the angular extent of the mod-10 means
  theta_points  mean over all pairs a < a' of angle(x(a), x(a')): the angular extent of the whole cloud
  theta_axis    RMS over a of angle(x(a), mhat): the cone half-width around the mean direction
  pr_tangent    participation ratio (sum lambda)^2 / sum lambda^2 of the eigenvalues lambda of T^T T, T (400 x d)
                the tangent parts xhat(a) - (xhat(a) . mhat) mhat: how many directions the cone spreads over

    python plot_angular.py --out <png>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_gram import BLOCKS, block_xv, residue_means
from repo import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from track_build import OUT, load_blocks, rmsnorm  # noqa: E402


def angles(X: np.ndarray) -> np.ndarray:
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = np.clip(U @ U.T, -1, 1)
    return np.degrees(np.arccos(C[np.triu_indices(len(X), 1)]))


def metrics(X: np.ndarray) -> dict[str, float]:
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    m = U.mean(0)
    m /= np.linalg.norm(m)
    T = U - np.outer(U @ m, m)
    lam = np.linalg.svd(T, compute_uv=False) ** 2
    return {
        "theta_means": float(angles(residue_means(X)).mean()),
        "theta_points": float(angles(X).mean()),
        "theta_axis": float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(U @ m, -1, 1))) ** 2))),
        "pr_tangent": float(lam.sum() ** 2 / (lam**2).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    blocks, _, _ = load_blocks()
    S = np.load(OUT / "stream.npz")
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    res = {"full": [], "span": []}
    for b in BLOCKS:
        res["full"].append(metrics(rmsnorm(S[f"mid_{b}"].astype(np.float64), blocks[b].g2)))
        res["span"].append(metrics(block_xv(b, readers)))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    keys = [("theta_means", "θ_means: mean angle between\nresidue means μ_r (deg)"),
            ("theta_points", "θ_points: mean angle between\npoints x(a), x(a') (deg)"),
            ("theta_axis", "θ_axis: RMS angle of x(a) to\nthe mean direction (deg)"),
            ("pr_tangent", "PR of the tangent spread\n(number of directions)")]  # fmt: skip
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    bl = list(BLOCKS)
    for ax, (k, lab) in zip(axs, keys, strict=True):
        for sp, c, name in (("full", "#1d1d1b", "full MLP input (4096 dims)"),
                            ("span", "#2a5db0", "readers' span x_V")):  # fmt: skip
            ax.plot(bl, [r[k] for r in res[sp]], marker="o", ms=3, color=c, label=name)
        ax.set_xlabel("block")
        ax.set_title(lab, fontsize=9)
        ax.set_xticks(bl)
        ax.tick_params(labelsize=7)
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    for sp in res:
        print(sp)
        for b, r in zip(bl, res[sp], strict=True):
            print(f"  {b:2d} " + "  ".join(f"{k} {v:6.2f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
