"""Area of the spherical convex hull of the ten residue-mean directions, blocks 0..16, position 1.

mu_r (r = a mod 10 = 0..9) is the mean over the 40 operands a with a mod 10 = r of the block's MLP input, in two spaces
(as in plot_angular.py): "full" x(a) in R^4096, and "span" x_V(a). Read from the origin, nothing centred.
Directions b_r = mu_r / ||mu_r||; B = [b_0 .. b_9] (4096 x 10, rank n = 10). The minimal convex set on the sphere
containing the b_r is the spherical simplex they span: the cone {B w : w >= 0} cut by the unit sphere of the
10-dimensional span of B, a piece of the 9-sphere S^9.
Its area (9-dimensional, absolute) is
    Omega = |det B| * integral over lambda in the simplex {lambda >= 0, sum lambda = 1} of ||B lambda||^(-n) dlambda
(dlambda: Lebesgue measure on (lambda_0..lambda_8), total 1/(n-1)!), from vol(cone ∩ unit ball) = Omega / n with
x = t B lambda. |det B| = sqrt(det C), C = B^T B the cosine matrix. Monte Carlo over lambda ~ Dirichlet(1, ..., 1):
Omega = sqrt(det C) E[||B lambda||^(-n)] / (n-1)!.
Reported: log10 of Omega / |S^9| (the fraction of the 9-sphere, |S^9| = 2 pi^5 / Gamma(5)), and rho = the half-angle of
the spherical cap on S^9 with the same area (degrees), from cap fraction = I(sin^2 rho; (n-1)/2, 1/2) / 2.

    python plot_hull.py --out <png>
"""

import argparse
import json
import sys
from math import factorial, gamma, log10, pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_gram import BLOCKS, block_xv, residue_means
from repo import ROOT
from scipy.optimize import brentq
from scipy.special import betainc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from track_build import OUT, load_blocks, rmsnorm  # noqa: E402

N = 10
SPHERE = 2 * pi ** (N / 2) / gamma(N / 2)


def hull(mu: np.ndarray, rng: np.random.Generator, n_mc: int = 200_000) -> dict[str, float]:
    Bm = (mu / np.linalg.norm(mu, axis=1, keepdims=True)).T
    C = Bm.T @ Bm
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    lam = rng.dirichlet(np.ones(N), n_mc)
    f = np.einsum("ij,jk,ik->i", lam, C, lam) ** (-N / 2)  # ||B lambda||^(-n)
    log_omega = 0.5 * logdet + np.log(f.mean()) - np.log(factorial(N - 1))
    frac = log_omega / np.log(10) - log10(SPHERE)
    target = 10.0**frac

    def cap(rho: float) -> float:
        return 0.5 * betainc((N - 1) / 2, 0.5, np.sin(rho) ** 2) - target

    rho = float(np.degrees(brentq(cap, 1e-9, pi / 2))) if target < 0.5 else float("nan")
    return {"log10_fraction": float(frac), "cap_deg": rho, "mc_rel_se": float(f.std() / f.mean() / np.sqrt(n_mc)),
            "log10_detC": float(logdet / np.log(10))}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    blocks, _, _ = load_blocks()
    S = np.load(OUT / "stream.npz")
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    res: dict[str, list[dict[str, float]]] = {"full": [], "span": []}
    for b in BLOCKS:
        x = rmsnorm(S[f"mid_{b}"].astype(np.float64), blocks[b].g2)
        res["full"].append(hull(residue_means(x), rng))
        res["span"].append(hull(residue_means(block_xv(b, readers)), rng))
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    bl = list(BLOCKS)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for sp, c, name in (
        ("full", "#1d1d1b", "full MLP input"),
        ("span", "#2a5db0", "readers' span x_V"),
    ):
        axs[0].plot(
            bl, [r["log10_fraction"] for r in res[sp]], marker="o", ms=3, color=c, label=name
        )
        axs[1].plot(bl, [r["cap_deg"] for r in res[sp]], marker="o", ms=3, color=c, label=name)
    axs[0].set_title("log10 (hull area / area of S⁹)", fontsize=9)
    axs[1].set_title("half-angle of the S⁹ cap with the same area (deg)", fontsize=9)
    axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.set_xlabel("block")
        ax.set_xticks(bl)
        ax.tick_params(labelsize=7)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    for sp in res:
        print(sp)
        for b, r in zip(bl, res[sp], strict=True):
            print(f"  {b:2d} " + "  ".join(f"{k} {v:8.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
