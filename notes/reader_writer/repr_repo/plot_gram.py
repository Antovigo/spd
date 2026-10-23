"""Gram matrices G[r, r'] = mu_r . mu_r' of the ten residue means of a mod 10, blocks 0..16, position 1.

mu_r in R^4096 is the mean, over the 40 operands a in 1..400 with a mod 10 = r, of the block's MLP input at
position 1 projected onto the span of the block's kept readers' read vectors (x_V, as in seed_blocks_mod10.py,
whose stored `mu` it reproduces for blocks 1..11). Nothing is centred: G includes the direction all ten means
share. G is circulant when G[r, r'] depends only on the lag (r' - r) mod 10. Each panel has its own colour range.
Titles: offdiag = share of |F G F^H|^2 off the diagonal within frequencies 1..5 (0 for circulant), and its median
over 200 random relabellings of the residues (seed_blocks_mod10.circulant).
With --cosine: cos(mu_r, mu_r') = G[r, r'] / (||mu_r|| ||mu_r'||) instead; the diagonal (1 by definition) is drawn
grey and left out of the colour range, which spans the panel's off-diagonal values.

With --centred (an explicit exception to the no-centring rule, for this view only): cos(d_r, d_r'), with
d_r = mu_r - (mean over the ten residues of mu_s), the relative orientations once the shared direction is removed;
diagonal grey, diverging colour range symmetric about 0 (+- the panel's largest off-diagonal |value|). Ten equally
spaced one-hot-like directions give -1/9 everywhere off the diagonal.

--log: logarithmic colour scale. --shared: one colour range (and one colour bar) for all blocks.

--cohen: D[r, r'] = Cohen's d between the clouds {x_V(a): a mod 10 = r} and {x_V(a): a mod 10 = r'} (40 operands
each) along the unit vector joining their centroids mu_r' -> mu_r (pooled within-cloud standard deviation, no
centring of the activations); diagonal grey.

    python plot_gram.py --out <png> [--cosine | --centred | --cohen] [--shared]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from repo import ROOT
from seed_blocks_mod10 import circulant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN, A, N  # noqa: E402

SRC = RUN / "analysis/reader_writer/variance/alive_only"
BLOCKS = range(17)


def block_xv(b: int, readers: list[dict]) -> np.ndarray:
    z = np.load(SRC / f"L{b}" / "comps.npz")
    rds = [r for r in readers if r["block"] == b]
    col = {(str(k), int(i)): j for j, (k, i) in enumerate(zip(z["kind"], z["index"], strict=True))}
    cols = np.asarray([col[(r["kind"], r["component"])] for r in rds])
    V = np.concatenate(
        [
            z[f"V_{r['kind']}"][:, [list(z["index"][z["kind"] == r["kind"]]).index(r["component"])]]
            for r in rds
        ],
        1,
    ).astype(np.float64)
    T = np.asarray(
        np.load(SRC / f"L{b}" / "inner.npy", mmap_mode="r")[(A - 1) * N, 0, :][:, cols], np.float64
    )
    return (V @ np.linalg.pinv(V.T @ V) @ T.T).T


def residue_means(xv: np.ndarray) -> np.ndarray:
    return np.stack([xv[r == A % 10].mean(0) for r in range(10)])


def cohen(xv: np.ndarray) -> np.ndarray:
    """D[r, r'] = (mu_r - mu_r') . u / s, u the unit vector from mu_r' to mu_r, s the pooled within-cloud standard
    deviation of x_V(a) . u over the two clouds (40 operands each). Symmetric; the diagonal is NaN."""
    mu = residue_means(xv)
    D = np.full((10, 10), np.nan)
    for r in range(10):
        for q in range(r + 1, 10):
            d = mu[r] - mu[q]
            u = d / np.linalg.norm(d)
            pr, pq = xv[r == A % 10] @ u, xv[q == A % 10] @ u
            s = np.sqrt((pr.var(ddof=1) + pq.var(ddof=1)) / 2)
            D[r, q] = D[q, r] = np.linalg.norm(d) / s
    return D


def matrix(mu: np.ndarray, mode: str) -> np.ndarray:
    match mode:
        case "gram":
            return mu @ mu.T
        case "cosine":
            n = np.linalg.norm(mu, axis=1)
            return (mu @ mu.T) / np.outer(n, n)
        case "centred":
            d = mu - mu.mean(0)
            n = np.linalg.norm(d, axis=1)
            return (d @ d.T) / np.outer(n, n)
        case _:
            raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--centred", action="store_true")
    ap.add_argument("--cohen", action="store_true")
    ap.add_argument("--log", action="store_true", help="logarithmic colour scale (Cohen's d)")
    ap.add_argument("--shared", action="store_true", help="one colour range for all blocks")
    args = ap.parse_args()
    mode = (
        "cohen"
        if args.cohen
        else "centred"
        if args.centred
        else "cosine"
        if args.cosine
        else "gram"
    )
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    reps = {
        r["id"]: r
        for r in json.loads((ROOT / "representations.json").read_text())["representations"]
    }
    eye = np.eye(10, dtype=bool)
    mats, circs = {}, {}
    for b in BLOCKS:
        xv = block_xv(b, readers)
        mu = residue_means(xv)
        rid = f"L{b}.p1.a_mod_10.means"
        if rid in reps:
            stored = np.load(ROOT / reps[rid]["vectors"])["mu"]
            assert np.abs(stored - mu).max() <= 1e-6 * np.abs(stored).max(), b
        circs[b] = circulant(mu)
        mats[b] = cohen(xv) if mode == "cohen" else matrix(mu, mode)

    def vrange(M: np.ndarray) -> tuple[float, float]:
        v = M if mode == "gram" else M[~eye]
        if mode == "centred":
            m = float(np.abs(v).max())
            return -m, m
        return float(v.min()), float(v.max())

    if args.shared:
        lims = np.array([vrange(M) for M in mats.values()])
        shared = (float(lims[:, 0].min()), float(lims[:, 1].max()))
    cmap = plt.get_cmap("RdBu_r" if mode == "centred" else "viridis").copy()
    cmap.set_bad("#bbbbbb")
    fig, grid = plt.subplots(3, 6, figsize=(16, 8.4))
    for b, M in mats.items():
        ax = grid[b // 6, b % 6]
        lo, hi = shared if args.shared else vrange(M)
        shown = M if mode == "gram" else np.ma.masked_where(eye, M)
        if args.log:
            im = ax.imshow(shown, cmap=cmap, norm=LogNorm(vmin=lo, vmax=hi))
        else:
            im = ax.imshow(shown, cmap=cmap, vmin=lo, vmax=hi)
        if not args.shared:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        c = circs[b]
        sub = (
            f"median d {np.median(M[~eye]):.1f}, min {M[~eye].min():.1f}"
            if mode == "cohen"
            else f"offdiag {c['offdiag_share_k1to5']:.2f} (shuffled {c['shuffled_median']:.2f})"
        )
        ax.set_title(f"block {b}\n{sub}", fontsize=8)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
    grid[-1, -1].axis("off")
    if args.shared:
        fig.colorbar(im, ax=grid[-1, -1], fraction=0.5, pad=0.0)
    label = {
        "cohen": "Cohen's d, clouds r and r'\nalong μ_r − μ_r'",
        "centred": "cos(μ_r − μ̄, μ_r' − μ̄)",
        "cosine": "cos(μ_r, μ_r')",
        "gram": "G[r, r'] = μ_r · μ_r'",
    }
    for i in range(3):
        grid[i, 0].set_ylabel(f"{label[mode]}\nrow r, column r'", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
