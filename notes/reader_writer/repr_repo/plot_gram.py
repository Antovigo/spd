"""Gram matrices G[r, r'] = mu_r . mu_r' of the ten residue means of the a mod 10 representations, blocks 0..11.

mu_r in R^4096 is the mean, over the 40 operands a in 1..400 with a mod 10 = r, of the block's MLP input at
position 1 projected onto the span of the kept readers' read vectors (x_V, as in seed_blocks_mod10.py; for
blocks 1..11 it is the stored `mu` of L<b>.p1.a_mod_10.means). Nothing is centred: G includes the direction all
ten means share. G is circulant when G[r, r'] depends only on the lag (r' - r) mod 10.
Rows 1 and 3: G (each panel its own colour range), blocks 0..5 and 6..11. Rows 2 and 4: G minus its circulant part C[r, r'] = mean of G over the
cyclic diagonal of lag (r' - r) mod 10, on a symmetric range of half the panel above's range. Titles: offdiag = share of |F G F^H|^2 off
the diagonal within frequencies 1..5 (0 for circulant), and the median over 200 random relabellings.

    python plot_gram.py --out <png>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from repo import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN, A, N  # noqa: E402

SRC = RUN / "analysis/reader_writer/variance/alive_only"


def block0_mu() -> np.ndarray:
    z = np.load(SRC / "L0" / "comps.npz")
    rds = [r for r in json.loads((ROOT / "readers.json").read_text())["readers"] if r["block"] == 0]
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
        np.load(SRC / "L0" / "inner.npy", mmap_mode="r")[(A - 1) * N, 0, :][:, cols], np.float64
    )
    xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T
    return np.stack([xv[r == A % 10].mean(0) for r in range(10)])


def circulant_part(G: np.ndarray) -> np.ndarray:
    lag = (np.arange(10)[None] - np.arange(10)[:, None]) % 10
    c = np.array([G[lag == k].mean() for k in range(10)])
    return c[lag]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reps = {
        r["id"]: r
        for r in json.loads((ROOT / "representations.json").read_text())["representations"]
    }
    fig, grid = plt.subplots(4, 6, figsize=(15, 10.5))
    for b in range(12):
        if b == 0:
            mu = block0_mu()
            from seed_blocks_mod10 import circulant

            circ = circulant(mu)
        else:
            rep = reps[f"L{b}.p1.a_mod_10.means"]
            mu = np.load(ROOT / rep["vectors"])["mu"]
            circ = rep["details"]["circulant"]
        G = mu @ mu.T
        top, bot = grid[2 * (b // 6), b % 6], grid[2 * (b // 6) + 1, b % 6]
        lo, hi = G.min(), G.max()
        im = top.imshow(G, cmap="viridis", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=top, fraction=0.046, pad=0.03)
        top.set_title(
            f"block {b}\noffdiag {circ['offdiag_share_k1to5']:.2f} (shuffled {circ['shuffled_median']:.2f})",
            fontsize=8,
        )
        R = G - circulant_part(G)
        m = (hi - lo) / 2
        im = bot.imshow(R, cmap="RdBu_r", vmin=-m, vmax=m)
        fig.colorbar(im, ax=bot, fraction=0.046, pad=0.03)
        for ax in (top, bot):
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.tick_params(labelsize=6)
    for i in (0, 2):
        grid[i, 0].set_ylabel("G[r, r'] = μ_r · μ_r'\nrow r, column r'", fontsize=8)
        grid[i + 1, 0].set_ylabel("G − circulant part", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
