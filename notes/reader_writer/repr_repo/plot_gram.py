"""Gram matrices G[r, r'] = mu_r . mu_r' of the ten residue means of a mod 10, blocks 0..16, position 1.

mu_r in R^4096 is the mean, over the 40 operands a in 1..400 with a mod 10 = r, of the block's MLP input at
position 1 projected onto the span of the block's kept readers' read vectors (x_V, as in seed_blocks_mod10.py,
whose stored `mu` it reproduces for blocks 1..11). Nothing is centred: G includes the direction all ten means
share. G is circulant when G[r, r'] depends only on the lag (r' - r) mod 10. Each panel has its own colour range.
Titles: offdiag = share of |F G F^H|^2 off the diagonal within frequencies 1..5 (0 for circulant), and its median
over 200 random relabellings of the residues (seed_blocks_mod10.circulant).
With --cosine: cos(mu_r, mu_r') = G[r, r'] / (||mu_r|| ||mu_r'||) instead; the diagonal (1 by definition) is drawn
grey and left out of the colour range, which spans the panel's off-diagonal values.

    python plot_gram.py --out <png> [--cosine]
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
from seed_blocks_mod10 import circulant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import RUN, A, N  # noqa: E402

SRC = RUN / "analysis/reader_writer/variance/alive_only"
BLOCKS = range(17)


def block_mu(b: int, readers: list[dict]) -> np.ndarray:
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
    xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T
    return np.stack([xv[r == A % 10].mean(0) for r in range(10)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cosine", action="store_true")
    args = ap.parse_args()
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    reps = {
        r["id"]: r
        for r in json.loads((ROOT / "representations.json").read_text())["representations"]
    }
    fig, grid = plt.subplots(3, 6, figsize=(16, 8.4))
    for b in BLOCKS:
        mu = block_mu(b, readers)
        rid = f"L{b}.p1.a_mod_10.means"
        if rid in reps:
            stored = np.load(ROOT / reps[rid]["vectors"])["mu"]
            assert np.abs(stored - mu).max() <= 1e-6 * np.abs(stored).max(), b
        circ = circulant(mu)
        G = mu @ mu.T
        ax = grid[b // 6, b % 6]
        if args.cosine:
            n = np.linalg.norm(mu, axis=1)
            C = G / np.outer(n, n)
            off = C[~np.eye(10, dtype=bool)]
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#bbbbbb")
            im = ax.imshow(
                np.ma.masked_where(np.eye(10, dtype=bool), C),
                cmap=cmap,
                vmin=off.min(),
                vmax=off.max(),
            )
        else:
            im = ax.imshow(G, cmap="viridis", vmin=G.min(), vmax=G.max())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(
            f"block {b}\noffdiag {circ['offdiag_share_k1to5']:.2f} (shuffled {circ['shuffled_median']:.2f})",
            fontsize=8,
        )
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
    grid[-1, -1].axis("off")
    for i in range(3):
        label = "cos(μ_r, μ_r')" if args.cosine else "G[r, r'] = μ_r · μ_r'"
        grid[i, 0].set_ylabel(f"{label}\nrow r, column r'", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
