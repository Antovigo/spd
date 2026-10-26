"""Pairwise geometry of the 20 residue means of a mod 20 at position 1, blocks 0..16 (alive-only model).

Definitions as on the mod-10 geometry page (../repr_repo/plot_gram.py), with a mod 10 replaced by a mod 20, in three spaces:
  x       the full block MLP input of the alive-only model (post-attention RMSNorm output of track_build.py's mid_<b>, R^4096)
  x_orig  the same for the original model (track_build_orig.py)
  x_V     x projected onto the span of the block's kept readers' read vectors (plot_gram.block_xv; the mod-10 page's space)
Nothing is centred.

Notation (a = integer operand 1..400; s = a mod 20, integer 0..19, 20 operands per residue):
  y(a)         the MLP input in one of the three spaces
  mu_s         mean of y(a) over the 20 operands with a mod 20 = s
  cos[s, s']   mu_s . mu_s' / (||mu_s|| ||mu_s'||)
  d[s, s']     Cohen's d of the clouds {y(a): a mod 20 = s} and {... = s'} along u = (mu_s - mu_s') / ||mu_s - mu_s'||:
               ||mu_s - mu_s'|| / sqrt((var_s + var_s') / 2), var the unbiased variance of y(a) . u within a cloud
  pair classes antipodal: s' = s + 10 (same a mod 10, other tens parity h); same r is the same thing;
               lag 1: s' = s +- 1 mod 20; other: every remaining pair
Writes <out>/mod20_geom_{cos,cohen}_{x,x_orig,x_V}.png and track_mod20/geometry.json. Cosine panels have their own
colour range (the shared direction puts every cosine near 1 from block 1 on); Cohen's d uses one logarithmic range per figure.

    python mod20_geometry.py --out notebook_mod20/fig
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
from mod20_track import OUT
from track_build import OUT as AOUT
from track_build import load_blocks, rmsnorm
from track_build_orig import OUT as OOUT
from track_build_orig import loader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repr_repo"))
from mod10_L0 import A  # noqa: E402
from plot_gram import BLOCKS, block_xv  # noqa: E402
from repo import ROOT  # noqa: E402

S = A % 20
EYE = np.eye(20, dtype=bool)
LAG = (np.arange(20)[:, None] - np.arange(20)[None]) % 20
CLASSES = {
    "antipodal": LAG == 10,
    "lag 1": (LAG == 1) | (LAG == 19),
    "other": ~EYE & (LAG != 10) & (LAG != 1) & (LAG != 19),
}


def residue_means(xv: np.ndarray) -> np.ndarray:
    return np.stack([xv[s == S].mean(0) for s in range(20)])


def cohen(xv: np.ndarray, mu: np.ndarray) -> np.ndarray:
    D = np.full((20, 20), np.nan)
    for s in range(20):
        for q in range(s + 1, 20):
            d = mu[s] - mu[q]
            u = d / np.linalg.norm(d)
            ps, pq = xv[s == S] @ u, xv[q == S] @ u
            D[s, q] = D[q, s] = np.linalg.norm(d) / np.sqrt((ps.var(ddof=1) + pq.var(ddof=1)) / 2)
    return D


def grid(mats: dict[int, np.ndarray], label: str, log: bool, path: Path, sub, title: str) -> None:
    vals = np.concatenate([M[~EYE] for M in mats.values()])
    lo, hi = float(vals.min()), float(vals.max())
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bbbbbb")
    fig, axes = plt.subplots(3, 6, figsize=(17, 8.8))
    for b, M in mats.items():
        ax = axes[b // 6, b % 6]
        shown = np.ma.masked_where(EYE, M)
        if log:
            im = ax.imshow(shown, cmap=cmap, norm=LogNorm(vmin=lo, vmax=hi))
        else:
            im = ax.imshow(shown, cmap=cmap, vmin=float(M[~EYE].min()), vmax=float(M[~EYE].max()))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        for e in (4.5, 9.5, 14.5):
            ax.axhline(e, color="w", lw=0.6)
            ax.axvline(e, color="w", lw=0.6)
        ax.set_xticks(range(0, 20, 5))
        ax.set_yticks(range(0, 20, 5))
        ax.tick_params(labelsize=6)
        ax.set_title(f"block {b}\n{sub(M)}", fontsize=7.5)
    axes[-1, -1].axis("off")
    if log:
        cb = fig.colorbar(im, ax=axes[-1, -1], fraction=0.5, pad=0.0)
        ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo <= t <= hi]
        cb.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    fig.suptitle(title, fontsize=10)
    for i in range(3):
        axes[i, 0].set_ylabel(f"{label}\nrow s, column s'", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def spaces(
    b: int, readers: list[dict], st: dict, blocks_alive, g2_orig: dict
) -> dict[str, np.ndarray]:
    return {
        "x": rmsnorm(st["alive"][f"mid_{b}"].astype(np.float64), blocks_alive[b].g2),
        "x_orig": rmsnorm(st["orig"][f"mid_{b}"].astype(np.float64), g2_orig[b]),
        "x_V": block_xv(b, readers),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    readers = json.loads((ROOT / "readers.json").read_text())["readers"]
    st = {"alive": np.load(AOUT / "stream.npz"), "orig": np.load(OOUT / "stream.npz")}
    blocks_alive = load_blocks()[0]
    _, _, wt = loader()
    g2_orig = {b: wt(f"model.layers.{b}.post_attention_layernorm.weight") for b in BLOCKS}
    names = {"x": "components model, full MLP input x", "x_orig": "original model (neurons), full MLP input x",
             "x_V": "components model, x_V (x projected onto the block's kept readers' span)"}  # fmt: skip
    cos = {k: {} for k in names}
    coh = {k: {} for k in names}
    stats: dict = {k: {} for k in names}
    for b in BLOCKS:
        for k, y in spaces(b, readers, st, blocks_alive, g2_orig).items():
            mu = residue_means(y)
            n = np.linalg.norm(mu, axis=1)
            cos[k][b] = (mu @ mu.T) / np.outer(n, n)
            coh[k][b] = cohen(y, mu)
            stats[k][b] = {
                "cos_mean": {c: float(cos[k][b][m].mean()) for c, m in CLASSES.items()},
                "cos_min": float(cos[k][b][~EYE].min()),
                "d_median": {c: float(np.median(coh[k][b][m])) for c, m in CLASSES.items()},
                "d_min": {c: float(coh[k][b][m].min()) for c, m in CLASSES.items()},
            }
            x = stats[k][b]
            print(b, k, "cos", {c: round(v, 4) for c, v in x["cos_mean"].items()}, "d median",
                  {c: round(v, 2) for c, v in x["d_median"].items()}, flush=True)  # fmt: skip
    for k, title in names.items():
        grid(cos[k], "cos(μ_s, μ_s')", False, args.out / f"mod20_geom_cos_{k}.png",
             lambda M: f"antipodal {M[CLASSES['antipodal']].mean():.3f}, lag 1 {M[CLASSES['lag 1']].mean():.3f}", title)  # fmt: skip
        grid(coh[k], "Cohen's d along μ_s − μ_s'", True, args.out / f"mod20_geom_cohen_{k}.png",
             lambda M: f"median: antipodal {np.median(M[CLASSES['antipodal']]):.1f}, lag 1 {np.median(M[CLASSES['lag 1']]):.1f}",
             title)  # fmt: skip
    (OUT / "geometry.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
