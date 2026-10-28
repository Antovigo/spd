"""Held-out decoding of every harmonic of a mod 20 from the MLP input at position 1, blocks 0..16.

For harmonic j = 1..10 (period 20/j) the targets are cos(2*pi*j*a/20) and sin(2*pi*j*a/20) (sin is identically
zero at j = 10 and dropped). The features are x = RMSNorm(mid_b) * g2_b, the post-norm input to block b's MLP
(4096 dims), for a = 1..400. Ridge regression without intercept (the network has no biases; nothing is centred).

Split: 5 outer folds stratified by a mod 20: each fold holds out 4 of the 20 a's of every residue (80 a's), so every
a is tested once on a fit that never saw it. The ridge penalty is chosen per (model, block, j, outer fold) by an inner
4-fold stratified CV on the training a's only, over a grid that reaches 1e-12 of the top squared singular value
(effectively unpenalised). r^2 = 1 - sum of held-out squared errors / sum of squared targets, pooled over the folds
and the two targets. "minnorm" is the same with the penalty fixed at the bottom of the grid.

Writes <run>/analysis/reader_writer/track_mod20/harmonics.json and notebook_mod20/fig/mod20_harmonics_{alive,orig}.png.

    python mod20_harmonics.py [--plot-only]   (--plot-only: redraw the figures from harmonics.json)
"""

import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import HUB, RUN, A, bf16_tensor
from track_build import EPS, LAST

OUT = RUN / "analysis/reader_writer/track_mod20"
STREAM = {
    "alive": RUN / "analysis/reader_writer/track_mod10/stream.npz",
    "orig": RUN / "analysis/reader_writer/track_orig/stream.npz",
}
TITLE = {"alive": "components (alive-only model)", "orig": "neurons (original model)"}
FIG = Path(__file__).parent / "notebook_mod20/fig"
J = range(1, 11)
GRID = 10.0 ** np.arange(-12, 2.5, 0.5)
S20 = A % 20


def folds(n_folds: int, idx: np.ndarray, seed: int) -> list[np.ndarray]:
    """Split idx into n_folds parts with an equal number of a's of every residue in each part."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_folds)]
    for s in range(20):
        mine = rng.permutation(idx[S20[idx] == s])
        for f, chunk in enumerate(np.array_split(mine, n_folds)):
            parts[f].extend(chunk.tolist())
    return [np.sort(np.asarray(p)) for p in parts]


def targets(j: int) -> np.ndarray:
    th = 2 * np.pi * j * A / 20
    Y = np.stack([np.cos(th), np.sin(th)], 1)
    return Y[:, np.abs(Y).max(0) > 1e-9]


def ridge_sse(X: np.ndarray, Ys: list[np.ndarray], tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """Held-out squared error for every target set in Ys and every penalty in GRID: (len(Ys), len(GRID))."""
    U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
    lam = GRID * s[0] ** 2
    P = X[te] @ Vt.T  # (n_te, r)
    out = np.zeros((len(Ys), len(GRID)))
    for k, Y in enumerate(Ys):
        UtY = U.T @ Y[tr]  # (r, t)
        for g, lm in enumerate(lam):
            pred = P @ (UtY * (s / (s**2 + lm))[:, None])
            out[k, g] = ((pred - Y[te]) ** 2).sum()
    return out


def decode(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r^2 (nested-CV penalty), r^2 (min-norm), chosen log10 penalty per fold; first two (10,), last (10, 5)."""
    Ys = [targets(j) for j in J]
    den = np.asarray([(Y**2).sum() for Y in Ys])
    outer = folds(5, np.arange(len(A)), seed=0)
    sse, sse0, pick = np.zeros(10), np.zeros(10), np.zeros((10, 5))
    for f, te in enumerate(outer):
        tr = np.setdiff1d(np.arange(len(A)), te)
        inner = sum(
            ridge_sse(X, Ys, np.setdiff1d(tr, ite), ite) for ite in folds(4, tr, seed=f + 1)
        )
        best = inner.argmin(1)
        held = ridge_sse(X, Ys, tr, te)
        sse += held[np.arange(10), best]
        sse0 += held[:, 0]
        pick[:, f] = np.log10(GRID[best])
    return 1 - sse / den, 1 - sse0 / den, pick


def label(j: int) -> str:
    p = Fraction(20, j)
    per = str(p.numerator) if p.denominator == 1 else f"{p.numerator}/{p.denominator}"
    tail = f"; mod 10: k={j // 2}" if j % 2 == 0 else ""
    return f"period {per}  (mod 20: j={j}{tail})"


def figure(R: np.ndarray, mdl: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(np.clip(R, 0, 1), aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="upper")
    for j in range(10):
        for b in range(LAST + 1):
            v = R[j, b]
            ax.text(b, j, f"{v:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=6.5,
                    color="w" if v > 0.6 else "#222")  # fmt: skip
    ax.set_yticks(range(10), [label(j) for j in J], fontsize=7.5)
    ax.set_xticks(range(LAST + 1), [str(b) for b in range(LAST + 1)], fontsize=8)
    ax.set_xlabel("block whose MLP input is decoded")
    ax.set_title(
        f"{TITLE[mdl]}: held-out r² of cos, sin(2π·j·a/20) from the MLP input at the token a",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, label="held-out r² (negative shown as 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import glob
    import sys

    if sys.argv[1:] == ["--plot-only"]:
        res = json.loads((OUT / "harmonics.json").read_text())
        for mdl in STREAM:
            figure(np.asarray(res[mdl]["r2"]), mdl, FIG / f"mod20_harmonics_{mdl}.png")
        return

    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    g2 = []
    for b in range(LAST + 1):
        name = f"model.layers.{b}.post_attention_layernorm.weight"
        g2.append(np.asarray(bf16_tensor(snap + wmap[name], name, None), np.float64))
    res = {}
    FIG.mkdir(parents=True, exist_ok=True)
    for mdl, path in STREAM.items():
        st = np.load(path)
        R, R0, pick = np.zeros((10, LAST + 1)), np.zeros((10, LAST + 1)), {}
        for b in range(LAST + 1):
            h = st[f"mid_{b}"].astype(np.float64)
            x = h / np.sqrt((h**2).mean(1, keepdims=True) + EPS) * g2[b]
            R[:, b], R0[:, b], pk = decode(x)
            pick[b] = pk.tolist()
            print(mdl, b, np.round(R[:, b], 2), flush=True)
        res[mdl] = {"r2": R.tolist(), "r2_minnorm": R0.tolist(), "log10_penalty": pick}
        figure(R, mdl, FIG / f"mod20_harmonics_{mdl}.png")
    res["periods"] = [label(j) for j in J]
    (OUT / "harmonics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
