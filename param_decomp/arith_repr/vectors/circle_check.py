"""Why is the (a + b) mod 50 circle drawn by the seven L18 neurons noisy when all their terms are
kept? Tests four explanations on the addition prompts.

    python -m param_decomp.arith_repr.vectors.circle_check   # -> OUT/figs/k2_circle_check.png + stdout

a and b are the operands (integers 1..100); s = (a + b) mod 50 is the result residue. A view is a set
of 2-D points, one per prompt: some vector (a write, or the residual stream) projected on a plane.
For each view:
* share_s: share of the points' variance explained by the class means of s (all periods of s),
* circle: share explained by the period-50 wave of a + b alone (harmonic k = 2 of the a + b line),
* share_a / share_b: share explained by the class means of a / of b (the operands' own content),
* order: |mean over s of exp(i (angle of class mean of s - 2 pi s / 50))|, 1 = class means in
  perfect circular order (up to a rotation), with the orientation that maximises it,
* nearest: share of prompts whose point is closer to their own class mean of s than to any other."""

from typing import Any

import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, number_token_ids
from param_decomp.arith_repr.vectors.case_k2 import FILE as K2
from param_decomp.arith_repr.vectors.case_k2 import plane
from param_decomp.arith_repr.vectors.common import DATASET, OUT, line_dft, point_names

L, T = 18, 50
i_ = np.arange(10000)
A_, B_ = i_ // 100 + 1, i_ % 100 + 1
S_ = (A_ + B_) % T


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def class_means(y: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, y.shape[1]))
    np.add.at(m, g, y)
    return m / np.bincount(g, minlength=n)[:, None]


def metrics(y: np.ndarray) -> dict[str, float]:
    y = y - y.mean(0)
    tot = (y**2).sum()
    out = {}
    for nm, g, n in (("s", S_, T), ("a", A_ - 1, 100), ("b", B_ - 1, 100)):
        m = class_means(y, g, n)
        out[f"share_{nm}"] = float((m[g] ** 2).sum() / tot)
    F = line_dft(y.reshape(100, 100, 2))[2, 1]  # a + b line, k = 2
    out["circle"] = float(2 * (np.abs(F) ** 2).sum() * 10000 / tot)
    m = class_means(y, S_, T)
    z = m[:, 0] + 1j * m[:, 1]
    q = np.exp(2j * np.pi * np.arange(T) / T)
    out["order"] = float(max(abs((z / abs(z) * np.conj(q)).mean()), abs((z / abs(z) * q).mean())))
    d = ((y[:, None, :] - m[None]) ** 2).sum(-1)
    out["nearest"] = float((d.argmin(1) == S_).mean())
    return out


def additive_removed(y: np.ndarray) -> np.ndarray:
    """y minus its class means by a and by b (both centred): what is left after the operands' own
    content is taken out (exact two-way decomposition on the full grid)."""
    y = y - y.mean(0)
    return y - class_means(y, A_ - 1, 100)[A_ - 1] - class_means(y, B_ - 1, 100)[B_ - 1]


def main() -> None:
    assert ml_dtypes.bfloat16
    z = dict(np.load(K2))
    top = z["top"]
    Wd = Weights().get(f"model.layers.{L}.mlp.down_proj.weight").astype(np.float32)
    g = np.asarray(
        np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")[L, :10000, 4], np.float32
    )
    u = np.asarray(
        np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")[L, :10000, 4], np.float32
    )
    act = silu(g) * u
    del g, u
    resid = np.load(DATASET / "original/resid.npy", mmap_mode="r")
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    names = point_names()

    def stream_plane(t: int) -> np.ndarray:
        F = fre[t, 3, 0, 2, 1].astype(np.float32) + 1j * fim[t, 3, 0, 2, 1].astype(np.float32)
        return plane(F)

    A7 = line_dft(act[:, top].reshape(100, 100, -1))[2, 1]  # (7,) a + b coefficients of the values
    P7 = plane(Wd[:, top] @ A7)  # the applet's plane: the seven neurons' a + b write
    Aall = np.empty(Wd.shape[1], np.complex64)
    for c in range(0, Wd.shape[1], 2048):
        Aall[c : c + 2048] = line_dft(act[:, c : c + 2048].reshape(100, 100, -1))[2, 1]
    Pmlp = plane(Wd @ Aall)  # the whole MLP's a + b write
    t_in, t_out = names.index(f"L{L}.attn"), names.index(f"L{L}.mlp")
    Ps = stream_plane(t_out)  # the stream's own (a + b) mod 50 plane after the L18 MLP
    w7 = act[:, top] @ Wd[:, top].T  # (10000, 4096) the seven neurons' write, all terms
    wall = act @ Wd.T  # the whole MLP's write
    x_in = np.asarray(resid[t_in, :10000, 4], np.float32)
    x_out = np.asarray(resid[t_out, :10000, 4], np.float32)
    print("check: x_out - x_in vs MLP write, relative error",
          float(np.linalg.norm((x_out - x_in) - wall) / np.linalg.norm(wall)))  # fmt: skip

    views = {
        "1 seven neurons, all terms, applet plane": w7 @ P7.T,
        "2 seven neurons, operands' own content removed": additive_removed(w7 @ P7.T),
        "3 seven neurons, all terms, stream's a+b plane": w7 @ Ps.T,
        "4 whole MLP write, its own a+b plane": wall @ Pmlp.T,
        "5 whole MLP write, stream's a+b plane": wall @ Ps.T,
        "6 stream before the MLP, stream's a+b plane": x_in @ Ps.T,
        "7 stream before + seven neurons": (x_in + w7) @ Ps.T,
        "8 stream after the MLP (L18.mlp)": x_out @ Ps.T,
        "9 stream after, operands' own content removed": additive_removed(x_out @ Ps.T),
    }
    for t in ("L19.mlp", "L20.mlp", "L22.mlp", "L25.mlp", "L28.mlp", "L31.mlp"):
        ti = names.index(t)
        x = np.asarray(resid[ti, :10000, 4], np.float32)
        views[f"stream at {t}, its own a+b plane"] = x @ stream_plane(ti).T
    res = {}
    for k, y in views.items():
        res[k] = metrics(y)
        r = res[k]
        print(f"{k:52s} share_s {r['share_s']:.3f} circle {r['circle']:.3f} share_a {r['share_a']:.3f} "
              f"share_b {r['share_b']:.3f} order {r['order']:.3f} nearest {r['nearest']:.3f}")  # fmt: skip
    # null hypothesis: are the model's errors on addition tied to a bad position on the circle?
    ids, _ = number_token_ids(200)
    tok2num = {int(t): n for n, t in enumerate(ids)}
    top1 = np.load(DATASET / "original/last_top_ids.npy", mmap_mode="r")[:10000, 0]
    pred = np.array([tok2num.get(int(t), -1) for t in top1])
    wrong = pred != A_ + B_
    print(f"model errors on addition: {wrong.sum()} of 10000")
    if wrong.any():
        diff = pred[wrong] - (A_ + B_)[wrong]
        vals, cnt = np.unique(diff, return_counts=True)
        order = np.argsort(-cnt)[:10]
        print("   pred - (a+b): " + ", ".join(f"{vals[j]}: {cnt[j]}" for j in order))
        print("   errors with pred == -1 (not a number 0..200):", int((pred[wrong] == -1).sum()))
        for k in (
            "8 stream after the MLP (L18.mlp)",
            "stream at L25.mlp, its own a+b plane",
            "stream at L31.mlp, its own a+b plane",
        ):
            y = views[k] - views[k].mean(0)
            m = class_means(y, S_, T)
            dist = np.sqrt(((y - m[S_]) ** 2).sum(1))
            own = ((y[:, None, :] - m[None]) ** 2).sum(-1).argmin(1) == S_
            print(f"   {k}: distance to own class mean, wrong {dist[wrong].mean():.3f} vs right "
                  f"{dist[~wrong].mean():.3f}; nearest-mean correct: wrong {own[wrong].mean():.2f} vs right {own[~wrong].mean():.2f}")  # fmt: skip
    arrays: dict[str, Any] = {f"view{j}": v for j, v in enumerate(views.values())}
    arrays.update(names=np.array(list(views)), wrong=wrong, pred=pred)
    np.savez(OUT / "circle_check.npz", **arrays)
    figure(views, res)


def figure(views: dict[str, np.ndarray], res: dict[str, dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(views)
    cols = 5
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.7 * rows))
    cmap = matplotlib.colormaps["twilight_shifted"]
    for ax, k in zip(axes.ravel(), keys, strict=False):
        y = views[k] - views[k].mean(0)
        sd = np.sqrt((y**2).sum(1).mean())
        y = y / sd
        ax.scatter(y[:, 0], y[:, 1], s=1.2, c=cmap(S_ / T), alpha=0.35, lw=0)
        m = class_means(y, S_, T)
        ax.plot(*np.r_[m, m[:1]].T, color="#1d1c1a", lw=0.7)
        ax.scatter(*m.T, s=9, c=cmap(np.arange(T) / T), edgecolor="#1d1c1a", lw=0.4, zorder=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        r = res[k]
        ax.set_title(f"{k}\ns {r['share_s']:.2f} | circle {r['circle']:.2f} | a {r['share_a']:.2f} | b {r['share_b']:.2f}\n"
                     f"order {r['order']:.2f} | nearest {r['nearest']:.2f}", fontsize=7)  # fmt: skip
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.suptitle("Addition prompts, coloured by (a + b) mod 50; black: class means of (a + b) mod 50. "
                 "Shares are of each view's variance.", fontsize=10)  # fmt: skip
    fig.tight_layout()
    (OUT / "figs").mkdir(exist_ok=True)
    fig.savefig(OUT / "figs/k2_circle_check.png", dpi=150, bbox_inches="tight")
    print("saved k2_circle_check")


if __name__ == "__main__":
    main()
