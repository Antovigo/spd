"""Do representations move between layers without changing their arrangement?

    python -m param_decomp.arith_repr.vectors.drift compute   # -> OUT/drift.npz
    python -m param_decomp.arith_repr.vectors.drift figures   # -> OUT/figs/drift_*.png

A representation here is the set of class means of the raw residual stream at one position: group the
10,000 addition prompts by one quantity (a, b or a + b), average the stream over each group, and
subtract the grand mean. At stream point t this gives a matrix X_t (n classes x 4096). Three
comparisons between two points t, t':

* direct similarity  <X_t, X_t'>_F / (|X_t| |X_t'|): the class means are compared in the stream's own
  coordinates, with no realignment. 1 = same directions and same arrangement; 0 = orthogonal.
* linear CKA         |X_t^T X_t'|_F^2 / (|X_t^T X_t|_F |X_t'^T X_t'|_F): stored but not plotted; it is
  dominated by the largest directions (e.g. the magnitude lin(a) + lin(b)) and scores high even
  where the codes differ.
* CCA similarity     mean squared canonical correlation between the top-K principal coordinates of
  X_t and X_t' (K = 10): unchanged by any invertible linear map of those coordinates; chance level
  about K / (n - 1).

When direct similarity falls while CCA stays high, the representation has moved to other directions
of the stream but its arrangement, up to a linear map, is the same. Figures show only the points
where the quantity is present (`presence`)."""

import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from param_decomp.arith_repr.vectors.common import DATASET, OUT, point_names

FILE = OUT / "drift.npz"
REPS = {  # name: (token position, grouping quantity)
    "a_at_a": (1, "a"),
    "b_at_b": (3, "b"),
    "a_at_eq": (4, "a"),
    "b_at_eq": (4, "b"),
    "sum_at_eq": (4, "sum"),
}
K_CCA = 10


def groups(q: str) -> tuple[np.ndarray, int]:
    i = np.arange(10000)
    a, b = i // 100 + 1, i % 100 + 1
    v = {"a": a - 1, "b": b - 1, "sum": a + b - 2}[q]
    return v, int(v.max()) + 1


def class_means(x: np.ndarray, q: str) -> np.ndarray:
    v, n = groups(q)
    cnt = np.bincount(v, minlength=n).astype(np.float64)
    m = np.zeros((n, x.shape[1]))
    np.add.at(m, v, x)
    m /= cnt[:, None]
    return m - m.mean(0)


def compute() -> None:
    resid = np.load(DATASET / "original/resid.npy", mmap_mode="r")  # (65, 20000, 5, 4096)
    n_pt = resid.shape[0]
    X = {k: [] for k in REPS}
    for t in range(n_pt):
        for pos in sorted({p for p, _ in REPS.values()}):
            x = np.asarray(resid[t, :10000, pos], np.float64)
            for k, (p, q) in REPS.items():
                if p == pos:
                    X[k].append(class_means(x, q).astype(np.float32))
        print(t, flush=True)
    out = {}
    for k in REPS:
        Xs = np.stack(X[k])  # (T, n, d)
        flat = Xs.reshape(n_pt, -1).astype(np.float64)
        nrm = np.linalg.norm(flat, axis=1)
        direct = flat @ flat.T / np.outer(nrm, nrm)
        G = np.einsum("tnd,tmd->tnm", Xs, Xs, dtype=np.float64)  # Gram matrices (n x n)
        Gn = G / np.linalg.norm(G, axis=(1, 2), keepdims=True)
        cka = np.einsum("tnm,snm->ts", Gn, Gn)
        # CCA on the top-K principal coordinates (left singular vectors of each X_t)
        bases = []
        for t in range(n_pt):
            u, _, _ = np.linalg.svd(Xs[t].astype(np.float64), full_matrices=False)
            bases.append(u[:, :K_CCA])
        Q = np.stack(bases)
        cca = np.zeros((n_pt, n_pt))
        for t in range(n_pt):
            s = np.linalg.svd(np.einsum("nk,snj->skj", Q[t], Q), compute_uv=False)  # (T, K)
            cca[t] = (s**2).mean(1)
        out[f"{k}_direct"], out[f"{k}_cka"], out[f"{k}_cca"] = direct, cka, cca
        out[f"{k}_var"] = (Xs.astype(np.float64) ** 2).sum((1, 2)) / Xs.shape[1]
        print(k, "done", flush=True)
    np.savez(FILE, **out)


# ---------------------------------------------------------------- figures

INK, INK2, GRAY = "#1d1c1a", "#52514e", "#b9b8b3"
TITLES = {
    "a_at_a": "a, at token a",
    "b_at_b": "b, at token b",
    "a_at_eq": "a, at '='",
    "b_at_eq": "b, at '='",
    "sum_at_eq": "a + b, at '=' (2..200)",
}
ROWS = ("a_at_a", "b_at_eq", "sum_at_eq")
REF = {
    "a_at_a": "L2.mlp",
    "b_at_b": "L2.mlp",
    "a_at_eq": "L17.attn",
    "b_at_eq": "L15.attn",
    "sum_at_eq": "L19.mlp",
}


def _ticks(ax: Any, names: list[str], axis: str = "both") -> None:
    idx = [
        i
        for i, n in enumerate(names)
        if n == "embed" or (n.endswith(".mlp") and int(n[1:].split(".")[0]) % 4 == 3)
    ]
    lab = [names[i].replace(".mlp", "") for i in idx]
    if axis in ("x", "both"):
        ax.set_xticks(idx)
        ax.set_xticklabels(lab, fontsize=6.5, rotation=90)
    if axis in ("y", "both"):
        ax.set_yticks(idx)
        ax.set_yticklabels(lab, fontsize=6.5)


def save(fig: Any, name: str) -> None:
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"figs/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


PRESENT = 0.05  # a representation is present where its codes carry >= 5 % of the stream's variance
LINE_IX = {"a": 0, "b": 1, "sum": 2}


def presence(k: str) -> np.ndarray:
    """(65,) bool: the quantity's codes (all harmonics of its line, plus its linear part for a / b)
    carry at least PRESENT of the stream's variance at that point and position (storage.npz)."""
    st = np.load(OUT / "storage.npz")
    pos, q = REPS[k]
    li = LINE_IX[q]
    e = st["energy"][:, pos - 1, 0, li].sum(-1)
    if li < 2:
        e = e + st["lin"][:, pos - 1, 0, li]
    return np.nan_to_num(e) >= PRESENT


def masked(sim: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = sim.copy()
    out[~p, :] = np.nan
    out[:, ~p] = np.nan
    return out


def shade_absent(ax: Any, p: np.ndarray) -> None:
    t = 0
    while t < len(p):
        if not p[t]:
            u = t
            while u + 1 < len(p) and not p[u + 1]:
                u += 1
            ax.axvspan(t - 0.5, u + 0.5, color="#eceee9", zorder=0)
            if u - t >= 6:
                ax.text((t + u) / 2, 0.5, "not present", ha="center", va="center", fontsize=7.5,
                        color=INK2, transform=ax.get_xaxis_transform())  # fmt: skip
            t = u + 1
        else:
            t += 1


def heatmaps(z: dict[str, np.ndarray], metric: str, title: str, name: str) -> None:
    names = point_names()
    fig, axes = plt.subplots(1, len(ROWS), figsize=(5.2 * len(ROWS), 5))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#eceee9")
    im: Any = None
    for ax, k in zip(axes, ROWS, strict=True):
        im = ax.imshow(
            masked(z[f"{k}_{metric}"], presence(k)), vmin=0, vmax=1, cmap=cmap, origin="lower"
        )
        _ticks(ax, names)
        ax.set_title(TITLES[k], fontsize=9.5, color=INK)
        ax.set_xlabel("stream point", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label=metric)
    fig.suptitle(title, fontsize=10, color=INK2)
    save(fig, name)


def lines(z: dict[str, np.ndarray]) -> None:
    """Similarity to the previous point and to a fixed reference point, and the sizes."""
    names = point_names()
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, len(ROWS), figsize=(5.2 * len(ROWS), 9.4), sharex=True)
    styles = {"direct": ("#d6336c", "same directions (no realignment)"),
              "cca": ("#2a78d6", "same up to any linear map (CCA, top 10)")}  # fmt: skip
    for c, k in enumerate(ROWS):
        ref = names.index(REF[k])
        p = presence(k)
        pv = np.where(p, 1.0, np.nan)
        for r, (lab, get) in enumerate((
            ("vs the previous stream point",
             lambda sim, pv=pv: np.r_[np.nan, np.diag(sim, 1)] * pv * np.r_[np.nan, pv[:-1]]),
            (f"vs {REF[k]}", lambda sim, ref=ref, pv=pv: sim[ref] * pv),
        )):  # fmt: skip
            ax = axes[r, c]
            shade_absent(ax, p)
            for m, (col, desc) in styles.items():
                ax.plot(x, get(z[f"{k}_{m}"]), color=col, lw=1.4, label=desc)
            ax.axhline(10 / 99, color="#2a78d6", lw=0.7, ls=":", alpha=0.7)
            if r == 1:
                ax.axvline(ref, color=GRAY, lw=0.8, ls=":")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{TITLES[k]}: {lab}", fontsize=9, color=INK)
            ax.grid(alpha=0.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            _ticks(ax, names, "x")
        # moved away, or diluted? The reference pattern's own amount in X_t, and the size of X_t
        ax = axes[2, c]
        shade_absent(ax, p)
        size = np.sqrt(z[f"{k}_var"] / z[f"{k}_var"][ref])
        kept = z[f"{k}_direct"][ref] * size  # <X_ref, X_t> / |X_ref|^2
        ax.plot(
            x, size * pv, color=INK, lw=1.4, label="size of the representation, |X_t| / |X_ref|"
        )
        ax.plot(
            x,
            kept * pv,
            color="#d6336c",
            lw=1.4,
            ls="--",
            label="amount of the reference pattern still in X_t",
        )
        ax.plot(x, np.sqrt(np.maximum(size**2 - kept**2, 0)) * pv, color="#8a8f8c", lw=1.2, ls=":",
                label="size of the part orthogonal to the reference pattern")  # fmt: skip
        ax.axvline(ref, color=GRAY, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 30)
        ax.set_title(f"{TITLES[k]}: sizes relative to {REF[k]}", fontsize=9, color=INK)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        _ticks(ax, names, "x")
    axes[0, 0].legend(fontsize=7.5, loc="lower left")
    axes[2, 0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Class means of the residual stream compared across stream points (addition prompts). Grey: the "
                 "quantity's codes carry < 5 % of the stream's variance there. Dotted blue: chance level of CCA "
                 "(10/99)", fontsize=10, color=INK2)  # fmt: skip
    fig.tight_layout()
    save(fig, "drift_lines")


def figures() -> None:
    z = dict(np.load(FILE))
    heatmaps(z, "direct", "Movement: similarity of the class means in the stream's own coordinates "
             "(no realignment allowed), between every pair of stream points where the quantity is present",
             "drift_direct")  # fmt: skip
    heatmaps(z, "cca", "Arrangement: CCA similarity of the class means (unchanged by any invertible linear map "
             "of the top 10 principal coordinates), where the quantity is present", "drift_cca")  # fmt: skip
    lines(z)
    names = point_names()
    for k in REPS:
        d, q = z[f"{k}_direct"], z[f"{k}_cca"]
        ref = names.index(REF[k])
        p = presence(k)
        first = names[int(np.argmax(p))]
        print(k, "ref", REF[k], "present from", first, "points present", int(p.sum()))
        for t in ("L5.mlp", "L10.mlp", "L15.mlp", "L19.mlp", "L23.mlp", "L27.mlp", "L31.mlp"):
            i = names.index(t)
            size = np.sqrt(z[f"{k}_var"][i] / z[f"{k}_var"][ref])
            print(f"   {t:9s} present {p[i]} step direct {d[i - 1, i]:.3f} cca {q[i - 1, i]:.3f} | vs ref direct "
                  f"{d[ref, i]:.3f} cca {q[ref, i]:.3f} | size {size:.2f} kept {d[ref, i] * size:.2f}")  # fmt: skip
        mlp = [names.index(f"L{li}.mlp") for li in range(3, 31) if p[names.index(f"L{li}.mlp")]]
        att = [names.index(f"L{li}.attn") for li in range(3, 31) if p[names.index(f"L{li}.attn")]]
        for nm, ids in (("attn", att), ("mlp", mlp)):
            if ids:
                print(f"   median step L3-L30 {nm}: direct {np.median([d[i - 1, i] for i in ids]):.3f} "
                      f"cca {np.median([q[i - 1, i] for i in ids]):.3f}")  # fmt: skip


if __name__ == "__main__":
    {"compute": compute, "figures": figures}[sys.argv[1]]()
