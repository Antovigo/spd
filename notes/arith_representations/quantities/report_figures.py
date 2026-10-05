"""Figures for the walkthrough in README.md: each iteration's fits, the discovery of divisibility
by 3, and a gallery of the quantities found with the readers that use them most.

    python report_figures.py   # refits every hypothesis set, writes figures/ and walkthrough.json
"""

import json
import re
from pathlib import Path

import hypotheses
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hyp_fit import ReaderFit, diagnostics, fit_site
from iterate import site_rho
from qsite import load_site

HERE = Path(__file__).parent
FIG = HERE / "figures"
A = np.arange(1, 101)
SETS = {"attn": ["s1_v1", "s1_v2", "s1_v3"], "mlp": ["s2_v1", "s2_v2"]}
SITE_NAME = {"attn": "site 1 (L0 attention input)", "mlp": "site 2 (L0 MLP input)"}


def run_all() -> tuple[dict, dict, dict]:
    data, fits, stats = {}, {}, {}
    for point, sets in SETS.items():
        site = load_site(0, point)
        rho = site_rho(0, point)
        X = site.X * (rho / rho.mean())[:, None]
        data[point] = (site.names, X)
        for h in sets:
            f = fit_site(X, getattr(hypotheses, h)())
            fits[h] = f
            R = X - np.stack([r.fitted for r in f], 1)
            d = diagnostics(R, site.names)
            stats[h] = {
                "explained_var": float(1 - (R**2).sum() / ((X - X.mean(0)) ** 2).sum()),
                "median_r2": float(np.median([r.r2 for r in f])),
                "top_sv": d["top_sv"][0], "null_sv": d["null_top_sv_95"],
                "flagged": len(d["patterned_readers"]), "n": len(site.names),
            }  # fmt: skip
            print(h, stats[h], flush=True)
    return data, fits, stats


def _panel(ax, x, fit, part=None, title="", part_label="this quantity"):  # noqa: ANN001, ANN202
    ax.plot(A, x, "k-", lw=0.9, label="raw read")
    ax.plot(A, fit, "C1--", lw=0.9, label="full fit")
    if part is not None:
        ax.plot(A, part, "C0-", lw=0.9, alpha=0.8, label=part_label)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(alpha=0.3, lw=0.4)


def fig_iterations(data: dict, fits: dict) -> None:
    names, X = data["attn"]
    pick = ["L0.k.c0", "L0.q.c21", "L0.v.c1", "L0.v.c28", "L0.k.c25"]
    J = [names.index(p) for p in pick]
    fig, axes = plt.subplots(4, len(J), figsize=(3.2 * len(J), 7.0))
    for col, j in enumerate(J):
        for row, h in enumerate(("s1_v1", "s1_v3")):
            f: ReaderFit = fits[h][j]
            _panel(
                axes[2 * row, col], X[:, j], f.fitted, title=f"{names[j]} | {h} | R2 = {f.r2:.3f}"
            )
            ax = axes[2 * row + 1, col]
            ax.plot(A, X[:, j] - f.fitted, "k-", lw=0.8)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_title(f"residual ({h})", fontsize=7)
            ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Site 1 (L0 attention input, token a): iteration 1 (log a + a for magnitude) vs final "
                 "(spline magnitude + token identity menu)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "walk_site1_iterations.png", dpi=85)
    plt.close(fig)


def fig_div3(data: dict, fits: dict) -> None:
    names, X = data["mlp"]
    j = names.index("L0.gate.c109")
    x = X[:, j]
    f1, f2 = fits["s2_v1"][j], fits["s2_v2"][j]
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.8))
    _panel(axes[0], x, f1.fitted, title=f"L0.gate.c109 | s2_v1 | R2 = {f1.r2:.2f}")
    r = x - f1.fitted
    p = np.abs(np.fft.rfft(r - r.mean())[1:]) ** 2
    axes[1].bar(np.arange(1, 51), p[:50] / p.sum(), color="0.3")
    axes[1].set_title("s2_v1 residual: power share per frequency k (period 100/k)", fontsize=7)
    axes[1].annotate("k = 33 (period ~3)", (33, p[32] / p.sum()), fontsize=7, color="C3")
    axes[1].tick_params(labelsize=6)
    part = f2.contrib.get("residue class (a mod m = r)")
    _panel(
        axes[2],
        x,
        f2.fitted,
        part,
        title=f"s2_v2 | R2 = {f2.r2:.2f} | " + ", ".join(i.split(": ")[1] for i in f2.items),
    )
    axes[3].stem(A, x, linefmt="0.6", markerfmt="k.", basefmt=" ")
    m3 = A % 3 == 0
    axes[3].plot(A[m3], x[m3], "C3o", ms=3, label="a mod 3 = 0")
    axes[3].set_title("raw read: multiples of 3 in red (amplitudes vary)", fontsize=7)
    axes[3].legend(fontsize=6)
    axes[3].tick_params(labelsize=6)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "walk_site2_div3.png", dpi=85)
    plt.close(fig)


GALLERY = {
    "attn": [
        ("magnitude curve (spline in log a)", ["magnitude (spline in log a)"], None),
        ("digit count ([a <= 9], [a = 100])", ["digit count ([a<=9], [a=100])"], None),
        ("round numbers (a mod 10 = 0, a mod 5 = 0)", ["round (a mod 10 = 0)", "half-round (a mod 5 = 0)"], None),
        ("single tokens", ["token identity (a = v)"], None),
    ],
    "mlp": [
        ("magnitude curve (spline in log a)", ["magnitude (spline in log a)"], None),
        ("digit count ([a <= 9], [a = 100])", ["digit count ([a<=9], [a=100])"], None),
        ("units digit (a mod 10 = r)", ["residue class (a mod m = r)"], r"a mod 10 = [1-46-9]$"),
        ("multiples of 5 / 10", ["residue class (a mod m = r)"], r"a mod (5|10) = (0|5)$"),
        ("divisibility by 3", ["residue class (a mod m = r)"], r"a mod (3|6|9|12) = 0$"),
        ("place code: bumps and intervals", ["bump in a (place code)", "interval of a (place code)"], None),
        ("soft thresholds", ["soft threshold in a"], None),
        ("single-value detectors", ["token identity (a = v)"], "ONLY_TOKENS"),
    ],
}  # fmt: skip


def fig_gallery(point: str, h: str, data: dict, fits: dict) -> list[dict]:
    names, X = data[point]
    entries = GALLERY[point]
    fig, axes = plt.subplots(len(entries), 4, figsize=(13, 2.3 * len(entries)), squeeze=False)
    rows = []
    for e, (title, qs, pat) in enumerate(entries):
        cand = []
        for j, f in enumerate(fits[h]):
            if not any(q in f.quantities for q in qs):
                continue
            items = [i.split(": ", 1)[1] for i in f.items if i.split(": ")[0] in qs]
            if pat == "ONLY_TOKENS":
                if set(f.quantities) != {"token identity (a = v)"}:
                    continue
            elif pat is not None and not any(re.search(pat, i) for i in items):
                continue
            part = sum(f.contrib.get(q, 0) for q in qs)
            x = X[:, j]
            share = float(np.var(part) / np.var(x))
            cand.append((share, j, part, items))
        # examples: readers where this quantity's part of the fit is not cancelled by another
        # quantity (share <= 1.2; overlapping families such as spline / bumps / thresholds can fit
        # large opposite terms), largest share first; cancelled fits are only used if none is clean
        clean = [t for t in cand if t[0] <= 1.2]
        cand = sorted(clean, key=lambda t: -t[0]) + sorted(
            [t for t in cand if t[0] > 1.2], key=lambda t: t[0]
        )
        rows.append({"feature": title, "n_readers": len(cand),
                     "examples": [(names[j], round(s, 3), round(fits[h][j].r2, 3), it[:6]) for s, j, _, it in cand[:4]]})  # fmt: skip
        for c in range(4):
            ax = axes[e, c]
            if c >= len(cand):
                ax.axis("off")
                continue
            s, j, part, items = cand[c]
            f = fits[h][j]
            lab = ", ".join(items[:4]) + (" ..." if len(items) > 4 else "")
            _panel(ax, X[:, j], f.fitted, part + X[:, j].mean() - np.mean(part),
                   title=f"{names[j]} | R2 {f.r2:.3f} | share {s:.2f}\n{lab}")  # fmt: skip
        axes[e, 0].set_ylabel(f"{title}\n({len(cand)} readers)", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{SITE_NAME[point]}, hypothesis set {h}: each row is one quantity, with the 4 readers it "
                 "explains most (blue: that quantity's part of the fit, shifted to the reader's mean)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / f"walk_gallery_{point}.png", dpi=80)
    plt.close(fig)
    return rows


def fig_r2(fits: dict, stats: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, (point, sets) in zip(axes, SETS.items(), strict=True):
        bins = np.linspace(0, 1, 21)
        for h in sets:
            ax.hist([f.r2 for f in fits[h]], bins=bins, histtype="step", lw=1.2,
                    label=f"{h}: explained {stats[h]['explained_var']:.3f}")  # fmt: skip
        ax.set_title(f"{SITE_NAME[point]}: per-reader R2", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "walk_r2.png", dpi=85)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    data, fits, stats = run_all()
    fig_iterations(data, fits)
    fig_div3(data, fits)
    gal = {p: fig_gallery(p, SETS[p][-1], data, fits) for p in SETS}
    fig_r2(fits, stats)
    (HERE / "walkthrough.json").write_text(json.dumps({"stats": stats, "gallery": gal}, indent=1))
    print(json.dumps(gal, indent=1))


if __name__ == "__main__":
    main()
