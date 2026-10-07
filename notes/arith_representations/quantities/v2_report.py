"""V2 report: tables, figures and JSON for the README's V2 section.

python v2_report.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v2 import (
    A,
    Q,
    circle_geometry,
    classes,
    fourier,
    reader_dirs,
    residual_svd,
    span_basis,
    spline,
    stagewise,  # fmt: skip
    stream,
    variance_table,
)

HERE = Path(__file__).parent
FIG = HERE / "figures"
TENS = np.where((A >= 10) & (A <= 99), A // 10, 0)


def H_main() -> list[Q]:
    return [
        Q("magnitude curve (1 dir)", "curve", spline(), 1),
        Q("smooth in log a, 2 more dirs", "curve", spline(), 2),
        Q("one digit [a <= 9]", "fixed", (A <= 9).astype(float)[:, None]),
        Q("a mod 10: circle (period 10)", "fixed", fourier(10, [1])),
        Q("a mod 10: harmonics 2-5", "fixed", fourier(10, [2, 3, 4, 5])),
        Q("a mod 3", "fixed", classes(A % 3)),
        Q("tens digit (10..99)", "fixed", classes(TENS)),
        Q("place code: smooth in a (25 dims)", "fixed", spline(25, log=False)),
        Q("one-hot remainder", "fixed", np.eye(100)),
    ]


def H_place_first() -> list[Q]:
    """Same quantities, the fine place code before the digit quantities (order check)."""
    H = H_main()
    return H[:2] + [H[7]] + H[2:7] + [H[8]]


def sites() -> dict:
    out = {"embedding": (stream(0), None, None)}
    for key, (blk, pt, lpos) in {"site1": (0, "attn", 0), "site2": (0, "mlp", 1)}.items():
        Vt, names, _ = reader_dirs(blk, pt)
        Qb = span_basis(Vt)
        out[key] = (stream(lpos) @ Qb, Qb.T @ Vt.T, names)
    return out


LABEL = {"embedding": "embedding (l = 0, all 4096 dims)", "site1": "site 1: L0 attention input (8 readers)",
         "site2": "site 2: L0 MLP input (115 readers)"}  # fmt: skip


def units_geometry(Z: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, list[float], dict]:
    """After the smooth stages: class means of the units digit in the metric M, their 2-D MDS,
    the variance of each period-10 harmonic j = 1..5, and the j = 1 circle's geometry."""
    mu, parts, objs = stagewise(Z, H_main()[:2])
    R = (Z - mu - sum(parts)) @ M
    means = np.stack([R[u == A % 10].mean(0) for u in range(10)])
    C = means - means.mean(0)
    Uu, s, _ = np.linalg.svd(C, full_matrices=False)
    emb2 = Uu[:, :2] * s[:2]
    tot = (R**2).sum()
    per_j = []
    for j in range(1, 6):
        F = fourier(10, [j])
        F = F - F.mean(0)
        G, *_ = np.linalg.lstsq(F, R, rcond=None)
        per_j.append(float(((F @ G) ** 2).sum() / tot))
    F = fourier(10, [1])
    G, *_ = np.linalg.lstsq(F - F.mean(0), R, rcond=None)
    return emb2, per_j, circle_geometry(G, (0, 1)) | {"class_mean_sv": (s / s[0]).round(3).tolist()}


def main() -> None:
    S = sites()
    res: dict = {"main": {}, "place_first": {}, "units": {}, "residual": {}}
    for key, (Z, W, _) in S.items():
        res["main"][key] = variance_table(Z, W, H_main())
        res["place_first"][key] = variance_table(Z, W, H_place_first())
        print(key, "done", flush=True)

    # variance figure (main order)
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
    for ax, key in zip(axes, S, strict=True):
        rows = res["main"][key][:-1]
        y = np.arange(len(rows))
        ax.barh(y, [r["incr"] for r in rows], color="C0", label="explained (increment)")
        ax.barh(
            y,
            [r["chance"] for r in rows],
            color="none",
            edgecolor="k",
            hatch="//",
            label="chance level",
        )
        ax.set_yticks(y, [f"{r['quantity']} (d={r['dims']})" for r in rows], fontsize=7)
        ax.invert_yaxis()
        rem = res["main"][key][-1]["incr"]
        ax.set_title(f"{LABEL[key]}\none-hot remainder {rem:.2f}; CV R2 of the structured part "
                     f"{res['main'][key][-2]['cv_cum']:.2f}", fontsize=8)  # fmt: skip
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("V2: variance explained by each quantity (stagewise, in order; reader metric at sites 1-2, "
                 "stream metric for the embedding) against its chance level", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "v2_variance.png", dpi=90)
    plt.close(fig)

    # units digit geometry
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for c, key in enumerate(S):
        Z, W, _ = S[key]
        M = W if W is not None else np.eye(Z.shape[1])
        e2, per_j, geo = units_geometry(Z, M)
        res["units"][key] = {"per_harmonic_var": per_j, **geo}
        ax = axes[0, c]
        ax.scatter(e2[:, 0], e2[:, 1], c=np.arange(10), cmap="tab10", s=60)
        for u in range(10):
            ax.annotate(str(u), e2[u], fontsize=9, xytext=(4, 4), textcoords="offset points")
        ax.set_title(f"{LABEL[key]}\nunits-digit class means, top-2 PCs (sv ratios "
                     f"{', '.join(f'{v:.2f}' for v in geo['class_mean_sv'][:4])})", fontsize=8)  # fmt: skip
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=6)
        ax = axes[1, c]
        ax.bar(range(1, 6), per_j, color="C2")
        ax.set_xticks(range(1, 6), [f"j={j}\n(period {10 / j:g})" for j in range(1, 6)], fontsize=7)
        ax.set_title(f"variance per period-10 harmonic j (j=5: 1 dim, others 2)\nj=1 circle: |D_sin|/|D_cos| "
                     f"{geo['norm_ratio']:.2f}, angle {geo['angle_deg']:.0f} deg", fontsize=8)  # fmt: skip
        ax.tick_params(labelsize=6)
    fig.suptitle(
        "V2: is a mod 10 a circle or a 10-class simplex? (after removing the smooth magnitude dims)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIG / "v2_units_geometry.png", dpi=90)
    plt.close(fig)

    # smooth curves (the learned magnitude features) and residual singular functions
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for c, key in enumerate(S):
        Z, W, _ = S[key]
        mu, parts, _ = stagewise(Z, H_main()[:2])
        sm = sum(parts)
        U, s, _ = np.linalg.svd(sm, full_matrices=False)
        ax = axes[0, c]
        for i in range(3):
            ax.plot(A, U[:, i] * s[i], lw=1.1, label=f"smooth feature {i + 1}")
        ax.axvline(9.5, color="0.7", lw=0.6)
        ax.set_title(f"{LABEL[key]}: learned smooth features (singular functions x sv)", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        mu, parts, _ = stagewise(Z, H_main()[:-1])
        E = Z - mu - sum(parts)
        if W is not None:
            E = E @ W
        Ue, se, null = residual_svd(E)
        res["residual"][key] = {"top_sv": se[:5].round(4).tolist(), "null95": round(null, 4)}
        ax = axes[1, c]
        for i in range(3):
            ax.plot(A, Ue[:, i] * se[i] + 0 * i, lw=0.9, label=f"sv{i + 1} = {se[i]:.3g}")
        ax.set_title(
            f"residual before the one-hot remainder: top singular functions (null 95%: {null:.3g})",
            fontsize=8,
        )
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "v2_smooth_and_residual.png", dpi=90)
    plt.close(fig)

    # example readers at site 2: raw read vs V2 structured prediction (all stages but the remainder)
    Z, W, names = S["site2"]
    mu, parts, _ = stagewise(Z, H_main()[:-1])
    Yhat = (mu + sum(parts)) @ W
    Y = Z @ W
    r2 = 1 - ((Y - Yhat) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    res["site2_reader_r2_structured"] = {
        "median": float(np.median(r2)),
        "quartiles": np.quantile(r2, [0.25, 0.75]).round(3).tolist(),
    }
    pick = [
        "L0.gate.c19",
        "L0.gate.c7",
        "L0.gate.c44",
        "L0.up.c52",
        "L0.up.c145",
        "L0.up.c101",
        "L0.up.c415",
        "L0.gate.c109",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5))
    for ax, nm in zip(axes.flat, pick, strict=True):
        j = names.index(nm)
        ax.plot(A, Y[:, j], "k-", lw=0.9, label="raw read")
        ax.plot(A, Yhat[:, j], "C1-", lw=0.9, label="V2 structured part")
        ax.set_title(f"{nm} | structured R2 = {r2[j]:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("V2 at site 2: each reader's raw read vs the projection of the shared structured quantities "
                 "(no per-reader selection, no one-hot remainder)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "v2_readers.png", dpi=90)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.hist(r2, bins=np.linspace(min(-0.2, r2.min()), 1, 25), color="C0")
    ax.set_title(
        f"site 2: per-reader R2 of the V2 structured part (median {np.median(r2):.2f})", fontsize=8
    )
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "v2_reader_r2.png", dpi=90)
    plt.close(fig)

    (HERE / "v2_results.json").write_text(json.dumps(res, indent=1))
    for key in S:
        print("==", key)
        for r in res["main"][key]:
            print(f"  {r['quantity']:<36} d={r['dims']:3d} incr={r['incr']:.3f} chance={r['chance']:.3f} "
                  f"excess={r['excess']:+.3f} cum={r['cum']:.3f} cv={r['cv_cum']:.3f} readers>10%={r['readers_10pct']}")  # fmt: skip
        print("  place-first order:")
        for r in res["place_first"][key]:
            print(f"    {r['quantity']:<34} incr={r['incr']:.3f} excess={r['excess']:+.3f}")
        print("  units:", res["units"][key])
        print("  residual:", res["residual"][key])
    print(res["site2_reader_r2_structured"])


if __name__ == "__main__":
    main()
