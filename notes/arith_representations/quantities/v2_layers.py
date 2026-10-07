"""V2 at every read site of the network, token "a" (position 1): the L0 hypothesis set fit at
each attention input (stream position l = 2b) and MLP input (l = 2b + 1), b = 0..31, plus residual
diagnostics that point to quantities the set misses.

    python v2_layers.py <hypothesis set name in v2_sets.py> <tag>

Writes `layers_<tag>.json` and figures `figures/v2_layers_<tag>_*.png`.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import v2_sets
from v2 import A, reader_dirs, residual_svd, span_basis, stagewise, stream, variance_table

HERE = Path(__file__).parent
FIG = HERE / "figures"


def site_list() -> list[tuple[int, str, int]]:
    return [(b, pt, 2 * b + (pt == "mlp")) for b in range(32) for pt in ("attn", "mlp")]


def residual_spectrum(E: np.ndarray) -> np.ndarray:
    """Share of the residual's power at each frequency k = 1..50 over a (period 100 / k), pooled
    over the reader dimensions."""
    F = np.abs(np.fft.rfft(E - E.mean(0), axis=0)[1:]) ** 2
    F[:-1] *= 2  # k = 1..49 appear twice in the full spectrum, k = 50 once
    p = F.sum(1)
    return p / p.sum()


def main() -> None:
    hname, tag = sys.argv[1], sys.argv[2]
    H = getattr(v2_sets, hname)()
    out = []
    for b, pt, lpos in site_list():
        try:
            Vt, names, _ = reader_dirs(b, pt)
        except IndexError:
            continue
        if len(names) == 0:
            continue
        Qb = span_basis(Vt)
        Z, W = stream(lpos) @ Qb, Qb.T @ Vt.T
        rows = variance_table(Z, W, H)
        mu, parts, _ = stagewise(Z, H[:-1])
        E = (Z - mu - sum(parts)) @ W
        U, s, null = residual_svd(E)
        spec = residual_spectrum(E)
        out.append({
            "site": f"L{b}.{pt}", "l": lpos, "n_readers": len(names), "k": int(Qb.shape[1]), "rows": rows,
            "resid_sv1": float(s[0]), "resid_null95": null, "resid_u1": (U[:, 0] * s[0]).tolist(),
            "resid_spectrum": spec.tolist(),
        })  # fmt: skip
        print(f"L{b}.{pt}: n={len(names)} k={Qb.shape[1]} remainder={rows[-1]['incr']:.3f} "
              f"cv={rows[-2]['cv_cum']:.3f} sv1/null={s[0] / null:.2f} top k={int(np.argmax(spec)) + 1}", flush=True)  # fmt: skip
    (HERE / f"layers_{tag}.json").write_text(json.dumps(out))
    plot(out, H, tag)


def plot(out: list[dict], H: list, tag: str) -> None:
    labels = [o["site"] for o in out]
    names = [r["quantity"] for r in out[0]["rows"]]
    ex = np.array([[r["excess"] for r in o["rows"]] for o in out]).T  # (quantities, sites)
    inc = np.array([[r["incr"] for r in o["rows"]] for o in out]).T
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    ax = axes[0]
    vmax = 0.3
    im = ax.imshow(ex[:-1], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(ex.shape[0] - 1):
        for j in range(ex.shape[1]):
            if abs(ex[i, j]) >= 0.05:
                ax.text(j, i, f"{ex[i, j]:.2f}", ha="center", va="center", fontsize=5)
    ax.set_yticks(range(len(names) - 1), [f"{n}" for n in names[:-1]], fontsize=7)
    ax.set_xticks(range(len(labels)), [f"{lab} ({o['n_readers']})" for lab, o in zip(labels, out, strict=True)],
                  rotation=90, fontsize=6)  # fmt: skip
    ax.set_title(
        "variance explained beyond chance (reader metric, stagewise in this order); site (number of active readers)",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.015)
    ax = axes[1]
    x = np.arange(len(out))
    ax.plot(x, inc[-1], "k.-", lw=0.8, label="one-hot remainder (token-specific)")
    ax.plot(x, [o["rows"][-2]["cum"] for o in out], "C0.-", lw=0.8, label="structured, in-sample")
    ax.plot(
        x,
        [o["rows"][-2]["cv_cum"] for o in out],
        "C1.-",
        lw=0.8,
        label="structured, held-out values of a",
    )
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(0, color="0.7", lw=0.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axes[2]
    ratio = [o["resid_sv1"] / o["resid_null95"] for o in out]
    ax.bar(x, ratio, color=["C3" if r > 1.2 else "0.6" for r in ratio])
    for j, o in enumerate(out):
        k = int(np.argmax(o["resid_spectrum"])) + 1
        ax.text(j, ratio[j] + 0.02, f"{k}", ha="center", fontsize=5)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_title("residual before the one-hot remainder: top singular value / 95% permutation null (red > 1.2); "
                 "number = dominant frequency k over a (period 100/k)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / f"v2_layers_{tag}_overview.png", dpi=90)
    plt.close(fig)

    flagged = [o for o in out if o["resid_sv1"] / o["resid_null95"] > 1.2]
    if flagged:
        n = len(flagged)
        fig, axes = plt.subplots(n, 2, figsize=(11, 1.6 * n), squeeze=False)
        for i, o in enumerate(flagged):
            axes[i, 0].plot(A, o["resid_u1"], "k-", lw=0.8)
            axes[i, 0].set_title(f"{o['site']}: top residual singular function (sv1/null "
                                 f"{o['resid_sv1'] / o['resid_null95']:.2f})", fontsize=7)  # fmt: skip
            axes[i, 1].bar(range(1, 51), o["resid_spectrum"], color="0.3")
            axes[i, 1].set_title("residual power share per frequency k", fontsize=7)
            for ax in axes[i]:
                ax.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(FIG / f"v2_layers_{tag}_flagged.png", dpi=80)
        plt.close(fig)


if __name__ == "__main__":
    main()
