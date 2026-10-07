"""Inspection for the agent (AGENT.md steps 1, 4 and 7): what is left at a site, and where it matters.

For a site of position t and its accepted list (qloop_t<t>[_tag].json):
* the readers ranked by CI-weighted residual (sum over the domain of CI x residual^2), with their
  share of the site's total and the domain values where their weighted errors are largest;
* a figure of the top readers: raw read, reconstruction and CI over the domain (t = 1: curves over
  a; t = 2: curves over a for each op; t >= 3: (a, b) grids for each op: read, residual, CI);
* the residual's top singular functions over the domain (shared patterns), as values and plots.

    python qinspect.py <t> <site> [--tag name] [--top 6]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qdata import DATA, load

FIG = DATA / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("t", type=int)
    ap.add_argument("site")
    ap.add_argument("--tag", default="")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    pos = load(args.t, [args.site])
    s = pos.sites[args.site]
    R = np.load(DATA / f"recon_t{args.t}{tag}.npz")
    run = {
        r["site"]: r for r in json.loads((DATA / f"qloop_t{args.t}{tag}.json").read_text())["sites"]
    }[args.site]
    Y, Yh = R[args.site + "/Y"].astype(np.float64), R[args.site + "/Yhat"].astype(np.float64)
    E = Y - Yh
    w = (s.CI * E**2).sum(0)
    order = np.argsort(-w)[: args.top]
    print(f"{args.site} t={args.t}: accepted {[a['name'] for a in run['accepted']]}")
    print(
        f"  important residual share in-sample {run['important_share_in']:.3f}, held-out {run['important_share_heldout']:.3f}"
    )
    names = [f"c{int(c)}" for c in s.cols]
    for r in order:
        top = np.argsort(-s.CI[:, r] * E[:, r] ** 2)[:6]
        where = [(int(pos.op[i]), int(pos.a[i]), int(pos.b[i])) for i in top]
        print(f"  reader {names[r]:<8} share {w[r] / w.sum():.2f}; important on {(s.CI[:, r] > 0.01).sum()} of {pos.D} rows; "
              f"largest weighted errors at (op, a, b) = {where}")  # fmt: skip
    U, sv, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    for j in range(2):
        top = np.argsort(-np.abs(U[:, j]))[:10]
        print(f"  residual singular function {j + 1} (sv {sv[j]:.3g}): largest at (op, a, b) = "
              f"{[(int(pos.op[i]), int(pos.a[i]), int(pos.b[i]), round(float(U[i, j]), 3)) for i in top]}")  # fmt: skip

    FIG.mkdir(parents=True, exist_ok=True)
    if args.t <= 2:
        fig, axes = plt.subplots(1, len(order), figsize=(3.6 * len(order), 2.8), squeeze=False)
        for ax, r in zip(axes[0], order, strict=True):
            for op in np.unique(pos.op):
                m = pos.op == op
                ax.plot(
                    pos.a[m], Y[m, r], "-", color=f"C{2 * op}", lw=0.9, label=f"read, op {'+-'[op]}"
                )
                ax.plot(
                    pos.a[m],
                    Yh[m, r],
                    "--",
                    color=f"C{2 * op + 1}",
                    lw=0.9,
                    label=f"reconstruction, op {'+-'[op]}",
                )
            ax2 = ax.twinx()
            ax2.bar(pos.a, s.CI[:, r], color="0.5", alpha=0.25, width=1.0)
            ax2.set_ylim(0, 1.05)
            ax.set_title(
                f"{args.site} {names[r]}: {w[r] / w.sum():.2f} of the CI-weighted residual",
                fontsize=7,
            )
            ax.tick_params(labelsize=6)
        axes[0, 0].legend(fontsize=5)
    else:
        fig, axes = plt.subplots(len(order), 6, figsize=(16, 2.6 * len(order)), squeeze=False)
        for i, r in enumerate(order):
            for op in (0, 1):
                m = pos.op == op
                g = lambda v: v[m].reshape(100, 100)  # noqa: B023, E731
                for j, (mat, title, cmap) in enumerate(
                    (
                        (g(Y[:, r]), "read", "RdBu_r"),
                        (g(E[:, r]), "residual", "RdBu_r"),
                        (g(s.CI[:, r]), "CI", "Greys"),
                    )
                ):
                    ax = axes[i, 3 * op + j]
                    lim = np.abs(g(Y[:, r])).max() if j < 2 else 1
                    ax.imshow(mat, cmap=cmap, vmin=-lim if j < 2 else 0, vmax=lim)
                    ax.set_title(f"{names[r]} a {'+-'[op]} b: {title}", fontsize=7)
                    ax.set_xticks([])
                    ax.set_yticks([])
        fig.suptitle(f"{args.site} t={args.t}: grids with a down, b across", fontsize=9)
    fig.tight_layout()
    out = FIG / f"inspect_t{args.t}_{args.site}{tag}.png"
    fig.savefig(out, dpi=80)
    print(f"  figure: {out}")


if __name__ == "__main__":
    main()
