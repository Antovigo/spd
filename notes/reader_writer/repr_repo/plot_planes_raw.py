"""Uncentred views of the block-0 a mod 10 representations of the representation repository.

The network has no biases, so every plot has the zero of the activation space at its origin; nothing is
centred or detrended. For each Fourier representation R (basis P_R, orthonormal columns in R^4096):
the coordinates P_R^T x_V(a) of the input of every prompt a = 1..400 at position 1, and those of the raw
residue means mu_r = mean of x_V(a) over a mod 10 = r. Also the cosines between the raw mu_r.

    python plot_planes_raw.py --out <notebook dir>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "readers"))
from mod10_L0 import SRC, A, N, kept_readers  # noqa: E402
from repo import Repo  # noqa: E402

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "fig").mkdir(parents=True, exist_ok=True)
    repo = Repo()
    reps = sorted(
        (r for r in repo.representations.values() if r["block"] == 0),
        key=lambda r: r["details"]["frequency_k"],
    )
    cols, names, V = kept_readers()
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T  # (400, 4096)
    res = rr = A % 10  # residue of every prompt
    mu = np.stack([xv[rr == r].mean(axis=0) for r in range(10)])  # (10, 4096)
    D = np.linalg.svd(np.concatenate([repo.basis(r["id"]) for r in reps], 1), full_matrices=False)[
        0
    ]
    out: dict = {
        "share_of_x_norm2_in_D": {
            "mean": float(((xv @ D) ** 2).sum(1).mean() / (xv**2).sum(1).mean()),
            "min": float((((xv @ D) ** 2).sum(1) / (xv**2).sum(1)).min()),
            "max": float((((xv @ D) ** 2).sum(1) / (xv**2).sum(1)).max()),
        },
        "planes": {},
    }
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
    cmap = plt.get_cmap("tab10")
    for ax, rep in zip(axs, reps, strict=True):
        k, B = rep["details"]["frequency_k"], repo.basis(rep["id"])
        pts, mpts = xv @ B, mu @ B
        centre = mpts.mean(axis=0)
        radius = float(np.linalg.norm(mpts - centre, axis=1).mean())
        out["planes"][rep["id"]] = {
            "norm_of_mean_of_residue_means": float(np.linalg.norm(centre)),
            "mean_distance_of_residue_means_from_their_mean": radius,
            "mean_norm_of_residue_means": float(np.linalg.norm(mpts, axis=1).mean()),
            "mean_norm_of_inputs": float(np.linalg.norm(pts, axis=1).mean()),
        }
        if B.shape[1] == 2:
            ax.axhline(0, color="#888", lw=0.6)
            ax.axvline(0, color="#888", lw=0.6)
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(r) for r in res], s=4, alpha=0.35, lw=0)
            ax.scatter(
                mpts[:, 0],
                mpts[:, 1],
                c=[cmap(r) for r in range(10)],
                s=36,
                edgecolor="k",
                lw=0.6,
                zorder=3,
            )
            for r in range(10):
                ax.annotate(str(r), mpts[r], fontsize=8, xytext=(4, 3), textcoords="offset points")
            ax.plot(0, 0, "k+", ms=10, mew=1.4, zorder=4)
            lim = 1.08 * max(np.abs(pts).max(), 1e-9)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("coordinate 1")
            ax.set_ylabel("coordinate 2")
        else:
            ax.axhline(0, color="#888", lw=0.6)
            ax.scatter(A, pts[:, 0], c=[cmap(r) for r in res], s=4)
            ax.set_xlabel("a")
            ax.set_ylabel("coordinate")
        ax.set_title(f"k = {k}: {rep['encodes']}")
    fig.suptitle(
        "x_V(a) projected onto each representation, origin = zero of the activation space "
        "(small dots: a = 1..400; large: residue means μ_r; colour: r = a mod 10)"
    )
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "l0_planes_raw.png")
    plt.close(fig)

    mun = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    cos = mun @ mun.T
    muD = mu @ D
    cosD = (muD / np.linalg.norm(muD, axis=1, keepdims=True)) @ (
        muD / np.linalg.norm(muD, axis=1, keepdims=True)
    ).T
    out["cos_mu_raw_offdiag"] = [
        float(cos[~np.eye(10, dtype=bool)].min()),
        float(cos[~np.eye(10, dtype=bool)].max()),
    ]
    out["cos_mu_in_D_offdiag"] = [
        float(cosD[~np.eye(10, dtype=bool)].min()),
        float(cosD[~np.eye(10, dtype=bool)].max()),
    ]
    out["cos_mu_in_D"] = cosD.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, c, t in (
        (axs[0], cos, "cos(μ_r, μ_r′), raw"),
        (axs[1], cosD, "cos(Π_D μ_r, Π_D μ_r′)"),
    ):
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("r′")
        ax.set_ylabel("r")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "l0_mu_cos_raw.png")
    plt.close(fig)

    sel = [
        "L0.gate.43",
        "L0.gate.47",
        "L0.gate.58",
        "L0.gate.59",
        "L0.gate.62",
        "L0.gate.69",
        "L0.up.31",
        "L0.up.52",
        "L0.up.54",
        "L0.up.60",
        "L0.up.794",
    ]
    out["readers"] = {s: repo.readers[s]["positions"]["1"] for s in sel}
    (args.out / "l0_planes_raw.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps({k: v for k, v in out.items() if k not in ("readers", "cos_mu_in_D")}, indent=1)
    )
    for s in sel:
        p = repo.readers[s]["positions"]["1"]
        print(
            s, round(p["r2"], 3), {r.split(".")[-1]: round(v, 3) for r, v in p["r2_by_rep"].items()}
        )


if __name__ == "__main__":
    main()
