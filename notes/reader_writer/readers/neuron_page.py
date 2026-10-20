"""Figures and numbers of the "Mod 10 neurons" page: the block-0 a mod 10 analysis in the neuron basis of the
original Llama-3.1-8B (no decomposition). Nothing is centred or detrended.

Notation (also defined on the page):
  a        integer operand 1..400; r = a mod 10
  x(a)     original model's block-0 MLP input at position 1 (neurons_l0_input.py), R^4096
  n        a neuron reader: a row w_n of gate_proj ("g#<i>") or up_proj ("u#<i>"), i in 0..14335
  f_n(a)   = x(a) . w_n / ||w_n||
  mu_r     mean of x(a) over a mod 10 = r;  Psi_k = sum_r mu_r exp(-2 pi i k r / 10), k = 1..5
  P_k      orthonormal basis of span{Re Psi_k, Im Psi_k};  D = span(P_1..P_5)
  g_n^X(a) = Pi_X x(a) . w_n / ||w_n||;  r2_X(n) = 1 - sum_a (f_n - g_n^X)^2 / sum_a f_n^2

    python neuron_page.py --data <dir with l0_input.npz> --out <page dir>
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import HUB, A, bf16_tensor, embedding_inputs, kept_readers, principal_cosines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repr_repo"))
from repo import Repo  # noqa: E402

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
TAU = 0.5
THR = 0.05
ENC = {1: "a mod 10", 2: "a mod 5", 3: "a mod 10", 4: "a mod 5", 5: "a mod 2"}
COLORS = ["#1f4e9c", "#6c9bd1", "#b5442d", "#e39a7d", "#7a7a7a"]


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def r2(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    return 1 - ((F - G) ** 2).sum(0) / (F**2).sum(0)


def fourier_planes(mu: np.ndarray) -> dict[int, np.ndarray]:
    rr = np.arange(10)
    out = {}
    for k in range(1, 6):
        psi = mu.T @ np.exp(-2j * np.pi * k * rr / 10)
        out[k] = orth(np.stack([psi.real, psi.imag], 1) if k < 5 else psi.real[:, None])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    x = np.load(args.data / "l0_input.npz")["x"].astype(np.float64)  # (400, 4096)
    (snap,) = glob.glob(HUB)
    idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    W = np.concatenate(
        [
            bf16_tensor(
                snap + idx[f"model.layers.0.mlp.{k}_proj.weight"],
                f"model.layers.0.mlp.{k}_proj.weight",
            )
            for k in ("gate", "up")
        ]
    ).astype(np.float64)  # (28672, 4096)
    names = [f"g#{i}" for i in range(14336)] + [f"u#{i}" for i in range(14336)]
    wn = np.linalg.norm(W, axis=1)
    Wu = W / wn[:, None]
    F = x @ Wu.T  # (400, 28672)
    rr = A % 10
    mu = np.stack([x[rr == r].mean(axis=0) for r in range(10)])  # (10, 4096)
    P = fourier_planes(mu)
    D = orth(np.concatenate(list(P.values()), 1))
    XD = x @ D

    def pred(B: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return (x @ B) @ (B.T @ Wu[rows].T)

    allr = np.arange(W.shape[0])
    r2D = r2(F, (XD @ (D.T @ Wu.T)))
    rng = np.random.default_rng(0)
    ctrl = np.stack(
        [r2(F, (x @ (B := orth(rng.standard_normal((4096, 9))))) @ (B.T @ Wu.T)) for _ in range(5)]
    )
    sel = np.nonzero(r2D >= TAU)[0]
    order = np.argsort(-r2D)
    res["n_neurons"] = int(W.shape[0])
    res["n_selected"] = int(sel.size)
    res["n_selected_gate"] = int((sel < 14336).sum())
    res["selected"] = {names[i]: float(r2D[i]) for i in order[: sel.size]}
    res["next"] = {names[i]: float(r2D[i]) for i in order[sel.size : sel.size + 8]}
    res["control_random9_max"] = float(ctrl.max())
    res["r2D_quantiles_all"] = np.quantile(r2D, [0.5, 0.9, 0.99, 0.999]).tolist()

    # comparison with the decomposition's D (span of the kept block-0 components)
    repo = Repo()
    comp_reps = sorted(
        (r for r in repo.representations.values() if r["block"] == 0),
        key=lambda r: r["details"]["frequency_k"],
    )
    Dc = orth(np.concatenate([repo.basis(r["id"]) for r in comp_reps], 1))
    res["principal_cos_D_neuron_vs_components"] = principal_cosines(D, Dc).tolist()
    res["plane_cos_vs_components"] = {
        str(k): float(
            np.linalg.svd(P[k].T @ repo.basis(comp_reps[k - 1]["id"]), compute_uv=False).min()
        )
        for k in P
    }

    # 1) planes, with origin at zero
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
    res["planes"] = {}
    for ax, k in zip(axs, P, strict=True):
        pts, mpts = x @ P[k], mu @ P[k]
        c = mpts.mean(axis=0)
        res["planes"][str(k)] = {
            "centre_norm": float(np.linalg.norm(c)),
            "radius": float(np.linalg.norm(mpts - c, axis=1).mean()),
        }
        ax.axhline(0, color="#888", lw=0.6)
        if P[k].shape[1] == 2:
            ax.axvline(0, color="#888", lw=0.6)
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(r) for r in rr], s=4, alpha=0.35, lw=0)
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
            lim = 1.08 * np.abs(pts).max()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
        else:
            ax.scatter(A, pts[:, 0], c=[cmap(r) for r in rr], s=4)
            ax.set_xlabel("a")
        ax.set_title(f"P_{k}: {ENC[k]}")
    fig.suptitle(
        "x(a) in each Fourier plane of the original model; origin = zero (small dots: a; large: μ_r; colour: r)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "n_planes.png")
    plt.close(fig)
    nD = (XD**2).sum(1) / (x**2).sum(1)
    res["share_x_in_D"] = [float(nD.mean()), float(nD.min()), float(nD.max())]
    pc = np.array([[np.linalg.svd(P[k].T @ P[q], compute_uv=False).max() for q in P] for k in P])
    res["plane_cos"] = pc.tolist()

    # 2) screen
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    axs[0].plot(
        np.arange(1, 401), r2D[order[:400]], color="#2a5db0", lw=1.2, label="r²_D, neurons sorted"
    )
    axs[0].plot(
        np.arange(1, 401),
        np.sort(ctrl.max(axis=0))[::-1][:400],
        color="#999",
        lw=1,
        label="random 9-dim subspaces (max of 5)",
    )
    axs[0].axhline(TAU, color="#c0392b", ls="--", lw=1, label=f"τ = {TAU}")
    axs[0].set_xlabel("rank (top 400 of 28672)")
    axs[0].set_ylabel("r² (uncentred)")
    axs[0].set_ylim(-0.05, 1.02)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title(
        f"{sel.size} neurons with r²_D ≥ τ ({res['n_selected_gate']} gate, {sel.size - res['n_selected_gate']} up)"
    )
    axs[1].hist(r2D, bins=np.linspace(-0.2, 1, 61), color="#2a5db0", log=True)
    axs[1].set_xlabel("r²_D")
    axs[1].set_ylabel("neurons (log)")
    axs[1].set_title("all 28672 block-0 neurons")
    fig.tight_layout()
    fig.savefig(fig_dir / "n_screen.png")
    plt.close(fig)

    # profiles and residue coverage
    m = np.stack([F[rr == r][:, sel].mean(axis=0) for r in range(10)])  # (10, |sel|)
    main_r = np.argmax(np.abs(m), axis=0)
    sgn = np.sign(m[main_r, np.arange(sel.size)])
    nm = sgn * m / np.abs(m).max(axis=0)
    second = np.sort(np.abs(nm), axis=0)[-2]
    res["coverage"] = {str(r): int((main_r == r).sum()) for r in range(10)}
    res["second_abs_quantiles"] = np.quantile(second, [0.5, 0.9, 1.0]).tolist()
    res["median_offresidue_quantiles"] = np.quantile(
        np.abs(np.median(nm, axis=0)), [0.5, 0.9, 1.0]
    ).tolist()
    res["flipped_count"] = int((sgn < 0).sum())
    res["flipped_gate"] = int(((sgn < 0) & (sel < 14336)).sum())
    pk = np.lexsort((-r2D[sel], main_r))
    fig, ax = plt.subplots(figsize=(4.6, max(3.0, 0.11 * sel.size + 1.2)))
    im = ax.imshow(
        nm[:, pk].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest"
    )
    ax.set_yticks(range(sel.size))
    ax.set_yticklabels(
        [f"{names[sel[i]]} ({r2D[sel[i]]:.2f})" for i in pk], fontsize=5 if sel.size > 40 else 7
    )
    ax.set_xticks(range(10))
    ax.set_xlabel("r = a mod 10")
    ax.set_title("s_n m_n(r) / max_r |m_n(r)|")
    fig.colorbar(im, ax=ax, shrink=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "n_profiles.png")
    plt.close(fig)

    # 3) per-plane reading
    share = np.stack([((P[k].T @ Wu[sel].T) ** 2).sum(0) for k in P])  # (5, |sel|)
    per = np.stack([r2(F[:, sel], pred(P[k], sel)) for k in P])  # (5, |sel|)
    res["share_in_D_quantiles"] = np.quantile(
        ((D.T @ Wu[sel].T) ** 2).sum(0), [0, 0.5, 1.0]
    ).tolist()
    res["max_single_plane_r2"] = float(per.max())
    res["mean_share_per_plane"] = share.mean(1).tolist()
    res["mean_r2_per_plane"] = per.mean(1).tolist()
    r13 = r2(F[:, sel], pred(orth(np.concatenate([P[1], P[3]], 1)), sel))
    res["r2_planes_1_3_range"] = [float(r13.min()), float(r13.max())]
    show = pk if sel.size <= 40 else np.concatenate([pk[main_r[pk] == r][:4] for r in range(10)])
    fig, axs = plt.subplots(1, 2, figsize=(max(10, 0.32 * show.size * 2), 4.2))
    for ax, vals_all, ttl in (
        (axs[0], share, "‖P_k^T w_n‖² / ‖w_n‖²"),
        (axs[1], per, "r² with the input projected onto plane k alone"),
    ):
        bottom = np.zeros(show.size)
        for ci, k in enumerate(P):
            ax.bar(
                range(show.size),
                vals_all[ci, show],
                bottom=bottom,
                color=COLORS[ci],
                label=f"k={k} ({ENC[k]})",
            )
            bottom = bottom + vals_all[ci, show]
        ax.set_xticks(range(show.size))
        ax.set_xticklabels(
            [f"{names[sel[j]]} (r={int(main_r[j])})" for j in show], rotation=90, fontsize=6
        )
        ax.set_title(ttl, fontsize=9)
    axs[1].plot(
        range(show.size), r2D[sel][show], "k_", ms=10, mew=2, label="r²_D (all planes jointly)"
    )
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(fig_dir / "n_per_plane.png")
    plt.close(fig)
    res["per_plane_shown"] = [names[sel[j]] for j in show]

    # arrows (sign s_n applied for display)
    cosres = np.zeros((4, sel.size))
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.0))
    for ax, k in zip(axs, P, strict=True):
        mpts = mu @ P[k]
        arr = (P[k].T @ Wu[sel].T) * sgn
        if k < 5:
            cosres[k - 1] = (
                (arr * mpts[main_r].T).sum(0)
                / np.linalg.norm(arr, axis=0)
                / np.linalg.norm(mpts[main_r], axis=1)
            )
        lim = 1.15 * np.abs(mpts).max()
        scale = 0.9 * lim / np.linalg.norm(arr[:, show], axis=0).max()
        ax.axhline(0, color="#888", lw=0.6)
        if k < 5:
            ax.axvline(0, color="#888", lw=0.6)
            ax.plot(*np.vstack([mpts, mpts[:1]]).T, "-", color="#ddd", lw=0.8)
            ax.scatter(
                mpts[:, 0],
                mpts[:, 1],
                c=[cmap(r) for r in range(10)],
                s=40,
                edgecolor="k",
                lw=0.6,
                zorder=3,
            )
            for r in range(10):
                ax.annotate(
                    str(r),
                    mpts[r],
                    fontsize=8,
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontweight="bold",
                )
            for j in show:
                ax.annotate(
                    "",
                    xy=tuple(scale * arr[:, j]),
                    xytext=(0, 0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": cmap(int(main_r[j])),
                        "lw": 0.9,
                        "alpha": 0.85,
                    },
                )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
        else:
            ax.scatter(
                mpts[:, 0],
                np.zeros(10),
                c=[cmap(r) for r in range(10)],
                s=40,
                edgecolor="k",
                lw=0.6,
                zorder=3,
            )
            for i2, j in enumerate(show):
                y = -0.05 - 0.04 * i2
                ax.annotate(
                    "",
                    xy=(scale * arr[0, j], y),
                    xytext=(0, y),
                    arrowprops={"arrowstyle": "-|>", "color": cmap(int(main_r[j])), "lw": 0.9},
                )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.1 - 0.04 * show.size, 0.2)
            ax.set_yticks([])
            ax.axvline(0, color="#888", lw=0.6)
        ax.set_title(f"P_{k}: {ENC[k]}  (arrows ×{scale:.2f})")
    fig.suptitle(
        f"residue means μ_r and the read directions s_n P_k^T w_n / ‖w_n‖ of {show.size} neurons (colour: main residue)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "n_arrows.png")
    plt.close(fig)
    res["arrow_cos_range"] = [float(cosres.min()), float(cosres.max())]
    res["arrow_cos_q05"] = float(np.quantile(cosres, 0.05))

    # lower-period-only neurons
    per_all = np.stack([r2(F, (x @ P[k]) @ (P[k].T @ Wu.T)) for k in P])  # (5, 28672)
    resp = per_all >= THR
    low_only = resp[[1, 3, 4]].any(0) & ~resp[[0, 2]].any(0)
    tot_low = np.zeros(W.shape[0])
    for i in np.nonzero(low_only)[0]:
        ks = [k for k in (2, 4, 5) if resp[k - 1, i]]
        B = orth(np.concatenate([P[k] for k in ks], 1))
        tot_low[i] = r2(F[:, [i]], (x @ B) @ (B.T @ Wu[[i]].T))[0]
    lo = np.nonzero(low_only)[0]
    res["low_only_count"] = int(lo.size)
    res["low_only_top"] = {names[i]: float(tot_low[i]) for i in lo[np.argsort(-tot_low[lo])][:8]}
    res["low_only_parity_only_top"] = {
        names[i]: float(per_all[4, i]) for i in lo[np.argsort(-per_all[4, lo])][:5]
    }
    res["responding_any"] = int(resp.any(0).sum())

    # embedding
    z = embedding_inputs()
    eps = np.stack([z[rr == r].mean(axis=0) for r in range(10)])
    De = orth(np.concatenate(list(fourier_planes(eps).values()), 1))
    res["principal_cos_D_vs_emb"] = principal_cosines(D, De).tolist()

    # relation to the 11 mod-10 components
    cols, cnames, V = kept_readers()
    comp_sel = ["g43", "g47", "g58", "g59", "g62", "g69", "u31", "u52", "u54", "u60", "u794"]
    best = {}
    for cn in comp_sel:
        v = V[:, cnames.index(cn)]
        cs = Wu @ (v / np.linalg.norm(v))
        j = int(np.argmax(np.abs(cs)))
        best[cn] = {"neuron": names[j], "cos": float(cs[j]), "neuron_r2D": float(r2D[j])}
    res["best_neuron_per_component"] = best

    # 4) the other neurons
    others = np.setdiff1d(allr, sel)
    Gd = XD @ (D.T @ Wu[others].T)
    change = np.linalg.norm(Gd, axis=0) / np.linalg.norm(F[:, others], axis=0)
    res["others_change_quantiles"] = np.quantile(change, [0.5, 0.9, 0.99, 1.0]).tolist()
    oo = np.argsort(-change)
    res["others_change_top"] = {names[others[j]]: float(change[j]) for j in oo[:10]}
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.hist(change, bins=80, color="#2a5db0", log=True)
    ax.set_xlabel("‖g_n^D‖ / ‖f_n‖")
    ax.set_ylabel("neurons (log)")
    ax.set_title(f"{others.size} other neurons: size of the part read from D")
    fig.tight_layout()
    fig.savefig(fig_dir / "n_others_change.png")
    plt.close(fig)
    fig, axs = plt.subplots(12, 2, figsize=(12, 15), sharex=True)
    for axx, j in zip(axs.T.ravel(), oo[:24], strict=True):
        i = others[j]
        axx.axhline(0, color="#ddd", lw=0.6)
        axx.plot(A, F[:, i], lw=0.7, color="#bbb")
        axx.plot(A, F[:, i] - Gd[:, j], lw=0.8, color="#2a5db0")
        axx.set_ylabel(names[i], fontsize=8)
        axx.text(
            0.995,
            0.92,
            f"change {change[j]:.2f}",
            transform=axx.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    axs[-1, 0].set_xlabel("a")
    axs[-1, 1].set_xlabel("a")
    fig.suptitle("other neurons, ordered by change: f_n (grey) and ρ_n = f_n − g_n^D (blue)")
    fig.tight_layout()
    fig.savefig(fig_dir / "n_others_residual.png")
    plt.close(fig)
    (args.out / "neuron_page.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps({k: v for k, v in res.items() if k not in ("selected", "plane_cos")}, indent=1)
    )
    print("selected:", list(res["selected"].items())[:60])


if __name__ == "__main__":
    main()
