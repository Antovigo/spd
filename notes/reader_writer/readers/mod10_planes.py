"""Follow-up to mod10_L0.py: the Fourier planes of the block-0 a mod 10 subspace D.

Notation as in mod10_L0.py. In addition:
  k          integer frequency 1..5; Psi_k = sum_r mu_r exp(-2 pi i k r / 10) (i the imaginary unit)
  P_k        orthonormal basis of span{Re Psi_k, Im Psi_k} (k = 1..4: a plane; k = 5: a line)
  D          span of all P_k (= span{mu_r - mu_bar}, 9-dimensional)
  Pi_X       orthogonal projection onto span X
  fhat_c     reader c's activation when its input x_V(a) is replaced by Pi_X x_V(a):
             centred version  V_c . Pi_X (x_V(a) - mean_a x_V) / ||V_c|| + mean f_c   (keeps the reader's mean)
             raw version      V_c . Pi_X x_V(a) / ||V_c||                           (discards the rest outright)

    python mod10_planes.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import SRC, VALID, A, N, class_means, kept_readers, moving_trend, periodic_share

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols, names, V = kept_readers()
    vnorm = np.linalg.norm(V, axis=0)
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    F = T / vnorm
    sel = np.nonzero(periodic_share(F, 10) >= args.tau)[0]
    W = V @ np.linalg.pinv(V.T @ V)
    xv = (W @ T.T).T  # (400, 4096)
    xbar = xv.mean(axis=0)
    mu = W @ class_means(T - moving_trend(T, 10), 10, VALID).T
    rr = np.arange(10)
    psi = {k: mu @ np.exp(-2j * np.pi * k * rr / 10) for k in range(1, 6)}
    P = {
        k: orth(np.stack([psi[k].real, psi[k].imag], 1) if k < 5 else psi[k].real[:, None])
        for k in psi
    }
    res: dict = {"dims": {str(k): int(P[k].shape[1]) for k in P}}
    # plane-to-plane principal cosines
    pc = np.zeros((5, 5))
    for k in P:
        for q in P:
            pc[k - 1, q - 1] = np.linalg.svd(P[k].T @ P[q], compute_uv=False).max()
    res["plane_max_principal_cos"] = pc.tolist()
    D = orth(np.concatenate([P[k] for k in P], 1))
    res["dim_D"] = int(D.shape[1])
    # same span as the centred residue means?
    Dmu = orth(mu - mu.mean(axis=1, keepdims=True))
    res["D_vs_mu_principal_cos_min"] = float(np.linalg.svd(D.T @ Dmu, compute_uv=False).min())

    subsets = {"all (D)": list(P)} | {f"k={k}": [k] for k in P} | {"k=1,3": [1, 3]}
    Fs = F[:, sel]
    Fc = Fs - Fs.mean(axis=0)
    xc = xv - xbar
    ev, rel = {}, {}
    for name, ks in subsets.items():
        B = orth(np.concatenate([P[k] for k in ks], 1))
        g = (xc @ B) @ (B.T @ V[:, sel]) / vnorm[sel]
        ev[name] = (1 - ((Fc - g) ** 2).sum(0) / (Fc**2).sum(0)).tolist()
        rel[name] = (np.linalg.norm(Fc - g, axis=0) / np.linalg.norm(Fc, axis=0)).tolist()
    res["readers"] = [names[i] for i in sel]
    res["ev_centred"] = ev
    res["rel_err_centred"] = rel
    # raw: discard everything outside D, including the mean input
    raw = (xv @ D) @ (D.T @ V[:, sel]) / vnorm[sel]
    res["raw_offset_shift_over_std"] = ((raw.mean(0) - Fs.mean(0)) / Fs.std(0)).tolist()
    res["raw_mean_f"] = Fs.mean(0).tolist()
    res["raw_mean_fhat"] = raw.mean(0).tolist()
    res["raw_corr"] = [float(np.corrcoef(raw[:, j], Fs[:, j])[0, 1]) for j in range(len(sel))]
    # the part of the mean input inside D
    res["share_of_mean_input_in_D"] = float(
        np.linalg.norm(D.T @ xbar) ** 2 / np.linalg.norm(xbar) ** 2
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    labels = list(subsets)
    width = 0.8 / len(labels)
    for li, name in enumerate(labels):
        ax.bar(np.arange(len(sel)) + li * width, ev[name], width, label=name)
    ax.set_xticks(np.arange(len(sel)) + 0.4 - width / 2)
    ax.set_xticklabels(res["readers"])
    ax.set_ylabel("variance of f_c explained")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=4)
    ax.set_title("mod-10 readers: input replaced by its projection onto the listed Fourier planes")
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_planes_ev.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(pc, cmap="Greys", vmin=0, vmax=1)
    for k in range(5):
        for q in range(5):
            ax.text(
                q,
                k,
                f"{pc[k, q]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if pc[k, q] > 0.5 else "black",
            )
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"k={k}" for k in P])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"k={k}" for k in P])
    ax.set_title("largest principal cos between planes")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_planes_cos.png")
    plt.close(fig)

    sl = slice(0, 100)
    fig, axs = plt.subplots(len(sel), 1, figsize=(9, 1.25 * len(sel)), sharex=True)
    g_all = (xc @ D) @ (D.T @ V[:, sel]) / vnorm[sel] + Fs.mean(0)
    for j, axx in enumerate(axs):
        axx.plot(A[sl], Fs[sl, j], lw=1.6, color="#bbb", label="f_c")
        axx.plot(
            A[sl], g_all[sl, j], lw=0.8, color="#2a5db0", label="input → Π_D (centred), + mean f_c"
        )
        axx.plot(A[sl], raw[sl, j], lw=0.8, color="#c0392b", ls="--", label="input → Π_D x_V (raw)")
        axx.set_ylabel(res["readers"][j], fontsize=8)
    axs[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.6))
    axs[-1].set_xlabel("a (first 100)")
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_planes_overlay.png")
    plt.close(fig)
    (args.out / "l0_planes.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
