"""Figures and numbers of the notebook page "Mod 10 representations" (block 0, position 1).

Nothing is centred or detrended anywhere: the network has no biases, so zero is the zero of the
activation space. Notation (also defined on the page):
  a        integer operand 1..400; r = a mod 10
  c        a kept block-0 reader (gate g<id> or up u<id>); V_c in R^4096 its read vector
  x_V(a)   block-0 MLP input at position 1, projected onto span of the kept readers' V_c
  f_c(a)   = x_V(a) . V_c / ||V_c||
  mu_r     mean of x_V(a) over the a with a mod 10 = r
  P_k      repository basis of frequency k (span{Re Psi_k, Im Psi_k}); D = span of P_1..P_5
  g_c^X(a) = Pi_X x_V(a) . V_c / ||V_c||;  r2_X(c) = 1 - sum_a (f_c - g_c^X)^2 / sum_a f_c^2

    python mod10_page.py --out <notebook dir>
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import SRC, A, N, embedding_inputs, kept_readers, principal_cosines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repr_repo"))
from repo import Repo  # noqa: E402

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
TAU = 0.5
SHOW_FIRST = ("u54", "u60")


def orth(X: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def r2(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    return 1 - ((F - G) ** 2).sum(0) / (F**2).sum(0)


def predict(xv: np.ndarray, B: np.ndarray, V: np.ndarray, vnorm: np.ndarray) -> np.ndarray:
    return (xv @ B) @ (B.T @ V) / vnorm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    repo = Repo()
    reps = sorted(
        (r for r in repo.representations.values() if r["block"] == 0),
        key=lambda r: r["details"]["frequency_k"],
    )
    P = {r["details"]["frequency_k"]: repo.basis(r["id"]) for r in reps}
    D = orth(np.concatenate(list(P.values()), 1))
    cols, names, V = kept_readers()
    K = len(names)
    vnorm = np.linalg.norm(V, axis=0)
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    F = T / vnorm
    Pv = np.linalg.pinv(V.T @ V)
    xv = (V @ Pv @ T.T).T
    rr = A % 10
    mu = np.stack([xv[rr == r].mean(axis=0) for r in range(10)])  # (10, 4096)

    # 1) which readers the planes explain
    r2D = r2(F, predict(xv, D, V, vnorm))
    rng = np.random.default_rng(0)
    Q = orth(V)
    ctrl = np.stack(
        [
            r2(F, predict(xv, Q @ orth(rng.standard_normal((Q.shape[1], 9))), V, vnorm))
            for _ in range(20)
        ]
    )
    sel = np.nonzero(r2D >= TAU)[0]
    order = np.argsort(-r2D)
    res["K"] = K
    res["tau"] = TAU
    res["selected"] = {names[i]: float(r2D[i]) for i in order if r2D[i] >= TAU}
    res["next"] = {names[i]: float(r2D[i]) for i in order[len(sel) : len(sel) + 5]}
    res["control_random9_max_over_readers_and_draws"] = float(ctrl.max())
    res["control_random9_mean"] = float(ctrl.mean())
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(K), r2D[order], color="#2a5db0", label="r²_D: input projected onto D")
    ax.plot(
        np.arange(K),
        np.maximum(ctrl.max(axis=0)[order], -0.2),
        "k.",
        ms=3,
        label="largest r² over 20 random 9-dim subspaces of span(V) (clipped at −0.2)",
    )
    ax.set_ylim(-0.22, 1.02)
    ax.axhline(TAU, color="#c0392b", lw=1, ls="--", label=f"τ = {TAU}")
    ax.set_xticks(np.arange(len(sel) + 10))
    ax.set_xticklabels([names[i] for i in order[: len(sel) + 10]], rotation=90, fontsize=6)
    ax.set_ylabel("r² (uncentred)")
    ax.set_title(f"block-0 readers at position 1, sorted by r²_D: {len(sel)} of {K} have r²_D ≥ τ")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_r2_sorted.png")
    plt.close(fig)

    # residue profiles: raw class means of f_c
    m = np.stack([F[rr == r].mean(axis=0) for r in range(10)])  # (10, K)
    msel = m[:, sel]
    main_r = np.argmax(np.abs(msel - np.median(msel, axis=0)), axis=0)
    # the sign of a read vector is a gauge choice (V_c, U_c -> -V_c, -U_c): flip each selected reader so that
    # its response at its main residue is positive
    sgn = np.sign(msel[main_r, np.arange(len(sel))])
    res["flipped"] = [names[sel[j]] for j in range(len(sel)) if sgn[j] < 0]
    prof = sgn * msel / np.abs(msel).max(axis=0)
    pk = np.argsort(main_r, kind="stable")
    fig, ax = plt.subplots(figsize=(4.4, 0.2 * len(sel) + 1.3))
    im = ax.imshow(prof[:, pk].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([names[sel[i]] for i in pk], fontsize=7)
    ax.set_xticks(range(10))
    ax.set_xlabel("r = a mod 10")
    ax.set_title("s_c m_c(r) / max_r |m_c(r)|")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_profiles.png")
    plt.close(fig)
    res["profiles"] = {names[i]: m[:, i].tolist() for i in sel}

    # overlay: f_c and its D prediction
    G = predict(xv, D, V, vnorm)
    showlist = [i for n in SHOW_FIRST for i in sel if names[i] == n] + [
        i for i in sel if names[i] not in SHOW_FIRST
    ]
    fig, axs = plt.subplots(len(showlist), 1, figsize=(9, 1.2 * len(showlist)), sharex=True)
    for axx, i in zip(axs, showlist, strict=True):
        axx.axhline(0, color="#ccc", lw=0.6)
        axx.plot(A[:100], F[:100, i], lw=1.6, color="#bbb")
        axx.plot(A[:100], G[:100, i], lw=0.8, color="#2a5db0")
        axx.set_ylabel(names[i], fontsize=8)
        axx.text(
            0.995,
            0.92,
            f"r²_D = {r2D[i]:.2f}",
            transform=axx.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    axs[-1].set_xlabel("a (first 100)")
    fig.suptitle("f_c(a) (grey) and g_c^D(a), its input projected onto D (blue)")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_overlay.png")
    plt.close(fig)

    # 2) the planes
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
    res["planes"] = {}
    for ax, k in zip(axs, P, strict=True):
        pts, mpts = xv @ P[k], mu @ P[k]
        centre = mpts.mean(axis=0)
        res["planes"][str(k)] = {
            "centre_norm": float(np.linalg.norm(centre)),
            "radius": float(np.linalg.norm(mpts - centre, axis=1).mean()),
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
            ax.set_xlabel("coordinate 1")
            ax.set_ylabel("coordinate 2")
        else:
            ax.scatter(A, pts[:, 0], c=[cmap(r) for r in rr], s=4)
            ax.set_xlabel("a")
            ax.set_ylabel("coordinate")
        ax.set_title(f"P_{k}: {reps[k - 1]['encodes']}")
    fig.suptitle(
        "x_V(a) in each representation; origin = zero of the activation space "
        "(small dots: a = 1..400; large: μ_r; colour: r = a mod 10)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_planes.png")
    plt.close(fig)

    # the selected readers' read directions inside each plane (arrows = P_k^T V_c / ||V_c||, scaled)
    Vu = sgn * V[:, sel] / vnorm[sel]
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.0))
    res["read_share_per_plane"] = {}
    for ax, k in zip(axs, P, strict=True):
        mpts = mu @ P[k]
        arr = P[k].T @ Vu  # (d, |sel|)
        res["read_share_per_plane"][str(k)] = dict(
            zip([names[i] for i in sel], (arr**2).sum(0).tolist(), strict=True)
        )
        lim = 1.15 * np.abs(mpts).max()
        scale = 0.9 * lim / max(np.linalg.norm(arr, axis=0).max(), 1e-12)
        ax.axhline(0, color="#888", lw=0.6)
        if P[k].shape[1] == 2:
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
            for j, i in enumerate(sel):
                vx, vy = scale * arr[:, j]
                ax.annotate(
                    "",
                    xy=(vx, vy),
                    xytext=(0, 0),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": cmap(int(main_r[j])),
                        "lw": 1.1,
                        "alpha": 0.9,
                    },
                )
                ax.annotate(
                    names[i],
                    (vx, vy),
                    fontsize=6,
                    color=cmap(int(main_r[j])),
                    xytext=(2, -7),
                    textcoords="offset points",
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
            for r in range(10):
                ax.annotate(
                    str(r),
                    (mpts[r, 0], 0),
                    fontsize=8,
                    xytext=(0, 6 + 8 * (r % 2)),
                    textcoords="offset points",
                    ha="center",
                )
            for j, i in enumerate(sel):
                y = -0.1 - 0.08 * j
                ax.annotate(
                    "",
                    xy=(scale * arr[0, j], y),
                    xytext=(0, y),
                    arrowprops={"arrowstyle": "-|>", "color": cmap(int(main_r[j])), "lw": 1.1},
                )
                ax.annotate(
                    names[i],
                    (scale * arr[0, j], y),
                    fontsize=6,
                    color=cmap(int(main_r[j])),
                    xytext=(3, -2),
                    textcoords="offset points",
                )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.2 - 0.08 * len(sel), 0.35)
            ax.set_yticks([])
            ax.axvline(0, color="#888", lw=0.6)
        ax.set_title(f"P_{k}: {reps[k - 1]['encodes']}  (arrows ×{scale:.2f})")
    fig.suptitle(
        "residue means μ_r (dots, labelled r) and the 11 readers' read directions P_k^T V_c / ‖V_c‖ "
        "(arrows from the origin, sign s_c applied; coloured by the reader's main residue)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_planes_arrows.png")
    plt.close(fig)
    nD = ((xv @ D) ** 2).sum(1) / (xv**2).sum(1)
    res["share_x_in_D"] = [float(nD.mean()), float(nD.min()), float(nD.max())]
    pc = np.array([[np.linalg.svd(P[k].T @ P[q], compute_uv=False).max() for q in P] for k in P])
    res["plane_cos"] = pc.tolist()
    per_plane = {k: r2(F[:, sel], predict(xv, P[k], V[:, sel], vnorm[sel])) for k in P}
    res["r2_per_plane"] = {
        str(k): dict(zip([names[i] for i in sel], v.tolist(), strict=True))
        for k, v in per_plane.items()
    }
    res["r2_D_sel"] = {names[i]: float(r2D[i]) for i in sel}
    po2 = np.argsort(main_r, kind="stable")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0))
    ks = list(P)
    colors = ["#1f4e9c", "#6c9bd1", "#b5442d", "#e39a7d", "#7a7a7a"]
    for ax, key, ttl in (
        (axs[0], "share", "‖P_k^T V_c‖² / ‖V_c‖²: share of the read vector in plane k"),
        (axs[1], "r2", "r² of f_c with the input projected onto plane k alone"),
    ):
        bottom = np.zeros(len(sel))
        for ci, k in enumerate(ks):
            if key == "share":
                vals = ((P[k].T @ (V[:, sel] / vnorm[sel])) ** 2).sum(0)[po2]
            else:
                vals = per_plane[k][po2]
            ax.bar(
                range(len(sel)),
                vals,
                bottom=bottom,
                color=colors[ci],
                label=f"k={k} ({reps[k - 1]['encodes']})",
            )
            bottom = bottom + vals
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels(
            [f"{names[sel[j]]} (r={int(main_r[j])})" for j in po2], rotation=90, fontsize=7
        )
        ax.set_title(ttl, fontsize=9)
    axs[1].plot(
        range(len(sel)), r2D[sel][po2], "k_", ms=14, mew=2, label="r²_D (all planes jointly)"
    )
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / "mod10_per_plane.png")
    plt.close(fig)
    pairs = {"1,3": [1, 3], "2,4": [2, 4]}
    for lab, ks in pairs.items():
        B = orth(np.concatenate([P[k] for k in ks], 1))
        res[f"r2_planes_{lab}"] = dict(
            zip(
                [names[i] for i in sel],
                r2(F[:, sel], predict(xv, B, V[:, sel], vnorm[sel])).tolist(),
                strict=True,
            )
        )
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
    fig.savefig(fig_dir / "mod10_planes_cos.png")
    plt.close(fig)

    # residue directions and the reader vectors
    mun = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    muD = mu @ D
    muDn = muD / np.linalg.norm(muD, axis=1, keepdims=True)
    cos_raw, cos_D = mun @ mun.T, muDn @ muDn.T
    off = ~np.eye(10, dtype=bool)
    diff = (np.arange(10)[:, None] - np.arange(10)[None, :]) % 10
    res["cos_mu_raw"] = [float(cos_raw[off].min()), float(cos_raw[off].max())]
    res["cos_mu_D"] = [float(cos_D[off].min()), float(cos_D[off].max())]
    is3 = (diff == 3) | (diff == 7)
    res["cos_mu_D_diff3"] = [
        float(cos_D[is3].min()),
        float(cos_D[is3].max()),
        float(cos_D[is3].mean()),
    ]
    res["cos_mu_D_other"] = [float(cos_D[off & ~is3].min()), float(cos_D[off & ~is3].max())]
    sv = np.linalg.svd(mu, compute_uv=False)
    U = np.linalg.svd(mu.T, full_matrices=False)[0]
    res["sv_mu_raw"] = sv.tolist()
    ev = [
        float(r2(F[:, sel], predict(xv, U[:, :n], V[:, sel], vnorm[sel])).mean()) if n else 0.0
        for n in range(11)
    ]
    res["ev_sel_vs_n_raw_svd"] = ev
    Vs = V[:, sel]
    Pd = D.T @ (sgn * Vs)
    res["sel_V_share_in_D"] = dict(
        zip([names[i] for i in sel], ((Pd**2).sum(0) / (Vs**2).sum(0)).tolist(), strict=True)
    )
    Pn = Pd / np.linalg.norm(Pd, axis=0)
    gram = Pn.T @ Pn
    res["sel_V_cos_in_D"] = gram.tolist()
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    axs[0].bar(range(1, 11), sv**2 / (sv**2).sum(), color="#2a5db0")
    axs[0].set_xlabel("singular direction j of M = [μ_0 … μ_9]")
    axs[0].set_ylabel("share of ‖M‖²")
    axs[0].set_title("SVD of the raw residue means")
    axs[1].plot(range(11), ev, "o-", color="#2a5db0")
    axs[1].set_xlabel("n (top-n singular directions of M)")
    axs[1].set_ylabel("mean r² over the selected readers")
    axs[1].set_ylim(0, 1)
    axs[1].set_title(f"selected readers ({len(sel)})")
    for ax, c, t in (
        (axs[2], cos_raw, "cos(μ_r, μ_r′)"),
        (axs[3], cos_D, "cos(Π_D μ_r, Π_D μ_r′)"),
    ):
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("r′")
        ax.set_ylabel("r")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_mu.png")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    po = np.argsort(main_r, kind="stable")
    axs[0].bar(
        range(len(sel)), [res["sel_V_share_in_D"][names[sel[j]]] for j in po], color="#2a5db0"
    )
    axs[0].set_xticks(range(len(sel)))
    axs[0].set_xticklabels([names[sel[j]] for j in po], rotation=90, fontsize=7)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("‖Π_D V_c‖² / ‖V_c‖²")
    im = axs[1].imshow(gram[np.ix_(po, po)], cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_xticks(range(len(sel)))
    axs[1].set_xticklabels([names[sel[j]] for j in po], rotation=90, fontsize=7)
    axs[1].set_yticks(range(len(sel)))
    axs[1].set_yticklabels([names[sel[j]] for j in po], fontsize=7)
    axs[1].set_title("cos(s_c Π_D V_c, s_c′ Π_D V_c′)")
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_reader_vectors.png")
    plt.close(fig)

    # the dual basis of the residue means: mu*_r . mu_r' = delta_rr', mu*_r in span{mu_r}
    Qm = orth(mu.T)  # (4096, 10) orthonormal basis of span{mu_r}
    dual = np.linalg.pinv(mu).T  # (10, 4096)
    assert np.allclose(dual @ mu.T, np.eye(10), atol=1e-8)
    Vm = Qm @ (Qm.T @ (sgn * V[:, sel]))  # Pi_M s_c V_c, (4096, |sel|)
    res["sel_V_share_in_M"] = dict(
        zip(
            [names[i] for i in sel],
            ((Vm**2).sum(0) / (V[:, sel] ** 2).sum(0)).tolist(),
            strict=True,
        )
    )

    def cosmat(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (
            Y / np.linalg.norm(Y, axis=0, keepdims=True)
        )

    def cos(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y))

    c_mu = cosmat(mu, Vm)  # (10, |sel|)
    c_dual = cosmat(dual, Vm)
    js = np.arange(len(sel))
    res["cos_V_mu_main"] = dict(
        zip([names[i] for i in sel], c_mu[main_r, js].tolist(), strict=True)
    )
    res["cos_V_dual_main"] = dict(
        zip([names[i] for i in sel], c_dual[main_r, js].tolist(), strict=True)
    )
    lobe = np.array([np.delete(prof[:, j], main_r[j]).min() for j in js])
    res["most_negative_lobe"] = dict(zip([names[i] for i in sel], lobe.tolist(), strict=True))
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.2, 1]})
    for ax, c, t in (
        (axs[0], c_mu, "cos(Π_M s_c V_c, μ_r)"),
        (axs[1], c_dual, "cos(Π_M s_c V_c, μ*_r)"),
    ):
        im = ax.imshow(c[:, po].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([f"{names[sel[j]]} (r={int(main_r[j])})" for j in po], fontsize=7)
        ax.set_xticks(range(10))
        ax.set_xlabel("r")
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8)
    w = 0.38
    axs[2].bar(
        np.arange(len(sel)) - w / 2, c_mu[main_r, js][po], w, color="#b5442d", label="with μ_r₀"
    )
    axs[2].bar(
        np.arange(len(sel)) + w / 2, c_dual[main_r, js][po], w, color="#1f4e9c", label="with μ*_r₀"
    )
    axs[2].set_xticks(range(len(sel)))
    axs[2].set_xticklabels([names[sel[j]] for j in po], rotation=90, fontsize=7)
    axs[2].set_ylim(0, 1.05)
    axs[2].set_title("cosine of Π_M s_c V_c at the reader's own residue r₀")
    axs[2].legend(frameon=False, fontsize=8, loc="lower right")
    axs[3].scatter(
        c_dual[main_r, js], lobe, c=[cmap(int(r)) for r in main_r], s=30, edgecolor="k", lw=0.5
    )
    for j in js:
        axs[3].annotate(
            names[sel[j]],
            (c_dual[main_r[j], j], lobe[j]),
            fontsize=7,
            xytext=(3, 2),
            textcoords="offset points",
        )
    axs[3].axhline(0, color="#888", lw=0.6)
    axs[3].set_xlabel("cos(Π_M s_c V_c, μ*_r₀)")
    axs[3].set_ylabel("most negative s_c m_c(r) / max |m_c|, r ≠ r₀")
    axs[3].set_title("side lobes appear when V_c leaves μ*_r₀")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_dual.png")
    plt.close(fig)

    # where the primal / dual difference lives: the shared direction u outside D, and inside D
    mubar = mu.mean(axis=0)
    u = mubar - D @ (D.T @ mubar)
    u = u / np.linalg.norm(u)
    res["share_mu_along_u"] = float(((mu @ u) ** 2 / (mu**2).sum(1)).mean())
    res["share_dual_along_u"] = float(((dual @ u) ** 2 / (dual**2).sum(1)).mean())
    Vs_ = sgn * V[:, sel]
    along = {
        "V_c": (u @ Vs_) / np.linalg.norm(Vs_, axis=0),
        "μ_r₀": (mu[main_r] @ u) / np.linalg.norm(mu[main_r], axis=1),
        "μ*_r₀": (dual[main_r] @ u) / np.linalg.norm(dual[main_r], axis=1),
    }
    VD = D @ (D.T @ Vs_)
    cD_mu = np.array([cos(VD[:, j], D @ (D.T @ mu[main_r[j]])) for j in js])
    cD_dual = np.array([cos(VD[:, j], D @ (D.T @ dual[main_r[j]])) for j in js])
    res["along_u"] = {
        k: dict(zip([names[i] for i in sel], v.tolist(), strict=True)) for k, v in along.items()
    }
    res["cos_in_D_mu_main"] = dict(zip([names[i] for i in sel], cD_mu.tolist(), strict=True))
    res["cos_in_D_dual_main"] = dict(zip([names[i] for i in sel], cD_dual.tolist(), strict=True))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    w = 0.27
    for t, (lab, v) in enumerate(along.items()):
        axs[0].bar(
            np.arange(len(sel)) + (t - 1) * w,
            v[po],
            w,
            label=lab,
            color=["#7a7a7a", "#b5442d", "#1f4e9c"][t],
        )
    axs[0].set_title("cosine with the shared direction u (outside D)")
    axs[1].bar(
        np.arange(len(sel)) - w / 2,
        cD_mu[po],
        w,
        color="#b5442d",
        label="cos(Π_D s_c V_c, Π_D μ_r₀)",
    )
    axs[1].bar(
        np.arange(len(sel)) + w / 2,
        cD_dual[po],
        w,
        color="#1f4e9c",
        label="cos(Π_D s_c V_c, Π_D μ*_r₀)",
    )
    axs[1].set_ylim(0.5, 1.02)
    axs[1].set_title("inside D")
    for ax in axs:
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels(
            [f"{names[sel[j]]} (r={int(main_r[j])})" for j in po], rotation=90, fontsize=7
        )
    axs[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.75))
    axs[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_dual_split.png")
    plt.close(fig)

    # 3) embedding
    z = embedding_inputs()  # (400, 4096)
    eps = np.stack([z[rr == r].mean(axis=0) for r in range(10)])
    rk = np.arange(10)
    De = orth(
        np.concatenate(
            [
                np.stack([p.real, p.imag], 1) if k < 5 else p.real[:, None]
                for k in range(1, 6)
                for p in [eps.T @ np.exp(-2j * np.pi * k * rk / 10)]
            ],
            1,
        )
    )
    PvV = V @ Pv @ V.T
    res["principal_cos_D_vs_emb_projected"] = principal_cosines(D, PvV @ De).tolist()
    res["principal_cos_D_vs_emb_full"] = principal_cosines(D, De).tolist()
    res["share_emb_D_in_spanV"] = float(np.linalg.norm(Q.T @ De) ** 2 / De.shape[1])
    obs = (
        (mu @ D) @ (D.T @ V[:, sel]) / vnorm[sel]
    )  # (10, |sel|) planes part of the residue profile
    pred = (eps @ De) @ (De.T @ V[:, sel]) / vnorm[sel]  # V_c . Pi_De eps_r / ||V_c||
    cosp = (obs * pred).sum(0) / np.linalg.norm(obs, axis=0) / np.linalg.norm(pred, axis=0)
    res["emb_profile_cos"] = dict(zip([names[i] for i in sel], cosp.tolist(), strict=True))
    res["emb_profile_norm_ratio"] = dict(
        zip(
            [names[i] for i in sel],
            (np.linalg.norm(obs, axis=0) / np.linalg.norm(pred, axis=0)).tolist(),
            strict=True,
        )
    )
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    pcs = res["principal_cos_D_vs_emb_projected"]
    axs[0].bar(range(1, len(pcs) + 1), pcs, color="#2a5db0")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("principal angle index")
    axs[0].set_ylabel("cos")
    axs[0].set_title("D vs Π_V D_ε")
    axs[1].bar(range(len(sel)), cosp, color="#2a5db0")
    axs[1].set_xticks(range(len(sel)))
    axs[1].set_xticklabels([names[i] for i in sel], rotation=90, fontsize=7)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("cos over r: observed vs embedding-predicted profile")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_embedding.png")
    plt.close(fig)

    # 4) the other readers after removing D
    others = np.setdiff1d(np.arange(K), sel)
    W = predict(xv, D, V[:, others], vnorm[others])
    Fo = F[:, others]
    R = Fo - W
    change = np.linalg.norm(W, axis=0) / np.linalg.norm(Fo, axis=0)
    oo = np.argsort(-change)
    res["others_change_quantiles"] = np.quantile(change, [0.5, 0.9, 1.0]).tolist()
    res["others_change_top"] = {names[others[j]]: float(change[j]) for j in oo[:10]}
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(others.size), change[oo], color="#2a5db0")
    ax.set_xticks(range(40))
    ax.set_xticklabels([names[others[j]] for j in oo[:40]], rotation=90, fontsize=6)
    ax.set_ylabel("‖g_c^D‖ / ‖f_c‖")
    ax.set_title(f"other readers ({others.size}): size of the part read from D")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_others_change.png")
    plt.close(fig)
    fig, axs = plt.subplots(12, 2, figsize=(12, 15), sharex=True)
    for axx, j in zip(axs.T.ravel(), oo[:24], strict=True):
        axx.axhline(0, color="#ddd", lw=0.6)
        axx.plot(A, Fo[:, j], lw=0.7, color="#bbb")
        axx.plot(A, R[:, j], lw=0.8, color="#2a5db0")
        axx.set_ylabel(names[others[j]], fontsize=8)
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
    fig.suptitle(
        "other block-0 readers, ordered by change: f_c (grey) and ρ_c = f_c − g_c^D (blue)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_others_residual.png")
    plt.close(fig)
    (args.out / "mod10_page.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("profiles", "sel_V_cos_in_D", "plane_cos")},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
