"""The layer-15 MLP input at `=` in the subspace read by the three op switches (alive-only model).

    python -m param_decomp.arith_repr.vectors.flip_op_readers   # -> OUT/flip/op_readers.png
    (needs OUT/flip/opflow/h_eq.npz from `flip_opflow capture`)

xin = RMSNorm(stream entering L15's MLP) * ln2 gain, per prompt (all 20000 prompts), position `=`; the
stream is rebuilt from the per-prompt inners at `=` (`flip_opflow.stream_at`). The three readers are
L15.gate.c72, L15.up.c117, L15.up.c34; their inner activations are h_c = xin . V_c (no bias), so
h_c = 0 on the plane through the origin orthogonal to V_c. S = span(V_72, V_117, V_34) (3
dimensions); every V_c lies in S, so h_c / |V_c| is the coordinate of xin's projection onto S along
V_c / |V_c|. Basis of S: e1 = the op flag (mean xin on add - mean xin on sub) projected onto S,
normalized; e2, e3 = the directions of S orthogonal to e1, ordered by the variance of xin along them.
* Top: the 3D cloud of xin's projections onto S (blue add, orange sub), with each dot's
  shadow on the floor of the box (grey); arrows from the origin along V_c / |V_c|, all drawn at the
  same length.
* Bottom: histograms of h_c per operation."""

import json
from typing import Any, cast

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from param_decomp.arith_repr.isa.components_model import ComponentsModel
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_opflow import OUT as OPFLOW
from param_decomp.arith_repr.vectors.flip_opflow import stream_at

L = 15
READERS = ("L15.gate.c72", "L15.up.c117", "L15.up.c34")
ADD, SUB, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a63"
SHADOW = "#b9b8b0"
RC = {"L15.gate.c72": "#1baf7a", "L15.up.c117": "#4a3aa7", "L15.up.c34": "#c2185b"}


def main() -> None:
    cm = ComponentsModel()
    H = dict(np.load(OPFLOW / "h_eq.npz"))
    x = stream_at(cm, H, 2 * L + 1)
    xin = x / np.sqrt((x * x).mean(-1, keepdims=True) + cm.eps) * np.asarray(cm.ln2[L], np.float64)
    op = cm.op
    cols = [cm.column(L, n)[1] for n in READERS]
    kinds = [n.split(".")[1] for n in READERS]
    V = np.stack(
        [np.asarray(cm.sites[L][k].V[:, j], np.float64) for k, j in zip(kinds, cols, strict=True)],
        1,
    )
    h = xin @ V
    hc = np.stack([H[f"{L}.{k}"][:, j] for k, j in zip(kinds, cols, strict=True)], 1)
    print("max |h - captured h|:", float(np.abs(h - hc).max()), flush=True)
    stats = {n: [float(h[op == o, i].mean()) for o in (0, 1)] for i, n in enumerate(READERS)}
    print("mean h add / sub:", {n: [round(v, 2) for v in s] for n, s in stats.items()}, flush=True)
    Q, _ = np.linalg.qr(V)  # (d, 3) orthonormal basis of S
    flag = xin[op == 0].mean(0) - xin[op == 1].mean(0)
    fS = Q.T @ flag
    share_flag = float(np.linalg.norm(fS) ** 2 / np.linalg.norm(flag) ** 2)
    e1 = fS / np.linalg.norm(fS)
    rest = np.linalg.svd(np.eye(3) - np.outer(e1, e1))[0][:, :2]  # (3, 2) complement of e1
    c = xin @ Q
    _, R = np.linalg.eigh(np.cov(((c - c.mean(0)) @ rest).T))
    rest = rest @ R[:, ::-1]
    B = np.stack([e1, rest[:, 0], rest[:, 1]], 1)  # (3, 3): coords in S of e1, e2, e3
    P = c @ B  # (N, 3) coordinates of xin on e1, e2, e3
    Vs = (Q.T @ (V / np.linalg.norm(V, axis=0))).T @ B  # unit V_c in (e1, e2, e3)
    assert np.allclose(np.linalg.norm(Vs, axis=1), 1, atol=1e-6)
    cos = {
        n: float(V[:, i] @ flag / np.linalg.norm(V[:, i]) / np.linalg.norm(flag))
        for i, n in enumerate(READERS)
    }

    arrow = 0.3 * float(np.ptp(P, axis=0).max())
    tips = np.concatenate(
        [P, np.zeros((1, 3)), 1.15 * arrow * Vs]
    )  # the box holds points, origin, arrow tips
    lo, hi = tips.min(0), tips.max(0)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    idx = np.random.default_rng(0).permutation(len(op))
    fig = plt.figure(figsize=(11, 11.5))
    for vi, (elev, azim) in enumerate(((22, -62),)):
        ax = cast(Any, fig.add_axes((0.02, 0.29, 0.92, 0.70), projection="3d"))
        ax.scatter(P[idx, 0], P[idx, 1], np.full(len(idx), lo[2]), s=1, c=SHADOW, alpha=0.15,
                   linewidths=0, depthshade=False)  # fmt: skip
        ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=1.5, c=np.where(op[idx] == 0, ADD, SUB),
                   alpha=0.35, linewidths=0, depthshade=False)  # fmt: skip
        for o, colr, nm in ((0, ADD, "add (+), mean"), (1, SUB, "sub (−), mean")):
            mu = P[op == o].mean(0)
            ax.scatter(
                *mu,
                s=70,
                color=colr,
                edgecolors="white",
                linewidths=1.5,
                depthshade=False,
                label=nm,
            )
            ax.plot([mu[0], mu[0]], [mu[1], mu[1]], [lo[2], mu[2]], color=colr, lw=0.8, ls=":")
        ax.scatter(0, 0, 0, s=25, color=INK, depthshade=False)
        ax.scatter(0, 0, lo[2], s=12, color=SHADOW, depthshade=False)
        ax.plot([0, 0], [0, 0], [lo[2], 0], color=MUTED, lw=0.8, ls=":")
        for i, n in enumerate(READERS):
            v = arrow * Vs[i]
            ax.quiver(0, 0, 0, *v, color=RC[n], lw=2.2, arrow_length_ratio=0.12)
            ax.plot([0, v[0]], [0, v[1]], [lo[2], lo[2]], color=RC[n], lw=1, alpha=0.5)
            ax.text(*(1.08 * v), n.replace("L15.", ""), fontsize=9, color=INK)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("e1: op flag within S", fontsize=9, color=INK)
        ax.set_ylabel("e2: most remaining variance", fontsize=9, color=INK)
        ax.set_zlabel("e3: least remaining variance", fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        if vi == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.06, 0.92), fontsize=9, frameon=False)
    for i, n in enumerate(READERS):
        ax = fig.add_axes((0.04 + 0.33 * i, 0.06, 0.28, 0.17))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        bins = np.linspace(h[:, i].min(), h[:, i].max(), 80)
        for o, colr, nm in ((0, ADD, "add"), (1, SUB, "sub")):
            ax.hist(h[op == o, i], bins=bins, color=colr, alpha=0.6, label=nm)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{n}: h_c\ncos(V_c, op flag) {cos[n]:+.2f}", fontsize=9, color=INK)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.text(0.5, 0.005, "layer-15 MLP input at `=`, alive-only model, 20000 prompts, projected onto S = span of the three "
             "readers' V (black dot: origin; grey: shadows on the floor).\nArrows: V_c / |V_c|, common length. The op flag "
             f"has {share_flag:.2f} of its energy in S.", ha="center", fontsize=8, color=MUTED)  # fmt: skip
    fig.savefig(DIR / "op_readers.png", dpi=140)
    (DIR / "op_readers.json").write_text(
        json.dumps({"mean_h": stats, "cos_flag": cos, "flag_share_in_S": share_flag}, indent=1)
    )
    print("saved", DIR / "op_readers.png", "flag share in S", round(share_flag, 3), flush=True)


if __name__ == "__main__":
    main()
