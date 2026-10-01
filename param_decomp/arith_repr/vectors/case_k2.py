"""Period 50 at the L18 MLP, worked through for the report: how a handful of neurons turn a's and b's
mod-50 circles into the a + b circle, why the a - b by-product of each product cancels, and on which
prompts each gate is open.

    python -m param_decomp.arith_repr.vectors.case_k2 compute   # -> OUT/case_k2.npz (+ stdout)
    python -m param_decomp.arith_repr.vectors.case_k2 figures   # -> OUT/figs/k2_*.png

Everything is at position `=` on the original model. Harmonic k = 2 of a quantity q is the pair
cos(2 pi q / 50), sin(2 pi q / 50); its complex coefficient F = mean x e^{-2 pi i q / 50} (common.py).
A plane is spanned by C = Re F and S = -Im F (the harmonic part of the class mean of residue q is
2 (C cos + S sin)); `plane` gives it an orthonormal basis in which residue q sits at angle 2 pi q / 50."""

import sys
from typing import Any

import matplotlib
import ml_dtypes
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.vectors.common import DATASET, OUT, comp_table, line_dft, load_uv
from param_decomp.arith_repr.vectors.periods_summary import unit_of

L, K, T = 18, 2, 50
IN_PT, OUT_PT = (
    2 * L + 1,
    2 * L + 2,
)  # stream the L18 MLP reads (after L18 attention) / after its add
N_TOP = 7
LINE_IX = {"a": 0, "b": 1, "sum": 2, "diff": 3}
FILE = OUT / "case_k2.npz"


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def plane(F: np.ndarray) -> np.ndarray:
    """(2, d) orthonormal basis of the plane of a complex code F: e1 along C = Re F, e2 along the part
    of S = -Im F orthogonal to it."""
    C, S = np.real(F), -np.imag(F)
    e1 = C / np.linalg.norm(C)
    s = S - (S @ e1) * e1
    return np.stack([e1, s / np.linalg.norm(s)])


def residues(line: str) -> np.ndarray:
    """(10000,) value mod 50 of a quantity on one operation's grid."""
    i = np.arange(10000)
    a, b = i // 100 + 1, i % 100 + 1
    q = {"a": a, "b": b, "sum": a + b, "diff": a - b}[line]
    return np.mod(q, T)


def class_means(x: np.ndarray, line: str) -> np.ndarray:
    """(T, ...) mean of x over the prompts of each residue of `line`, minus the grid mean."""
    r = residues(line)
    m = np.stack([x[r == q].mean(0) for q in range(T)])
    return m - x.mean(0)


def compute() -> None:
    assert ml_dtypes.bfloat16  # numpy must know bfloat16 before safetensors is read
    Wd = Weights().get(f"model.layers.{L}.mlp.down_proj.weight").astype(np.float32)  # (4096, 14336)
    g_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")
    resid = np.load(DATASET / "original/resid.npy", mmap_mode="r")
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    comps = comp_table()
    owner, dcols = unit_of(comps, L)
    gcols = np.flatnonzero((comps["layer"] == L) & (comps["kind"] == "gate"))
    ucols = np.flatnonzero((comps["layer"] == L) & (comps["kind"] == "up"))
    _, Ug = load_uv(comps, gcols)
    _, Uu = load_uv(comps, ucols)
    Ug, Uu = np.stack(Ug), np.stack(Uu)  # (n_comp, 14336)
    inner = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    wn2 = (Wd**2).sum(0)
    out: dict[str, Any] = {}
    top = np.empty(0, int)
    for o, op in enumerate(("add", "sub")):
        sl = slice(o * 10000, (o + 1) * 10000)
        g = np.asarray(g_mm[L, sl, 4], np.float32)
        u = np.asarray(u_mm[L, sl, 4], np.float32)
        act = silu(g) * u
        coef = {}
        for nm, x in (("g", g), ("u", u), ("act", act)):
            F = np.empty((4, x.shape[1]), np.complex64)
            for i in range(0, x.shape[1], 2048):
                F[:, i : i + 2048] = line_dft(x[:, i : i + 2048].reshape(100, 100, -1))[:, K - 1]
            coef[nm] = F  # (4 lines, 14336)
        A = coef["act"]
        Tw = (Wd @ A.T).T  # (4 lines, 4096): the MLP's write on each line at k = 2
        coh = (np.abs(Tw) ** 2).sum(1)
        inc = (wn2[None] * np.abs(A) ** 2).sum(1)
        res = "sum" if o == 0 else "diff"
        share = (np.conj(Tw[LINE_IX[res]]) @ Wd * A[LINE_IX[res]]).real / coh[LINE_IX[res]]
        if o == 0:
            top = np.argsort(-share)[:N_TOP]
        At = A[:, top]
        Tt = (Wd[:, top] @ At.T).T
        print(f"== {op}: k = 2 write of the L18 MLP (all neurons | top {N_TOP})")
        for nm, li in LINE_IX.items():
            print(f"  {nm:4s} |T|^2 {coh[li]:.3f}  sum_n |w_n A_n|^2 {inc[li]:.3f}  ratio {coh[li] / inc[li]:.2f}"
                  f" | top: |T|^2 {(np.abs(Tt[li]) ** 2).sum():.3f}")  # fmt: skip
        print(f"  top {N_TOP} carry {share[top].sum():.3f} of the {res} write")
        # planes: operand codes in the stream the MLP reads, result codes after its add, and the
        # MLP write's own planes
        F_in = fre[IN_PT, 3, o, :, K - 1].astype(np.float32) + 1j * fim[IN_PT, 3, o, :, K - 1]
        F_out = fre[OUT_PT, 3, o, :, K - 1].astype(np.float32) + 1j * fim[OUT_PT, 3, o, :, K - 1]
        B = {"a": plane(F_in[0]), "b": plane(F_in[1]), "res": plane(F_out[LINE_IX[res]]),
             "w_sum": plane(Tw[2]), "w_diff": plane(Tw[3])}  # fmt: skip
        x_in = np.asarray(resid[IN_PT, sl, 4], np.float32)
        x_out = np.asarray(resid[OUT_PT, sl, 4], np.float32)
        out[f"xa_o{o}"] = (x_in - x_in.mean(0)) @ B["a"].T
        out[f"xb_o{o}"] = (x_in - x_in.mean(0)) @ B["b"].T
        out[f"xres_o{o}"] = (x_out - x_out.mean(0)) @ B["res"].T
        for nm in ("res", "w_sum", "w_diff"):
            Pn = B[nm] @ Wd  # (2, 14336): every neuron's down vector in the plane
            out[f"P_{nm}_o{o}"] = Pn[:, top].T
            out[f"mlp_{nm}_o{o}"] = act @ Pn.T  # (10000, 2) the whole MLP write in the plane
        out[f"g_o{o}"], out[f"u_o{o}"] = g[:, top].T, u[:, top].T
        for nm in ("g", "u", "act"):
            out[f"{nm}_coef_o{o}"] = coef[nm][:, top].T  # (7, 4 lines)
        out[f"coh_o{o}"], out[f"inc_o{o}"], out[f"share_o{o}"] = coh, inc, share[top]
        out[f"cohtop_o{o}"] = (np.abs(Tt) ** 2).sum(1)
        out[f"inctop_o{o}"] = (wn2[top][None] * np.abs(At) ** 2).sum(1)
        # the alive gate / up components feeding each top neuron: U_c[n] x (inner's k = 2 coefficient)
        for kind, cols, Uk in (("gate", gcols, Ug), ("up", ucols, Uu)):
            h = np.asarray(inner[sl, 4][:, cols], np.float32)
            Hc = line_dft(h.reshape(100, 100, -1))[:, K - 1]  # (4, n_comp)
            names, fracs = [], []
            for n in top:
                c = Uk[:, n][None] * Hc[:2]  # (2, n_comp): a and b lines
                tot = np.abs(coef["g" if kind == "gate" else "u"][:2, n]) ** 2
                j = np.argsort(-(np.abs(c) ** 2).sum(0))[:3]
                names.append([int(comps["cidx"][cols[i]]) for i in j])
                fracs.append([float((np.abs(c[:, i]) ** 2).sum() / tot.sum()) for i in j])
            out[f"{kind}_feed_o{o}"] = np.array(names)
            out[f"{kind}_feedfrac_o{o}"] = np.array(fracs)
    out["top"] = top
    out["unit"] = np.array([comps["cidx"][dcols[owner[n]]] if owner[n] >= 0 else -1 for n in top])
    np.savez(FILE, **out)
    for j, n in enumerate(top):
        print(f"neuron {n} (down c{out['unit'][j]}): gate fed by {out['gate_feed_o0'][j]} "
              f"{np.round(out['gate_feedfrac_o0'][j], 2)}, up by {out['up_feed_o0'][j]} "
              f"{np.round(out['up_feedfrac_o0'][j], 2)}")  # fmt: skip


# ---------------------------------------------------------------- figures

INK, INK2, GRAY, FAINT = "#1d1c1a", "#52514e", "#b9b8b3", "#e6e5e0"
C_SUM, C_DIFF, C_A, C_B = "#1baf7a", "#eda100", "#2a78d6", "#eb6834"
NEURON_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#d6336c", "#795548", "#00838f")


def hue(q: np.ndarray | float) -> Any:
    return matplotlib.colormaps["twilight_shifted"](np.asarray(q) / T)


def peak(F: complex | np.ndarray) -> np.ndarray:
    """Residue mod 50 at which the harmonic-2 wave with coefficient F peaks."""
    return np.mod(-np.angle(F) * T / (2 * np.pi), T)


def save(fig: Any, name: str) -> None:
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"figs/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def ring(ax: Any, pts: np.ndarray, cloud: np.ndarray | None, line: str, title: str,
         labels: bool = True) -> float:  # fmt: skip
    """Class means of `line` (T, 2) as a coloured ring, over an optional cloud of single prompts."""
    if cloud is not None:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(cloud), 1500, replace=False)
        ax.scatter(*cloud[idx].T, s=3, c=hue(residues(line)[idx]), alpha=0.25, lw=0)
    ax.plot(*np.r_[pts, pts[:1]].T, color=GRAY, lw=0.6, zorder=1)
    ax.scatter(*pts.T, s=22, c=hue(np.arange(T)), edgecolor="white", lw=0.4, zorder=2)
    m = float(np.abs(pts).max())
    if labels:
        for q in range(0, T, 5):
            ax.text(*(pts[q] * (1 + 0.17 * m / max(np.linalg.norm(pts[q]), 1e-9))), str(q),
                    fontsize=7, color=INK2, ha="center", va="center")  # fmt: skip
    ax.set_aspect("equal")
    ax.axhline(0, color=FAINT, lw=0.6, zorder=0)
    ax.axvline(0, color=FAINT, lw=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK)
    return m


def arrows(ax: Any, Z: np.ndarray, m: float, labels: list[str], dashed: bool = False) -> None:
    """Arrows from the origin at complex positions Z, scaled so the longest reaches 0.85 m."""
    s = 0.85 * m / np.abs(Z).max()
    for j, z in enumerate(Z * s):
        ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=NEURON_COLORS[j], lw=1.6,
                                    ls="--" if dashed else "-", shrinkA=0, shrinkB=0))  # fmt: skip
        if labels[j]:
            ax.text(z.real * 1.08, z.imag * 1.08, labels[j], fontsize=6.5, color=NEURON_COLORS[j],
                    ha="center", va="center", zorder=4)  # fmt: skip


def fig_circles(z: dict[str, np.ndarray]) -> None:
    """The three circles of the case study (add), with the neurons' reads and writes drawn on them."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ga, ub = z["g_coef_o0"][:, 0], z["u_coef_o0"][:, 1]
    specs = (
        ("a", z["xa_o0"], "a mod 50 at '=', in the stream the L18 MLP reads\narrows: what each gate reads of a",
         np.conj(ga), [f"n{n}" for n in z["top"]]),
        ("b", z["xb_o0"], "b mod 50, same stream\narrows: what each up reads of b", np.conj(ub),
         [f"n{n}" for n in z["top"]]),
        ("sum", z["xres_o0"], "(a + b) mod 50, right after the L18 MLP's add\n"
         "arrows: each neuron's down vector; label = the a + b residue\nwhere the neuron's value peaks", None, [f"n{n}" for n in z["top"]]),
    )  # fmt: skip
    for ax, (line, x, title, zs, lab) in zip(axes, specs, strict=True):
        pts = class_means(x, line)
        m = ring(ax, pts, x, line, title)
        if zs is None:
            P = z["P_res_o0"]  # (7, 2): down vectors in the result plane
            zs = P[:, 0] + 1j * P[:, 1]
            pk = peak(z["act_coef_o0"][:, LINE_IX["sum"]])
            lab = [f"n{n}: {p:.0f}" for n, p in zip(z["top"], pk, strict=True)]
        arrows(ax, zs, m, lab)
        ax.set_xlim(-1.5 * m, 1.5 * m)
        ax.set_ylim(-1.5 * m, 1.5 * m)
    fig.suptitle("Period 50 at '=' (addition). Dots: mean stream position of the prompts with each residue, "
                 "projected on the code's plane (colour = residue); faint: single prompts", fontsize=9.5,
                 color=INK2)  # fmt: skip
    save(fig, "k2_circles")


def fig_gates(z: dict[str, np.ndarray]) -> None:
    """(a, b) grids of four of the neurons: gate pre-activation with its off region, up, and product."""
    unit, top = z["unit"], z["top"]
    rows = [j for j in range(len(top)) if unit[j] in (12, 21, 27, 36)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.4 * len(rows)))
    ext = (0.5, 100.5, 0.5, 100.5)
    for r, j in enumerate(rows):
        g = z["g_o0"][j].reshape(100, 100)
        u = z["u_o0"][j].reshape(100, 100)
        s = silu(g)
        panels = ((g, "gate pre-activation g"), (s, "silu(g): the gate's output"), (u, "up u"),
                  (s * u, "neuron value silu(g)·u"))  # fmt: skip
        for c, (img, lab) in enumerate(panels):
            ax = axes[r, c]
            v = float(np.abs(img).max())
            ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
            if c < 2:
                ax.contourf(np.arange(1, 101), np.arange(1, 101), g, levels=[-1e9, 0], colors="none",
                            hatches=["////"])  # fmt: skip
                ax.contour(
                    np.arange(1, 101), np.arange(1, 101), g, levels=[0], colors=INK, linewidths=0.8
                )
            ax.set_title(f"n{top[j]} (read by down c{unit[j]}): {lab}", fontsize=7.5, color=INK)
            ax.set_xlabel("b", fontsize=7)
            ax.set_ylabel("a", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[r, 0].text(2, 94, f"open on {(g > 0).mean():.0%} of prompts", fontsize=7,
                        color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))  # fmt: skip
    fig.suptitle("Where the gates are open (addition prompts, a on the vertical axis, b on the horizontal). "
                 "Hatched = g < 0: the gate is (mostly) closed", fontsize=10, color=INK2)  # fmt: skip
    fig.tight_layout()
    save(fig, "k2_gates")


def textbook() -> dict[str, Any]:
    """The four textbook neurons on the full grid: products of cos / sin of a and b, written along
    +e1, -e1, +e2, +e2 of a plane (period 50)."""
    i = np.arange(10000)
    ta, tb = 2 * np.pi * (i // 100 + 1) / T, 2 * np.pi * (i % 100 + 1) / T
    return {
        "cos a · cos b → +e1": (np.cos(ta) * np.cos(tb), np.array([1.0, 0.0])),
        "sin a · sin b → −e1": (np.sin(ta) * np.sin(tb), np.array([-1.0, 0.0])),
        "cos a · sin b → +e2": (np.cos(ta) * np.sin(tb), np.array([0.0, 1.0])),
        "sin a · cos b → +e2": (np.sin(ta) * np.cos(tb), np.array([0.0, 1.0])),
    }


def fig_textbook() -> None:
    nrn = textbook()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
    for r, line in enumerate(("sum", "diff")):
        total = np.zeros((T, 2))
        for c, (lab, (v, w)) in enumerate(nrn.items()):
            pts = class_means(v, line)[:, None] * w[None]  # (T, 2): this neuron's write per class
            total += pts
            ax = axes[r, c]
            ring(ax, pts + 1e-9, None, line, lab if r == 0 else "", labels=False)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
        ring(
            axes[r, 4],
            total + 1e-9,
            None,
            line,
            "all four together" if r == 0 else "",
            labels=r == 0,
        )
        axes[r, 4].set_xlim(-1.3, 1.3)
        axes[r, 4].set_ylim(-1.3, 1.3)
        axes[r, 0].text(-1.5, 0, {"sum": "prompts grouped\nby (a + b) mod 50",
                                   "diff": "prompts grouped\nby (a − b) mod 50"}[line],
                        fontsize=9, color=INK, ha="right", va="center")  # fmt: skip
    fig.suptitle("Textbook circuit: four products, each written along one direction. Each dot is the mean write "
                 "of the prompts with one residue (colour).\nEvery single product draws a line through the "
                 "origin for a + b AND for a − b; together the a + b lines make a circle and the a − b lines "
                 "cancel.", fontsize=9.5, color=INK2)  # fmt: skip
    save(fig, "k2_textbook")


def _plain(ax: Any, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=FAINT, lw=0.6, zorder=0)
    ax.axvline(0, color=FAINT, lw=0.6, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK)


def fig_cancel(z: dict[str, np.ndarray]) -> None:
    """Measured: each neuron alone draws a line for a + b and for a - b; together they draw the a + b
    circle and cancel the a - b one."""
    top = z["top"]
    act = silu(z["g_o0"]) * z["u_o0"]  # (7, 10000)
    P = z["P_w_sum_o0"]  # (7, 2) down vectors in the plane of the MLP's a + b write
    zoom = 2.0
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    lim = float(np.abs(class_means(z["mlp_w_sum_o0"], "sum")).max())
    for r, line in enumerate(("sum", "diff")):
        q = "a + b" if line == "sum" else "a − b"
        each = [class_means(act[j], line)[:, None] * P[j][None] for j in range(len(top))]
        ax = axes[r, 0]
        for j, pts in enumerate(each):
            ax.plot(*pts.T, color=NEURON_COLORS[j], lw=1.0, alpha=0.7)
            ax.scatter(*pts.T, s=7, c=hue(np.arange(T)), lw=0, zorder=3)
            end = pts[np.argmax(np.linalg.norm(pts, axis=1))]
            ax.text(
                *(end * 1.18),
                f"n{top[j]}",
                fontsize=7,
                color=NEURON_COLORS[j],
                ha="center",
                va="center",
            )
        _plain(ax, f"each of the {len(top)} neurons alone (zoomed ×{zoom:.0f})")
        ax.set_xlim(-1.3 * lim / zoom, 1.3 * lim / zoom)
        ax.set_ylim(-1.3 * lim / zoom, 1.3 * lim / zoom)
        ring(
            axes[r, 1],
            np.sum(each, 0),
            None,
            line,
            f"the {len(top)} neurons added",
            labels=line == "sum",
        )
        if line == "sum":
            ring(axes[r, 2], class_means(z["mlp_w_sum_o0"], line), None, line,
                 "the whole L18 MLP (14,336 neurons)", labels=True)  # fmt: skip
        else:
            ring(axes[r, 2], class_means(z["mlp_w_diff_o0"], line), None, line,
                 "the whole L18 MLP, in the plane where\nits a − b write is largest", labels=False)  # fmt: skip
        for c in (1, 2):
            axes[r, c].set_xlim(-1.3 * lim, 1.3 * lim)
            axes[r, c].set_ylim(-1.3 * lim, 1.3 * lim)
        axes[r, 0].text(-0.08, 0.5, f"prompts grouped\nby ({q}) mod 50", transform=axes[r, 0].transAxes,
                        fontsize=10, ha="right", va="center", color=INK)  # fmt: skip
    fig.suptitle("Measured, addition prompts, period 50: the L18 MLP's write, projected on a plane; each dot is the "
                 "mean write of the prompts with one residue.\nColumns 1-2: the plane of the MLP's a + b write. "
                 "Columns 2-3 share one scale; column 1 is zoomed.", fontsize=9.5, color=INK2)  # fmt: skip
    save(fig, "k2_cancel")


def fig_phasors(z: dict[str, np.ndarray]) -> None:
    """Each neuron's contribution to the circle as an arrow (amplitude and angle), added head to tail."""
    P = z["P_w_sum_o0"]
    Pc = P[:, 0] + 1j * P[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    walks = {
        "sum": Pc * z["act_coef_o0"][:, LINE_IX["sum"]],
        "diff": Pc * z["act_coef_o0"][:, LINE_IX["diff"]],
    }
    lim = max(float(np.abs(np.cumsum(w)).max()) for w in walks.values())
    lim = max(lim, float(np.abs(walks["sum"]).max()))
    for ax, (line, ph) in zip(axes, walks.items(), strict=True):
        q = "a + b" if line == "sum" else "a − b"
        pos = np.r_[0, np.cumsum(ph)]
        for j in range(len(ph)):
            ax.annotate("", xy=(pos[j + 1].real, pos[j + 1].imag), xytext=(pos[j].real, pos[j].imag),
                        arrowprops=dict(arrowstyle="-|>", color=NEURON_COLORS[j], lw=1.8, shrinkA=0, shrinkB=0))  # fmt: skip
            mid = (pos[j] + pos[j + 1]) / 2
            ax.text(
                mid.real,
                mid.imag + 0.04 * lim,
                f"n{z['top'][j]}",
                fontsize=7,
                color=NEURON_COLORS[j],
                ha="center",
            )
        ax.plot([0, pos[-1].real], [0, pos[-1].imag], color=INK, lw=1.0, ls="--")
        ax.scatter([0], [0], s=12, color=INK)
        _plain(
            ax, f"{q}: sum of arrows = {abs(pos[-1]):.2f}, sum of lengths = {np.abs(ph).sum():.2f}"
        )
        ax.set_xlim(-0.1 * lim, 1.1 * lim)
        ax.set_ylim(-0.3 * lim, 0.2 * lim)
    fig.suptitle("Each neuron's contribution to the circle, as an arrow: length = how much it writes, angle = how far "
                 "its write direction is\nrotated from where its own wave peaks. Aligned arrows add up (a + b); "
                 "arrows pointing different ways cancel (a − b).", fontsize=9.5, color=INK2)  # fmt: skip
    save(fig, "k2_phasors")


def table(z: dict[str, np.ndarray]) -> None:
    """The numbers of the report's case-study table."""
    g, u, a = z["g_coef_o0"], z["u_coef_o0"], z["act_coef_o0"]
    P = z["P_w_sum_o0"]
    Pc = P[:, 0] + 1j * P[:, 1]
    spoke = np.mod(np.angle(Pc) * T / (2 * np.pi), T)
    print("unit neuron | gate: a (amp@peak) b | up: a b | value: a+b a-b | spoke | share | open")
    for j, n in enumerate(z["top"]):
        f = lambda F: f"{2 * abs(F):.2f}@{float(peak(F)):4.1f}"  # noqa: E731
        print(
            f"c{z['unit'][j]:<3d} n{n:<5d} | {f(g[j, 0])} {f(g[j, 1])} | {f(u[j, 0])} {f(u[j, 1])} | "
            f"{f(a[j, 2])} {f(a[j, 3])} | {spoke[j]:4.1f} | {z['share_o0'][j]:.3f} | {(z['g_o0'][j] > 0).mean():.2f}"
        )
    for o in range(2):
        print(
            f"op {o}: coherent",
            np.round(z[f"coh_o{o}"], 3),
            "incoherent",
            np.round(z[f"inc_o{o}"], 3),
            "top7 coh",
            np.round(z[f"cohtop_o{o}"], 3),
            "top7 inc",
            np.round(z[f"inctop_o{o}"], 3),
        )
    for line in ("sum", "diff"):
        ph = Pc * a[:, LINE_IX[line]]
        print(
            line,
            "phasor angles (deg):",
            np.round(np.degrees(np.angle(ph))).astype(int),
            "lengths",
            np.round(np.abs(ph), 3),
            "sum",
            round(float(abs(ph.sum())), 3),
        )


def fig_planes(z: dict[str, np.ndarray]) -> None:
    """Why the a - b leftover lands outside the a + b plane: the pair rule (sum vs difference of two
    write vectors), a 3-D toy, and the measured shares."""
    assert ml_dtypes.bfloat16
    top = z["top"]
    W = Weights().get(f"model.layers.{L}.mlp.down_proj.weight").astype(np.float32)[:, top]
    A = z["act_coef_o0"]
    Tp, Tm = W @ A[:, LINE_IX["sum"]], W @ A[:, LINE_IX["diff"]]
    Bp = plane(Tp)
    share_w = ((Bp @ W) ** 2).sum(0) / (W**2).sum(0)
    in_top = float((np.abs(Bp @ Tm) ** 2).sum() / (np.abs(Tm) ** 2).sum())
    y = z["mlp_w_sum_o0"]  # whole MLP write in the plane of its a + b write, per prompt
    in_mlp = float(
        (np.abs(line_dft(y.reshape(100, 100, 2))[LINE_IX["diff"], K - 1]) ** 2).sum()
    ) / float(z["coh_o0"][LINE_IX["diff"]])
    print(
        f"a - b write inside the a + b plane: top {len(top)} {in_top:.2f}, whole MLP {in_mlp:.2f}"
    )
    fig = plt.figure(figsize=(17, 5.6))
    # A: the pair rule
    ax: Any = fig.add_subplot(1, 3, 1)
    for x0, ang, ttl in (
        (0.0, 0.0, "ideal: w1 = w2"),
        (3.2, 62.0, "measured pair n1448, n6339:\nangle 62° (cos 0.47), equal lengths"),
    ):
        h = np.radians(ang / 2)
        w1, w2 = np.array([np.cos(h), np.sin(h)]), np.array([np.cos(h), -np.sin(h)])
        o = np.array([x0, 0.0])
        for w, c, lab in ((w1, NEURON_COLORS[1], "w1"), (w2, NEURON_COLORS[3], "w2")):
            ax.annotate("", xy=o + w, xytext=o, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
            if ang > 0:
                ax.text(*(o + w * 1.08), lab, color=c, fontsize=9)
        if ang == 0:
            ax.text(*(o + np.array([0.1, 0.45])), "w1 = w2", color=INK2, fontsize=9)
        sm, df = (w1 + w2) / 2, (w1 - w2) / 2
        ax.annotate(
            "", xy=o + sm, xytext=o, arrowprops=dict(arrowstyle="-|>", color=C_SUM, lw=3, alpha=0.8)
        )
        ax.text(
            *(o + sm + np.array([0.05, -0.32 if ang == 0 else -0.12])),
            "a + b:\n(w1 + w2)/2",
            color=C_SUM,
            fontsize=8.5,
        )
        if np.linalg.norm(df) > 1e-6:
            ax.annotate(
                "", xy=o + df, xytext=o, arrowprops=dict(arrowstyle="-|>", color=C_DIFF, lw=3)
            )
            ax.text(
                *(o + df + np.array([-0.15, 0.1])),
                "a − b:\n(w1 − w2)/2",
                color=C_DIFF,
                fontsize=8.5,
            )
            ax.plot(*(o + np.array([[0.09, 0], [0.09, 0.09], [0, 0.09]])).T, color=INK2, lw=0.8)
        else:
            ax.text(
                *(o + np.array([0.1, 0.25])), "a − b: (w1 − w2)/2 = 0", color=C_DIFF, fontsize=8.5
            )
        ax.text(x0 + 0.5, -0.95, ttl, ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(-0.3, 4.6)
    ax.set_ylim(-1.1, 1.0)
    _plain(ax, "A. Two neurons with the same a + b phase and opposite a − b phases:\n"
           "a + b is written along w1 + w2, a − b along w1 − w2")  # fmt: skip
    # B: a 3-D toy
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    i = np.arange(10000)
    ta, tb = 2 * np.pi * (i // 100 + 1) / T, 2 * np.pi * (i % 100 + 1) / T
    hgt, s_ = 0.45, 0.15
    vals = (
        np.cos(ta) * np.cos(tb),
        -np.sin(ta) * np.sin(tb),
        np.cos(ta) * np.sin(tb),
        np.sin(ta) * np.cos(tb),
    )
    ws = np.array([[1, 0, hgt], [1, 0, -hgt], [-s_, 1, 0], [s_, 1, 0]], float)
    ws /= np.linalg.norm(ws, axis=1, keepdims=True)
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot_trisurf(
        np.r_[0, np.cos(th)] * 1.3,
        np.r_[0, np.sin(th)] * 1.3,
        np.zeros(81),
        color=C_SUM,
        alpha=0.08,
    )
    for line, col in (("sum", C_SUM), ("diff", C_DIFF)):
        pts = np.sum(
            [class_means(v, line)[:, None] * w[None] for v, w in zip(vals, ws, strict=True)], 0
        )
        pts = np.r_[pts, pts[:1]]
        ax.plot(*pts.T, color=col, lw=2.2)
        ax.scatter(*pts[:-1].T, color=col, s=8)
    for j, w in enumerate(ws):
        ax.quiver(0, 0, 0, *w, color=NEURON_COLORS[j], lw=1.6, arrow_length_ratio=0.12)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-60)
    ax.set_title("B. Toy: four write vectors (arrows) that agree in the shaded plane\nbut not above / below it. "
                 "Green: a + b classes (circle, in the plane);\namber: a − b classes (small loop, mostly vertical)",
                 fontsize=8.5)  # fmt: skip
    # C: measured
    ax = fig.add_subplot(1, 3, 3)
    labels = [f"w of n{n}" for n in top] + [
        "a + b write",
        f"a − b write\n({len(top)} neurons)",
        "a − b write\n(whole MLP)",
    ]
    vals_in = np.r_[share_w, 1.0, in_top, in_mlp]
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals_in, color=[GRAY] * len(top) + [C_SUM, C_DIFF, C_DIFF], height=0.65)
    ax.barh(yy, 1 - vals_in, left=vals_in, color=FAINT, height=0.65)
    for y_, v in zip(yy, vals_in, strict=True):
        ax.text(v + 0.01, y_, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of the squared length inside the plane of the a + b write", fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_title("C. Measured (L18, period 50, addition)", fontsize=8.5)
    fig.suptitle("Why the a − b leftover lands outside the a + b plane", fontsize=10.5, color=INK2)
    save(fig, "k2_planes")


def figures() -> None:
    z = dict(np.load(FILE))
    fig_textbook()
    fig_circles(z)
    fig_gates(z)
    fig_cancel(z)
    fig_phasors(z)
    fig_planes(z)
    table(z)


if __name__ == "__main__":
    {"compute": compute, "figures": figures}[sys.argv[1]]()
