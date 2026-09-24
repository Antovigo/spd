"""Figures of the vector-level auto-interp. `python -m param_decomp.arith_repr.vectors.figures [name ...]`

Every figure goes to OUT/figs/<name>.png."""

import sys
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from param_decomp.arith_repr.vectors.common import LINES, OUT, comp_table, peak_value, period
from param_decomp.arith_repr.vectors.load import load_reads

FIGS = OUT / "figs"
LINE_COLORS = {"a": "#2a78d6", "b": "#eb6834", "sum": "#1baf7a", "diff": "#eda100"}
LINE_LABEL = {"a": "a", "b": "b", "sum": "a+b", "diff": "a−b"}
GRAY = "#b9b8b3"
INK2 = "#52514e"
PERIOD_ORDER = (2, 4, 5, 10, 20, 25, 50, 100)


def style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def read_composition() -> None:
    """Share of the component inners' grid variance on each line, by layer, at `=` (var-weighted)."""
    comps = comp_table()
    r = load_reads()
    E = r.energy.sum(-1)  # (A, 4, 2, 4)
    lin = r.lin.sum(-1)  # (A, 4, 2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    groups = (("gate/up (read the stream)", ("gate", "up")), ("down (read neurons)", ("down",)),
              ("q/k/v/o", ("q", "k", "v", "o")))  # fmt: skip
    for o, op in enumerate(("add", "sub")):
        for g, (title, kinds) in enumerate(groups):
            ax = axes[o, g]
            shares = np.zeros((32, 6))
            for li in range(32):
                sel = np.flatnonzero((comps["layer"] == li) & np.isin(comps["kind"], kinds))
                w = r.var[sel, 3, o]
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                shares[li, :4] = (E[sel, 3, o] * w[:, None]).sum(0)
                shares[li, 4] = (lin[sel, 3, o] * w).sum()
                shares[li, 5] = max(0.0, 1 - shares[li, :5].sum())
            bottom = np.zeros(32)
            cols = [LINE_COLORS[q] for q in LINES] + ["#4a3aa7", "#e6e5e0"]
            labs = [LINE_LABEL[q] for q in LINES] + ["linear a, b", "off-line / token"]
            for j in range(6):
                ax.bar(np.arange(32), shares[:, j], bottom=bottom, color=cols[j], width=0.85,
                       label=labs[j], edgecolor="white", linewidth=0.5)  # fmt: skip
                bottom += shares[:, j]
            style(ax)
            ax.set_title(f"{title} — {op}", fontsize=9, color=INK2)
            if g == 0:
                ax.set_ylabel("share of inner variance at '='", fontsize=8)
            if o == 1:
                ax.set_xlabel("layer", fontsize=8)
    axes[0, 2].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("What the components' V vectors read at '=': periodic codes of a, b, a+b, a−b",
                 fontsize=10)  # fmt: skip
    save(fig, "read_composition")


def mirror_readers() -> None:
    """Same vs mirrored b code on sub, as seen by the gate/up readers at `=`, per harmonic."""
    comps = comp_table()
    r = load_reads()
    ks = (1, 2, 3, 5, 10, 20, 25)
    layers = np.arange(10, 26)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    ims = []
    for ax, (line, li) in zip(axes, (("b", 1), ("a", 0)), strict=False):
        M = np.zeros((len(ks), layers.size))
        for j, L in enumerate(layers):
            sel = np.flatnonzero((comps["layer"] == L) & np.isin(comps["kind"], ("gate", "up")))
            for i, k in enumerate(ks):
                a, s = r.R[sel, 3, 0, li, k - 1], r.R[sel, 3, 1, li, k - 1]
                den = np.sqrt((np.abs(a) ** 2).sum() * (np.abs(s) ** 2).sum()) + 1e-12
                same = np.abs((s * np.conj(a)).sum()) / den
                refl = np.abs((s * a).sum()) / den
                M[i, j] = refl - same
        ims.append(ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"))
        ax.set_xticks(range(layers.size), [str(v) for v in layers], fontsize=7)
        ax.set_yticks(range(len(ks)), [f"k={k} (mod {period(k)})" for k in ks], fontsize=7)
        ax.set_xlabel("layer of the gate/up readers (mlp_in.l)", fontsize=8)
        ax.set_title(f"code of {line} at '=': sub vs add", fontsize=9, color=INK2)
    cb = fig.colorbar(ims[-1], ax=axes, shrink=0.8)
    cb.set_label("|<R_sub, R_add*>| − |<R_sub, R_add>|\n(> 0: mirrored, q → −q)", fontsize=7)
    save(fig, "mirror_readers")


def qk_routing() -> None:
    z = np.load(OUT / "qk.npz")
    heads = [(0, 23), (1, 6), (2, 2), (13, 7), (15, 13), (16, 21), (17, 9), (18, 30), (20, 2),
             (24, 10)]  # fmt: skip
    M = np.zeros((2, len(heads), 5))
    for i, (L, H) in enumerate(heads):
        P = z[f"pair_{L}"]
        for o in range(2):
            M[o, i] = P[:, :, H, :, o].sum((0, 1))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    vmax = np.abs(M).max()
    ims = []
    for o, ax in enumerate(axes):
        ims.append(ax.imshow(M[o], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"))
        for i in range(len(heads)):
            for s in range(5):
                ax.text(s, i, f"{M[o, i, s]:.1f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(5), ["BOS", "a", "op", "b", "="], fontsize=8)
        ax.set_yticks(range(len(heads)), [f"L{L}H{H}" for L, H in heads], fontsize=8)
        ax.set_title(("add", "sub")[o], fontsize=9, color=INK2)
    fig.colorbar(ims[-1], ax=axes, shrink=0.8).set_label(
        "alive q·k logit (query at '=')", fontsize=7
    )
    fig.suptitle("Where the heads at '=' look, from q and k component vectors alone", fontsize=10)
    save(fig, "qk_routing")


def ov_phase() -> None:
    """Phase at the source (v comp, pos a or b) vs phase written at `=` (o comp), strongest pairs."""
    r = load_reads()
    wz = np.load(OUT.parent / "autointerp/wiring.npz")
    head = wz["head"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (L, H, src, line) in zip(axes, ((15, 13, 2, 1), (16, 21, 0, 0)), strict=False):
        ids = wz[f"vo_{L}_ids"]
        vo = wz[f"vo_{L}"]
        nv = vo.shape[0]
        vs, os_ = ids[:nv], ids[nv:]
        xs, ys, cs, ss = [], [], [], []
        for j, oc in enumerate(os_):
            if head[oc].argmax() != H:
                continue
            cpl = vo[:, j, H]
            for i in np.argsort(-np.abs(cpl) * np.sqrt(r.var[vs, src, 0]))[:3]:
                e_v = r.energy[vs[i], src, 0, line]
                e_o = r.energy[oc, 3, 0, line]
                k0 = int(np.argmax(e_v * e_o))
                k = k0 + 1
                if e_v[k0] < 0.1 or e_o[k0] < 0.1:
                    continue
                Rv = r.R[vs[i], src, 0, line, k0] * np.sign(cpl[i])
                Ro = r.R[oc, 3, 0, line, k0]
                T = 100 / k
                xs.append(peak_value(Rv, k) / T)
                ys.append(peak_value(Ro, k) / T)
                cs.append(k)
                ss.append(abs(cpl[i]))
        sc = ax.scatter(xs, ys, c=np.log2(cs), cmap="viridis", s=12 + 200 * np.array(ss),
                        edgecolor="white", linewidth=0.5)  # fmt: skip
        ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style(ax)
        q = ("a", "b")[line]
        ax.set_xlabel(f"v comp: phase of {q} it reads at the {q} token (fraction of period)",
                      fontsize=8)  # fmt: skip
        ax.set_ylabel(f"o comp: phase of {q} it reads at '='", fontsize=8)
        ax.set_title(f"L{L}H{H}: {len(xs)} (v, o) pairs", fontsize=9, color=INK2)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("log2 k", fontsize=7)
    fig.suptitle("OV circuits copy the operand code without changing its phase", fontsize=10)
    save(fig, "ov_phase")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        globals()[nm]()


FAMILY_T = (2, 4, 5, 10, 20, 25, 50, 100)
PER = np.array([period(k) for k in range(1, 51)])
PT_NAMES = ["emb"] + [f"{li}{s}" for li in range(32) for s in ("a", "m")]


def storage_map() -> None:
    """Share of the raw stream's variance per code family, every stream point, positions a / b / =."""
    z = np.load(OUT / "storage.npz")
    E, lin = z["energy"], z["lin"]  # (65, 4, 2, 4, 50), (65, 4, 2, 2)
    panels = [("position a (1), add", 0, 0, ("a",)), ("position b (3), add", 2, 0, ("a", "b")),
              ("position '=' (4), add", 3, 0, ("a", "b", "sum", "diff")),
              ("position '=' (4), sub", 3, 1, ("a", "b", "sum", "diff"))]  # fmt: skip
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 13), sharex=True)
    for ax, (title, p, o, lines) in zip(axes, panels, strict=False):
        rows, labels = [], []
        for q in lines:
            li = LINES.index(q)
            if q in ("a", "b"):
                rows.append(lin[:, p, o, li])
                labels.append(f"{LINE_LABEL[q]} linear")
            for T in FAMILY_T:
                if q == "diff" and T == 2:
                    continue  # (a-b) mod 2 = (a+b) mod 2: the same code
                rows.append(E[:, p, o, li][:, PER == T].sum(1))
                labels.append(
                    f"{LINE_LABEL[q]} mod {T}" + (" (+ identity)" if T == 100 and q in "ab" else "")
                )
        M = np.array(rows)
        im = ax.imshow(np.maximum(M, 1e-4), aspect="auto", cmap="Blues", interpolation="nearest",
                       norm=LogNorm(vmin=1e-3, vmax=0.5))  # fmt: skip
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        ax.set_title(title, fontsize=9, color=INK2, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        cb.ax.tick_params(labelsize=6)
        y = 0
        for q in lines:
            y += 9 if q in ("a", "b") else (7 if q == "diff" else 8)
            ax.axhline(y - 0.5, color="white", lw=1.5)
    ticks = np.arange(0, 65, 4)
    axes[-1].set_xticks(ticks, [PT_NAMES[t] for t in ticks], fontsize=7)
    axes[-1].set_xlabel("raw stream point (emb, then <layer>a after attention / <layer>m after MLP)",
                        fontsize=8)  # fmt: skip
    fig.suptitle(
        "Share of the raw residual variance (full 4096-d stream, over the 100 x 100 grid) carried by each code (log scale)",
        fontsize=10,
    )
    save(fig, "storage_map")


def code_geometry() -> None:
    """Are a's codes at position a and b's at position b the same directions? And do codes keep
    their directions from one stream point to the next?"""
    z = np.load(OUT / "storage.npz")
    cp = np.abs(z["cos_pos"][:, 0])  # (65, 50) add
    E = z["energy"]
    cn = np.abs(z["cos_next"])  # (64, 4, 2, 4, 50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    ax = axes[0]
    for T, col in zip((10, 5, 50, 100), ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"), strict=False):
        ks = np.flatnonzero(PER == T)
        w = E[:, 0, 0, 0][:, ks] + E[:, 2, 0, 1][:, ks]
        ax.plot(np.arange(65), (cp[:, ks] * w).sum(1) / np.maximum(w.sum(1), 1e-12), color=col, lw=2,
                label=f"mod {T}")  # fmt: skip
    style(ax)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 65, 8), [PT_NAMES[t] for t in range(0, 65, 8)], fontsize=7)
    ax.set_ylabel("|cos| (a code at pos a, b code at pos b)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("operands share their code directions across positions", fontsize=9, color=INK2)
    ax = axes[1]
    for q, p, col in (("a", 0, "#2a78d6"), ("b", 2, "#eb6834"), ("sum", 3, "#1baf7a")):
        li = LINES.index(q)
        w = E[1:, p, 0, li]
        c = (cn[:, p, 0, li] * w).sum(1) / np.maximum(w.sum(1), 1e-12)
        ax.plot(np.arange(1, 65), c, color=col, lw=2, label=f"{LINE_LABEL[q]} at pos {p + 1}")
    style(ax)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 65, 8), [PT_NAMES[t] for t in range(0, 65, 8)], fontsize=7)
    ax.set_ylabel("|cos| of a code with itself one step earlier", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("code directions are kept from one sublayer to the next", fontsize=9, color=INK2)
    save(fig, "code_geometry")


def units_spokes() -> None:
    """U of the L0 MLP down components, drawn in the plane of a's units-digit code (k = 10) right
    after the L0 MLP, labelled with the digit their inner prefers."""
    comps = comp_table()
    r = load_reads()
    from param_decomp.arith_repr.vectors.load import load_writes
    from param_decomp.arith_repr.vectors.storage import load_frame

    wr = load_writes()
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    F = load_frame(fre, fim, st, 2)[0, 0, 0]  # after L0 MLP, pos a, add, a line: (50, d)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, k in zip(axes, (10, 20, 2), strict=False):
        Fk = F[k - 1]
        nrm = np.linalg.norm(Fk)
        sel = np.flatnonzero((comps["layer"][wr.cols] == 0) & (comps["kind"][wr.cols] == "down"))
        cols = wr.cols[sel]
        e_in = r.energy[cols, 0, 0, 0, k - 1]
        zc = np.conj(wr.W[sel, 0, 0, 0, k - 1]) / nrm / wr.unorm[sel]  # U . conj(F) / |F||U|
        # contribution sign: the inner's k-harmonic says which residue turns the comp up
        keep = e_in > 0.15
        T = 100 / k
        for j in np.flatnonzero(keep):
            c = cols[j]
            pv = peak_value(r.R[c, 0, 0, 0, k - 1], k)
            s = np.sign(np.cos(np.angle(r.R[c, 0, 0, 0, k - 1] * wr.W[sel[j], 0, 0, 0, k - 1])))
            zz = zc[j]
            col = matplotlib.colormaps["hsv"](float(pv / T))
            ax.annotate("", xy=(zz.real, zz.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))  # fmt: skip
            ax.text(zz.real * 1.12, zz.imag * 1.12, f"{int(round(float(pv))) % int(round(T))}", fontsize=7, ha="center", va="center",
                    color=INK2)  # fmt: skip
            del s
        # the code's own residues, for reference (unit circle positions)
        th = 2 * np.pi * np.arange(int(round(T))) / T
        m = max(0.05, np.abs(zc[keep]).max() * 1.3) if keep.any() else 0.1
        ax.scatter(m * np.cos(th), m * np.sin(th), s=8, color=GRAY)
        for i, t_ in enumerate(th):
            ax.text(m * 1.12 * np.cos(t_), m * 1.12 * np.sin(t_), str(i), fontsize=6, color=GRAY,
                    ha="center", va="center")  # fmt: skip
        ax.set_aspect("equal")
        ax.set_xlim(-1.3 * m, 1.3 * m)
        ax.set_ylim(-1.3 * m, 1.3 * m)
        ax.axis("off")
        ax.set_title(
            f"k = {k} (a mod {period(k)}): {keep.sum()} L0 down comps", fontsize=9, color=INK2
        )
    fig.suptitle("L0 MLP down components write spokes of a's residue circles: arrow = U projected on the "
                 "code plane, label = residue its inner prefers, grey = where each residue sits",
                 fontsize=9)  # fmt: skip
    save(fig, "units_spokes")


def code_steps() -> None:
    """Which sublayer writes each code (|dF|^2 per step), at a / b / '=' — and how much of the step
    the alive writers' own contributions R_c U_c account for."""
    z = np.load(OUT / "transfer.npz")
    dE = z["dF_energy"]  # (64, 4, 2, 4, 50)
    share, cols = z["share"], z["cols"]
    comps = comp_table()
    wl, wk = comps["layer"][cols], comps["kind"][cols]
    step_of = 2 * wl + (wk == "down").astype(int)  # step t -> t+1 index
    panels = [("a codes at position a", 0, 0, 0), ("b codes at position b", 2, 0, 1),
              ("a+b codes at '=' (add)", 3, 0, 2), ("a−b codes at '=' (sub)", 3, 1, 3)]  # fmt: skip
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 10), sharex=True)
    for ax, (title, p, o, li) in zip(axes, panels, strict=False):
        tot = dE[:, p, o, li].sum(-1)
        # alive-explained share, weighted over k by |dF|^2
        expl = np.zeros(64)
        for t in range(64):
            sel = step_of == t
            if sel.any():
                w = dE[t, p, o, li]
                expl[t] = (share[sel, p, o, li].sum(0) * w).sum() / max(w.sum(), 1e-12)
        x = np.arange(64)
        colors = ["#2a78d6" if t % 2 == 0 else "#eb6834" for t in x]
        ax.bar(x, tot, color=colors, width=0.8)
        big = np.argsort(-tot)[:6]
        for t in big:
            ax.text(t, tot[t], f"{expl[t]:.2f}", fontsize=6, ha="center", va="bottom", color=INK2)
        style(ax)
        ax.set_ylabel("|ΔF|² / var", fontsize=8)
        ax.set_title(title + "  (blue = attention step, orange = MLP step)", fontsize=9, color=INK2,
                     loc="left")  # fmt: skip
    axes[-1].set_xticks(np.arange(0, 64, 4), [f"L{t // 2}" for t in range(0, 64, 4)], fontsize=7)
    fig.suptitle("Which sublayer writes each code; numbers = share of the step accounted for by the "
                 "alive components' own R_c·U_c", fontsize=10)  # fmt: skip
    save(fig, "code_steps")


def logit_votes() -> None:
    """Logit lens of the result code: each harmonic's vote for token n = res, per stream point."""
    z = np.load(OUT / "output.npz")
    vote = z["vote"]  # (65, 2, 50) complex
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for o, ax in enumerate(axes[:2]):
        for T, col in zip(FAMILY_T, ("#e34948", "#9085e9", "#eda100", "#2a78d6", "#eb6834", "#e87ba4",
                                     "#1baf7a", "#4a3aa7"), strict=False):  # fmt: skip
            ks = np.flatnonzero(PER == T)
            w = np.where(ks == 49, 1.0, 2.0)
            ax.plot(
                np.arange(65), (vote[:, o, ks].real * w).sum(1), color=col, lw=1.8, label=f"mod {T}"
            )
        style(ax)
        ax.set_xticks(np.arange(0, 65, 8), [PT_NAMES[t] for t in range(0, 65, 8)], fontsize=7)
        ax.set_title(f"{('add', 'sub')[o]}: logit given to token n = res by each code family",
                     fontsize=9, color=INK2)  # fmt: skip
        ax.set_ylabel("logit (vote at n = res, relative to the mean number)", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]
    # the full vote profile over n - res at the last point (add)
    d = np.arange(-50, 51)
    for o, col in ((0, "#1baf7a"), (1, "#eda100")):
        v = vote[-1, o]
        w = np.r_[np.full(49, 2.0), 1.0]
        prof = (
            w[None]
            * (v[None] * np.exp(-2j * np.pi * np.arange(1, 51)[None] * d[:, None] / 100)).real
        ).sum(1)
        ax.plot(d, prof, color=col, lw=2, label=("add: a+b", "sub: a−b")[o])
    style(ax)
    ax.set_xlabel("n − res (mod 100)", fontsize=8)
    ax.set_ylabel("logit from the periodic result codes", fontsize=8)
    ax.set_title("after the last layer: votes add up only at n = res", fontsize=9, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    save(fig, "logit_votes")


def mlp_creation() -> None:
    """Per layer at '=': result-line share of what the neurons are handed (linear g, u) vs what
    they output (act), and how well the a x b product predicts the newly created part."""
    df = pd.read_parquet(OUT / "mlp_units.parquet")
    d = cast(pd.DataFrame, df[df["pos"] == 4])
    layers = np.arange(10, 32)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for o, op in enumerate(("add", "sub")):
        g = cast(pd.DataFrame, d[d["op"] == o])
        res = "sum" if o == 0 else "diff"
        rin, rout, fit = [], [], []
        for layer in layers:
            x = cast(pd.DataFrame, g[g["layer"] == layer])
            w = np.asarray(x["var_act"], float)
            ro = np.asarray(x[f"out_{res}"], float)
            rin.append(
                np.average(
                    (np.asarray(x[f"in_g_{res}"]) + np.asarray(x[f"in_u_{res}"])) / 2, weights=w
                )
            )
            rout.append(np.average(ro, weights=w))
            fit.append(np.average(np.asarray(x["cross_fit"], float), weights=w * ro + 1e-12))
        ax = axes[o]
        ax.bar(
            layers - 0.2, rin, width=0.4, color="#b9d3f2", label="handed in: g, u (linear reads)"
        )
        ax.bar(layers + 0.2, rout, width=0.4, color=LINE_COLORS[res],
               label="handed out: silu(g)·u (what down reads)")  # fmt: skip
        for layer, f_, r_, i_ in zip(layers, fit, rout, rin, strict=True):
            if r_ > 0.05 and i_ < 0.25:
                ax.text(layer + 0.2, r_ + 0.02, f"{f_:.2f}", fontsize=6.5, ha="center", color=INK2)
        style(ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"share on the {LINE_LABEL[res]} line", fontsize=8)
        ax.set_title(f"{op}: numbers = fit of the a×b product to the newly created {LINE_LABEL[res]} code (where little is handed in)",
                     fontsize=9, color=INK2, loc="left")  # fmt: skip
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[-1].set_xticks(layers, [str(v) for v in layers], fontsize=7)
    axes[-1].set_xlabel("MLP layer (at '=')", fontsize=8)
    fig.suptitle("L16-L18 MLPs create the result code by multiplying operand codes; from L19 they "
                 "receive it", fontsize=10)  # fmt: skip
    save(fig, "mlp_creation")


def result_units() -> None:
    """(a, b) grids of a few result-creating units at '=': the gate / up pre-activations of their top
    neuron (sums of a- and b-stripes) and the neuron output (anti-diagonal stripes = a + b code)."""
    from param_decomp.arith_repr.vectors.common import DATASET, load_uv

    comps = comp_table()
    units = ((16, 10), (18, 23), (18, 12), (19, 0))
    g_mm = np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")
    u_mm = np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")
    fig, axes = plt.subplots(len(units), 4, figsize=(12, 3 * len(units)))
    for row, (L, cid) in enumerate(units):
        d = np.flatnonzero(
            (comps["layer"] == L) & (comps["kind"] == "down") & (comps["cidx"] == cid)
        )
        V, _ = load_uv(comps, d)
        v = V[0]
        n = int(np.argmax(np.abs(v)))
        for o in range(2):
            sl = slice(o * 10000, (o + 1) * 10000)
            g = np.asarray(g_mm[L, sl, 4, n], np.float32).reshape(100, 100)
            u = np.asarray(u_mm[L, sl, 4, n], np.float32).reshape(100, 100)
            act = g / (1 + np.exp(-g)) * u
            if o == 0:
                for j, (img, lab) in enumerate(
                    ((g, "gate pre-act g"), (u, "up u"), (act, "silu(g)·u"))
                ):
                    ax = axes[row, j]
                    ax.imshow(img, origin="lower", extent=(0.5, 100.5, 0.5, 100.5), cmap="RdBu_r",
                              vmin=-np.abs(img).max(), vmax=np.abs(img).max())  # fmt: skip
                    ax.set_title(
                        f"L{L} down c{cid}, neuron {n}: {lab} (add)", fontsize=7.5, color=INK2
                    )
                    ax.set_xlabel("b", fontsize=7)
                    ax.set_ylabel("a", fontsize=7)
                    ax.tick_params(labelsize=6)
            else:
                ax = axes[row, 3]
                ax.imshow(act, origin="lower", extent=(0.5, 100.5, 0.5, 100.5), cmap="RdBu_r",
                          vmin=-np.abs(act).max(), vmax=np.abs(act).max())  # fmt: skip
                ax.set_title("silu(g)·u (sub): diagonal = a − b", fontsize=7.5, color=INK2)
                ax.set_xlabel("b", fontsize=7)
                ax.tick_params(labelsize=6)
    fig.suptitle("Result-creating MLP units at '=': gate and up are sums of a- and b-codes; their "
                 "product is a code of a+b (add) / a−b (sub)", fontsize=10)  # fmt: skip
    fig.tight_layout()
    save(fig, "result_units")


def op_flag() -> None:
    """The op flag at '=': its size along the stream, and how its direction drifts."""
    st = np.load(OUT / "frames_stats.npz")
    od = st["mean"][:, 3, 0] - st["mean"][:, 3, 1]
    nrm = np.linalg.norm(od, axis=1)
    size = nrm / (st["rms"][:, 3].mean(1) * np.sqrt(od.shape[1]))
    odn = od / np.maximum(nrm[:, None], 1e-9)
    C = odn @ odn.T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(np.arange(65), size, color="#4a3aa7", lw=2)
    style(ax)
    ax.set_xticks(np.arange(0, 65, 8), [PT_NAMES[t] for t in range(0, 65, 8)], fontsize=7)
    ax.set_ylabel("|mean(add) − mean(sub)| / |x| at '='", fontsize=8)
    ax.set_title("the op flag is a large direction at '=' from L1", fontsize=9, color=INK2)
    ax = axes[1]
    im = ax.imshow(C[1:, 1:], cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = np.arange(0, 64, 8)
    ax.set_xticks(ticks, [PT_NAMES[t + 1] for t in ticks], fontsize=7)
    ax.set_yticks(ticks, [PT_NAMES[t + 1] for t in ticks], fontsize=7)
    ax.set_title("cosine of the op-flag direction between stream points", fontsize=9, color=INK2)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "op_flag")


def schematic() -> None:
    """Pipeline read off the vectors: positions across, layers down."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 10.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 10.9)
    ax.axis("off")
    X = {"a": 1.8, "op": 5.0, "b": 8.2, "=": 11.8}
    for k, x in X.items():
        ax.text(x, 10.75, f"position {k}", ha="center", fontsize=11, weight="bold", color=INK2)

    def box(x: float, y: float, w: float, h: float, text: str, col: str) -> None:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=col, ec="white", lw=2))  # fmt: skip
        ax.text(x, y, text, ha="center", va="center", fontsize=7.6, color="#0b0b0b", wrap=True)

    def arrow(
        p: tuple[float, float], q: tuple[float, float], col: str = INK2, rad: float = 0.0,
        text: str | None = None, tx: float = 0.0, ty: float = 0.0,
    ) -> None:  # fmt: skip
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=col, lw=1.6,
                                     connectionstyle=f"arc3,rad={rad}"))  # fmt: skip
        if text:
            ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=7, color=col,
                    ha="center")  # fmt: skip

    A, B, S, OP, M = "#d4e4f7", "#fbd9c9", "#c8eedd", "#e3def5", "#fbe8b8"
    ax.text(0.05, 9.6, "L0-L4", fontsize=9, color=INK2)
    box(
        X["a"],
        9.5,
        2.9,
        1.2,
        "token embedding: a:lin, a mod 5\nL0 MLP down comps = residue\ndetectors; U = spoke of the\nresidue circle (mod 10, 5, 50...)",
        A,
    )
    box(
        X["b"],
        9.5,
        2.9,
        1.2,
        "same for b: the SAME read/write\ndirections as for a (v comps read\na at pos a and b at pos b with\nthe same phase)",
        B,
    )
    box(X["op"], 9.5, 2.4, 1.0, "op token\n(+ / −)", OP)
    box(
        X["="],
        9.5,
        3.2,
        1.2,
        "L0 attn: b (prev. token) arrives;\nL0H23 / L1H6 / L2H2 copy the op\n→ op flag: ~70 % of |x| at '='",
        OP,
    )
    arrow(
        (X["op"], 10.0),
        (X["="] - 0.6, 10.12),
        "#4a3aa7",
        rad=-0.12,
        text="L0H23: k c6 = 'is the op token'",
        ty=0.42,
    )
    ax.text(0.05, 7.9, "L5-L14", fontsize=9, color=INK2)
    box(
        X["a"], 7.8, 2.9, 0.9, "codes kept in place (cos ≥ 0.9\nstep to step); refreshed by MLPs", A
    )
    box(
        X["b"],
        7.8,
        2.9,
        0.9,
        "slot feature 'second operand'\n(k c48 of L15) built by L7-L14 MLPs",
        B,
    )
    box(
        X["="],
        7.8,
        3.2,
        1.2,
        "op flag relayed by one-bit MLP comps\n(add-only / sub-only; U along the flag,\ndirection drifts L2 → L8 → L15);\na−b comparator (sign, magnitude)",
        OP,
    )
    ax.text(0.05, 6.3, "L15-L16", fontsize=9, color=INK2)
    box(
        X["="],
        6.2,
        3.2,
        1.6,
        "L15H13 copies b, L16H21 copies a\n(OV keeps the phase: identity on\nthe code). L15 MLP, gated by the op\nflag: add-gated comps write +c(b),\nsub-gated comps −c(b)\n→ later readers see b mirrored",
        B,
    )
    arrow(
        (X["b"] + 1.4, 7.4),
        (X["="] - 1.6, 6.5),
        "#eb6834",
        rad=-0.1,
        text="L15H13: q c138 × k c48",
        tx=0.2,
        ty=0.1,
    )
    arrow(
        (X["a"] + 1.4, 7.4),
        (X["="] - 1.6, 5.9),
        "#2a78d6",
        rad=0.15,
        text="L16H21: q c136 × k c5",
        tx=-0.4,
        ty=-0.35,
    )
    ax.text(0.05, 4.6, "L16-L18", fontsize=9, color=INK2)
    box(
        X["="],
        4.6,
        3.2,
        1.4,
        "MLP units: gate & up read a- and b-\ncodes at the SAME harmonic k; their\nproduct makes cos k(a±b) (fit 0.99);\nU writes it into the result plane\n(parity: XOR of L17/L18 AND units)",
        S,
    )
    ax.text(0.05, 3.0, "L19-L30", fontsize=9, color=INK2)
    box(
        X["="],
        3.0,
        3.2,
        1.3,
        "result codes (all periods) now read\nand re-written by every MLP (75-90 %\nof inner variance); magnitude path:\nlin(a)+lin(b) → '≥ 100' units",
        S,
    )
    ax.text(0.05, 1.4, "logits", fontsize=9, color=INK2)
    box(
        X["="],
        1.3,
        3.2,
        1.3,
        "each code plane is phase-aligned\nwith the unembedding's own Fourier\nplane → votes add only at n = res\n(Σ direct votes 3.7 logits, add)",
        M,
    )
    for y0, y1 in ((8.9, 8.35), (7.2, 7.0), (5.4, 5.3), (3.9, 3.65), (2.35, 1.95)):
        arrow((X["="], y0), (X["="], y1))
    fig.suptitle("How Llama-3.1-8B computes a ± b, as read off the ceiling-filter components' U and V",
                 fontsize=12)  # fmt: skip
    save(fig, "schematic")


def mirror_mechanism() -> None:
    """b's code at '=' as the L16 gate/up readers see it (copied code + the L15 MLP's write), drawn on
    the two axes the analysis finds: e = axis of the op-even part, w = axis of the op-odd part.
    Add and sub trace the same ellipse in opposite directions: b -> -b."""
    from param_decomp.arith_repr.vectors.common import RESID, load_uv
    from param_decomp.arith_repr.vectors.storage import load_frame

    comps = comp_table()
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    F32 = load_frame(fre, fim, st, 32)[3][:, 1]  # after the L15 MLP: (2 op, 50, d)
    cols = np.flatnonzero((comps["layer"] == 16) & np.isin(comps["kind"], ("gate", "up")))
    V, _ = load_uv(comps, cols)
    Vm = np.stack(V) * ln2[16][None]

    def axis(z: np.ndarray) -> np.ndarray:
        _, _, vt = np.linalg.svd(np.stack([np.real(z), np.imag(z)]), full_matrices=False)
        return vt[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, k in zip(axes, (2, 10, 20), strict=True):
        Fa, Fs = Vm @ F32[0, k - 1], Vm @ F32[1, k - 1]
        e, w = axis((Fa + Fs) / 2), axis((Fa - Fs) / 2)
        if (e @ ((Fa + Fs) / 2)).real < 0:
            e = -e
        T = int(period(k))
        bs = np.arange(T) if T <= 10 else np.arange(0, T, 5)
        bb = np.linspace(0, T, 200)
        for F, col, lab in ((Fa, "#1baf7a", "add"), (Fs, "#eda100", "sub")):
            ce, cw = e @ F, w @ F
            xe = 2 * (ce * np.exp(2j * np.pi * k * bb / 100)).real
            xw = 2 * (cw * np.exp(2j * np.pi * k * bb / 100)).real
            ax.plot(xe, xw, color=col, lw=2, label=lab)
            j = 20
            ax.annotate("", xy=(xe[j + 3], xw[j + 3]), xytext=(xe[j], xw[j]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2))  # fmt: skip
            for b in bs:
                pe = 2 * (ce * np.exp(2j * np.pi * k * b / 100)).real
                pw = 2 * (cw * np.exp(2j * np.pi * k * b / 100)).real
                ax.scatter([pe], [pw], s=14, color=col, zorder=3)
                ax.text(
                    pe * 1.1, pw * 1.1, str(b), fontsize=6.5, color=col, ha="center", va="center"
                )
        style(ax)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.axvline(0, color=GRAY, lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("e: op-even axis", fontsize=8)
        ax.set_ylabel("w: op-odd axis (written by the L15 MLP)", fontsize=8)
        ax.set_title(f"k = {k}: b mod {T}", fontsize=9, color=INK2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("b's code at '=' seen by the L16 readers: the op flips the sign of the sine axis only "
                 "→ the same circle traversed backwards, b → −b", fontsize=10)  # fmt: skip
    save(fig, "mirror_mechanism")


MECH_COLORS = {"X": "#2a78d6", "Sx": "#eb6834", "M": "#1baf7a", "P": "#b9b8b3", "O": "#eda100"}
MECH_LABELS = {
    "X": "a-code × b-code across gate and up",
    "Sx": "a-code × b-code inside silu(gate)",
    "M": "result × result (harmonic mixing)",
    "P": "result code passed through",
    "O": "other (e.g. (a−b)-code × b-code)",
}


def period_map() -> None:
    """Per MLP at '=' and result period: size = result-code write, colour = dominant mechanism."""
    z = np.load(OUT / "periods/summary.npz")
    power, groups = z["power"], z["groups"]
    fams, layers, names = z["families"], z["layers"], list(z["group_names"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for o, ax in enumerate(axes):
        for f in range(len(fams)):
            for j in range(len(layers)):
                p = power[o, f, j]
                if p < 2e-4:
                    continue
                g = names[int(np.argmax(groups[o, f, j]))]
                ax.scatter(
                    layers[j], f, s=6 + 9000 * p, color=MECH_COLORS[g], edgecolor="white", lw=0.8
                )
        ax.set_yticks(range(len(fams)), [f"a{'+−'[o]}b mod {T}" for T in fams], fontsize=8)
        ax.invert_yaxis()
        style(ax)
        ax.grid(axis="x", color="#e6e5e0", lw=0.6)
        ax.set_title(("addition", "subtraction")[o], fontsize=9, color=INK2, loc="left")
    axes[-1].set_xticks(layers, [str(v) for v in layers], fontsize=8)
    axes[-1].set_xlabel("MLP layer (position '=')", fontsize=8)
    for g in names:
        axes[0].scatter([], [], s=60, color=MECH_COLORS[g], label=MECH_LABELS[g])
    axes[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("How each MLP writes each period of the result (area = size of the write; colour = "
                 "dominant term of the exact gate × up bookkeeping)", fontsize=10)  # fmt: skip
    save(fig, "period_map")


def period_tree() -> None:
    """Which period is made from which: a×b at L16-L18, then result × result from L19."""
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    y0, y1 = 4.5, 1.5
    direct = {"mod 50": (1.2, "L18"), "mod 5": (3.4, "L16, L18"), "mod 20": (5.4, "L18-L19"),
              "mod 2": (7.2, "L17-L18, inside silu"), "mod 10": (9.0, "L17-L18"),
              "mod 100": (10.8, "L16-L18")}  # fmt: skip
    derived = {
        "mod 25": (1.2, "L19-L30: k2 + k2"),
        "mod 4": (4.4, "L22-L30: k20 + k5"),
        "re-derived": (9.9, "L21-L30: mod 5 (k10+k10), mod 10 (k20−k10),\n"
                            "mod 100 (k2−k1), mod 50 (k1+k1), mod 20 (k3+k2)"),
    }  # fmt: skip
    ax.text(0.2, 5.55, "created from a-code × b-code at the same harmonic (gate × up; parity inside silu)",
            fontsize=9, color=INK2)  # fmt: skip
    ax.text(0.2, 0.45, "created or re-derived later from result × result (harmonic mixing inside one neuron)",
            fontsize=9, color=INK2)  # fmt: skip
    for name, (x, lay) in direct.items():
        ax.text(x, y0, f"a+b {name}\n{lay}", ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#d4e4f7", ec="white"))  # fmt: skip
    for name, (x, lay) in derived.items():
        title = "a+b " + name if name != "re-derived" else "every other period"
        ax.text(x, y1, f"{title}\n{lay}", ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#c8eedd", ec="white"))  # fmt: skip
    edges = [("mod 50", "mod 25"), ("mod 5", "mod 4"), ("mod 20", "mod 4"), ("mod 10", "re-derived"),
             ("mod 100", "re-derived")]  # fmt: skip
    for a, b in edges:
        xa, xb = direct[a][0], derived[b][0]
        ax.add_patch(FancyArrowPatch((xa, y0 - 0.45), (xb, y1 + 0.5), arrowstyle="-|>",
                                     mutation_scale=12, color=INK2, lw=1.2))  # fmt: skip
    fig.suptitle("Where each period of the result comes from (addition; subtraction uses the same units)",
                 fontsize=10)  # fmt: skip
    save(fig, "period_tree")


def routing() -> None:
    """For the copy heads at '=': query feature, key feature and attention, by source position."""
    comps = comp_table()
    hm = np.load(OUT / "qk.npz")["hmean"]
    ap = np.load(OUT.parent / "autointerp/attn_patterns.npy", mmap_mode="r")
    rows = [(0, 23, 17, 6, "op token"), (15, 13, 138, 48, "b"), (16, 21, 136, 5, "a"),
            (18, 30, 104, 95, "a and b")]  # fmt: skip
    pos = ["BOS", "a", "op", "b", "="]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.3 * len(rows)), sharex=True)
    for i, (L, H, qc, kc, tgt) in enumerate(rows):
        q = np.flatnonzero((comps["layer"] == L) & (comps["kind"] == "q") & (comps["cidx"] == qc))[
            0
        ]
        k = np.flatnonzero((comps["layer"] == L) & (comps["kind"] == "k") & (comps["cidx"] == kc))[
            0
        ]
        att = [
            np.asarray(ap[L, o * 10000 : (o + 1) * 10000 : 10, H, 4, :], np.float32).mean(0)
            for o in (0, 1)
        ]
        for j, (vals, title) in enumerate(
            (
                ([hm[q, :, 0]], f"query: L{L} q c{qc} (mean inner)"),
                ([hm[k, :, 0]], f"key: L{L} k c{kc} (mean inner)"),
                (att, f"L{L}H{H} attention from '=' → {tgt}"),
            )
        ):
            ax = axes[i, j]
            for o, v in enumerate(vals):
                ax.bar(np.arange(5) + (0.2 * o - 0.1 if len(vals) > 1 else 0), v, width=0.4 if len(vals) > 1 else 0.6,
                       color=("#2a78d6", "#eda100")[o] if len(vals) > 1 else INK2,
                       label=("add", "sub")[o] if len(vals) > 1 else None)  # fmt: skip
            style(ax)
            ax.axhline(0, color=GRAY, lw=0.8)
            ax.set_title(title, fontsize=8, color=INK2)
            ax.set_xticks(range(5), pos, fontsize=7)
        if i == 0:
            axes[i, 2].legend(fontsize=7, frameon=False)
    fig.suptitle("Routing is content-based: a query feature that fires only at '=' meets a key feature that "
                 "fires only at one slot (RoPE term flat across positions)", fontsize=10)  # fmt: skip
    fig.tight_layout()
    save(fig, "routing")


def operand_separation() -> None:
    """a and b share their code planes at their own tokens, but not at '=': the two copy heads write
    into near-orthogonal output spaces."""
    from param_decomp.arith_repr.vectors.storage import load_frame

    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))

    def plane(z: np.ndarray) -> np.ndarray:
        Q, _ = np.linalg.qr(np.stack([np.real(z), np.imag(z)], 1))
        return Q

    ts = list(range(1, 45))
    ks = (2, 10, 20)
    src = np.zeros((len(ts), len(ks)))
    dst = np.zeros_like(src)
    for i, t in enumerate(ts):
        F = load_frame(fre, fim, st, t)
        for j, k in enumerate(ks):
            src[i, j] = np.linalg.svd(
                plane(F[0, 0, 0, k - 1]).T @ plane(F[2, 0, 1, k - 1]), compute_uv=False
            )[0]
            dst[i, j] = np.linalg.svd(
                plane(F[3, 0, 0, k - 1]).T @ plane(F[3, 0, 1, k - 1]), compute_uv=False
            )[0]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, (k, col) in enumerate(zip(ks, ("#1baf7a", "#2a78d6", "#eb6834"), strict=True)):
        ax.plot(ts, src[:, j], color=col, lw=2, label=f"k={k}: a at pos a vs b at pos b")
        ax.plot(ts, dst[:, j], color=col, lw=2, ls="--", label=f"k={k}: a vs b at '='")
    style(ax)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ts[::4], [PT_NAMES[t] for t in ts[::4]], fontsize=7)
    ax.set_ylabel("top principal cosine of the two code planes", fontsize=8)
    ax.axvline(31, color=GRAY, lw=1)
    ax.axvline(33, color=GRAY, lw=1)
    ax.text(31, 1.0, " L15H13 copies b", fontsize=7, color=INK2)
    ax.text(33, 0.93, " L16H21 copies a", fontsize=7, color=INK2)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower right")
    fig.suptitle(
        "Same code directions at the operand tokens, different directions at '='", fontsize=10
    )
    save(fig, "operand_separation")
