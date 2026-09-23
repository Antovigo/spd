"""Figures of report_auto_interp.md.

    python -m param_decomp.arith_repr.autointerp.figures --run <run_dir> [--only name ...]

Reads the products of tuning / profiles / wiring / attn_patterns / op_phase / code_attrib /
delta_profiles under `<run>/analysis/arith_repr/autointerp/` and writes `figs/<name>.png`."""

import argparse
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from param_decomp.arith_repr.autointerp.onsets import describe

FILTER = "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
POS = ["<BOS>", "a", "op", "b", "="]


class Ctx:
    def __init__(self, run: Path) -> None:
        self.run = run
        self.ai = run / "analysis/arith_repr/autointerp"
        self.figs = self.ai / "figs"
        self.figs.mkdir(exist_ok=True)
        self.ds = run / FILTER
        self.df = pd.read_parquet(self.ai / "catalogue.parquet")
        ix = np.load(self.ds / "index.npz")
        self.a, self.b, self.op = ix["a"], ix["b"], ix["op"]

    def name(self, c: int) -> str:
        r = self.df.loc[c]
        return f"L{r.layer} {r.kind} c{r.cidx}"

    def save(self, fig: Figure, name: str) -> None:
        fig.savefig(self.figs / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("wrote", name, flush=True)


def fig_pos1_onsets(ctx: Ctx) -> None:
    """Taxonomy of on-sets over `a` of the components whose main position is `a`."""
    pr = np.load(ctx.ai / "profiles.npz")
    on_a = pr["on_a"][1].mean(0)
    vals = np.arange(1, 101)
    df = ctx.df
    rows = []
    for c in df.index[df.main_pos == 1]:
        d = describe(vals, on_a[c])
        n = len(d["R"])
        if d["tau"] == -1:
            cls = "flat"
        elif d["tau"] == 100 and n == 1:
            cls = "single value"
        elif d["tau"] == 100 and d["arc"]:
            cls = "window"
        elif d["tau"] == 100:
            cls = "scattered values"
        else:
            cls = f"residue mod {d['tau']}"
        rows.append((c, df.layer[c], cls, d))
    t = pd.DataFrame(rows, columns=["col", "layer", "cls", "d"])
    t["cls"] = t.cls.replace(
        {"residue mod 20": "residue mod 20/25", "residue mod 25": "residue mod 20/25"}
    )
    order = ["single value", "window", "scattered values", "residue mod 10", "residue mod 50",
             "residue mod 5", "residue mod 20/25", "flat"]  # fmt: skip
    ct = pd.crosstab(t.layer, t.cls).reindex(columns=order, fill_value=0)
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(len(ct))
    cmap = plt.get_cmap("tab10")
    for i, cname in enumerate(order):
        ax.bar(ct.index, ct[cname], bottom=bottom, label=cname, color=cmap(i))
        bottom += np.asarray(ct[cname].values)
    ax.set_xlabel("layer")
    ax.set_ylabel("# components whose main position is `a`")
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("What turns an `a`-position component on: on-set over a in 1..100 (on-rate > 0.5)")
    examples = []
    for cname in ("single value", "window", "residue mod 10", "residue mod 50"):
        cand = cast(pd.DataFrame, t[(t.cls == cname) & (t.layer <= 3)])
        if cname == "window":
            cand = cast(pd.DataFrame, cand[cand.d.apply(lambda d: 8 <= len(d["R"]) <= 12)])
        examples.append(cand.iloc[len(cand) // 2])
    for j, r in enumerate(examples):
        ax = fig.add_subplot(gs[1, j])
        c = int(r.col)
        ax.bar(vals, on_a[c], width=1.0, color="k")
        w = pr["w_a"][1].mean(0)[c]
        ax2 = ax.twinx()
        ax2.plot(vals, w, color="tab:red", lw=0.8)
        ax2.set_ylabel("mean write coeff", color="tab:red", fontsize=8)
        ax.set_title(f"{ctx.name(c)}: {r.cls}", fontsize=9)
        ax.set_xlabel("a")
        ax.set_ylabel("on-rate")
    fig.tight_layout()
    ctx.save(fig, "pos1_onsets")


KEY_HEADS = [
    (0, 23, 4),
    (1, 6, 4),
    (2, 2, 4),
    (1, 24, 3),
    (5, 22, 3),
    (15, 13, 4),
    (16, 21, 4),
    (18, 30, 4),
    (13, 7, 4),
    (20, 2, 4),
    (17, 9, 3),
    (24, 10, 3),
]  # (layer, head, query position)


def fig_attention(ctx: Ctx) -> None:
    att = np.load(ctx.ai / "attn_patterns.npy", mmap_mode="r")
    wz = np.load(ctx.ai / "wiring.npz")
    head = wz["head"]
    df = ctx.df
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5), sharey=True)
    for ax, (li, h, qpos) in zip(axes.flat, KEY_HEADS, strict=True):
        o = df[(df.kind == "o") & (df.layer == li) & (head.argmax(1) == h)]
        pats = [
            np.asarray(att[li, ctx.op == o_, h, qpos, : qpos + 1], np.float32).mean(0)
            for o_ in (0, 1)
        ]
        x = np.arange(qpos + 1)
        ax.bar(x - 0.2, pats[0], 0.4, label="add")
        ax.bar(x + 0.2, pats[1], 0.4, label="sub")
        ax.set_xticks(x)
        ax.set_xticklabels(POS[: qpos + 1])
        ax.set_title(f"L{li}H{h}: query {POS[qpos]!r}, {len(o)} alive o comps", fontsize=9)
    axes.flat[0].legend()
    fig.suptitle(
        "Attention of the heads that carry alive o components (original model, mean over prompts)"
    )
    fig.tight_layout()
    ctx.save(fig, "attention_heads")


def fig_result_switch(ctx: Ctx) -> None:
    t = np.load(ctx.ai / "tuning.npz")
    layer = ctx.df.layer.values
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    Ls = np.arange(8, 32)
    for ax, (o, name) in zip(axes[:2], ((0, "add"), (1, "sub")), strict=True):
        wv = t["w_std"][4, o] ** 2
        series = {k: [] for k in ("a-only", "b-only", "sum", "diff")}
        for L in Ls:
            m = layer == L
            for k in series:
                series[k].append((t[f"dft_{k}"][4, o][m].sum(1) * wv[m]).sum() / wv[m].sum())
        for k, v in series.items():
            ax.plot(
                Ls,
                v,
                marker="o",
                ms=3,
                label={"sum": "a+b diagonal", "diff": "a-b diagonal"}.get(k, k),
            )
        ax.set_title(f"{name}: where the `=` writes live on the (a, b) torus")
        ax.set_xlabel("layer")
        ax.set_ylabel("share of write variance (DFT)")
        ax.legend(fontsize=8)
        ax.axvspan(18.5, 20.5, color="grey", alpha=0.15)
    ax = axes[2]
    wv = t["w_std"][4, 0] ** 2
    groups = {"100": [k for k in range(1, 51) if np.gcd(k, 100) == 1], "50": [k for k in range(1, 51) if np.gcd(k, 100) == 2],
              "25": [k for k in range(1, 51) if np.gcd(k, 100) == 4], "20": [k for k in range(1, 51) if np.gcd(k, 100) == 5],
              "10": [10, 30], "5": [20, 40], "2": [50]}  # fmt: skip
    for g, ks in groups.items():
        v = [
            ((t["dft_sum"][4, 0][layer == L][:, ks].sum(1)) * wv[layer == L]).sum()
            / wv[layer == L].sum()
            for L in Ls
        ]
        ax.plot(Ls, v, marker="o", ms=3, label=f"a+b mod {g}")
    ax.set_title("add: a+b diagonal energy split by period")
    ax.set_xlabel("layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    ctx.save(fig, "result_switch")


def fig_op_phase(ctx: Ctx) -> None:
    z = np.load(ctx.ai / "op_phase.npz")
    S, R = z["same"], z["reflect"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    pts = [(li, s) for li in range(32) for s in range(2)]
    xs = np.arange(len(pts)) / 2
    for ax, (p_i, q_i, title) in zip(
        axes,
        ((1, 1, "b at `=`"), (1, 0, "a at `=` (control)"), (0, 1, "b at its own position")),
        strict=True,
    ):
        for k, col in ((10, "tab:blue"), (20, "tab:orange"), (1, "tab:green"), (2, "tab:purple")):
            ax.plot(
                xs,
                [S[li, s, p_i, q_i, k] for li, s in pts],
                color=col,
                lw=1.2,
                label=f"k={k} (period {100 // np.gcd(k, 100)}) same",
            )
            ax.plot(
                xs,
                [R[li, s, p_i, q_i, k] for li, s in pts],
                color=col,
                lw=1.2,
                ls="--",
                label=f"k={k} mirrored",
            )
        ax.axvline(15.25, color="grey", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("read point (layer; attn_in at li, mlp_in at li+0.5)")
    axes[0].set_ylabel("|cos| between sub and add Fourier code")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Subtraction mirrors b's code at `=` (b -> -b) right after layer 15's MLP; a's code and b at its own token are never mirrored"
    )
    fig.tight_layout()
    ctx.save(fig, "op_mirror")


def fig_output(ctx: Ctx) -> None:
    z = np.load(ctx.ai / "delta_profiles.npz")
    writers, deltas = z["writers"], z["deltas"]
    ons = pd.read_parquet(ctx.ai / "onsets_all.parquet")
    ons = ons[(ons.pos == 4) & (ons.op == "add")].set_index("col")
    fam = np.asarray(ons["label"].reindex(writers).fillna("-").values)
    P = z["add"] - z["add_base"][:, None]
    fig, axes = plt.subplots(2, 3, figsize=(17, 7.5))
    sel_d = (deltas >= -120) & (deltas <= 120)
    for ax, f in zip(
        axes.flat, ["res%10", "res%100", "res", "res//10", "tens(a,b)", "ALL"], strict=True
    ):
        m = np.ones(len(fam), bool) if f == "ALL" else fam == f
        prof = np.nansum(P[m], 0)
        ax.bar(deltas[sel_d], prof[sel_d], width=1.0, color="k")
        ax.set_title(
            f"{'all `=` writers L14-31' if f == 'ALL' else 'on-set family ' + f} (n={m.sum()})",
            fontsize=10,
        )
        ax.set_xlabel("token - correct answer")
        ax.axhline(0, color="grey", lw=0.5)
    axes[0, 0].set_ylabel("summed direct logit (mean over add prompts)")
    fig.suptitle(
        "Direct effect of the late `=` writers on the number tokens, relative to the correct answer (add)"
    )
    fig.tight_layout()
    ctx.save(fig, "output_families")


def _col(ctx: Ctx, layer: int, kind: str, cidx: int) -> int:
    df = ctx.df
    matches = np.asarray(df.index[(df.layer == layer) & (df.kind == kind) & (df.cidx == cidx)])
    return int(matches[0])


def _gated(ctx: Ctx, col: int, pos: int) -> np.ndarray:
    ci = np.load(ctx.ds / "original" / "ci.npy", mmap_mode="r")
    inner = np.load(ctx.ds / "original" / "inner.npy", mmap_mode="r")
    return np.asarray(inner[:, pos, col]) * (np.asarray(ci[:, pos, col], np.float32) > 0.01)


CURATED = [
    ((15, "o", 60), "L15H13 copy of b mod 2"), ((16, "o", 120), "L16H21 copy of a (window)"),
    ((16, "gate", 493), "conjunction a~50 & b~55"), ((19, "gate", 41), "digit-pair lattice"),
    ((19, "gate", 0), "a+b mod 50 band"), ((21, "up", 13), "a+b mod 10 in {5,6,7}"),
    ((22, "down", 6), "a+b = 100 line"), ((29, "up", 549), "(a+b) mod 100 in 20s"),
    ((30, "gate", 399), "(a+b) mod 100 in 1..9"),
]  # fmt: skip


def fig_curated_grids(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, len(CURATED), figsize=(2.3 * len(CURATED), 5.2))
    for j, ((li, k, c), title) in enumerate(CURATED):
        col = _col(ctx, li, k, c)
        w = _gated(ctx, col, 4)
        for o in (0, 1):
            g = np.full((100, 100), np.nan)
            m = ctx.op == o
            g[ctx.a[m] - 1, ctx.b[m] - 1] = w[m]
            vm = np.nanmax(np.abs(g)) or 1.0
            ax = axes[o, j]
            ax.imshow(
                g, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=(0.5, 100.5, 0.5, 100.5)
            )
            ax.set_xticks([1, 50, 100])
            ax.set_yticks([1, 50, 100])
            ax.tick_params(labelsize=6)
            if o == 0:
                ax.set_title(f"L{li} {k} c{c}\n{title}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{['a + b', 'a - b'][o]}\na", fontsize=9)
            if o == 1:
                ax.set_xlabel("b", fontsize=8)
    fig.suptitle(
        "Gated write coefficient at `=` over the (a, b) grid (top: addition, bottom: subtraction; white = off)"
    )
    fig.tight_layout()
    ctx.save(fig, "curated_grids")


DIGIT_COMPS = [
    ((17, "down", 67), "a odd AND b odd"), ((17, "down", 22), "a odd AND b even"),
    ((18, "down", 4), "a+b even (parity)"), ((17, "down", 52), "a%10 in 2-4 AND b%10 in 0-4"),
    ((18, "down", 88), "band a%10+b%10 ~ 4-8"), ((18, "down", 32), "band"), ((18, "down", 26), "(a+b) mod 5 stripes"),
    ((17, "down", 39), "band + wrap"),
]  # fmt: skip


def fig_digit_plane(ctx: Ctx) -> None:
    fig, axes = plt.subplots(1, len(DIGIT_COMPS), figsize=(2.4 * len(DIGIT_COMPS), 3.0))
    m = ctx.op == 0
    for ax, ((li, k, c), title) in zip(axes, DIGIT_COMPS, strict=True):
        on = _gated(ctx, _col(ctx, li, k, c), 4) != 0
        g = np.zeros((10, 10))
        n = np.zeros((10, 10))
        np.add.at(g, (ctx.a[m] % 10, ctx.b[m] % 10), on[m])
        np.add.at(n, (ctx.a[m] % 10, ctx.b[m] % 10), 1)
        ax.imshow(g / n, origin="lower", cmap="Greys", vmin=0, vmax=1)
        ax.set_title(f"L{li} {k} c{c}\n{title}", fontsize=8)
        ax.set_xlabel("b mod 10", fontsize=8)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("a mod 10")
    fig.suptitle(
        "Layer 17-18 MLP components at `=` (addition): on-rate over the units digits of a and b"
    )
    fig.tight_layout()
    ctx.save(fig, "digit_plane")


def fig_code_attrib(ctx: Ctx) -> None:
    z = np.load(ctx.ai / "code_attrib.npz")
    S, W = z["share"], z["writers"]
    cases = [tuple(json.loads(c)) for c in z["cases"]]
    df = ctx.df
    is_o = df.kind.values[W] == "o"
    lay = df.layer.values[W]
    groups = {"period 10 (k=10,30)": [10, 30], "period 2 (k=50)": [50],
              "period 50 & 100": [k for k in range(1, 51) if np.gcd(k, 100) in (1, 2)]}  # fmt: skip
    show = [c for c in cases if not (c[1] == 4 and c[0] in ("mlp_in.15",) and False)]
    labels = [f"{r} @{POS[p]} {['add', 'sub'][o]} q={q}" for r, p, o, q in show]
    fig, axes = plt.subplots(1, 3, figsize=(20, 12), sharey=True)
    im = None
    for ax, (gname, ks) in zip(axes, groups.items(), strict=True):
        M = np.zeros((len(show), 33 * 2))
        for i, case in enumerate(show):
            sh = S[cases.index(case)][:, ks].mean(1)
            for L in range(32):
                M[i, 2 * L] = sh[:-1][(lay == L) & is_o].sum()
                M[i, 2 * L + 1] = sh[:-1][(lay == L) & ~is_o].sum()
            M[i, 64] = sh[-1]
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        ax.set_xticks(np.arange(0, 64, 4) + 0.5)
        ax.set_xticklabels([f"L{L}" for L in range(0, 32, 2)], fontsize=7)
        ax.set_title(gname, fontsize=10)
    axes[0].set_yticks(range(len(show)))
    axes[0].set_yticklabels(labels, fontsize=7)
    assert im is not None
    fig.colorbar(im, ax=axes, shrink=0.5)
    fig.suptitle(
        "Share of each Fourier code written by each layer (per layer: attention column, then MLP column; last column = token embedding)"
    )
    ctx.save(fig, "code_attrib")


def fig_schematic(ctx: Ctx) -> None:
    """Hand-drawn summary of sections 1-5 (no data)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 11))
    X = {"a": 0.0, "op": 1.25, "b": 2.5, "=": 5.0}
    ax.set_xlim(-1.3, 8.6)
    ax.set_ylim(33, -2.2)
    for name, x in X.items():
        ax.text(x, -1.3, f"`{name}`", ha="center", fontsize=14, weight="bold")
        ax.plot([x, x], [-0.6, 32], color="0.85", lw=1, zorder=0)
    for L in range(0, 33, 4):
        ax.text(-1.2, L, f"L{L}", va="center", fontsize=9, color="0.4")

    def box(
        x: float, y0: float, y1: float, text: str, color: str, w: float = 1.1, fs: float = 8
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y0),
                w,
                y1 - y0,
                boxstyle="round,pad=0.05",
                fc=color,
                ec="k",
                lw=0.6,
                alpha=0.9,
            )
        )
        ax.text(x, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(
        x0: float, y0: float, x1: float, y1: float, text: str, color: str = "k", dy: float = -0.3
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4
            )
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=8, color=color)

    blue, green, orange, red, grey = "#cfe2ff", "#d7f5d0", "#ffe0b3", "#ffd0d0", "#eeeeee"
    box(
        X["a"],
        -0.6,
        5.0,
        "embedding:\nlin + mod 5\n\nL0-L4 MLPs:\nvalue-bin\ndetectors\n(single values,\nwindows,\nmod 10/50/5)",
        blue,
        fs=7.5,
    )
    box(X["a"], 11, 15.5, "L11-L15 MLPs\nre-write the\nmod 10/5/2\ncodes", blue)
    box(X["b"], -0.4, 3.0, "same for b\n(embedding +\nL0-L4 MLPs)", blue)
    box(X["b"], 5.5, 10, "copies of a\n(L1H24, L5H22)\na==b, a>b\ncomparators", grey)
    box(X["b"], 11, 14.5, "L12-L14 MLPs\nre-write b", blue)
    box(X["op"], -0.4, 2.8, "op-specific\ncomponents", orange)
    box(
        X["="],
        4.5,
        14.0,
        "op FLAG relay:\n1-4 MLP comps\nper layer, on\nfor exactly one\noperation",
        orange,
        w=1.4,
    )
    arrow(X["op"], 3.0, X["="] - 0.75, 4.0, "L0H23 L1H6 L2H2", color="tab:orange", dy=0.9)
    arrow(
        X["b"] + 0.55, 14.4, X["="] - 0.75, 14.4, "L15H13: b (mod 2/5/10, digits)", color="tab:blue"
    )
    arrow(X["a"] + 0.55, 16, X["="] - 0.75, 16, "L16H21: a", color="tab:blue", dy=0.7)
    box(X["="] + 0.9, 14.7, 15.6, "L15 MLP: on sub, mirror b -> -b", orange, w=2.2)
    box(
        X["="],
        16.2,
        18.8,
        "L16-L18 MLPs: digit-class AND gates,\nparity XOR, digit-sum bands\n-> first result code (mod 2, 5, 10, 20, 50)",
        green,
        w=3.0,
    )
    box(
        X["="],
        19.2,
        24.8,
        "L19-L24 MLPs: result codes\nmod 10/20 (units digit),\nmod 50/100; a+b stripes on add,\nthe SAME comps give a-b stripes on sub",
        green,
        w=3.0,
    )
    box(
        X["="],
        25.2,
        31.2,
        "L25-L31 MLPs: result-WINDOW detectors\n(a+b mod 100 in a narrow window,\ntens-digit windows, exact values)\nwrite straight into the number logits",
        red,
        w=3.0,
    )
    ax.text(7.0, 29.5, "direct logit effects:\n- mod-10 family: every token\n  with the right units digit\n- mod-100 family: right last\n  two digits (x and x+-100)\n- magnitude families: a\n  +-2 / +-30 window around\n  the answer (resolve hundreds)",
            fontsize=8.5, va="center")  # fmt: skip
    ax.axis("off")
    ax.set_title(
        "How the decomposition says Llama-3.1-8B computes a +/- b = (one reading of the alive components)",
        fontsize=12,
    )
    ctx.save(fig, "schematic")


FIGS = {
    "schematic": fig_schematic,
    "curated_grids": fig_curated_grids,
    "digit_plane": fig_digit_plane,
    "code_attrib": fig_code_attrib,
    "pos1_onsets": fig_pos1_onsets,
    "attention_heads": fig_attention,
    "result_switch": fig_result_switch,
    "op_mirror": fig_op_phase,
    "output_families": fig_output,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    ctx = Ctx(args.run)
    for name, fn in FIGS.items():
        if args.only is None or name in args.only:
            fn(ctx)


if __name__ == "__main__":
    main()
