"""Figures of the flip report (notes/arith_representations/report_flip_mechanism.md), from saved results.

    python -m param_decomp.arith_repr.vectors.flip_report_figs   # -> OUT/flip/report_figs/*.png

* pipeline.png: the components of the mechanism, layer by layer (a diagram, no data);
* flip_share.png: flip share Phi / (Phi + Sig) per reader layer, alive-only vs original model
  (alive_baseline.json);
* patching.png: share of answers that follow the partner prompt when the stream at one position and
  layer is copied from it, alive-only model, pairs with a > b (resid_comp_all.json);
* removals.png: flip left at the L16 readers (Phi / unablated Phi) after removals
  (eval_alive.json, core.json);
* swap.png: the flip-swap test (flipswap.json).
The Fourier-plane figures are drawn by `flip_fig`."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from param_decomp.arith_repr.vectors.flip_components import DIR

ADD, SUB = "#2a78d6", "#eb6834"
POS_C = {"op": "#4a3aa7", "b": "#1baf7a", "=": "#eda100"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
OUT = DIR / "report_figs"


def style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def flip_share() -> None:
    b = json.loads((DIR / "alive_baseline.json").read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for key, colr, nm in (("alive", ADD, "alive-only model"), ("dense", MUTED, "original model")):
        fl = b[key]["flip"]
        ls = sorted(int(x) for x in fl)
        y = [fl[str(li)][0] / (fl[str(li)][0] + fl[str(li)][1]) for li in ls]
        ax.plot(ls, y, "-o", color=colr, lw=2, ms=4)
        ax.annotate(
            nm,
            (ls[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=8,
            va="center",
        )
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.annotate("0.5: flip as large as the shared part", (31, 0.5), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=7, ha="right")  # fmt: skip
    ax.set_xticks(range(16, 32, 2))
    ax.set_xlabel("layer whose MLP readers look at `=`", color=INK, fontsize=9)
    ax.set_ylabel("flip share Φ / (Φ + Σ)", color=INK, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(15.5, 34)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "flip_share.png", dpi=150)


def patching() -> None:
    r = json.loads((DIR / "resid_comp_all.json").read_text())
    pts = sorted({int(k.split("|")[0]) for k in r if "|" in k})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, (key, title) in zip(axes, (("swap_add_gt", "addition prompt → answer becomes the subtraction answer"),
                                       ("swap_sub_gt", "subtraction prompt → answer becomes the addition answer")), strict=True):  # fmt: skip
        for pos, lab in (("op", "op token"), ("b", "b token"), ("=", "`=` token")):
            y = [r[f"{t}|{pos}"][key] for t in pts]
            x = [t / 2 for t in pts]
            ax.plot(x, y, "-", color=POS_C[pos], lw=2)
            anchor = {"op": (x[1], y[1]), "b": (x[-1], y[-1]), "=": (x[-1], y[-1])}[pos]
            off = {"op": (8, -2), "b": (-4, 6), "=": (-4, -14)}[pos]
            ax.annotate(lab, anchor, xytext=off, textcoords="offset points", color=INK, fontsize=8,
                        ha="left" if pos == "op" else "right")  # fmt: skip
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(
            "layer at whose input the stream is copied from the partner prompt",
            fontsize=8,
            color=INK,
        )
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("share of prompts (a > b)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "patching.png", dpi=150)


def removals() -> None:
    e = json.loads((DIR / "eval_alive.json").read_text())
    c = json.loads((DIR / "core.json").read_text())
    base16 = e["base"]["flip"]["16"][0]
    rows = [(nm.replace("L15.", "").replace("@=", ""), e[nm]["flip"]["16"][0] / base16)
            for nm in ("L15.gate.c0@=", "L15.up.c117@=", "L15.down.c4@=", "L15.gate.c72@=", "L15.down.c16@=", "L15.up.c34@=")]  # fmt: skip
    rows.append(
        ("4 switches, path through the 8 flip neurons", c["all switches via F8"]["phi"]["16"])
    )
    rows.append(("4 switches + down c16 + down c4", c["core off"]["phi"]["16"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    ax.barh(names, vals, color=ADD, height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(
            f"{v:.2f}",
            (v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xlim(0, 1)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("flip left at the L16 readers (1 = unablated)", fontsize=9, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "removals.png", dpi=150)


def swap() -> None:
    f = json.loads((DIR / "flipswap.json").read_text())
    conds = (("base", "no edit"), ("clamp16", "flip removed"), ("swap16", "flip reversed"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
    for ax, (op, title) in zip(
        axes, (("add", "addition prompts"), ("sub", "subtraction prompts")), strict=True
    ):
        for i, (key, _) in enumerate(conds):
            other, same = f[key][f"swap_{op}"], f[key][f"same_{op}"]
            ax.barh(i, same, color=ADD if op == "add" else SUB, height=0.55)
            ax.barh(i, other, left=same, color=SUB if op == "add" else ADD, height=0.55)
            ax.annotate(f"own {same:.2f} · other op {other:.3f}", (same + other, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)  # fmt: skip
        ax.set_yticks(range(3), [c[1] for c in conds])
        ax.set_xlim(0, 1.45)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("share of prompts whose top-1 equals the unedited model's answer on\n"
                      "the prompt itself (own) / on the partner prompt (other op)", fontsize=8, color=INK)  # fmt: skip
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "swap.png", dpi=150)


def pipeline() -> None:
    """Boxes per stage, left to right; text only, no data."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = {
        "op": (
            0.1,
            3.0,
            1.9,
            1.3,
            "op token (+ / −)\nL0 v c2, v c23\nL2 k c93, k c11",
            POS_C["op"],
        ),
        "relay": (
            2.3,
            3.0,
            2.0,
            1.3,
            "op flag at `=`\nrelays L3 down c28,\nL4 c16, L7 c23, L13 c127",
            POS_C["op"],
        ),
        "bcopy": (
            2.3,
            0.4,
            2.0,
            1.3,
            "b's code at `=`\ncopied from b's token\nby L15 attention",
            POS_C["b"],
        ),
        "switch": (
            4.7,
            3.0,
            2.0,
            1.3,
            "L15 op switches\ngate c72, up c117 (sub)\nup c34 (add), gate c0",
            POS_C["op"],
        ),
        "reader": (
            4.7,
            0.4,
            2.0,
            1.3,
            "L15 b readers (sine axis)\nup c16, up c64, gate c21,\ngate c4, gate c35",
            POS_C["b"],
        ),
        "neuron": (7.0, 1.7, 1.7, 1.3, "8 L15 neurons\nact = silu(gate) × up\n= b × op sign", INK),
        "writer": (
            9.0,
            1.7,
            1.9,
            1.3,
            "L15 down writers\nc16 (add) / c64 (sub) T5\nc21 T10, c4 T100,\nc19, c35 T50",
            INK,
        ),
    }
    for _, (x, y, w, h, txt, colr) in boxes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                fc="white",
                ec=colr,
                lw=1.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8, color=INK)

    def arrow(a: str, b: str) -> None:
        xa, ya, wa, ha_, *_ = boxes[a]
        xb, yb, _, hb, *_ = boxes[b]
        ax.add_patch(FancyArrowPatch((xa + wa, ya + ha_ / 2), (xb, yb + hb / 2), arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, lw=1.4, connectionstyle="arc3,rad=0"))  # fmt: skip

    for a, b in (
        ("op", "relay"),
        ("relay", "switch"),
        ("bcopy", "reader"),
        ("switch", "neuron"),
        ("reader", "neuron"),
        ("neuron", "writer"),
    ):
        arrow(a, b)
    ax.text(9.95, 1.35, "→ L16–L18 MLP readers of b at the same periods\n(the adder: a's code × b's code)",
            ha="center", va="top", fontsize=8, color=INK)  # fmt: skip
    ax.text(5.7, 2.55, "also: an operation constant through other\nneurons (9816 → down c6, 10519)",
            ha="center", va="center", fontsize=7, color=MUTED)  # fmt: skip
    ax.text(
        0.1,
        0.1,
        "violet: carries the operation · green: carries b · black: their product",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=150)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for f in (pipeline, flip_share, patching, removals, swap):
        f()
        print("saved", f.__name__, flush=True)


if __name__ == "__main__":
    main()
