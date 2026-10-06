"""Figures of the attention-pattern comparison, from OUT/summary.npz, OUT/<cond>_P.npy and
OUT/patch.npz, into OUT/figures. Example heads are passed as `L<l>H<h>` arguments:
`python -m ...figures agree=L18H30,L0H5 disagree=L20H1 grid=L18H30,L20H1`."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from param_decomp.arith_repr.attn_patterns.extract import OUT, H, L, T
from param_decomp.arith_repr.vectors.common import DATASET, POS_NAMES

FIG = OUT / "figures"
C_ORIG, C_DEC, C_LOC = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
QNAMES = POS_NAMES[1:]  # query rows a, op, b, = (the BOS row is trivially [1])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 150, "savefig.bbox": "tight"})  # fmt: skip


def head(s: str) -> tuple[int, int]:
    li, h = s[1:].split("H")
    return int(li), int(h)


def prompt_text(n: int, ix: dict[str, np.ndarray]) -> str:
    return f"{ix['a'][n]}{'+-'[ix['op'][n]]}{ix['b'][n]}="


def overview(S: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey=True)
    im = None
    for r, c in enumerate(("dec", "loc")):
        for j, q in enumerate(range(1, T)):
            ax = axes[r, j]
            im = ax.imshow(S[f"tv_{c}"][:, :, q], cmap="Blues", vmin=0, vmax=1, origin="lower",
                           aspect="auto", interpolation="nearest")  # fmt: skip
            ax.set_title(f"{'decomposed' if c == 'dec' else 'components on original stream'}\nquery = {QNAMES[j]}", fontsize=8)  # fmt: skip
            if j == 0:
                ax.set_ylabel("layer l")
            if r == 1:
                ax.set_xlabel("head h")
    assert im is not None
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean TV distance to the original row")
    fig.savefig(FIG / "fig1_overview_tv.png")
    plt.close(fig)


def layers(S: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(L)
    for ax, (title, qs) in zip(axes, (("query = '='", [4]), ("all queries a, op, b, =", [1, 2, 3, 4])), strict=True):  # fmt: skip
        ax.plot(
            x, S["tv_dec"][:, :, qs].mean((1, 2)), color=C_DEC, lw=2, label="decomposed, head mean"
        )
        ax.plot(
            x,
            S["tvw_dec"][:, qs].mean(1),
            color=C_DEC,
            lw=2,
            ls="--",
            label="decomposed, weighted by original write",
        )
        ax.plot(
            x,
            S["tvwd_dec"][:, qs].mean(1),
            color=C_DEC,
            lw=2,
            ls=":",
            label="decomposed, weighted by decomposed write",
        )
        ax.plot(
            x,
            S["tv_loc"][:, :, qs].mean((1, 2)),
            color=C_LOC,
            lw=2,
            label="components on original stream, head mean",
        )
        ax.set_title(title)
        ax.set_xlabel("layer l")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e5e4df", lw=0.6)
    axes[0].set_ylabel("TV distance to the original row")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(FIG / "fig2_layers_tv.png")
    plt.close(fig)


def importance(S: dict[str, np.ndarray], marks: dict[str, list[str]]) -> None:
    q = 4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    W, tv, dW, Wd = (
        S["W_orig"][:, :, q],
        S["tv_dec"][:, :, q],
        S["dW"][:, :, q],
        S["W_dec"][:, :, q],
    )
    sc = axes[0].scatter(
        W.ravel(), tv.ravel(), c=np.repeat(np.arange(L), H), cmap="viridis", s=10, lw=0
    )
    axes[0].set_xscale("log")
    axes[0].set_xlim(right=W.max() * 4)
    axes[0].set_xlabel("original head write norm at '=' (mean over prompts)")
    axes[0].set_ylabel("TV distance, decomposed vs original, query '='")
    fig.colorbar(sc, ax=axes[0], label="layer l")
    axes[1].scatter(
        Wd.ravel() + 1e-3,
        dW.ravel() + 1e-3,
        c=np.repeat(np.arange(L), H),
        cmap="viridis",
        s=10,
        lw=0,
    )
    lim = [1e-3, max(Wd.max(), dW.max()) * 1.5]
    axes[1].plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("decomposed head write norm at '=' (+1e-3)")
    axes[1].set_ylabel("change of that write with the original pattern (+1e-3)")
    for hs in marks.values():
        for s in hs:
            li, h = head(s)
            for ax, (xv, yv) in zip(axes, ((W[li, h], tv[li, h]), (Wd[li, h] + 1e-3, dW[li, h] + 1e-3)), strict=True):  # fmt: skip
                ax.scatter([xv], [yv], s=40, facecolor="none", edgecolor=INK, lw=1)
                ax.annotate(
                    s, (xv, yv), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK
                )
    fig.savefig(FIG / "fig3_importance.png")
    plt.close(fig)


def sink(S: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    for j, q in enumerate(range(1, T)):
        o, d = S["meanP_orig"][:, :, q, 0].ravel(), S["meanP_dec"][:, :, q, 0].ravel()
        ax = axes[j]
        ax.hist2d(o, d, bins=40, range=[[0, 1], [0, 1]], cmap="Blues", norm=LogNorm())
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax.axhline(1 / (q + 1), color=C_DEC, lw=0.8, ls="--")
        ax.set_title(f"query = {QNAMES[j]}  (dashed: uniform 1/{q + 1})", fontsize=8)
        ax.set_xlabel("original attention to <BOS>")
    axes[0].set_ylabel("decomposed attention to <BOS>")
    fig.savefig(FIG / "fig4_bos_mass.png")
    plt.close(fig)


def _pat(ax: plt.Axes, P: np.ndarray, title: str, show_y: bool) -> None:
    M = np.ma.masked_where(~np.tril(np.ones((T, T), bool)), P)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for i in range(T):
        for k in range(i + 1):
            ax.text(k, i, f"{P[i, k]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if P[i, k] > 0.6 else INK)  # fmt: skip
    ax.set_xticks(range(T), POS_NAMES, fontsize=7, rotation=45)
    ax.set_yticks(range(T), POS_NAMES if show_y else [], fontsize=7)
    ax.set_title(title, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def gallery(name: str, hs: list[str], S: dict[str, np.ndarray], P: dict[str, np.ndarray],
            ix: dict[str, np.ndarray], n_ex: int) -> None:  # fmt: skip
    fig, axes = plt.subplots(len(hs), 5, figsize=(13, 3.3 * len(hs)), squeeze=False)
    fig.subplots_adjust(hspace=0.75, wspace=0.25)
    for r, s in enumerate(hs):
        li, h = head(s)
        for j, c in enumerate(("orig", "dec", "loc")):
            lab = {"orig": "original", "dec": "decomposed", "loc": "comps on orig. stream"}[c]
            _pat(axes[r, j], S[f"meanP_{c}"][li, h], f"{s} {lab}\nmean over 20000 prompts", j == 0)
        for j, c in enumerate(("orig", "dec")):
            lab = {"orig": "original", "dec": "decomposed"}[c]
            _pat(axes[r, 3 + j], np.asarray(P[c][li, n_ex, h], np.float32),
                 f"{s} {lab}\nprompt '{prompt_text(n_ex, ix)}'", False)  # fmt: skip
        axes[r, 4].text(5.3, 2, f"TV at '=': {S['tv_dec'][li, h, 4]:.2f}\n"
                        f"orig. write |W|: {S['W_orig'][li, h, 4]:.2f}\n"
                        f"dec. write |W|: {S['W_dec'][li, h, 4]:.2f}\n"
                        f"swap change: {S['dW'][li, h, 4]:.2f}\n"
                        f"R2 at '=': {S['r2_dec'][li, h, 4]:.2f}", fontsize=7, va="center", color=INK)  # fmt: skip
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def grid(hs: list[str], P: dict[str, np.ndarray]) -> None:
    """The '=' row's weight on the a and b tokens over the (a, b) grid, addition block."""
    fig, axes = plt.subplots(len(hs), 4, figsize=(15, 3.3 * len(hs)), squeeze=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.35)
    for r, s in enumerate(hs):
        li, h = head(s)
        for j, (c, key) in enumerate((("orig", 1), ("dec", 1), ("orig", 3), ("dec", 3))):
            M = np.asarray(P[c][li, :10000, h, 4, key], np.float32).reshape(100, 100)
            ax = axes[r, j]
            vmax = float(np.asarray(P["orig"][li, :10000, h, 4, key], np.float32).max())
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(vmax, 1e-3), origin="lower",
                           extent=(0.5, 100.5, 0.5, 100.5), interpolation="nearest")  # fmt: skip
            ax.set_title(f"{s} {'original' if c == 'orig' else 'decomposed'}\nweight of '=' on {POS_NAMES[key]}", fontsize=8)  # fmt: skip
            ax.set_xlabel("b")
            if j == 0:
                ax.set_ylabel("a")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "fig7_ab_grid.png")
    plt.close(fig)


def patch() -> None:
    """KL(original || variant) minus KL(original || decomposed) for the pattern-patching variants."""
    R = dict(np.load(OUT / "patch.npz"))
    Rh = dict(np.load(OUT / "patch_heads.npz")) if (OUT / "patch_heads.npz").exists() else None
    base = R["dec"].mean()
    n = 3 if Rh is not None else 2
    fig, axes = plt.subplots(
        1, n, figsize=(5.2 * n + 2, 3.8), gridspec_kw={"width_ratios": [2.2, 1, 1.6][:n]}
    )
    ax = axes[0]
    x = np.arange(L)
    ax.bar(x, [R[f"L{li}"].mean() - base for li in x], color=C_DEC, width=0.7)
    ax.axhline(
        R["all"].mean() - base,
        color=C_ORIG,
        lw=1,
        ls="--",
        label=f"all 32 blocks patched ({R['all'].mean() - base:+.4f})",
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("block l whose pattern is replaced by the original one")
    ax.set_ylabel(f"change of KL vs. unpatched decomposed ({base:.4f})")
    ax.legend(frameon=False, fontsize=7.5, loc="center left")
    ax = axes[1]
    ks = np.arange(0, L, 4)
    ax.plot(
        ks,
        [R[f"upto{k}"].mean() - base for k in ks],
        color=C_ORIG,
        lw=2,
        marker="o",
        ms=4,
        label="blocks 0..l patched",
    )
    ax.plot(
        ks,
        [R[f"from{k}"].mean() - base for k in ks],
        color=C_LOC,
        lw=2,
        marker="o",
        ms=4,
        label="blocks l..31 patched",
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("l")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    if Rh is not None:
        ax = axes[2]
        names = [k for k in Rh if k not in ("rows", "dec")]
        vals = [Rh[k].mean() - Rh["dec"].mean() for k in names]
        ax.barh(
            np.arange(len(names)),
            vals,
            color=[C_ORIG if "mean" in k else C_DEC for k in names],
            height=0.7,
        )
        ax.set_yticks(np.arange(len(names)), names, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("change of KL vs. unpatched decomposed")
        ax.set_title("single heads / prompt-mean patterns", fontsize=8)
    fig.savefig(FIG / "fig8_patch_kl.png")
    plt.close(fig)


def copy_tests() -> None:
    """The L15-16 split tests (`copy_tests.py`): KL change per variant, both models."""
    R = dict(np.load(OUT / "copy_patch.npz"))
    base = R["dec/none"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.6)
    ax = axes[0]
    subs = [
        ("all", "all\nkeys"),
        ("focus", "focus\nkey"),
        ("other", "other\nkeys"),
        ("bos", "<BOS>\nonly"),
        ("rest", "other keys\nexcept <BOS>"),
    ]
    x = np.arange(len(subs))
    for i, (src, col, lab) in enumerate(
        (
            ("dec_values", C_DEC, "decomposed values u_d"),
            ("orig_values", C_ORIG, "original values u_o"),
        )
    ):
        ax.bar(
            x + (i - 0.5) * 0.36,
            [R[f"dec/{k}/{src}"].mean() - base for k, _ in subs],
            width=0.34,
            color=col,
            label=lab,
        )
    ax.set_xticks(x, [lab for _, lab in subs])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel(f"KL change vs. unpatched decomposed ({base:.4f})")
    ax.set_title(
        "decomposed model + the original pattern's change on the keys S (4 heads, '=' row)",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax = axes[1]
    names = [("orig/swap_all_rows", "decomposed pattern, all rows"), ("orig/dec_row", "decomposed '=' row"),
             ("orig/focus_amp", "focus weight -> decomposed"), ("orig/drop_other", "drop other keys"),
             ("orig/drop_rest", "drop other non-<BOS> keys")]  # fmt: skip
    ax.barh(np.arange(len(names)), [R[k].mean() for k, _ in names], color=C_LOC, height=0.6)
    ax.set_yticks(np.arange(len(names)), [lab for _, lab in names], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(base, color=INK, lw=0.8, ls="--")
    ax.text(base, len(names) - 0.4, " decomposed model", fontsize=7, color=INK)
    ax.set_xlabel("KL(original || modified original)")
    ax.set_title("original model, 4 heads modified", fontsize=8)
    fig.savefig(FIG / "fig11_copy_tests.png")
    plt.close(fig)


def active(S: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for j, kind in enumerate(("q", "k", "v", "o")):
        ax = axes[j]
        for p in range(T):
            ax.plot(np.arange(L), S["n_active"][:, j, p], color=cols[p], lw=1.6, label=POS_NAMES[p])
        ax.plot(np.arange(L), S["n_alive"][:, j], color=MUTED, lw=1, ls=":", label="alive")
        ax.set_title(f"{kind} components", fontsize=9)
        ax.set_xlabel("layer l")
        ax.set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("active per position (mean over prompts)")
    axes[3].legend(frameon=False, fontsize=7, title="position", title_fontsize=7)
    fig.savefig(FIG / "fig9_active_components.png")
    plt.close(fig)


def prompt_dependence(S: dict[str, np.ndarray], marks: dict[str, list[str]]) -> None:
    """Per head, the prompt-to-prompt spread of the '=' row (root of the summed per-key variance)
    in the original vs the decomposed model."""
    q = 4
    so, sd = np.sqrt(S["var_orig"][:, :, q]), np.sqrt(S["var_dec"][:, :, q])
    keep = S["W_dec"][:, :, q] > 0.1
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(so[~keep], sd[~keep], s=8, color="#c3c2b7", lw=0, label="decomposed write < 0.1")
    sc = ax.scatter(so[keep], sd[keep], s=14, c=np.nonzero(keep)[0], cmap="viridis", vmin=0, vmax=L - 1,
                    lw=0, label="decomposed write >= 0.1")  # fmt: skip
    lim = [0, max(so.max(), sd.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=":")
    for hs in marks.values():
        for s in hs:
            li, h = head(s)
            ax.annotate(
                s, (so[li, h], sd[li, h]), xytext=(4, 3), textcoords="offset points", fontsize=7
            )
    ax.set_xlabel("original: spread of the '=' row across prompts")
    ax.set_ylabel("decomposed: spread of the '=' row across prompts")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.colorbar(sc, ax=ax, label="layer l")
    fig.savefig(FIG / "fig10_prompt_dependence.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    marks = {k: args[k].split(",") for k in ("agree", "disagree") if k in args}
    S = dict(np.load(OUT / "summary.npz"))
    P = {c: np.load(OUT / f"{c}_P.npy", mmap_mode="r") for c in ("orig", "dec", "loc")}
    ix = dict(np.load(DATASET / "index.npz"))
    n_ex = int(args.get("prompt", 3647))  # 37+48=
    overview(S)
    layers(S)
    sink(S)
    active(S)
    importance(S, marks)
    prompt_dependence(S, marks)
    for k, name in (("agree", "fig5_agree"), ("disagree", "fig6_disagree")):
        if k in marks:
            gallery(name, marks[k], S, P, ix, n_ex)
    if "grid" in args:
        grid(args["grid"].split(","), P)
    if (OUT / "patch.npz").exists():
        patch()
    if (OUT / "copy_patch.npz").exists():
        copy_tests()


if __name__ == "__main__":
    main()
