"""Report figures: the map of accepted quantities per position, and the patching results.

qmap_t<t>.png:

One panel per position t: rows are quantities (grouped by the variable they are a function of),
columns are the read sites in stream order (L0.attn, L0.mlp, ..., L31.mlp). A cell is coloured by
the quantity's drop-one excess at that site (the variance of the reads it alone explains, beyond
the others and beyond chance, as a fraction of the reads' variance; log scale), empty where the
quantity is not in the site's accepted list; a dot marks dependence > 0.9 (its features are
reproduced by the other accepted quantities, so its directions are not separable from theirs).

patching.png (from patch_t<t>_alive.json, qpatch.py): one row per position t. Left: KL at the last
position against the unpatched alive-only model when one site's readers are patched at t (dead-
at-t readers off in every run), per site in stream order, for the mean patch (the readers' mean
over the domain), the reconstruction from the accepted quantities, and the held-out
reconstruction. Right: every site of the position patched together; the switch of the dead-at-t
readers alone for reference.

    python qmap.py [--tag name] [--out dir]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from qdata import DATA, POS_NAMES

AMBIGUOUS = 0.9


def group(name: str) -> tuple[int, str]:
    """Sort key: op first, then quantities of a, b, s/d/r and carry/borrow, op-gated last within each."""
    gated = name.startswith("[op = -] x ")
    base = name.removeprefix("[op = -] x ")
    if base.startswith("op "):
        g = 0
    elif any(k in base for k in ("a + b", "a - b", " of r", " r ", "carry", "borrow")):
        g = 3
    elif " of a" in base or base.endswith(" a") or base.startswith(("a ", "log a", "one digit [a")):
        g = 1
    elif " of b" in base or base.endswith(" b") or base.startswith(("b ", "log b", "one digit [b")):
        g = 2
    else:
        g = 3
    return (2 * g + gated, base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "figures"))
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    labels = [f"L{c}.{p}" for c in range(32) for p in ("attn", "mlp")]
    for t in (1, 2, 3, 4):
        run = json.loads((DATA / f"qloop_t{t}{tag}.json").read_text())
        dep = {(s["site"], q["name"]): q["dependence"]
               for s in json.loads((DATA / f"dirs_t{t}{tag}.json").read_text())["sites"] for q in s["quantities"]}  # fmt: skip
        cell = {}
        for s in run["sites"]:
            for a in s["accepted"]:
                cell[(a["name"], s["site"])] = max(a["drop_one_loss"] - a["chance"], 1e-5)
        names = sorted({n for n, _ in cell}, key=group)
        M = np.full((len(names), len(labels)), np.nan)
        for (n, lab), v in cell.items():
            M[names.index(n), labels.index(lab)] = v
        fig, ax = plt.subplots(figsize=(15, 0.24 * len(names) + 1.6))
        im = ax.imshow(
            M, aspect="auto", cmap="viridis", norm=LogNorm(1e-4, 1), interpolation="nearest"
        )
        ys, xs = zip(*[(names.index(n), labels.index(lab)) for (lab, n), d in dep.items() if d > AMBIGUOUS], strict=True) if any(d > AMBIGUOUS for d in dep.values()) else ((), ())  # fmt: skip
        ax.scatter(xs, ys, s=4, c="white", marker="o", linewidths=0)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_xticks(range(0, len(labels), 2), [f"L{c}" for c in range(32)], fontsize=6)
        ax.set_xticks(np.arange(-0.5, len(labels), 2), minor=True)
        ax.grid(which="minor", axis="x", color="0.85", lw=0.3)
        consts = [s["site"] for s in run["sites"] if s.get("constant")]
        ax.set_title(f"t = {t} (token {POS_NAMES[t]}): accepted quantities at each read site (attention input, then MLP input, per block)"
                     + (f"; constant, not fit: {', '.join(consts)}" if consts else ""), fontsize=8)  # fmt: skip
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
        cb.set_label("drop-one excess (share of the reads' variance)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.tight_layout()
        out = Path(args.out) / f"qmap_t{t}{tag}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(out)
    print(patch_figure(tag, Path(args.out)))


def patch_figure(tag: str, out_dir: Path) -> Path:
    labels = [f"L{c}.{p}" for c in range(32) for p in ("attn", "mlp")]
    variants = (
        ("mean", "mean patch", "0.6"),
        ("recon", "reconstruction", "C0"),
        ("heldout", "held-out reconstruction", "C1"),
    )
    fig, axes = plt.subplots(
        4, 2, figsize=(15, 11), gridspec_kw={"width_ratios": [5, 1]}, squeeze=False
    )
    for row, t in enumerate((1, 2, 3, 4)):
        r = json.loads((DATA / f"patch_t{t}_alive{tag}.json").read_text())
        ax, bx = axes[row]
        for v, lab, col in variants:
            xs = [labels.index(s) for s in r["sites"]]
            ys = [max(e[v]["kl_mean"], 1e-6) for e in r["sites"].values()]
            ax.plot(xs, ys, "o", ms=3, color=col, label=lab)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3)
        ax.set_xticks(range(0, len(labels), 2), [f"L{c}" for c in range(32)], fontsize=6)
        ax.set_xlim(-1, len(labels))
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_ylabel("KL (mean over prompts)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"t = {t} (token {POS_NAMES[t]}): one site patched at a time", fontsize=8)
        bars = [("dead-at-t\nreaders off", r["masked_only"]["kl_mean"], "0.3")]
        bars += [
            (lab.replace(" ", "\n", 1), r["all_sites_" + v]["kl_mean"], col)
            for v, lab, col in variants
        ]
        bx.bar(range(len(bars)), [max(b[1], 1e-6) for b in bars], color=[b[2] for b in bars])
        for i, b in enumerate(bars):
            bx.text(i, max(b[1], 1e-6) * 1.3, f"{b[1]:.2g}", ha="center", fontsize=6)
        bx.set_yscale("log")
        bx.set_ylim(1e-6, 10)
        bx.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=5.5)
        bx.tick_params(labelsize=6)
        bx.set_title("all sites together", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Closed patching test, alive-only model, 2000 prompts: KL at the last position against the unpatched model",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / f"patching{tag}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
