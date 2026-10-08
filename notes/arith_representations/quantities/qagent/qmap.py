"""Map of the accepted quantities over the sites of each token position (for the report).

One panel per position t: rows are quantities (grouped by the variable they are a function of),
columns are the read sites in stream order (L0.attn, L0.mlp, ..., L31.mlp). A cell is coloured by
the quantity's drop-one excess at that site (the variance of the reads it alone explains, beyond
the others and beyond chance, as a fraction of the reads' variance; log scale), empty where the
quantity is not in the site's accepted list; a dot marks dependence > 0.9 (its features are
reproduced by the other accepted quantities, so its directions are not separable from theirs).

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


if __name__ == "__main__":
    main()
