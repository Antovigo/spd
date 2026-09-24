"""Tables and figures of `mlp_linearity` (per-layer split of each MLP's write) and `mlp_patch`
(linearised forwards), for report_mechanisms.md.

    python -m param_decomp.arith_repr.autointerp.mlp_lin_report --run <run_dir> --resid <dir>

Adds, per layer, where the gates that switch sit relative to the classes of the operand at its
own token: for every neuron whose gate crosses zero on the population (2-98 % of the prompts
have g > 0), `eta` = share of the variance of [g > 0] that lies between the classes of the
variable (1 = the gate is on for whole classes and off for the others: it flips between
classes, never within one). `eta_w` is its mean weighted by the neuron's |share| of the silu
term in the per-value write (a%100 / b%100), i.e. over the switches that matter.
Writes `mech/mlp_lin.parquet`, `mech/mlp_flips.parquet`, `mech/mlp_patch.parquet` and
`figs/mech_mlp_linear.png`, `figs/mech_mlp_patch.png`."""

import argparse
import glob
import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.autointerp.mechanisms import AI
from param_decomp.arith_repr.autointerp.mlp_linearity import value
from param_decomp.arith_repr.resid import load_read_input

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK2 = "#0b0b0b", "#52514e"
FLIP_VARS = {"a": ("a%10", "a//10", "a%2"), "b": ("b%10", "b//10", "b%2")}


def eta(on: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """(neurons,) between-class share of the variance of the 0/1 columns of `on` (n, neurons)."""
    keys, inv = np.unique(cls, return_inverse=True)
    onehot = np.zeros((keys.size, len(cls)), np.float32)
    onehot[inv, np.arange(len(cls))] = 1
    n = onehot.sum(1)
    cm = (onehot @ on) / n[:, None]
    mu = on.mean(0)
    between = (n[:, None] * (cm - mu) ** 2).sum(0) / len(cls)
    return between / np.maximum(on.var(0), 1e-12)


def flips(run: Path, resid: Path) -> pd.DataFrame:
    w = Weights()
    meta = json.loads((resid / "pool.json").read_text())
    labels = np.array([(0 if o == "add" else 1, a, b) for o, a, b in meta["labels"]])
    lin = run / AI / "mech" / "mlp_lin"
    rows = []
    for layer in range(w.config["num_hidden_layers"]):
        Wg = w.get(f"model.layers.{layer}.mlp.gate_proj.weight")
        st = np.load(lin / f"L{layer}.npz")
        for pname, p in (("a", 1), ("b", 3)):
            x = load_read_input(resid, f"mlp_in.{layer}", p)
            for o in (0, 1):
                sel = labels[:, 0] == o
                if pname == "a":
                    sel &= labels[:, 2] == 1
                on = (x[sel] @ Wg.T > 0).astype(np.float32)
                frac = on.mean(0)
                cross = (frac > 0.02) & (frac < 0.98)
                op = "both" if pname == "a" else ("add", "sub")[o]
                silu = np.abs(st[f"{pname}.{op}.{pname}%100.neuron_share"][:, 2])
                row: dict[str, object] = dict(
                    layer=layer, pos=pname, op=op, n_cross=int(cross.sum())
                )
                for var in FLIP_VARS[pname]:
                    e = eta(on[:, cross], value(labels[sel], var))
                    wt = silu[cross]
                    row[f"eta_{var}"] = float(e.mean())
                    row[f"eta_w_{var}"] = float((e * wt).sum() / wt.sum())
                    row[f"pure_{var}"] = float((e > 0.99).mean())
                rows.append(row)
                print(row, flush=True)
                if pname == "a":
                    break
    return pd.DataFrame(rows)


def patch_table(run: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((run / AI / "mech" / "mlp_patch").glob("*.npz")):
        if f.stem.startswith("test"):
            continue
        z = np.load(f)
        names = sorted({k.rsplit(".", 1)[0] for k in z.files if k.endswith(".kl")})
        op = z["labels"][:, 0]
        for name in names:
            kind, poss, layers = name.split(":") if name != "clean" else ("clean", "", "")
            for o, m in (("both", np.ones(len(op), bool)), ("add", op == 0), ("sub", op == 1)):
                rows.append(
                    dict(
                        run=f.stem,
                        variant=name,
                        kind=kind,
                        positions=poss,
                        layers=layers,
                        op=o,
                        kl=float(z[f"{name}.kl"][m].mean()),
                        agree=float((z[f"{name}.top"][m] == z["clean.top"][m]).mean()),
                        acc=float((z[f"{name}.top"][m] == z["answer"][m]).mean()),
                        n=int(m.sum()),
                    )
                )
    return pd.DataFrame(rows)


def fig_linear(d: pd.DataFrame, figs: Path) -> None:
    panels = [
        ("a", "both", "a%10", "a mod 10 at `a`"),
        ("b", "add", "b%10", "b mod 10 at `b` (add)"),
        ("=", "add", "res%10", "res mod 10 at `=` (add; contrast, from L15)"),
    ]
    terms = [
        ("share_up", "up (linear)"),
        ("share_gate", "gate read (linear)"),
        ("share_silu", "silu bend (switching)"),
        ("share_prod", "gate × up (product)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), sharex="col")
    for j, (p, o, v, title) in enumerate(panels):
        x = cast(pd.DataFrame, d[(d.pos == p) & (d.op == o) & (d["var"] == v)])
        x = x.sort_values("layer")
        if p == "=":  # no result code at `=` before L16: the shares of a ~0 write mean nothing
            x = x[x.layer >= 15]
        ax = axes[0, j]
        for (c, lab), col in zip(terms, SERIES, strict=True):
            ax.plot(x.layer, x[c], color=col, lw=2, label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_ylim(-0.45, 1.0)
        ax = axes[1, j]
        ax.plot(x.layer, x.r2_lin, color=SERIES[0], lw=2, label="one linear map (up + gate read)")
        ax.plot(x.layer, x.r2_noflip, color=SERIES[1], lw=2, label="no switching (silu as a line)")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("layer")
        for a_ in axes[:, j]:
            a_.grid(alpha=0.25, lw=0.5)
            a_.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("share of the MLP's write\nto the arrangement")
    axes[1, 0].set_ylabel("fraction of the write\nreproduced (R²)")
    fig.tight_layout(h_pad=3.5)
    axes[0, 1].legend(
        frameon=False, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.06)
    )
    axes[1, 1].legend(
        frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16)
    )
    fig.savefig(figs / "mech_mlp_linear.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_patch(pt: pd.DataFrame, figs: Path) -> None:
    x = pt[(pt.op == "both") & (pt.kind != "clean")]
    kinds = [
        ("lin", "one linear map"),
        ("sl", "no switching (silu as a line)"),
        ("mean", "MLP write removed (mean)"),
    ]
    panels = [
        ("all", [(p, "all") for p in ("a", "b", "a,b", "=")], "all 32 MLPs at one position"),
        (
            "blocks",
            [(p, r) for p in ("a", "b") for r in ("0-7", "8-15", "16-23", "24-31")],
            "one block of 8 MLPs at one position",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    hgt = 0.26
    for ax, (run, rows, title) in zip(axes, panels, strict=True):
        ys = np.arange(len(rows))
        for k, ((kind, lab), col) in enumerate(zip(kinds, SERIES, strict=False)):
            kl = []
            for p, layers in rows:
                m = (x.run == run) & (x.kind == kind) & (x.positions == p) & (x.layers == layers)
                v = cast(pd.DataFrame, x[m])
                kl.append(max(float(v.kl.iloc[0]), 1e-4) if len(v) else np.nan)
            ax.barh(ys + (k - 1) * hgt, kl, hgt * 0.9, color=col, label=lab)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 3)
        ax.set_yticks(
            ys, [f"`{p}`" + ("" if layers == "all" else f"  L{layers}") for p, layers in rows]
        )
        ax.invert_yaxis()
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel("KL(clean ‖ patched) of the answer at `=`, nats (log)")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.savefig(figs / "mech_mlp_patch.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--skip_flips", action="store_true")
    args = parser.parse_args()
    mech = args.run / AI / "mech"
    figs = args.run / AI / "figs"
    d = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(str(mech / "mlp_lin" / "L*.parquet")))]
    )
    d.to_parquet(mech / "mlp_lin.parquet")
    if not args.skip_flips:
        flips(args.run, args.resid).to_parquet(mech / "mlp_flips.parquet")
    pt = patch_table(args.run)
    pt.to_parquet(mech / "mlp_patch.parquet")
    fig_linear(d, figs)
    if len(pt) and (pt.run == "all").any():
        fig_patch(pt, figs)
        print(pt[pt.op == "both"].to_string(), flush=True)


if __name__ == "__main__":
    main()
