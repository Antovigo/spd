"""Figures of report_mechanisms.md (arrangements of residue groups, mechanisms, checks).

    python -m param_decomp.arith_repr.autointerp.mech_figures --run <run_dir>

Reads `mech/arrangements.parquet`, `mech/arr_fourier.npz`, `mech/mechanisms.parquet`,
`mech/codes.parquet`, `mech/members.parquet`, `mech/code_pairs.parquet`; writes `figs/mech_*.png`."""

import argparse
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from param_decomp.arith_repr.autointerp.mechanisms import AI

POS = ["<BOS>", "a", "op", "b", "="]
PERIODS = (2, 4, 5, 10, 20, 25, 50, 100)
GROUPS = [(1, "a"), (3, "b"), (4, "a"), (4, "b"), (4, "res")]
QI = {"a": 0, "b": 1, "res": 2}


def point_label(t: int) -> str:
    if t == 0:
        return "emb"
    layer, s = divmod(t - 1, 2)
    return f"L{layer}{'am'[s]}"


def save(fig: Figure, figs: Path, name: str) -> None:
    fig.savefig(figs / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def period_power(power: np.ndarray) -> np.ndarray:
    """(65, 51) power per k -> (65, len(PERIODS)) power per exact period 100 / gcd(k, 100)."""
    out = np.zeros((power.shape[0], len(PERIODS)))
    for k in range(1, 51):
        out[:, PERIODS.index(100 // np.gcd(k, 100))] += power[:, k]
    return out


def fig_code_power(mech: Path, figs: Path) -> None:
    fz = np.load(mech / "arr_fourier.npz")
    fig, axes = plt.subplots(len(GROUPS), 2, figsize=(15, 2.3 * len(GROUPS)), sharex=True)
    for r, (p, q) in enumerate(GROUPS):
        for o in (0, 1):
            ax = axes[r, o]
            pw = period_power(np.nan_to_num(fz[f"{p}_{o}_{QI[q]}.power"]))
            im = ax.imshow(
                np.log10(pw.T + 1e-5),
                aspect="auto",
                cmap="magma",
                vmin=-4,
                vmax=-0.5,
                origin="lower",
            )
            ax.set_yticks(range(len(PERIODS)), [str(x) for x in PERIODS])
            ax.set_ylabel(f"{q} @ {POS[p]}\nperiod")
            ax.set_title(f"{q} mod period at `{POS[p]}` ({'add' if o == 0 else 'sub'})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.02, label="log10 share of stream var.")
    ticks = list(range(0, 65, 4))
    for ax in axes[-1]:
        ax.set_xticks(ticks, [point_label(t) for t in ticks], rotation=90, fontsize=7)
    fig.suptitle(
        "Code strength per exact period (Fourier power of the residue class means / "
        "total variance of the raw residual), every residual point"
    )
    save(fig, figs, "mech_code_power")


def fig_geometry(mech: Path, figs: Path) -> None:
    t = pd.read_parquet(mech / "arrangements.parquet")
    specs = [
        (1, "a", 10),
        (3, "b", 10),
        (4, "a", 10),
        (4, "b", 10),
        (4, "res", 10),
        (4, "res", 100),
    ]
    cols = [
        ("share", "code share (log)"),
        ("circulant", "shift-symmetric part"),
        ("ordered", "ordered (corr w/ cyclic dist.)"),
        ("pr_rel", "dimension (PR / (tau-1))"),
        ("cka_prev", "same geometry as prev. point"),
        ("in_span_prev", "inside prev. span"),
    ]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.0 * len(cols)), sharex=True)
    for p, q, tau in specs:
        s = cast(pd.DataFrame, t[(t.p == p) & (t.o == 0) & (t.q == q) & (t.tau == tau)])
        s = s.sort_values("point")
        s = s.assign(pr_rel=s["pr"] / (tau - 1))
        for ax, (c, lab) in zip(axes, cols, strict=True):
            y = np.log10(s[c] + 1e-6) if c == "share" else s[c]
            ax.plot(s["point"], y, label=f"{q} mod {tau} @ {POS[p]}", lw=1.2)
            ax.set_ylabel(lab, fontsize=8)
    axes[0].legend(fontsize=7, ncol=6)
    ticks = list(range(0, 65, 2))
    axes[-1].set_xticks(ticks, [point_label(x) for x in ticks], rotation=90, fontsize=7)
    fig.suptitle("Shape of the residue arrangements over the residual points (add)")
    save(fig, figs, "mech_geometry")


def fig_snapshots(mech: Path, figs: Path) -> None:
    fz = np.load(mech / "arr_fourier.npz")
    rows = [
        (1, "a", [0, 2, 8, 24, 30, 34]),
        (3, "b", [0, 2, 8, 24, 30, 34]),
        (4, "a", [30, 32, 34, 36, 40, 64]),
        (4, "res", [34, 36, 38, 40, 52, 64]),
    ]
    fig, axes = plt.subplots(len(rows), 6, figsize=(16, 2.8 * len(rows)))
    cmap = plt.get_cmap("tab10")
    for r, (p, q, pts) in enumerate(rows):
        pcs = fz[f"{p}_0_{QI[q]}.pcs_10"]
        for c, t in enumerate(pts):
            ax = axes[r, c]
            xy = pcs[t]
            ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(v) for v in range(10)], s=30)
            for v in range(10):
                ax.annotate(str(v), xy[v, :2], fontsize=8, ha="center", va="bottom")
            ax.set_title(f"{q} mod 10 @ {POS[p]}, {point_label(t)}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", "datalim")
    fig.suptitle("Class means of the residues mod 10 on their top-2 principal axes (add)")
    save(fig, figs, "mech_snapshots")


def fig_map(mech: Path, figs: Path) -> None:
    m = pd.read_parquet(mech / "mechanisms.parquet")
    c = pd.read_parquet(mech / "codes.parquet")
    mem = pd.read_parquet(mech / "members.parquet")
    first = cast(pd.Series, mem.groupby("code")["layer"].min())
    fig, axes = plt.subplots(4, 1, figsize=(15, 18))
    for ax, p in zip(axes, (1, 2, 3, 4), strict=True):
        mm = cast(pd.DataFrame, m[(m.p == p) & (m.o == 0)])
        n_mem = cast(pd.Series, mm.groupby("var")["n_members"].sum())
        vars_ = cast(pd.Index, n_mem.sort_values(ascending=False).index[:18])
        for yi, var in enumerate(vars_):
            mv = cast(pd.DataFrame, mm[mm["var"] == var]).sort_values("layer_min")
            offs = np.linspace(-0.3, 0.3, len(mv)) if len(mv) > 1 else [0.0]
            for k, (off, (_, row)) in enumerate(zip(offs, mv.iterrows(), strict=True)):
                cs = c[c.mech == row.mech]
                xs = [first[x] for x in cs.code]
                y = [yi + off] * len(xs)
                col = f"C{k % 10}"
                if len(xs) > 1:
                    ax.plot(xs, y, color=col, lw=1.0, zorder=1)
                ax.scatter(xs, y, s=3 * cs.n_members.to_numpy() + 8, color=col, alpha=0.7, zorder=2)
        ax.set_yticks(range(len(vars_)), list(vars_), fontsize=7)
        ax.set_xlim(-0.5, 31.5)
        ax.set_title(
            f"position `{POS[p]}` (add): codes (dots, size ~ members) joined into "
            "mechanisms (lines)",
            fontsize=9,
        )
        ax.set_xlabel("layer")
    save(fig, figs, "mech_map")


def with_classes(pairs: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Class count of each code pair's variable (from the mechanisms table if not stored)."""
    if "nc" in pairs:
        return pairs
    nc = m[["p", "o", "var", "n_classes"]].drop_duplicates()
    out = pairs.merge(nc, on=["p", "o", "var"], how="left")
    return out.rename(columns={"n_classes": "nc"})


def fig_checks(mech: Path, figs: Path) -> None:
    pairs = pd.read_parquet(mech / "code_pairs.parquet")
    m = pd.read_parquet(mech / "mechanisms.parquet")
    c = pd.read_parquet(mech / "codes.parquet")
    fig, axes = plt.subplots(1, 4, figsize=(21, 4))
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    big = pairs[with_classes(pairs, m)["nc"] >= 10]
    ax.hist(big["cka"], bins, alpha=0.6, label="code pairs, same variable")
    ax.hist(big["cka_null"], bins, alpha=0.6, label="classes of one code permuted")
    ax.axvline(0.7, color="k", ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("CKA of two codes' arrangements (variables with >= 10 classes)")
    ax.legend(fontsize=8)
    ax = axes[1]
    mm = m[m.n_members > 1]
    ax.scatter(
        mm["groups_member"] / mm["n_classes"], mm["groups"] / mm["n_classes"], s=8, alpha=0.5
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("classes told apart by the best member (fraction)")
    ax.set_ylabel("... by the whole mechanism")
    ax = axes[2]
    cc = cast(pd.DataFrame, c[c.n_members > 2])
    ax.scatter(
        cc["overlap_null"],
        cc["overlap"],
        s=8,
        alpha=0.5,
        c=np.where(cc["overlap_p"] <= 0.05, "C3", "C0"),
    )
    lim = float(max(cast(float, cc["overlap"].max()), cast(float, cc["overlap_null"].max())))
    ax.plot([1, lim], [1, lim], "k--", lw=0.8)
    ax.set_xlabel("support overlap of random writer sets (median)")
    ax.set_ylabel("support overlap of the code (1 = tiling)")
    ax.set_title("red: tiling better than random (p <= 0.05)", fontsize=8)
    ax = axes[3]
    ax.hist(mm["kappa"].clip(0, 3), bins=40)
    ax.axvline(1, color="k", ls="--")
    ax.set_xlabel("kappa = joint between-class energy / sum over members")
    fig.suptitle("Checks of the grouping")
    save(fig, figs, "mech_checks")


DRIFT_PTS = [0, 2, 8, 16, 24, 30, 32, 34, 36, 40, 48, 56, 64]


def fig_drift(mech: Path, figs: Path) -> None:
    """Frobenius cos between the mod-10 arrangements at two points: same shape in the same place."""
    from param_decomp.arith_repr.autointerp.arrangements import residue_means

    cm = np.load(mech / "class_means.npy", mmap_mode="r")
    ix = np.load(mech / "class_means_index.npz")
    specs = [(1, "a", 10), (3, "b", 10), (4, "res", 10), (4, "res", 100)]
    fig, axes = plt.subplots(1, len(specs), figsize=(4.6 * len(specs), 4.4))
    for ax, (p, q, tau) in zip(axes, specs, strict=True):
        sel = (ix["pos"] == p) & (ix["op"] == 0) & (ix["q"] == QI[q])
        X = [
            residue_means(
                np.asarray(cm[t][sel], np.float64), ix["value"][sel], ix["count"][sel], tau
            ).ravel()
            for t in DRIFT_PTS
        ]
        C = np.array(
            [[x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30) for y in X] for x in X]
        )
        ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
        for i in range(len(X)):
            for j in range(len(X)):
                ax.text(
                    j,
                    i,
                    f"{C[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="w" if C[i, j] < 0.6 else "k",
                )
        labels = [point_label(t) for t in DRIFT_PTS]
        ax.set_xticks(range(len(X)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(X)), labels, fontsize=7)
        ax.set_title(f"{q} mod {tau} @ `{POS[p]}` (add)", fontsize=9)
    fig.suptitle(
        "Is the arrangement still in the same place? cos of the class-mean matrices "
        "between two points (1 = same shape, same directions)"
    )
    save(fig, figs, "mech_drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    mech = args.run / AI / "mech"
    figs = args.run / AI / "figs"
    fns = {
        "code_power": fig_code_power,
        "geometry": fig_geometry,
        "snapshots": fig_snapshots,
        "map": fig_map,
        "checks": fig_checks,
        "drift": fig_drift,
    }
    for name, fn in fns.items():
        if args.only and name not in args.only:
            continue
        fn(mech, figs)


if __name__ == "__main__":
    main()
