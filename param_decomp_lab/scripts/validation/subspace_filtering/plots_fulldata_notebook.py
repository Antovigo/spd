"""Marimo notebook plotting the `collect_filtered_kl_fulldata.py` outputs.

Reads `kl_fulldata_<block>.tsv` / `meta_fulldata_<block>.json`
from a run's `analysis/datasets/subspace_filtering/` and renders, per block:

- grouped boxplots of the per-(prompt, position) KL distribution, one group per
  intervention (raw / centered / bias) plus the circuit baseline;
- mean KL vs position, one line per intervention (chosen flavor) plus the baseline;
- per-intervention raw-vs-centered scatters, coloured by position.

Every figure is saved as PNG under `<run>/analysis/subspace_filtering/fulldata_<block>/`.

Run with `uv pip install marimo`, then `marimo edit plots_fulldata_notebook.py`; or
headless for one combo: `SUBFILT_BLOCK=mlp python plots_fulldata_notebook.py`.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    return Path, json, mo, np, os, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Subspace filtering — full-data run, every layer / every position

        Per-(prompt, position) `KL(target ‖ variant)` from `collect_filtered_kl_fulldata.py`.
        """
    )
    return


@app.cell
def _(mo, os):
    run_dir_ui = mo.ui.text(
        value=os.environ.get("SUBFILT_RUN", "~/out/runs/s-55ea3f9b"),
        label="run dir",
        full_width=True,
    )
    block_ui = mo.ui.dropdown(
        options=["mlp", "attn"], value=os.environ.get("SUBFILT_BLOCK", "mlp"), label="block"
    )
    flavor_ui = mo.ui.dropdown(
        options=["raw", "centered", "bias"],
        value=os.environ.get("SUBFILT_FLAVOR", "centered"),
        label="position-curve flavor",
    )
    mo.hstack([run_dir_ui, block_ui, flavor_ui])
    return block_ui, flavor_ui, run_dir_ui


@app.cell
def _(Path, block_ui, json, pd, run_dir_ui):
    run_dir = Path(run_dir_ui.value).expanduser()
    block = block_ui.value
    data_dir = run_dir / "analysis" / "datasets" / "subspace_filtering"
    kl_path = data_dir / f"kl_fulldata_{block}.tsv"
    assert kl_path.exists(), f"missing {kl_path}; run collect_filtered_kl_fulldata first"
    df = pd.read_csv(kl_path, sep="\t")
    meta = json.loads((data_dir / f"meta_fulldata_{block}.json").read_text())
    fig_dir = run_dir / "analysis" / "subspace_filtering" / f"fulldata_{block}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    experiments = [e for e in df["experiment"].unique() if e != "circuit_baseline"]
    baseline = df[df["experiment"] == "circuit_baseline"]
    return baseline, block, df, experiments, fig_dir, meta


@app.cell
def _(baseline, block, df, experiments, fig_dir, np, plt):
    _FLAVORS = ("raw", "centered", "bias")
    _floor = 1e-8

    def _boxplot():
        fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(experiments)), 5))
        colors = {"raw": "tab:red", "centered": "tab:blue", "bias": "tab:orange"}
        for fi, flavor in enumerate(_FLAVORS):
            data = [
                np.maximum(
                    df[(df["experiment"] == e) & (df["flavor"] == flavor)]["kl"].to_numpy(),
                    _floor,
                )
                for e in experiments
            ]
            bp = ax.boxplot(
                data,
                positions=np.arange(len(experiments)) * 4.0 + fi,
                widths=0.8,
                showfliers=False,
                patch_artist=True,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[flavor])
                patch.set_alpha(0.6)
        bp = ax.boxplot(
            [np.maximum(baseline["kl"].to_numpy(), _floor)],
            positions=[len(experiments) * 4.0],
            widths=0.8,
            showfliers=False,
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("tab:green")
        ax.axhline(float(baseline["kl"].mean()), color="tab:green", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks([*(np.arange(len(experiments)) * 4.0 + 1), len(experiments) * 4.0])
        ax.set_xticklabels([*experiments, "circuit_baseline"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("KL(target ‖ variant), all positions")
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6)
            for c in [*colors.values(), "tab:green"]
        ]
        ax.legend(handles, [*_FLAVORS, "baseline"], fontsize=8)
        ax.set_title(f"fulldata/{block}: per-(prompt, position) KL by intervention and flavor")
        fig.tight_layout()
        fig.savefig(fig_dir / "boxplot.png", dpi=150)
        return fig

    _boxplot()
    return


@app.cell
def _(baseline, block, df, experiments, fig_dir, flavor_ui, plt):
    def _kl_vs_position():
        flavor = flavor_ui.value
        fig, ax = plt.subplots(figsize=(10, 5))
        for e in experiments:
            sub = df[(df["experiment"] == e) & (df["flavor"] == flavor)]
            curve = sub.groupby("pos")["kl"].mean()
            ax.plot(curve.index, curve.to_numpy(), lw=1, label=e)
        base_curve = baseline.groupby("pos")["kl"].mean()
        ax.plot(base_curve.index, base_curve.to_numpy(), "k--", lw=1.5, label="circuit_baseline")
        ax.set_yscale("log")
        ax.set_xlabel("position")
        ax.set_ylabel(f"mean KL ({flavor})")
        ax.legend(fontsize=7)
        ax.set_title(f"fulldata/{block}: mean KL vs position ({flavor})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"kl_vs_position_{flavor}.png", dpi=150)
        return fig

    _kl_vs_position()
    return


@app.cell
def _(block, df, experiments, fig_dir, np, plt):
    _floor = 1e-8

    def _scatter():
        ncols = 3
        nrows = -(-len(experiments) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
        sc = None
        for ei, e in enumerate(experiments):
            ax = axes[ei // ncols][ei % ncols]
            sub_r = df[(df["experiment"] == e) & (df["flavor"] == "raw")].sort_values(
                ["prompt", "pos"]
            )
            sub_c = df[(df["experiment"] == e) & (df["flavor"] == "centered")].sort_values(
                ["prompt", "pos"]
            )
            x = np.maximum(sub_r["kl"].to_numpy(), _floor)
            y = np.maximum(sub_c["kl"].to_numpy(), _floor)
            sc = ax.scatter(x, y, s=1.5, c=sub_r["pos"].to_numpy(), cmap="plasma", alpha=0.4)
            lo, hi = _floor, max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("raw KL")
            ax.set_ylabel("centered KL")
            ax.set_title(e, fontsize=9)
        assert sc is not None
        fig.colorbar(sc, ax=axes[0][min(len(experiments), ncols) - 1], label="position")
        for j in range(len(experiments), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"fulldata/{block}: raw vs centered KL per (prompt, position)")
        fig.tight_layout()
        fig.savefig(fig_dir / "scatter_raw_vs_centered.png", dpi=150)
        return fig

    _scatter()
    return


@app.cell
def _(json, meta, mo):
    mo.md(f"### meta\n```json\n{json.dumps(meta, indent=2)[:5000]}\n```")
    return


if __name__ == "__main__":
    app.run()
