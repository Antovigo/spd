"""Marimo notebook plotting the `collect_filtered_kl.py` outputs (see `plan.md`).

Reads the long-format KL TSVs from a run's `analysis/datasets/subspace_filtering/` and
renders, per (block, subset, op):

- grouped boxplots of the per-prompt KL distribution, one group per intervention
  (raw / centered / bias flavors) plus the circuit baseline;
- per-intervention KL(a, b) heatmaps: raw | centered | bias | baseline on a shared log
  colour scale;
- per-intervention attribution scatters: raw vs centered KL per prompt, coloured by the
  per-prompt selected-set size when available (active subset).

Every rendered figure is also saved as PNG under `<run>/analysis/subspace_filtering/`.

Run with `uv pip install marimo`, then `marimo edit plots_notebook.py` (or `marimo run`).
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
    from matplotlib.colors import LogNorm

    return LogNorm, Path, json, mo, np, os, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Subspace filtering — KL under projections

        Per-prompt last-position `KL(target ‖ variant)` for the projection battery of
        `collect_filtered_kl.py`. Pick the run, block, subset, and operation; figures
        are saved to `<run>/analysis/subspace_filtering/` as a side effect.
        """
    )
    return


@app.cell
def _(mo, os):
    # env-var defaults let `python plots_notebook.py` render one combo headlessly
    run_dir_ui = mo.ui.text(
        value=os.environ.get("SUBFILT_RUN", "~/out/runs/addsub-L18-04-2x-beta0.75-LR"),
        label="run dir",
        full_width=True,
    )
    block_ui = mo.ui.dropdown(
        options=["mlp", "attn"], value=os.environ.get("SUBFILT_BLOCK", "mlp"), label="block"
    )
    subset_ui = mo.ui.dropdown(
        options=["alive", "active"],
        value=os.environ.get("SUBFILT_SUBSET", "alive"),
        label="subset",
    )
    op_ui = mo.ui.dropdown(
        options=["add", "sub"], value=os.environ.get("SUBFILT_OP", "add"), label="op"
    )
    mo.hstack([run_dir_ui, block_ui, subset_ui, op_ui])
    return block_ui, op_ui, run_dir_ui, subset_ui


@app.cell
def _(Path, block_ui, json, op_ui, pd, run_dir_ui, subset_ui):
    run_dir = Path(run_dir_ui.value).expanduser()
    block, subset, op = block_ui.value, subset_ui.value, op_ui.value
    data_dir = run_dir / "analysis" / "datasets" / "subspace_filtering"
    kl_path = data_dir / f"kl_{block}_{subset}_{op}.tsv"
    assert kl_path.exists(), f"missing {kl_path}; run collect_filtered_kl first"
    df = pd.read_csv(kl_path, sep="\t")
    meta = json.loads((data_dir / f"meta_{block}_{subset}.json").read_text())
    nci_path = data_dir / f"nci_{block}_{op}.tsv"
    nci = pd.read_csv(nci_path, sep="\t") if nci_path.exists() else None
    fig_dir = run_dir / "analysis" / "subspace_filtering" / f"{block}_{subset}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    experiments = [e for e in df["experiment"].unique() if e != "circuit_baseline"]
    n_grid = int(df["a"].max())
    baseline_kl = df[df["experiment"] == "circuit_baseline"]["kl"].to_numpy()
    return baseline_kl, block, data_dir, df, experiments, fig_dir, meta, n_grid, nci, op, subset


@app.cell
def _(df, n_grid, np):
    def grid_of(experiment: str, flavor: str) -> np.ndarray:
        sub = df[(df["experiment"] == experiment) & (df["flavor"] == flavor)]
        assert len(sub) == n_grid * n_grid, f"{experiment}/{flavor}: {len(sub)} rows"
        grid = np.zeros((n_grid, n_grid), np.float32)
        grid[sub["a"].to_numpy() - 1, sub["b"].to_numpy() - 1] = sub["kl"].to_numpy()
        return grid

    return (grid_of,)


@app.cell
def _(baseline_kl, block, df, experiments, fig_dir, np, op, plt, subset):
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
            [np.maximum(baseline_kl, _floor)],
            positions=[len(experiments) * 4.0],
            widths=0.8,
            showfliers=False,
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("tab:green")
        ax.axhline(float(np.mean(baseline_kl)), color="tab:green", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xticks([*(np.arange(len(experiments)) * 4.0 + 1), len(experiments) * 4.0])
        ax.set_xticklabels([*experiments, "circuit_baseline"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("KL(target ‖ variant) at `=`")
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6)
            for c in [*colors.values(), "tab:green"]
        ]
        ax.legend(handles, [*_FLAVORS, "baseline"], fontsize=8)
        ax.set_title(f"{block}/{subset}/{op}: per-prompt KL by intervention and flavor")
        fig.tight_layout()
        fig.savefig(fig_dir / f"boxplot_{op}.png", dpi=150)
        return fig

    _boxplot()
    return


@app.cell
def _(LogNorm, baseline_kl, block, experiments, fig_dir, grid_of, mo, n_grid, np, op, plt, subset):
    _floor = 1e-8

    def _heatmap(experiment: str):
        grids = [grid_of(experiment, f) for f in ("raw", "centered", "bias")]
        grids.append(grid_of("circuit_baseline", "none"))
        titles = ["raw", "centered", "bias", "circuit baseline"]
        clipped = [np.maximum(g, _floor) for g in grids]
        vmax = max(g.max() for g in clipped)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        norm = LogNorm(vmin=_floor, vmax=vmax)
        im = None
        for ax, g, t, raw_g in zip(axes, clipped, titles, grids, strict=True):
            im = ax.imshow(
                g, origin="lower", extent=(0.5, n_grid + 0.5, 0.5, n_grid + 0.5),
                norm=norm, cmap="viridis",
            )  # fmt: skip
            ax.set_title(f"{t}\nmean KL {raw_g.mean():.4g}", fontsize=9)
            ax.set_xlabel("b")
        axes[0].set_ylabel("a")
        assert im is not None
        fig.colorbar(im, ax=list(axes), label="KL", fraction=0.03)
        fig.suptitle(f"{block}/{subset}/{op}: {experiment}")
        safe = experiment.replace(":", "-")
        fig.savefig(fig_dir / f"heatmap_{safe}_{op}.png", dpi=150, bbox_inches="tight")
        return fig

    _figs = [_heatmap(e) for e in experiments]
    mo.vstack(_figs)
    return


@app.cell
def _(block, df, experiments, fig_dir, nci, np, op, plt, subset):
    _floor = 1e-8

    def _scatter():
        ncols = 3
        nrows = -(-len(experiments) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
        color = None
        if nci is not None:
            color = nci.sort_values(["a", "b"])["n_ci_total"].to_numpy()
        for ei, e in enumerate(experiments):
            ax = axes[ei // ncols][ei % ncols]
            sub_r = df[(df["experiment"] == e) & (df["flavor"] == "raw")].sort_values(["a", "b"])
            sub_c = df[(df["experiment"] == e) & (df["flavor"] == "centered")].sort_values(
                ["a", "b"]
            )
            x = np.maximum(sub_r["kl"].to_numpy(), _floor)
            y = np.maximum(sub_c["kl"].to_numpy(), _floor)
            sc = ax.scatter(x, y, s=2, c=color, cmap="plasma", alpha=0.5)
            lo, hi = _floor, max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("raw KL")
            ax.set_ylabel("centered KL")
            ax.set_title(e, fontsize=9)
            if color is not None and ei == 0:
                fig.colorbar(sc, ax=ax, label="n_ci_total", fraction=0.05)
        for j in range(len(experiments), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"{block}/{subset}/{op}: raw vs centered KL per prompt")
        fig.tight_layout()
        fig.savefig(fig_dir / f"scatter_raw_vs_centered_{op}.png", dpi=150)
        return fig

    _scatter()
    return


@app.cell
def _(json, meta, mo):
    mo.md(f"### meta\n```json\n{json.dumps(meta, indent=2)[:4000]}\n```")
    return


if __name__ == "__main__":
    app.run()
