"""Summarise ridge-CV Fourier probes: which variables live at which period, where.

Reads one `ridge_cv_probes_<op>.json` per op (from `fit_ridge_cv_probes`) and renders a
small-multiples heatmap grid — rows = ops, columns = variables; within each panel,
y = residual-stream position (after block L, shallow→deep), x = period, cell = `cv_r2`
(fold-mean held-out R² at the CV-selected λ). Cells whose permutation-null p-value
exceeds `--alpha` are greyed out: no evidence of generalizable circular structure there.
Also writes a flat TSV of every (op, layer, variable, period) cell.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.plot_ridge_cv_probes \
        <ridge_cv_probes_add.json> [<ridge_cv_probes_sub.json> ...] [--alpha=0.05] \
        [--output-fig=PATH] [--output-tsv=PATH] [--slurm [--gpus=0 ...]]

Output (default beside the first json): `ridge_cv_heatmap.png` + `ridge_cv_summary.tsv`.
"""

import csv
import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from param_decomp.log import logger
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm

_MODULE = "param_decomp_lab.scripts.validation.probes.plot_ridge_cv_probes"
_GATED_GREY = "#e8e8ea"
_SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#0b3d91"])


def plot_ridge_cv_probes(
    *ridge_cv_jsons: str,
    alpha: float = 0.05,
    output_fig: str | None = None,
    output_tsv: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 0,
    slurm_time: str = "0:20:00",
    slurm_mem: str | None = None,
    dependency: str | None = None,
) -> tuple[Path, Path] | None:
    assert ridge_cv_jsons, "pass at least one ridge_cv_probes_<op>.json"
    json_paths = [Path(p).expanduser() for p in ridge_cv_jsons]
    if slurm:
        argv = [str(p) for p in json_paths] + [f"--alpha={alpha}"]
        if output_fig is not None:
            argv.append(f"--output-fig={Path(output_fig).expanduser()}")
        if output_tsv is not None:
            argv.append(f"--output-tsv={Path(output_tsv).expanduser()}")
        opts = SlurmOptions(
            partition=partition,
            gpus=gpus,
            slurm_time=slurm_time,
            slurm_mem=slurm_mem,
            dependency=dependency,
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-plot-ridge-cv")
        return None

    payloads = [json.loads(p.read_text()) for p in json_paths]
    variables: list[str] = payloads[0]["meta"]["variables"]
    periods: list[int] = payloads[0]["meta"]["periods"]
    for payload in payloads[1:]:
        assert payload["meta"]["variables"] == variables
        assert payload["meta"]["periods"] == periods

    fig_path = (
        Path(output_fig).expanduser()
        if output_fig
        else json_paths[0].parent / "ridge_cv_heatmap.png"
    )
    tsv_path = (
        Path(output_tsv).expanduser()
        if output_tsv
        else json_paths[0].parent / "ridge_cv_summary.tsv"
    )

    rows: list[dict[str, Any]] = []
    n_ops = len(payloads)
    fig, axes = plt.subplots(
        n_ops,
        len(variables),
        figsize=(3.1 * len(variables) + 1.2, 2.6 * n_ops + 0.9),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for row_i, payload in enumerate(payloads):
        op = payload["meta"]["op"]
        positions: list[str] = payload["meta"]["positions"]
        for col_i, variable in enumerate(variables):
            ax = axes[row_i][col_i]
            score = np.zeros((len(positions), len(periods)))
            accepted = np.zeros_like(score, dtype=bool)
            for yi, pos in enumerate(positions):
                for xi, period in enumerate(periods):
                    entry = payload["results"][pos][variable][str(period)]
                    score[yi, xi] = entry["cv_r2"]
                    # beating the null is not enough: a probe with negative held-out R²
                    # (all scores deeply negative, real merely least-bad) is still no probe
                    accepted[yi, xi] = entry["p_value"] <= alpha and entry["cv_r2"] > 0
                    rows.append(
                        {
                            "op": op,
                            "position": pos,
                            "variable": variable,
                            "period": period,
                            "cv_r2": entry["cv_r2"],
                            "p_value": entry["p_value"],
                            "accepted": int(accepted[yi, xi]),
                            "lambda_rel": entry["lambda_rel"],
                            "cv_angerr_deg": entry["cv_angerr_deg"],
                            "full_r2": entry["full_r2"],
                            "min_block_r2": min(entry["block_r2"]),
                            "max_null_cv_r2": max(entry["null_cv_r2"]),
                        }
                    )
            shown = np.where(accepted, np.clip(score, 0.0, 1.0), np.nan)
            last_image = ax.imshow(
                shown, cmap=_SEQ_CMAP, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
            )
            ax.set_facecolor(_GATED_GREY)
            for yi in range(len(positions)):
                for xi in range(len(periods)):
                    if accepted[yi, xi]:
                        dark_cell = np.clip(score[yi, xi], 0.0, 1.0) > 0.55
                        ax.text(
                            xi,
                            yi,
                            f"{score[yi, xi]:.2f}".replace("0.", "."),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="#ffffff" if dark_cell else "#1c1c1e",
                        )
                    else:
                        ax.text(xi, yi, "·", ha="center", va="center", fontsize=7, color="#8e8e93")
            ax.set_xticks(range(len(periods)), [str(p) for p in periods], fontsize=8)
            ax.set_yticks(range(len(positions)), positions, fontsize=8)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_i == 0:
                ax.set_title(variable, fontsize=11)
            if row_i == n_ops - 1:
                ax.set_xlabel("period T", fontsize=9, color="#555")
            if col_i == 0:
                ax.set_ylabel(f"{op}: resid after block", fontsize=9, color="#555")
    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label("CV R² (held-out, rotating value blocks)", fontsize=9)
    n_perm = payloads[0]["meta"]["n_perm"]
    fig.suptitle(
        f"Ridge Fourier probes — grey · = fails permutation null (p > {alpha}, {n_perm} perms)",
        fontsize=12,
    )
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {fig_path} + {tsv_path} ({len(rows)} cells)")
    return fig_path, tsv_path


if __name__ == "__main__":
    fire.Fire(plot_ridge_cv_probes)
