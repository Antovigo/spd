"""Plot objective-2 results: fine-tuned subjects vs obj-1 baselines + trajectories."""

import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt

from param_decomp_lab.combine.plot_obj1 import (
    NONTARGET_SERIES,
    SERIES_COLORS,
    TARGET_SERIES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _panel,
)

TRAJECTORY_KEYS = {
    "rounded": "eval/target_recon/rounded",
    "PGD": "eval/loss/PGDReconLoss",
}


def _eval_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if TRAJECTORY_KEYS["rounded"] in d:
                records.append(d)
    return records


def _trajectory_figure(
    run_dirs: dict[str, Path], singles_band: tuple[float, float], out_path: Path
) -> None:
    fig, (ax_recon, ax_l0) = plt.subplots(
        1, 2, figsize=(10, 3.8), facecolor="white", width_ratios=[1.4, 1]
    )
    linestyles = {"combine-L16-19-both-02": "-", "combine-L16-19-frozenci-04": "--"}
    short = {"combine-L16-19-both-02": "both", "combine-L16-19-frozenci-04": "frozen CI"}
    for run_name, run_dir in run_dirs.items():
        records = _eval_records(run_dir)
        steps = [d["step"] for d in records]
        for label, key in TRAJECTORY_KEYS.items():
            values = [d[key] for d in records]
            ax_recon.plot(
                steps,
                values,
                color=SERIES_COLORS[label],
                linestyle=linestyles[run_name],
                linewidth=1.8,
                marker="o",
                markersize=4,
            )
            ax_recon.annotate(
                f"{label} ({short[run_name]})",
                (steps[-1], values[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                color=SERIES_COLORS[label],
                fontsize=8,
                va="center",
            )
        l0 = [d["eval/target_recon/total_l0"] for d in records]
        ax_l0.plot(
            steps,
            l0,
            color=TEXT_PRIMARY,
            linestyle=linestyles[run_name],
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
        ax_l0.annotate(
            short[run_name],
            (steps[-1], l0[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=TEXT_PRIMARY,
            fontsize=8,
            va="center",
        )

    ax_recon.set_yscale("log")
    ax_recon.axhspan(*singles_band, color="#e6e5df", alpha=0.6, zorder=0)
    ax_recon.annotate(
        "single-block range",
        (0, singles_band[1]),
        textcoords="offset points",
        xytext=(4, 3),
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    ax_recon.set_ylabel("recon loss (KL per position)", color=TEXT_SECONDARY, fontsize=9)
    ax_recon.set_title(
        "Target recon during fine-tuning", color=TEXT_PRIMARY, fontsize=11, loc="left"
    )
    ax_l0.set_ylabel("total L0 (CI > 0.1)", color=TEXT_SECONDARY, fontsize=9)
    ax_l0.set_ylim(bottom=0)
    ax_l0.set_title("Sparsity during fine-tuning", color=TEXT_PRIMARY, fontsize=11, loc="left")
    for ax in (ax_recon, ax_l0):
        ax.set_xlabel("fine-tuning step", color=TEXT_SECONDARY, fontsize=9)
        ax.grid(axis="y", color="#e6e5df", linewidth=0.8, zorder=0)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.margins(x=0.18)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def main(obj1_json: str, obj2_json: str, out_dir: str, runs_dir: str) -> None:
    """Render obj-2 figures: merged subject dot plot + fine-tuning trajectories.

    Args:
        obj1_json: eval_combined output with singles + raw combined.
        obj2_json: eval_combined output with the fine-tuned subjects.
        out_dir: Directory for the PNGs.
        runs_dir: Runs root (for the fine-tuned runs' metrics.jsonl trajectories).
    """
    obj1 = json.loads(Path(obj1_json).read_text())
    obj2 = json.loads(Path(obj2_json).read_text())
    results = dict(obj1["results"])
    rename = {
        "combine-L16-19-frozenci-04": "combined + FT (frozen CI)",
        "combine-L16-19-both-02": "combined + FT (both)",
    }
    for name, r in obj2["results"].items():
        results[rename.get(name, name)] = r
    singles = [s for s in obj1["results"] if s != "combined"]
    subjects = singles + ["combined"] + [rename[n] for n in obj2["results"] if n in rename]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor="white")
    _panel(
        axes[0], subjects, results, TARGET_SERIES, "Target distribution (addsub, delta off)", None
    )
    _panel(
        axes[1],
        subjects,
        results,
        NONTARGET_SERIES,
        "Nontarget distribution (FineWeb, delta on)",
        None,
    )
    for ax in axes:
        ax.set_xlabel("reconstruction loss (KL per position)", color=TEXT_SECONDARY, fontsize=9)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=6,
            label=label,
        )
        for label, color in SERIES_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Recon per eval batch: singles, raw combination, and 2000-step fine-tunes",
        y=1.09,
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    fig.tight_layout()
    out_path = Path(out_dir) / "obj2_recon.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    singles_rounded = [obj1["results"][s]["mean"]["target_recon/rounded"] for s in singles]
    _trajectory_figure(
        {
            "combine-L16-19-both-02": Path(runs_dir) / "combine-L16-19-both-02",
            "combine-L16-19-frozenci-04": Path(runs_dir) / "combine-L16-19-frozenci-04",
        },
        (min(singles_rounded), max(singles_rounded)),
        Path(out_dir) / "obj2_trajectory.png",
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
