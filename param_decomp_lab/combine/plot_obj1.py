"""Plot the objective-1 eval JSON: per-batch recon losses, singles vs combined."""

import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np

SERIES_COLORS = {
    "rounded": "#2a78d6",
    "PGD": "#008300",
    "ci_masked": "#e87ba4",
    "stochastic": "#eda100",
    "delta_only": "#1baf7a",
}
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e56"
REFERENCE_GRAY = "#8a897f"

TARGET_SERIES = {
    "rounded": "target_recon/rounded",
    "PGD": "loss/PGDReconLoss",
    "ci_masked": "target_recon/ci_masked",
    "stochastic": "target_recon/stochastic",
}
NONTARGET_SERIES = {
    "rounded": "nontarget_recon/rounded",
    "ci_masked": "nontarget_recon/ci_masked",
    "stochastic": "nontarget_recon/stochastic",
    "delta_only": "nontarget_recon/delta_only",
}


def _short_name(subject: str) -> str:
    if subject == "combined":
        return "combined (4 blocks)"
    return subject.removeprefix("addsub-")


def _training_log_final(run_dir: Path, key: str) -> float | None:
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None
    value = None
    with open(metrics_file) as f:
        for line in f:
            record = json.loads(line)
            if key in record:
                value = record[key]
    return value


def _panel(
    ax: plt.Axes,
    subjects: list[str],
    results: dict[str, dict[str, Any]],
    series: dict[str, str],
    title: str,
    log_refs: dict[str, float] | None,
) -> None:
    n_series = len(series)
    offsets = np.linspace(-0.28, 0.28, n_series)
    for row, subject in enumerate(subjects):
        per_batch = results[subject]["per_batch"]
        means = results[subject]["mean"]
        for offset, (label, key) in zip(offsets, series.items(), strict=True):
            y = row + offset
            values = per_batch[key]
            color = SERIES_COLORS[label]
            ax.scatter(
                values,
                np.full(len(values), y),
                s=14,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.75,
                zorder=3,
            )
            ax.plot(
                [means[key], means[key]],
                [y - 0.11, y + 0.11],
                color=color,
                linewidth=2.2,
                zorder=4,
                solid_capstyle="butt",
            )
        if log_refs is not None and subject in log_refs:
            ax.scatter(
                [log_refs[subject]],
                [row],
                marker="D",
                s=26,
                facecolors="none",
                edgecolors=REFERENCE_GRAY,
                linewidths=1.2,
                zorder=5,
            )

    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([_short_name(s) for s in subjects], color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_xscale("log")
    ax.grid(axis="x", color="#e6e5df", linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8bf")


def _scaling_figure(
    chain_points: list[tuple[int, str, dict[str, Any]]],
    extra_points: list[tuple[int, str, dict[str, Any]]],
    out_path: Path,
) -> None:
    """Recon vs number of blocks replaced: chain connected, extras annotated."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor="white")
    series = {"rounded": "target_recon/rounded", "PGD": "loss/PGDReconLoss"}
    for label, key in series.items():
        color = SERIES_COLORS[label]
        xs = [n for n, _, _ in chain_points]
        means = [r["mean"][key] for _, _, r in chain_points]
        for n, _, r in chain_points:
            values = r["per_batch"][key]
            ax.scatter(
                np.full(len(values), n),
                values,
                s=14,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.75,
                zorder=3,
            )
        ax.plot(xs, means, color=color, linewidth=2.0, zorder=4, marker="_", markersize=10)
        ax.annotate(
            label,
            (xs[-1], means[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            color=color,
            fontsize=9,
            va="center",
        )
        for n, _name, r in extra_points:
            values = r["per_batch"][key]
            ax.scatter(
                np.full(len(values), n + 0.13),
                values,
                s=14,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.45,
                zorder=3,
            )
    for n, name, r in extra_points:
        ax.annotate(
            name,
            (n + 0.13, max(r["mean"][key] for key in series.values())),
            textcoords="offset points",
            xytext=(6, 6),
            color=TEXT_SECONDARY,
            fontsize=8,
        )

    chain_labels = [name for _, name, _ in chain_points]
    ax.set_xticks([n for n, _, _ in chain_points])
    ax.set_xticklabels(
        [f"{n}\n{lbl}" for (n, _, _), lbl in zip(chain_points, chain_labels, strict=True)],
        fontsize=8,
        color=TEXT_PRIMARY,
    )
    ax.set_yscale("log")
    ax.set_xlabel("number of blocks replaced", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("recon loss (KL per position)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(
        "Target recon vs number of combined blocks (chain L16→L19)",
        color=TEXT_PRIMARY,
        fontsize=11,
        loc="left",
        pad=10,
    )
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


def main(
    results_json: str,
    out_dir: str,
    runs_dir: str | None = None,
    prefix2_json: str | None = None,
    prefix3_json: str | None = None,
    pair_json: str | None = None,
) -> None:
    """Render the objective-1 figures from `eval_combined` outputs.

    Args:
        results_json: Path to the JSON written by `combine.eval_combined`.
        out_dir: Directory for the PNGs.
        runs_dir: Runs root for end-of-training reference markers (gray diamonds);
            omit to skip them.
        prefix2_json / prefix3_json: 2- and 3-block prefix-chain evals for the
            scaling figure; omit to skip it.
        pair_json: Extra non-chain pair eval, annotated on the scaling figure.
    """
    payload = json.loads(Path(results_json).read_text())
    results = payload["results"]
    singles = [s for s in results if s != "combined"]
    subjects = singles + ["combined"]

    log_refs: dict[str, float] = {}
    if runs_dir is not None:
        for subject in singles:
            final = _training_log_final(Path(runs_dir) / subject, "eval/target_recon/rounded")
            if final is not None:
                log_refs[subject] = final

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="white")
    _panel(
        axes[0],
        subjects,
        results,
        TARGET_SERIES,
        "Target distribution (addsub prompts, delta off)",
        log_refs,
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
    if log_refs:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=REFERENCE_GRAY,
                markersize=6,
                label="end-of-training log (rounded)",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Recon loss per eval batch: single-block decompositions vs all four combined "
        f"(rounding thr {payload['meta']['ci_thr']})",
        y=1.10,
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    fig.tight_layout()

    out_path = Path(out_dir) / "obj1_recon.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")

    if prefix2_json is not None and prefix3_json is not None:
        first_single = singles[0]
        chain_points = [
            (1, _short_name(first_single), results[first_single]),
            (2, "+L17", json.loads(Path(prefix2_json).read_text())["results"]["combined"]),
            (3, "+L18", json.loads(Path(prefix3_json).read_text())["results"]["combined"]),
            (4, "+L19", results["combined"]),
        ]
        extra_points = []
        if pair_json is not None:
            extra_points.append(
                (2, "L18+L19", json.loads(Path(pair_json).read_text())["results"]["combined"])
            )
        _scaling_figure(chain_points, extra_points, Path(out_dir) / "obj1_scaling.png")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
