"""Active-subcomponent heatmaps for a toy-model-redundancy decomposition, vs redundancy ground truth.

For every decomposed matrix, plots the `[subcomponent, token]` lower-leaky CI heatmap
(subcomponents with max CI > `--ci-thr`, sorted by their argmax token), from one
forward per vocab token (seq len 1 — positions are independent). A second figure is
the redundancy diagnostic: per block, the toy's ground-truth per-token accuracy map
(`per_token_accuracy.npz`: does block b alone implement token v?) against the
decomposition's coverage (max CI over the block's subcomponents at v), with cells the
decomposition SKIPPED (ground truth says the mechanism exists, no subcomponent claims
it) marked. CPU-only.

Usage:
    python -m param_decomp_lab.experiments.toy_model_redundancy.plot_ci <model_path> [--ci-thr=0.1]

Outputs (in the PD run's `analysis/`): `active_subcomponents.png`, `coverage_vs_truth.png`.
"""

from pathlib import Path
from typing import cast

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp_lab.experiments.toy_model_redundancy.ci_figure import (  # noqa: E402
    plot_subcomponent_grid,
)
from param_decomp_lab.experiments.toy_model_redundancy.run import (  # noqa: E402
    AnyRedundancyToy,
    SavedToyModelRedundancyRun,
    resolve_toy_run_dir,
)
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402


def _token_cis(run: SavedToyModelRedundancyRun) -> dict[str, np.ndarray]:
    """module -> `[vocab, seq, C]` lower-leaky CI, one probe sequence per vocab entry."""
    model = run.load_model()
    tokens = cast(AnyRedundancyToy, model.target_model).enumerate_inputs()
    with torch.no_grad():
        cached = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(cached.cache, sampling="continuous")
    return {m: ci.lower_leaky[m].float().numpy(force=True) for m in sorted(ci.lower_leaky)}


def plot_ci(model_path: ModelPath, ci_thr: float = 0.1) -> Path:
    """Write the active-subcomponent and coverage-vs-truth figures; returns the out dir."""
    run = SavedToyModelRedundancyRun.from_path(model_path)
    cis = _token_cis(run)
    toy_dir = resolve_toy_run_dir(run.cfg.target.run_path)
    truth = np.load(toy_dir / "per_token_accuracy.npz")["accuracy"]  # [n_blocks, vocab]
    n_blocks, vocab = truth.shape

    out_dir = run.checkpoint_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: all-subcomponent CI heatmaps on a [matrix, block] grid, permuted to
    # identity. Only decomposed blocks appear; a partial (per-block) decomposition is
    # scored against its own blocks' ground truth.
    fig = plot_subcomponent_grid(cis)
    fig.savefig(out_dir / "active_subcomponents.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    blocks_present = sorted({int(module.split(".")[1]) for module in cis})

    # Figure 2: per-block coverage vs ground truth, with skipped mechanisms marked.
    truth = truth[blocks_present]
    n_blocks = len(blocks_present)
    # Routing matrices (q, k) are token-unselective — a single routing subcomponent is
    # active on every token — so only token-selective matrices count as coverage.
    coverage = np.zeros((n_blocks, vocab))
    for module, ci in cis.items():
        if module.rsplit(".", 1)[-1] in ("q", "k"):
            continue
        row = blocks_present.index(int(module.split(".")[1]))
        coverage[row] = np.maximum(coverage[row], ci.max(axis=(1, 2)))
    skipped = (truth > 0.9) & (coverage < ci_thr)

    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), facecolor="white", sharex=True)
    panels = [
        (truth, "Greys", "ground truth: block-alone accuracy per token"),
        (coverage, "RdPu", "decomposition: max CI over the block's subcomponents"),
        (skipped.astype(float), "Reds", f"SKIPPED (truth > 0.9, max CI < {ci_thr})"),
    ]
    for ax, (data, cmap, title) in zip(axes, panels, strict=True):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="none")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks(range(n_blocks), [f"block {b}" for b in blocks_present], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02)
    axes[-1].set_xlabel("input token", fontsize=8)
    fig.suptitle(f"{run.checkpoint_path.parent.name}: mechanism coverage vs redundancy", y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_vs_truth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_skipped = int(skipped.sum())
    logger.info(
        f"{n_skipped} skipped (block, token) mechanisms of {int((truth > 0.9).sum())} "
        f"ground-truth ones → {out_dir}"
    )
    return out_dir


def cli() -> None:
    fire.Fire(plot_ci)


if __name__ == "__main__":
    cli()
