"""Compare active components across two SPD decompositions.

For each shared module_path, computes pairwise cosine similarities between active
components from both runs and plots heatmaps.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_components \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions_components
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from spd.scripts.compare_decompositions.utils import (
    compute_ci,
    compute_pairwise_cosine_sim,
    get_active_component_indices,
    get_component_weight,
    get_tokenizer,
    load_decomposition,
)

TILE_SIZE = 0.5  # inches per tile
LABEL_PAD = 1.5  # inches for axis labels/titles
COLORBAR_WIDTH_RATIO = 2  # gridspec width ratio units for colorbar column


def plot_component_heatmaps(
    sim_matrices: dict[str, torch.Tensor],
    active_indices_a: dict[str, list[int]],
    active_indices_b: dict[str, list[int]],
    label_a: str,
    label_b: str,
) -> Figure:
    """Plot one heatmap subplot per module_path showing pairwise cosine similarity."""
    assert len(sim_matrices) > 0

    sorted_paths = sorted(sim_matrices)
    n_b_per_module = [sim_matrices[p].shape[1] for p in sorted_paths]
    max_n_a = max(sim_matrices[p].shape[0] for p in sorted_paths)

    width_ratios = n_b_per_module + [COLORBAR_WIDTH_RATIO]
    fig_width = (
        sum(n_b_per_module) * TILE_SIZE
        + LABEL_PAD * len(sorted_paths)
        + COLORBAR_WIDTH_RATIO * TILE_SIZE
    )
    fig_height = max_n_a * TILE_SIZE + LABEL_PAD

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=len(sorted_paths) + 1,
        width_ratios=width_ratios,
        wspace=0.4,
    )

    im = None
    for idx, module_path in enumerate(sorted_paths):
        ax = fig.add_subplot(gs[0, idx])
        sim_np = sim_matrices[module_path].cpu().numpy()
        im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

        indices_a = active_indices_a[module_path]
        indices_b = active_indices_b[module_path]
        ax.set_xticks(range(len(indices_b)))
        ax.set_xticklabels([str(i) for i in indices_b], fontsize=7)
        ax.set_yticks(range(len(indices_a)))
        ax.set_yticklabels([str(i) for i in indices_a], fontsize=7)
        ax.set_xlabel(f"{label_b} components", fontsize=8)
        ax.set_ylabel(f"{label_a} components", fontsize=8)
        ax.set_title(module_path, fontsize=9)

    fig.suptitle("Component Cosine Similarity", fontsize=14)
    assert im is not None
    cbar_ax = fig.add_subplot(gs[0, -1])
    fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")
    return fig


@torch.no_grad()
def main(
    wandb_path_a: str,
    wandb_path_b: str,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    prompt: str,
    ci_threshold: float = 0.1,
    device: str = "cuda",
    output_dir: str = "/tmp/compare_decompositions_components",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decomp_a = load_decomposition(wandb_path_a, label_a, device)
    decomp_b = load_decomposition(wandb_path_b, label_b, device)
    print(f"Run A: {decomp_a.label}")
    print(f"Run B: {decomp_b.label}")

    tokenizer = get_tokenizer(decomp_a)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Prompt: {prompt!r}  (tokens: {tokens.shape[1]})")

    ci_a = compute_ci(decomp_a.model, tokens, decomp_a.run_info.config.sampling)
    ci_b = compute_ci(decomp_b.model, tokens, decomp_b.run_info.config.sampling)

    active_a = get_active_component_indices(ci_a, ci_threshold)
    active_b = get_active_component_indices(ci_b, ci_threshold)

    print(f"\nActive components (CI > {ci_threshold}):")
    for path in sorted(set(active_a) | set(active_b)):
        n_a = len(active_a.get(path, []))
        n_b = len(active_b.get(path, []))
        print(f"  {path}: {n_a} ({decomp_a.label}), {n_b} ({decomp_b.label})")

    shared_modules = [p for p in active_a if p in active_b and active_a[p] and active_b[p]]

    sim_matrices: dict[str, torch.Tensor] = {}
    for module_path in sorted(shared_modules):
        indices_a = active_a[module_path]
        indices_b = active_b[module_path]

        weights_a = torch.stack(
            [get_component_weight(decomp_a.model, module_path, i).flatten() for i in indices_a]
        )
        weights_b = torch.stack(
            [get_component_weight(decomp_b.model, module_path, i).flatten() for i in indices_b]
        )

        sim_matrices[module_path] = compute_pairwise_cosine_sim(weights_a, weights_b)

    if not sim_matrices:
        print("No shared modules with active components in both runs.")
        return

    print(f"\n{len(sim_matrices)} shared modules with active components")

    fig = plot_component_heatmaps(sim_matrices, active_a, active_b, decomp_a.label, decomp_b.label)
    path = out / "component_similarity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")

    # Print summary table
    print(f"\n{'Module':<50} {'Best A→B':>10} {'Best B→A':>10}")
    print("-" * 72)
    for module_path in sorted(sim_matrices):
        sim = sim_matrices[module_path].abs()
        best_a_to_b = sim.max(dim=1).values.mean().item()
        best_b_to_a = sim.max(dim=0).values.mean().item()
        print(f"{module_path:<50} {best_a_to_b:>10.4f} {best_b_to_a:>10.4f}")


if __name__ == "__main__":
    fire.Fire(main)
