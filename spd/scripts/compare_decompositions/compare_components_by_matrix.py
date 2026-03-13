"""Compare active components across two SPD decompositions, one image per module.

For each shared module_path, produces one PNG with positions arranged horizontally.
Tiles are uniform squares sized by the component counts.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_components_by_matrix \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions_by_matrix
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from spd.scripts.compare_decompositions.utils import (
    compute_ci,
    compute_pairwise_cosine_sim,
    get_active_component_indices_per_position,
    get_component_uv,
    get_component_weight,
    get_tokenizer,
    load_decomposition,
)

TILE_SIZE = 0.5 / 3  # inches per tile
LABEL_PAD = 1.5  # inches padding for axis labels/titles
COLORBAR_WIDTH_RATIO = 0.2 / 3  # gridspec width ratio units for colorbar column


ROW_LABEL_PAD = 0.8  # inches padding for row labels on the left

ROW_LABELS = ["Full", "V", "U"]


def plot_matrix_heatmaps(
    module_path: str,
    pos_sims_full: dict[int, torch.Tensor],
    pos_sims_v: dict[int, torch.Tensor],
    pos_sims_u: dict[int, torch.Tensor],
    active_a: dict[int, list[int]],
    active_b: dict[int, list[int]],
    token_strs: list[str],
    label_a: str,
    label_b: str,
) -> Figure:
    """3-row grid of heatmaps: Full subcomponents, V vectors, U vectors."""
    sorted_positions = sorted(pos_sims_full)
    all_sims = [pos_sims_full, pos_sims_v, pos_sims_u]

    col_widths = [pos_sims_full[pos].shape[1] for pos in sorted_positions]
    max_na = max(pos_sims_full[pos].shape[0] for pos in sorted_positions)

    width_ratios = col_widths + [COLORBAR_WIDTH_RATIO]
    fig_width = (
        sum(col_widths) * TILE_SIZE + LABEL_PAD * (len(sorted_positions) + 1) + ROW_LABEL_PAD
    )
    fig_height = max_na * TILE_SIZE * 3 + LABEL_PAD * 4

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=len(sorted_positions) + 1,
        width_ratios=width_ratios,
        wspace=0.4,
        hspace=0.5,
    )

    im = None
    for row_idx, sims_dict in enumerate(all_sims):
        for col_idx, pos in enumerate(sorted_positions):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            sim_np = sims_dict[pos].cpu().numpy()
            im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

            indices_a = active_a[pos]
            indices_b = active_b[pos]
            ax.set_xticks(range(len(indices_b)))
            ax.set_xticklabels([str(i) for i in indices_b], fontsize=7)
            ax.set_yticks(range(len(indices_a)))
            ax.set_yticklabels([str(i) for i in indices_a], fontsize=7)

            if row_idx == 0:
                tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
                ax.set_title(f"pos {pos} '{tok}'", fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(label_a, fontsize=8)
            if row_idx == 2:
                ax.set_xlabel(label_b, fontsize=8)

    # Bold row labels on the left
    for row_idx, row_label in enumerate(ROW_LABELS):
        row_center = (
            gs[row_idx, 0].get_position(fig).y0
            + (gs[row_idx, 0].get_position(fig).y1 - gs[row_idx, 0].get_position(fig).y0) / 2
        )
        fig.text(
            0.01,
            row_center,
            row_label,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
            rotation=90,
        )

    fig.suptitle(module_path, fontsize=12)
    assert im is not None
    cbar_ax = fig.add_subplot(gs[:, -1])
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
    output_dir: str = "/tmp/compare_decompositions_by_matrix",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    decomp_a = load_decomposition(wandb_path_a, label_a, device)
    decomp_b = load_decomposition(wandb_path_b, label_b, device)
    print(f"Run A: {decomp_a.label}")
    print(f"Run B: {decomp_b.label}")

    tokenizer = get_tokenizer(decomp_a)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    token_strs = [tokenizer.decode(t) for t in tokens[0]]
    print(f"Prompt: {prompt!r}  (tokens: {tokens.shape[1]})")

    ci_a = compute_ci(decomp_a.model, tokens, decomp_a.run_info.config.sampling)
    ci_b = compute_ci(decomp_b.model, tokens, decomp_b.run_info.config.sampling)

    active_a = get_active_component_indices_per_position(ci_a, ci_threshold)
    active_b = get_active_component_indices_per_position(ci_b, ci_threshold)

    print(f"\nActive components per position (CI > {ci_threshold}):")
    for path in sorted(set(active_a) | set(active_b)):
        positions_a = active_a.get(path, {})
        positions_b = active_b.get(path, {})
        all_pos = sorted(set(positions_a) | set(positions_b))
        for pos in all_pos:
            n_a = len(positions_a.get(pos, []))
            n_b = len(positions_b.get(pos, []))
            tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
            print(f"  {path} pos {pos} '{tok}': {n_a} ({decomp_a.label}), {n_b} ({decomp_b.label})")

    shared_modules = sorted(p for p in active_a if p in active_b)
    saved_count = 0

    print(f"\n{'Module':<50} {'Pos':>4} {'Token':<10} {'Best A→B':>10} {'Best B→A':>10}")
    print("-" * 88)

    for module_path in shared_modules:
        shared_positions = sorted(set(active_a[module_path]) & set(active_b[module_path]))
        if not shared_positions:
            continue

        pos_sims_full: dict[int, torch.Tensor] = {}
        pos_sims_v: dict[int, torch.Tensor] = {}
        pos_sims_u: dict[int, torch.Tensor] = {}
        for pos in shared_positions:
            indices_a = active_a[module_path][pos]
            indices_b = active_b[module_path][pos]
            weights_a = torch.stack(
                [get_component_weight(decomp_a.model, module_path, i).flatten() for i in indices_a]
            )
            weights_b = torch.stack(
                [get_component_weight(decomp_b.model, module_path, i).flatten() for i in indices_b]
            )
            pos_sims_full[pos] = compute_pairwise_cosine_sim(weights_a, weights_b)

            v_a = torch.stack(
                [get_component_uv(decomp_a.model, module_path, i)[1] for i in indices_a]
            )
            v_b = torch.stack(
                [get_component_uv(decomp_b.model, module_path, i)[1] for i in indices_b]
            )
            pos_sims_v[pos] = compute_pairwise_cosine_sim(v_a, v_b)

            u_a = torch.stack(
                [get_component_uv(decomp_a.model, module_path, i)[0] for i in indices_a]
            )
            u_b = torch.stack(
                [get_component_uv(decomp_b.model, module_path, i)[0] for i in indices_b]
            )
            pos_sims_u[pos] = compute_pairwise_cosine_sim(u_a, u_b)

        fig = plot_matrix_heatmaps(
            module_path,
            pos_sims_full,
            pos_sims_v,
            pos_sims_u,
            active_a[module_path],
            active_b[module_path],
            token_strs,
            decomp_a.label,
            decomp_b.label,
        )
        sanitized = module_path.replace(".", "_").replace("/", "_")
        path = out / f"{sanitized}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_count += 1
        print(f"  Saved {path}")

        # Summary rows for this module
        for pos in sorted(pos_sims_full):
            sim = pos_sims_full[pos].abs()
            best_a_to_b = sim.max(dim=1).values.mean().item()
            best_b_to_a = sim.max(dim=0).values.mean().item()
            tok = token_strs[pos] if pos < len(token_strs) else f"pos{pos}"
            print(f"{module_path:<50} {pos:>4} {tok:<10} {best_a_to_b:>10.4f} {best_b_to_a:>10.4f}")

    if saved_count == 0:
        print("No shared modules with active components in both runs at any position.")
    else:
        print(f"\nSaved {saved_count} images to {out}")


if __name__ == "__main__":
    fire.Fire(main)
