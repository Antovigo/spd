"""Compare components across two SPD decompositions using Hungarian matching.

Components are matched 1-to-1 per matrix by maximizing cosine similarity of their
flattened weights (V @ U). Produces a single figure with a grid: rows = matrix types,
columns = layer indices. Each cell has 4 subpanels showing weight cosine similarity,
component norms, V cosine similarity, and U cosine similarity for the matched pairs.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_matched_components \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_matched_components
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import linear_sum_assignment

from spd.models.component_model import ComponentModel
from spd.scripts.compare_decompositions.utils import (
    compute_ci,
    compute_pairwise_cosine_sim,
    get_active_component_indices,
    get_component_uv,
    get_component_weight,
    get_tokenizer,
    load_decomposition,
)

_MODULE_PATH_PATTERN = re.compile(r"h\.(\d+)\.(.*)")


@dataclass
class MatchedPairStats:
    weight_cos_sim: float
    v_cos_sim: float
    u_cos_sim: float
    norm_a: float
    norm_b: float


@dataclass
class MatrixMatchResult:
    layer_idx: int
    matrix_type: str
    pairs: list[MatchedPairStats]


def parse_module_path(path: str) -> tuple[int, str] | None:
    m = _MODULE_PATH_PATTERN.match(path)
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


def match_components(
    model_a: ComponentModel,
    model_b: ComponentModel,
    module_path: str,
    indices_a: list[int],
    indices_b: list[int],
) -> list[MatchedPairStats]:
    na, nb = len(indices_a), len(indices_b)
    n = max(na, nb)
    if n == 0:
        return []

    # Get component weights and UV vectors
    comp_a = model_a.components[module_path]
    d_flat = comp_a.V.shape[0] * comp_a.U.shape[1]

    weights_a = [get_component_weight(model_a, module_path, i).flatten() for i in indices_a]
    weights_b = [get_component_weight(model_b, module_path, i).flatten() for i in indices_b]
    uv_a = [get_component_uv(model_a, module_path, i) for i in indices_a]
    uv_b = [get_component_uv(model_b, module_path, i) for i in indices_b]

    # Pad with zeros if different counts
    zero_weight = torch.zeros(d_flat, device=comp_a.V.device)
    u_dim = comp_a.U.shape[1]
    v_dim = comp_a.V.shape[0]
    zero_u = torch.zeros(u_dim, device=comp_a.V.device)
    zero_v = torch.zeros(v_dim, device=comp_a.V.device)

    while len(weights_a) < n:
        weights_a.append(zero_weight)
        uv_a.append((zero_u, zero_v))
    while len(weights_b) < n:
        weights_b.append(zero_weight)
        uv_b.append((zero_u, zero_v))

    stacked_a = torch.stack(weights_a)
    stacked_b = torch.stack(weights_b)

    # Pairwise cosine similarity; handle zero vectors (NaN → 0)
    cos_sim = compute_pairwise_cosine_sim(stacked_a, stacked_b)
    cos_sim = torch.nan_to_num(cos_sim, nan=0.0)

    # Hungarian matching: minimize cost = 1 - |cos_sim|
    cost = (1.0 - cos_sim.abs()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: list[MatchedPairStats] = []
    for r, c in zip(row_ind, col_ind, strict=True):
        u_a, v_a = uv_a[r]
        u_b, v_b = uv_b[c]

        norm_a_val = u_a.norm().item() * v_a.norm().item()
        norm_b_val = u_b.norm().item() * v_b.norm().item()

        # Cosine similarity for V and U vectors; zero vectors → 0
        if v_a.norm() > 0 and v_b.norm() > 0:
            v_sim = F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
        else:
            v_sim = 0.0
        if u_a.norm() > 0 and u_b.norm() > 0:
            u_sim = F.cosine_similarity(u_a.unsqueeze(0), u_b.unsqueeze(0)).item()
        else:
            u_sim = 0.0

        pairs.append(
            MatchedPairStats(
                weight_cos_sim=cos_sim[r, c].item(),
                v_cos_sim=v_sim,
                u_cos_sim=u_sim,
                norm_a=norm_a_val,
                norm_b=norm_b_val,
            )
        )

    # Sort pairs by descending weight cosine similarity
    pairs.sort(key=lambda p: -abs(p.weight_cos_sim))
    return pairs


def plot_matched_grid(
    results: list[MatrixMatchResult],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    matrix_types = sorted({r.matrix_type for r in results})
    layer_indices = sorted({r.layer_idx for r in results})
    n_rows = len(matrix_types)
    n_cols = len(layer_indices)

    # Build lookup
    lookup: dict[tuple[str, int], MatrixMatchResult] = {}
    for r in results:
        lookup[(r.matrix_type, r.layer_idx)] = r

    # Global max norm for shared scatter plot scale
    max_norm = max(max(max(p.norm_a, p.norm_b) for p in r.pairs) for r in results if r.pairs)
    norm_lim = max_norm * 1.05 if max_norm > 0 else 1.0

    cell_w, cell_h = 3.5, 3.0
    fig = plt.figure(figsize=(n_cols * cell_w + 1.5, n_rows * cell_h + 1.5))

    outer_gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=0.08,
        right=0.96,
        top=0.93,
        bottom=0.05,
        wspace=0.35,
        hspace=0.45,
    )

    for row_idx, mtype in enumerate(matrix_types):
        for col_idx, layer_idx in enumerate(layer_indices):
            result = lookup.get((mtype, layer_idx))

            inner_gs = GridSpecFromSubplotSpec(
                2, 2, subplot_spec=outer_gs[row_idx, col_idx], wspace=0.4, hspace=0.5
            )

            ax_weight = fig.add_subplot(inner_gs[0, 0])
            ax_norm = fig.add_subplot(inner_gs[0, 1])
            ax_v = fig.add_subplot(inner_gs[1, 0])
            ax_u = fig.add_subplot(inner_gs[1, 1])

            if result is None or not result.pairs:
                for ax in (ax_weight, ax_norm, ax_v, ax_u):
                    ax.set_visible(False)
                continue

            n_pairs = len(result.pairs)
            x = np.arange(n_pairs)

            # Top-left: weight cosine similarity bars
            weight_sims = [abs(p.weight_cos_sim) for p in result.pairs]
            ax_weight.bar(x, weight_sims, color="steelblue", width=0.7)
            ax_weight.set_ylim(0, 1)
            ax_weight.set_title("Weight cos sim", fontsize=7)
            ax_weight.tick_params(labelsize=6)
            ax_weight.set_xticks(x)
            ax_weight.set_xticklabels([str(i) for i in x], fontsize=5)

            # Top-right: norm scatter plot
            norms_a = [p.norm_a for p in result.pairs]
            norms_b = [p.norm_b for p in result.pairs]
            ax_norm.scatter(norms_b, norms_a, s=15, c="steelblue", alpha=0.7)
            ax_norm.plot([0, norm_lim], [0, norm_lim], "k--", linewidth=0.5, alpha=0.3)
            ax_norm.set_xlim(0, norm_lim)
            ax_norm.set_ylim(0, norm_lim)
            ax_norm.set_aspect("equal")
            ax_norm.set_title("||U||·||V|| norm", fontsize=7)
            ax_norm.tick_params(labelsize=5)
            if row_idx == n_rows - 1:
                ax_norm.set_xlabel(label_b, fontsize=6)
            if col_idx == 0:
                ax_norm.set_ylabel(label_a, fontsize=6)

            # Bottom-left: V cosine similarity bars
            v_sims = [abs(p.v_cos_sim) for p in result.pairs]
            ax_v.bar(x, v_sims, color="coral", width=0.7)
            ax_v.set_ylim(0, 1)
            ax_v.set_title("V cos sim", fontsize=7)
            ax_v.tick_params(labelsize=6)
            ax_v.set_xticks(x)
            ax_v.set_xticklabels([str(i) for i in x], fontsize=5)

            # Bottom-right: U cosine similarity bars
            u_sims = [abs(p.u_cos_sim) for p in result.pairs]
            ax_u.bar(x, u_sims, color="mediumpurple", width=0.7)
            ax_u.set_ylim(0, 1)
            ax_u.set_title("U cos sim", fontsize=7)
            ax_u.tick_params(labelsize=6)
            ax_u.set_xticks(x)
            ax_u.set_xticklabels([str(i) for i in x], fontsize=5)

            # Row / column labels
            if col_idx == 0:
                ax_weight.set_ylabel(mtype, fontsize=8, fontweight="bold")
            if row_idx == 0:
                ax_weight.annotate(
                    f"Layer {layer_idx}",
                    xy=(0.5, 1.0),
                    xycoords="axes fraction",
                    xytext=(0, 25),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                )

    fig.suptitle(f"Hungarian-matched components: {label_a} vs {label_b}", fontsize=12, y=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def write_tsv(results: list[MatrixMatchResult], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "layer_idx",
                "matrix_type",
                "pair_idx",
                "weight_cos_sim",
                "v_cos_sim",
                "u_cos_sim",
                "norm_a",
                "norm_b",
            ]
        )
        for r in results:
            for i, p in enumerate(r.pairs):
                writer.writerow(
                    [
                        r.layer_idx,
                        r.matrix_type,
                        i,
                        f"{p.weight_cos_sim:.6f}",
                        f"{p.v_cos_sim:.6f}",
                        f"{p.u_cos_sim:.6f}",
                        f"{p.norm_a:.6f}",
                        f"{p.norm_b:.6f}",
                    ]
                )
    print(f"Saved {output_path}")


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
    output_dir: str = "/tmp/compare_matched_components",
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

    shared_modules = sorted(p for p in active_a if p in active_b)

    results: list[MatrixMatchResult] = []
    for module_path in shared_modules:
        parsed = parse_module_path(module_path)
        if parsed is None:
            print(f"  Skipping {module_path} (cannot parse layer/matrix type)")
            continue

        layer_idx, matrix_type = parsed
        indices_a = active_a[module_path]
        indices_b = active_b[module_path]

        pairs = match_components(decomp_a.model, decomp_b.model, module_path, indices_a, indices_b)

        print(
            f"  {module_path}: {len(indices_a)} vs {len(indices_b)} components → "
            f"{len(pairs)} matched pairs"
        )
        for i, p in enumerate(pairs):
            print(
                f"    pair {i}: weight={p.weight_cos_sim:.3f}  "
                f"V={p.v_cos_sim:.3f}  U={p.u_cos_sim:.3f}  "
                f"norm_a={p.norm_a:.3f}  norm_b={p.norm_b:.3f}"
            )

        results.append(MatrixMatchResult(layer_idx=layer_idx, matrix_type=matrix_type, pairs=pairs))

    if not results:
        print("No shared modules with active components found.")
        return

    plot_matched_grid(results, decomp_a.label, decomp_b.label, out / "matched_components.png")
    write_tsv(results, out / "matched_components.tsv")


if __name__ == "__main__":
    fire.Fire(main)
