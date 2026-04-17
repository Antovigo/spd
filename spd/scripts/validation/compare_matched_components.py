"""Compare components across two SPD decompositions using Hungarian matching.

Alive components of each run (as found by `find_alive_components`) are matched 1-to-1 per
(layer, matrix) by maximizing cosine similarity of their flattened weights (V @ U).
Produces a TSV of matched-pair statistics and a figure grid (rows = matrix types, columns
= layers). Each cell shows weight cosine similarity, component norms, V cosine similarity,
and U cosine similarity for the matched pairs.

Usage:
    python -m spd.scripts.validation.compare_matched_components <model_path_a> \\
        <model_path_b> [--alive-components-a=PATH] [--alive-components-b=PATH] \\
        [--label-a=NAME] [--label-b=NAME] \\
        [--output-tsv=PATH] [--output-fig=PATH]

By default the alive-components TSVs are read from each model's directory (as written by
`find_alive_components`), at `<run_dir>/alive_components.tsv`.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponents
from spd.scripts.validation.common import build_module_lookup, load_spd_run
from spd.spd_types import ModelPath


@dataclass
class MatchedPairStats:
    weight_cos_sim: float
    v_cos_sim: float
    u_cos_sim: float
    norm_a: float
    norm_b: float


@dataclass
class MatrixMatchResult:
    layer: int
    matrix: str
    pairs: list[MatchedPairStats]


def _load_alive_components(path: Path) -> dict[tuple[int, str], list[int]]:
    """Return {(layer, matrix): [component indices]} from an alive_components TSV."""
    result: dict[tuple[int, str], list[int]] = defaultdict(list)
    with path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            key = (int(record["layer"]), record["matrix"])
            result[key].append(int(record["component"]))
    for key in result:
        result[key].sort()
    return dict(result)


def _get_component_uv(model: ComponentModel, module_path: str, idx: int) -> tuple[Tensor, Tensor]:
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    return comp.U[idx], comp.V[:, idx]


def _get_component_weight(model: ComponentModel, module_path: str, idx: int) -> Tensor:
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    return comp.V[:, idx : idx + 1] @ comp.U[idx : idx + 1, :]


def _pairwise_cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


def _match_components(
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

    comp_a = model_a.components[module_path]
    assert isinstance(comp_a, LinearComponents)
    device = comp_a.V.device
    d_flat = comp_a.V.shape[0] * comp_a.U.shape[1]
    u_dim = comp_a.U.shape[1]
    v_dim = comp_a.V.shape[0]

    weights_a = [_get_component_weight(model_a, module_path, i).flatten() for i in indices_a]
    weights_b = [_get_component_weight(model_b, module_path, i).flatten() for i in indices_b]
    uv_a = [_get_component_uv(model_a, module_path, i) for i in indices_a]
    uv_b = [_get_component_uv(model_b, module_path, i) for i in indices_b]

    zero_w = torch.zeros(d_flat, device=device)
    zero_u = torch.zeros(u_dim, device=device)
    zero_v = torch.zeros(v_dim, device=device)

    while len(weights_a) < n:
        weights_a.append(zero_w)
        uv_a.append((zero_u, zero_v))
    while len(weights_b) < n:
        weights_b.append(zero_w)
        uv_b.append((zero_u, zero_v))

    stacked_a = torch.stack(weights_a)
    stacked_b = torch.stack(weights_b)

    cos_sim = _pairwise_cosine_sim(stacked_a, stacked_b)
    cos_sim = torch.nan_to_num(cos_sim, nan=0.0)

    cost = (1.0 - cos_sim.abs()).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    pairs: list[MatchedPairStats] = []
    for r, c in zip(row_ind, col_ind, strict=True):
        u_a, v_a = uv_a[r]
        u_b, v_b = uv_b[c]

        norm_a = u_a.norm().item() * v_a.norm().item()
        norm_b = u_b.norm().item() * v_b.norm().item()

        v_sim = (
            F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
            if v_a.norm() > 0 and v_b.norm() > 0
            else 0.0
        )
        u_sim = (
            F.cosine_similarity(u_a.unsqueeze(0), u_b.unsqueeze(0)).item()
            if u_a.norm() > 0 and u_b.norm() > 0
            else 0.0
        )

        pairs.append(
            MatchedPairStats(
                weight_cos_sim=cos_sim[r, c].item(),
                v_cos_sim=v_sim,
                u_cos_sim=u_sim,
                norm_a=norm_a,
                norm_b=norm_b,
            )
        )

    pairs.sort(key=lambda p: -abs(p.weight_cos_sim))
    return pairs


def _plot_matched_grid(
    results: list[MatrixMatchResult],
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    matrix_types = sorted({r.matrix for r in results})
    layer_indices = sorted({r.layer for r in results})
    n_rows = len(matrix_types)
    n_cols = len(layer_indices)

    lookup: dict[tuple[str, int], MatrixMatchResult] = {}
    for r in results:
        lookup[(r.matrix, r.layer)] = r

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

            ax_weight.bar(
                x, [abs(p.weight_cos_sim) for p in result.pairs], color="steelblue", width=0.7
            )
            ax_weight.set_ylim(0, 1)
            ax_weight.set_title("Weight cos sim", fontsize=7)
            ax_weight.tick_params(labelsize=6)
            ax_weight.set_xticks(x)
            ax_weight.set_xticklabels([str(i) for i in x], fontsize=5)

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

            ax_v.bar(x, [abs(p.v_cos_sim) for p in result.pairs], color="coral", width=0.7)
            ax_v.set_ylim(0, 1)
            ax_v.set_title("V cos sim", fontsize=7)
            ax_v.tick_params(labelsize=6)
            ax_v.set_xticks(x)
            ax_v.set_xticklabels([str(i) for i in x], fontsize=5)

            ax_u.bar(x, [abs(p.u_cos_sim) for p in result.pairs], color="mediumpurple", width=0.7)
            ax_u.set_ylim(0, 1)
            ax_u.set_title("U cos sim", fontsize=7)
            ax_u.tick_params(labelsize=6)
            ax_u.set_xticks(x)
            ax_u.set_xticklabels([str(i) for i in x], fontsize=5)

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


def _write_tsv(results: list[MatrixMatchResult], output_path: Path) -> None:
    fieldnames = [
        "layer",
        "matrix",
        "pair_idx",
        "weight_cos_sim",
        "v_cos_sim",
        "u_cos_sim",
        "norm_a",
        "norm_b",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in results:
            for i, p in enumerate(r.pairs):
                writer.writerow(
                    {
                        "layer": r.layer,
                        "matrix": r.matrix,
                        "pair_idx": i,
                        "weight_cos_sim": f"{p.weight_cos_sim:.6f}",
                        "v_cos_sim": f"{p.v_cos_sim:.6f}",
                        "u_cos_sim": f"{p.u_cos_sim:.6f}",
                        "norm_a": f"{p.norm_a:.6f}",
                        "norm_b": f"{p.norm_b:.6f}",
                    }
                )


def compare_matched_components(
    model_path_a: ModelPath,
    model_path_b: ModelPath,
    alive_components_a: str | None = None,
    alive_components_b: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    output_tsv: str | None = None,
    output_fig: str | None = None,
) -> Path:
    """Match alive components of two SPD decompositions and write a TSV + figure of the matches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, _config_a, run_dir_a = load_spd_run(model_path_a)
    model_b, _config_b, run_dir_b = load_spd_run(model_path_b)
    model_a = model_a.to(device)
    model_b = model_b.to(device)

    if label_a is None:
        label_a = run_dir_a.name
    if label_b is None:
        label_b = run_dir_b.name

    alive_path_a = (
        Path(alive_components_a).expanduser()
        if alive_components_a
        else run_dir_a / "alive_components.tsv"
    )
    alive_path_b = (
        Path(alive_components_b).expanduser()
        if alive_components_b
        else run_dir_b / "alive_components.tsv"
    )
    assert alive_path_a.exists(), f"alive components file not found for A: {alive_path_a}"
    assert alive_path_b.exists(), f"alive components file not found for B: {alive_path_b}"

    alive_a = _load_alive_components(alive_path_a)
    alive_b = _load_alive_components(alive_path_b)
    logger.info(
        f"Run A alive components: {sum(len(v) for v in alive_a.values())} across {len(alive_a)} matrices"
    )
    logger.info(
        f"Run B alive components: {sum(len(v) for v in alive_b.values())} across {len(alive_b)} matrices"
    )

    module_lookup = build_module_lookup(model_a.target_module_paths)
    module_lookup_b = build_module_lookup(model_b.target_module_paths)
    assert module_lookup == module_lookup_b, (
        "Models A and B have different module paths; cannot match components across them."
    )

    shared_keys = sorted(alive_a.keys() & alive_b.keys())

    results: list[MatrixMatchResult] = []
    with torch.no_grad():
        for key in shared_keys:
            layer, matrix = key
            assert key in module_lookup, (
                f"(layer={layer}, matrix={matrix}) from alive components is not a decomposed "
                f"module. Available: {sorted(module_lookup.keys())}"
            )
            module_path = module_lookup[key]
            indices_a = alive_a[key]
            indices_b = alive_b[key]

            pairs = _match_components(model_a, model_b, module_path, indices_a, indices_b)
            logger.info(
                f"  {module_path}: {len(indices_a)} vs {len(indices_b)} components -> "
                f"{len(pairs)} matched pairs"
            )
            results.append(MatrixMatchResult(layer=layer, matrix=matrix, pairs=pairs))

    assert results, "No (layer, matrix) shared between the two alive-components files."

    tsv_path = Path(output_tsv).expanduser() if output_tsv else run_dir_a / "matched_components.tsv"
    fig_path = Path(output_fig).expanduser() if output_fig else run_dir_a / "matched_components.png"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    _write_tsv(results, tsv_path)
    _plot_matched_grid(results, label_a, label_b, fig_path)
    logger.info(f"Saved {tsv_path}")
    logger.info(f"Saved {fig_path}")
    return tsv_path


if __name__ == "__main__":
    fire.Fire(compare_matched_components)
