"""Compare components across two SPD decompositions by max cosine similarity.

For each alive component in model A, finds the component in model B with the highest absolute
cosine similarity on the flattened rank-one weight (`V[:, i] @ U[i, :]`). For the matched
pair, also reports the cosine similarity of their V and U vectors and their `||U|| * ||V||`
norms. Unlike `compare_matched_components` (Hungarian 1-to-1), multiple A components may map
to the same B component.

With `--random-b`, model B's component weights are re-initialized (Kaiming-normal, same as
at construction time) and, per `(layer, matrix)`, a random subset of size
`len(alive_b_for_that_matrix)` is drawn from model B's total `C` components. This controls
for the pool-size inflation of max-cosine against an untrained model. The random draw is
repeated `--n-random-samples` times (each with a different seed) and one TSV row is written
per `(a_component, draw)` so downstream code can aggregate.

Usage:
    python -m spd.scripts.validation.compare_components <model_path_a> <model_path_b> \\
        [--alive-components-a=PATH] [--alive-components-b=PATH] \\
        [--random-b] [--n-random-samples=N] [--random-seed=S] \\
        [--label-a=NAME] [--label-b=NAME] \\
        [--output-tsv=PATH] [--output-fig=PATH]
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
from torch import Tensor

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponents
from spd.scripts.validation.common import build_module_lookup, load_spd_run
from spd.spd_types import ModelPath
from spd.utils.module_utils import init_param_


@dataclass
class MaxCosMatch:
    a_idx: int
    b_idx: int
    weight_cos_sim: float
    v_cos_sim: float
    u_cos_sim: float
    norm_a: float
    norm_b: float
    draw: int


@dataclass
class MatrixMatchResult:
    layer: int
    matrix: str
    matches: list[MaxCosMatch]


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


def _stack_weights(
    model: ComponentModel, module_path: str, indices: list[int]
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Return (stacked_flat_weights, per-component V vectors, per-component U vectors)."""
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    weights = []
    vs = []
    us = []
    for i in indices:
        v = comp.V[:, i]
        u = comp.U[i]
        vs.append(v)
        us.append(u)
        weights.append(torch.outer(v, u).flatten())
    return torch.stack(weights), vs, us


def _pairwise_cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


def _cos_sim_or_zero(a: Tensor, b: Tensor) -> float:
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _match_max_cos(
    model_a: ComponentModel,
    model_b: ComponentModel,
    module_path: str,
    indices_a: list[int],
    indices_b: list[int],
    draw: int,
) -> list[MaxCosMatch]:
    """For each a in indices_a, pick the b in indices_b with highest |cos sim| on flat weight."""
    if not indices_a or not indices_b:
        return []

    weights_a, vs_a, us_a = _stack_weights(model_a, module_path, indices_a)
    weights_b, vs_b, us_b = _stack_weights(model_b, module_path, indices_b)

    cos_sim = torch.nan_to_num(_pairwise_cosine_sim(weights_a, weights_b), nan=0.0)
    best_cols = cos_sim.abs().argmax(dim=1).cpu().tolist()

    matches: list[MaxCosMatch] = []
    for r, c in enumerate(best_cols):
        v_a, u_a = vs_a[r], us_a[r]
        v_b, u_b = vs_b[c], us_b[c]
        matches.append(
            MaxCosMatch(
                a_idx=indices_a[r],
                b_idx=indices_b[c],
                weight_cos_sim=cos_sim[r, c].item(),
                v_cos_sim=_cos_sim_or_zero(v_a, v_b),
                u_cos_sim=_cos_sim_or_zero(u_a, u_b),
                norm_a=u_a.norm().item() * v_a.norm().item(),
                norm_b=u_b.norm().item() * v_b.norm().item(),
                draw=draw,
            )
        )
    return matches


def _reinit_components_(model: ComponentModel, seed: int) -> None:
    """Re-initialize every LinearComponents' V and U with fresh Kaiming-normal samples."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for comp in model.components.values():
            if not isinstance(comp, LinearComponents):
                continue
            v_dim, c_dim = comp.V.shape
            init_param_(comp.V, fan_val=v_dim, nonlinearity="linear", generator=gen)
            init_param_(comp.U, fan_val=c_dim, nonlinearity="linear", generator=gen)


def _sample_random_pool(total_c: int, pool_size: int, rng: np.random.Generator) -> list[int]:
    assert pool_size <= total_c, f"Cannot sample pool of size {pool_size} from {total_c} components"
    return sorted(rng.choice(total_c, size=pool_size, replace=False).tolist())


def _write_tsv(results: list[MatrixMatchResult], output_path: Path) -> None:
    fieldnames = [
        "layer",
        "matrix",
        "draw",
        "a_component",
        "b_component",
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
            for m in r.matches:
                writer.writerow(
                    {
                        "layer": r.layer,
                        "matrix": r.matrix,
                        "draw": m.draw,
                        "a_component": m.a_idx,
                        "b_component": m.b_idx,
                        "weight_cos_sim": f"{m.weight_cos_sim:.6f}",
                        "v_cos_sim": f"{m.v_cos_sim:.6f}",
                        "u_cos_sim": f"{m.u_cos_sim:.6f}",
                        "norm_a": f"{m.norm_a:.6f}",
                        "norm_b": f"{m.norm_b:.6f}",
                    }
                )


def _aggregate_per_a(matches: list[MaxCosMatch]) -> dict[int, dict[str, float]]:
    """Mean of |cos sims| across draws, grouped by a_idx."""
    by_a: dict[int, list[MaxCosMatch]] = defaultdict(list)
    for m in matches:
        by_a[m.a_idx].append(m)
    out: dict[int, dict[str, float]] = {}
    for a_idx, ms in by_a.items():
        out[a_idx] = {
            "weight_cos_sim": float(np.mean([abs(m.weight_cos_sim) for m in ms])),
            "v_cos_sim": float(np.mean([abs(m.v_cos_sim) for m in ms])),
            "u_cos_sim": float(np.mean([abs(m.u_cos_sim) for m in ms])),
            "norm_a": float(np.mean([m.norm_a for m in ms])),
            "norm_b": float(np.mean([m.norm_b for m in ms])),
        }
    return out


def _plot_max_cos_grid(
    results: list[MatrixMatchResult],
    label_a: str,
    label_b: str,
    output_path: Path,
    is_random: bool,
) -> None:
    matrix_types = sorted({r.matrix for r in results})
    layer_indices = sorted({r.layer for r in results})
    n_rows = len(matrix_types)
    n_cols = len(layer_indices)

    lookup: dict[tuple[str, int], MatrixMatchResult] = {(r.matrix, r.layer): r for r in results}

    all_norms = [max(m.norm_a, m.norm_b) for r in results for m in r.matches]
    max_norm = max(all_norms) if all_norms else 1.0
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

            if result is None or not result.matches:
                for ax in (ax_weight, ax_norm, ax_v, ax_u):
                    ax.set_visible(False)
                continue

            per_a = _aggregate_per_a(result.matches)
            a_order = sorted(per_a.keys(), key=lambda a: -per_a[a]["weight_cos_sim"])
            x = np.arange(len(a_order))

            ax_weight.bar(
                x, [per_a[a]["weight_cos_sim"] for a in a_order], color="steelblue", width=0.7
            )
            ax_weight.set_ylim(0, 1)
            ax_weight.set_title("Max |weight cos|", fontsize=7)
            ax_weight.tick_params(labelsize=6)
            ax_weight.set_xticks(x)
            ax_weight.set_xticklabels([str(a) for a in a_order], fontsize=5, rotation=90)

            norms_a = [per_a[a]["norm_a"] for a in a_order]
            norms_b = [per_a[a]["norm_b"] for a in a_order]
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

            ax_v.bar(x, [per_a[a]["v_cos_sim"] for a in a_order], color="coral", width=0.7)
            ax_v.set_ylim(0, 1)
            ax_v.set_title("V cos sim (matched)", fontsize=7)
            ax_v.tick_params(labelsize=6)
            ax_v.set_xticks(x)
            ax_v.set_xticklabels([str(a) for a in a_order], fontsize=5, rotation=90)

            ax_u.bar(x, [per_a[a]["u_cos_sim"] for a in a_order], color="mediumpurple", width=0.7)
            ax_u.set_ylim(0, 1)
            ax_u.set_title("U cos sim (matched)", fontsize=7)
            ax_u.tick_params(labelsize=6)
            ax_u.set_xticks(x)
            ax_u.set_xticklabels([str(a) for a in a_order], fontsize=5, rotation=90)

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

    mode = "random-B baseline" if is_random else "max-cos matched"
    fig.suptitle(f"Components: {label_a} vs {label_b} ({mode})", fontsize=12, y=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compare_components(
    model_path_a: ModelPath,
    model_path_b: ModelPath,
    alive_components_a: str | None = None,
    alive_components_b: str | None = None,
    random_b: bool = False,
    n_random_samples: int = 10,
    random_seed: int = 0,
    label_a: str | None = None,
    label_b: str | None = None,
    output_tsv: str | None = None,
    output_fig: str | None = None,
) -> Path:
    """Max-cosine match alive components of A against B (or random-B baseline)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, _config_a, run_dir_a = load_spd_run(model_path_a)
    model_b, _config_b, run_dir_b = load_spd_run(model_path_b)
    model_a = model_a.to(device)
    model_b = model_b.to(device)

    if label_a is None:
        label_a = run_dir_a.name
    if label_b is None:
        label_b = f"{run_dir_b.name}-random" if random_b else run_dir_b.name

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
        f"Run A alive components: {sum(len(v) for v in alive_a.values())} across "
        f"{len(alive_a)} matrices"
    )
    logger.info(
        f"Run B alive components: {sum(len(v) for v in alive_b.values())} across "
        f"{len(alive_b)} matrices"
    )

    module_lookup = build_module_lookup(model_a.target_module_paths)
    module_lookup_b = build_module_lookup(model_b.target_module_paths)
    assert module_lookup == module_lookup_b, (
        "Models A and B have different module paths; cannot match components across them."
    )

    shared_keys = sorted(alive_a.keys() & alive_b.keys())
    assert shared_keys, "No (layer, matrix) shared between the two alive-components files."

    results: list[MatrixMatchResult] = []

    with torch.no_grad():
        if not random_b:
            for key in shared_keys:
                layer, matrix = key
                module_path = module_lookup[key]
                matches = _match_max_cos(
                    model_a, model_b, module_path, alive_a[key], alive_b[key], draw=0
                )
                logger.info(
                    f"  {module_path}: {len(alive_a[key])} A comps vs {len(alive_b[key])} B "
                    f"comps -> {len(matches)} matches"
                )
                results.append(MatrixMatchResult(layer=layer, matrix=matrix, matches=matches))
        else:
            per_matrix_matches: dict[tuple[int, str], list[MaxCosMatch]] = {
                key: [] for key in shared_keys
            }
            for draw in range(n_random_samples):
                _reinit_components_(model_b, seed=random_seed + draw)
                rng = np.random.default_rng(random_seed + draw)
                for key in shared_keys:
                    module_path = module_lookup[key]
                    comp_b = model_b.components[module_path]
                    assert isinstance(comp_b, LinearComponents)
                    total_c = comp_b.U.shape[0]
                    pool = _sample_random_pool(total_c, len(alive_b[key]), rng)
                    matches = _match_max_cos(
                        model_a, model_b, module_path, alive_a[key], pool, draw=draw
                    )
                    per_matrix_matches[key].extend(matches)
                logger.info(f"random draw {draw + 1}/{n_random_samples} done")
            for key in shared_keys:
                layer, matrix = key
                results.append(
                    MatrixMatchResult(layer=layer, matrix=matrix, matches=per_matrix_matches[key])
                )

    stem = "compare_components_random" if random_b else "compare_components"
    tsv_path = Path(output_tsv).expanduser() if output_tsv else run_dir_a / f"{stem}.tsv"
    fig_path = Path(output_fig).expanduser() if output_fig else run_dir_a / f"{stem}.png"
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    _write_tsv(results, tsv_path)
    _plot_max_cos_grid(results, label_a, label_b, fig_path, is_random=random_b)
    logger.info(f"Saved {tsv_path}")
    logger.info(f"Saved {fig_path}")
    return tsv_path


if __name__ == "__main__":
    fire.Fire(compare_components)
