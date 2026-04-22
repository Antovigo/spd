"""All pairwise cosine similarities between alive components of two SPD decompositions.

For each `(layer, matrix)` shared between the two models' alive-components TSVs, computes the
full pairwise cosine-similarity matrix over alive components on three vectors each:
- the flattened rank-one weight `V[:, i] @ U[i, :]`
- the `V` vector
- the `U` vector

Also runs the same comparison against a Kaiming-normal-reinitialised copy of model B (same
alive indices, same shapes) as a random baseline; the baseline cosine similarities are written
as extra columns on each row. Unlike `compare_components.py`, there is no argmax or Hungarian
matching — every pair is written.

Usage:
    python -m spd.scripts.validation.all_cosine_similarities <model_path_a> <model_path_b> \\
        [--alive-components-a=PATH] [--alive-components-b=PATH] \\
        [--random-seed=S] [--output=PATH]
"""

import copy
import csv
from collections import defaultdict
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch import Tensor

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponents
from spd.scripts.validation.common import build_module_lookup, load_spd_run
from spd.spd_types import ModelPath
from spd.utils.module_utils import init_param_


def _load_alive_components(path: Path) -> dict[tuple[int, str], list[int]]:
    result: dict[tuple[int, str], list[int]] = defaultdict(list)
    with path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            key = (int(record["layer"]), record["matrix"])
            result[key].append(int(record["component"]))
    for key in result:
        result[key].sort()
    return dict(result)


def _stack_vectors(
    model: ComponentModel, module_path: str, indices: list[int]
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (flat_weights [N, d_in*d_out], v_vectors [N, d_in], u_vectors [N, d_out])."""
    comp = model.components[module_path]
    assert isinstance(comp, LinearComponents)
    vs = torch.stack([comp.V[:, i] for i in indices])
    us = torch.stack([comp.U[i] for i in indices])
    weights = torch.stack([torch.outer(vs[i], us[i]).flatten() for i in range(len(indices))])
    return weights, vs, us


def _pairwise_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    """Cosine similarity matrix with zero rows/cols wherever the underlying norm is zero."""
    a_norms = a.norm(dim=-1)
    b_norms = b.norm(dim=-1)
    cos = F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T
    cos = torch.nan_to_num(cos, nan=0.0)
    zero_a = (a_norms == 0).unsqueeze(1)
    zero_b = (b_norms == 0).unsqueeze(0)
    cos = cos.masked_fill(zero_a, 0.0).masked_fill(zero_b, 0.0)
    return cos


def _reinit_components_(model: ComponentModel, seed: int) -> None:
    """Re-initialize every LinearComponents' V and U with Kaiming-normal samples."""
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


def _compute_cos_sims(
    model_a: ComponentModel,
    model_b: ComponentModel,
    module_path: str,
    indices_a: list[int],
    indices_b: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (weight_cos, v_cos, u_cos) with double-negative correction applied to v/u."""
    weights_a, vs_a, us_a = _stack_vectors(model_a, module_path, indices_a)
    weights_b, vs_b, us_b = _stack_vectors(model_b, module_path, indices_b)

    weight_cos = _pairwise_cos_sim(weights_a, weights_b).cpu()
    v_cos = _pairwise_cos_sim(vs_a, vs_b).cpu()
    u_cos = _pairwise_cos_sim(us_a, us_b).cpu()

    # Double-negative correction: when both V and U cos sims are negative, the two sign
    # flips cancel in the outer product V @ U, so the rank-one components are equivalent
    # up to a shared sign flip on each side. Flip both to positive.
    both_neg = (v_cos < 0) & (u_cos < 0)
    v_cos = torch.where(both_neg, -v_cos, v_cos)
    u_cos = torch.where(both_neg, -u_cos, u_cos)
    return weight_cos, v_cos, u_cos


def all_cosine_similarities(
    model_path_a: ModelPath,
    model_path_b: ModelPath,
    alive_components_a: str | None = None,
    alive_components_b: str | None = None,
    random_seed: int = 0,
    output: str | None = None,
) -> Path:
    """Compute all pairwise U/V/weight cosine similarities between alive components of A and B,
    plus a random-init baseline for B written as extra columns on each row."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, _config_a, run_dir_a = load_spd_run(model_path_a)
    model_b, _config_b, run_dir_b = load_spd_run(model_path_b)
    model_a = model_a.to(device)
    model_b = model_b.to(device)

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
        "Models A and B have different module paths; cannot compare components across them."
    )

    shared_keys = sorted(alive_a.keys() & alive_b.keys())
    assert shared_keys, "No (layer, matrix) shared between the two alive-components files."

    # Build a random-init clone of model B (keeps the same module shapes + alive index set).
    model_b_random = copy.deepcopy(model_b)
    _reinit_components_(model_b_random, seed=random_seed)

    out_path = (
        Path(output).expanduser()
        if output
        else run_dir_a / f"all_cosine_similarities_{run_dir_b.name}.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer",
        "matrix",
        "a_component",
        "b_component",
        "weight_cos_sim",
        "v_cos_sim",
        "u_cos_sim",
        "weight_cos_sim_random",
        "v_cos_sim_random",
        "u_cos_sim_random",
    ]

    n_rows = 0
    with torch.no_grad(), out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in shared_keys:
            layer, matrix = key
            module_path = module_lookup[key]
            indices_a = alive_a[key]
            indices_b = alive_b[key]

            weight_cos, v_cos, u_cos = _compute_cos_sims(
                model_a, model_b, module_path, indices_a, indices_b
            )
            weight_cos_r, v_cos_r, u_cos_r = _compute_cos_sims(
                model_a, model_b_random, module_path, indices_a, indices_b
            )

            for r, a_idx in enumerate(indices_a):
                for c, b_idx in enumerate(indices_b):
                    writer.writerow(
                        {
                            "layer": layer,
                            "matrix": matrix,
                            "a_component": a_idx,
                            "b_component": b_idx,
                            "weight_cos_sim": f"{weight_cos[r, c].item():.6f}",
                            "v_cos_sim": f"{v_cos[r, c].item():.6f}",
                            "u_cos_sim": f"{u_cos[r, c].item():.6f}",
                            "weight_cos_sim_random": f"{weight_cos_r[r, c].item():.6f}",
                            "v_cos_sim_random": f"{v_cos_r[r, c].item():.6f}",
                            "u_cos_sim_random": f"{u_cos_r[r, c].item():.6f}",
                        }
                    )
                    n_rows += 1
            logger.info(
                f"  {module_path}: {len(indices_a)} × {len(indices_b)} = "
                f"{len(indices_a) * len(indices_b)} pairs"
            )

    logger.info(f"Wrote {n_rows} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(all_cosine_similarities)
