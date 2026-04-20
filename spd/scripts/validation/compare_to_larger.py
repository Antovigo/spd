"""Compare targeted-decomposition components against a larger decomposition.

For each alive component in the targeted model, find the component in the larger model (searched
across ALL of its components, not only alive ones) with the highest absolute cosine similarity on
the flattened rank-one weight (`V[:, i] @ U[i, :]`). Report per-pair weight / V / U cos sim and
`||U|| * ||V||` norms, matching the schema of `compare_components.py`.

A random baseline is also produced by re-initializing the targeted model's components
(Kaiming-normal) and rerunning the match `--n-random-samples` times. This controls for how much
structure a random rank-one slice picks up when matched against a large component pool.

Outputs two TSVs next to the targeted run (suffixed with the larger run's folder name):
    compare_to_larger_<folder-b>.tsv
    compare_to_larger_random_<folder-b>.tsv

Usage:
    python -m spd.scripts.validation.compare_to_larger <targeted_path> <larger_path> \\
        [--alive-components-targeted=PATH] \\
        [--n-random-samples=N] [--random-seed=S] \\
        [--chunk-size=N] [--output-dir=PATH]
"""

from pathlib import Path

import fire
import torch
import torch.nn.functional as F

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponents
from spd.scripts.validation.common import build_module_lookup, load_spd_run
from spd.scripts.validation.compare_components import (
    MatrixMatchResult,
    MaxCosMatch,
    _cos_sim_or_zero,
    _load_alive_components,
    _reinit_components_,
    _stack_weights,
    _write_tsv,
)
from spd.spd_types import ModelPath


def _match_against_all_b_chunked(
    model_a: ComponentModel,
    model_b: ComponentModel,
    module_path: str,
    indices_a: list[int],
    draw: int,
    chunk_size: int,
) -> list[MaxCosMatch]:
    """For each a in indices_a, find the b in ALL of model_b's components with max |weight cos sim|.

    Iterates b in chunks of `chunk_size` to avoid materialising the full `(C_b, d_in*d_out)` weight
    stack at once.
    """
    if not indices_a:
        return []

    comp_b = model_b.components[module_path]
    assert isinstance(comp_b, LinearComponents)
    total_c_b = comp_b.U.shape[0]

    weights_a, vs_a, us_a = _stack_weights(model_a, module_path, indices_a)
    weights_a_norm = F.normalize(weights_a, dim=-1)
    device = weights_a.device
    n_a = len(indices_a)

    best_abs = torch.full((n_a,), -1.0, device=device)
    best_signed = torch.zeros(n_a, device=device)
    best_b_idx = torch.zeros(n_a, dtype=torch.long, device=device)

    for start in range(0, total_c_b, chunk_size):
        chunk_ids = list(range(start, min(start + chunk_size, total_c_b)))
        weights_b_chunk, _, _ = _stack_weights(model_b, module_path, chunk_ids)
        weights_b_norm = F.normalize(weights_b_chunk, dim=-1)
        cos = torch.nan_to_num(weights_a_norm @ weights_b_norm.T, nan=0.0)
        chunk_max_abs, chunk_argmax = cos.abs().max(dim=1)
        chunk_max_signed = cos.gather(1, chunk_argmax.unsqueeze(1)).squeeze(1)

        improved = chunk_max_abs > best_abs
        best_abs = torch.where(improved, chunk_max_abs, best_abs)
        best_signed = torch.where(improved, chunk_max_signed, best_signed)
        best_b_idx = torch.where(improved, chunk_argmax + start, best_b_idx)

    matches: list[MaxCosMatch] = []
    for r in range(n_a):
        b_idx = int(best_b_idx[r].item())
        v_b = comp_b.V[:, b_idx]
        u_b = comp_b.U[b_idx]
        v_a, u_a = vs_a[r], us_a[r]
        matches.append(
            MaxCosMatch(
                a_idx=indices_a[r],
                b_idx=b_idx,
                weight_cos_sim=best_signed[r].item(),
                v_cos_sim=_cos_sim_or_zero(v_a, v_b),
                u_cos_sim=_cos_sim_or_zero(u_a, u_b),
                norm_a=u_a.norm().item() * v_a.norm().item(),
                norm_b=u_b.norm().item() * v_b.norm().item(),
                draw=draw,
            )
        )
    return matches


def _match_all_pool(
    model_a: ComponentModel,
    model_b: ComponentModel,
    module_lookup: dict[tuple[int, str], str],
    alive_a: dict[tuple[int, str], list[int]],
    shared_keys: list[tuple[int, str]],
    draw: int,
    chunk_size: int,
) -> list[MatrixMatchResult]:
    results: list[MatrixMatchResult] = []
    for key in shared_keys:
        layer, matrix = key
        module_path = module_lookup[key]
        matches = _match_against_all_b_chunked(
            model_a, model_b, module_path, alive_a[key], draw=draw, chunk_size=chunk_size
        )
        results.append(MatrixMatchResult(layer=layer, matrix=matrix, matches=matches))
    return results


def compare_to_larger(
    model_path_targeted: ModelPath,
    model_path_larger: ModelPath,
    alive_components_targeted: str | None = None,
    n_random_samples: int = 10,
    random_seed: int = 0,
    chunk_size: int = 64,
    output_dir: str | None = None,
) -> tuple[Path, Path]:
    """Match targeted alive components to the nearest component in a larger decomposition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_a, _config_a, run_dir_a = load_spd_run(model_path_targeted)
    model_b, _config_b, run_dir_b = load_spd_run(model_path_larger)
    model_a = model_a.to(device)
    model_b = model_b.to(device)

    alive_path_a = (
        Path(alive_components_targeted).expanduser()
        if alive_components_targeted
        else run_dir_a / "alive_components.tsv"
    )
    assert alive_path_a.exists(), f"alive components file not found for targeted: {alive_path_a}"

    alive_a = _load_alive_components(alive_path_a)
    logger.info(
        f"Targeted alive components: {sum(len(v) for v in alive_a.values())} across "
        f"{len(alive_a)} matrices"
    )

    module_lookup = build_module_lookup(model_a.target_module_paths)
    module_lookup_b = build_module_lookup(model_b.target_module_paths)
    assert module_lookup == module_lookup_b, (
        "Targeted and larger models have different module paths; cannot match components."
    )

    shared_keys = sorted(k for k in alive_a if k in module_lookup)
    assert shared_keys, "No (layer, matrix) keys from alive file match the model's modules."

    out_base = Path(output_dir).expanduser() if output_dir else run_dir_a
    out_base.mkdir(parents=True, exist_ok=True)
    suffix = run_dir_b.name
    real_tsv = out_base / f"compare_to_larger_{suffix}.tsv"
    rand_tsv = out_base / f"compare_to_larger_random_{suffix}.tsv"

    with torch.no_grad():
        real_results = _match_all_pool(
            model_a, model_b, module_lookup, alive_a, shared_keys, draw=0, chunk_size=chunk_size
        )
        for r in real_results:
            logger.info(
                f"  layer={r.layer} matrix={r.matrix}: {len(r.matches)} targeted comps matched"
            )

        per_matrix: dict[tuple[int, str], list[MaxCosMatch]] = {key: [] for key in shared_keys}
        for draw in range(n_random_samples):
            _reinit_components_(model_a, seed=random_seed + draw)
            draw_results = _match_all_pool(
                model_a,
                model_b,
                module_lookup,
                alive_a,
                shared_keys,
                draw=draw,
                chunk_size=chunk_size,
            )
            for r in draw_results:
                per_matrix[(r.layer, r.matrix)].extend(r.matches)
            logger.info(f"random draw {draw + 1}/{n_random_samples} done")
        random_results = [
            MatrixMatchResult(layer=layer, matrix=matrix, matches=per_matrix[(layer, matrix)])
            for (layer, matrix) in shared_keys
        ]

    _write_tsv(real_results, real_tsv)
    _write_tsv(random_results, rand_tsv)
    logger.info(f"Saved {real_tsv}")
    logger.info(f"Saved {rand_tsv}")
    return real_tsv, rand_tsv


if __name__ == "__main__":
    fire.Fire(compare_to_larger)
