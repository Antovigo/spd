"""Find components that are active (CI > threshold) at least once over the chosen data.

Usage:
    python -m spd.scripts.validation.find_alive_components <model_path> \
        [--ci-thr=0.01] [--n-batches=1] [--nontarget] [--output=PATH]
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import fire
import torch
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.scripts.validation.common import (
    build_lm_loader,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    parse_module_name,
    resolve_task_config,
)
from spd.spd_types import ModelPath


@dataclass
class PerComponentStats:
    count_active: Tensor  # int64 [C]
    count_total: int
    max_ci: Tensor  # float [C]
    activation_sum: Tensor  # float [C]
    activation_count: Tensor  # int64 [C]


def _init_stats(C: int, device: torch.device) -> PerComponentStats:
    return PerComponentStats(
        count_active=torch.zeros(C, dtype=torch.int64, device=device),
        count_total=0,
        max_ci=torch.zeros(C, device=device),
        activation_sum=torch.zeros(C, device=device),
        activation_count=torch.zeros(C, dtype=torch.int64, device=device),
    )


@torch.no_grad()
def _update_stats(
    stats: PerComponentStats,
    ci: Tensor,  # [..., C]
    activations: Tensor,  # [..., C]
    ci_thr: float,
) -> None:
    flat_ci = ci.reshape(-1, ci.shape[-1])
    flat_acts = activations.reshape(-1, activations.shape[-1])

    active = flat_ci > ci_thr  # [N, C]
    stats.count_active += active.sum(dim=0).to(torch.int64)
    stats.count_total += flat_ci.shape[0]
    stats.max_ci = torch.maximum(stats.max_ci, flat_ci.amax(dim=0))
    stats.activation_sum += (flat_acts * active).sum(dim=0)
    stats.activation_count += active.sum(dim=0).to(torch.int64)


def find_alive_components(
    model_path: ModelPath,
    ci_thr: float = 0.01,
    n_batches: int = 1,
    nontarget: bool = False,
    output: str | None = None,
) -> Path:
    """Run the decomposed model and collect components that ever reach CI > ci_thr."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = resolve_task_config(config, use_nontarget=nontarget)
    loader = build_lm_loader(task_config, config)
    single_batch = is_prompt_task(task_config)

    stats: dict[str, PerComponentStats] = {
        name: _init_stats(C, device) for name, C in spd_model.module_to_c.items()
    }

    n_to_run = 1 if single_batch else n_batches
    iterator = iterate_input_ids(loader, device)

    with torch.no_grad():
        for _ in tqdm(range(n_to_run), desc="batches"):
            batch = next(iterator)
            output_with_cache = spd_model(batch, cache_type="input")
            ci_outputs = spd_model.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                sampling=config.sampling,
            )
            component_acts = spd_model.get_all_component_acts(output_with_cache.cache)

            for module_name in stats:
                _update_stats(
                    stats=stats[module_name],
                    ci=ci_outputs.lower_leaky[module_name],
                    activations=component_acts[module_name],
                    ci_thr=ci_thr,
                )

    rows: list[dict[str, object]] = []
    for module_name, s in stats.items():
        layer, matrix = parse_module_name(module_name)
        count_active = s.count_active.cpu()
        count_total = s.count_total
        max_ci = s.max_ci.cpu()
        activation_sum = s.activation_sum.cpu()
        activation_count = s.activation_count.cpu()
        C = count_active.shape[0]
        for c in range(C):
            if count_active[c].item() == 0:
                continue
            n_active = int(activation_count[c].item())
            mean_activation = (
                float(activation_sum[c].item() / n_active) if n_active > 0 else float("nan")
            )
            rows.append(
                {
                    "layer": layer,
                    "matrix": matrix,
                    "component": c,
                    "fraction_active": float(count_active[c].item()) / count_total,
                    "max_ci": float(max_ci[c].item()),
                    "mean_activation": mean_activation,
                }
            )

    rows.sort(key=lambda r: (r["layer"], r["matrix"], r["component"]))

    default_name = "alive_components_nontarget.tsv" if nontarget else "alive_components.tsv"
    out_path = Path(output).expanduser() if output else run_dir / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["layer", "matrix", "component", "fraction_active", "max_ci", "mean_activation"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} alive components to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(find_alive_components)
