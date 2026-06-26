"""Find subcomponents that are active on at least one target prompt.

Runs every prompt in the run's target distribution through the decomposed model and records
which subcomponents reach lower-leaky causal importance > `--ci-thr` on at least one
(prompt, position). Only real (non-pad) positions count.

An 8B forward pass needs a GPU; pass `--slurm` to submit this invocation as a single-GPU
SLURM job instead of running it on the (GPU-less) login node.

Usage:
    python -m param_decomp_lab.scripts.validation.find_alive_components <model_path> \
        [--ci-thr=0.1] [--batch-size=8] [--output=PATH] [--output-json=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Outputs (default in the run's `analysis/datasets/`):
- `alive_components.tsv` — one row per alive subcomponent: layer, matrix, component,
  fraction_active, max_ci, mean_activation.
- `alive_components_per_position.json` — per (prompt, position), the active components with
  their CI at that position, organised as prompt > position > matrix (full module path) >
  [{component, ci}]. Consumed by `plot_ci_heatmaps.py`.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import fire
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.component_model_io import get_all_component_acts
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    analysis_datasets_dir,
    load_lm_run,
    parse_module_name,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.find_alive_components"
_TSV_FIELDS = ["layer", "matrix", "component", "fraction_active", "max_ci", "mean_activation"]


@dataclass
class _Stats:
    count_active: Tensor  # int64 [C]: positions where CI > ci_thr
    max_ci: Tensor  # float [C]
    activation_sum: Tensor  # float [C]: sum of V^T x over active positions


def _init_stats(c: int, device: torch.device) -> _Stats:
    return _Stats(
        count_active=torch.zeros(c, dtype=torch.int64, device=device),
        max_ci=torch.zeros(c, device=device),
        activation_sum=torch.zeros(c, device=device),
    )


def find_alive_components(
    model_path: ModelPath,
    ci_thr: float = 0.1,
    batch_size: int = 8,
    prompts: str | None = None,
    output: str | None = None,
    output_json: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Write the alive-components TSV + per-(prompt, position) active-components JSON.

    Returns the TSV path, or `None` when `--slurm` submits the job instead of running.
    """
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--ci-thr={ci_thr}",
            f"--batch-size={batch_size}",
        ]
        if prompts is not None:
            argv.append(f"--prompts={Path(prompts).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        if output_json is not None:
            argv.append(f"--output-json={Path(output_json).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-find-alive-components")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer

    prompts_path = prompts if prompts is not None else cfg.data.prompts_file
    assert prompts_path is not None, (
        "find_alive_components requires prompts-based target data (or pass --prompts)"
    )
    prompts_file = Path(prompts_path).expanduser()
    prompt_texts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
    assert len(set(prompt_texts)) == len(prompt_texts), (
        "duplicate prompts in the prompts file would collide as JSON keys"
    )
    pool = load_prompts_dataset(str(prompts_file), cast(Any, tokenizer))
    pool = pool.to(device)
    assert pool.shape[0] == len(prompt_texts)

    # Prompts are constant-length and unpadded (load_prompts_dataset enforces this), so every
    # position is a real token — no pad-position masking.
    stats: dict[str, _Stats] = {}
    count_total = 0
    # prompt -> position -> module -> [{component, ci}], for components active at that position.
    per_position: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}

    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            count_total += chunk.numel()

            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            component_acts = get_all_component_acts(model, cached.cache)

            for module, ci_ll in ci.lower_leaky.items():
                c = ci_ll.shape[-1]
                stats.setdefault(module, _init_stats(c, device))
                flat_ci = ci_ll.reshape(-1, c)  # [b*T, C]
                flat_acts = component_acts[module].reshape(-1, c).float()
                active = flat_ci > ci_thr
                stats[module].count_active += active.sum(dim=0).to(torch.int64)
                stats[module].max_ci = torch.maximum(stats[module].max_ci, flat_ci.amax(dim=0))
                stats[module].activation_sum += (flat_acts * active).sum(dim=0)

            # Per-position JSON: move CI to CPU/numpy once per chunk to avoid per-element syncs.
            ci_np = {
                module: ci_ll.float().cpu().numpy() for module, ci_ll in ci.lower_leaky.items()
            }
            for i in range(chunk.shape[0]):
                pos_entry: dict[str, dict[str, list[dict[str, Any]]]] = {}
                for pos in range(chunk.shape[1]):
                    per_module: dict[str, list[dict[str, Any]]] = {}
                    for module, arr in ci_np.items():
                        ci_vec = arr[i, pos]  # [C]
                        active_idx = (ci_vec > ci_thr).nonzero()[0]
                        if active_idx.size > 0:
                            per_module[module] = sorted(
                                (
                                    {"component": int(comp), "ci": round(float(ci_vec[comp]), 3)}
                                    for comp in active_idx
                                ),
                                key=lambda d: d["ci"],
                                reverse=True,
                            )
                    pos_entry[str(pos)] = per_module
                per_position[prompt_texts[start + i]] = pos_entry

    rows: list[dict[str, Any]] = []
    for module, s in stats.items():
        layer, matrix = parse_module_name(module)
        count_active = s.count_active.tolist()
        max_ci = s.max_ci.tolist()
        activation_sum = s.activation_sum.tolist()
        for component, n_active in enumerate(count_active):
            if n_active == 0:
                continue
            rows.append(
                {
                    "layer": layer,
                    "matrix": matrix,
                    "component": component,
                    "fraction_active": n_active / count_total,
                    "max_ci": max_ci[component],
                    "mean_activation": activation_sum[component] / n_active,
                }
            )
    rows.sort(key=lambda r: (r["layer"], r["matrix"], r["component"]))

    data_dir = analysis_datasets_dir(run.run_dir)
    out_path = Path(output).expanduser() if output else data_dir / "alive_components.tsv"
    json_path = (
        Path(output_json).expanduser()
        if output_json
        else data_dir / "alive_components_per_position.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(per_position, separators=(",", ":")))

    logger.info(
        f"{len(rows)} alive components over {pool.shape[0]} prompts → {out_path} and {json_path}"
    )
    return out_path


if __name__ == "__main__":
    fire.Fire(find_alive_components)
