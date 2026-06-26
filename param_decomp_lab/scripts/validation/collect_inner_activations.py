"""Per-subcomponent normalized inner activations over an operation's 100x100 grid.

For every alive L18 MLP subcomponent and every `a<op>b=` prompt, computes the inner
activation at the **last token**: the dot product of the component's input with its V
vector normalized to unit norm, `(x · V_c) / ||V_c||`. The component input is taken
directly per matrix — post-RMSNorm MLP input for gate/up, post-SwiGLU neuron activations
for down — which is exactly what the cached pre-weight acts hold, so no manual RMSNorm /
nonlinearity is reapplied.

"Alive" is the intersection of the `find_alive_components` set (ever causally important on
the run's original data — its default unsuffixed `alive_components.tsv`) with a **mean-CI**
filter applied *here* for this operation: a component is kept only if its mean lower-leaky CI
at the last token over this op's whole grid exceeds `--mean-ci-thr` (default 0.1). The
surviving set is written to `alive_filtered_<op>.tsv` and consumed by the period / cosine /
explorer scripts.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.collect_inner_activations <model_path> \
        [--op=add] [--mean-ci-thr=0.1] [--alive-tsv=PATH] [--batch-size=256] \
        [--output=PATH] [--output-alive=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Outputs (default in the run's `analysis/datasets/`):
- `inner_activations_<op>.tsv` — one row per (filtered-alive component, prompt): columns
  `a, operation, b, matrix, subcomponent, inner_act`.
- `alive_filtered_<op>.tsv` — the surviving alive set: `layer, matrix, component, mean_ci`.
"""

import csv
from pathlib import Path
from typing import Any, cast

import fire
import numpy as np
import torch

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    SlurmOptions,
    analysis_datasets_dir,
    load_lm_run,
    op_prompts_file,
    op_symbol,
    parse_operands,
    read_alive_components,
    square_grid_size,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.collect_inner_activations"
_TSV_FIELDS = ["a", "operation", "b", "matrix", "subcomponent", "inner_act"]
_ALIVE_FIELDS = ["layer", "matrix", "component", "mean_ci"]


def collect_inner_activations(
    model_path: ModelPath,
    op: str = "add",
    mean_ci_thr: float = 0.1,
    alive_tsv: str | None = None,
    batch_size: int = 256,
    output: str | None = None,
    output_alive: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--op={op}",
            f"--mean-ci-thr={mean_ci_thr}",
            f"--batch-size={batch_size}",
        ]
        if alive_tsv is not None:
            argv.append(f"--alive-tsv={Path(alive_tsv).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        if output_alive is not None:
            argv.append(f"--output-alive={Path(output_alive).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-inner-acts-{op}")
        return None

    run = load_lm_run(model_path)
    model, cfg, device = run.model, run.cfg, run.device

    data_dir = analysis_datasets_dir(run.run_dir)
    alive_path = Path(alive_tsv).expanduser() if alive_tsv else data_dir / "alive_components.tsv"
    alive = read_alive_components(alive_path, keep_projs=MLP_MATRICES)
    logger.info(f"{len(alive)} existing-alive MLP components from {alive_path.name}")

    prompts_file = op_prompts_file(op)
    prompt_texts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
    ab = [parse_operands(t, op) for t in prompt_texts]
    n = square_grid_size(ab)
    pool = load_prompts_dataset(str(prompts_file), cast(Any, run.tokenizer)).to(device)
    assert pool.shape[0] == len(prompt_texts)

    # Cache EVERY decomposed module: the CI fn is a shared transformer over all matrices, so
    # it needs every module's input. The alive subset only decides which rows we keep.
    modules = sorted({a.module for a in alive})
    v_unit = {
        m: model.components[m].V.detach()
        / model.components[m].V.detach().norm(dim=0).clamp_min(1e-12)
        for m in modules
    }

    # Per-alive-component (a, b) grids of normalized inner act and last-token CI.
    inner = np.zeros((len(alive), n, n), dtype=np.float32)
    ci_grid = np.zeros((len(alive), n, n), dtype=np.float32)
    comp_index = {m: [i for i, a in enumerate(alive) if a.module == m] for m in modules}

    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            rows = [a - 1 for a, _ in ab[start : start + chunk.shape[0]]]
            cols = [b - 1 for _, b in ab[start : start + chunk.shape[0]]]

            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            for m in modules:
                # .float() the einsum result: autocast can return bf16 even for float inputs.
                inner_last = (
                    torch.einsum("bd,dc->bc", cached.cache[m][:, -1].float(), v_unit[m].float())
                    .float()
                    .cpu()
                    .numpy()
                )  # (x · V_c)/||V_c|| at last token, [b, C]
                ci_last = ci.lower_leaky[m][:, -1].float().cpu().numpy()  # [b, C]
                for i in comp_index[m]:
                    c = alive[i].component
                    inner[i, rows, cols] = inner_last[:, c]
                    ci_grid[i, rows, cols] = ci_last[:, c]
            logger.info(
                f"batch {start // batch_size + 1}: prompts {start}..{start + chunk.shape[0]}"
            )

    mean_ci = ci_grid.reshape(len(alive), -1).mean(axis=1)
    keep = mean_ci > mean_ci_thr
    logger.info(f"{int(keep.sum())}/{len(alive)} pass mean-CI > {mean_ci_thr}")

    out_path = Path(output).expanduser() if output else data_dir / f"inner_activations_{op}.tsv"
    alive_out = (
        Path(output_alive).expanduser() if output_alive else data_dir / f"alive_filtered_{op}.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alive_out.parent.mkdir(parents=True, exist_ok=True)

    sym = op_symbol(op)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
        writer.writeheader()
        for i in np.nonzero(keep)[0]:
            comp = alive[i]
            grid = inner[i]
            for a in range(n):
                for b in range(n):
                    writer.writerow(
                        {
                            "a": a + 1,
                            "operation": sym,
                            "b": b + 1,
                            "matrix": comp.proj,
                            "subcomponent": comp.component,
                            "inner_act": round(float(grid[a, b]), 6),
                        }
                    )

    with alive_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALIVE_FIELDS, delimiter="\t")
        writer.writeheader()
        for i in np.nonzero(keep)[0]:
            comp = alive[i]
            writer.writerow(
                {
                    "layer": comp.layer,
                    "matrix": comp.matrix,
                    "component": comp.component,
                    "mean_ci": round(float(mean_ci[i]), 6),
                }
            )

    logger.info(f"wrote inner activations → {out_path} and filtered-alive set → {alive_out}")
    return out_path


if __name__ == "__main__":
    fire.Fire(collect_inner_activations)
