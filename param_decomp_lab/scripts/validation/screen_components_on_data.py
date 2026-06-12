"""In what natural-text contexts do the L18 addition components fire?

Streams sequences from the run's broad **nontarget** distribution (the data used to keep
these components inactive on general text — fineweb here) through the decomposed model,
computes lower-leaky causal importance for every L18 component, and records (a) how often
each component fires (CI > `--ci-thr`) on broad text and (b) its top-`--top-k`
max-activating contexts (the firing token plus a window of left context). This screens
which components are addition-specific (≈never fire on broad text) vs. generic (fire on
numbers / other contexts), and surfaces the non-addition situations that drive the
addition module.

Components alive on the addition task are flagged in the summary via `--alive-tsv`
(default `alive_components.tsv` in the run folder).

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.screen_components_on_data <model_path> \
        [--n-batches=600] [--batch-size=128] [--ci-thr=0.1] [--top-k=30] \
        [--context-window=24] [--seed=0] [--output-tsv=PATH] [--output=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=0:30:00 --slurm-mem=...]]

Outputs (default in the run folder):
- `screen_components_on_data.tsv` — one row per component that ever fires: matrix,
  component, alive_on_addition, count_active, frac_active, max_ci.
- `screen_components_on_data.jsonl` — one object per component with its top-k contexts.
"""

import bisect
import csv
import json
from pathlib import Path
from typing import Any

import fire
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.run import build_lm_loader
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    parse_module_name,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.screen_components_on_data"
_TSV_FIELDS = ["matrix", "component", "alive_on_addition", "count_active", "frac_active", "max_ci"]


def _load_alive(alive_path: Path) -> set[tuple[str, int]]:
    if not alive_path.exists():
        return set()
    out: set[tuple[str, int]] = set()
    for row in csv.DictReader(alive_path.open(), delimiter="\t"):
        out.add((row["matrix"], int(row["component"])))
    return out


def screen_components_on_data(
    model_path: ModelPath,
    n_batches: int = 600,
    batch_size: int = 128,
    ci_thr: float = 0.1,
    top_k: int = 30,
    context_window: int = 24,
    seed: int = 0,
    alive_tsv: str | None = None,
    output_tsv: str | None = None,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "0:30:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--n-batches={n_batches}",
            f"--batch-size={batch_size}",
            f"--ci-thr={ci_thr}",
            f"--top-k={top_k}",
            f"--context-window={context_window}",
            f"--seed={seed}",
        ]
        if alive_tsv is not None:
            argv.append(f"--alive-tsv={Path(alive_tsv).expanduser()}")
        if output_tsv is not None:
            argv.append(f"--output-tsv={Path(output_tsv).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-screen-components")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    assert cfg.nontarget is not None, "run has no nontarget config to screen"

    tsv_path = (
        Path(output_tsv).expanduser()
        if output_tsv
        else run.run_dir / "screen_components_on_data.tsv"
    )
    jsonl_path = (
        Path(output).expanduser() if output else run.run_dir / "screen_components_on_data.jsonl"
    )

    alive_path = Path(alive_tsv).expanduser() if alive_tsv else run.run_dir / "alive_components.tsv"
    alive = _load_alive(alive_path)

    loader = build_lm_loader(
        cfg.target,
        cfg.nontarget.data,
        split="train",
        device=str(device),
        batch_size=batch_size,
        seed=seed,
    )

    # Per-module running top-k (by CI) with the global flat position id of each survivor,
    # plus firing counts. Modules are discovered from the first batch's CI keys.
    topk_vals: dict[str, Tensor] = {}
    topk_pos: dict[str, Tensor] = {}
    count_active: dict[str, Tensor] = {}
    all_ids: list[Tensor] = []  # CPU input_ids per batch, for context reconstruction
    batch_end: list[int] = []  # cumulative flat-position offset after each batch
    offset = 0
    count_total = 0

    def context_for(flat_pos: int, seq_len: int) -> dict[str, Any]:
        bi = bisect.bisect_right(batch_end, flat_pos)
        within = flat_pos - (batch_end[bi - 1] if bi > 0 else 0)
        row, col = within // seq_len, within % seq_len
        ids = all_ids[bi][row]
        left = ids[max(0, col - context_window) : col].tolist()
        return {
            "pos": int(col),
            "token": repr(tokenizer.decode([int(ids[col])])),
            "left_context": tokenizer.decode(left),
        }

    def flush(seen: int) -> int:
        """Write the summary TSV + per-component contexts from the running state. Returns
        the number of components that ever fired. Safe to call mid-stream (checkpointing)."""
        seq_len = all_ids[0].shape[1]
        rows: list[dict[str, Any]] = []
        jsonl: list[dict[str, Any]] = []
        for module in topk_vals:
            _, matrix = parse_module_name(module)
            ca = count_active[module].tolist()
            mx = topk_vals[module][:, 0].tolist()
            for comp in range(len(ca)):
                if ca[comp] == 0:
                    continue
                rows.append(
                    {
                        "matrix": matrix,
                        "component": comp,
                        "alive_on_addition": int((matrix, comp) in alive),
                        "count_active": ca[comp],
                        "frac_active": ca[comp] / seen,
                        "max_ci": mx[comp],
                    }
                )
                ctxs = []
                for k in range(top_k):
                    v = topk_vals[module][comp, k].item()
                    p = int(topk_pos[module][comp, k])
                    if p < 0 or v <= ci_thr:
                        continue
                    ctxs.append({"ci": round(v, 3), **context_for(p, seq_len)})
                jsonl.append(
                    {
                        "matrix": matrix,
                        "component": comp,
                        "alive_on_addition": (matrix, comp) in alive,
                        "contexts": ctxs,
                    }
                )
        rows.sort(key=lambda r: -r["frac_active"])
        with tsv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        with jsonl_path.open("w") as f:
            for obj in jsonl:
                f.write(json.dumps(obj) + "\n")
        return len(rows)

    it = iter(loader)
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for b in range(n_batches):
            chunk = next(it, None)
            if chunk is None:
                logger.warning(f"stream exhausted after {b} batches")
                break
            chunk = chunk.to(device)
            bsz, T = chunk.shape
            n = bsz * T

            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")

            posids = torch.arange(offset, offset + n, device=device)
            for module, ci_ll in ci.lower_leaky.items():
                c = ci_ll.shape[-1]
                vals = ci_ll.reshape(n, c).T.float()  # [C, n]
                if module not in topk_vals:
                    topk_vals[module] = torch.full((c, top_k), -1.0, device=device)
                    topk_pos[module] = torch.full((c, top_k), -1, dtype=torch.long, device=device)
                    count_active[module] = torch.zeros(c, dtype=torch.long, device=device)
                count_active[module] += (vals > ci_thr).sum(dim=1)
                cat_v = torch.cat([topk_vals[module], vals], dim=1)  # [C, K+n]
                cat_p = torch.cat([topk_pos[module], posids.expand(c, n)], dim=1)
                tv, ti = cat_v.topk(top_k, dim=1)
                topk_vals[module] = tv
                topk_pos[module] = cat_p.gather(1, ti)

            all_ids.append(chunk.cpu())
            offset += n
            batch_end.append(offset)
            count_total += n

            if (b + 1) % 50 == 0:
                n_fired = flush(count_total)
                logger.info(
                    f"batch {b + 1}/{n_batches}: {count_total} positions, {n_fired} fired (checkpointed)"
                )

    n_fired = flush(count_total)
    logger.info(
        f"Screened {count_total} positions; {n_fired} components fired → {tsv_path}, {jsonl_path}"
    )
    return tsv_path


if __name__ == "__main__":
    fire.Fire(screen_components_on_data)
