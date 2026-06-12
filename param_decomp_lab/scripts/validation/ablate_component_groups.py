"""What does each *family* of pos-`=` components do to the predicted sum?

Runs a sample of the run's `X+Y=` target prompts through the circuit (CI > `--ci-thr`,
delta off) and, for each named component group, re-runs with exactly that group forced
off. The answer is a single token, so we read the argmax at the `=` position and compare
the predicted integer to `X+Y`. Contrasting which groups break the *ones digit* vs the
*magnitude* of the answer turns the geometric grid story (see `lab_notebook.md`,
`figures/pos4_grid*.png`) into a causal one.

Groups are defined in `_GROUPS` below, keyed by the families found in the grid analysis.

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.ablate_component_groups <model_path> \
        [--n-examples=1024] [--ci-thr=0.1] [--batch-size=128] [--seed=0] [--output=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Output (default `<run_dir>/ablate_component_groups.tsv`): one row per (prompt, condition).
Columns: example, x, y, correct, condition, pred (decoded argmax token, repr'd),
pred_int (parsed int or empty), correct_flag.
"""

import csv
import re
from pathlib import Path
from typing import Any, cast

import fire
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.masks import make_mask_infos
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.ablate_component_groups"

# Component families found in the (X,Y)-grid analysis at the `=` position. Each entry maps
# a short matrix name to the component indices to force off. See lab_notebook.md Finding 3.
_GROUPS: dict[str, dict[str, list[int]]] = {
    "units_lattice_gate": {"gate_proj": [389, 248, 460, 228, 34, 65, 485, 174]},
    "sum_band_down": {"down_proj": [230, 84, 404, 488, 314, 334, 240, 56, 123, 182, 351, 463]},
    "always_on_down": {"down_proj": [118, 467]},
    "operand_mag_gate": {"gate_proj": [429, 163, 276]},
    "operand_mag_up": {"up_proj": [323, 400, 225, 17]},
}


def _parse_xy(prompt: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\+(\d+)=", prompt)
    assert m is not None, f"prompt is not 'X+Y=': {prompt!r}"
    return int(m.group(1)), int(m.group(2))


def _ablated_masks(base: dict[str, Tensor], group: dict[str, list[int]]) -> dict[str, Tensor]:
    """Clone the circuit mask and zero the group's components (all positions)."""
    out = {name: m.clone() for name, m in base.items()}
    for short_matrix, comps in group.items():
        full = f"model.layers.18.mlp.{short_matrix}"
        assert full in out, f"{full} not among decomposed modules {list(out)}"
        out[full][..., comps] = 0.0
    return out


def ablate_component_groups(
    model_path: ModelPath,
    n_examples: int = 1024,
    ci_thr: float = 0.1,
    batch_size: int = 128,
    seed: int = 0,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--n-examples={n_examples}",
            f"--ci-thr={ci_thr}",
            f"--batch-size={batch_size}",
            f"--seed={seed}",
        ]
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-ablate-component-groups")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer

    assert cfg.data.prompts_file is not None, "requires prompts-based target data"
    prompts_file = Path(cfg.data.prompts_file).expanduser()
    prompt_texts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
    pool = load_prompts_dataset(str(prompts_file), cast(Any, tokenizer), cfg.data.max_seq_len)

    # Deterministic subsample of prompts for speed.
    n = min(n_examples, pool.shape[0])
    perm = torch.randperm(pool.shape[0], generator=torch.Generator().manual_seed(seed))[:n]
    pool = pool[perm].to(device)
    sampled_texts = [prompt_texts[i] for i in perm.tolist()]

    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    assert isinstance(pad_id, int)
    # The `=` is the last real token of every prompt; predict from there.
    eq_pos = (pool != pad_id).sum(dim=1) - 1  # [n]

    conditions = ["baseline", *(_GROUPS.keys())]
    rows: list[dict[str, Any]] = []
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            eqp = eq_pos[start : start + batch_size]
            rng = torch.arange(chunk.shape[0], device=device)

            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            base_masks = {name: (v > ci_thr).to(v.dtype) for name, v in ci.lower_leaky.items()}

            for cond in conditions:
                masks = (
                    base_masks if cond == "baseline" else _ablated_masks(base_masks, _GROUPS[cond])
                )
                logits = model(chunk, mask_infos=make_mask_infos(masks))
                eq_logits = logits[rng, eqp].float()  # [b, vocab] at the `=` position
                pred_ids = eq_logits.argmax(dim=-1).tolist()
                for i, pid in enumerate(pred_ids):
                    text = sampled_texts[start + i]
                    x, y = _parse_xy(text)
                    pred = tokenizer.decode([pid])
                    pred_int = int(pred) if pred.strip().lstrip("-").isdigit() else None
                    rows.append(
                        {
                            "example": start + i,
                            "x": x,
                            "y": y,
                            "correct": x + y,
                            "condition": cond,
                            "pred": repr(pred),
                            "pred_int": "" if pred_int is None else pred_int,
                            "correct_flag": int(pred_int == x + y),
                        }
                    )

    out_path = Path(output).expanduser() if output else run.run_dir / "ablate_component_groups.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"Wrote {len(rows)} rows ({len(conditions)} conditions × {n} prompts) to {out_path}"
    )
    return out_path


if __name__ == "__main__":
    fire.Fire(ablate_component_groups)
