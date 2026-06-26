"""Compare original vs. circuit next-token predictions on a sample of target data.

Draws a random sample of `--n-examples` sequences from the run's target distribution and,
for each one, runs two forward passes:

- the original target model, and
- the "circuit": for every decomposed matrix, only the subcomponents whose lower-leaky
  causal importance exceeds `--ci-thr` are enabled (binary mask), and the delta component
  is disabled.

For every real (non-pad) position of every sampled sequence it reports each model's top-n
next-token predictions, one token per `rank_<r>` column. Each sequence yields
`2 * n_positions` rows: one per `(position, model)`.

An 8B forward pass needs a GPU; pass `--slurm` to submit this invocation as a single-GPU
SLURM job instead of running it on the (GPU-less) login node.

Usage:
    python -m param_decomp_lab.scripts.validation.sample_target_data <model_path> \
        [--n-examples=50] [--ci-thr=0.1] [--top-n=5] [--batch-size=8] [--seed=0] \
        [--output=PATH] [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Output (default `<run_dir>/analysis/datasets/sample_target_data.tsv`): one row per
(sequence, position, model).
"""

import csv
from pathlib import Path
from typing import Any

import fire
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.masks import make_mask_infos
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.run import build_lm_loader
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    analysis_datasets_dir,
    load_lm_run,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.sample_target_data"


def _cell(prob: float, token_id: int, tokenizer: Any) -> str:
    """`'<tok>' (0.834)` — the decoded token (repr, so whitespace is visible) and its prob."""
    return f"{tokenizer.decode([token_id])!r} ({prob:.3f})"


def _token_repr(tokenizer: Any, token_id: int) -> str:
    """The input token at a position, decoded and repr'd so whitespace is visible."""
    return repr(tokenizer.decode([token_id]))


def _n_real_positions(sample: Tensor, pad_token_id: int | None) -> Tensor:
    """Per-row count of real (non-pad) positions. Prompts are right-padded; packed dataset
    rows have no padding, so every position is real."""
    if pad_token_id is None:
        return torch.full((sample.shape[0],), sample.shape[1], device=sample.device)
    lengths = (sample != pad_token_id).sum(dim=1)
    assert (lengths > 0).all(), "a sampled sequence is entirely padding"
    return lengths


def sample_target_data(
    model_path: ModelPath,
    n_examples: int = 50,
    ci_thr: float = 0.1,
    top_n: int = 5,
    batch_size: int = 8,
    seed: int = 0,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Write a long-format TSV of original vs. circuit top-n next-token predictions.

    Returns the output path, or `None` when `--slurm` submits the job instead of running.
    """
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--n-examples={n_examples}",
            f"--ci-thr={ci_thr}",
            f"--top-n={top_n}",
            f"--batch-size={batch_size}",
            f"--seed={seed}",
        ]
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-sample-target-data")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer

    loader = build_lm_loader(
        cfg.target, cfg.data, split="train", device=str(device), batch_size=n_examples, seed=seed
    )
    sample: Tensor = next(iter(loader)).to(device)
    if sample.shape[0] < n_examples:
        logger.warning(f"Only {sample.shape[0]} examples available (asked for {n_examples})")

    if cfg.data.prompts_file is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        assert isinstance(pad_token_id, int), "tokenizer has neither pad_token_id nor eos_token_id"
    else:
        pad_token_id = None
    n_positions = _n_real_positions(sample, pad_token_id)

    rank_fields = [f"rank_{r + 1}" for r in range(top_n)]
    rows: list[dict[str, Any]] = []
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, sample.shape[0], batch_size):
            chunk = sample[start : start + batch_size]

            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            component_masks = {
                name: (ci_vals > ci_thr).to(ci_vals.dtype)
                for name, ci_vals in ci.lower_leaky.items()
            }
            circuit_logits = model(chunk, mask_infos=make_mask_infos(component_masks))

            # Top-n tokens at every position for both models: values/indices are [b, T, top_n].
            orig_top = cached.output.float().softmax(dim=-1).topk(top_n, dim=-1)
            circ_top = circuit_logits.float().softmax(dim=-1).topk(top_n, dim=-1)

            for i in range(chunk.shape[0]):
                for pos in range(int(n_positions[start + i])):
                    token = _token_repr(tokenizer, int(chunk[i, pos]))
                    for label, top in (("original", orig_top), ("ablated", circ_top)):
                        row: dict[str, Any] = {
                            "example": start + i,
                            "pos": pos,
                            "token": token,
                            "model": label,
                        }
                        for r, field in enumerate(rank_fields):
                            row[field] = _cell(
                                top.values[i, pos, r].item(),
                                int(top.indices[i, pos, r]),
                                tokenizer,
                            )
                        rows.append(row)

    out_path = (
        Path(output).expanduser()
        if output
        else analysis_datasets_dir(run.run_dir) / "sample_target_data.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(sample_target_data)
