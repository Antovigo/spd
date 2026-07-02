"""Collect Llama-3.1-8B's layer-18 residual stream at four points around the MLP, over `a+b=`.

Reproduces Feucht et al.'s (2026, "Arithmetic in the Wild") activation extraction: runs every
`{a}+{b}=` prompt for `a, b` in `1..max_value` (default 200, as they use) through the model and
stores, at the **last token** (the `=`), four residual-stream sites around `layer`'s MLP, captured
in one forward pass (see `RESID_SITES`):

- `resid_pre`     — input to `post_attention_layernorm`: resid after attention, *before* the MLP.
- `resid_norm`    — the MLP's actual input: `post_attention_layernorm` output (after RMSNorm).
- `resid_mlp_out` — the MLP output: the Δ the MLP writes into the residual stream.
- `resid_post`    — the **decoder-block output** (`out[0] = pre + mlp_out`): after attention *and*
  MLP (Feucht's `source="resid"`).

Batches are **left-padded**, so position `-1` is the `=` for every prompt.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.collect_resid_activations <model_path> \
        [--layer=18] [--max-value=200] [--batch-size=64] [--output=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Output (default `resid_activations.npz` under `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`):
`resid_<site>` grids `[G, G, d_model]` fp16 indexed `[a-1, b-1]` (`G = max_value`) for each site,
plus `a`, `b` axes, `layer`, `max_value`.
"""

from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import (
    RESID_SITES,
    SlurmOptions,
    load_lm_run,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.probes.collect_resid_activations"


def collect_resid_activations(
    model_path: ModelPath,
    layer: int = 18,
    max_value: int = 200,
    batch_size: int = 64,
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
            f"--layer={layer}",
            f"--max-value={max_value}",
            f"--batch-size={batch_size}",
        ]
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-resid-acts")
        return None

    run = load_lm_run(model_path)
    hf = run.model.target_model  # the bare, frozen Llama-3.1-8B
    tokenizer = run.tokenizer
    tokenizer.padding_side = "left"  # so position -1 is the `=` for every prompt length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    values = [str(i) for i in range(1, max_value + 1)]
    prompts = [f"{a}+{b}=" for a in values for b in values]
    ab = [(int(a), int(b)) for a in values for b in values]
    logger.info(f"{len(prompts)} prompts (a,b in 1..{max_value}), layer {layer}")

    layer_mod = hf.get_submodule(f"model.layers.{layer}")
    norm_mod = layer_mod.get_submodule("post_attention_layernorm")
    mlp_mod = layer_mod.get_submodule("mlp")
    captured: dict[str, torch.Tensor] = {}

    def last(t: Any) -> torch.Tensor:
        return (t[0] if isinstance(t, tuple) else t)[:, -1].float()  # last token (`=`), left-padded

    def hook_pre(_m: Any, args: Any) -> None:  # input to RMSNorm = resid after attn, before MLP
        captured["pre"] = last(args[0])

    def hook_norm(_m: Any, args: Any) -> None:  # input to MLP = RMSNorm output
        captured["norm"] = last(args[0])

    def hook_mlp_out(_m: Any, _i: Any, out: Any) -> None:  # MLP output = Δ added to resid
        captured["mlp_out"] = last(out)

    def hook_post(_m: Any, _i: Any, out: Any) -> None:  # block output = pre + mlp_out
        captured["post"] = last(out)

    handles = [
        norm_mod.register_forward_pre_hook(hook_pre),
        mlp_mod.register_forward_pre_hook(hook_norm),
        mlp_mod.register_forward_hook(hook_mlp_out),
        layer_mod.register_forward_hook(hook_post),
    ]

    grids: dict[str, np.ndarray] = {}  # allocated on the first batch, once d_model is known
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(run.device)
            captured.clear()
            hf(**inputs)
            for site in RESID_SITES:
                acts = captured[site].cpu().numpy().astype(np.float16)
                if site not in grids:
                    grids[site] = np.zeros((max_value, max_value, acts.shape[-1]), dtype=np.float16)
                for i, (a, b) in enumerate(ab[start : start + len(batch)]):
                    grids[site][a - 1, b - 1] = acts[i]
            if (start // batch_size) % 50 == 0:
                logger.info(f"batch {start // batch_size + 1}/{-(-len(prompts) // batch_size)}")
    for h in handles:
        h.remove()
    assert set(grids) == set(RESID_SITES), f"missing sites: {set(RESID_SITES).difference(grids)}"

    out_path = (
        Path(output).expanduser()
        if output
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_probes" / "resid_activations.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        a=np.arange(1, max_value + 1, dtype=np.int32),
        b=np.arange(1, max_value + 1, dtype=np.int32),
        layer=layer,
        max_value=max_value,
        resid_pre=grids["pre"],
        resid_norm=grids["norm"],
        resid_mlp_out=grids["mlp_out"],
        resid_post=grids["post"],
    )
    logger.info(
        f"wrote {', '.join(RESID_SITES)} ({out_path.stat().st_size / 1e6:.0f} MB) → {out_path}"
    )
    return out_path


if __name__ == "__main__":
    fire.Fire(collect_resid_activations)
