"""Collect the residual stream around layers 15-20 at the `=` token, one grid per position.

Generalises `collect_resid_activations` (four sites around L18's MLP) to the stream itself:
one forward pass per batch captures, at the last token, every requested stream *position*
for the full `a<op>b=` grid (`a, b` in `1..max_value`), one output file per op:

- `L<i>` — block `i`'s output (`out[0]`), the stream after block `i` = input to block `i+1`;
- `L<i>att` (all but the first layer) — the stream after block `i`'s **attention**, before
  its MLP (forward-pre input of `post_attention_layernorm`).

So the default `--layers=14..20` yields positions `L14, L15att, L15, ..., L20att, L20`, and
consecutive deltas isolate each attention write (`into L<i>att`) and each MLP write
(`into L<i>`). Batches are left-padded, so position `-1` is the `=` for every prompt.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.collect_resid_stream <model_path> \
        [--ops=add,sub] [--layers=14,15,16,17,18,19,20] [--max-value=200] [--batch-size=64] \
        [--output-dir=DIR] [--slurm [--partition=... --gpus=1 --slurm-time=3:00:00]]

Output (default dir `<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`): one
`resid_stream_<op>.npz` per op with `resid_<pos>` grids `[G, G, d_model]` fp16 indexed
`[a-1, b-1]` (`G = max_value`), plus `a`, `b`, `positions` (strings, stream order),
`layers`, `max_value`, `op`.
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
    SlurmOptions,
    load_lm_run,
    op_symbol,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.probes.collect_resid_stream"
_DEFAULT_LAYERS = (14, 15, 16, 17, 18, 19, 20)


def collect_resid_stream(
    model_path: ModelPath,
    ops: tuple[str, ...] = ("add", "sub"),
    layers: tuple[int, ...] = _DEFAULT_LAYERS,
    max_value: int = 200,
    batch_size: int = 64,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "3:00:00",
    slurm_mem: str | None = None,
    dependency: str | None = None,
) -> list[Path] | None:
    ops = _as_tuple(ops)
    layers = _as_tuple(layers)
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--ops={','.join(ops)}",
            f"--layers={','.join(str(i) for i in layers)}",
            f"--max-value={max_value}",
            f"--batch-size={batch_size}",
        ]
        if output_dir is not None:
            argv.append(f"--output-dir={Path(output_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition,
            gpus=gpus,
            slurm_time=slurm_time,
            slurm_mem=slurm_mem,
            dependency=dependency,
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-resid-stream")
        return None

    run = load_lm_run(model_path)
    hf = run.model.target_model  # the bare, frozen Llama-3.1-8B
    tokenizer = run.tokenizer
    tokenizer.padding_side = "left"  # so position -1 is the `=` for every prompt length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    positions = [f"L{layers[0]}"]
    for i in layers[1:]:
        positions += [f"L{i}att", f"L{i}"]
    captured: dict[str, torch.Tensor] = {}

    def make_block_hook(pos: str) -> Any:
        def hook(_m: Any, _i: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out  # tuple or bare tensor by version
            captured[pos] = hidden[:, -1].float()  # block output at the `=` (left-padded)

        return hook

    def make_att_hook(pos: str) -> Any:
        def hook(_m: Any, args: Any) -> None:  # input to post-attn RMSNorm = resid after attention
            captured[pos] = args[0][:, -1].float()

        return hook

    handles = [
        hf.get_submodule(f"model.layers.{i}").register_forward_hook(make_block_hook(f"L{i}"))
        for i in layers
    ] + [
        hf.get_submodule(f"model.layers.{i}.post_attention_layernorm").register_forward_pre_hook(
            make_att_hook(f"L{i}att")
        )
        for i in layers[1:]
    ]

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_probes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    values = [str(i) for i in range(1, max_value + 1)]
    ab = [(int(a), int(b)) for a in values for b in values]

    for op in ops:
        sym = op_symbol(op)
        prompts = [f"{a}{sym}{b}=" for a in values for b in values]
        logger.info(
            f"op {op!r}: {len(prompts)} prompts (a,b in 1..{max_value}), positions {positions}"
        )
        grids: dict[str, np.ndarray] = {}  # allocated on the first batch, once d_model is known
        with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True).to(run.device)
                captured.clear()
                hf(**inputs)
                for pos in positions:
                    acts = captured[pos].cpu().numpy().astype(np.float16)
                    if pos not in grids:
                        grids[pos] = np.zeros(
                            (max_value, max_value, acts.shape[-1]), dtype=np.float16
                        )
                    for i, (a, b) in enumerate(ab[start : start + len(batch)]):
                        grids[pos][a - 1, b - 1] = acts[i]
                if (start // batch_size) % 50 == 0:
                    logger.info(
                        f"{op}: batch {start // batch_size + 1}/{-(-len(prompts) // batch_size)}"
                    )
        assert set(grids) == set(positions), (
            f"missing positions: {set(positions).difference(grids)}"
        )

        out_path = out_dir / f"resid_stream_{op}.npz"
        arrays: dict[str, Any] = {
            "a": np.arange(1, max_value + 1, dtype=np.int32),
            "b": np.arange(1, max_value + 1, dtype=np.int32),
            "positions": np.array(positions),
            "layers": np.array(layers, dtype=np.int32),
            "max_value": max_value,
            "op": op,
            **{f"resid_{pos}": grids[pos] for pos in positions},
        }
        np.savez_compressed(out_path, **arrays)
        logger.info(f"wrote {op} ({out_path.stat().st_size / 1e6:.0f} MB) → {out_path}")
        out_paths.append(out_path)
        grids.clear()
    for h in handles:
        h.remove()
    return out_paths


def _as_tuple(x: "tuple[Any, ...] | Any") -> tuple[Any, ...]:
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


if __name__ == "__main__":
    fire.Fire(collect_resid_stream)
