"""Collect L18 MLP hidden activations over an operation's 100x100 prompt grid.

Runs every `a<op>b=` prompt of one operation (add / sub / mult) through the target model
and stores, at the **last token** (the `=` answer position), the activations at five hook
points around layer 18's MLP:
- `resid_pre_mlp`  — residual stream entering the MLP (input to `post_attention_layernorm`)
- `mlp_input`      — MLP input after the RMSNorm (the gate/up projection input)
- `gate_preact`    — gate projection output, before the SwiGLU nonlinearity
- `up_preact`      — up projection output, before the SwiGLU nonlinearity
- `mlp_output`     — MLP output after down projection (what is added back to the residual)

Each is stored as a `[N, N, dim]` grid indexed `[a-1, b-1]` (float16 to bound size). The
post-SwiGLU neuron activation `silu(gate_preact) * up_preact` is derivable downstream and
not stored separately.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.collect_hidden_activations <model_path> \
        [--op=add] [--layer=18] [--batch-size=256] [--output=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Output (default `hidden_activations_<op>.npz` in the run folder): the five grids above
plus `a`, `b` axis values, `op`, and `layer`.
"""

from pathlib import Path
from typing import Any, cast

import fire
import numpy as np
import torch
from torch import Tensor, nn

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    op_prompts_file,
    op_symbol,
    parse_operands,
    square_grid_size,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.collect_hidden_activations"
# hook-point name -> (submodule path suffix under the layer, "input" | "output")
_HOOKS = {
    "resid_pre_mlp": ("post_attention_layernorm", "input"),
    "mlp_input": ("post_attention_layernorm", "output"),
    "gate_preact": ("mlp.gate_proj", "output"),
    "up_preact": ("mlp.up_proj", "output"),
    "mlp_output": ("mlp.down_proj", "output"),
}


def _mlp_layer(component_modules: list[str]) -> int:
    """The single transformer block whose MLP is decomposed (the analysis target)."""
    layers = {int(m.split(".layers.")[1].split(".")[0]) for m in component_modules if ".mlp." in m}
    assert len(layers) == 1, f"expected exactly one decomposed-MLP layer, got {sorted(layers)}"
    return layers.pop()


def collect_hidden_activations(
    model_path: ModelPath,
    op: str = "add",
    layer: int | None = None,
    batch_size: int = 256,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [str(Path(model_path).expanduser()), f"--op={op}", f"--batch-size={batch_size}"]
        if layer is not None:
            argv.append(f"--layer={layer}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-hidden-acts-{op}")
        return None

    run = load_lm_run(model_path)
    model, cfg, device = run.model, run.cfg, run.device
    layer = layer if layer is not None else _mlp_layer(list(model.components.keys()))

    prompts_file = op_prompts_file(op)
    prompt_texts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
    ab = [parse_operands(t, op) for t in prompt_texts]
    n = square_grid_size(ab)
    pool = load_prompts_dataset(str(prompts_file), cast(Any, run.tokenizer)).to(device)
    assert pool.shape[0] == len(prompt_texts)
    logger.info(f"{op} ({op_symbol(op)}): {len(prompt_texts)} prompts, N={n}, layer {layer}")

    # Forward hooks stash each batch's last-token activations; a plain forward (no masks)
    # runs the bare target model, so these are the true module in/outputs.
    captured: dict[str, Tensor] = {}
    handles = []
    base = model.target_model.get_submodule(f"model.layers.{layer}")
    for name, (suffix, which) in _HOOKS.items():
        submodule = base.get_submodule(suffix)
        if which == "input":

            def pre_hook(_m: nn.Module, inputs: tuple[Tensor, ...], _name: str = name) -> None:
                captured[_name] = inputs[0][:, -1].float()

            handles.append(submodule.register_forward_pre_hook(pre_hook))
        else:

            def out_hook(
                _m: nn.Module, _inp: tuple[Tensor, ...], output: Tensor, _name: str = name
            ) -> None:
                captured[_name] = output[:, -1].float()

            handles.append(submodule.register_forward_hook(out_hook))

    grids: dict[str, np.ndarray] = {}

    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            captured.clear()
            model(chunk)
            assert set(captured) == set(_HOOKS), (
                f"missing hook captures: {set(_HOOKS) - set(captured)}"
            )
            for name, act in captured.items():
                if name not in grids:
                    grids[name] = np.zeros((n, n, act.shape[-1]), dtype=np.float16)
                acts = act.cpu().numpy().astype(np.float16)
                for i, (a, b) in enumerate(ab[start : start + chunk.shape[0]]):
                    grids[name][a - 1, b - 1] = acts[i]
            logger.info(
                f"batch {start // batch_size + 1}: prompts {start}..{start + chunk.shape[0]}"
            )

    for h in handles:
        h.remove()

    out_path = Path(output).expanduser() if output else run.run_dir / f"hidden_activations_{op}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "a": np.arange(1, n + 1, dtype=np.int32),
        "b": np.arange(1, n + 1, dtype=np.int32),
        "op": op,
        "layer": layer,
        **grids,
    }
    np.savez_compressed(out_path, **payload)
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"wrote {len(grids)} hidden-activation grids ({size_mb:.0f} MB) → {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(collect_hidden_activations)
