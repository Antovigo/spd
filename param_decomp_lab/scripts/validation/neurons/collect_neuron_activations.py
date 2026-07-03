"""L18 neuron gate/up preactivations + model-answer baseline over the 0..200 operand grids.

For each op (add, sub) runs every `a<op>b=` prompt (a, b in 0..200 — all prompts are exactly
5 tokens, so no padding) and stores, at the `=` position:

- `gate_preact`, `up_preact` — `[201, 201, 14336]` fp16 grids indexed `[a, b]`. The combined
  post-SwiGLU activation `silu(gate) * up` is derivable downstream and not stored.
- `mlp_input` — the post-RMSNorm MLP input, `[201, 201, 4096]` fp16: what gate/up neuron rows
  (and gate/up subcomponent V vectors) read, so subcomponent inner activations are
  CPU-derivable on the same grid.
- baseline next-token stats (per-prompt `[201, 201]` grids): the model's argmax token and its
  probability, the true answer's first-token probability / logprob, and whether the argmax
  equals that first token.

The decomposition checkpoint is only used to locate the frozen base model (L18 MLP runs all
decompose base Llama-3.1-8B); nothing here depends on the decomposition itself, so outputs
land in the shared `runs/neurons/` census dir, not the run's `analysis/`.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.collect_neuron_activations \
        <model_path> [--ops=add,sub] [--layer=18] [--batch-size=256] [--out-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Outputs (default under `<PARAM_DECOMP_OUT_DIR>/runs/neurons/`):
- `activations_<op>.npz` — `gate_preact`, `up_preact`, `mlp_input`, plus `a`, `b`, `layer`, `op`.
- `baseline_<op>.npz` — `orig_token`, `orig_prob`, `correct_token`, `correct_prob`,
  `correct_logprob`, `is_correct`, plus `a`, `b`, `op`.
"""

from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.neurons.common import (
    D_INT,
    N_VALUES,
    NEURONS_DIR,
    VALUES,
    correct_first_token_grid,
    tokenize_grid,
)

_MODULE = "param_decomp_lab.scripts.validation.neurons.collect_neuron_activations"


def collect_neuron_activations(
    model_path: ModelPath,
    ops: str | tuple[str, ...] = "add,sub",
    layer: int = 18,
    batch_size: int = 256,
    out_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> list[Path] | None:
    ops_list = list(ops) if isinstance(ops, tuple) else ops.split(",")  # fire parses a,b as tuple
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--ops={','.join(ops_list)}",
            f"--layer={layer}",
            f"--batch-size={batch_size}",
        ]
        if out_dir is not None:
            argv.append(f"--out-dir={Path(out_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-neuron-acts")
        return None

    run = load_lm_run(model_path)
    hf = run.model.target_model  # the bare, frozen Llama-3.1-8B
    out_root = Path(out_dir).expanduser() if out_dir else NEURONS_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    mlp = hf.get_submodule(f"model.layers.{layer}.mlp")
    captured: dict[str, torch.Tensor] = {}

    def hook_gate(_m: Any, _i: Any, out: torch.Tensor) -> None:
        captured["gate"] = out[:, -1].float()

    def hook_up(_m: Any, _i: Any, out: torch.Tensor) -> None:
        captured["up"] = out[:, -1].float()

    def hook_mlp_in(_m: Any, args: Any) -> None:
        captured["mlp_input"] = args[0][:, -1].float()

    handles = [
        mlp.get_submodule("gate_proj").register_forward_hook(hook_gate),
        mlp.get_submodule("up_proj").register_forward_hook(hook_up),
        mlp.register_forward_pre_hook(hook_mlp_in),
    ]

    written: list[Path] = []
    for op in ops_list:
        input_ids = tokenize_grid(run.tokenizer, op)
        correct_token = correct_first_token_grid(run.tokenizer, op).reshape(-1)
        n = input_ids.shape[0]

        gate = np.zeros((n, D_INT), dtype=np.float16)
        up = np.zeros((n, D_INT), dtype=np.float16)
        mlp_input = np.zeros((n, 4096), dtype=np.float16)
        orig_token = np.zeros(n, dtype=np.int32)
        orig_prob = np.zeros(n, dtype=np.float32)
        correct_prob = np.zeros(n, dtype=np.float32)
        correct_logprob = np.zeros(n, dtype=np.float32)

        with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
            for start in range(0, n, batch_size):
                batch = input_ids[start : start + batch_size].to(run.device)
                captured.clear()
                logits = hf(input_ids=batch).logits[:, -1].float()
                logprobs = F.log_softmax(logits, dim=-1)
                argmax = logprobs.argmax(dim=-1)
                corr = torch.from_numpy(
                    correct_token[start : start + batch.shape[0]].astype(np.int64)
                ).to(run.device)

                sl = slice(start, start + batch.shape[0])
                gate[sl] = captured["gate"].cpu().numpy().astype(np.float16)
                up[sl] = captured["up"].cpu().numpy().astype(np.float16)
                mlp_input[sl] = captured["mlp_input"].cpu().numpy().astype(np.float16)
                orig_token[sl] = argmax.cpu().numpy().astype(np.int32)
                orig_prob[sl] = logprobs.gather(1, argmax[:, None])[:, 0].exp().cpu().numpy()
                clp = logprobs.gather(1, corr[:, None])[:, 0]
                correct_logprob[sl] = clp.cpu().numpy()
                correct_prob[sl] = clp.exp().cpu().numpy()
                if (start // batch_size) % 20 == 0:
                    logger.info(f"{op}: batch {start // batch_size + 1}/{-(-n // batch_size)}")

        grid = (N_VALUES, N_VALUES)
        acts_path = out_root / f"activations_{op}.npz"
        np.savez_compressed(
            acts_path,
            gate_preact=gate.reshape(*grid, D_INT),
            up_preact=up.reshape(*grid, D_INT),
            mlp_input=mlp_input.reshape(*grid, -1),
            a=VALUES,
            b=VALUES,
            layer=layer,
            op=op,
        )
        base_path = out_root / f"baseline_{op}.npz"
        np.savez_compressed(
            base_path,
            orig_token=orig_token.reshape(grid),
            orig_prob=orig_prob.reshape(grid),
            correct_token=correct_token.reshape(grid).astype(np.int32),
            correct_prob=correct_prob.reshape(grid),
            correct_logprob=correct_logprob.reshape(grid),
            is_correct=(orig_token == correct_token).reshape(grid),
            a=VALUES,
            b=VALUES,
            op=op,
        )
        acc = float((orig_token == correct_token).mean())
        logger.info(f"{op}: accuracy {acc:.3f}; wrote {acts_path} + {base_path}")
        written += [acts_path, base_path]

    for h in handles:
        h.remove()
    return written


if __name__ == "__main__":
    fire.Fire(collect_neuron_activations)
