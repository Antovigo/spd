"""Per-subcomponent last-position ablation effect on the `a<op>b=` grid — measured, not CI.

The subcomponent analogue of `collect_neuron_ablation_kl`, for a decomposition run's L18 MLP
matrices (`gate_proj` / `up_proj` / `down_proj`). Learned causal importances are the model's
own prediction of maskability; this measures the real thing: remove one component's rank-1
weight `U_c V_c^T` from the frozen target weight (the full reconstruction incl. the weight
delta sums to exactly `W`, so this is `W - U_c V_c^T` with no reconstruction floor) **at the
`=` position only**, and read how the next-token distribution moves.

Mechanics per matrix, all from one clean forward's captures (MLP input `x`, gate/up preacts,
post-SwiGLU acts, block output `h`, tail K/V):
- `down`: `h' = h - (acts · V_c) U_c` — exactly like a neuron but with a distributed read.
- `gate`: `gate' = gate - (x · V_c) U_c`, `h' = h + W_down (silu(gate')·up - silu(gate)·up)`.
- `up`:   `up'   = up   - (x · V_c) U_c`, `h' = h + W_down (silu(gate)·up' - silu(gate)·up)`.
The patched row then re-runs layers 19..31 against the clean KV cache (`_PatchedTail`), so the
same caveat applies: an effect flowing through operand positions is invisible.

Ablation is only valid for L18 MLP matrices: ablating attention k/v projections would change
the prefix KV cache, which this patch keeps frozen — the script asserts the projs it targets.

Components default to **all C per matrix** (the point is not trusting CI); `--stride` gives
the same screen/full split as the neuron script. Output rows are keyed `(matrix, component)`.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.collect_subcomp_ablation_kl \
        <model_path> [--op=add] [--stride=5] [--components-tsv=PATH] [--layer=18] \
        [--batch-size=64] [--chunk=128] [--output=PATH] [--slurm [...]]

Output (default `subcomp_ablation_<screen|full>_<op>.npz` in the run's `analysis/datasets/`):
same arrays as the neuron script (`kl`, `abl_token`, `abl_prob`, `answer_flip`,
`delta_correct_logprob`, stride-1 `offset_logprob` + `clean_offset_logprob`, `null_kl`,
`orig_token`, `a`, `b`, `offsets`, `layer`, `op`, `stride`) with `matrix` (str array) and
`component` (int32) in place of `neuron_ids`.
"""

import csv
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
    analysis_datasets_dir,
    load_component_uv,
    load_lm_run,
    op_symbol,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.neurons.collect_neuron_ablation_kl import (
    _AblationStats,
    _PatchedTail,
    _to_grid,
)
from param_decomp_lab.scripts.validation.neurons.common import (
    OFFSETS,
    VALUES,
    correct_first_token_grid,
    offset_first_token_grid,
)

_MODULE = "param_decomp_lab.scripts.validation.neurons.collect_subcomp_ablation_kl"

MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
_NULL_KL_MAX = 0.02


def _read_component_list(components_tsv: Path) -> list[tuple[str, int]]:
    with open(components_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows and "matrix" in rows[0] and "component" in rows[0], (
        f"expected matrix/component columns in {components_tsv}"
    )
    return [(r["matrix"].removeprefix("mlp."), int(r["component"])) for r in rows]


def collect_subcomp_ablation_kl(
    model_path: ModelPath,
    op: str = "add",
    stride: int = 5,
    components_tsv: str | None = None,
    layer: int = 18,
    batch_size: int = 64,
    chunk: int = 128,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "8:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--op={op}",
            f"--stride={stride}",
            f"--layer={layer}",
            f"--batch-size={batch_size}",
            f"--chunk={chunk}",
        ]
        if components_tsv is not None:
            argv.append(f"--components-tsv={Path(components_tsv).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-scabl-{op}")
        return None

    from param_decomp_lab.experiments.lm.run import SavedLMRun

    checkpoint = SavedLMRun.from_path(model_path).checkpoint_path
    run = load_lm_run(model_path)
    hf = run.model.target_model
    tokenizer = run.tokenizer
    device = run.device

    model_dtype = next(hf.parameters()).dtype
    uv = load_component_uv(checkpoint, layer, MLP_PROJS)
    uv_t = {
        proj: (
            torch.from_numpy(v).to(device, model_dtype),
            torch.from_numpy(u).to(device, model_dtype),
        )
        for proj, (v, u) in uv.items()
    }
    if components_tsv is None:
        comps = [(proj, c) for proj in MLP_PROJS for c in range(uv[proj][0].shape[1])]
    else:
        comps = _read_component_list(Path(components_tsv).expanduser())
        assert all(proj in MLP_PROJS for proj, _ in comps), (
            "last-position patching is only valid for the L18 MLP projections"
        )
    by_proj: dict[str, list[int]] = {proj: [] for proj in MLP_PROJS}
    for proj, c in comps:
        by_proj[proj].append(c)
    row_of: dict[tuple[str, int], int] = {}
    for proj in MLP_PROJS:
        for c in by_proj[proj]:
            row_of[(proj, c)] = len(row_of)
    n_rows = len(row_of)

    values = VALUES[::stride]
    n_val = len(values)
    sym = op_symbol(op)
    prompts = [f"{a}{sym}{b}=" for a in values for b in values]
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    assert isinstance(input_ids, torch.Tensor) and input_ids.shape == (len(prompts), 5)
    correct = correct_first_token_grid(tokenizer, op)[::stride, ::stride].reshape(-1)
    with_offsets = stride == 1
    offset_tok_all = (
        offset_first_token_grid(tokenizer, op)[::stride, ::stride].reshape(-1, len(OFFSETS))
        if with_offsets
        else None
    )

    mlp = hf.get_submodule(f"model.layers.{layer}.mlp")
    w_down = mlp.get_submodule("down_proj").weight
    assert isinstance(w_down, torch.Tensor)
    tail = _PatchedTail(hf, layer, device)
    captured: dict[str, torch.Tensor] = {}

    def hook_out(name: str) -> Any:
        def hook(_m: Any, _i: Any, out: torch.Tensor) -> None:
            captured[name] = out[:, -1]

        return hook

    def hook_x(_m: Any, args: Any) -> None:
        captured["x"] = args[0][:, -1]

    extra_handles = [
        mlp.get_submodule("gate_proj").register_forward_hook(hook_out("gate")),
        mlp.get_submodule("up_proj").register_forward_hook(hook_out("up")),
        mlp.register_forward_pre_hook(hook_x),
    ]

    n_p = len(prompts)
    logger.info(f"{op}: {n_rows} subcomponents x {n_p} prompts (stride {stride}), layer {layer}")
    kl_out = np.zeros((n_rows, n_p), dtype=np.float16)
    abl_token_out = np.zeros((n_rows, n_p), dtype=np.int32)
    abl_prob_out = np.zeros((n_rows, n_p), dtype=np.float16)
    dlp_out = np.zeros((n_rows, n_p), dtype=np.float16)
    offset_out = np.zeros((n_rows, n_p, len(OFFSETS)), dtype=np.float16) if with_offsets else None
    clean_offset_out = np.zeros((n_p, len(OFFSETS)), dtype=np.float32) if with_offsets else None
    null_kl = np.zeros(n_p, dtype=np.float16)
    orig_token = np.zeros(n_p, dtype=np.int32)

    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        n_batches = -(-n_p // batch_size)
        for bi, start in enumerate(range(0, n_p, batch_size)):
            batch = input_ids[start : start + batch_size].to(device)
            bsz = batch.shape[0]
            tail.capture = True
            clean_logits = hf(input_ids=batch).logits[:, -1].float()
            clean_logprobs = F.log_softmax(clean_logits, dim=-1)
            clean_argmax = clean_logprobs.argmax(dim=-1)
            correct_tok = torch.from_numpy(correct[start : start + bsz].astype(np.int64)).to(device)
            orig_token[start : start + bsz] = clean_argmax.cpu().numpy().astype(np.int32)
            offset_tok = None
            if offset_tok_all is not None:
                offset_tok = torch.from_numpy(
                    offset_tok_all[start : start + bsz].astype(np.int64)
                ).to(device)
                assert clean_offset_out is not None
                clean_offset_out[start : start + bsz] = (
                    clean_logprobs.gather(1, offset_tok).cpu().numpy()
                )

            clean_kv = tail.clean_kv()
            tail.capture = False
            h_site = tail.h_patch_site
            acts = tail.neuron_acts  # silu(gate)·up at `=`, [bsz, d_int]
            x = captured["x"]  # MLP input, [bsz, d_model]
            gate = captured["gate"]
            up = captured["up"]
            assert h_site is not None and acts is not None

            null_logits = tail.forward(h_site, clean_kv, expand=1)
            null_stats = _AblationStats(
                null_logits,
                torch.arange(bsz, device=device),
                clean_logprobs,
                correct_tok,
                offset_tok=None,
            )
            null_kl[start : start + bsz] = null_stats.kl.cpu().numpy().astype(np.float16)
            batch_null_max = float(null_stats.kl.max())
            assert batch_null_max < _NULL_KL_MAX, (
                f"null-patch KL noise floor too high on batch {bi}: {batch_null_max}"
            )

            for proj in MLP_PROJS:
                comp_ids = by_proj[proj]
                v_all, u_all = uv_t[proj]
                for cs in range(0, len(comp_ids), chunk):
                    nc = torch.tensor(comp_ids[cs : cs + chunk], device=device)
                    c = nc.shape[0]
                    v = v_all[:, nc]  # [d_in, c]
                    u = u_all[nc]  # [c, d_out]
                    match proj:
                        case "down_proj":
                            inner = acts @ v  # [bsz, c]
                            h_rows = h_site[:, None, :] - inner[:, :, None] * u[None]
                        case "gate_proj":
                            inner = x @ v
                            gate_abl = gate[:, None, :] - inner[:, :, None] * u[None]
                            acts_abl = F.silu(gate_abl) * up[:, None, :]
                            h_rows = h_site[:, None, :] + (acts_abl - acts[:, None, :]) @ w_down.T
                        case "up_proj":
                            inner = x @ v
                            up_abl = up[:, None, :] - inner[:, :, None] * u[None]
                            acts_abl = F.silu(gate)[:, None, :] * up_abl
                            h_rows = h_site[:, None, :] + (acts_abl - acts[:, None, :]) @ w_down.T
                    h_rows = h_rows.reshape(bsz * c, -1)
                    prompt_idx = torch.arange(bsz, device=device).repeat_interleave(c)
                    abl_logits = tail.forward(h_rows, clean_kv, expand=c)
                    stats = _AblationStats(
                        abl_logits, prompt_idx, clean_logprobs, correct_tok, offset_tok
                    )
                    r0 = row_of[(proj, comp_ids[cs])]
                    nsl = slice(r0, r0 + c)
                    psl = slice(start, start + bsz)
                    kl_out[nsl, psl] = _to_grid(stats.kl, bsz, c, np.float16)
                    abl_token_out[nsl, psl] = _to_grid(stats.abl_argmax, bsz, c, np.int32)
                    abl_prob_out[nsl, psl] = _to_grid(stats.abl_prob, bsz, c, np.float16)
                    dlp_out[nsl, psl] = _to_grid(stats.delta_correct_logprob, bsz, c, np.float16)
                    if offset_out is not None:
                        assert stats.offset_logprob is not None
                        offset_out[nsl, psl] = _to_grid(stats.offset_logprob, bsz, c, np.float16)
            if bi % 10 == 0:
                logger.info(f"{op}: prompt batch {bi + 1}/{n_batches}")

    tail.remove_hooks()
    for h in extra_handles:
        h.remove()
    logger.info(f"null-patch KL floor: mean {null_kl.mean():.2e}, max {float(null_kl.max()):.2e}")

    mode = "screen" if stride > 1 else "full"
    out_path = (
        Path(output).expanduser()
        if output
        else analysis_datasets_dir(run.run_dir) / f"subcomp_ablation_{mode}_{op}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = (n_rows, n_val, n_val)
    matrix_col = np.array([proj for proj, _ in row_of], dtype=np.str_)
    component_col = np.array([c for _, c in row_of], dtype=np.int32)
    arrays: dict[str, Any] = {
        "matrix": matrix_col,
        "component": component_col,
        "kl": kl_out.reshape(grid),
        "abl_token": abl_token_out.reshape(grid),
        "abl_prob": abl_prob_out.reshape(grid),
        "answer_flip": (abl_token_out != orig_token[None, :]).reshape(grid),
        "delta_correct_logprob": dlp_out.reshape(grid),
        "null_kl": null_kl.reshape(n_val, n_val),
        "orig_token": orig_token.reshape(n_val, n_val),
        "a": values,
        "b": values,
        "offsets": np.array(OFFSETS, dtype=np.int32),
        "layer": layer,
        "op": op,
        "stride": stride,
    }
    if offset_out is not None and clean_offset_out is not None:
        arrays["offset_logprob"] = offset_out.reshape(*grid, len(OFFSETS))
        arrays["clean_offset_logprob"] = clean_offset_out.reshape(n_val, n_val, len(OFFSETS))
    np.savez_compressed(out_path, **arrays)
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


if __name__ == "__main__":
    fire.Fire(collect_subcomp_ablation_kl)
