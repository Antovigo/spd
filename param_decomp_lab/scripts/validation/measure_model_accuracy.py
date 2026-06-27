"""Per-prompt arithmetic accuracy of the decomposed model, optionally ablating subcomponents.

Runs every `a<op>b=` prompt of the run's target distribution through the **all-on
reconstruction** (every component on + the delta component on, which reproduces the target
model) and, for each prompt, records the probability the model puts on the correct answer
token and on every wrong answer in a `±--range` window around it (offset `k` → the first
token of `str(result + k)`). With `--ablate`, the listed subcomponents are masked off (their
`U_c V_c^T` removed from the weight) so you can read how each prompt's answer distribution
moves; the delta and all other components stay on, so the only change is the ablation.

The answer sits at the last (`=`) position — the prompt pool tokenizes to one shared length.
The correct answer is the first token of `str(result)`; for `1..100` operands every result in
the window is a 1–3 digit number, which the Llama-3 tokenizer encodes as a single token, so the
first-token probability is the number's probability.

Output: `<run_dir>/model_accuracy/accuracy[_<ablation>].json` — the filename is suffixed with the
ablated subcomponents (e.g. `accuracy_gate163_down240.json`), empty for the un-ablated model. The
JSON holds run/ablation metadata plus, per prompt, `a`, `b`, `result`, the correct token and its
probability, the model's argmax token and whether it is correct, and `offset_probs` mapping each
offset in `-range..+range` to its token's probability (offset 0 is the correct answer). A sibling
`model_accuracy_notebook.py` (marimo) is copied in to plot the results.

An 8B forward needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.measure_model_accuracy <model_path> \
        [--ablate=gate_proj:163,down_proj:240] [--range=5] [--batch-size=512] \
        [--output-dir=PATH] [--slurm [--partition=... --slurm-time=1:00:00 --slurm-mem=...]]
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import fire
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.masks import ComponentsMaskInfo
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    OP_SYMBOL,
    SlurmOptions,
    load_lm_run,
    op_symbol,
    parse_module_name,
    parse_operands,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.measure_model_accuracy"
_NOTEBOOK = Path(__file__).with_name("model_accuracy_notebook.py")
_RESULT: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mult": lambda a, b: a * b,
}
_PROJ_RANK = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}


def _offsets(radius: int) -> list[int]:
    return list(range(-radius, radius + 1))


def _chunk_starts(total: int, size: int) -> list[int]:
    return list(range(0, total, size))


def _detect_op(prompt: str) -> str:
    """The operation whose `a<sym>b=` pattern this prompt matches."""
    for op in OP_SYMBOL:
        try:
            parse_operands(prompt, op)
            return op
        except AssertionError:
            continue
    raise AssertionError(f"prompt matches no known operation: {prompt!r}")


def _resolve_ablations(ablate: str, modules: list[str]) -> list[tuple[str, int]]:
    """Parse `matrix:component,...` into `(module_path, component)`, sorted gate<up<down.

    `matrix` may be bare (`gate`, `gate_proj`) or qualified (`mlp.gate_proj`); it is matched
    by suffix against the decomposed module paths and must resolve to exactly one module.
    """
    out: list[tuple[str, int]] = []
    for item in ablate.split(","):
        name, sep, cstr = item.partition(":")
        assert sep and cstr, f"ablation spec must be 'matrix:component', got {item!r}"
        proj = name if name.endswith("_proj") else f"{name}_proj"
        matches = [m for m in modules if m.endswith(f".{proj}")]
        assert len(matches) == 1, (
            f"ablation {item!r}: matrix {proj!r} matched {len(matches)} modules {matches}; "
            "qualify it (e.g. layers.18.mlp.gate_proj:163)"
        )
        out.append((matches[0], int(cstr)))
    return sorted(out, key=lambda mc: (_PROJ_RANK.get(mc[0].split(".")[-1], 9), mc[1]))


def _ablation_label(module: str, component: int) -> dict[str, Any]:
    layer, matrix = parse_module_name(module)
    proj = matrix.split(".")[-1]
    return {"module": module, "layer": layer, "proj": proj, "component": component}


def _ablation_suffix(ablations: list[tuple[str, int]]) -> str:
    """Filesystem-safe `_gate163_down240`-style suffix (empty for no ablation)."""
    short = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
    parts = [f"{short.get(m.split('.')[-1], m.split('.')[-1])}{c}" for m, c in ablations]
    return ("_" + "_".join(parts)) if parts else ""


def _all_on_infos(
    modules: list[str],
    weight_deltas: dict[str, Tensor],
    n_comp: dict[str, int],
    batch: int,
    seq: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Tensor], dict[str, ComponentsMaskInfo]]:
    """Component masks all-ones + delta fully on (= exact reconstruction); returns the mutable
    per-module mask tensors (to zero ablated columns) and the assembled infos sharing them."""
    masks = {m: torch.ones(batch, seq, n_comp[m], device=device, dtype=dtype) for m in modules}
    delta_mask = torch.ones(batch, seq, device=device, dtype=dtype)
    infos = {
        m: ComponentsMaskInfo(
            component_mask=masks[m],
            routing_mask="all",
            weight_delta_and_mask=(weight_deltas[m], delta_mask),
        )
        for m in modules
    }
    return masks, infos


def measure_model_accuracy(
    model_path: ModelPath,
    ablate: str | None = None,
    range: int = 5,
    batch_size: int = 512,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    radius = range
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--range={radius}",
            f"--batch-size={batch_size}",
        ]
        if ablate is not None:
            argv.append(f"--ablate={ablate}")
        if output_dir is not None:
            argv.append(f"--output-dir={Path(output_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-measure-accuracy")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    assert cfg.data.prompts_file is not None, "requires prompts-based target data"

    prompt_texts = [
        ln.strip()
        for ln in Path(cfg.data.prompts_file).expanduser().read_text().splitlines()
        if ln.strip()
    ]
    op = _detect_op(prompt_texts[0])
    result_fn = _RESULT[op]
    ab = [parse_operands(t, op) for t in prompt_texts]

    pool = load_prompts_dataset(cfg.data.prompts_file, cast(Any, tokenizer)).to(device)
    assert pool.shape[0] == len(prompt_texts)
    seq = pool.shape[1]

    modules = sorted(model.components.keys(), key=parse_module_name)
    n_comp = {m: model.components[m].V.shape[1] for m in modules}
    ablations = _resolve_ablations(ablate, modules) if ablate else []
    weight_deltas = model.calc_weight_deltas()
    dtype = next(model.parameters()).dtype
    logger.info(
        f"op={op} ({op_symbol(op)}), {pool.shape[0]} prompts, {len(modules)} modules, "
        f"range=±{radius}, ablations={[(m.split('.')[-1], c) for m, c in ablations]}"
    )

    # Per-prompt correct + windowed token ids (first token of str(result + offset)). Single-token
    # numbers for the 1..100 grid, so this is the number's probability.
    offsets = _offsets(radius)
    token_cache: dict[str, int] = {}

    def first_token(s: str) -> int:
        if s not in token_cache:
            ids = tokenizer.encode(s, add_special_tokens=False)
            assert ids, f"{s!r} tokenized to nothing"
            token_cache[s] = ids[0]
        return token_cache[s]

    results = [result_fn(a, b) for a, b in ab]
    correct_ids = torch.tensor([first_token(str(c)) for c in results], device=device)
    offset_ids = torch.tensor(
        [[first_token(str(c + off)) for off in offsets] for c in results], device=device
    )  # [P, n_off]

    p_correct = torch.empty(pool.shape[0])
    p_offsets = torch.empty(pool.shape[0], len(offsets))
    pred_ids = torch.empty(pool.shape[0], dtype=torch.long)
    p_pred = torch.empty(pool.shape[0])

    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in _chunk_starts(pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            sl = slice(start, start + b)
            masks, infos = _all_on_infos(modules, weight_deltas, n_comp, b, seq, device, dtype)
            for m, c in ablations:
                masks[m][:, :, c] = 0.0
            logits = model(chunk, mask_infos=infos)  # [b, seq, vocab]
            probs = logits[:, -1].float().softmax(dim=-1)  # [b, vocab] at the = position

            if start == 0:
                _reconstruction_check(model, chunk, logits, bool(ablations))
            p_correct[sl] = probs.gather(1, correct_ids[sl, None])[:, 0].cpu()
            p_offsets[sl] = probs.gather(1, offset_ids[sl]).cpu()
            pred = probs.argmax(dim=-1)
            pred_ids[sl] = pred.cpu()
            p_pred[sl] = probs.gather(1, pred[:, None])[:, 0].cpu()
            logger.info(f"prompts {start}..{start + b}")

    decode = {int(t): tokenizer.decode([int(t)]) for t in set(pred_ids.tolist()) | set(results)}
    correct_decode = {int(t): tokenizer.decode([int(t)]) for t in correct_ids.tolist()}
    prompts_out: list[dict[str, Any]] = []
    n_correct = 0
    for i, ((a, b), c) in enumerate(zip(ab, results, strict=True)):
        cid, pid = int(correct_ids[i]), int(pred_ids[i])
        is_correct = pid == cid
        n_correct += int(is_correct)
        prompts_out.append(
            {
                "a": a,
                "b": b,
                "result": c,
                "correct_token_id": cid,
                "correct_token": correct_decode[cid],
                "p_correct": round(float(p_correct[i]), 6),
                "predicted_token_id": pid,
                "predicted_token": decode[pid],
                "correct": is_correct,
                "offset_probs": {
                    str(off): round(float(p_offsets[i, j]), 6) for j, off in enumerate(offsets)
                },
            }
        )

    accuracy = n_correct / len(prompts_out)
    payload = {
        "run_id": run.run_dir.name,
        "op": op,
        "symbol": op_symbol(op),
        "range": radius,
        "position": "last (= answer)",
        "model": "all-on reconstruction" + (" with ablations" if ablations else ""),
        "ablated": [_ablation_label(m, c) for m, c in ablations],
        "n_prompts": len(prompts_out),
        "accuracy": round(accuracy, 6),
        "mean_p_correct": round(float(p_correct.mean()), 6),
        "prompts": prompts_out,
    }

    out_dir = Path(output_dir).expanduser() if output_dir else run.run_dir / "model_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"accuracy{_ablation_suffix(ablations)}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    assert _NOTEBOOK.exists(), f"notebook template missing: {_NOTEBOOK}"
    shutil.copyfile(_NOTEBOOK, out_dir / _NOTEBOOK.name)
    logger.info(
        f"accuracy {accuracy:.4f}, mean P(correct) {payload['mean_p_correct']:.4f} → {out_path}"
    )
    return out_path


def _reconstruction_check(model: Any, chunk: Tensor, recon_logits: Tensor, ablated: bool) -> None:
    """All-on + delta must reproduce the target model (only meaningful un-ablated)."""
    tgt = torch.log_softmax(model(chunk)[:, -1].float(), dim=-1)
    rec = torch.log_softmax(recon_logits[:, -1].float(), dim=-1)
    kl = (tgt.exp() * (tgt - rec)).sum(-1).mean().item()
    agree = (tgt.argmax(-1) == rec.argmax(-1)).float().mean().item()
    logger.info(f"reconstruction check: mean KL(target||recon)={kl:.4g}, argmax agree={agree:.3f}")
    if not ablated:
        assert kl < 0.5, f"all-on+delta != target (KL {kl}); mask/delta wiring bug"


if __name__ == "__main__":
    fire.Fire(measure_model_accuracy)
