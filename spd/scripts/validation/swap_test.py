"""Swap two components' U vectors (rescaled by their mean activations) and test the result.

Builds a modified decomposed model where the output directions of a pair of components in the same
matrix are swapped. The swap is rescaled by the ratio of the components' mean inner activations
(V^T x) so the post-swap output magnitudes match what the other component used to produce. The
swapped model is then evaluated on the LM target prompts and on `--n-nontarget-batches` of the
nontarget dataset; for each (prompt, pos) the script stores the orig/swapped predictions and the
probability of both task target tokens.

Usage:
    python -m spd.scripts.validation.swap_test <model_path> <alive_components_tsv> \
        --layer=1 --matrix=attn.q_proj --a-component=3 --b-component=7 \
        --target-a=" np" --target-b=" pd" \
        [--n-nontarget-batches=1] [--prompts=PATH] [--batch-size=N] \
        [--output-target=PATH] [--output-nontarget=PATH]
"""

import csv
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    build_module_lookup,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath


@dataclass
class SwapSpec:
    module_name: str
    a_component: int
    b_component: int
    a_mean: float
    b_mean: float
    target_a_id: int
    target_b_id: int


def _read_mean_activations(
    alive_components_path: Path, layer: int, matrix: str, components: tuple[int, int]
) -> tuple[float, float]:
    """Look up mean_activation for both components in one TSV scan."""
    wanted = set(components)
    found: dict[int, float] = {}
    with alive_components_path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            if int(record["layer"]) != layer or record["matrix"] != matrix:
                continue
            c = int(record["component"])
            if c in wanted:
                found[c] = float(record["mean_activation"])
                if len(found) == len(wanted):
                    break
    missing = wanted - found.keys()
    assert not missing, (
        f"No row for (layer={layer}, matrix={matrix!r}, component in {sorted(missing)}) in "
        f"{alive_components_path}"
    )
    return found[components[0]], found[components[1]]


def _single_token_id(tokenizer: PreTrainedTokenizer, text: str, label: str) -> int:
    encoded: Any = tokenizer(text, add_special_tokens=False)  # pyright: ignore[reportCallIssue]
    ids: list[int] = encoded["input_ids"]
    assert len(ids) == 1, f"{label} {text!r} must tokenize to exactly one token, got {ids}"
    return ids[0]


@contextmanager
def _swapped_u_vectors(spd_model: ComponentModel, spec: SwapSpec) -> Generator[None]:
    """Swap U[A] and U[B] on the given module, rescaled by the activation ratio.

    Saves a clone of the two rows on entry and restores them on exit, so the model's state is
    unchanged once the `with` block is done.
    """
    U = spd_model.components[spec.module_name].U
    assert spec.a_component < U.shape[0] and spec.b_component < U.shape[0], (
        f"Component indices out of range for {spec.module_name} (C={U.shape[0]})"
    )
    u_a_old = U.data[spec.a_component].clone()
    u_b_old = U.data[spec.b_component].clone()
    U.data[spec.a_component] = (spec.b_mean / spec.a_mean) * u_b_old
    U.data[spec.b_component] = (spec.a_mean / spec.b_mean) * u_a_old
    try:
        yield
    finally:
        U.data[spec.a_component] = u_a_old
        U.data[spec.b_component] = u_b_old


def _build_full_mask_infos(
    spd_model: ComponentModel,
    weight_deltas: dict[str, Tensor],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, ComponentsMaskInfo]:
    """All-ones component masks + delta enabled everywhere, using the pre-swap weight deltas."""
    delta_mask = torch.ones(batch_shape, device=device)
    component_masks = {
        name: torch.ones((*batch_shape, C), device=device)
        for name, C in spd_model.module_to_c.items()
    }
    deltas_and_masks = {name: (weight_deltas[name], delta_mask) for name in weight_deltas}
    return make_mask_infos(component_masks, weight_deltas_and_masks=deltas_and_masks)


def _probs_from_logits(
    logits: Tensor, target_a_id: int, target_b_id: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (log_probs, pred, pred_prob, p_target_a, p_target_b)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pred_prob, pred = probs.max(dim=-1)
    return log_probs, pred, pred_prob, probs[..., target_a_id], probs[..., target_b_id]


_CORE_FIELDS = [
    "prompt",
    "pos",
    "token",
    "orig_pred",
    "orig_prob",
    "swapped_pred",
    "swapped_prob",
    "p_target_a_orig",
    "p_target_a_swapped",
    "p_target_b_orig",
    "p_target_b_swapped",
    "kl",
]


def _write_rows(
    writer: "csv.DictWriter[str]",
    batch: Tensor,
    orig_logits: Tensor,
    swapped_logits: Tensor,
    spec: SwapSpec,
    prompt_offset: int,
    batch_idx: int | None,
) -> None:
    """Compute per-(prompt, pos) rows from the two logit tensors and write them out."""
    orig_log_probs, orig_pred, orig_prob, p_a_orig, p_b_orig = _probs_from_logits(
        orig_logits, spec.target_a_id, spec.target_b_id
    )
    swapped_log_probs, swapped_pred, swapped_prob, p_a_swapped, p_b_swapped = _probs_from_logits(
        swapped_logits, spec.target_a_id, spec.target_b_id
    )
    kl = (orig_log_probs.exp() * (orig_log_probs - swapped_log_probs)).sum(dim=-1)

    batch_size, seq_len = batch.shape
    batch_cpu = batch.cpu().tolist()
    orig_pred_cpu = orig_pred.cpu().tolist()
    orig_prob_cpu = orig_prob.cpu().tolist()
    swapped_pred_cpu = swapped_pred.cpu().tolist()
    swapped_prob_cpu = swapped_prob.cpu().tolist()
    p_a_orig_cpu = p_a_orig.cpu().tolist()
    p_b_orig_cpu = p_b_orig.cpu().tolist()
    p_a_swapped_cpu = p_a_swapped.cpu().tolist()
    p_b_swapped_cpu = p_b_swapped.cpu().tolist()
    kl_cpu = kl.cpu().tolist()
    extra = {"batch_idx": batch_idx} if batch_idx is not None else {}

    for b in range(batch_size):
        prompt_idx = prompt_offset + b
        for t in range(seq_len):
            writer.writerow(
                {
                    **extra,
                    "prompt": prompt_idx,
                    "pos": t,
                    "token": batch_cpu[b][t],
                    "orig_pred": orig_pred_cpu[b][t],
                    "orig_prob": orig_prob_cpu[b][t],
                    "swapped_pred": swapped_pred_cpu[b][t],
                    "swapped_prob": swapped_prob_cpu[b][t],
                    "p_target_a_orig": p_a_orig_cpu[b][t],
                    "p_target_a_swapped": p_a_swapped_cpu[b][t],
                    "p_target_b_orig": p_b_orig_cpu[b][t],
                    "p_target_b_swapped": p_b_swapped_cpu[b][t],
                    "kl": kl_cpu[b][t],
                }
            )


def _run_pass(
    spd_model: ComponentModel,
    iterator: Iterator[Tensor],
    n_batches: int,
    weight_deltas: dict[str, Tensor],
    spec: SwapSpec,
    out_path: Path,
    device: torch.device,
    include_batch_idx: bool,
) -> None:
    """Run `n_batches` through orig + swapped, streaming rows to `out_path`.

    `mask_infos` is built from the first batch's shape and reused (DataLoader uses `drop_last=True`
    and `StaticBatchLoader` yields the same cached batch, so shapes are stable across the pass).
    """
    fieldnames = (["batch_idx"] if include_batch_idx else []) + _CORE_FIELDS
    first_shape: tuple[int, ...] | None = None
    mask_infos: dict[str, ComponentsMaskInfo] | None = None

    with out_path.open("w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for batch_idx in tqdm(range(n_batches), desc=out_path.name):
            batch = next(iterator)
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            shape = tuple(batch.shape)
            if first_shape is None:
                first_shape = shape
                mask_infos = _build_full_mask_infos(spd_model, weight_deltas, shape, device)
            else:
                assert shape == first_shape, (
                    f"Batch shape changed mid-pass: got {shape}, expected {first_shape}"
                )
            assert mask_infos is not None

            orig_logits = spd_model(batch)
            assert isinstance(orig_logits, Tensor)

            with _swapped_u_vectors(spd_model, spec):
                swapped_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(swapped_logits, Tensor)

            _write_rows(
                writer=writer,
                batch=batch,
                orig_logits=orig_logits,
                swapped_logits=swapped_logits,
                spec=spec,
                prompt_offset=batch_idx * first_shape[0],
                batch_idx=batch_idx if include_batch_idx else None,
            )


def swap_test(
    model_path: ModelPath,
    alive_components_path: str,
    layer: int,
    matrix: str,
    a_component: int,
    b_component: int,
    target_a: str,
    target_b: str,
    n_nontarget_batches: int = 1,
    prompts: str | None = None,
    batch_size: int | None = None,
    output_target: str | None = None,
    output_nontarget: str | None = None,
) -> tuple[Path, Path]:
    """Run the swap test on target prompts and on nontarget batches."""
    assert a_component != b_component, "a_component and b_component must be different"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    module_name = module_lookup.get((layer, matrix))
    assert module_name is not None, (
        f"No decomposed module matches (layer={layer}, matrix={matrix!r}). "
        f"Available: {sorted(module_lookup.keys())}"
    )

    a_mean, b_mean = _read_mean_activations(
        Path(alive_components_path).expanduser(), layer, matrix, (a_component, b_component)
    )
    assert a_mean != 0 and b_mean != 0, (
        f"Mean activations must be non-zero; got a_mean={a_mean}, b_mean={b_mean}"
    )

    assert config.tokenizer_name is not None, "config.tokenizer_name is required"
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    target_a_id = _single_token_id(tokenizer, target_a, "target_a")
    target_b_id = _single_token_id(tokenizer, target_b, "target_b")

    spec = SwapSpec(
        module_name=module_name,
        a_component=a_component,
        b_component=b_component,
        a_mean=a_mean,
        b_mean=b_mean,
        target_a_id=target_a_id,
        target_b_id=target_b_id,
    )
    logger.info(
        f"Swap on {module_name}: A={a_component} (mean={a_mean:.4g}, target={target_a!r}={target_a_id}); "
        f"B={b_component} (mean={b_mean:.4g}, target={target_b!r}={target_b_id})"
    )

    target_task_config = resolve_task_config(config, use_nontarget=False, prompts_override=prompts)
    assert is_prompt_task(target_task_config), (
        "swap_test requires a prompt-based target task (config.task_config.prompts_file)"
    )
    target_iter = iterate_input_ids(
        build_lm_loader(target_task_config, config, batch_size_override=batch_size), device
    )

    nontarget_task_config = resolve_task_config(config, use_nontarget=True)
    nontarget_iter = iterate_input_ids(
        build_lm_loader(nontarget_task_config, config, batch_size_override=batch_size), device
    )

    slug = f"{module_name.replace('.', '_')}_c{a_component}_c{b_component}"
    target_out = (
        Path(output_target).expanduser()
        if output_target
        else run_dir / f"swap_test_{slug}_target.tsv"
    )
    nontarget_out = (
        Path(output_nontarget).expanduser()
        if output_nontarget
        else run_dir / f"swap_test_{slug}_nontarget.tsv"
    )
    for p in (target_out, nontarget_out):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Weight deltas depend only on the (unswapped) model parameters, which don't change across
    # batches or between the two passes. Compute once.
    weight_deltas = spd_model.calc_weight_deltas()

    _run_pass(
        spd_model=spd_model,
        iterator=target_iter,
        n_batches=1,
        weight_deltas=weight_deltas,
        spec=spec,
        out_path=target_out,
        device=device,
        include_batch_idx=False,
    )
    _run_pass(
        spd_model=spd_model,
        iterator=nontarget_iter,
        n_batches=n_nontarget_batches,
        weight_deltas=weight_deltas,
        spec=spec,
        out_path=nontarget_out,
        device=device,
        include_batch_idx=True,
    )
    return target_out, nontarget_out


if __name__ == "__main__":
    fire.Fire(swap_test)
