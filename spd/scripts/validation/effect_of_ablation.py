"""Measure the per-position KL divergence induced by ablating one component at a time.

Usage:
    python -m spd.scripts.validation.effect_of_ablation <model_path> <components_tsv> \
        [--n-batches=1] [--nontarget] [--output=PATH]
"""

import csv
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    build_module_lookup,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath


def _load_components(
    components_path: Path, module_lookup: dict[tuple[int, str], str]
) -> list[tuple[str, int, int, str]]:
    """Return [(module_name, layer, component, matrix), ...] rows."""
    rows: list[tuple[str, int, int, str]] = []
    with components_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in reader:
            layer = int(record["layer"])
            matrix = record["matrix"]
            component = int(record["component"])
            key = (layer, matrix)
            assert key in module_lookup, (
                f"No decomposed module matches layer={layer}, matrix={matrix}. "
                f"Available: {sorted(module_lookup.keys())}"
            )
            rows.append((module_lookup[key], layer, component, matrix))
    return rows


def _build_ablation_masks(
    module_to_c: dict[str, int],
    batch_shape: tuple[int, ...],
    target_module: str,
    target_component: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """All-ones masks, with a single (module, component) entry zeroed at every position."""
    masks: dict[str, Tensor] = {}
    for name, C in module_to_c.items():
        mask = torch.ones((*batch_shape, C), device=device)
        if name == target_module:
            mask[..., target_component] = 0.0
        masks[name] = mask
    return masks


def _build_full_delta_masks(
    weight_deltas: dict[str, Tensor],
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, WeightDeltaAndMask]:
    """Delta enabled everywhere (mask=1) for every decomposed module."""
    delta_mask = torch.ones(batch_shape, device=device)
    return {name: (weight_deltas[name], delta_mask) for name in weight_deltas}


@torch.no_grad()
def _kl_and_preds(
    orig_logits: Tensor, ablated_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute per-position KL(orig || ablated) and top-1 (pred, prob) for both."""
    orig_log_probs = F.log_softmax(orig_logits, dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits, dim=-1)
    orig_probs = orig_log_probs.exp()
    kl = (orig_probs * (orig_log_probs - ablated_log_probs)).sum(dim=-1)

    orig_prob_max, orig_pred = orig_probs.max(dim=-1)
    ablated_probs = ablated_log_probs.exp()
    ablated_prob_max, ablated_pred = ablated_probs.max(dim=-1)
    return kl, orig_pred, orig_prob_max, ablated_pred, ablated_prob_max


def effect_of_ablation(
    model_path: ModelPath,
    components_path: str,
    n_batches: int = 1,
    nontarget: bool = False,
    output: str | None = None,
) -> Path:
    """Write a TSV of per-(component, prompt, position) KL divergences under ablation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = resolve_task_config(config, use_nontarget=nontarget)
    loader = build_lm_loader(task_config, config)
    single_batch = is_prompt_task(task_config)

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    components_file = Path(components_path).expanduser()
    components = _load_components(components_file, module_lookup)
    logger.info(f"Loaded {len(components)} components to ablate from {components_file}")

    default_name = "effect_of_ablation_nontarget.tsv" if nontarget else "effect_of_ablation.tsv"
    out_path = Path(output).expanduser() if output else run_dir / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer",
        "matrix",
        "component",
        "prompt",
        "pos",
        "token",
        "kl",
        "orig_pred",
        "orig_prob",
        "ablated_pred",
        "ablated_prob",
    ]

    n_to_run = 1 if single_batch else n_batches
    iterator = iterate_input_ids(loader, device)

    # Delta weights are fixed (target_weight - component_weight); compute once.
    weight_deltas = spd_model.calc_weight_deltas()

    total_writes = 0
    with out_path.open("w", newline="") as f, torch.no_grad():
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for batch_idx in tqdm(range(n_to_run), desc="batches"):
            batch = next(iterator)
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            batch_size, seq_len = batch.shape

            orig_logits = spd_model(batch)
            assert isinstance(orig_logits, Tensor)

            delta_masks = _build_full_delta_masks(weight_deltas, (batch_size, seq_len), device)
            prompt_offset = batch_idx * batch_size

            for module_name, layer, component, matrix in tqdm(
                components, desc=f"components (batch {batch_idx})", leave=False
            ):
                component_masks = _build_ablation_masks(
                    module_to_c=spd_model.module_to_c,
                    batch_shape=(batch_size, seq_len),
                    target_module=module_name,
                    target_component=component,
                    device=device,
                )
                mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=delta_masks)
                ablated_logits = spd_model(batch, mask_infos=mask_infos)
                assert isinstance(ablated_logits, Tensor)

                kl, orig_pred, orig_prob, abl_pred, abl_prob = _kl_and_preds(
                    orig_logits, ablated_logits
                )

                kl_cpu = kl.cpu()
                orig_pred_cpu = orig_pred.cpu()
                orig_prob_cpu = orig_prob.cpu()
                abl_pred_cpu = abl_pred.cpu()
                abl_prob_cpu = abl_prob.cpu()
                batch_cpu = batch.cpu()

                for b in range(batch_size):
                    prompt_idx = prompt_offset + b
                    for t in range(seq_len):
                        writer.writerow(
                            {
                                "layer": layer,
                                "matrix": matrix,
                                "component": component,
                                "prompt": prompt_idx,
                                "pos": t,
                                "token": int(batch_cpu[b, t].item()),
                                "kl": float(kl_cpu[b, t].item()),
                                "orig_pred": int(orig_pred_cpu[b, t].item()),
                                "orig_prob": float(orig_prob_cpu[b, t].item()),
                                "ablated_pred": int(abl_pred_cpu[b, t].item()),
                                "ablated_prob": float(abl_prob_cpu[b, t].item()),
                            }
                        )
                        total_writes += 1

    logger.info(f"Wrote {total_writes} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(effect_of_ablation)
