"""Check whether each alive component's function is also captured elsewhere in the decomposition.

For each component X from the alive-components TSV we run two ablations and compare each to the
original target model output:
- case a ("circuit"): every decomposed module has 1 on its alive components and 0 elsewhere,
  delta is OFF, and X is additionally set to 0. If the alive set is mechanistically complete,
  this should still approximate the target; any large KL here shows the function of X is not
  carried by another alive component.
- case b ("all-on"): every component is on and delta is ON, except X. If KL stays small here,
  something outside the alive set (delta or an inactive component) is doing X's job in parallel.

We also record a "circuit, X on" baseline KL per (prompt, pos): the same case-a setup but with
no component ablated. Subtracting this from `kl_circuit` isolates X's marginal effect from the
irreducible imperfection of the circuit approximation. (Case b with no ablation is KL=0 by
construction since components + delta = target exactly.) The baseline is the same across
components at a given (prompt, pos), so it's written to its own smaller TSV — one row per
(prompt, pos) in iteration order, no prompt/pos columns — mergeable via `i = prompt * seq_len + pos`.

Rows in the main TSV are filtered to positions where X's lower-leaky CI exceeds `--ci-threshold`
(default 0.1). The baseline TSV is always dense.

Loop order is batches outer, components inner: per batch we do one target-model forward (for the
baseline distribution), one circuit no-ablation forward (for the shared baseline), then two
forward passes per component — one per case.

Usage:
    python -m spd.scripts.validation.completeness <model_path> <alive_components_tsv> \\
        [--n-batches=1] [--nontarget] [--prompts=PATH] [--split=SPLIT] \\
        [--batch-size=N] [--ci-threshold=0.1] \\
        [--output=PATH] [--output-baseline=PATH]
"""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fire
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.scripts.validation.common import (
    build_lm_loader,
    build_module_lookup,
    is_prompt_task,
    iterate_input_ids,
    load_spd_run,
    resolve_task_config,
)
from spd.spd_types import ModelPath

FIELDS = [
    "layer",
    "matrix",
    "component",
    "prompt",
    "pos",
    "kl_circuit",
    "kl_all",
    "ci",
]
BASELINE_FIELDS = ["kl_circuit_baseline"]


def _load_alive(
    path: Path, module_lookup: dict[tuple[int, str], str]
) -> list[tuple[str, int, str, int]]:
    """Return [(module_name, layer, matrix, component), ...] rows from alive_components.tsv."""
    rows: list[tuple[str, int, str, int]] = []
    with path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            layer = int(record["layer"])
            matrix = record["matrix"]
            component = int(record["component"])
            key = (layer, matrix)
            assert key in module_lookup, (
                f"No decomposed module matches layer={layer}, matrix={matrix}. "
                f"Available: {sorted(module_lookup.keys())}"
            )
            rows.append((module_lookup[key], layer, matrix, component))
    return rows


def _build_component_masks(
    module_to_c: dict[str, int],
    alive_idx_by_module: dict[str, list[int]],
    batch_shape: tuple[int, ...],
    device: torch.device,
    only_alive: bool,
) -> dict[str, Tensor]:
    """Per-module component mask, broadcast to `batch_shape`.

    `only_alive=True`: 1 on alive component indices, 0 everywhere else.
    `only_alive=False`: all 1s.
    """
    masks: dict[str, Tensor] = {}
    for name, c in module_to_c.items():
        if only_alive:
            row = torch.zeros(c, device=device)
            alive = alive_idx_by_module.get(name, [])
            if alive:
                row[torch.tensor(alive, device=device, dtype=torch.long)] = 1.0
        else:
            row = torch.ones(c, device=device)
        masks[name] = row.expand(*batch_shape, c).clone()
    return masks


def completeness(
    model_path: ModelPath,
    alive_components_path: str,
    n_batches: int = 1,
    nontarget: bool = False,
    prompts: str | None = None,
    split: str | None = None,
    batch_size: int | None = None,
    ci_threshold: float = 0.1,
    output: str | None = None,
    output_baseline: str | None = None,
) -> tuple[Path, Path]:
    """Write per-(alive-component, prompt, pos) KL for both ablation cases + the component's CI.

    Rows in the main TSV are filtered to positions where the component's lower-leaky CI exceeds
    `ci_threshold`. The circuit baseline KL (case a, no component ablated) is saved to a separate
    TSV with one row per (prompt, pos) in iteration order — no prompt/pos columns, since row index
    `i = prompt * seq_len + pos` recovers them.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spd_model, config, run_dir = load_spd_run(model_path)
    spd_model = spd_model.to(device)

    task_config = resolve_task_config(
        config, use_nontarget=nontarget, prompts_override=prompts, split_override=split
    )
    loader = build_lm_loader(task_config, config, batch_size_override=batch_size)
    single_batch = is_prompt_task(task_config)

    module_lookup = build_module_lookup(spd_model.target_module_paths)
    alive_file = Path(alive_components_path).expanduser()
    alive_rows = _load_alive(alive_file, module_lookup)
    alive_idx_by_module: dict[str, list[int]] = {}
    for module_name, _layer, _matrix, component in alive_rows:
        alive_idx_by_module.setdefault(module_name, []).append(component)

    n_to_run = 1 if single_batch else n_batches
    iterator = iterate_input_ids(loader, device)
    weight_deltas = spd_model.calc_weight_deltas()

    default_name = "completeness_nontarget.tsv" if nontarget else "completeness.tsv"
    baseline_default = (
        "completeness_baseline_nontarget.tsv" if nontarget else "completeness_baseline.tsv"
    )
    out_path = Path(output).expanduser() if output else run_dir / default_name
    baseline_path = (
        Path(output_baseline).expanduser() if output_baseline else run_dir / baseline_default
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    total_prompts, seq_len = _run(
        spd_model=spd_model,
        config_sampling=config.sampling,
        iterator=iterator,
        n_to_run=n_to_run,
        alive_rows=alive_rows,
        alive_idx_by_module=alive_idx_by_module,
        weight_deltas=weight_deltas,
        device=device,
        ci_threshold=ci_threshold,
        out_path=out_path,
        baseline_path=baseline_path,
    )
    logger.info(
        f"Saw {total_prompts} prompts of {seq_len} positions each "
        f"(total: {total_prompts * seq_len})"
    )
    logger.info(f"Saved {out_path}")
    logger.info(f"Saved {baseline_path}")
    return out_path, baseline_path


def _run(
    *,
    spd_model: ComponentModel,
    config_sampling: Any,
    iterator: Iterator[Tensor],
    n_to_run: int,
    alive_rows: list[tuple[str, int, str, int]],
    alive_idx_by_module: dict[str, list[int]],
    weight_deltas: dict[str, Tensor],
    device: torch.device,
    ci_threshold: float,
    out_path: Path,
    baseline_path: Path,
) -> tuple[int, int]:
    """Stream one TSV row per (alive component, prompt, pos) with ci > threshold, plus a
    separate baseline TSV with one row per (prompt, pos) in iteration order."""
    total_prompts = 0
    seq_len = 0
    batches_done = 0

    masks_alive: dict[str, Tensor] | None = None
    masks_all: dict[str, Tensor] | None = None
    mask_infos_alive: Any = None
    mask_infos_all: Any = None
    first_shape: tuple[int, ...] | None = None

    with (
        out_path.open("w", newline="") as f,
        baseline_path.open("w", newline="") as fb,
        torch.no_grad(),
    ):
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        baseline_writer = csv.DictWriter(fb, fieldnames=BASELINE_FIELDS, delimiter="\t")
        baseline_writer.writeheader()

        for batch_idx, batch in zip(tqdm(range(n_to_run), desc="batches"), iterator, strict=False):
            batches_done = batch_idx + 1
            assert batch.ndim == 2, f"Expected (batch, seq), got {tuple(batch.shape)}"
            batch_size, seq_len = batch.shape
            if masks_alive is None:
                first_shape = (batch_size, seq_len)
                masks_alive = _build_component_masks(
                    spd_model.module_to_c,
                    alive_idx_by_module,
                    first_shape,
                    device,
                    only_alive=True,
                )
                masks_all = _build_component_masks(
                    spd_model.module_to_c,
                    alive_idx_by_module,
                    first_shape,
                    device,
                    only_alive=False,
                )
                delta_mask_on = torch.ones(first_shape, device=device)
                delta_on = {name: (weight_deltas[name], delta_mask_on) for name in weight_deltas}
                mask_infos_alive = make_mask_infos(masks_alive, weight_deltas_and_masks=None)
                mask_infos_all = make_mask_infos(masks_all, weight_deltas_and_masks=delta_on)
            assert (batch_size, seq_len) == first_shape, (
                f"Batch shape changed mid-pass: got {(batch_size, seq_len)}, expected {first_shape}"
            )
            assert masks_all is not None  # for type narrowing
            total_prompts += batch_size
            prompt_offset = batch_idx * batch_size

            orig_out = spd_model(batch, cache_type="input")
            orig_logits = orig_out.output
            ci_outputs = spd_model.calc_causal_importances(
                pre_weight_acts=orig_out.cache, sampling=config_sampling
            )
            orig_log_probs = F.log_softmax(orig_logits, dim=-1)
            orig_probs = orig_log_probs.exp()

            # Baseline for the circuit: alive components on (no component ablated), delta off.
            # Case b with no ablation is KL=0 by construction (components + delta = target), so
            # only case a needs a baseline.
            logits_circuit_baseline = spd_model(batch, mask_infos=mask_infos_alive)
            assert isinstance(logits_circuit_baseline, Tensor)
            kl_circuit_baseline = (
                orig_probs * (orig_log_probs - F.log_softmax(logits_circuit_baseline, dim=-1))
            ).sum(dim=-1)
            kl_circuit_baseline_cpu = kl_circuit_baseline.cpu().tolist()
            for b in range(batch_size):
                for t in range(seq_len):
                    baseline_writer.writerow({"kl_circuit_baseline": kl_circuit_baseline_cpu[b][t]})

            for module_name, layer, matrix, component in tqdm(
                alive_rows, desc=f"components (batch {batch_idx})", leave=False
            ):
                mask_a = masks_alive[module_name]
                mask_b = masks_all[module_name]
                mask_a[..., component] = 0.0
                mask_b[..., component] = 0.0
                try:
                    logits_a = spd_model(batch, mask_infos=mask_infos_alive)
                    logits_b = spd_model(batch, mask_infos=mask_infos_all)
                finally:
                    mask_a[..., component] = 1.0
                    mask_b[..., component] = 1.0
                assert isinstance(logits_a, Tensor) and isinstance(logits_b, Tensor)

                kl_a = (orig_probs * (orig_log_probs - F.log_softmax(logits_a, dim=-1))).sum(dim=-1)
                kl_b = (orig_probs * (orig_log_probs - F.log_softmax(logits_b, dim=-1))).sum(dim=-1)
                ci = ci_outputs.lower_leaky[module_name][..., component]
                keep = ci > ci_threshold
                if not keep.any():
                    continue

                kl_a_cpu = kl_a.cpu().tolist()
                kl_b_cpu = kl_b.cpu().tolist()
                ci_cpu = ci.cpu().tolist()
                keep_cpu = keep.cpu().tolist()
                for b in range(batch_size):
                    for t in range(seq_len):
                        if not keep_cpu[b][t]:
                            continue
                        writer.writerow(
                            {
                                "layer": layer,
                                "matrix": matrix,
                                "component": component,
                                "prompt": prompt_offset + b,
                                "pos": t,
                                "kl_circuit": kl_a_cpu[b][t],
                                "kl_all": kl_b_cpu[b][t],
                                "ci": ci_cpu[b][t],
                            }
                        )

    if batches_done < n_to_run:
        logger.warning(
            f"Loader exhausted after {batches_done}/{n_to_run} batches (dataset too small?)"
        )
    return total_prompts, seq_len


if __name__ == "__main__":
    fire.Fire(completeness)
