"""Rank (component_A, component_B) pairs for U-vector swapping between two prompts.

Reads:
- target KL + orig-predictions files (produced by `effect_of_ablation`) to score each task's
  per-component importance;
- a nontarget summary TSV (produced by `summarize_nontarget`) that carries the precomputed
  per-component KL quantile — the side-effect score.

Usage:
    python -m spd.scripts.validation.find_swap_candidates <model_path> \
        <effect_target_kl_tsv> <effect_target_orig_tsv> <nontarget_summary_tsv> \
        --task-a='{"prompt": "import numpy as", "target": " np"}' \
        --task-b='{"prompt": "import pandas as", "target": " pd"}' \
        [--top-k=20] [--prompts=PATH] [--output=PATH]
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.scripts.validation.common import (
    TaskSpec,
    load_prompts,
    resolve_task,
)
from spd.spd_types import ModelPath

ComponentKey = tuple[int, str, int]  # (layer, matrix, component)
MatrixKey = tuple[int, str]


@dataclass
class TargetRow:
    importance: float  # KL at task position after ablation
    orig_prob: float


@dataclass
class PairScore:
    layer: int
    matrix: str
    a_component: int
    b_component: int
    a_row: TargetRow
    b_row: TargetRow
    a_nontarget_quantile_kl: float
    b_nontarget_quantile_kl: float
    combined_score: float


def _read_orig_at_task_position(orig_path: Path, task: TaskSpec) -> float:
    """Return the original model's probability for task's target token at its task position.

    Asserts the model actually predicts the target token there.
    """
    with orig_path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            if int(record["prompt"]) != task.prompt_idx or int(record["pos"]) != task.last_pos:
                continue
            orig_pred = int(record["orig_pred"])
            assert orig_pred == task.target_token_id, (
                f"Task {task.name}: orig_predictions row at prompt={task.prompt_idx}, "
                f"pos={task.last_pos} has orig_pred={orig_pred}, but target_token_id="
                f"{task.target_token_id}. The original model doesn't predict the target token "
                "— aborting."
            )
            return float(record["orig_prob"])
    raise AssertionError(
        f"Task {task.name}: no row for (prompt={task.prompt_idx}, pos={task.last_pos}) in "
        f"{orig_path}"
    )


def _read_target_rows(
    kl_path: Path, orig_path: Path, task_a: TaskSpec, task_b: TaskSpec
) -> tuple[dict[ComponentKey, TargetRow], dict[ComponentKey, TargetRow]]:
    """Extract the (prompt, pos) row for each component under each task."""
    orig_prob_a = _read_orig_at_task_position(orig_path, task_a)
    orig_prob_b = _read_orig_at_task_position(orig_path, task_b)

    rows_a: dict[ComponentKey, TargetRow] = {}
    rows_b: dict[ComponentKey, TargetRow] = {}
    with kl_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="target kl TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            for task, bucket, orig_prob in (
                (task_a, rows_a, orig_prob_a),
                (task_b, rows_b, orig_prob_b),
            ):
                if prompt != task.prompt_idx or pos != task.last_pos:
                    continue
                key: ComponentKey = (
                    int(record["layer"]),
                    record["matrix"],
                    int(record["component"]),
                )
                bucket[key] = TargetRow(
                    importance=float(record["kl"]),
                    orig_prob=orig_prob,
                )
    return rows_a, rows_b


def _read_nontarget_summary(summary_path: Path) -> dict[ComponentKey, float]:
    side_effects: dict[ComponentKey, float] = {}
    with summary_path.open() as f:
        for record in csv.DictReader(f, delimiter="\t"):
            key: ComponentKey = (
                int(record["layer"]),
                record["matrix"],
                int(record["component"]),
            )
            side_effects[key] = float(record["quantile_kl"])
    return side_effects


def _rank_pairs(
    task_a_rows: dict[ComponentKey, TargetRow],
    task_b_rows: dict[ComponentKey, TargetRow],
    side_effects: dict[ComponentKey, float],
) -> list[PairScore]:
    by_matrix_a: dict[MatrixKey, list[tuple[int, TargetRow]]] = {}
    by_matrix_b: dict[MatrixKey, list[tuple[int, TargetRow]]] = {}
    for (layer, matrix, component), row in task_a_rows.items():
        by_matrix_a.setdefault((layer, matrix), []).append((component, row))
    for (layer, matrix, component), row in task_b_rows.items():
        by_matrix_b.setdefault((layer, matrix), []).append((component, row))

    pairs: list[PairScore] = []
    for matrix_key, a_items in by_matrix_a.items():
        if matrix_key not in by_matrix_b:
            continue
        b_items = by_matrix_b[matrix_key]
        layer, matrix = matrix_key
        for a_component, a_row in a_items:
            a_side = side_effects.get((layer, matrix, a_component), 0.0)
            for b_component, b_row in b_items:
                if a_component == b_component:
                    continue
                b_side = side_effects.get((layer, matrix, b_component), 0.0)
                mean_side = (a_side + b_side) / 2.0
                score = min(a_row.importance, b_row.importance) / (1e-6 + mean_side)
                pairs.append(
                    PairScore(
                        layer=layer,
                        matrix=matrix,
                        a_component=a_component,
                        b_component=b_component,
                        a_row=a_row,
                        b_row=b_row,
                        a_nontarget_quantile_kl=a_side,
                        b_nontarget_quantile_kl=b_side,
                        combined_score=score,
                    )
                )

    pairs.sort(key=lambda p: p.combined_score, reverse=True)
    return pairs


def _write_pairs(pairs: list[PairScore], out_path: Path) -> None:
    fieldnames = [
        "rank",
        "layer",
        "matrix",
        "a_component",
        "b_component",
        "a_importance",
        "a_orig_prob",
        "b_importance",
        "b_orig_prob",
        "a_nontarget_quantile_kl",
        "b_nontarget_quantile_kl",
        "combined_score",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rank, p in enumerate(pairs, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "layer": p.layer,
                    "matrix": p.matrix,
                    "a_component": p.a_component,
                    "b_component": p.b_component,
                    "a_importance": p.a_row.importance,
                    "a_orig_prob": p.a_row.orig_prob,
                    "b_importance": p.b_row.importance,
                    "b_orig_prob": p.b_row.orig_prob,
                    "a_nontarget_quantile_kl": p.a_nontarget_quantile_kl,
                    "b_nontarget_quantile_kl": p.b_nontarget_quantile_kl,
                    "combined_score": p.combined_score,
                }
            )


def find_swap_candidates(
    model_path: ModelPath,
    effect_target_kl_path: str,
    effect_target_orig_path: str,
    nontarget_summary_path: str,
    task_a: str | dict[str, Any],
    task_b: str | dict[str, Any],
    top_k: int = 20,
    prompts: str | None = None,
    output: str | None = None,
) -> Path:
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    run_dir = run_info.checkpoint_path.parent
    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompt_texts = load_prompts(config, prompts)

    spec_a, _, _ = resolve_task("A", task_a, prompt_texts, tokenizer)
    spec_b, _, _ = resolve_task("B", task_b, prompt_texts, tokenizer)

    target_kl = Path(effect_target_kl_path).expanduser()
    target_orig = Path(effect_target_orig_path).expanduser()
    summary_path = Path(nontarget_summary_path).expanduser()

    task_a_rows, task_b_rows = _read_target_rows(target_kl, target_orig, spec_a, spec_b)
    assert task_a_rows, f"No target-ablation rows found for task A (prompt_idx={spec_a.prompt_idx})"
    assert task_b_rows, f"No target-ablation rows found for task B (prompt_idx={spec_b.prompt_idx})"

    side_effects = _read_nontarget_summary(summary_path)

    pairs = _rank_pairs(task_a_rows, task_b_rows, side_effects)
    logger.info(f"Built {len(pairs)} candidate pairs; keeping top {top_k}")
    pairs = pairs[:top_k]

    out_path = Path(output).expanduser() if output else run_dir / "swap_candidates.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pairs(pairs, out_path)
    return out_path


if __name__ == "__main__":
    fire.Fire(find_swap_candidates)
