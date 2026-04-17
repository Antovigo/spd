"""Rank (component_A, component_B) pairs for U-vector swapping between two prompts.

Usage:
    python -m spd.scripts.validation.find_swap_candidates <model_path> \
        <effect_target_tsv> <effect_nontarget_tsv> \
        --task-a='{"prompt": "import numpy as", "target": " np"}' \
        --task-b='{"prompt": "import pandas as", "target": " pd"}' \
        [--top-k=20] [--quantile=0.99] [--prompts=PATH] [--output=PATH]
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import LMTaskConfig
from spd.log import logger
from spd.models.component_model import SPDRunInfo
from spd.spd_types import ModelPath

ComponentKey = tuple[int, str, int]  # (layer, matrix, component)
MatrixKey = tuple[int, str]


@dataclass
class TaskSpec:
    name: str
    prompt_idx: int
    prompt_token_ids: list[int]
    last_pos: int
    target_token_id: int


@dataclass
class TargetRow:
    importance: float  # KL at target position after ablation
    orig_prob: float
    ablated_pred: int
    ablated_prob: float


@dataclass
class NontargetHit:
    task_name: str
    prompt: int
    pos: int
    context_token_ids: list[int]
    orig_pred: int
    orig_prob: float


def _resolve_task(
    name: str,
    raw: Any,
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
) -> tuple[TaskSpec, str, str]:
    """Parse a `--task-*` arg, locate its prompt, and tokenise its target.

    Returns (spec, prompt_text, target_text); the text strings are only used for logging.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    assert isinstance(data, dict) and "prompt" in data and "target" in data, (
        f"--task-{name} must be a JSON dict with 'prompt' and 'target' keys, got {raw!r}"
    )
    prompt_text, target_text = str(data["prompt"]), str(data["target"])

    matches = [i for i, p in enumerate(prompts) if p == prompt_text]
    assert len(matches) == 1, (
        f"Task {name}: expected exactly one prompt matching {prompt_text!r} in the prompts file, "
        f"found {len(matches)}"
    )

    prompt_encoded: Any = tokenizer(prompt_text)  # pyright: ignore[reportCallIssue]
    prompt_ids: list[int] = prompt_encoded["input_ids"]

    target_encoded: Any = tokenizer(target_text, add_special_tokens=False)  # pyright: ignore[reportCallIssue]
    target_ids: list[int] = target_encoded["input_ids"]
    assert len(target_ids) == 1, (
        f"Task {name}: target {target_text!r} must tokenize to exactly one token, got {target_ids}"
    )

    spec = TaskSpec(
        name=name,
        prompt_idx=matches[0],
        prompt_token_ids=prompt_ids,
        last_pos=len(prompt_ids) - 1,
        target_token_id=target_ids[0],
    )
    return spec, prompt_text, target_text


def _read_target_rows(
    effect_target_path: Path, task_a: TaskSpec, task_b: TaskSpec
) -> tuple[dict[ComponentKey, TargetRow], dict[ComponentKey, TargetRow]]:
    """Extract the (prompt, pos) row for each component under each task."""
    rows_a: dict[ComponentKey, TargetRow] = {}
    rows_b: dict[ComponentKey, TargetRow] = {}

    with effect_target_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in tqdm(reader, desc="target TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            for task, bucket in ((task_a, rows_a), (task_b, rows_b)):
                if prompt != task.prompt_idx or pos != task.last_pos:
                    continue
                orig_pred = int(record["orig_pred"])
                assert orig_pred == task.target_token_id, (
                    f"Task {task.name}: row at prompt={prompt}, pos={pos} has orig_pred="
                    f"{orig_pred}, but target_token_id={task.target_token_id}. "
                    "The original model doesn't predict the target token — aborting."
                )
                key: ComponentKey = (
                    int(record["layer"]),
                    record["matrix"],
                    int(record["component"]),
                )
                bucket[key] = TargetRow(
                    importance=float(record["kl"]),
                    orig_prob=float(record["orig_prob"]),
                    ablated_pred=int(record["ablated_pred"]),
                    ablated_prob=float(record["ablated_prob"]),
                )
    return rows_a, rows_b


def _contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


_CONTEXT_SIZE = 5


def _scan_nontarget(
    effect_nontarget_path: Path, task_a: TaskSpec, task_b: TaskSpec, quantile: float
) -> tuple[dict[ComponentKey, float], set[int], dict[tuple[int, int], NontargetHit]]:
    """Single pass: per-component KL quantile (excluding target-containing prompts), excluded set, alerts.

    effect_of_ablation writes rows in the order `(batch, component, prompt, pos)`, so all
    positions of a given `(component, prompt)` block are contiguous, and the *first* component's
    block for a prompt sees every one of its positions before any later component does. That lets
    us decide exclusion for a prompt the moment its first block ends, then either fold the block's
    KL values into the global per-component collection or discard them — never needing a second
    pass. Alerts (positions where `orig_pred` is one of the target tokens) are collected in the
    same pass, deduplicated by `(prompt, pos)`.
    """
    values_by_key: dict[ComponentKey, list[float]] = {}
    tokens_per_prompt: dict[int, dict[int, int]] = {}
    excluded: set[int] = set()
    excluded_status: dict[int, bool] = {}
    hits: dict[tuple[int, int], NontargetHit] = {}
    target_to_task_name = {
        task_a.target_token_id: task_a.name,
        task_b.target_token_id: task_b.name,
    }

    current_prompt: int | None = None
    current_key: ComponentKey | None = None
    block_values: list[float] = []

    def _finalize(prompt: int, key: ComponentKey, block_values: list[float]) -> None:
        if prompt not in excluded_status:
            pos_to_tok = tokens_per_prompt[prompt]
            seq = [pos_to_tok[p] for p in sorted(pos_to_tok)]
            is_excl = _contains_subsequence(seq, task_a.prompt_token_ids) or _contains_subsequence(
                seq, task_b.prompt_token_ids
            )
            excluded_status[prompt] = is_excl
            if is_excl:
                excluded.add(prompt)
        if excluded_status[prompt]:
            return
        values_by_key.setdefault(key, []).extend(block_values)

    with effect_nontarget_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for record in tqdm(reader, desc="nontarget TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            key: ComponentKey = (
                int(record["layer"]),
                record["matrix"],
                int(record["component"]),
            )

            if (prompt, key) != (current_prompt, current_key):
                if current_prompt is not None and current_key is not None:
                    _finalize(current_prompt, current_key, block_values)
                current_prompt = prompt
                current_key = key
                block_values = []

            tokens_per_prompt.setdefault(prompt, {})[pos] = int(record["token"])
            block_values.append(float(record["kl"]))

            pos_key = (prompt, pos)
            if pos_key in hits:
                continue
            orig_pred = int(record["orig_pred"])
            matched_task = target_to_task_name.get(orig_pred)
            if matched_task is None:
                continue
            pos_to_tok = tokens_per_prompt[prompt]
            context_ids = [
                pos_to_tok[p] for p in range(max(0, pos - _CONTEXT_SIZE), pos) if p in pos_to_tok
            ]
            hits[pos_key] = NontargetHit(
                task_name=matched_task,
                prompt=prompt,
                pos=pos,
                context_token_ids=context_ids,
                orig_pred=orig_pred,
                orig_prob=float(record["orig_prob"]),
            )

    if current_prompt is not None and current_key is not None:
        _finalize(current_prompt, current_key, block_values)

    quantile_kl = {
        k: float(np.quantile(np.asarray(vs, dtype=np.float32), quantile))
        for k, vs in values_by_key.items()
    }
    return quantile_kl, excluded, hits


def _decode_pred(token_id: int, tokenizer: PreTrainedTokenizer) -> str:
    decoded: str = tokenizer.decode([token_id])  # pyright: ignore[reportAttributeAccessIssue]
    return f"{token_id}:{decoded!r}"


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


def _write_pairs(pairs: list[PairScore], tokenizer: PreTrainedTokenizer, out_path: Path) -> None:
    fieldnames = [
        "rank",
        "layer",
        "matrix",
        "a_component",
        "b_component",
        "a_importance",
        "a_orig_prob",
        "a_ablated_pred",
        "a_ablated_prob",
        "b_importance",
        "b_orig_prob",
        "b_ablated_pred",
        "b_ablated_prob",
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
                    "a_ablated_pred": _decode_pred(p.a_row.ablated_pred, tokenizer),
                    "a_ablated_prob": p.a_row.ablated_prob,
                    "b_importance": p.b_row.importance,
                    "b_orig_prob": p.b_row.orig_prob,
                    "b_ablated_pred": _decode_pred(p.b_row.ablated_pred, tokenizer),
                    "b_ablated_prob": p.b_row.ablated_prob,
                    "a_nontarget_quantile_kl": p.a_nontarget_quantile_kl,
                    "b_nontarget_quantile_kl": p.b_nontarget_quantile_kl,
                    "combined_score": p.combined_score,
                }
            )


def find_swap_candidates(
    model_path: ModelPath,
    effect_target_path: str,
    effect_nontarget_path: str,
    task_a: str | dict[str, Any],
    task_b: str | dict[str, Any],
    top_k: int = 20,
    quantile: float = 0.99,
    prompts: str | None = None,
    output: str | None = None,
) -> Path:
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    run_dir = run_info.checkpoint_path.parent

    assert isinstance(config.task_config, LMTaskConfig) and config.task_config.prompts_file, (
        "find_swap_candidates requires a prompts-based LM task_config"
    )
    assert config.tokenizer_name is not None, "config.tokenizer_name is required"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompts_path = Path(prompts or config.task_config.prompts_file).expanduser()
    prompt_texts = [ln.strip() for ln in prompts_path.read_text().splitlines() if ln.strip()]

    spec_a, _, a_target = _resolve_task("A", task_a, prompt_texts, tokenizer)
    spec_b, _, b_target = _resolve_task("B", task_b, prompt_texts, tokenizer)
    logger.info(
        f"Task A: prompt_idx={spec_a.prompt_idx} last_pos={spec_a.last_pos} "
        f"target_id={spec_a.target_token_id} ({a_target!r})"
    )
    logger.info(
        f"Task B: prompt_idx={spec_b.prompt_idx} last_pos={spec_b.last_pos} "
        f"target_id={spec_b.target_token_id} ({b_target!r})"
    )

    effect_target = Path(effect_target_path).expanduser()
    effect_nontarget = Path(effect_nontarget_path).expanduser()

    task_a_rows, task_b_rows = _read_target_rows(effect_target, spec_a, spec_b)
    assert task_a_rows, f"No target-ablation rows found for task A (prompt_idx={spec_a.prompt_idx})"
    assert task_b_rows, f"No target-ablation rows found for task B (prompt_idx={spec_b.prompt_idx})"
    logger.info(
        f"Found {len(task_a_rows)} task-A rows and {len(task_b_rows)} task-B rows in the target TSV"
    )

    assert 0.0 < quantile < 1.0, f"--quantile must be in (0, 1), got {quantile}"
    logger.info(f"Side-effect score: per-component KL quantile at q={quantile}")
    side_effects, excluded_prompts, hits = _scan_nontarget(
        effect_nontarget, spec_a, spec_b, quantile
    )
    if excluded_prompts:
        print(
            f"[nontarget-monitor] excluding {len(excluded_prompts)} nontarget prompt(s) that "
            f"contain a target prompt as a token sub-sequence: {sorted(excluded_prompts)}"
        )
    else:
        print("[nontarget-monitor] no nontarget prompts contain a target prompt as a sub-sequence.")

    if hits:
        print(
            f"[nontarget-monitor] {len(hits)} nontarget position(s) where the original model "
            f"predicts one of the target tokens:"
        )
        for hit in sorted(hits.values(), key=lambda h: (h.prompt, h.pos)):
            ctx = tokenizer.decode(hit.context_token_ids)  # pyright: ignore[reportAttributeAccessIssue]
            orig_tok = tokenizer.decode([hit.orig_pred])  # pyright: ignore[reportAttributeAccessIssue]
            in_excluded = " [in-excluded-prompt]" if hit.prompt in excluded_prompts else ""
            print(
                f"[nontarget-hit] task={hit.task_name} prompt={hit.prompt} pos={hit.pos} "
                f"{ctx!r} -> orig: {orig_tok!r} ({hit.orig_prob:.3f}){in_excluded}"
            )
    else:
        print("[nontarget-monitor] no nontarget positions predict a target token.")

    pairs = _rank_pairs(task_a_rows, task_b_rows, side_effects)
    logger.info(f"Built {len(pairs)} candidate pairs; keeping top {top_k}")
    pairs = pairs[:top_k]

    out_path = Path(output).expanduser() if output else run_dir / "swap_candidates.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pairs(pairs, tokenizer, out_path)
    logger.info(f"Wrote {len(pairs)} candidate pairs to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(find_swap_candidates)
