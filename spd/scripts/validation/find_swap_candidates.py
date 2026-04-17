"""Rank (component_A, component_B) pairs for U-vector swapping between two prompts.

Usage:
    python -m spd.scripts.validation.find_swap_candidates <model_path> \
        <effect_target_kl_tsv> <effect_target_orig_tsv> \
        <effect_nontarget_kl_tsv> <effect_nontarget_orig_tsv> \
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


def _read_orig_at_task_position(orig_path: Path, task: TaskSpec) -> float:
    """Return the original model's probability for task's target token at its task position.

    Also asserts the model actually predicts the target token there (otherwise the task is
    mis-specified — the swap test's importance signal assumes the original model is correct).
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


def _contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


_CONTEXT_SIZE = 5


def _scan_positions(
    positions_path: Path, task_a: TaskSpec, task_b: TaskSpec
) -> tuple[dict[int, list[int]], set[int], dict[tuple[int, int], NontargetHit]]:
    """Read the nontarget positions TSV: per-prompt token sequence, excluded prompt set, alerts."""
    tokens_per_prompt: dict[int, dict[int, int]] = {}
    hits: dict[tuple[int, int], NontargetHit] = {}
    target_to_task_name = {
        task_a.target_token_id: task_a.name,
        task_b.target_token_id: task_b.name,
    }
    with positions_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="positions TSV"):
            prompt = int(record["prompt"])
            pos = int(record["pos"])
            tokens_per_prompt.setdefault(prompt, {})[pos] = int(record["token"])
            orig_pred = int(record["orig_pred"])
            matched_task = target_to_task_name.get(orig_pred)
            if matched_task is None:
                continue
            pos_key = (prompt, pos)
            if pos_key in hits:
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

    prompt_tokens: dict[int, list[int]] = {}
    excluded: set[int] = set()
    for prompt, pos_to_tok in tokens_per_prompt.items():
        seq = [pos_to_tok[p] for p in sorted(pos_to_tok)]
        prompt_tokens[prompt] = seq
        if _contains_subsequence(seq, task_a.prompt_token_ids) or _contains_subsequence(
            seq, task_b.prompt_token_ids
        ):
            excluded.add(prompt)
    return prompt_tokens, excluded, hits


def _scan_nontarget(
    kl_path: Path, positions_path: Path, task_a: TaskSpec, task_b: TaskSpec, quantile: float
) -> tuple[dict[ComponentKey, float], set[int], dict[tuple[int, int], NontargetHit]]:
    """Compute per-component KL quantile, excluded-prompt set, and nontarget-hit alerts.

    The nontarget output of `effect_of_ablation` is split into two files: a positions file (one
    row per prompt × pos) and a kl file (one row per component × prompt × pos). The positions
    file is read first to determine prompt exclusions and alerts; the kl file is then streamed
    and each row's KL is added to its component's accumulator iff the prompt isn't excluded.
    """
    _, excluded, hits = _scan_positions(positions_path, task_a, task_b)

    values_by_key: dict[ComponentKey, list[float]] = {}
    with kl_path.open() as f:
        for record in tqdm(csv.DictReader(f, delimiter="\t"), desc="kl TSV"):
            prompt = int(record["prompt"])
            if prompt in excluded:
                continue
            key: ComponentKey = (
                int(record["layer"]),
                record["matrix"],
                int(record["component"]),
            )
            values_by_key.setdefault(key, []).append(float(record["kl"]))

    quantile_kl = {
        k: float(np.quantile(np.asarray(vs, dtype=np.float32), quantile))
        for k, vs in values_by_key.items()
    }
    return quantile_kl, excluded, hits


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
    effect_nontarget_kl_path: str,
    effect_nontarget_orig_path: str,
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

    effect_target_kl = Path(effect_target_kl_path).expanduser()
    effect_target_orig = Path(effect_target_orig_path).expanduser()
    effect_nontarget_kl = Path(effect_nontarget_kl_path).expanduser()
    effect_nontarget_orig = Path(effect_nontarget_orig_path).expanduser()

    task_a_rows, task_b_rows = _read_target_rows(
        effect_target_kl, effect_target_orig, spec_a, spec_b
    )
    assert task_a_rows, f"No target-ablation rows found for task A (prompt_idx={spec_a.prompt_idx})"
    assert task_b_rows, f"No target-ablation rows found for task B (prompt_idx={spec_b.prompt_idx})"
    logger.info(
        f"Found {len(task_a_rows)} task-A rows and {len(task_b_rows)} task-B rows in the target TSV"
    )

    assert 0.0 < quantile < 1.0, f"--quantile must be in (0, 1), got {quantile}"
    logger.info(f"Side-effect score: per-component KL quantile at q={quantile}")
    side_effects, excluded_prompts, hits = _scan_nontarget(
        effect_nontarget_kl, effect_nontarget_orig, spec_a, spec_b, quantile
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
    _write_pairs(pairs, out_path)
    logger.info(f"Wrote {len(pairs)} candidate pairs to {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(find_swap_candidates)
