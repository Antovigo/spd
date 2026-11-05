"""Base and source prompt pairs for DAS (plan section 5), saved as rows of `tokenized.npz`.

    python -m param_decomp.das.build_prompt_pairs --config <das_experiment_config.yaml>

`references.npz` has one row per prompt of the pruning run's problem (true answer >= 1), so its rows
are mapped through `build_problem`'s `pool_rows`, and the tokens are asserted equal on every row."""

import argparse
from pathlib import Path
from typing import get_args

import numpy as np
import yaml

from param_decomp.ci_filter.config import ArithmeticPoolConfig
from param_decomp.ci_filter.pool import Tokenizer
from param_decomp.das.load_das_experiment_config import (
    DasExperimentConfig,
    Task,
    load_das_experiment_config,
)
from param_decomp.pruning.problem import build_problem
from param_decomp.pruning.stats import QUANTITIES

N_OPERAND = 400
TRAIN, VALIDATION, TEST = 0, 1, 2


def sign_of(task: Task) -> int:
    match task:
        case "addition":
            return 1
        case "subtraction":
            return -1


def correct_by_tokenized_row(config: DasExperimentConfig, tokenizer: Tokenizer) -> np.ndarray:
    """`(N,)` bool over the rows of `tokenized.npz`: answer >= 1 and top integer correct."""
    tokens = np.load(config.paths.tokenized_prompts)["tokens"]
    pool = ArithmeticPoolConfig.model_validate(
        yaml.safe_load(config.paths.pruning_run_config.read_text())["pool"]
    )
    problem = build_problem(pool, tokenizer)
    assert np.array_equal(problem.tokens, tokens[problem.pool_rows]), "rows do not align"
    references = np.load(config.paths.correctness_references)["llama"]
    assert references.shape[1] == problem.n, (references.shape, problem.n)
    correct = np.zeros(tokens.shape[0], bool)
    correct[problem.pool_rows] = references[QUANTITIES.index("correct")] > 0.5
    return correct


def split_prompts(
    correct: np.ndarray, train_fraction: float, validation_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """`correct.shape` int8: the split of every correct prompt, -1 elsewhere."""
    fractions = [train_fraction, validation_fraction, 1 - train_fraction - validation_fraction]
    splits = np.full(correct.shape, -1, np.int8)
    splits[correct] = rng.choice(3, int(correct.sum()), p=fractions)
    return splits


def sample_pairs(
    splits: np.ndarray, split: int, sign: int, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`n_pairs` of `(a, b, a', b')`, 1-based operands: all four prompts in `split` (so correct), and
    the answers of `(a, b)`, `(a', b)`, `(a, b')`, `(a', b')` pairwise distinct. `splits` is
    `(N_OPERAND, N_OPERAND)`, indexed by 0-based operands."""
    a_all, b_all = np.nonzero(splits == split)
    found: list[np.ndarray] = []
    n_found = 0
    while n_found < n_pairs:
        i, j = rng.integers(0, a_all.size, (2, 4 * n_pairs + 4096))
        a0, b0, a1, b1 = a_all[i], b_all[i], a_all[j], b_all[j]
        answers = [a + sign * b for a, b in ((a0, b0), (a1, b0), (a0, b1), (a1, b1))]
        keep = (splits[a1, b0] == split) & (splits[a0, b1] == split)
        for x in range(4):
            for y in range(x + 1, 4):
                keep &= answers[x] != answers[y]
        found.append(np.stack([a0[keep], b0[keep], a1[keep], b1[keep]], axis=1))
        n_found += int(keep.sum())
    pairs = np.concatenate(found)[:n_pairs] + 1
    return pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3]


def tokenized_row(task_index: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The row of `tokenized.npz`: per operation, row-major in `(a, b)`, 1-based operands."""
    return task_index * N_OPERAND**2 + (a - 1) * N_OPERAND + (b - 1)


def build_prompt_pairs(config: DasExperimentConfig, tokenizer: Tokenizer) -> None:
    settings = config.prompt_pairs
    correct = correct_by_tokenized_row(config, tokenizer)
    prompts = np.load(config.paths.tokenized_prompts)
    rng = np.random.default_rng(settings.seed)
    config.paths.prompt_pairs.mkdir(parents=True, exist_ok=True)
    tasks: tuple[Task, ...] = get_args(Task.__value__)
    for task_index, task in enumerate(tasks):
        rows = np.arange(N_OPERAND**2) + task_index * N_OPERAND**2
        assert np.array_equal(prompts["a"][rows], np.repeat(np.arange(1, N_OPERAND + 1), N_OPERAND))
        assert np.array_equal(prompts["op"][rows], np.full(rows.size, task_index))
        grid = correct[rows].reshape(N_OPERAND, N_OPERAND)
        splits = split_prompts(
            grid, settings.train_prompt_fraction, settings.validation_prompt_fraction, rng
        )
        base_rows, source_rows, pair_splits = [], [], []
        for split, n_pairs in (
            (TRAIN, settings.train_pairs),
            (VALIDATION, settings.validation_pairs),
            (TEST, settings.test_pairs),
        ):
            a0, b0, a1, b1 = sample_pairs(splits, split, sign_of(task), n_pairs, rng)
            base_rows.append(tokenized_row(task_index, a0, b0))
            source_rows.append(tokenized_row(task_index, a1, b1))
            pair_splits.append(np.full(n_pairs, split, np.int8))
        np.savez(
            config.paths.prompt_pairs / f"{task}_prompt_pairs.npz",
            base_prompt_row=np.concatenate(base_rows),
            source_prompt_row=np.concatenate(source_rows),
            split=np.concatenate(pair_splits),
        )
        print(f"{task}: {int(grid.sum())} correct prompts", flush=True)


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    config = load_das_experiment_config(parser.parse_args().config)
    build_prompt_pairs(config, AutoTokenizer.from_pretrained(str(config.paths.llama_weights)))


if __name__ == "__main__":
    main()
