"""Is the gap with Feucht et al. at 16 dimensions due to the operand range?

Fits 16 dimensions at the `=` position alone, at addition's stream point before block 18's MLP, for each
variable, with operands up to 100 (theirs) and up to 400 (ours), at their 3,584 and our 14,336 training
pairs. Their IIA there: first operand 61.7%, second operand 63.8%, result 87.6%.

    python notes/das/compare_with_feucht_operand_range.py --config <das_experiment_config.yaml> --out <folder>
"""

import argparse
import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from param_decomp.das.build_prompt_pairs import (
    N_OPERAND,
    TEST,
    TRAIN,
    VALIDATION,
    correct_by_tokenized_row,
    sample_pairs,
    split_prompts,
    tokenized_row,
)
from param_decomp.das.load_das_experiment_config import Variable, load_das_experiment_config
from param_decomp.das.patched_llama_forward import load_llama
from param_decomp.das.train_das_subspaces import (
    SiteFunctions,
    dimension_mask,
    evaluate,
    load_split,
    svd_initialization,
    timed_full_fit,
)
from param_decomp.pruning.problem import integer_token_ids

STREAM_POINT = 37
DIMS = (0, 0, 0, 16)
VARIABLES: tuple[Variable, ...] = ("first_operand", "second_operand", "result")
TRAIN_PAIRS = (3584, 14336)


def build_small_range_pairs(
    correct: np.ndarray, max_operand: int, n: tuple[int, int, int], out: Path
) -> None:
    rng = np.random.default_rng(0)
    grid = correct[: N_OPERAND**2].reshape(N_OPERAND, N_OPERAND).copy()
    grid[max_operand:, :] = False
    grid[:, max_operand:] = False
    print(f"correct prompts with operands up to {max_operand}: {int(grid.sum())}", flush=True)
    splits = split_prompts(grid, 0.6, 0.2, rng)
    base_rows, source_rows, pair_splits = [], [], []
    for split, n_pairs in zip((TRAIN, VALIDATION, TEST), n, strict=True):
        a0, b0, a1, b1 = sample_pairs(splits, split, 1, n_pairs, rng)
        base_rows.append(tokenized_row(0, a0, b0))
        source_rows.append(tokenized_row(0, a1, b1))
        pair_splits.append(np.full(n_pairs, split, np.int8))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "addition_prompt_pairs.npz",
        base_prompt_row=np.concatenate(base_rows),
        source_prompt_row=np.concatenate(source_rows),
        split=np.concatenate(pair_splits),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_das_experiment_config(arguments.config)
    tokenizer = AutoTokenizer.from_pretrained(str(config.paths.llama_weights))
    settings = config.training
    validation_pairs = config.prompt_pairs.validation_pairs
    small_range = arguments.out / "prompt_pairs_operands_up_to_100"
    build_small_range_pairs(
        correct_by_tokenized_row(config, tokenizer),
        100,
        (max(TRAIN_PAIRS), validation_pairs, config.prompt_pairs.test_pairs),
        small_range,
    )
    functions = SiteFunctions(
        load_llama(config.paths.llama_weights),
        STREAM_POINT,
        jnp.asarray(integer_token_ids(tokenizer)),
        settings.learning_rate,
    )
    mask = dimension_mask(DIMS)
    rows: list[dict[str, object]] = []
    for max_operand, pairs_dir in ((100, small_range), (400, config.paths.prompt_pairs)):
        for variable in VARIABLES:
            train = load_split(
                functions, config.paths.tokenized_prompts, pairs_dir, "addition", variable, TRAIN,
                max(TRAIN_PAIRS),
            )  # fmt: skip
            validation = load_split(
                functions, config.paths.tokenized_prompts, pairs_dir, "addition", variable,
                VALIDATION, validation_pairs,
            )  # fmt: skip
            initial_basis, _ = svd_initialization(train, settings.svd_pairs)
            for n_train in TRAIN_PAIRS:
                subset = train.head(n_train)
                basis, seconds, steps = timed_full_fit(
                    functions, initial_basis, DIMS, subset, settings.batch_size, settings.epochs, 0
                )
                held_out = evaluate(functions, basis, mask, validation)
                rows.append(
                    {
                        "max_operand": max_operand,
                        "variable": variable,
                        "train_pairs": n_train,
                        "validation_iia": held_out.iia,
                        "validation_full_vocabulary_iia": held_out.full_vocabulary_iia,
                        "train_iia": evaluate(functions, basis, mask, subset.head(3584)).iia,
                        "steps": steps,
                        "seconds": seconds,
                    }
                )
                print(rows[-1], flush=True)
    with (arguments.out / "iia_at_16_dimensions_by_operand_range.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("comparison finished", flush=True)


if __name__ == "__main__":
    main()
