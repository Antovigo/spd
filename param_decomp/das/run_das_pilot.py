"""The DAS pilot (plan section 7.1), one site per array task.

    python -m param_decomp.das.run_das_pilot --config <yaml> --stage fit_quality --site <index>
    python -m param_decomp.das.run_das_pilot --config <yaml> --stage search --site <index>
    python -m param_decomp.das.run_das_pilot --config <yaml> --stage summary

`fit_quality` sweeps the learning rate and the training pairs at 64 dimensions. `search` compares
pruning with fresh fits at the settings chosen from that sweep and written in the config. Each file
under the site's folder appears only when its arm is complete, so a restarted task skips finished arms."""

import argparse
import csv
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.das.build_prompt_pairs import TRAIN, VALIDATION
from param_decomp.das.load_das_experiment_config import (
    DasExperimentConfig,
    PilotSite,
    load_das_experiment_config,
)
from param_decomp.das.patched_llama_forward import Llama, load_llama, logits_from, stream_at
from param_decomp.das.prune_das_subspaces import (
    Schedule,
    SearchRound,
    fresh_fit_search,
    prune_search,
    rank_positions,
)
from param_decomp.das.train_das_subspaces import (
    EVALUATION_CHUNK,
    PairStreams,
    SiteFunctions,
    dimension_mask,
    evaluate,
    load_split,
    svd_initialization,
    timed_full_fit,
)
from param_decomp.pruning.problem import integer_token_ids

type Stage = Literal["fit_quality", "search", "drop_then_bisect", "summary"]
CORRECTNESS_SAMPLE = 1024
MIN_CORRECTNESS_AGREEMENT = 0.99
TRAIN_IIA_PAIRS = 3584


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write through a temporary name, so the file exists only when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def start_dims(
    config: DasExperimentConfig, site: PilotSite, differs: np.ndarray
) -> tuple[int, ...]:
    """64 dimensions at every patched position with a base-source difference, 0 elsewhere."""
    patched = config.patched_positions[site.variable]
    return tuple(
        64 if position + 1 in patched and bool(differs[position]) else 0 for position in range(4)
    )


def check_correctness(config: DasExperimentConfig, model: Llama, integer_tokens: jax.Array) -> None:
    """The forward reproduces the reference correctness on a sample of each task's prompts."""
    from transformers import AutoTokenizer

    from param_decomp.das.build_prompt_pairs import correct_by_tokenized_row

    tokenizer = AutoTokenizer.from_pretrained(str(config.paths.llama_weights))
    correct = correct_by_tokenized_row(config, tokenizer)
    prompts = np.load(config.paths.tokenized_prompts)
    rng = np.random.default_rng(0)
    forward = jax.jit(
        lambda weights, tokens: logits_from(weights, stream_at(weights, tokens, 1), 1)
    )
    for op, name in enumerate(("addition", "subtraction")):
        sign = 1 if op == 0 else -1
        rows = rng.choice(
            np.flatnonzero((prompts["op"] == op) & (prompts["a"] + sign * prompts["b"] >= 1)),
            CORRECTNESS_SAMPLE,
            replace=False,
        )
        answers = prompts["a"][rows] + sign * prompts["b"][rows]
        hits = []
        for s in range(0, CORRECTNESS_SAMPLE, EVALUATION_CHUNK):
            chunk = jnp.asarray(prompts["tokens"][rows[s : s + EVALUATION_CHUNK]])
            hits.append(np.asarray(jnp.argmax(forward(model, chunk)[:, integer_tokens], -1)))
        agreement = float(((np.concatenate(hits) == answers) == correct[rows]).mean())
        print(f"correctness agreement with references, {name}: {agreement:.4f}", flush=True)
        assert agreement >= MIN_CORRECTNESS_AGREEMENT, (name, agreement)


def run_fit_quality(
    config: DasExperimentConfig,
    site: PilotSite,
    site_dir: Path,
    build: Callable[[float], SiteFunctions],
) -> None:
    path = site_dir / "hyperparameter_and_pair_count_sweep.csv"
    if path.exists():
        print(f"{path} exists; nothing to do", flush=True)
        return
    rows: list[dict[str, object]] = []
    settings = config.training
    max_pairs = max(config.pilot.train_pairs)
    loaded: dict[str, PairStreams] = {}
    initial_basis, differs = None, None
    for learning_rate in config.pilot.learning_rates:
        functions = build(learning_rate)
        if not loaded:
            loaded["train"] = load_split(
                functions, config.paths.tokenized_prompts, config.paths.prompt_pairs,
                site.task, site.variable, TRAIN, max_pairs,
            )  # fmt: skip
            loaded["validation"] = load_split(
                functions, config.paths.tokenized_prompts, config.paths.prompt_pairs,
                site.task, site.variable, VALIDATION, config.prompt_pairs.validation_pairs,
            )  # fmt: skip
            initial_basis, differs = svd_initialization(loaded["train"], settings.svd_pairs)
        assert initial_basis is not None and differs is not None
        dims = start_dims(config, site, np.asarray(differs))
        mask = dimension_mask(dims)
        for n_train in config.pilot.train_pairs:
            train = loaded["train"].head(n_train)
            basis, seconds, steps = timed_full_fit(
                functions, initial_basis, dims, train, settings.batch_size, settings.epochs, 0
            )
            validation = evaluate(functions, basis, mask, loaded["validation"])
            training = evaluate(functions, basis, mask, train.head(TRAIN_IIA_PAIRS))
            rows.append(
                {
                    "site": site.name,
                    "learning_rate": learning_rate,
                    "train_pairs": n_train,
                    "validation_iia": validation.iia,
                    "validation_full_vocabulary_iia": validation.full_vocabulary_iia,
                    "validation_cross_entropy": validation.cross_entropy,
                    "train_iia": training.iia,
                    "steps": steps,
                    "seconds": seconds,
                }
            )
            print(rows[-1], flush=True)
    write_rows(path, rows)


def round_row(site: PilotSite, arm: str, entry: SearchRound) -> dict[str, object]:
    fields = dataclasses.asdict(entry)
    dims = fields.pop("dims")
    return {
        "site": site.name,
        "arm": arm,
        **fields,
        **{f"k_position_{p + 1}": dims[p] for p in range(4)},
    }


def run_search(
    config: DasExperimentConfig,
    site: PilotSite,
    site_dir: Path,
    build: Callable[[float], SiteFunctions],
    only_drop_then_bisect: bool,
) -> None:
    settings = config.training
    functions = build(settings.learning_rate)
    train = load_split(
        functions, config.paths.tokenized_prompts, config.paths.prompt_pairs,
        site.task, site.variable, TRAIN, settings.train_pairs,
    )  # fmt: skip
    validation = load_split(
        functions, config.paths.tokenized_prompts, config.paths.prompt_pairs,
        site.task, site.variable, VALIDATION, config.prompt_pairs.validation_pairs,
    )  # fmt: skip
    scoring = train.head(config.pruning.prune_score_pairs)
    initial_basis, differs = svd_initialization(train, settings.svd_pairs)
    dims = start_dims(config, site, np.asarray(differs))
    start_path = site_dir / "start_64_dims.npz"
    site_dir.mkdir(parents=True, exist_ok=True)
    if start_path.exists():
        saved = np.load(start_path)
        start_basis, seconds, steps = (
            jnp.asarray(saved["basis"]),
            float(saved["seconds"]),
            int(saved["steps"]),
        )
    else:
        start_basis, seconds, steps = timed_full_fit(
            functions, initial_basis, dims, train, settings.batch_size, settings.epochs, 0
        )
        temporary = site_dir / "start_64_dims.partial.npz"
        np.savez(temporary, basis=np.asarray(start_basis), seconds=seconds, steps=steps)
        temporary.replace(start_path)
    reference = evaluate(functions, start_basis, dimension_mask(dims), validation)
    print(f"reference validation IIA {reference.iia:.4f}, fit seconds {seconds:.0f}", flush=True)
    if reference.iia < config.pruning.min_iia_for_position_search:
        print(f"reference IIA below the search threshold; skipping {site.name}", flush=True)
        return
    active = tuple(p for p in range(4) if dims[p] > 0)
    order = rank_positions(functions, start_basis, dims, active, scoring)
    print(f"position order {[p + 1 for p in order]}", flush=True)
    margin = config.pruning.position_margin

    def run_arm(
        name: str, search: Callable[[Callable[[SearchRound], None]], tuple[int, ...]]
    ) -> None:
        path = site_dir / f"{name}.csv"
        if path.exists():
            return
        rows: list[dict[str, object]] = []

        def report(entry: SearchRound) -> None:
            rows.append(round_row(site, name, entry))
            print(rows[-1], flush=True)

        started = time.monotonic()
        final = search(report)
        rows.append(
            round_row(
                site,
                name,
                SearchRound(
                    position=0,
                    dims=final,
                    validation_iia=reference.iia,
                    validation_full_vocabulary_iia=reference.full_vocabulary_iia,
                    validation_cross_entropy=reference.cross_entropy,
                    passed=True,
                    training_steps=0,
                    seconds=time.monotonic() - started,
                ),
            )
        )
        write_rows(path, rows)
        print(f"{name}: final dimensions {final}", flush=True)

    if not only_drop_then_bisect:
        run_arm(
            "fresh_fits",
            lambda report: fresh_fit_search(
                functions, initial_basis, dims, order, config.pruning.fresh_fit_ladder, train,
                validation, settings.batch_size, settings.epochs, margin, reference.iia, 0, report,
            ),
        )  # fmt: skip

    def prune_arm(
        schedule: Schedule, finetune_epochs: int
    ) -> Callable[[Callable[[SearchRound], None]], tuple[int, ...]]:
        def search(report: Callable[[SearchRound], None]) -> tuple[int, ...]:
            return prune_search(
                functions, start_basis, dims, order, train, validation, scoring, schedule,
                finetune_epochs * train.n // settings.batch_size, settings.batch_size,
                margin, reference.iia, 0, report,
            )[1]  # fmt: skip

        return search

    if only_drop_then_bisect:
        run_arm("pruning_drop_then_bisect_1_epochs", prune_arm("drop_then_bisect", 1))
        return
    for epochs in config.pilot.finetune_epochs:
        run_arm(f"pruning_halve_then_single_steps_{epochs}_epochs",
                prune_arm("halve_then_single_steps", epochs))  # fmt: skip
    if site.task == "addition":
        run_arm("pruning_single_steps_1_epochs", prune_arm("single_steps", 1))


def summarize(config: DasExperimentConfig) -> None:
    """Print the sweep and each arm's final dimensions, from the files present."""
    for site in config.pilot.sites:
        site_dir = config.paths.pilot / site.name
        print(f"== {site.name}")
        for path in sorted(site_dir.glob("*.csv")):
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if path.name.startswith("hyperparameter"):
                for row in rows:
                    print(
                        f"  lr {row['learning_rate']} pairs {row['train_pairs']}: "
                        f"validation IIA {float(row['validation_iia']):.3f}, "
                        f"train IIA {float(row['train_iia']):.3f}, {float(row['seconds']):.0f} s"
                    )
                continue
            final = rows[-1]
            dims = [int(final[f"k_position_{p}"]) for p in (1, 2, 3, 4)]
            passed = [row for row in rows[:-1] if row["passed"] == "True"]
            best = max((float(row["validation_iia"]) for row in rows[:-1]), default=float("nan"))
            print(
                f"  {path.stem}: k {dims}, sum {sum(dims)}, reference IIA "
                f"{float(final['validation_iia']):.3f}, best round IIA {best:.3f}, "
                f"{len(rows) - 1} rounds ({len(passed)} passed), {float(final['seconds']):.0f} s"
            )


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("fit_quality", "search", "drop_then_bisect", "summary"), required=True
    )
    parser.add_argument("--site", type=int)
    arguments = parser.parse_args()
    config = load_das_experiment_config(arguments.config)
    stage: Stage = arguments.stage
    if stage == "summary":
        summarize(config)
        return
    site = config.pilot.sites[arguments.site]
    print(f"DAS pilot {stage}, site {site.name}, devices {jax.devices()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(config.paths.llama_weights))
    integer_tokens = jnp.asarray(integer_token_ids(tokenizer))
    model = load_llama(config.paths.llama_weights)
    if arguments.site == 0 and stage == "fit_quality":
        check_correctness(config, model, integer_tokens)

    def build(learning_rate: float) -> SiteFunctions:
        return SiteFunctions(model, site.stream_point, integer_tokens, learning_rate)

    site_dir = config.paths.pilot / site.name
    match stage:
        case "fit_quality":
            run_fit_quality(config, site, site_dir, build)
        case "search":
            run_search(config, site, site_dir, build, only_drop_then_bisect=False)
        case "drop_then_bisect":
            run_search(config, site, site_dir, build, only_drop_then_bisect=True)
    print(f"DAS pilot {stage} finished: {json.dumps(site.name)}", flush=True)


if __name__ == "__main__":
    main()
