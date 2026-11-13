"""Entry point: one subcommand per stage of the integer-only-adv pipeline (plan:
`notes/integer_only/plan_adv.md`; layout: `layout.py`).

    python -m param_decomp.integer_only.adv.run <stage> --run_dir <run> --data_root <root>
        --config <yaml> [--compilation_cache_dir <dir>]

Stages, in order: `train` (the CI-fn fine-tune, resumable), `alive` (the fine-tuned CI over the
pool, `integer-only-adv.json`), `score` (the integer-only-adv model on `Q`), `collect` (the
dataset), `applet`. Every stage reads the integer-only pipeline's `pool.npz`, `check_set.npy`
and `config.yaml` from `<run>/integer_only/` (its stages 1 and 2 must be done). The first stage
pins `--config` as `config.yaml`; later stages require the same settings (hardware-only fields
excepted). A stage whose `status/<stage>.done` exists is skipped; each writes
`status/<stage>.done` (facts) or `status/<stage>.failed` (the error)."""

import argparse
import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.checkpoint import restore_ci_fn, save_ci_fn, starting_ci
from param_decomp.ci_filter.config import CIFilterConfig
from param_decomp.ci_filter.pool import load_tokenizer
from param_decomp.ci_filter.ppgd import init_adversary
from param_decomp.ci_filter.step import CIConstraints, FilterCIFn, index_batches, prepare_components
from param_decomp.core.configs import PlacementSpec, TargetedPDConfig
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.load_run import RestoredRun, restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.run_data import product_document
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.integer_only import applet as ap
from param_decomp.integer_only.adv.adversary import restore_run_adversary
from param_decomp.integer_only.adv.coefficients import TARGET_TERM, filter_config, resolve_final
from param_decomp.integer_only.adv.config import MUTABLE, IntegerOnlyAdvConfig
from param_decomp.integer_only.adv.layout import (
    ADV_MODEL,
    DATASET_README,
    README,
    STAGES,
    VIEW_LABELS,
    AdvLayout,
)
from param_decomp.integer_only.adv.train import Problem, fresh_state, train
from param_decomp.integer_only.ci_pass import chunk_max_ci
from param_decomp.integer_only.collect import collect_dataset
from param_decomp.integer_only.config import IntegerOnlyConfig
from param_decomp.integer_only.layout import Layout
from param_decomp.integer_only.model import build_factors, build_target, slots_of
from param_decomp.integer_only.pick import (
    KL_INT,
    SUMMARY_HEADER,
    TARGET,
    mask_json,
    read_mask_json,
    score_masks,
    summary,
    summary_line,
)
from param_decomp.integer_only.pool import PromptPool, integer_token_ids
from param_decomp.integer_only.run import (
    commit,
    keep_of,
    labels_text,
    load_pool,
    run_stage,
    site_sizes,
)
from param_decomp.migrations.fork_2026_10 import read_stored_config
from param_decomp.targets.transformer import TransformerDecomposedModel


def pin_config(layout: AdvLayout, config: IntegerOnlyAdvConfig) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    if layout.config.exists():
        old = IntegerOnlyAdvConfig.from_file(layout.config)
        for key in IntegerOnlyAdvConfig.model_fields:
            if key not in MUTABLE:
                assert getattr(old, key) == getattr(config, key), (
                    f"{key} differs from {layout.config}"
                )
        return
    layout.config.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))


def restore(
    run_dir: Path, config: IntegerOnlyAdvConfig, data_root: Path, sharding: PlacementSpec = "zero1"
) -> RestoredRun:
    restored = restore_jax_run(run_dir, config.step, data_root=data_root, sharding=sharding)
    assert restored.step == config.step, (restored.step, config.step)
    assert isinstance(restored.placed.model, TransformerDecomposedModel)
    return restored


def assert_one_device(stage: str) -> None:
    """`score` and `collect` build the single-device factored model (`integer_only.model`) from
    the restored run, as integer-only stages 4 and 5 do on one GPU; on a 2-L40 smoke the same
    build under a 2-device mesh ran for over 20 minutes without finishing."""
    assert jax.device_count() == 1, f"{stage}: run with one visible device, not {jax.devices()}"


def inputs(io: Layout) -> tuple[IntegerOnlyConfig, PromptPool, np.ndarray]:
    """The integer-only pipeline's pinned config, pool and check set `Q`."""
    io_config = IntegerOnlyConfig.from_file(io.config)
    return io_config, load_pool(io, io_config), np.load(io.check_set)


def fine_tuned(
    layout: AdvLayout, restored: RestoredRun, run_dir: Path
) -> tuple[FilterCIFn, CIConstraints]:
    """The fine-tuned CI fn and the ceiling it is read under. Call under the run's mesh."""
    resolved = CIFilterConfig.from_file(layout.filter_config)
    _, constraints = starting_ci(resolved, run_dir, restored.step, restored.ci_fn)
    return restore_ci_fn(layout.ci_fn, restored.ci_fn), constraints


def alive_document(
    max_ci: dict[str, np.ndarray], threshold: float, sizes: dict[str, int], rule: str, source: str
) -> dict[str, object]:
    """The components whose max CI (`{site: (T, C)}`) exceeds `threshold` at some position, in
    the schema of `integer-only.json` over every site of `sizes` (`{site: C}`)."""
    keep = {site: np.asarray((v > threshold).any(axis=0)) for site, v in max_ci.items()}
    slots = slots_of(keep)
    return mask_json(
        slots, np.ones(slots.n, bool), sizes, rule=rule, source=source, threshold=threshold
    )


# ----------------------------------- stages -----------------------------------


def stage_train(
    config: IntegerOnlyAdvConfig, layout: AdvLayout, io: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    _, pool, check = inputs(io)
    train_rows = np.setdiff1d(pool.problem_rows(), check)
    pd = TargetedPDConfig.model_validate(read_stored_config(product_document(run_dir))["pd"])
    coefficients = resolve_final(pd, pool.seq_len)
    resolved = filter_config(coefficients, config, run_dir)
    layout.training.mkdir(parents=True, exist_ok=True)
    layout.coefficients.write_text(json.dumps(coefficients.as_json(), indent=1) + "\n")
    resolved.to_file(layout.filter_config)
    logger.info(f"final-step coefficients: {coefficients.as_json()}")
    restored = restore(run_dir, config, data_root, config.placement)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    int_ids = integer_token_ids(load_tokenizer(target.model_name))
    mesh = restored.mesh
    assert config.microbatch % mesh.size == 0, f"microbatch must tile the {mesh.size}-device mesh"
    placed = restored.placed
    with jax.set_mesh(mesh):
        ci_fn, constraints = starting_ci(resolved, run_dir, restored.step, restored.ci_fn)
        assert constraints.kept is None and len(constraints.ceilings) == 1, constraints
        prepared, _ = prepare_components(placed, restored.components)
        problem = Problem(
            tokens_all=jax.sharding.reshard(jnp.asarray(pool.tokens), P()),
            answer_ids=jax.sharding.reshard(jnp.asarray(int_ids), P()),
            targets=jax.sharding.reshard(jnp.asarray(pool.r.astype(np.int32)), P()),
            train=train_rows,
        )
        like = init_adversary(
            placed, coefficients.run_batch, pool.seq_len, jax.random.PRNGKey(config.seed + 2)
        )
        resuming = layout.latest_state.exists()
        match config.adversary_init:
            case "run" if not resuming:
                t0 = time.time()
                adversary = restore_run_adversary(run_dir, restored.step, TARGET_TERM, like, mesh)
                logger.info(
                    f"adversary restored from step {restored.step} ({time.time() - t0:.0f}s), "
                    f"{float(adversary.opt_state.step_count):.0f} ascents"
                )
            case "run" | "fresh":
                adversary = like
        del restored
        start = fresh_state(ci_fn, resolved, adversary)
        adversary_ascents_at_start = float(adversary.opt_state.step_count)
        del adversary, like
        trained, facts = train(
            config, resolved, layout, placed, prepared, constraints, problem, start
        )
        save_ci_fn(layout.ci_fn, trained)
    for old in layout.training.glob("state_*"):
        shutil.rmtree(old)
    layout.latest_state.unlink(missing_ok=True)
    return facts | {
        "n_train_rows": int(train_rows.size),
        "adversary_init": config.adversary_init,
        "adversary_rows": coefficients.run_batch,
        "adversary_ascents_at_start": adversary_ascents_at_start,
        "coefficients": coefficients.as_json(),
    }


def stage_alive(
    config: IntegerOnlyAdvConfig, layout: AdvLayout, io: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    _, pool, _ = inputs(io)
    restored = restore(run_dir, config, data_root)
    placed = restored.placed
    t0 = time.time()
    running: dict[str, np.ndarray] = {}
    with jax.set_mesh(restored.mesh):
        ci_fn, constraints = fine_tuned(layout, restored, run_dir)
        prepared, _ = prepare_components(placed, restored.components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        batches = index_batches(pool.n, config.ci_batch)
        for i, (idx, real) in enumerate(batches):
            got = chunk_max_ci(
                placed, prepared, ci_fn, constraints, tokens_all, jnp.asarray(idx),
                jnp.asarray(real),
            )  # fmt: skip
            for site, value in got.items():
                v = np.asarray(value)
                running[site] = np.maximum(running[site], v) if site in running else v
            if i % 50 == 0 or i == len(batches) - 1:
                logger.info(f"alive: batch {i + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    np.savez(layout.max_ci, **running)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    sizes = site_sizes(restored)
    doc = alive_document(
        running, config.threshold, sizes,
        rule=f"integer-only-adv CI (the run's CI fn fine-tuned under the ceiling: "
        f"{layout.ci_fn}) > {config.threshold} at some position of some pool prompt",
        source=str(layout.max_ci),
    )  # fmt: skip
    layout.alive.write_text(json.dumps(doc) + "\n")
    per_position = [
        int(sum(int((v[t] > config.threshold).sum()) for v in running.values()))
        for t in range(pool.seq_len)
    ]
    return {
        "n_integer_only_adv": doc["n_alive"],
        "n_components": doc["n_components"],
        "on_per_position": per_position,
        "seconds_ci_pass": round(time.time() - t0, 1),
    }


def stage_score(
    config: IntegerOnlyAdvConfig, layout: AdvLayout, io: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    assert_one_device("score")
    io_config, pool, check = inputs(io)
    restored = restore(run_dir, config, data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    int_ids = integer_token_ids(load_tokenizer(target.model_name))
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    keep = keep_of(read_mask_json(layout.alive), site_sizes(restored))
    with jax.set_mesh(restored.mesh):
        arch, shared, dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
    del restored
    logger.info(f"score: {slots.n} components stored, scoring {check.size} prompts of Q")
    t0 = time.time()
    scored = score_masks(
        arch, shared, factors, dense, slots, {ADV_MODEL: np.ones(slots.n, bool)}, pool, check,
        int_ids, io_config,
    )  # fmt: skip
    ops = pool.op[check]
    adv = summary(scored[ADV_MODEL], ops, pool.ops)
    final = {"model": ADV_MODEL, "size": slots.n, "prompts": int(check.size)} | adv
    layout.final.write_text(json.dumps(final, indent=1) + "\n")
    adv_line = summary_line(ADV_MODEL, slots.n, adv, chosen=False)
    if io.summary.exists():
        lines = io.summary.read_text().splitlines()
        assert lines[0] == SUMMARY_HEADER, (lines[0], SUMMARY_HEADER)
        source = str(io.summary)
    else:
        lines = [
            SUMMARY_HEADER,
            summary_line(TARGET, None, summary(scored[TARGET], ops, pool.ops), False),
        ]
        source = "none (integer-only stage pick not done): the target row is scored here"
    layout.comparison.write_text("\n".join([*lines, adv_line]) + "\n")
    return {
        "size": slots.n,
        "kl_int": float(np.mean(scored[ADV_MODEL][KL_INT])),
        "comparison_rows_from": source,
        "seconds_scoring": round(time.time() - t0, 1),
    }


def stage_collect(
    config: IntegerOnlyAdvConfig, layout: AdvLayout, io: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    assert_one_device("collect")
    io_config, pool, _ = inputs(io)
    restored = restore(run_dir, config, data_root)
    model = restored.placed.model
    assert isinstance(model, TransformerDecomposedModel)
    keep = keep_of(read_mask_json(layout.alive), site_sizes(restored))
    with np.load(layout.max_ci) as z:
        max_ci = {site: z[site] for site in z.files}
    placed = restored.placed
    with jax.set_mesh(restored.mesh):
        ci_fn, constraints = fine_tuned(layout, restored, run_dir)
        arch, shared, _dense = build_target(model)
        factors, slots = build_factors(restored.components, keep, arch.n_layer)
        prepared, v_norms = prepare_components(placed, restored.components)

    def readme(n_columns: int, per_position: str, n_first_only: int, n_second_only: int) -> str:
        return DATASET_README.format(
            run_dir=run_dir, step=config.step, commit=commit(), n_components=slots.n,
            n_prompts=pool.n, prompts=io_config.prompts, seq_len=pool.seq_len,
            labels=labels_text(pool), threshold=config.threshold, n_columns=n_columns,
            per_position=per_position, ops=", ".join(f"{k} = {o}" for k, o in enumerate(pool.ops)),
            precision=io_config.collect_precision, last=pool.seq_len - 1,
            n_first_only=n_first_only, n_second_only=n_second_only,
        )  # fmt: skip

    facts = collect_dataset(
        placed, prepared, v_norms, ci_fn, constraints, restored.mesh, arch, shared, factors,
        slots, pool, max_ci, config.threshold, config.ci_batch, io_config.collect_precision,
        layout.datasets, readme,
    )  # fmt: skip
    return {"n_integer_only_adv": slots.n} | facts


def stage_applet(
    config: IntegerOnlyAdvConfig, layout: AdvLayout, io: Layout, run_dir: Path, data_root: Path
) -> dict[str, object]:
    del run_dir, data_root
    _, pool, _ = inputs(io)
    sizes = {site: int(v.shape[1]) for site, v in np.load(layout.max_ci).items()}
    return ap.write_applet(
        layout.datasets, pool, sizes, config.step, config.threshold,
        f"integer-only-adv components, step {config.step}", VIEW_LABELS, layout.applet_data,
        layout.applet_page,
    )  # fmt: skip


STAGE_FNS = {
    "train": stage_train,
    "alive": stage_alive,
    "score": stage_score,
    "collect": stage_collect,
    "applet": stage_applet,
}
assert tuple(STAGE_FNS) == STAGES


# ----------------------------------- driver -----------------------------------


def write_readme(
    layout: AdvLayout, config: IntegerOnlyAdvConfig, io: Layout, run_dir: Path
) -> None:
    first = io.facts("candidates") or {}
    alive = layout.facts("alive") or {}
    stage_lines = []
    for stage in STAGES:
        state = (
            "done"
            if layout.done(stage).exists()
            else "FAILED"
            if layout.failed(stage).exists()
            else "not run"
        )
        stage_lines.append(f"- `{stage}`: {state}")
    layout.readme.write_text(
        README.format(
            run_dir=run_dir, step=config.step, commit=commit(),
            seq_len=first.get("seq_len", "?"), n_prompts=first.get("n_prompts", "?"),
            threshold=config.threshold, steps=config.steps, batch=config.batch,
            n_alive=alive.get("n_integer_only_adv", "?"), adv_model=ADV_MODEL,
            stages="\n".join(stage_lines),
        )
    )  # fmt: skip


def main() -> None:
    setup_console_logger()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="IntegerOnlyAdvConfig YAML")
    parser.add_argument("--compilation_cache_dir", type=Path, default=None)
    args = parser.parse_args()
    if args.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(args.compilation_cache_dir)
    config = IntegerOnlyAdvConfig.from_file(args.config)
    run_dir: Path = args.run_dir
    layout = AdvLayout.of_run(run_dir)
    io = Layout.of_run(run_dir)
    pin_config(layout, config)
    stage: str = args.stage
    fn = STAGE_FNS[stage]
    run_stage(layout, stage, lambda: fn(config, layout, io, run_dir, args.data_root))
    write_readme(layout, config, io, run_dir)


if __name__ == "__main__":
    main()
