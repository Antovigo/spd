"""Ablate EVERY alive component, one at a time, and record what it does to each metric.

    python -m param_decomp.ci_filter.scripts.ablation_sweep --config <yaml> --data_root <root>
        --source <cf-id|run> [--prompts 512] [--limit N]

For each component alive in the filter's own CI (or in `--alive_npz`) (max output CI > `alive_threshold` on the pool),
zeroes its mask at every position of a FIXED prompt subset and records the change against the
un-ablated baseline in full-vocabulary KL, answer-restricted KL, the true answer's cross-entropy
and accuracy. The clean forward and the CI evaluation are computed once for the subset, so the
inner loop is one masked forward per component.

Writes `<filter>/attribution/ablation_sweep.tsv` incrementally (one flushed row per component,
so a killed job keeps its progress); rerunning skips the components already in the file."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.attribution import (
    answer_targets,
    make_ablation_metrics,
    make_ablation_state,
)
from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import answer_token_ids, build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    prepare_components,
    remove_components,
)
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache

METRICS = ("kl", "integer_kl", "answer_ce", "accuracy")
COLUMNS = ("site", "component", *[f"d_{m}" for m in METRICS])


def ablation_sweep(
    config: CIFilterConfig,
    data_root: Path,
    source: str,
    prompts: int,
    limit: int | None,
    alive_npz: Path | None,
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out_dir = (
        run_dir / "analysis" / "ci_filter" / f"step_{step}" / "attribution_run"
        if source == "run"
        else CIFilterOutputs.for_run(run_dir, step, source).root / "attribution"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ablation_sweep.tsv"

    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(config.pool, tokenizer)
    answers = answer_token_ids(tokenizer, include_minus=True)
    targets = answer_targets(pool, tokenizer, answers)

    with jax.set_mesh(mesh):
        ci_fn, constraints = (
            (restored.ci_fn, UNCONSTRAINED)
            if source == "run"
            else trained_ci(run_dir, step, source, restored.ci_fn)
        )
        components = (
            restored.components
            if constraints.kept is None
            else remove_components(restored.components, constraints.kept)
        )
        del restored
        prepared, _ = prepare_components(placed, components)
        del components
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())
        target_all = jax.sharding.reshard(jnp.asarray(targets.indices), P())

        # The subset is a fixed, seed-derived sample; the whole sweep scores the same prompts.
        rng = np.random.default_rng(np.random.SeedSequence((config.seed, 11)))
        subset = np.sort(rng.choice(pool.n_prompts, size=prompts, replace=False)).astype(np.int32)
        state = make_ablation_state()(
            placed, ci_fn, constraints, tokens_all, jnp.asarray(subset), target_all
        )
        if alive_npz is None:
            alive = {
                site: np.asarray(jnp.max(value.astype(jnp.float32), axis=(0, 1)))
                > config.alive_threshold
                for site, value in state.masks.items()
            }
        else:  # an explicit set (e.g. the decomposition's own `alive/kept.npz`), so the sweep
            with np.load(alive_npz) as saved:  # covers the same rows as another sweep's
                alive = {site: saved[site].astype(bool) for site in saved.files}
        del ci_fn, constraints  # the ceilings are only read by `output_ci`, already done

        metrics = make_ablation_metrics()
        ones = {
            site: jax.sharding.reshard(jnp.ones(mask.size, jnp.float32), P())
            for site, mask in alive.items()
        }
        base = {k: float(v) for k, v in metrics(placed, prepared, state, answer_ids, ones).items()}
        logger.info(f"baseline on {prompts} prompts: {base}")

        done: set[tuple[str, int]] = set()
        if out.exists():
            for line in out.read_text().splitlines()[1:]:
                site, component = line.split("\t")[:2]
                done.add((site, int(component)))
            logger.info(f"resuming: {len(done)} components already swept")
        else:
            out.write_text("\t".join(COLUMNS) + "\n")

        todo = [
            (site, int(c))
            for site in sorted(alive)
            for c in np.nonzero(alive[site])[0]
            if (site, int(c)) not in done
        ]
        if limit is not None:
            todo = todo[:limit]
        logger.info(f"{len(todo)} components to ablate (baseline excluded)")

        t0 = time.time()
        with out.open("a") as sink:
            for i, (site, component) in enumerate(todo):
                keep = dict(ones)
                keep[site] = ones[site].at[component].set(0.0)
                got = metrics(placed, prepared, state, answer_ids, keep)
                row = [site, str(component)] + [f"{float(got[m]) - base[m]:.6g}" for m in METRICS]
                sink.write("\t".join(row) + "\n")
                sink.flush()
                if i % 200 == 0:
                    rate = (time.time() - t0) / max(i, 1)
                    logger.info(
                        f"{i}/{len(todo)} ({rate:.2f}s/component, "
                        f"{(len(todo) - i) * rate / 3600:.1f}h left)"
                    )
    logger.info(f"done -> {out}")
    return out


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="the filter's pinned config.yaml")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--source", default="run", help="`run` or a CI filter id")
    ap.add_argument("--prompts", type=int, default=512, help="fixed pool subset scored")
    ap.add_argument("--limit", type=int, default=None, help="stop after N components (smoke)")
    ap.add_argument(
        "--alive_npz",
        type=Path,
        default=None,
        help="{site: (C,)} bool mask of components to ablate; default = alive on the subset",
    )
    args = ap.parse_args()
    ablation_sweep(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.source,
        args.prompts,
        args.limit,
        args.alive_npz,
    )


if __name__ == "__main__":
    main()
