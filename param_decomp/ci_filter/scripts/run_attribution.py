"""Helpers vs interferers: score every component by its effect on the TRUE arithmetic answer.

Runs `param_decomp/ci_filter/attribution.py` over the pool with a filter's CI fn (or the
decomposition's own): the first-order screen for every component, then a causal single-component
ablation of the extremes. A POSITIVE effect means ablating the component RAISES the correct
answer's renormalized log-probability — it interferes.

    python -m param_decomp.ci_filter.scripts.run_attribution --config <yaml> --data_root <root>
        --source <cf-id|run> [--batch_size 500] [--verify_top 100] [--verify_prompts 1000]

Writes `<run_dir>/analysis/ablations/step_<step>/<source>/`: `components.npz` (per-site mean effect, the count of prompts where
ablating HELPS the true answer, and activity counts, each overall and per operation), `summary.json` (accuracies, score means) and `verified.json` (the
ablated extremes, first-order vs causal)."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.attribution import (
    answer_targets,
    make_ablation_scores,
    make_attribution_batch,
)
from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.paths import ablation_source_dir
from param_decomp.ci_filter.pool import answer_token_ids, build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    index_batches,
    prepare_components,
    remove_components,
)
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache


def run_attribution(
    config: CIFilterConfig,
    data_root: Path,
    source: str,
    batch_size: int,
    verify_top: int,
    verify_prompts: int,
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out_dir = ablation_source_dir(run_dir, step, source)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(config.pool, tokenizer)
    answers = answer_token_ids(tokenizer, include_minus=True)
    targets = answer_targets(pool, tokenizer, answers)
    logger.info(
        f"attribution of {source} on {pool.n_prompts} prompts "
        f"({[b.operation for b in pool.blocks]}), {answers.size} answer tokens -> {out_dir}"
    )

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

        attribution_batch = make_attribution_batch(config.remat)
        blocks = {block.operation: (block.start, block.stop) for block in pool.blocks}
        totals: dict[str, dict[str, np.ndarray]] = {}
        counts: dict[str, dict[str, np.ndarray]] = {}
        positives: dict[str, dict[str, np.ndarray]] = {}
        n_rows = {name: 0 for name in ("all", *blocks)}
        scores: list[np.ndarray] = []
        clean_scores: list[np.ndarray] = []
        correct: list[np.ndarray] = []
        clean_correct: list[np.ndarray] = []

        t0 = time.time()
        for idx, real in index_batches(pool.n_prompts, batch_size):
            rows = attribution_batch(
                placed,
                prepared,
                ci_fn,
                constraints,
                tokens_all,
                jnp.asarray(idx),
                answer_ids,
                target_all,
            )
            keep_rows = idx[:real]
            scores.append(np.asarray(rows.score)[:real])
            clean_scores.append(np.asarray(rows.clean_score)[:real])
            correct.append(np.asarray(rows.correct)[:real])
            clean_correct.append(np.asarray(rows.clean_correct)[:real])
            n_rows["all"] += real
            for operation, (start, stop) in blocks.items():
                n_rows[operation] += int(((keep_rows >= start) & (keep_rows < stop)).sum())
            for site, value in rows.effect.items():
                effect = np.asarray(value)[:real]
                active = np.asarray(rows.active[site])[:real] > config.alive_threshold
                site_totals = totals.setdefault(site, {})
                site_counts = counts.setdefault(site, {})
                site_positives = positives.setdefault(site, {})
                for name, mask in [
                    ("all", np.ones(real, bool)),
                    *[
                        (operation, (keep_rows >= start) & (keep_rows < stop))
                        for operation, (start, stop) in blocks.items()
                    ],
                ]:
                    site_totals[name] = site_totals.get(name, 0.0) + effect[mask].sum(axis=0)
                    site_counts[name] = site_counts.get(name, 0.0) + active[mask].sum(axis=0)
                    # Sign per prompt, not size: "interferes on MOST prompts" is this count.
                    site_positives[name] = site_positives.get(name, 0.0) + (effect[mask] > 0.0).sum(
                        axis=0
                    )
        logger.info(f"screen over the pool: {time.time() - t0:.0f}s")

        means = {
            f"{site}|{name}": total / max(n_rows[name], 1)
            for site, per_name in totals.items()
            for name, total in per_name.items()
        }
        activity = {
            f"{site}|{name}": count
            for site, per_name in counts.items()
            for name, count in per_name.items()
        }
        positive = {
            f"{site}|{name}": count
            for site, per_name in positives.items()
            for name, count in per_name.items()
        }
        saved = {
            **means,
            **{f"active:{k}": v for k, v in activity.items()},
            **{f"positive:{k}": v for k, v in positive.items()},
        }
        np.savez(out_dir / "components.npz", **saved)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)

        flat_score = np.concatenate(scores)
        summary = {
            "source": source,
            "n_prompts": int(flat_score.size),
            "mean_score": float(flat_score.mean()),
            "mean_clean_score": float(np.concatenate(clean_scores).mean()),
            "accuracy": float(np.concatenate(correct).mean()),
            "clean_accuracy": float(np.concatenate(clean_correct).mean()),
            "n_rows": n_rows,
        }

        # The extremes of the screen, verified by really ablating each one alone.
        ranked = sorted(
            (
                (site, int(c), float(v))
                for site, per in totals.items()
                for c, v in enumerate(per["all"] / max(n_rows["all"], 1))
            ),
            key=lambda row: row[2],
        )
        picks = ranked[:verify_top] + ranked[-verify_top:] if verify_top else []
        verified: list[dict[str, float | str | int]] = []
        if picks:
            rng = np.random.default_rng(np.random.SeedSequence((config.seed, 7)))
            subset = np.sort(
                rng.choice(pool.n_prompts, size=min(verify_prompts, pool.n_prompts), replace=False)
            ).astype(np.int32)
            batches = [
                (subset[start : start + batch_size], min(batch_size, subset.size - start))
                for start in range(0, subset.size, batch_size)
            ]
            batches = [
                (np.pad(idx, (0, batch_size - idx.size), mode="edge"), real)
                for idx, real in batches
            ]
            ablation_scores = make_ablation_scores()
            site_c = {site: int(value["all"].size) for site, value in totals.items()}
            ones = {
                site: jax.sharding.reshard(jnp.ones(c, jnp.float32), P())
                for site, c in site_c.items()
            }

            def scored(keep: dict[str, jax.Array]) -> tuple[float, float]:
                score_parts, correct_parts = [], []
                for idx, real in batches:
                    s, c = ablation_scores(
                        placed,
                        prepared,
                        ci_fn,
                        constraints,
                        tokens_all,
                        jnp.asarray(idx),
                        answer_ids,
                        target_all,
                        keep,
                    )
                    score_parts.append(np.asarray(s)[:real])
                    correct_parts.append(np.asarray(c)[:real])
                return float(np.concatenate(score_parts).mean()), float(
                    np.concatenate(correct_parts).mean()
                )

            t0 = time.time()
            base_score, base_accuracy = scored(ones)
            summary |= {"verify_base_score": base_score, "verify_base_accuracy": base_accuracy}
            for site, component, first_order in picks:
                keep = dict(ones)
                keep[site] = ones[site].at[component].set(0.0)
                score, accuracy = scored(keep)
                verified.append(
                    {
                        "site": site,
                        "component": component,
                        "first_order": first_order,
                        "ablated_delta_score": score - base_score,
                        "ablated_delta_accuracy": accuracy - base_accuracy,
                    }
                )
            verified.sort(key=lambda row: float(row["ablated_delta_score"]), reverse=True)
            logger.info(
                f"verified {len(verified)} components on {subset.size} prompts ({time.time() - t0:.0f}s)"
            )

    (out_dir / "verified.json").write_text(json.dumps(verified, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"{json.dumps(summary, indent=2)}\n-> {out_dir}")
    return out_dir


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="CIFilterConfig YAML (run, pool)")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--source", default="run", help="`run` or a CI filter id")
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--verify_top", type=int, default=100, help="components verified per side")
    ap.add_argument("--verify_prompts", type=int, default=1000)
    args = ap.parse_args()
    run_attribution(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.source,
        args.batch_size,
        args.verify_top,
        args.verify_prompts,
    )


if __name__ == "__main__":
    main()
