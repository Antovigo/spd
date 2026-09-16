"""Fine-tune a frozen decomposition's output CI fn against a narrower output objective.

Restores the decomposition, builds the arithmetic pool, evaluates the starting CI fn,
trains, re-evaluates, then writes the alive list and the `(a, b)`-grid applet under the
decomposition's own run dir (layout: `param_decomp/ci_filter/paths.py`). `data_root` resolves
the run id, named datasets and target caches."""

import argparse
import json
import time
from pathlib import Path
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from transformers import AutoTokenizer

from param_decomp.ci_filter.checkpoint import restore_ci_fn, save_ci_fn
from param_decomp.ci_filter.config import (
    CIFilterConfig,
    InitFromCIFilter,
    InitFromRun,
    LastPositionIntegerKL,
    LastPositionKL,
    MaskScores,
    PoolEval,
    resolve_run_dir,
)
from param_decomp.ci_filter.grid import collect_operation, write_applet, write_operation
from param_decomp.ci_filter.paths import CIFilterOutputs, new_ci_filter_id
from param_decomp.ci_filter.pool import Tokenizer, answer_token_ids, build_pool
from param_decomp.ci_filter.step import (
    Prepared,
    RowArrays,
    accumulate,
    index_batches,
    make_apply_update,
    make_eval_batch,
    make_micro_grads,
    make_optimizer,
    make_score_fixed_masks,
    sample_batch,
    trainable,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import PlacedModel, prepare_compute_weights
from param_decomp.core.run_state import optax_schedule
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.targets.glu_transformer import hf_snapshot_dir


@eqx.filter_jit
def _prepare(placed: PlacedModel, components: ComponentStacks) -> tuple[Prepared, dict[str, Array]]:
    v_norms = {
        site: jax.sharding.reshard(
            jnp.linalg.norm(components.site(site).V.astype(jnp.float32), axis=0), P()
        )
        for site in placed.site_names
    }
    return prepare_compute_weights(placed, components), v_norms


def _mask_scores(rows: dict[str, np.ndarray]) -> MaskScores:
    return MaskScores(
        kl=float(rows["kl"].mean()),
        integer_kl=float(rows["integer_kl"].mean()),
        top1=float(rows["top1"].mean()),
        integer_top1=float(rows["integer_top1"].mean()),
        max_prompt_objective=float(rows["objective"].max()),
    )


def _concat(parts: list[RowArrays], reals: list[int]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate(
            [np.asarray(p[key])[:real] for p, real in zip(parts, reals, strict=True)]
        )
        for key in parts[0]
    }


def run_ci_filter(config: CIFilterConfig, data_root: Path, filter_id: str) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig), "the pool reads an HF-family target's tokenizer"
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    outputs = CIFilterOutputs.for_run(run_dir, step, filter_id)
    outputs.create()
    config.to_file(outputs.config)
    logger.info(f"ci filter {filter_id}: run {run_dir} step {step} -> {outputs.root}")

    tokenizer = cast(
        Tokenizer,
        cast(
            object,
            AutoTokenizer.from_pretrained(
                str(hf_snapshot_dir(target.model_name)), local_files_only=True
            ),
        ),
    )
    pool = build_pool(config.pool, tokenizer)
    include_minus = (
        config.objective.include_minus
        if isinstance(config.objective, LastPositionIntegerKL)
        else True
    )
    answers = answer_token_ids(tokenizer, include_minus)
    logger.info(
        f"pool {pool.tokens.shape} ({[b.operation for b in pool.blocks]}), "
        f"{answers.size} answer tokens, objective {config.objective.kind}"
    )
    for size in (config.microbatch_size or config.batch_size, config.eval_batch_size):
        assert size % mesh.size == 0, f"batch {size} must tile the {mesh.size}-device mesh"

    with jax.set_mesh(mesh):
        prepared, v_norms = _prepare(placed, restored.components)
        ci_fn = restored.ci_fn
        del restored
        match config.init:
            case InitFromRun():
                pass
            case InitFromCIFilter(id=source_id):
                source = CIFilterOutputs.for_run(run_dir, step, source_id)
                ci_fn = restore_ci_fn(source.ci_fn, ci_fn)
                logger.info(f"CI fn initialized from {source.root}")
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())

        eval_batch = make_eval_batch(config)
        score_fixed = make_score_fixed_masks(config)

        def evaluate(
            at_step: int, ci_fn: PlacedCIFn, references: bool
        ) -> tuple[PoolEval, dict[str, np.ndarray]]:
            t0 = time.time()
            parts: dict[str, list[RowArrays]] = {"ci": [], "rounded": []}
            l0_parts: list[RowArrays] = []
            reals: list[int] = []
            site_max: dict[str, np.ndarray] = {}
            for idx, real in index_batches(pool.n_prompts, config.eval_batch_size):
                rows, smax, l0 = eval_batch(
                    placed, prepared, ci_fn, tokens_all, jnp.asarray(idx), answer_ids
                )
                for name, r in rows.items():
                    parts[name].append(r)
                l0_parts.append(l0)
                reals.append(real)
                for site, v in smax.items():
                    site_max[site] = np.maximum(site_max.get(site, 0.0), np.asarray(v))
            alive = {site: m > config.alive_threshold for site, m in site_max.items()}

            def fixed(masks: dict[str, np.ndarray]) -> MaskScores:
                device_masks = {k: jnp.asarray(v, jnp.float32) for k, v in masks.items()}
                fixed_parts = [
                    score_fixed(
                        placed, prepared, device_masks, tokens_all, jnp.asarray(idx), answer_ids
                    )
                    for idx, _ in index_batches(pool.n_prompts, config.eval_batch_size)
                ]
                return _mask_scores(_concat(fixed_parts, reals))

            l0 = _concat(l0_parts, reals)
            result = PoolEval(
                step=at_step,
                ci=_mask_scores(_concat(parts["ci"], reals)),
                rounded=_mask_scores(_concat(parts["rounded"], reals)),
                l0_per_token=float(l0["l0_per_token"].mean()),
                l0_per_token_last=float(l0["l0_last"].mean()),
                n_alive=int(sum(int(a.sum()) for a in alive.values())),
                alive_per_site={site: int(a.sum()) for site, a in alive.items()},
                alive_set=fixed(alive) if references else None,
                all_on=fixed({k: np.ones_like(v) for k, v in alive.items()})
                if references and at_step == 0
                else None,
            )
            with outputs.pool_evals.open("a") as sink:
                sink.write(result.model_dump_json() + "\n")
            logger.info(
                f"eval @ {at_step} ({time.time() - t0:.0f}s): n_alive {result.n_alive}, "
                f"L0/token {result.l0_per_token:.1f} (last {result.l0_per_token_last:.1f}), "
                f"ci {result.ci}, rounded {result.rounded}, alive_set {result.alive_set}, "
                f"all_on {result.all_on}"
            )
            return result, site_max

        optimizer = make_optimizer(config)
        lr = optax_schedule(config.lr_schedule, config.steps)
        opt_state = optimizer.init(trainable(ci_fn))
        micro_grads = make_micro_grads(config)
        apply_update = make_apply_update(optimizer)
        micro = config.microbatch_size or config.batch_size
        # Warm the training step (result discarded) BEFORE the first evaluation, so its large
        # temporary arena is carved from an unfragmented pool: evaluating first OOMed microbatches
        # that fit on their own (jobs 11804 at 256 and 11809 at 512, 1x L40).
        jax.block_until_ready(
            micro_grads(
                placed,
                prepared,
                ci_fn,
                tokens_all,
                jnp.asarray(
                    sample_batch(pool.n_prompts, config.batch_size, config.seed, 0)[:micro]
                ),
                answer_ids,
                jnp.float32(0.0),
            )
        )
        evaluate(0, ci_fn, references=True)

        with outputs.metrics.open("w") as sink:
            t_start = time.time()
            t_log = t_start
            for i in range(config.steps):
                idx = sample_batch(pool.n_prompts, config.batch_size, config.seed, i)
                train_frac = jnp.float32(i / max(config.steps - 1, 1))
                grads = None
                metrics: dict[str, Array] = {}
                schedules: dict[str, Array] = {}
                for start in range(0, config.batch_size, micro):
                    g, m, schedules = micro_grads(
                        placed,
                        prepared,
                        ci_fn,
                        tokens_all,
                        jnp.asarray(idx[start : start + micro]),
                        answer_ids,
                        train_frac,
                    )
                    grads = g if grads is None else accumulate(grads, g)
                    metrics = {k: metrics[k] + v if metrics else v for k, v in m.items()}
                assert grads is not None
                ci_fn, opt_state, grad_norm = apply_update(ci_fn, opt_state, grads)
                del grads
                if i % config.log_every == 0 or i == config.steps - 1:
                    now = time.time()
                    record = {
                        "step": i,
                        **{k: float(v) for k, v in (metrics | schedules).items()},
                        "grad_norm": float(grad_norm),
                        "lr": float(lr(i)),
                        "elapsed_s": now - t_start,
                        "s_per_step": (now - t_log) / (config.log_every if i else 1),
                    }
                    t_log = now
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    logger.info(
                        f"step {i}: loss {record['loss']:.4f} recon {record['recon']:.4f} "
                        f"act {record['imp_activity']:.0f} L0 {record['l0']:.0f} "
                        f"L0>{config.alive_threshold} {record['l0_alive']:.0f} "
                        f"gamma {record['gamma']:.3f} {record['s_per_step']:.2f}s/step"
                    )
                if (
                    config.eval_every is not None
                    and (i + 1) % config.eval_every == 0
                    and i + 1 < config.steps
                ):
                    evaluate(i + 1, ci_fn, references=False)

        del opt_state
        save_ci_fn(outputs.ci_fn, ci_fn)
        logger.info(f"CI fn saved to {outputs.ci_fn}")
        final, site_max = evaluate(config.steps, ci_fn, references=True)
        outputs.summary.write_text(final.model_dump_json(indent=2))
        np.savez(outputs.max_ci, **site_max)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
        alive = {
            site: np.nonzero(m > config.alive_threshold)[0].tolist() for site, m in site_max.items()
        }
        outputs.alive.write_text(
            json.dumps(
                {
                    "threshold": config.alive_threshold,
                    "n_alive": final.n_alive,
                    "n_components": int(sum(m.size for m in site_max.values())),
                    "components": alive,
                },
                indent=1,
            )
        )
        logger.info(f"{final.n_alive} alive components -> {outputs.alive}")

        match config.objective:
            case LastPositionKL():
                title = "last-position KL"
            case LastPositionIntegerKL():
                title = "last-position integer KL"
        write_applet(outputs.grids, pool, f"{run_dir.name} step {step} · {filter_id} · {title}")
        for block in pool.blocks:
            t0 = time.time()
            slices = collect_operation(
                placed,
                prepared,
                ci_fn,
                v_norms,
                tokens_all,
                block,
                pool.seq_len,
                config.grid.chunk_prompts,
                config.grid.mean_ci_floor,
            )
            counts = write_operation(
                outputs.grids, slices, pool, config.steps, config.grid.mean_ci_floor
            )
            logger.info(f"grid {block.operation} ({time.time() - t0:.0f}s): saved {counts}")
    logger.info(f"done: {outputs.root}")
    return outputs.root


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="CIFilterConfig YAML/JSON")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--filter_id", type=str, default=None, help="pre-assigned id (cf-xxxxxxxx)")
    args = ap.parse_args()
    run_ci_filter(
        CIFilterConfig.from_file(args.config), args.data_root, args.filter_id or new_ci_filter_id()
    )


if __name__ == "__main__":
    main()
