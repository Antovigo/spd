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
import wandb
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.checkpoint import restore_ci_fn, save_ci_fn
from param_decomp.ci_filter.config import (
    CIFilterConfig,
    CIMaskedRecon,
    InitFromCIFilter,
    InitFromRun,
    LastPositionIntegerKL,
    LastPositionKL,
    MaskScores,
    MergedPPGDRecon,
    PoolEval,
    resolve_run_dir,
)
from param_decomp.ci_filter.grid import collect_operation, write_applet, write_operation
from param_decomp.ci_filter.paths import CIFilterOutputs, new_ci_filter_id
from param_decomp.ci_filter.pool import answer_token_ids, build_pool, load_tokenizer
from param_decomp.ci_filter.ppgd import PPGDStep, init_adversary
from param_decomp.ci_filter.step import (
    RowArrays,
    batch_gradients,
    ci_masked_micro_call,
    index_batches,
    make_apply_update,
    make_eval_batch,
    make_micro_grads,
    make_optimizer,
    make_pgd_eval,
    make_score_fixed_masks,
    pgd_eval_batches,
    prepare_components,
    sample_batch,
    trainable,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.run_state import optax_schedule
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.infra.wandb import init_wandb, try_wandb


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


def _flat_eval(result: PoolEval) -> dict[str, float]:
    """`eval/<masking>/<score>` scalars of a pool evaluation (per-site alive counts stay in
    the jsonl)."""
    out: dict[str, float] = {}
    for key, value in result.model_dump(exclude={"step", "alive_per_site"}).items():
        if isinstance(value, dict):
            out |= {f"eval/{key}/{k}": float(v) for k, v in value.items()}
        elif value is not None:
            out[f"eval/{key}"] = float(value)
    return out


def _wandb_log(values: dict[str, float], step: int) -> None:
    if wandb.run is not None:
        try_wandb(wandb.log, values, step=step)


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
    if config.wandb is not None:
        init_wandb(
            config.wandb.project,
            filter_id,
            {
                **config.model_dump(mode="json"),
                "run_dir": str(run_dir),
                "checkpoint_step": step,
                "filter_id": filter_id,
            },
            resume=False,
            entity=config.wandb.entity,
            name=f"{run_dir.name}-{config.objective.kind}-{filter_id}",
            group=run_dir.name,
            tags=[config.objective.kind, config.init.kind],
        )

    tokenizer = load_tokenizer(target.model_name)
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
        prepared, v_norms = prepare_components(placed, restored.components)
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
        pgd_eval = make_pgd_eval(config) if config.pgd_eval is not None else None

        def pgd_recon(ci_fn: PlacedCIFn) -> float:
            assert pgd_eval is not None
            key = jax.random.PRNGKey(config.seed + 1)
            values = [
                float(
                    pgd_eval(
                        placed,
                        prepared,
                        ci_fn,
                        tokens_all,
                        jnp.asarray(idx),
                        jax.random.fold_in(key, b),
                    )
                )
                for b, idx in enumerate(pgd_eval_batches(pool.n_prompts, config))
            ]
            return float(np.mean(values))

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
            _wandb_log(_flat_eval(result), at_step)
            with outputs.pool_evals.open("a") as sink:
                sink.write(result.model_dump_json() + "\n")
            logger.info(
                f"eval @ {at_step} ({time.time() - t0:.0f}s): n_alive {result.n_alive}, "
                f"L0/token {result.l0_per_token:.1f} (last {result.l0_per_token_last:.1f}), "
                f"ci {result.ci}, rounded {result.rounded}, alive_set {result.alive_set}, "
                f"all_on {result.all_on}"
            )
            return result, site_max

        evaluate(0, ci_fn, references=True)
        # The starting CI fn's PGD eval runs at the END, beside the final one: run first, its
        # 20-step adversarial backward left a 1x L40 unable to fit the training step (smoke
        # 11820). Its masters wait in host memory.
        initial_ci_fn = jax.tree.map(np.asarray, trainable(ci_fn)) if pgd_eval is not None else None

        optimizer = make_optimizer(config)
        lr = optax_schedule(config.lr_schedule, config.steps)
        opt_state = optimizer.init(trainable(ci_fn))
        apply_update = make_apply_update(optimizer)
        micro = config.microbatch_size or config.batch_size
        match config.recon:
            case CIMaskedRecon():
                micro_grads = make_micro_grads(config)
                ppgd_step, adversary = None, None
            case MergedPPGDRecon():
                micro_grads = None
                ppgd_step = PPGDStep.build(config)
                adversary = init_adversary(
                    placed, config.batch_size, pool.seq_len, jax.random.PRNGKey(config.seed + 2)
                )
        step_key = jax.random.PRNGKey(config.seed + 3)

        with outputs.metrics.open("w") as sink:
            t_start = time.time()
            t_log = t_start
            for i in range(config.steps):
                idx = sample_batch(pool.n_prompts, config.batch_size, config.seed, i)
                train_frac = jnp.float32(i / max(config.steps - 1, 1))
                if ppgd_step is None:
                    assert micro_grads is not None
                    call = ci_masked_micro_call(
                        micro_grads, placed, prepared, ci_fn, tokens_all, answer_ids, train_frac
                    )
                    grads, metrics, schedules = batch_gradients(
                        call, idx, micro, config.accumulate_on_host
                    )
                else:
                    assert adversary is not None
                    grads, metrics, schedules, adversary = ppgd_step(
                        placed,
                        prepared,
                        ci_fn,
                        tokens_all,
                        idx,
                        answer_ids,
                        train_frac,
                        adversary,
                        jax.random.fold_in(step_key, i),
                        micro,
                        config.accumulate_on_host,
                    )
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
                    _wandb_log({f"train/{k}": v for k, v in record.items() if k != "step"}, i)
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

        del opt_state, adversary
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
        if pgd_eval is not None:
            assert initial_ci_fn is not None and config.pgd_eval is not None
            t0 = time.time()
            pgd_final = pgd_recon(ci_fn)
            initial_params = jax.tree.map(
                lambda host, like: jax.device_put(host, like.sharding),
                initial_ci_fn,
                trainable(ci_fn),
            )
            pgd_initial = pgd_recon(cast(PlacedCIFn, eqx.combine(initial_params, ci_fn)))
            outputs.pgd.write_text(
                json.dumps(
                    {"initial": pgd_initial, "final": pgd_final, **config.pgd_eval.model_dump()}
                )
            )
            final = final.model_copy(update={"pgd_recon": pgd_final})
            outputs.summary.write_text(final.model_dump_json(indent=2))
            _wandb_log(
                {"eval/pgd_recon_initial": pgd_initial, "eval/pgd_recon": pgd_final}, config.steps
            )
            logger.info(
                f"PGD recon ({time.time() - t0:.0f}s): initial {pgd_initial:.4f}, final {pgd_final:.4f}"
            )
    if config.wandb is not None:
        assert wandb.run is not None
        wandb.run.summary.update({"outputs": str(outputs.root), "final/n_alive": final.n_alive})
        wandb.finish()
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
