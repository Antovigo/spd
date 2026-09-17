"""Mask ablations (`param_decomp/ci_filter/ablation.py`) of a CI fn over the arithmetic pool.

The CI fn is the decomposition's own (`--source run`) or a CI filter's (`--source cf-xxxxxxxx`).
The run, pool, evaluation batch and alive threshold come from a `CIFilterConfig`. Writes
`<run_dir>/analysis/ci_filter/step_<step>/mask_ablations/<source>.json`."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.ablation import MASKINGS, ablation_rows
from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    index_batches,
    make_eval_batch,
    prepare_components,
    remove_components,
)
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache


def mask_ablations(config: CIFilterConfig, data_root: Path, source: str) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out_path = (
        run_dir / "analysis" / "ci_filter" / f"step_{step}" / "mask_ablations" / f"{source}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool = build_pool(config.pool, load_tokenizer(target.model_name))
    threshold = config.alive_threshold

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
        answer_ids = jax.sharding.reshard(jnp.asarray(np.zeros(1, np.int32)), P())
        batches = index_batches(pool.n_prompts, config.eval_batch_size)

        t0 = time.time()
        eval_batch = make_eval_batch(config)
        site_max: dict[str, np.ndarray] = {}
        for idx, _ in batches:
            _, smax, _ = eval_batch(
                placed, prepared, ci_fn, constraints, tokens_all, jnp.asarray(idx), answer_ids
            )
            for site, v in smax.items():
                site_max[site] = np.maximum(site_max.get(site, 0.0), np.asarray(v))
        alive_np = {site: (m > threshold).astype(np.float32) for site, m in site_max.items()}
        n_alive = int(sum(a.sum() for a in alive_np.values()))
        logger.info(f"{source}: {n_alive} alive at CI > {threshold} ({time.time() - t0:.0f}s)")

        alive = {k: jnp.asarray(v) for k, v in alive_np.items()}
        rows: dict[str, dict[str, list[np.ndarray]]] = {
            m: {"last": [], "all_positions": []} for m in MASKINGS
        }
        for idx, real in batches:
            got = ablation_rows(
                placed, prepared, ci_fn, constraints, alive, tokens_all, jnp.asarray(idx), threshold
            )
            for m, reductions in got.items():
                for r, v in reductions.items():
                    rows[m][r].append(np.asarray(v)[:real])
    result = {
        "source": source,
        "threshold": threshold,
        "n_prompts": pool.n_prompts,
        "n_alive": n_alive,
        "n_components": int(sum(a.size for a in alive_np.values())),
        "kl": {
            m: {r: float(np.concatenate(v).mean()) for r, v in reductions.items()}
            for m, reductions in rows.items()
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(f"{json.dumps(result, indent=2)}\n-> {out_path}")
    return out_path


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", type=Path, required=True, help="CIFilterConfig YAML (run, pool, batch)"
    )
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--source", default="run", help="`run` or a CI filter id")
    args = ap.parse_args()
    mask_ablations(CIFilterConfig.from_file(args.config), args.data_root, args.source)


if __name__ == "__main__":
    main()
