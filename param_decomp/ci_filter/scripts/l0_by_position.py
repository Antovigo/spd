"""L0 per layer, matrix kind and prompt position, for the decomposition and every active filter.

    python -m param_decomp.ci_filter.scripts.l0_by_position --config <yaml> --data_root <root>
        [--prompts 4000] [--batch_size 250]

L0 here is the mean number of components with output CI above `alive_threshold` at one position
of one prompt, per site. Sources: `run` (the decomposition's own CI fn) and every filtering run
under `ci_filter/step_<step>/` outside `Trash/`, each with the constraints it trained under
(ceilings and removed components). The prompt subset is fixed by the seed, so every source is
scored on the same prompts.

Writes `<run_dir>/analysis/ci_filter/step_<step>/l0_by_position.tsv`: one row per
`(source, site, position)` with the mean L0."""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.paths import TRASH, ci_filter_dir
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    CIConstraints,
    gather_rows,
    output_ci,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache


def make_site_l0(threshold: float) -> Callable[..., dict[str, Float[Array, " T"]]]:
    """`{site: (T,)}`: components above `threshold`, summed over the site's C and over the
    batch's prompts (the caller divides by the prompt count)."""

    @eqx.filter_jit
    def site_l0(
        placed: PlacedModel,
        ci_fn: PlacedCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
    ) -> dict[str, Float[Array, " T"]]:
        tokens = gather_rows(tokens_all, idx)
        captures = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys).captures
        lower = output_ci(placed, ci_fn, constraints, captures, remat=False).lower
        return {
            site: jax.sharding.reshard(
                jnp.sum(value > threshold, axis=(0, 2), dtype=jnp.float32), P()
            )
            for site, value in lower.items()
        }

    return site_l0


def l0_by_position(config: CIFilterConfig, data_root: Path, prompts: int, batch_size: int) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out = ci_filter_dir(run_dir, step) / "l0_by_position.tsv"
    pool = build_pool(config.pool, load_tokenizer(target.model_name))
    filters = sorted(
        d.name
        for d in ci_filter_dir(run_dir, step).iterdir()
        if d.name != TRASH and (d / "training" / "ci_fn").exists()
    )
    sources = ["run", *filters]
    logger.info(f"L0 by position for {sources} on {prompts} prompts -> {out}")

    rng = np.random.default_rng(np.random.SeedSequence((config.seed, 17)))
    subset = np.sort(rng.choice(pool.n_prompts, size=min(prompts, pool.n_prompts), replace=False))
    batches = [subset[s : s + batch_size] for s in range(0, subset.size, batch_size)]
    batches = [b.astype(np.int32) for b in batches if b.size == batch_size]
    n_scored = len(batches) * batch_size

    rows = ["\t".join(["source", "site", "layer", "kind", "position", "label", "l0"])]
    with jax.set_mesh(mesh):
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        site_l0 = make_site_l0(config.alive_threshold)
        for source in sources:
            t0 = time.time()
            ci_fn, constraints = (
                (restored.ci_fn, UNCONSTRAINED)
                if source == "run"
                else trained_ci(run_dir, step, source, restored.ci_fn)
            )
            totals: dict[str, np.ndarray] = {}
            for block in batches:
                got = site_l0(placed, ci_fn, constraints, tokens_all, jnp.asarray(block))
                for site, value in got.items():
                    totals[site] = totals.get(site, 0.0) + np.asarray(value)
            for site in sorted(totals):
                layer, kind = site.split(".", 2)[1], site.split(".", 2)[2]
                for position, value in enumerate(totals[site] / n_scored):
                    rows.append(
                        "\t".join(
                            [
                                source,
                                site,
                                layer,
                                kind,
                                str(position),
                                pool.position_labels[position],
                                f"{value:.4f}",
                            ]
                        )
                    )
            mean_total = sum(float(v.sum()) for v in totals.values()) / n_scored / pool.seq_len
            logger.info(f"{source}: L0 per token {mean_total:.1f} ({time.time() - t0:.0f}s)")
            del ci_fn, constraints
    out.write_text("\n".join(rows) + "\n")
    logger.info(f"-> {out}")
    return out


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="any filter config (run, pool)")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=250)
    args = ap.parse_args()
    l0_by_position(
        CIFilterConfig.from_file(args.config), args.data_root, args.prompts, args.batch_size
    )


if __name__ == "__main__":
    main()
