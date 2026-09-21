"""Does subtracting one component from the MODEL improve its arithmetic?

    python -m param_decomp.ci_filter.scripts.model_ablation --config <yaml> --data_root <root>
        --table <components.tsv> [--min_layer 13] [--max_layer 17] [--min_impairs 0.8]
        [--prompts 4096]

The arithmetic sweep (`ablation_sweep.py`) masks components with the weight delta OFF, so it
scores the DECOMPOSITION. This script scores the model: every component on and the delta on —
which reproduces `x @ W` — minus the one component under test. For each candidate it reports the
change in the integer-renormalized `log p(correct answer)` and in accuracy, and how CONSISTENT
the change is (the share of prompts improved), which is the claim "this component impairs
arithmetic" stated about Llama rather than about its decomposition.

Writes `<run_dir>/analysis/ablations/step_<step>/model_ablation.tsv`."""

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

from param_decomp.ci_filter.attribution import answer_targets
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.nontarget import subtracting
from param_decomp.ci_filter.objective import answer_logprob_rows, restrict
from param_decomp.ci_filter.paths import ablations_dir
from param_decomp.ci_filter.pool import answer_token_ids, build_pool, load_tokenizer
from param_decomp.ci_filter.scripts.nontarget_probe import select_components
from param_decomp.ci_filter.step import Prepared, gather_rows, prepare_components
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache

COLUMNS = (
    "site",
    "layer",
    "kind",
    "component",
    "impairs_frac_run",
    "swept_danswer_ce_run",
    "d_logp",
    "d_accuracy",
    "improved_frac",
    "baseline_logp",
    "baseline_accuracy",
)


def make_scores() -> Callable[..., tuple[Float[Array, " B"], Float[Array, " B"]]]:
    @eqx.filter_jit
    def scores(
        placed: PlacedModel,
        prepared: Prepared,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        target_all: Int[Array, " N"],
        keep: dict[str, Array],
    ) -> tuple[Float[Array, " B"], Float[Array, " B"]]:
        """`(log p(correct), correct)` per prompt of the model minus the zeroed components."""
        tokens = gather_rows(tokens_all, idx)
        target_idx = target_all[idx]
        logits = subtracting(placed, prepared, tokens, keep)[:, -1, :]
        logp = answer_logprob_rows(logits, answer_ids, target_idx)
        correct = (jnp.argmax(restrict(logits, answer_ids), axis=-1) == target_idx).astype(
            jnp.float32
        )
        return jax.sharding.reshard(logp, P()), jax.sharding.reshard(correct, P())

    return scores


def model_ablation(
    config: CIFilterConfig,
    data_root: Path,
    table: Path,
    prompts: int,
    batch_size: int,
    min_layer: int,
    max_layer: int,
    min_impairs: float,
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    out = ablations_dir(run_dir, step) / "model_ablation.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(config.pool, tokenizer)
    answers = answer_token_ids(tokenizer, include_minus=True)
    targets = answer_targets(pool, tokenizer, answers)
    chosen = select_components(table, min_layer, max_layer, min_impairs)
    logger.info(f"{len(chosen)} components, {min(prompts, pool.n_prompts)} prompts -> {out}")

    with jax.set_mesh(mesh):
        prepared, _ = prepare_components(placed, restored.components)
        del restored
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())
        target_all = jax.sharding.reshard(jnp.asarray(targets.indices), P())
        ones = {
            site.name: jax.sharding.reshard(jnp.ones(site.C, jnp.float32), P())
            for site in placed.sites
        }
        rng = np.random.default_rng(np.random.SeedSequence((config.seed, 13)))
        subset = np.sort(
            rng.choice(pool.n_prompts, size=min(prompts, pool.n_prompts), replace=False)
        ).astype(np.int32)
        batches = [subset[s : s + batch_size] for s in range(0, subset.size, batch_size)]
        batches = [b for b in batches if b.size == batch_size]

        scores = make_scores()

        def over_pool(keep: dict[str, Array]) -> tuple[np.ndarray, np.ndarray]:
            logps, corrects = [], []
            for block in batches:
                logp, correct = scores(
                    placed, prepared, tokens_all, jnp.asarray(block), answer_ids, target_all, keep
                )
                logps.append(np.asarray(logp))
                corrects.append(np.asarray(correct))
            return np.concatenate(logps), np.concatenate(corrects)

        t0 = time.time()
        base_logp, base_correct = over_pool(ones)
        logger.info(
            f"model baseline on {base_logp.size} prompts ({time.time() - t0:.0f}s): "
            f"log p(correct) {base_logp.mean():.4f}, accuracy {base_correct.mean():.4f}"
        )

        rows: list[str] = ["\t".join(COLUMNS)]
        for row in chosen:
            site, component = row["site"], int(row["component"])
            keep = dict(ones)
            keep[site] = ones[site].at[component].set(0.0)
            logp, correct = over_pool(keep)
            values = {
                "site": site,
                "layer": row["layer"],
                "kind": row["kind"],
                "component": component,
                "impairs_frac_run": row["impairs_frac_run"],
                "swept_danswer_ce_run": row["swept_danswer_ce_run"],
                "d_logp": f"{logp.mean() - base_logp.mean():.6g}",
                "d_accuracy": f"{correct.mean() - base_correct.mean():.6g}",
                "improved_frac": f"{float((logp > base_logp).mean()):.4g}",
                "baseline_logp": f"{base_logp.mean():.6g}",
                "baseline_accuracy": f"{base_correct.mean():.6g}",
            }
            rows.append("\t".join(str(values[c]) for c in COLUMNS))
            logger.info(
                f"{site} c{component}: d log p {values['d_logp']}, "
                f"d accuracy {values['d_accuracy']}, improved on {values['improved_frac']}"
            )
    out.write_text("\n".join(rows) + "\n")
    logger.info(f"-> {out}")
    return out


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--table", type=Path, required=True, help="components.tsv")
    ap.add_argument("--prompts", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--min_layer", type=int, default=13)
    ap.add_argument("--max_layer", type=int, default=17)
    ap.add_argument("--min_impairs", type=float, default=0.8)
    args = ap.parse_args()
    model_ablation(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.table,
        args.prompts,
        args.batch_size,
        args.min_layer,
        args.max_layer,
        args.min_impairs,
    )


if __name__ == "__main__":
    main()
