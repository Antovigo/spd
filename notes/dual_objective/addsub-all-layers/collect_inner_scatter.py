#!/usr/bin/env python
"""Inner activation of every ALIVE component, original model vs decomposed model, with NO CI
filter — the whole decomposition, not a filter's alive set.

    python collect_inner_scatter.py --run-dir <run> --step N --data-root <root> --out <dir>
        [--threshold 0.01] [--batch-size 100] [--limit N] [--seed 0]
        [--operations add sub] [--a-range 1 100] [--b-range 1 100]

Same quantity as `param_decomp/ci_filter/scripts/collect_dataset.py` + `notes/ci_filter/
plot_inner_scatter.py`: a component's inner activation `x · V_c` at ONE (prompt, position)
entry drawn uniformly from those where it is ON, in the original model (x) and in the
decomposed model (y). "On" is the ORIGINAL model's output CI over `threshold`; the decomposed
forward turns on exactly those components, with the weight delta OFF.

TWO DIFFERENCES FROM THE FILTER VERSION, both because there is no filter here:
  * the candidate set is EVERY component of the decomposition, not a filter's alive set, and
    the CI is read unconstrained (no ceilings, no pruning);
  * a component with no on-entry anywhere in the pool is DEAD and is simply absent from the
    output — it never gets a sample.

It also samples as it goes instead of writing the full (prompt, position, component) tensors:
the filter dataset is 454 GB for 11.6k components, which at ~62k components would be tens of
terabytes. The reservoir is the same "uniform key per on-entry, largest key wins" draw the
plot script does, so the sample is uniform over on-entries either way. Output is one TSV plus
meta.json, a few MB.

GPU JOB: run it through SLURM.
"""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.config import ArithmeticPoolConfig
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    Prepared,
    gather_rows,
    output_ci,
    prepare_components,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import MaterializedMasking, PlacedModel, select_captures
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.arithmetic_eval import component_activation_model
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache


def make_collect(threshold: float):  # noqa: ANN201 (jitted closure)
    @eqx.filter_jit
    def collect(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
    ) -> tuple[dict[str, Array], dict[str, Array], dict[str, Array]]:
        """Per site: the original model's output CI and inner activations, and the decomposed
        model's inner activations, all `(B, T, C)` and replicated for the host."""
        tokens = gather_rows(tokens_all, idx)
        keys = frozenset(ci_fn.capture_keys)
        clean, clean_inner = placed.component_activation_forward(
            prepared, tokens, capture_keys=keys
        )
        lower = output_ci(
            placed, ci_fn, UNCONSTRAINED, select_captures(clean.captures, ci_fn.capture_keys), False
        ).lower
        masking = MaterializedMasking(
            component_masks={k: (v > threshold).astype(COMPUTE_DT) for k, v in lower.items()}
        )
        masked_inner = component_activation_model(placed).masked_component_activations(
            prepared, tokens, masking, placement=placed.placement
        )
        out = [
            {s: jax.sharding.reshard(v[s].astype(jnp.float32), P()) for s in placed.site_names}
            for v in (lower, clean_inner, masked_inner)
        ]
        return out[0], out[1], out[2]

    return collect


class Reservoir:
    """One uniformly drawn on-entry per component, streamed over batches. Each on-entry gets a
    uniform key and the largest key wins, which is the plot script's draw exactly."""

    def __init__(self, sites: dict[str, int], seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.best = {s: np.full(c, -1.0, np.float64) for s, c in sites.items()}
        self.n_on = {s: np.zeros(c, np.int64) for s, c in sites.items()}
        self.x = {s: np.full(c, np.nan, np.float32) for s, c in sites.items()}
        self.y = {s: np.full(c, np.nan, np.float32) for s, c in sites.items()}
        self.ci = {s: np.full(c, np.nan, np.float32) for s, c in sites.items()}

    def update(
        self, site: str, lower: np.ndarray, inner: np.ndarray, masked: np.ndarray, threshold: float
    ) -> None:
        c = lower.shape[-1]
        on = (lower > threshold).reshape(-1, c)
        self.n_on[site] += on.sum(0)
        keys = np.where(on, self.rng.random(on.shape), -1.0)
        arg = keys.argmax(0)
        cols = np.arange(c)
        won = keys[arg, cols] > self.best[site]
        self.best[site][won] = keys[arg, cols][won]
        self.x[site][won] = inner.reshape(-1, c)[arg, cols][won]
        self.y[site][won] = masked.reshape(-1, c)[arg, cols][won]
        self.ci[site][won] = lower.reshape(-1, c)[arg, cols][won]


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--operations", nargs="+", default=["add", "sub"])
    ap.add_argument("--a-range", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--b-range", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--cache-dir", type=Path, default=Path("~/.cache/param-decomp/xla"))
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # The pool is built here rather than read from a CIFilterConfig yaml: those carry filter
    # training fields, and an older filter's yaml no longer validates against the schema.
    pool_config = ArithmeticPoolConfig(
        operations=tuple(a.operations), a_range=tuple(a.a_range), b_range=tuple(a.b_range)
    )
    if a.cache_dir is not None:
        enable_persistent_compilation_cache(a.cache_dir.expanduser())
    restored = restore_jax_run(a.run_dir, a.step, data_root=a.data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    pool = build_pool(pool_config, load_tokenizer(target.model_name))
    n_prompts = pool.n_prompts if a.limit is None else min(a.limit, pool.n_prompts)
    logger.info(f"{a.run_dir.name} step {step}: {n_prompts} prompts, threshold {a.threshold}")

    with jax.set_mesh(mesh):
        prepared, v_norms = prepare_components(placed, restored.components)
        ci_fn = restored.ci_fn
        sites = {s: int(np.asarray(v_norms[s]).shape[0]) for s in placed.site_names}
        reservoir = Reservoir(sites, a.seed)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        collect = make_collect(a.threshold)
        starts = range(0, n_prompts, a.batch_size)
        for i, start in enumerate(starts):
            idx = jnp.arange(start, min(start + a.batch_size, n_prompts))
            lower, inner, masked = collect(placed, prepared, ci_fn, tokens_all, idx)
            for s in placed.site_names:
                reservoir.update(
                    s,
                    np.asarray(lower[s]),
                    np.asarray(inner[s]),
                    np.asarray(masked[s]),
                    a.threshold,
                )
            if i % 20 == 0:
                seen = sum(int((v > 0).sum()) for v in reservoir.n_on.values())
                logger.info(f"  batch {i + 1}/{len(starts)}: {seen} components on so far")

    a.out.mkdir(parents=True, exist_ok=True)
    rows = ["site\tlayer\tkind\tcomponent\tn_on\tci\toriginal\tdecomposed\tv_norm"]
    n_alive = 0
    for s in sorted(placed.site_names):
        layer, kind = s.split(".")[1], s.split(".", 2)[2]
        v_norm = np.asarray(v_norms[s], np.float32)
        for c in np.flatnonzero(reservoir.n_on[s] > 0):
            n_alive += 1
            rows.append(
                f"{s}\t{layer}\t{kind}\t{c}\t{reservoir.n_on[s][c]}\t{reservoir.ci[s][c]:.6g}\t"
                f"{reservoir.x[s][c]:.6g}\t{reservoir.y[s][c]:.6g}\t{v_norm[c]:.6g}"
            )
    (a.out / "inner_scatter.tsv").write_text("\n".join(rows) + "\n")
    total = sum(sites.values())
    meta = {
        "run_dir": str(a.run_dir),
        "step": int(step),
        "threshold": a.threshold,
        "n_prompts": int(n_prompts),
        "seq_len": int(pool.seq_len),
        "n_layer": len({s.split(".")[1] for s in placed.site_names}),
        "n_components": total,
        "n_alive": n_alive,
        "seed": a.seed,
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(f"{n_alive} of {total} components ever on -> {a.out / 'inner_scatter.tsv'}")


if __name__ == "__main__":
    main()
