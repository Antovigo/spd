#!/usr/bin/env python
"""Pre-RMSNorm logit of the original model's top-1 next token, decomposed vs original.

    python collect_logit_shift.py --run-dir <run> --step N --data-root <root> --out <dir>
        [--threshold 0.0] [--batch-size 100] [--limit N]
        [--operations add sub] [--a-range 1 100] [--b-range 1 100]

For every prompt in the pool, at the ANSWER position (the last one):

  1. the original model's most likely next token, `t = argmax logits`;
  2. that token's PRE-RMSNORM logit in both models — the residual leaving the last block
     (`resid_tap_key(n_layer)`, before the final norm) dotted with the unembedding row
     `head_weight[t]`. Taking it before the norm is the point: the final RMSNorm rescales
     the whole residual, so a decomposition that shrinks or grows it uniformly looks
     unchanged downstream (see the logit-magnitude work on the L18 line);
  3. the same token's ordinary POST-norm logit in both models, for reference.

The DECOMPOSED model is the rounded-mask forward: components whose ORIGINAL-model output CI
exceeds `threshold` are on, everything else and the weight delta are off. The default
threshold 0.0 matches what these runs call "rounded" in their own eval
(`CEandKLLosses.rounding_threshold: 0.0`, both configs), i.e. any CI above zero is on.

Writes one row per prompt to `<out>/logit_shift.tsv` plus meta.json.

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
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.targets.transformer_taps import resid_tap_key


def make_collect(threshold: float, resid_key: str):  # noqa: ANN201 (jitted closure)
    @eqx.filter_jit
    def collect(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
    ) -> dict[str, Array]:
        tokens = gather_rows(tokens_all, idx)
        keys = frozenset(ci_fn.capture_keys) | {resid_key}
        clean, _ = placed.component_activation_forward(prepared, tokens, capture_keys=keys)
        lower = output_ci(
            placed, ci_fn, UNCONSTRAINED, select_captures(clean.captures, ci_fn.capture_keys), False
        ).lower
        masking = MaterializedMasking(
            component_masks={k: (v > threshold).astype(COMPUTE_DT) for k, v in lower.items()}
        )
        masked = placed.masked_forward(
            prepared, tokens, masking=masking, capture_keys=keys, remat=False
        )

        clean_last = clean.output[:, -1].astype(jnp.float32)
        masked_last = masked.output[:, -1].astype(jnp.float32)
        token = jnp.argmax(clean_last, axis=-1)
        # One-hot contractions rather than gathers: `token` carries the batch sharding while
        # the unembedding and logits do not, and an explicit-sharding gather across the two
        # cannot resolve its output spec (ShardingTypeError on the vocab gather).
        onehot = jax.nn.one_hot(token, clean_last.shape[-1], dtype=jnp.float32)
        rows = jnp.einsum("bv,vd->bd", onehot, placed.model.head_weight.astype(jnp.float32))
        pre = {
            name: jnp.einsum("bd,bd->b", out.captures[resid_key][:, -1].astype(jnp.float32), rows)
            for name, out in (("original", clean), ("decomposed", masked))
        }
        out = {
            "token": token,
            "pre_original": pre["original"],
            "pre_decomposed": pre["decomposed"],
            "post_original": jnp.einsum("bv,bv->b", clean_last, onehot),
            "post_decomposed": jnp.einsum("bv,bv->b", masked_last, onehot),
            "n_on": sum(jnp.sum(v > threshold, dtype=jnp.int32) for v in lower.values()),
        }
        return {k: jax.sharding.reshard(v, P()) for k, v in out.items()}

    return collect


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--threshold", type=float, default=0.0, help="rounded-mask CI threshold")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    ap.add_argument("--operations", nargs="+", default=["add", "sub"])
    ap.add_argument("--a-range", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--b-range", type=int, nargs=2, default=[1, 100])
    ap.add_argument("--cache-dir", type=Path, default=Path("~/.cache/param-decomp/xla"))
    a = ap.parse_args()

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
    n_layer = len({s.split(".")[1] for s in placed.site_names})
    resid_key = resid_tap_key(n_layer)  # leaves the last block, before the final RMSNorm
    logger.info(f"{a.run_dir.name} step {step}: {n_prompts} prompts, rounded at CI > {a.threshold}")

    cols: dict[str, list[np.ndarray]] = {}
    with jax.set_mesh(mesh):
        prepared, _ = prepare_components(placed, restored.components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        collect = make_collect(a.threshold, resid_key)
        starts = range(0, n_prompts, a.batch_size)
        for i, start in enumerate(starts):
            idx = jnp.arange(start, min(start + a.batch_size, n_prompts))
            got = collect(placed, prepared, restored.ci_fn, tokens_all, idx)
            for k, v in got.items():
                cols.setdefault(k, []).append(np.asarray(v))
            if i % 40 == 0:
                logger.info(f"  batch {i + 1}/{len(starts)}")

    data = {k: np.concatenate([np.atleast_1d(x) for x in v]) for k, v in cols.items()}
    a.out.mkdir(parents=True, exist_ok=True)
    shift = data["pre_decomposed"] - data["pre_original"]
    rows = ["token\tpre_original\tpre_decomposed\tpost_original\tpost_decomposed"]
    rows += [
        f"{t}\t{po:.6g}\t{pd:.6g}\t{qo:.6g}\t{qd:.6g}"
        for t, po, pd, qo, qd in zip(
            data["token"],
            data["pre_original"],
            data["pre_decomposed"],
            data["post_original"],
            data["post_decomposed"],
            strict=True,
        )
    ]
    (a.out / "logit_shift.tsv").write_text("\n".join(rows) + "\n")
    meta = {
        "run_dir": str(a.run_dir),
        "step": int(step),
        "threshold": a.threshold,
        "n_prompts": int(n_prompts),
        "n_layer": n_layer,
        "resid_key": resid_key,
        "pre_shift_mean": float(shift.mean()),
        "pre_shift_median": float(np.median(shift)),
        "pre_shift_p05": float(np.percentile(shift, 5)),
        "pre_shift_p95": float(np.percentile(shift, 95)),
        "mean_components_on_per_batch": float(np.mean(data["n_on"])),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info(
        f"pre-RMSNorm shift: mean {shift.mean():.4g}, median {np.median(shift):.4g}, "
        f"5-95% [{np.percentile(shift, 5):.4g}, {np.percentile(shift, 95):.4g}] -> {a.out}"
    )


if __name__ == "__main__":
    main()
