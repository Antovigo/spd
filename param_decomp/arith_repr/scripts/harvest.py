"""Post-norm residuals of the frozen target at every read point and position of the
arithmetic pool, on one GPU.

    python -m param_decomp.arith_repr.scripts.harvest --run_dir <run> --data_root <root>
        --out_dir <dir> [--batch 250]

Loads ONLY the frozen target (no components, no CI fn). Writes `pool.json` (prompt labels,
positions) and, per read point, `attn_in.<l>.npy` / `mlp_in.<l>.npy` of shape
`(n_prompts, T, 4096)` in bfloat16-as-uint16 (`view(bfloat16)` on read with ml_dtypes)."""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.config import ArithmeticPoolConfig
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import gather_rows, index_batches
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.sharding import hsdp_mesh
from param_decomp.experiments.lm.deliverable import load_deliverable
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.targets.transformer_taps import attention_input_tap_key, mlp_input_tap_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=250)
    args = parser.parse_args()
    setup_console_logger()

    deliverable = load_deliverable(args.run_dir, args.data_root)
    target = deliverable.target
    assert isinstance(target, TargetConfig)
    mesh = hsdp_mesh(1, jax.device_count(), 1)
    placed = build_target(target, mesh, args.data_root, "ddp" if mesh.size == 1 else "zero1")
    n_layer = len({int(s.split(".")[1]) for s in placed.site_names})
    logger.info(f"target {target.model_name}: {n_layer} layers, {mesh.size} device(s)")

    tokenizer = load_tokenizer(target.model_name)
    pool = build_pool(ArithmeticPoolConfig(), tokenizer)
    n, seq_len = pool.tokens.shape
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for block in pool.blocks:
        for a in block.grid.a_values:
            for b in block.grid.b_values:
                labels.append((block.operation, int(a), int(b)))
    (args.out_dir / "pool.json").write_text(
        json.dumps(
            {
                "position_labels": pool.position_labels,
                "labels": labels,
                "n_layer": n_layer,
                "d_model": None,
            }
        )
    )
    keys = frozenset(
        [attention_input_tap_key(layer) for layer in range(n_layer)]
        + [mlp_input_tap_key(layer) for layer in range(n_layer)]
    )

    @eqx.filter_jit
    def captures(tokens_all: Int[Array, "N T"], idx: Int[Array, " B"]) -> dict[str, Array]:
        tokens = gather_rows(tokens_all, idx)
        result = placed.clean_forward(tokens, capture_keys=keys)
        return {
            key: jax.sharding.reshard(value.astype(jnp.bfloat16), P())
            for key, value in result.captures.items()
        }

    batches = index_batches(n, args.batch)
    memmaps: dict[str, np.memmap] = {}
    with jax.set_mesh(mesh):
        tokens_all = jnp.asarray(pool.tokens)
        t0 = time.time()
        for i, (idx, real) in enumerate(batches):
            got = captures(tokens_all, jnp.asarray(idx))
            for key, value in got.items():
                arr = np.asarray(value)[:real]  # bfloat16 (ml_dtypes) numpy array
                if key not in memmaps:
                    d_model = arr.shape[-1]
                    memmaps[key] = np.lib.format.open_memmap(
                        args.out_dir / f"{key}.npy",
                        mode="w+",
                        dtype=np.uint16,
                        shape=(n, seq_len, d_model),
                    )
                start = int(idx[0])
                memmaps[key][start : start + real] = arr.view(np.uint16)
            if i % 10 == 0:
                logger.info(f"batch {i + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    for m in memmaps.values():
        m.flush()
    meta = json.loads((args.out_dir / "pool.json").read_text())
    meta["d_model"] = int(next(iter(memmaps.values())).shape[-1])
    meta["keys"] = sorted(memmaps)
    (args.out_dir / "pool.json").write_text(json.dumps(meta))
    logger.info(f"done: {len(memmaps)} read points x {n} prompts x {seq_len} positions")


if __name__ == "__main__":
    main()
