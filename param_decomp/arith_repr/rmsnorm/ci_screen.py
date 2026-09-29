"""Screen for the non-target study of the temperature components: the -05 run's OWN output CI (lower
leaky, no filter) of every component of blocks 30-31 on pre-tokenized fineweb, plus those blocks' full
U / V (every component, alive on addsub or not) for the dense ablations of `temperature.py`.

    python -m param_decomp.arith_repr.rmsnorm.ci_screen <n_rows> <batch>

Writes `OUT/fineweb/`: `tokens.npy` (n_rows, 64), `ci_L{30,31}.npy` (n_rows, 64, C_layer) float16 in
site order `SITES`, `uv_L{30,31}.npz` (`<kind>.U`, `<kind>.V` float32)."""

import sys
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from jax.sharding import PartitionSpec as P

from param_decomp.arith_repr.isa.components_model import KINDS
from param_decomp.arith_repr.rmsnorm.run import OUT, savez
from param_decomp.arith_repr.vectors.common import RUN
from param_decomp.ci_filter.step import UNCONSTRAINED, output_ci
from param_decomp.experiments.lm.load_run import restore_jax_run

DATA_ROOT = Path("/mnt/nw/home/a.vigouroux/out")
DATASET = DATA_ROOT / "datasets/fineweb_llama_tok_64_eval"
LAYERS = (30, 31)


def text_rows(n: int) -> np.ndarray:
    f = pq.ParquetFile(sorted(DATASET.glob("shard_*.parquet"))[0])
    rows, got = [], 0
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=["input_ids"])
        rows.append(np.stack([np.asarray(x, np.int32) for x in t["input_ids"].to_pylist()]))
        got += len(rows[-1])
        if got >= n:
            break
    return np.concatenate(rows)[:n]


def main(n_rows: int, batch: int) -> None:
    out = OUT / "fineweb"
    out.mkdir(parents=True, exist_ok=True)
    tokens = text_rows(n_rows)
    np.save(out / "tokens.npy", tokens)
    print("tokens", tokens.shape, "first tokens", tokens[:3, :4].tolist(), flush=True)
    restored = restore_jax_run(RUN, 40000, data_root=DATA_ROOT)
    placed, ci_fn = restored.placed, restored.ci_fn
    sites = {li: [f"layers.{li}.{KINDS[k]}" for k in KINDS] for li in LAYERS}
    with jax.set_mesh(restored.mesh):
        for li in LAYERS:
            uv = {}
            for k, name in zip(KINDS, sites[li], strict=False):
                sc = restored.components.site(name)
                uv[f"{k}.V"] = np.asarray(jax.sharding.reshard(sc.V.astype(jnp.float32), P()))
                uv[f"{k}.U"] = np.asarray(jax.sharding.reshard(sc.U.astype(jnp.float32), P()))
            savez(out / f"uv_L{li}.npz", **uv)
            print("saved uv", li, {k: v.shape for k, v in uv.items()}, flush=True)
        del restored

        @eqx.filter_jit
        def ci_of(placed: Any, ci_fn: Any, toks: jax.Array) -> dict[int, jax.Array]:  # noqa: ANN401
            captures = placed.clean_forward(toks, capture_keys=ci_fn.capture_keys).captures
            lower = output_ci(placed, ci_fn, UNCONSTRAINED, captures, remat=False).lower
            return {li: jax.sharding.reshard(jnp.concatenate([lower[s] for s in sites[li]], -1).astype(jnp.float16), P())
                    for li in LAYERS}  # fmt: skip

        cis: dict[int, list[np.ndarray]] = {li: [] for li in LAYERS}
        t0 = time.time()
        for s in range(0, n_rows, batch):
            got = ci_of(placed, ci_fn, jnp.asarray(tokens[s : s + batch]))
            for li in LAYERS:
                cis[li].append(np.asarray(got[li]))
            print(f"ci {s + batch}/{n_rows} ({time.time() - t0:.0f}s)", flush=True)
    for li in LAYERS:
        np.save(out / f"ci_L{li}.npy", np.concatenate(cis[li]))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
