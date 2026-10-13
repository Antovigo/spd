"""Masks of every block's alive gate / up / down components in the -08 filter's decomposed model.

Same model as `extract_layer.py` (on each prompt and position, exactly the components whose
output CI on the ORIGINAL model exceeds the filter's alive_threshold are on; the weight delta is
off), for all 32 blocks in one pass. Only the masks are written: the inner activations come from
the filter's cached dataset (`dataset/decomposed/inner_pos<t>.npy`, same model, float16).

    OUT/masks_L<l>.npy  (N, T, ceil(K_l / 8)) uint8, np.packbits(m, axis=-1) of the bool masks
                        (N, T, K_l); columns in comps_L<l>.npz order (gate, up, down)
    OUT/comps_L<l>.npz  kind, index (K_l,); U_<kind>, V_<kind> float32 of the alive components
    OUT/meta.json

OUT = <run>/analysis/reader_writer/all_layers.

    python extract_masks_all.py
"""

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from extract_layer import DATA_ROOT, FILTER, KINDS, RUN_DIR, STEP  # noqa: E402
from jax import Array
from jax.sharding import PartitionSpec as P
from numpy.lib.format import open_memmap

from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    gather_rows,
    output_ci,
    prepare_components,
    remove_components,
)
from param_decomp.core.model import select_captures
from param_decomp.experiments.lm.load_run import restore_jax_run

N_LAYER = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = RUN_DIR / "analysis" / "reader_writer" / "all_layers"
    out.mkdir(parents=True, exist_ok=True)

    outputs = CIFilterOutputs.for_run(RUN_DIR, STEP, FILTER)
    config = CIFilterConfig.from_file(outputs.config)
    threshold = config.alive_threshold
    alive_json = json.loads(outputs.alive.read_text())
    sites = {
        (layer_i, k): f"layers.{layer_i}.mlp.{k}_proj" for layer_i in range(N_LAYER) for k in KINDS
    }
    alive = {lk: np.asarray(alive_json["components"][s], np.int32) for lk, s in sites.items()}

    restored = restore_jax_run(RUN_DIR, STEP, data_root=DATA_ROOT)
    placed, mesh = restored.placed, restored.mesh
    pool = build_pool(config.pool, load_tokenizer(restored.deliverable.target.model_name))
    n = pool.n_prompts if args.limit is None else args.limit
    T = pool.seq_len

    with jax.set_mesh(mesh):
        ci_fn, constraints = trained_ci(RUN_DIR, STEP, FILTER, restored.ci_fn)
        components = (
            restored.components
            if constraints.kept is None
            else remove_components(restored.components, constraints.kept)
        )
        for layer_i in range(N_LAYER):
            uv = {}
            for k in KINDS:
                site = components.site(sites[(layer_i, k)])
                uv[f"U_{k}"] = np.asarray(site.U, np.float32)[alive[(layer_i, k)]]
                uv[f"V_{k}"] = np.asarray(site.V, np.float32)[:, alive[(layer_i, k)]]
            np.savez(out / f"comps_L{layer_i}.npz",
                     kind=np.asarray([k for k in KINDS for _ in alive[(layer_i, k)]]),
                     index=np.concatenate([alive[(layer_i, k)] for k in KINDS]), **uv)  # fmt: skip
        prepared, _ = prepare_components(placed, components)
        del components
        cols = {lk: jax.sharding.reshard(jnp.asarray(v), P()) for lk, v in alive.items()}
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())

        @eqx.filter_jit
        def collect(placed, prepared, ci_fn, constraints, cols, tokens_all, idx):  # noqa: ANN001, ANN202
            tokens = gather_rows(tokens_all, idx)
            clean, _ = placed.component_activation_forward(
                prepared, tokens, capture_keys=frozenset(ci_fn.capture_keys)
            )
            lower = output_ci(
                placed,
                ci_fn,
                constraints,
                select_captures(clean.captures, ci_fn.capture_keys),
                False,
            ).lower

            def layer(layer_i: int) -> Array:
                on = jnp.concatenate(
                    [jnp.take(jax.sharding.reshard(lower[sites[(layer_i, k)]], P()), cols[(layer_i, k)], axis=-1)
                     for k in KINDS], axis=-1) > threshold  # fmt: skip
                return jax.sharding.reshard(jnp.packbits(on, axis=-1), P())

            return [layer(layer_i) for layer_i in range(N_LAYER)]

        widths = [sum(alive[(layer_i, k)].size for k in KINDS) for layer_i in range(N_LAYER)]
        mms = [open_memmap(out / f"masks_L{layer_i}.npy", mode="w+", dtype=np.uint8,
                           shape=(n, T, (widths[layer_i] + 7) // 8)) for layer_i in range(N_LAYER)]  # fmt: skip
        t0 = time.time()
        starts = list(range(0, n, args.batch_size))
        for i, start in enumerate(starts):
            stop = min(start + args.batch_size, n)
            idx = np.arange(start, start + args.batch_size, dtype=np.int32) % pool.n_prompts
            got = collect(placed, prepared, ci_fn, constraints, cols, tokens_all, jnp.asarray(idx))
            for mm, g in zip(mms, got, strict=True):
                mm[start:stop] = np.asarray(g)[: stop - start]
            if i % 50 == 0 or stop == n:
                el = time.time() - t0
                print(f"{stop}/{n} {el:.0f}s eta {el / (i + 1) * (len(starts) - i - 1):.0f}s",
                      flush=True)  # fmt: skip
        for mm in mms:
            mm.flush()
    meta = {"run_dir": str(RUN_DIR), "step": STEP, "filter": FILTER, "threshold": threshold,
            "n_prompts": n, "T": T, "widths": widths, "seconds": time.time() - t0,
            "inner_source": str(outputs.config.parent / "dataset" / "decomposed")}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
