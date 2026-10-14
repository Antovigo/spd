"""`extract_layer.py` for every block in one pass over the pool (one model load, one forward).

Same rounded-CI decomposed model of the -08 CI filter. For each block l it writes, in the layout
`attribute.py` and `build_app.py` read:

    <run>/analysis/reader_writer/L<l>/inner.npy  (N, T, K_l) float16  a_c (before the mask)
    <run>/analysis/reader_writer/L<l>/masks.npy  (N, T, K_l) bool     m_c
    <run>/analysis/reader_writer/L<l>/comps.npz  kind, index (K_l,); U_<kind>, V_<kind> float32
    <run>/analysis/reader_writer/L<l>/meta.json

Blocks whose folder already holds meta.json are skipped (L18 comes from extract_layer.py, float32
inner and a check.npz). The cached filter dataset (dataset/decomposed/inner_pos<t>.npy, H100 run)
is not reused: on L18 it differs from this L40 extraction by 1.5 % relative on the on entries.

    python extract_all_layers.py
"""

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from extract_layer import DATA_ROOT, FILTER, KINDS, RUN_DIR, STEP
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
from param_decomp.core.model import MaterializedMasking, select_captures
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.arithmetic_eval import component_activation_model
from param_decomp.experiments.lm.load_run import restore_jax_run

N_LAYER = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    root = RUN_DIR / "analysis" / "reader_writer"

    outputs = CIFilterOutputs.for_run(RUN_DIR, STEP, FILTER)
    config = CIFilterConfig.from_file(outputs.config)
    threshold = config.alive_threshold
    alive_json = json.loads(outputs.alive.read_text())
    layers = [b for b in range(N_LAYER) if not (root / f"L{b}" / "meta.json").exists()]
    print(f"blocks to extract: {layers}", flush=True)
    sites = {(b, k): f"layers.{b}.mlp.{k}_proj" for b in layers for k in KINDS}
    alive = {bk: np.asarray(alive_json["components"][s], np.int32) for bk, s in sites.items()}

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
        for b in layers:
            (root / f"L{b}").mkdir(parents=True, exist_ok=True)
            uv = {}
            for k in KINDS:
                site = components.site(sites[(b, k)])
                uv[f"U_{k}"] = np.asarray(site.U, np.float32)[alive[(b, k)]]
                uv[f"V_{k}"] = np.asarray(site.V, np.float32)[:, alive[(b, k)]]
            np.savez(root / f"L{b}" / "comps.npz",
                     kind=np.asarray([k for k in KINDS for _ in alive[(b, k)]]),
                     index=np.concatenate([alive[(b, k)] for k in KINDS]), **uv)  # fmt: skip
        prepared, _ = prepare_components(placed, components)
        del components
        cols = {bk: jax.sharding.reshard(jnp.asarray(v), P()) for bk, v in alive.items()}
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
            masking = MaterializedMasking(
                component_masks={s: (v > threshold).astype(COMPUTE_DT) for s, v in lower.items()}
            )
            inner = component_activation_model(placed).masked_component_activations(
                prepared, tokens, masking, placement=placed.placement
            )

            def pick(values: dict, b: int, dtype) -> Array:  # noqa: ANN001
                return jnp.concatenate(
                    [jnp.take(jax.sharding.reshard(values[sites[(b, k)]], P()), cols[(b, k)], axis=-1)
                     for k in KINDS], axis=-1).astype(dtype)  # fmt: skip

            on = {s: v > threshold for s, v in lower.items()}
            return [
                (jax.sharding.reshard(pick(inner, b, jnp.float16), P()),
                 jax.sharding.reshard(pick(on, b, jnp.bool_), P()))
                for b in layers
            ]  # fmt: skip

        widths = {b: sum(alive[(b, k)].size for k in KINDS) for b in layers}
        mm = {
            b: (open_memmap(root / f"L{b}" / "inner.npy", mode="w+", dtype=np.float16, shape=(n, T, widths[b])),
                open_memmap(root / f"L{b}" / "masks.npy", mode="w+", dtype=np.bool_, shape=(n, T, widths[b])))
            for b in layers
        }  # fmt: skip
        absmax = 0.0
        t0 = time.time()
        starts = list(range(0, n, args.batch_size))
        for i, start in enumerate(starts):
            stop = min(start + args.batch_size, n)
            idx = np.arange(start, start + args.batch_size, dtype=np.int32) % pool.n_prompts
            got = collect(placed, prepared, ci_fn, constraints, cols, tokens_all, jnp.asarray(idx))
            for b, (inn, on) in zip(layers, got, strict=True):
                inn = np.asarray(inn)[: stop - start]
                absmax = max(absmax, float(np.abs(inn.astype(np.float32)).max()))
                mm[b][0][start:stop] = inn
                mm[b][1][start:stop] = np.asarray(on)[: stop - start]
            if i % 50 == 0 or stop == n:
                el = time.time() - t0
                print(f"{stop}/{n} {el:.0f}s eta {el / (i + 1) * (len(starts) - i - 1):.0f}s "
                      f"inner absmax {absmax:.1f}", flush=True)  # fmt: skip
        assert absmax < 65504, f"inner overflows float16 ({absmax})"
        for inn, on in mm.values():
            inn.flush()
            on.flush()
    for b in layers:
        meta = {"run_dir": str(RUN_DIR), "step": STEP, "filter": FILTER, "layer": b,
                "threshold": threshold, "n_prompts": n, "T": T, "inner_dtype": "float16",
                "n_alive": {k: int(alive[(b, k)].size) for k in KINDS},
                "seconds": time.time() - t0}  # fmt: skip
        (root / f"L{b}" / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done {layers} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
