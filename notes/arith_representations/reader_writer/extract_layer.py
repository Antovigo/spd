"""One MLP block's gate / up / down components in the rounded-CI decomposed model of a CI filter.

The decomposed model is the filter's (as in `ci_filter/scripts/collect_dataset.py`): on each
prompt and position, exactly the components whose output CI on the ORIGINAL model exceeds the
filter's `alive_threshold` are on; every other component and the weight delta are off. In that
model block `l`'s MLP is, at every position, with x = the RMSNorm'd residual stream
(d_model = 4096) and h the neuron values (d_mlp = 14336):

    gate_pre = sum_c m_c a_c U_c   over gate components c, a_c = x . V_c (inner activation)
    up_pre   = sum_c m_c a_c U_c   over up components
    h        = silu(gate_pre) * up_pre
    a_w      = h . V_w             (inner activation of down component w)

m_c in {0, 1} is the component's mask. This job writes, for the block's ALIVE gate/up/down
components (the filter's alive set), every prompt and position:

    OUT/inner.npy  (N, T, K) float32  a_c in the decomposed model (before the mask)
    OUT/masks.npy  (N, T, K) bool     m_c
    OUT/comps.npz  kind (K,) in {gate, up, down}, index (K,) component index in its site,
                   U_<kind> (K_kind, d_out), V_<kind> (d_in, K_kind) float32
    OUT/check.npz  gate_pre, up_pre, hidden (N_check, T, d_mlp) float16 captured in the
                   decomposed forward, for the first N_check prompts (to verify the identity)
    OUT/meta.json

    python extract_layer.py --layer 18
"""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from param_decomp.targets.transformer_taps import mlp_hidden_tap_key, site_output_tap_key

RUN_DIR = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
DATA_ROOT = Path("/mnt/nw/home/a.vigouroux/out")
STEP = 40000
FILTER = "addsub-08-last-pos"
KINDS = ("gate", "up", "down")
N_CHECK = 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    layer = args.layer
    out = RUN_DIR / "analysis" / "reader_writer" / f"L{layer}"
    out.mkdir(parents=True, exist_ok=True)

    outputs = CIFilterOutputs.for_run(RUN_DIR, STEP, FILTER)
    config = CIFilterConfig.from_file(outputs.config)
    threshold = config.alive_threshold
    alive_json = json.loads(outputs.alive.read_text())
    sites = {k: f"layers.{layer}.mlp.{k}_proj" for k in KINDS}
    alive = {k: np.asarray(alive_json["components"][s], np.int32) for k, s in sites.items()}

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
        uv = {}
        for k, s in sites.items():
            site = components.site(s)
            uv[f"U_{k}"] = np.asarray(site.U, np.float32)[alive[k]]
            uv[f"V_{k}"] = np.asarray(site.V, np.float32)[:, alive[k]]
        prepared, _ = prepare_components(placed, components)
        del components
        cols = {k: jax.sharding.reshard(jnp.asarray(alive[k]), P()) for k in KINDS}
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        taps = {
            "gate_pre": site_output_tap_key(sites["gate"]),
            "up_pre": site_output_tap_key(sites["up"]),
            "hidden": mlp_hidden_tap_key(layer),
        }

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
            masked = placed.masked_forward(
                prepared,
                tokens,
                masking=masking,
                capture_keys=frozenset(taps.values()),
                remat=False,
            )
            inner = component_activation_model(placed).masked_component_activations(
                prepared, tokens, masking, placement=placed.placement
            )

            def pick(values: dict, dtype) -> Array:  # noqa: ANN001
                return jnp.concatenate(
                    [
                        jnp.take(jax.sharding.reshard(values[sites[k]], P()), cols[k], axis=-1)
                        for k in KINDS
                    ],
                    axis=-1,
                ).astype(dtype)

            res = {
                "inner": pick(inner, jnp.float32),
                "masks": pick({s: v > threshold for s, v in lower.items()}, jnp.bool_),
                **{name: masked.captures[key].astype(jnp.float16) for name, key in taps.items()},
            }
            return {k: jax.sharding.reshard(v, P()) for k, v in res.items()}

        K = sum(v.size for v in alive.values())
        inner_mm = open_memmap(out / "inner.npy", mode="w+", dtype=np.float32, shape=(n, T, K))
        masks_mm = open_memmap(out / "masks.npy", mode="w+", dtype=np.bool_, shape=(n, T, K))
        check = {name: [] for name in taps}
        t0 = time.time()
        starts = list(range(0, n, args.batch_size))
        for i, start in enumerate(starts):
            stop = min(start + args.batch_size, n)
            idx = np.arange(start, start + args.batch_size, dtype=np.int32) % pool.n_prompts
            got = collect(placed, prepared, ci_fn, constraints, cols, tokens_all, jnp.asarray(idx))
            real = stop - start
            inner_mm[start:stop] = np.asarray(got["inner"])[:real]
            masks_mm[start:stop] = np.asarray(got["masks"])[:real]
            if start < N_CHECK:
                for name in taps:
                    check[name].append(np.asarray(got[name])[: min(real, N_CHECK - start)])
            if i % 50 == 0 or stop == n:
                el = time.time() - t0
                print(f"{stop}/{n} {el:.0f}s eta {el / (i + 1) * (len(starts) - i - 1):.0f}s",
                      flush=True)  # fmt: skip
        inner_mm.flush()
        masks_mm.flush()
    np.savez(out / "check.npz", **{k: np.concatenate(v) for k, v in check.items()})
    np.savez(
        out / "comps.npz",
        kind=np.asarray([k for k in KINDS for _ in alive[k]]),
        index=np.concatenate([alive[k] for k in KINDS]),
        **uv,
    )
    meta = {"run_dir": str(RUN_DIR), "step": STEP, "filter": FILTER, "layer": layer,
            "threshold": threshold, "n_prompts": n, "T": T,
            "n_alive": {k: int(v.size) for k, v in alive.items()},
            "seconds": time.time() - t0}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
