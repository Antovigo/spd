"""MLP component activations of the ALIVE-ONLY model of the -08 CI filter, every block (plan §10).

The alive-only model: every component of the filter's alive set (`alive/alive.json`) is on at 100 %
on every prompt and position; every other component and the weight delta are off. For every block
l this writes, for the block's alive gate / up / down components (columns in that order, as
`../L<l>/comps.npz`), positions 1..4 (position 0, BOS, is the same on every prompt):

    OUT/L<l>/inner.npy  (N, 4, K_l) float32   t_c = x . V_c (raw gauge), alive-only model
    OUT/L<l>/comps.npz  kind (K_l,), index (K_l,); U_<kind> (K_kind, d_out), V_<kind> (d_in, K_kind)
    OUT/kl_last.npy     (N,) float32           KL(original || alive-only) at the last position
    OUT/meta.json

    python extract_alive_only.py [--limit N] [--out DIR]
"""

import argparse
import json
import os
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from numpy.lib.format import open_memmap

from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig
from param_decomp.ci_filter.objective import kl_rows
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    all_logits,
    gather_batch,
    prepare_components,
    remove_components,
)
from param_decomp.core.components import require_full_emission
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.arithmetic_eval import component_activation_model
from param_decomp.experiments.lm.load_run import restore_jax_run

RUN_DIR = Path(os.environ.get("RUN_DIR", "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/nw/home/a.vigouroux/out"))
STEP = 40000
FILTER = "addsub-08-last-pos"
KINDS = ("gate", "up", "down")
POSITIONS = (1, 2, 3, 4)
N_LAYER = 32
OUT = RUN_DIR / "analysis" / "reader_writer" / "variance" / "alive_only"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--profile", action="store_true", help="time the forward pieces, write nothing")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "meta.json").exists(), f"{out} is complete already"

    outputs = CIFilterOutputs.for_run(RUN_DIR, STEP, FILTER)
    config = CIFilterConfig.from_file(outputs.config)
    alive_json = json.loads(outputs.alive.read_text())
    alive = {s: np.asarray(ids, np.int32) for s, ids in alive_json["components"].items()}
    sites = {(b, k): f"layers.{b}.mlp.{k}_proj" for b in range(N_LAYER) for k in KINDS}

    restored = restore_jax_run(RUN_DIR, STEP, data_root=DATA_ROOT)
    placed, mesh = restored.placed, restored.mesh
    pool = build_pool(config.pool, load_tokenizer(restored.deliverable.target.model_name))
    n = pool.n_prompts if args.limit is None else args.limit
    assert pool.seq_len == 5, pool.seq_len

    with jax.set_mesh(mesh):
        _, constraints = trained_ci(RUN_DIR, STEP, FILTER, restored.ci_fn)
        components = (
            restored.components
            if constraints.kept is None
            else remove_components(restored.components, constraints.kept)
        )
        assert set(alive) == set(placed.model.site_names), set(alive) ^ set(placed.model.site_names)
        n_comp = {s: int(components.site(s).V.shape[1]) for s in placed.model.site_names}
        for b in range(N_LAYER):
            (out / f"L{b}").mkdir(exist_ok=True)
            uv = {}
            for k in KINDS:
                site = components.site(sites[(b, k)])
                uv[f"U_{k}"] = np.asarray(site.U, np.float32)[alive[sites[(b, k)]]]
                uv[f"V_{k}"] = np.asarray(site.V, np.float32)[:, alive[sites[(b, k)]]]
            np.savez(out / f"L{b}" / "comps.npz",
                     kind=np.asarray([k for k in KINDS for _ in alive[sites[(b, k)]]]),
                     index=np.concatenate([alive[sites[(b, k)]] for k in KINDS]), **uv)  # fmt: skip
        prepared, _ = prepare_components(placed, components)
        del restored, components
        # (C,) 1.0 on the site's alive components, 0.0 elsewhere; broadcast over (B, T) in-jit
        on = {}
        for s in placed.model.site_names:
            vec = np.zeros(n_comp[s], np.float32)
            vec[alive[s]] = 1.0
            on[s] = jax.sharding.reshard(jnp.asarray(vec), P())
        cols = {bk: jax.sharding.reshard(jnp.asarray(alive[s]), P()) for bk, s in sites.items()}
        pos = jnp.asarray(POSITIONS)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())

        @eqx.filter_jit
        def collect(placed, prepared, on, cols, tokens_all, idx):  # noqa: ANN001, ANN202
            batch = gather_batch(tokens_all, idx)
            clean, clean_acts = placed.component_activation_forward(
                prepared, batch, sites=placed.model.site_names, capture_keys=frozenset()
            )
            masking = MaterializedMasking(
                component_masks={
                    s: (jnp.zeros_like(require_full_emission(v)) + on[s]).astype(COMPUTE_DT)
                    for s, v in clean_acts.items()
                }
            )
            masked = placed.masked_forward(
                prepared,
                batch,
                masking=placed.model.prepare_masking(masking),
                routes=None,
                remat=False,
            )
            inner = component_activation_model(placed).masked_component_activations(
                prepared, batch, masking, placement=placed.placement
            )
            per_block = []
            for b in range(N_LAYER):
                got = jnp.concatenate(
                    [
                        jnp.take(
                            jax.sharding.reshard(inner[sites[(b, k)]], P()), cols[(b, k)], axis=-1
                        )
                        for k in KINDS
                    ],
                    axis=-1,
                )[:, pos]
                per_block.append(jax.sharding.reshard(got.astype(jnp.float32), P()))
            kl = kl_rows(all_logits(masked.output)[:, -1], all_logits(clean.output)[:, -1])
            return per_block, jax.sharding.reshard(kl, P())

        if args.profile:
            profile(placed, prepared, on, cols, tokens_all, collect, args.batch_size)
            return
        widths = [sum(alive[sites[(b, k)]].size for k in KINDS) for b in range(N_LAYER)]
        mm = [
            open_memmap(
                out / f"L{b}" / "inner.npy",
                mode="w+",
                dtype=np.float32,
                shape=(n, len(POSITIONS), widths[b]),
            )  # fmt: skip
            for b in range(N_LAYER)
        ]
        kl_last = np.zeros(n, np.float32)
        absmax = 0.0
        t0 = time.time()
        starts = list(range(0, n, args.batch_size))
        for i, start in enumerate(starts):
            stop = min(start + args.batch_size, n)
            idx = np.arange(start, start + args.batch_size, dtype=np.int32) % pool.n_prompts
            per_block, kl = collect(placed, prepared, on, cols, tokens_all, jnp.asarray(idx))
            real = stop - start
            for b, got in enumerate(per_block):
                got = np.asarray(got)[:real]
                absmax = max(absmax, float(np.abs(got).max()))
                mm[b][start:stop] = got
            kl_last[start:stop] = np.asarray(kl)[:real]
            if i % 50 == 0 or stop == n:
                el = time.time() - t0
                print(f"{stop}/{n} {el:.0f}s eta {el / (i + 1) * (len(starts) - i - 1):.0f}s "
                      f"absmax {absmax:.1f} kl_last mean so far {kl_last[:stop].mean():.4f}",
                      flush=True)  # fmt: skip
        for m in mm:
            m.flush()
    np.save(out / "kl_last.npy", kl_last)
    meta = {"run_dir": str(RUN_DIR), "step": STEP, "filter": FILTER, "model": "alive-only",
            "n_prompts": n, "positions": list(POSITIONS), "inner_dtype": "float32",
            "absmax": absmax, "kl_last_mean": float(kl_last.mean()),
            "n_alive": {b: {k: int(alive[sites[(b, k)]].size) for k in KINDS}
                        for b in range(N_LAYER)},
            "seconds": time.time() - t0}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "n_alive"}, indent=2), flush=True)


def profile(placed, prepared, on, cols, tokens_all, collect, bs: int) -> None:  # noqa: ANN001
    """Seconds per batch of each forward piece (after a warm-up call), and of the full collect."""

    def masks_of(clean_acts):  # noqa: ANN001, ANN202
        return MaterializedMasking(
            component_masks={
                s: (jnp.zeros_like(require_full_emission(v)) + on[s]).astype(COMPUTE_DT)
                for s, v in clean_acts.items()
            }
        )

    @eqx.filter_jit
    def clean_only(placed, prepared, tokens_all, idx):  # noqa: ANN001, ANN202
        batch = gather_batch(tokens_all, idx)
        clean, acts = placed.component_activation_forward(
            prepared, batch, sites=placed.model.site_names, capture_keys=frozenset()
        )
        return all_logits(clean.output)[:, -1], acts

    @eqx.filter_jit
    def masked_only(placed, prepared, tokens_all, idx, acts):  # noqa: ANN001, ANN202
        batch = gather_batch(tokens_all, idx)
        out = placed.masked_forward(prepared, batch, masking=placed.model.prepare_masking(
            masks_of(acts)), routes=None, remat=False)  # fmt: skip
        return all_logits(out.output)[:, -1]

    @eqx.filter_jit
    def inner_only(placed, prepared, tokens_all, idx, acts):  # noqa: ANN001, ANN202
        batch = gather_batch(tokens_all, idx)
        return component_activation_model(placed).masked_component_activations(
            prepared, batch, masks_of(acts), placement=placed.placement
        )

    idx = jnp.arange(bs, dtype=jnp.int32)
    _, acts = clean_only(placed, prepared, tokens_all, idx)
    pieces = {
        "clean_forward": lambda: clean_only(placed, prepared, tokens_all, idx),
        "masked_forward": lambda: masked_only(placed, prepared, tokens_all, idx, acts),
        "masked_inner": lambda: inner_only(placed, prepared, tokens_all, idx, acts),
        "collect": lambda: collect(placed, prepared, on, cols, tokens_all, idx),
        "collect+host": lambda: [
            np.asarray(a) for a in collect(placed, prepared, on, cols, tokens_all, idx)[0]
        ],  # fmt: skip
    }
    for name, fn in pieces.items():
        jax.block_until_ready(fn())
        t0 = time.time()
        for _ in range(3):
            jax.block_until_ready(fn())
        print(f"profile {name}: {(time.time() - t0) / 3:.2f} s per {bs}-prompt batch", flush=True)


if __name__ == "__main__":
    main()
