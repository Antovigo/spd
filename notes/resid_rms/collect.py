"""Per-token RMS of the residual stream of three models on the whole -08 pool (p-ba8b0c08, step
40000, CI filter `addsub-08-last-pos`, 320,000 prompts "a+b=" / "a-b=", a, b in 1..400).

Models (index m of the output):
    0 original    Llama-3.1-8B (dense weights)
    1 ci_masked   the filter dataset's decomposed model: on each prompt and position exactly the
                  components whose filter CI on the ORIGINAL model's activations exceeds 0.01 are
                  on; every other component and the weight delta are off
    2 alive_only  every one of the filter's 15,768 alive components on at every prompt and
                  position; every other component and the weight delta are off

Stream points p = 0..64 (`index.npz` `resid_labels`): 0 = the embedding, 2l + 1 = after block
l's attention add, 2l + 2 = after its MLP add (l = 0..31) -- the raw stream, before any norm;
point p is the input of the p-th RMSNorm (point 64 feeds the final norm). Positions t = 0..4
(<BOS>, a, op, b, =). The value is r = sqrt(mean_i x_i^2) over the d = 4096 stream dimensions i,
no epsilon, from the bf16 stream in float32.

    OUT/rms.npy      (3, 65, N, 5) float32   r[m, p, n, t]; prompt row n = op * 160000 + (a - 1) * 400 + (b - 1)
    OUT/kl_last.npy  (2, N) float32          KL(original || ci_masked), KL(original || alive_only), last position
    OUT/meta.json

The KLs must reproduce `<filter>/dataset/kl_last.npy` and `analysis/reader_writer/variance/
alive_only/kl_last.npy` (checked at the end when present).

    python collect.py [--limit N] [--batch_size B] [--out DIR]
"""

import argparse
import json
import os
import subprocess
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
    CIConstraints,
    FilterCIFn,
    Prepared,
    all_logits,
    gather_batch,
    output_ci,
    prepare_components,
)
from param_decomp.core.components import require_full_emission
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.load_run import PlacedTransformer, restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.targets.transformer_taps import post_attention_tap_key, resid_tap_key

RUN_DIR = Path(os.environ.get("RUN_DIR", "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/mnt/nw/home/a.vigouroux/out"))
STEP = 40000
FILTER = "addsub-08-last-pos"
MODELS = ("original", "ci_masked", "alive_only")
N_LAYER = 32
OUT = RUN_DIR / "analysis" / "resid_rms"


def residual_keys(n_layer: int) -> tuple[str, ...]:
    keys = [resid_tap_key(0)]
    for block in range(n_layer):
        keys += [post_attention_tap_key(block), resid_tap_key(block + 1)]
    return tuple(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "meta.json").exists(), f"{out} is complete already"

    outputs = CIFilterOutputs.for_run(RUN_DIR, STEP, FILTER)
    config = CIFilterConfig.from_file(outputs.config)
    threshold = config.alive_threshold
    alive_json = json.loads(outputs.alive.read_text())
    assert alive_json["threshold"] == threshold, (alive_json["threshold"], threshold)
    alive = {s: np.asarray(ids, np.int32) for s, ids in alive_json["components"].items()}
    assert sum(v.size for v in alive.values()) == alive_json["n_alive"]

    restored = restore_jax_run(RUN_DIR, STEP, data_root=DATA_ROOT)
    placed, mesh, components = restored.placed, restored.mesh, restored.components
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    pool = build_pool(config.pool, load_tokenizer(target.model_name))
    n = pool.n_prompts if args.limit is None else args.limit
    assert pool.seq_len == 5, pool.seq_len
    resid = residual_keys(N_LAYER)

    with jax.set_mesh(mesh):
        ci_fn, constraints = trained_ci(RUN_DIR, STEP, FILTER, restored.ci_fn)
        del restored
        # Removed components (`alive/kept.npz`) have CI forced to 0 by `constraints` and are never
        # alive, so every mask below is 0 on them: the pruned copy of the components is not needed.
        if constraints.kept is not None:
            for s, ids in alive.items():
                assert np.asarray(constraints.kept[s])[ids].all(), f"{s}: alive but removed"
        assert set(alive) == set(placed.model.site_names), set(alive) ^ set(placed.model.site_names)
        n_comp = {s: int(components.site(s).V.shape[1]) for s in placed.model.site_names}
        prepared, _ = prepare_components(placed, components)
        del components
        on = {}
        for s in placed.model.site_names:
            vec = np.zeros(n_comp[s], np.float32)
            vec[alive[s]] = 1.0
            on[s] = jax.sharding.reshard(jnp.asarray(vec), P())
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())

        def rms(captures: dict[str, jax.Array]) -> jax.Array:
            """(65, B, T) float32."""
            x = jnp.stack([captures[k] for k in resid]).astype(jnp.float32)
            return jax.sharding.reshard(jnp.sqrt(jnp.mean(x * x, axis=-1)), P())

        @eqx.filter_jit
        def collect(
            placed: PlacedTransformer,
            prepared: Prepared,
            ci_fn: FilterCIFn,
            constraints: CIConstraints,
            on: dict[str, jax.Array],
            tokens_all: jax.Array,
            idx: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            batch = gather_batch(tokens_all, idx)
            keys = ci_fn.capture_keys | frozenset(resid)
            clean, clean_acts = placed.component_activation_forward(
                prepared, batch, sites=placed.model.site_names, capture_keys=keys
            )
            lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
            clean_logits = all_logits(clean.output)[:, -1]
            rs, kls = [rms(clean.captures)], []
            maskings = (
                {k: (v > threshold).astype(COMPUTE_DT) for k, v in lower.items()},
                {s: (jnp.zeros_like(require_full_emission(v)) + on[s]).astype(COMPUTE_DT)
                 for s, v in clean_acts.items()},
            )  # fmt: skip
            for masks in maskings:
                masked = placed.masked_forward(
                    prepared,
                    batch,
                    masking=placed.model.prepare_masking(
                        MaterializedMasking(component_masks=masks)
                    ),
                    routes=None,
                    capture_keys=frozenset(resid),
                    remat=False,
                )
                rs.append(rms(masked.captures))
                kls.append(kl_rows(all_logits(masked.output)[:, -1], clean_logits))
            return jnp.stack(rs), jax.sharding.reshard(jnp.stack(kls), P())

        mm = open_memmap(out / "rms.npy", mode="w+", dtype=np.float32, shape=(3, len(resid), n, 5))
        kl_last = np.zeros((2, n), np.float32)
        t0 = time.time()
        starts = list(range(0, n, args.batch_size))
        for i, start in enumerate(starts):
            stop = min(start + args.batch_size, n)
            idx = np.arange(start, start + args.batch_size, dtype=np.int32) % pool.n_prompts
            r, kl = collect(placed, prepared, ci_fn, constraints, on, tokens_all, jnp.asarray(idx))
            real = stop - start
            mm[:, :, start:stop] = np.asarray(r)[:, :, :real]
            kl_last[:, start:stop] = np.asarray(kl)[:, :real]
            if i % 50 == 0 or stop == n:
                el = time.time() - t0
                print(f"{stop}/{n} {el:.0f}s eta {el / (i + 1) * (len(starts) - i - 1):.0f}s "
                      f"kl_last so far {kl_last[0, :stop].mean():.4f} {kl_last[1, :stop].mean():.4f}",
                      flush=True)  # fmt: skip
        mm.flush()
    np.save(out / "kl_last.npy", kl_last)

    # the two masked models must be the ones of the filter dataset and of the alive-only extraction
    refs = {
        "ci_masked": outputs.dataset / "kl_last.npy",
        "alive_only": RUN_DIR / "analysis/reader_writer/variance/alive_only/kl_last.npy",
    }
    checks = {}
    for j, (name, path) in enumerate(refs.items()):
        if path.exists():
            ref = np.load(path)[:n]
            checks[name] = {"ref": str(path), "max_abs_diff": float(np.abs(kl_last[j] - ref).max()),
                            "mean": float(kl_last[j].mean()), "ref_mean": float(ref.mean())}  # fmt: skip
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()  # fmt: skip
    meta = {"run_dir": str(RUN_DIR), "step": STEP, "filter": FILTER, "threshold": threshold,
            "commit": commit, "models": list(MODELS), "n_prompts": n, "seq_len": 5,
            "n_points": len(resid), "kl_checks": checks,
            "rms_min": float(np.min(mm)), "rms_max": float(np.max(mm)),
            "seconds": time.time() - t0}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
