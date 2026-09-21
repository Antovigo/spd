"""Activation dataset of the original model and of one CI filter's decomposed model over the pool.

    python -m param_decomp.ci_filter.scripts.collect_dataset --config <yaml> --data_root <root>
        --source <cf-id> [--batch_size 100] [--limit N]

The DECOMPOSED model is the filter's rounded-CI forward: on each prompt and position, exactly
the components whose output CI (computed on the ORIGINAL model's activations, under the
filter's constraints) exceeds `alive_threshold` are on; every other component and the weight
delta are off. Component data covers the filter's alive set (`alive/alive.json`) only; that
no other component crosses the threshold on the original model is checked on every batch.

Writes `<filter dir>/dataset/` (layout in `DATASET_README`, also written as `README.md`)."""

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.checkpoint import trained_ci
from param_decomp.ci_filter.config import CIFilterConfig, resolve_run_dir
from param_decomp.ci_filter.objective import kl_rows
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import ArithmeticPool, build_pool, load_tokenizer
from param_decomp.ci_filter.step import (
    CIConstraints,
    Prepared,
    gather_rows,
    output_ci,
    prepare_components,
    remove_components,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import MaterializedMasking, PlacedModel, select_captures
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.arithmetic_eval import component_activation_model
from param_decomp.experiments.lm.load_run import restore_jax_run
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.training import enable_persistent_compilation_cache
from param_decomp.targets.transformer_taps import (
    mlp_hidden_tap_key,
    post_attention_tap_key,
    resid_tap_key,
    site_output_tap_key,
)

MODELS = ("original", "decomposed")
TOP_K = 10
CHECK_LAYERS = (0, 18, 31)
"""Blocks whose `mlp_hidden` is captured to check `silu(gate) * up` against it."""

DATASET_README = """\
# Activation dataset: original model vs rounded-CI decomposed model

Source: run `{run_dir}` step {step}, CI filter `{source}`, threshold {threshold}.
Commit: `{commit}`.

The DECOMPOSED model runs, on each prompt and position, exactly the components whose output CI
on the ORIGINAL model's activations exceeds {threshold}; every other component and the weight
delta are off. Every array is a numpy `.npy`, open with `np.load(path, mmap_mode="r")`.

N = {n_prompts} prompts (`index.npz`: `tokens`, `a`, `b`, `op` 0=add 1=sub, row-major (a, b)
per operation block), T = {seq_len} positions ({position_labels}), A = {n_alive} alive
components, L = {n_layer} blocks.

## index.npz
- `tokens` (N, T), `a`, `b`, `op` (N,), `position_labels` (T,)
- `resid_labels` (2L+1,): `embed`, then `L<l>.attn` (after block l's attention add) and
  `L<l>.mlp` (after its MLP add) per block — raw residual stream, before any norm
- `comp_site`, `comp_layer`, `comp_kind`, `comp_index` (A,): the component of each CI/inner
  column (sites in sorted name order, component index ascending)
- `comp_v_norm`, `comp_u_norm` (A,): ||V_c|| and ||U_c||; when on, the component writes
  `inner * U[c]` (row c of its site's U) into the site's output

## <model>/ (`original/`, `decomposed/`)
- `resid.npy` (2L+1, N, T, d_model) float16 — residual stream, layer-major
- `mlp_gate.npy`, `mlp_up.npy` (L, N, T, d_mlp) float16 — gate_proj / up_proj outputs
  (pre-activation); the neuron's value into down_proj is `silu(gate) * up`
- `ci.npy` (N, T, A) float16 — output-head lower-leaky CI (the value thresholded for masks);
  in `decomposed/` it is the CI fn evaluated on the decomposed model's own activations
- `inner.npy` (N, T, A) float32 — the component's inner activation `x @ V_c` (before any mask)
- `last_top_ids.npy` int32, `last_top_logprobs.npy` float32 (N, {top_k}) — top next-token
  predictions at the last position

## Top level
- `kl_last.npy` (N,) float32 — KL(original || decomposed) of the last-position distribution
- `meta.json` — counts, checks (fp16 range, gate/up vs mlp_hidden, threshold crossings outside
  the alive set), timings
"""


def residual_keys(n_layer: int) -> tuple[str, ...]:
    keys = [resid_tap_key(0)]
    for block in range(n_layer):
        keys += [post_attention_tap_key(block), resid_tap_key(block + 1)]
    return tuple(keys)


def residual_labels(n_layer: int) -> tuple[str, ...]:
    labels = ["embed"]
    for block in range(n_layer):
        labels += [f"L{block}.attn", f"L{block}.mlp"]
    return tuple(labels)


def mlp_keys(n_layer: int, kind: str) -> tuple[str, ...]:
    return tuple(site_output_tap_key(f"layers.{block}.mlp.{kind}") for block in range(n_layer))


def make_collect(n_layer: int, threshold: float):  # noqa: ANN201 (jitted closure)
    resid = residual_keys(n_layer)
    gate, up = mlp_keys(n_layer, "gate_proj"), mlp_keys(n_layer, "up_proj")
    hidden = tuple(mlp_hidden_tap_key(block) for block in CHECK_LAYERS)

    def extract(
        captures: dict[str, Array],
        logits: Array,
        lower: dict[str, Array],
        inner: dict[str, Array],
        alive_idx: dict[str, Array],
    ) -> dict[str, Array]:
        sites = sorted(alive_idx)
        stacked = {
            "resid": jnp.stack([captures[k] for k in resid]),
            "mlp_gate": jnp.stack([captures[k] for k in gate]),
            "mlp_up": jnp.stack([captures[k] for k in up]),
        }
        out: dict[str, Array] = {k: v.astype(jnp.float16) for k, v in stacked.items()}
        out["absmax"] = jnp.stack(
            [jnp.max(jnp.abs(v.astype(jnp.float32))) for v in stacked.values()]
        )
        errs = []
        for block, key in zip(CHECK_LAYERS, hidden, strict=True):
            g = captures[gate[block]].astype(jnp.float32)
            ref = captures[key].astype(jnp.float32)
            got = jax.nn.silu(g) * captures[up[block]].astype(jnp.float32)
            errs.append(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-6))
        out["hidden_rel_err"] = jnp.max(jnp.stack(errs))
        ci = jnp.concatenate([jnp.take(lower[s], alive_idx[s], axis=-1) for s in sites], axis=-1)
        out["ci"] = ci.astype(jnp.float16)
        out["inner"] = jnp.concatenate(
            [jnp.take(inner[s], alive_idx[s], axis=-1) for s in sites], axis=-1
        ).astype(jnp.float32)
        above = sum(jnp.sum(v > threshold, dtype=jnp.int32) for v in lower.values())
        above_alive = jnp.sum(ci > threshold, dtype=jnp.int32)
        out["above_outside_alive"] = above - above_alive
        logprobs = jax.nn.log_softmax(logits[:, -1].astype(jnp.float32), axis=-1)
        out["last_top_logprobs"], out["last_top_ids"] = jax.lax.top_k(logprobs, TOP_K)
        return {k: jax.sharding.reshard(v, P()) for k, v in out.items()}

    @eqx.filter_jit
    def collect(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        constraints: CIConstraints,
        alive_idx: dict[str, Array],
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        tokens = gather_rows(tokens_all, idx)
        keys = frozenset(ci_fn.capture_keys) | frozenset((*resid, *gate, *up, *hidden))

        clean, clean_inner = placed.component_activation_forward(
            prepared, tokens, capture_keys=keys
        )
        clean_lower = output_ci(
            placed, ci_fn, constraints, select_captures(clean.captures, ci_fn.capture_keys), False
        ).lower
        original = extract(clean.captures, clean.output, clean_lower, clean_inner, alive_idx)

        masking = MaterializedMasking(
            component_masks={k: (v > threshold).astype(COMPUTE_DT) for k, v in clean_lower.items()}
        )
        masked = placed.masked_forward(
            prepared, tokens, masking=masking, capture_keys=keys, remat=False
        )
        masked_inner = component_activation_model(placed).masked_component_activations(
            prepared, tokens, masking, placement=placed.placement
        )
        masked_lower = output_ci(
            placed, ci_fn, constraints, select_captures(masked.captures, ci_fn.capture_keys), False
        ).lower
        decomposed = extract(masked.captures, masked.output, masked_lower, masked_inner, alive_idx)
        kl = jax.sharding.reshard(kl_rows(masked.output[:, -1], clean.output[:, -1]), P())
        return original, decomposed, kl

    return collect


@dataclass
class Checks:
    """One model's running checks over the batches."""

    absmax: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Max |value| of resid, mlp_gate, mlp_up (must stay under the fp16 max, 65504)."""
    hidden_rel_err: float = 0.0
    """Max |silu(gate) * up - mlp_hidden| / max |mlp_hidden| over `CHECK_LAYERS`."""
    above_outside_alive: int = 0
    """(prompt, position, component) CIs over the threshold outside the alive set."""


def pool_columns(pool: ArithmeticPool) -> dict[str, np.ndarray]:
    a, b, op = [], [], []
    for code, block in enumerate(pool.blocks):
        grid = block.grid
        a.append(np.repeat(np.asarray(grid.a_values, np.int32), grid.n_b))
        b.append(np.tile(np.asarray(grid.b_values, np.int32), grid.n_a))
        op.append(np.full(grid.n_a * grid.n_b, code, np.int8))
        assert block.operation == ("add", "sub")[code], block.operation
    return {"a": np.concatenate(a), "b": np.concatenate(b), "op": np.concatenate(op)}


def collect_dataset(
    config: CIFilterConfig, data_root: Path, source: str, batch_size: int, limit: int | None
) -> Path:
    if config.compilation_cache_dir is not None:
        enable_persistent_compilation_cache(config.compilation_cache_dir)
    run_dir = resolve_run_dir(config.run, data_root)
    restored = restore_jax_run(run_dir, config.step, data_root=data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig)
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    outputs = CIFilterOutputs.for_run(run_dir, step, source)
    out_dir = outputs.dataset
    out_dir.mkdir(parents=True, exist_ok=False)
    pool = build_pool(config.pool, load_tokenizer(target.model_name))
    threshold = config.alive_threshold
    n_prompts = pool.n_prompts if limit is None else limit
    n_layer = len({site.split(".")[1] for site in placed.site_names})

    alive_json = json.loads(outputs.alive.read_text())
    assert alive_json["threshold"] == threshold, (alive_json["threshold"], threshold)
    alive = {site: np.asarray(ids, np.int32) for site, ids in alive_json["components"].items()}
    sites = sorted(alive)
    n_alive = sum(v.size for v in alive.values())
    assert n_alive == alive_json["n_alive"], (n_alive, alive_json["n_alive"])
    logger.info(f"{source}: {n_alive} alive components, {n_prompts} prompts -> {out_dir}")

    with jax.set_mesh(mesh):
        ci_fn, constraints = trained_ci(run_dir, step, source, restored.ci_fn)
        components = (
            restored.components
            if constraints.kept is None
            else remove_components(restored.components, constraints.kept)
        )
        u_norms = {
            s: np.linalg.norm(np.asarray(components.site(s).U, np.float32), axis=1) for s in sites
        }
        prepared, v_norms = prepare_components(placed, components)
        del restored, components
        assert set(alive) == set(placed.site_names), set(placed.site_names) ^ set(alive)
        alive_idx = {s: jax.sharding.reshard(jnp.asarray(alive[s]), P()) for s in sites}
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())

        T = pool.seq_len
        d_model = int(site_d_in(placed, "self_attn.q_proj"))
        d_mlp = int(site_d_in(placed, "mlp.down_proj"))
        n_resid = 2 * n_layer + 1
        files = {
            model: {
                "resid": ((n_resid, n_prompts, T, d_model), np.float16),
                "mlp_gate": ((n_layer, n_prompts, T, d_mlp), np.float16),
                "mlp_up": ((n_layer, n_prompts, T, d_mlp), np.float16),
                "ci": ((n_prompts, T, n_alive), np.float16),
                "inner": ((n_prompts, T, n_alive), np.float32),
                "last_top_ids": ((n_prompts, TOP_K), np.int32),
                "last_top_logprobs": ((n_prompts, TOP_K), np.float32),
            }
            for model in MODELS
        }
        mms: dict[str, dict[str, np.memmap]] = {}
        for model, spec in files.items():
            (out_dir / model).mkdir()
            mms[model] = {
                name: np.lib.format.open_memmap(
                    out_dir / model / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
                )
                for name, (shape, dtype) in spec.items()
            }
        kl_last = np.zeros(n_prompts, np.float32)
        checks = {m: Checks() for m in MODELS}

        collect = make_collect(n_layer, threshold)
        t0 = time.time()
        starts = list(range(0, n_prompts, batch_size))
        for i, start in enumerate(starts):
            stop = min(start + batch_size, n_prompts)
            idx = np.arange(start, start + batch_size, dtype=np.int32) % n_prompts
            got = collect(
                placed, prepared, ci_fn, constraints, alive_idx, tokens_all, jnp.asarray(idx)
            )
            real = stop - start
            for model, result in zip(MODELS, got[:2], strict=True):
                host = {k: np.asarray(v) for k, v in result.items()}
                mm = mms[model]
                for name in ("resid", "mlp_gate", "mlp_up"):
                    mm[name][:, start:stop] = host[name][:, :real]
                for name in ("ci", "inner", "last_top_ids", "last_top_logprobs"):
                    mm[name][start:stop] = host[name][:real]
                c = checks[model]
                c.absmax = [max(x, float(y)) for x, y in zip(c.absmax, host["absmax"], strict=True)]
                c.hidden_rel_err = max(c.hidden_rel_err, float(host["hidden_rel_err"]))
                c.above_outside_alive += int(host["above_outside_alive"])
                assert max(c.absmax) < 65504, f"{model}: fp16 overflow {c}"
            kl_last[start:stop] = np.asarray(got[2])[:real]
            if i % 10 == 0 or stop == n_prompts:
                elapsed = time.time() - t0
                logger.info(
                    f"{stop}/{n_prompts} ({elapsed:.0f}s, eta {elapsed / (i + 1) * (len(starts) - i - 1):.0f}s)"
                    f" KL so far {kl_last[:stop].mean():.4f} checks {checks}"
                )
        for model in MODELS:
            for mm in mms[model].values():
                mm.flush()
        del mms

    assert checks["original"].above_outside_alive == 0, checks
    comp_site = np.asarray([s for s in sites for _ in alive[s]])
    index: dict[str, np.ndarray] = {
        "tokens": pool.tokens[:n_prompts],
        **{k: v[:n_prompts] for k, v in pool_columns(pool).items()},
        "position_labels": np.asarray(pool.position_labels),
        "resid_labels": np.asarray(residual_labels(n_layer)),
        "comp_site": comp_site,
        "comp_layer": np.asarray([int(s.split(".")[1]) for s in comp_site], np.int32),
        "comp_kind": np.asarray([s.split(".", 2)[2] for s in comp_site]),
        "comp_index": np.concatenate([alive[s] for s in sites]),
        "comp_v_norm": np.concatenate([np.asarray(v_norms[s])[alive[s]] for s in sites]),
        "comp_u_norm": np.concatenate([u_norms[s][alive[s]] for s in sites]),
    }
    np.savez(out_dir / "index.npz", **index)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)
    np.save(out_dir / "kl_last.npy", kl_last)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    meta = {
        "run_dir": str(run_dir),
        "step": step,
        "source": source,
        "threshold": threshold,
        "commit": commit,
        "n_prompts": n_prompts,
        "seq_len": pool.seq_len,
        "n_layer": n_layer,
        "n_alive": n_alive,
        "kl_last_mean": float(kl_last.mean()),
        "checks": {m: asdict(c) for m, c in checks.items()},
        "seconds": time.time() - t0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "README.md").write_text(
        DATASET_README.format(
            run_dir=run_dir,
            step=step,
            source=source,
            threshold=threshold,
            commit=commit,
            n_prompts=n_prompts,
            seq_len=pool.seq_len,
            position_labels=", ".join(pool.position_labels),
            n_alive=n_alive,
            n_layer=n_layer,
            top_k=TOP_K,
        )
    )
    logger.info(f"{json.dumps(meta, indent=2)}\n-> {out_dir}")
    return out_dir


def site_d_in(placed: PlacedModel, kind: str) -> int:
    """Block 0's `kind` input width: d_model for `self_attn.q_proj`, d_mlp for `mlp.down_proj`."""
    (spec,) = [s for s in placed.sites if s.name == f"layers.0.{kind}"]
    return spec.d_in


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="CIFilterConfig YAML (run, pool)")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--source", required=True, help="the CI filter id")
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke)")
    args = ap.parse_args()
    collect_dataset(
        CIFilterConfig.from_file(args.config),
        args.data_root,
        args.source,
        args.batch_size,
        args.limit,
    )


if __name__ == "__main__":
    main()
