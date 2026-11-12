"""Alive sets, evaluation and `(a, b)` grids of a CI-capped tPD fine-tune ("integer-only").

    python -m param_decomp.ci_filter.scripts.integer_only --finetune_run_dir <runs>/<ft-id>
        --parent_run_dir <runs>/<parent-id> --parent_step 40000 --data_root <root> --out <dir>

The fine-tune (`pd.ci_ceiling`) checkpoints its trained, UNCAPPED CI fn; its effective CI is
the entrywise minimum of that fn's output-head CI and the parent's (`CIConstraints` ceiling).
Every CI below is that capped output-head lower-leaky CI on the ORIGINAL model's activations.

The pool is every `(a, b)` of both operations rendered with `--template` (`{a}`, `{op}`, `{b}`;
the op as a word), a and b over `--a_range` x `--b_range`; subtraction cells with a < b lie
outside the pool: they are never scored and their grid cells are NaN (grey). The valid
prompts must be exactly the lines of `--prompts_file` when one is given (the training pool).

Outputs under `--out`:

- `alive.json`: per position t, the components ALIVE at t — whose CI exceeds `--threshold` at
  t on some pool prompt — plus `max_ci.npz` (`<site>`: `(T, C)` max CI over the pool).
- `eval.json`: KL of three weight-delta-off models from Llama over the pool, at the last
  position (among the integer tokens `0..999` renormalized, and over the full vocabulary) and
  averaged over every position (full vocabulary): `integer_only` (at each position t exactly
  the components alive at t are on, for every prompt), `ci_masked` (masks = the CI) and
  `rounded_ci` (masks = CI > threshold); per operation and overall, with the integer top-1
  agreement and the alive counts.
- `ab_grids/`: the slice applet (`index.html`, `grids_app.html` with the integer-only view on)
  over operation x position (`--positions`), one file per slice saving the components alive
  at that position: their CI and their normalized inner activation `(x . V_c) / ||V_c||` on
  the original model; `alive/<slice>` holds the same components' normalized inner
  activations in the INTEGER-ONLY model. Grids wider than 100 are split (thumbnails + one
  full-resolution file per module).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.grid import APPLET, alive_page, slice_filename, slice_key, write_slice
from param_decomp.ci_filter.objective import kl_rows, restrict
from param_decomp.ci_filter.pool import load_tokenizer
from param_decomp.ci_filter.step import (
    CIConstraints,
    FilterCIFn,
    Prepared,
    all_logits,
    gather_batch,
    index_batches,
    masked_output,
    output_ci,
    prepare_components,
)
from param_decomp.core.components import SiteCI, require_full_emission
from param_decomp.core.log import logger, setup_console_logger
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.experiments.lm.ab_grid_dataset import (
    ABGridSnapshot,
    _b64,
    ab_grid_payload,
    ab_grid_split_payload,
    pool_grid,
    thumbnail_factor,
)
from param_decomp.experiments.lm.arithmetic_eval import ArithmeticGrid, component_activation_model
from param_decomp.experiments.lm.load_run import (
    PlacedTransformer,
    restore_jax_run,
    restore_other_decomposition,
)
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.experiments.lm.target_output import integer_token_ids

OPERATIONS = {"add": ("+", "plus"), "sub": ("-", "minus")}
"""operation -> (applet symbol, the word the template renders)."""


# ───────────────────────────── the pool ─────────────────────────────


class Pool:
    """Both operations' full `(a, b)` grids, operation-major then row-major `(a, b)`."""

    def __init__(
        self, template: str, a_range: tuple[int, int], b_range: tuple[int, int], tokenizer: Any
    ) -> None:
        self.a_values = tuple(range(a_range[0], a_range[1] + 1))
        self.b_values = tuple(range(b_range[0], b_range[1] + 1))
        a = np.repeat(np.asarray(self.a_values), len(self.b_values))
        b = np.tile(np.asarray(self.b_values), len(self.a_values))
        self.cells = a.size
        self.ops = tuple(OPERATIONS)
        self.texts: list[str] = []
        valid = []
        for op in self.ops:
            word = OPERATIONS[op][1]
            self.texts.extend(template.format(a=x, op=word, b=y) for x, y in zip(a, b, strict=True))
            valid.append(np.ones(a.size, bool) if op == "add" else a >= b)
        self.valid = np.concatenate(valid)
        rows = [[int(t) for t in tokenizer.encode(s, add_special_tokens=True)] for s in self.texts]
        lengths = sorted({len(r) for r in rows})
        assert len(lengths) == 1, f"prompts tokenize to several lengths: {lengths}"
        self.tokens = np.asarray(rows, np.int32)
        self.position_labels = [tokenizer.decode([int(t)]) for t in self.tokens[0]]

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[1])

    def block(self, op: str) -> tuple[int, int]:
        start = self.ops.index(op) * self.cells
        return start, start + self.cells

    def grid(self, op: str) -> ArithmeticGrid:
        return ArithmeticGrid(self.a_values, self.b_values, OPERATIONS[op][0])


def check_against_prompts_file(pool: Pool, path: Path) -> None:
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    valid = [t for t, v in zip(pool.texts, pool.valid, strict=True) if v]
    assert sorted(lines) == sorted(valid), (
        f"the valid grid prompts ({len(valid)}) are not the training pool {path} ({len(lines)})"
    )


# ───────────────────────────── jitted passes ─────────────────────────────


@eqx.filter_jit
def _chunk_ci_stats(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    tokens_all: Int[Array, "N T"],
    valid_all: Array,
    idx: Int[Array, " B"],
    n_real: Array,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """`({site: (T, C) max CI}, {site: (T, C) summed CI})` over the chunk's valid real rows."""
    clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
    lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
    keep = ((jnp.arange(idx.shape[0]) < n_real) & valid_all[idx])[:, None, None]
    mx = {
        s: jax.sharding.reshard(jnp.where(keep, v.astype(jnp.float32), 0.0).max(axis=0), P())
        for s, v in lower.items()
    }
    sm = {
        s: jax.sharding.reshard(jnp.where(keep, v.astype(jnp.float32), 0.0).sum(axis=0), P())
        for s, v in lower.items()
    }
    return mx, sm


def _broadcast_masks(alive: dict[str, Array], batch: int) -> dict[str, Array]:
    """`{site: (T, C)}` 0/1 tables -> `{site: (B, T, C)}` compute-precision masks."""
    return {
        s: jnp.broadcast_to(v[None].astype(COMPUTE_DT), (batch, *v.shape)) for s, v in alive.items()
    }


@eqx.filter_jit
def _chunk_eval(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    alive: dict[str, Array],
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    integer_ids: Int[Array, " K"],
    threshold: float,
) -> dict[str, dict[str, Array]]:
    """`{model: {metric: (B,)}}`, replicated, for the three delta-off models."""
    clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
    clean_logits = all_logits(clean.output)
    lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
    maskings = {
        "integer_only": _broadcast_masks(alive, idx.shape[0]),
        "ci_masked": {s: v.astype(COMPUTE_DT) for s, v in lower.items()},
        "rounded_ci": {s: (v > threshold).astype(COMPUTE_DT) for s, v in lower.items()},
    }
    clean_int = restrict(clean_logits[:, -1], integer_ids)
    out: dict[str, dict[str, Array]] = {}
    for name, masks in maskings.items():
        logits = all_logits(
            masked_output(
                placed,
                prepared,
                clean.conditioning,
                MaterializedMasking(component_masks=masks),
                remat=False,
            )
        )
        masked_int = restrict(logits[:, -1], integer_ids)
        per_position = kl_rows(logits, clean_logits)
        rows = {
            "kl_last_integer": kl_rows(masked_int, clean_int),
            "kl_last": per_position[:, -1],
            "kl_all_positions": jnp.mean(per_position, axis=1),
            "integer_top1": (
                jnp.argmax(masked_int, axis=-1) == jnp.argmax(clean_int, axis=-1)
            ).astype(jnp.float32),
        }
        out[name] = {k: jax.sharding.reshard(v, P()) for k, v in rows.items()}
    return out


@eqx.filter_jit
def _chunk_columns(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    alive: dict[str, Array],
    v_norms: dict[str, Array],
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    position: int,
    columns: dict[str, Array],
) -> dict[str, tuple[Array, Array, Array]]:
    """`{site: (ci, inner, inner_integer_only)}`, each `(B, len(columns[site]))` fp32 and
    replicated, at `position`: the CI, the original model's and the integer-only model's
    normalized inner activations of the requested components."""
    batch = gather_batch(tokens_all, idx)
    clean, acts = placed.component_activation_forward(
        prepared, batch, sites=tuple(columns), capture_keys=ci_fn.capture_keys
    )
    lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
    masked_acts = component_activation_model(placed).masked_component_activations(
        prepared,
        batch,
        MaterializedMasking(component_masks=_broadcast_masks(alive, idx.shape[0])),
        placement=placed.placement,
    )

    def take(v: SiteCI, cols: Array) -> Array:
        return (
            require_full_emission(v).astype(jnp.float32).at[:, position, cols].get(out_sharding=P())
        )

    out: dict[str, tuple[Array, Array, Array]] = {}
    for site, cols in columns.items():
        norm = jnp.maximum(v_norms[site][cols], 1e-12)
        out[site] = (
            take(lower[site], cols),
            take(acts[site], cols) / norm,
            take(masked_acts[site], cols) / norm,
        )
    return out


# ───────────────────────────── the passes ─────────────────────────────


def alive_tables(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    tokens_all: Array,
    valid_all: Array,
    pool: Pool,
    chunk: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """`({site: (T, C) max CI over the pool}, {op: {site: (T, C) mean CI over its valid rows}})`."""
    max_ci: dict[str, np.ndarray] = {}
    mean_ci: dict[str, dict[str, np.ndarray]] = {}
    for op in pool.ops:
        start, stop = pool.block(op)
        sums: dict[str, np.ndarray] = {}
        for idx, real in index_batches(stop - start, chunk):
            mx, sm = _chunk_ci_stats(
                placed,
                prepared,
                ci_fn,
                constraints,
                tokens_all,
                valid_all,
                jnp.asarray(idx + start),
                jnp.asarray(real),
            )
            for s in mx:
                m, t = np.asarray(mx[s]), np.asarray(sm[s])
                max_ci[s] = m if s not in max_ci else np.maximum(max_ci[s], m)
                sums[s] = t if s not in sums else sums[s] + t
        n_valid = int(pool.valid[start:stop].sum())
        mean_ci[op] = {s: v / n_valid for s, v in sums.items()}
        logger.info(f"CI stats {op}: {n_valid} prompts")
    return max_ci, mean_ci


def evaluate(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    alive_dev: dict[str, Array],
    tokens_all: Array,
    pool: Pool,
    chunk: int,
    integer_ids: Array,
    threshold: float,
) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    for op in pool.ops:
        start, stop = pool.block(op)
        for idx, real in index_batches(stop - start, chunk):
            got = _chunk_eval(
                placed,
                prepared,
                ci_fn,
                constraints,
                alive_dev,
                tokens_all,
                jnp.asarray(idx + start),
                integer_ids,
                threshold,
            )
            keep = pool.valid[start + idx[:real]]
            for model, metrics in got.items():
                for metric, v in metrics.items():
                    rows.setdefault(op, {}).setdefault(model, {}).setdefault(metric, []).append(
                        np.asarray(v)[:real][keep]
                    )
    out: dict[str, Any] = {}
    for op, models in rows.items():
        out[op] = {
            model: {metric: float(np.concatenate(v).mean()) for metric, v in metrics.items()}
            for model, metrics in models.items()
        }
        out[op]["n_prompts"] = int(len(np.concatenate(models["integer_only"]["kl_last"])))
    out["all"] = {
        model: {
            metric: float(
                np.concatenate([x for op in rows for x in rows[op][model][metric]]).mean()
            )
            for metric in rows[pool.ops[0]][model]
        }
        for model in rows[pool.ops[0]]
    }
    return out


def write_grids(
    grid_dir: Path,
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    alive_np: dict[str, np.ndarray],
    alive_dev: dict[str, Array],
    v_norms: dict[str, Array],
    mean_ci: dict[str, dict[str, np.ndarray]],
    tokens_all: Array,
    pool: Pool,
    positions: tuple[int, ...],
    chunk: int,
    step: int,
) -> dict[str, int]:
    """One slice per (operation, position): the components alive there, CI + both inners."""
    counts: dict[str, int] = {}
    (grid_dir / "alive").mkdir(parents=True, exist_ok=True)
    for op in pool.ops:
        grid = pool.grid(op)
        start, stop = pool.block(op)
        invalid = ~pool.valid[start:stop]
        factor = thumbnail_factor(grid)
        for t in positions:
            t0 = time.time()
            saved = {s: np.nonzero(a[t])[0] for s, a in alive_np.items()}
            live = {s: ids for s, ids in saved.items() if ids.size}
            device_cols = {s: jnp.asarray(ids.astype(np.int32)) for s, ids in live.items()}
            parts: dict[str, list[tuple[np.ndarray, ...]]] = {s: [] for s in live}
            for idx, real in index_batches(stop - start, chunk) if live else ():
                got = _chunk_columns(
                    placed,
                    prepared,
                    ci_fn,
                    constraints,
                    alive_dev,
                    v_norms,
                    tokens_all,
                    jnp.asarray(idx + start),
                    t,
                    device_cols,
                )
                for s, arrays in got.items():
                    parts[s].append(tuple(np.asarray(x)[:real] for x in arrays))
            ci_cols, inner_cols, model_cols = {}, {}, {}
            for s in saved:
                if s not in live:
                    ci_cols[s] = inner_cols[s] = model_cols[s] = np.zeros(
                        (grid.n_a * grid.n_b, 1, 0)
                    )
                    continue
                ci, inner, model = (np.concatenate([p[k] for p in parts[s]]) for k in range(3))
                for arr in (ci, inner, model):
                    arr[invalid] = np.nan
                ci_cols[s], inner_cols[s], model_cols[s] = (
                    ci[:, None],
                    inner[:, None],
                    model[:, None],
                )
            snapshot = ABGridSnapshot(
                mean_ci={"output": {s: m[t : t + 1] for s, m in mean_ci[op].items()}},
                saved=saved,
                ci_columns={"output": ci_cols},
                inner_columns=inner_cols,
            )
            args = (snapshot, grid, (t,), pool.seq_len, step, 0.0)
            if factor > 1:
                index, modules = ab_grid_split_payload(*args)
            else:
                index, modules = ab_grid_payload(*args), {}
            write_slice(grid_dir, op, grid.symbol, t, index, modules)
            write_model_slice(grid_dir, op, grid, t, model_cols, saved, factor)
            counts[slice_filename(op, t)] = int(sum(ids.size for ids in saved.values()))
            logger.info(
                f"slice {op} pos {t}: {counts[slice_filename(op, t)]} components "
                f"({time.time() - t0:.0f}s)"
            )
    return counts


def write_model_slice(
    grid_dir: Path,
    op: str,
    grid: ArithmeticGrid,
    position: int,
    model_cols: dict[str, np.ndarray],
    saved: dict[str, np.ndarray],
    factor: int,
) -> None:
    """`alive/<slice>`: the integer-only model's inner grids, in `alive_grids.py`'s format."""
    key = slice_key(grid.symbol, position)
    stem = slice_filename(op, position).removesuffix(".js")
    doc: dict[str, Any] = {"slice": key, "modules": {}}
    for site, ids in saved.items():
        if ids.size == 0:
            continue
        grids = np.transpose(
            model_cols[site][:, 0].reshape(grid.n_a, grid.n_b, ids.size), (2, 0, 1)
        )
        if factor == 1:
            doc["modules"][site] = {"inner": _b64(grids.astype(np.float16))}
            continue
        file = f"{stem}/{site}.js"
        path = grid_dir / "alive" / file
        path.parent.mkdir(parents=True, exist_ok=True)
        module = {"slice": key, "name": site, "inner": _b64(grids.astype(np.float16))}
        path.write_text(f"window.registerAliveModule({json.dumps(module)});")
        pooled = pool_grid(grids, factor)
        doc["modules"][site] = {"inner": _b64(pooled.astype(np.float16)), "file": file}
    (grid_dir / "alive" / slice_filename(op, position)).write_text(
        f"window.registerAliveSlice({json.dumps(doc)});"
    )


def write_applet(grid_dir: Path, pool: Pool, positions: tuple[int, ...], title: str) -> None:
    manifest = {
        "title": title,
        "ops": [OPERATIONS[op][0] for op in pool.ops],
        "positions": list(positions),
        "position_labels": pool.position_labels,
        "alive_label": "integer-only model",
        "files": {
            slice_key(OPERATIONS[op][0], t): slice_filename(op, t)
            for op in pool.ops
            for t in positions
        },
    }
    (grid_dir / "manifest.js").write_text(f"window.AB_GRID_SLICES = {json.dumps(manifest)};\n")
    (grid_dir / "index.html").write_text(alive_page())
    (grid_dir / "grids_app.html").write_bytes(APPLET.read_bytes())


# ───────────────────────────── entry ─────────────────────────────


def main() -> None:
    setup_console_logger()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--finetune_run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="fine-tune step (default: latest)")
    ap.add_argument("--parent_run_dir", type=Path, required=True)
    ap.add_argument("--parent_step", type=int, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", default="Answer: {a} {op} {b} is ")
    ap.add_argument("--a_range", type=int, nargs=2, default=(0, 399))
    ap.add_argument("--b_range", type=int, nargs=2, default=(0, 399))
    ap.add_argument("--prompts_file", type=Path, default=None)
    ap.add_argument("--positions", type=int, nargs="+", default=(4, 5, 6, 7, 8, 9))
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument(
        "--eval_chunk", type=int, default=200, help="eval prompts per forward (full logits)"
    )
    ap.add_argument("--skip_grids", action="store_true")
    args = ap.parse_args()

    restored = restore_jax_run(args.finetune_run_dir, args.step, data_root=args.data_root)
    target = restored.deliverable.target
    assert isinstance(target, TargetConfig), "the pool reads an HF-family target's tokenizer"
    placed, mesh, step = restored.placed, restored.mesh, restored.step
    assert args.chunk % mesh.size == 0 and args.eval_chunk % mesh.size == 0, mesh.size
    parent = restore_other_decomposition(restored, args.parent_run_dir, args.parent_step)
    tokenizer = load_tokenizer(target.model_name)
    pool = Pool(args.template, tuple(args.a_range), tuple(args.b_range), tokenizer)
    if args.prompts_file is not None:
        check_against_prompts_file(pool, args.prompts_file)
    positions = tuple(args.positions)
    assert all(0 <= t < pool.seq_len for t in positions), (positions, pool.seq_len)
    logger.info(
        f"fine-tune {args.finetune_run_dir} step {step}, ceiling {args.parent_run_dir} step "
        f"{args.parent_step}; pool {pool.tokens.shape}, {int(pool.valid.sum())} valid, "
        f"positions {pool.position_labels}"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    with jax.set_mesh(mesh):
        constraints = CIConstraints(ceilings=(cast_floating(parent.ci_fn, COMPUTE_DT),), kept=None)
        del parent
        ci_fn = restored.ci_fn
        prepared, v_norms = prepare_components(placed, restored.components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        valid_all = jax.sharding.reshard(jnp.asarray(pool.valid), P())

        t0 = time.time()
        max_ci, mean_ci = alive_tables(
            placed, prepared, ci_fn, constraints, tokens_all, valid_all, pool, args.chunk
        )
        alive_np = {s: m > args.threshold for s, m in max_ci.items()}
        np.savez(args.out / "max_ci.npz", **max_ci)  # pyright: ignore[reportArgumentType]
        n_alive = [int(sum(a[t].sum() for a in alive_np.values())) for t in range(pool.seq_len)]
        alive_doc = {
            "rule": (
                f"component c of site s is alive at position t iff its capped output-head lower "
                f"CI exceeds {args.threshold} at t on some pool prompt"
            ),
            "threshold": args.threshold,
            "source": {
                "finetune_run_dir": str(args.finetune_run_dir),
                "finetune_step": step,
                "ceiling_run_dir": str(args.parent_run_dir),
                "ceiling_step": args.parent_step,
            },
            "n_components": int(sum(a.shape[1] for a in alive_np.values())),
            "position_labels": pool.position_labels,
            "n_alive": n_alive,
            "n_alive_union": int(sum(a.any(axis=0).sum() for a in alive_np.values())),
            "positions": [
                {s: [int(c) for c in np.nonzero(a[t])[0]] for s, a in sorted(alive_np.items())}
                for t in range(pool.seq_len)
            ],
        }
        (args.out / "alive.json").write_text(json.dumps(alive_doc))
        logger.info(f"alive per position {n_alive} ({time.time() - t0:.0f}s)")

        alive_dev = {
            s: jax.sharding.reshard(jnp.asarray(a.astype(np.float32)), P())
            for s, a in alive_np.items()
        }
        integer_ids = jax.sharding.reshard(jnp.asarray(integer_token_ids(tokenizer, 0, 999)), P())
        t0 = time.time()
        metrics = evaluate(
            placed,
            prepared,
            ci_fn,
            constraints,
            alive_dev,
            tokens_all,
            pool,
            args.eval_chunk,
            integer_ids,
            args.threshold,
        )
        metrics["n_alive"] = n_alive
        (args.out / "eval.json").write_text(json.dumps(metrics, indent=1))
        logger.info(f"eval ({time.time() - t0:.0f}s): {json.dumps(metrics['all'])}")

        if not args.skip_grids:
            grid_dir = args.out / "ab_grids"
            counts = write_grids(
                grid_dir,
                placed,
                prepared,
                ci_fn,
                constraints,
                alive_np,
                alive_dev,
                v_norms,
                mean_ci,
                tokens_all,
                pool,
                positions,
                args.chunk,
                step,
            )
            write_applet(grid_dir, pool, positions, f"{args.parent_run_dir.name} integer-only")
            logger.info(f"grids: {counts}")


if __name__ == "__main__":
    main()
