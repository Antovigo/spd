"""The `(a, b)`-grid applet of a CI filter: every operation x every prompt position.

A slice `(operation, position)` is one file in the payload format of
`experiments/lm/ab_grid_dataset.py` (one op, one position), loaded lazily by the applet
`grids_app.html`, so the browser never holds the whole sweep.

Two passes over the pool keep only one chunk's grids on device at a time: pass 1 sums the
lower-leaky CI per (position, component) and decides which components each slice saves;
pass 2 recomputes each chunk and gathers only the columns some position of that operation
saves (their union), which the host splits per position."""

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int

from param_decomp.ci_filter.pool import ArithmeticPool, OperationBlock
from param_decomp.ci_filter.step import (
    Ceilings,
    Prepared,
    gather_rows,
    index_batches,
    output_ci,
)
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.ab_grid_dataset import (
    ABGridSnapshot,
    ab_grid_payload,
    saved_indices,
)

APPLET = Path(__file__).parent / "grids_app.html"
GATHER_INDEX_MULTIPLE = 64


@eqx.filter_jit
def _chunk_ci_sums(
    placed: PlacedModel,
    ci_fn: PlacedCIFn,
    ceilings: Ceilings,
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    n_real: Array,
) -> dict[str, Array]:
    """`{site: (T, C)}` lower CI summed over the chunk's real rows."""
    tokens = gather_rows(tokens_all, idx)
    clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
    lower = output_ci(placed, ci_fn, ceilings, clean.captures, remat=False).lower
    real = (jnp.arange(tokens.shape[0]) < n_real)[:, None, None]
    return {
        site: jax.sharding.reshard(jnp.where(real, v.astype(jnp.float32), 0.0).sum(axis=0), P())
        for site, v in lower.items()
    }


@eqx.filter_jit
def _chunk_columns(
    placed: PlacedModel,
    prepared: Prepared,
    ci_fn: PlacedCIFn,
    ceilings: Ceilings,
    v_norms: dict[str, Array],
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    columns: dict[str, Array],
) -> dict[str, tuple[Array, Array]]:
    """`{site: (ci, inner)}`, each `(B, T, len(columns[site]))` fp32 and replicated: the lower
    CI and the normalized inner activation `(x . V_c) / ||V_c||` of the requested components."""
    tokens = gather_rows(tokens_all, idx)
    clean, acts = placed.component_activation_forward(
        prepared, tokens, capture_keys=ci_fn.capture_keys
    )
    lower = output_ci(placed, ci_fn, ceilings, clean.captures, remat=False).lower
    out: dict[str, tuple[Array, Array]] = {}
    for site, cols in columns.items():
        ci = lower[site].astype(jnp.float32).at[:, :, cols].get(out_sharding=P())
        act = acts[site].astype(jnp.float32).at[:, :, cols].get(out_sharding=P())
        out[site] = (ci, act / jnp.maximum(v_norms[site][cols], 1e-12))
    return out


@dataclass(frozen=True)
class GridSlices:
    """One operation's snapshots, one per position."""

    block: OperationBlock
    snapshots: tuple[ABGridSnapshot, ...]


def collect_operation(
    placed: PlacedModel,
    prepared: Prepared,
    ci_fn: PlacedCIFn,
    ceilings: Ceilings,
    v_norms: dict[str, Array],
    tokens_all: Int[Array, "N T"],
    block: OperationBlock,
    seq_len: int,
    chunk_prompts: int,
    mean_ci_floor: float,
) -> GridSlices:
    n = block.stop - block.start
    chunks = [(idx + block.start, real) for idx, real in index_batches(n, chunk_prompts)]

    totals: dict[str, np.ndarray] = {}
    for idx, real in chunks:
        sums = _chunk_ci_sums(
            placed, ci_fn, ceilings, tokens_all, jnp.asarray(idx), jnp.asarray(real)
        )
        for site, value in sums.items():
            totals[site] = np.asarray(value) + totals.get(site, 0.0)
    mean_ci = {site: total / n for site, total in totals.items()}
    saved = [
        {site: saved_indices(m[p : p + 1], mean_ci_floor) for site, m in mean_ci.items()}
        for p in range(seq_len)
    ]
    union = {site: np.unique(np.concatenate([s[site] for s in saved])) for site in mean_ci}
    live = {site: u for site, u in union.items() if u.size}
    # Pad each index to a multiple so the gather retraces rarely; the host trims the pad.
    padded = {
        site: np.pad(u, (0, -(-u.size // GATHER_INDEX_MULTIPLE) * GATHER_INDEX_MULTIPLE - u.size))
        for site, u in live.items()
    }
    ci_parts: dict[str, list[np.ndarray]] = {site: [] for site in live}
    inner_parts: dict[str, list[np.ndarray]] = {site: [] for site in live}
    if live:
        device_cols = {site: jnp.asarray(p.astype(np.int32)) for site, p in padded.items()}
        for idx, real in chunks:
            got = _chunk_columns(
                placed,
                prepared,
                ci_fn,
                ceilings,
                v_norms,
                tokens_all,
                jnp.asarray(idx),
                device_cols,
            )
            for site, (ci, inner) in got.items():
                k = live[site].size
                ci_parts[site].append(np.asarray(ci)[:real, :, :k])
                inner_parts[site].append(np.asarray(inner)[:real, :, :k])
    ci_all = {site: np.concatenate(parts) for site, parts in ci_parts.items()}
    inner_all = {site: np.concatenate(parts) for site, parts in inner_parts.items()}

    snapshots = []
    for p in range(seq_len):
        ci_cols: dict[str, np.ndarray] = {}
        inner_cols: dict[str, np.ndarray] = {}
        for site in mean_ci:
            ids = saved[p][site]
            if ids.size == 0:
                ci_cols[site] = np.zeros((n, 1, 0), np.float32)
                inner_cols[site] = np.zeros((n, 1, 0), np.float32)
                continue
            where = np.searchsorted(live[site], ids)
            ci_cols[site] = ci_all[site][:, p : p + 1, where]
            inner_cols[site] = inner_all[site][:, p : p + 1, where]
        snapshots.append(
            ABGridSnapshot(
                mean_ci={"output": {site: m[p : p + 1] for site, m in mean_ci.items()}},
                saved=saved[p],
                ci_columns={"output": ci_cols},
                inner_columns=inner_cols,
            )
        )
    return GridSlices(block=block, snapshots=tuple(snapshots))


def slice_filename(operation: str, position: int) -> str:
    return f"{operation}_pos{position}.js"


def write_operation(
    grid_dir: Path, slices: GridSlices, pool: ArithmeticPool, step: int, mean_ci_floor: float
) -> dict[str, int]:
    """Write one file per position of this operation; returns saved-component counts."""
    grid_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for p, snapshot in enumerate(slices.snapshots):
        payload = ab_grid_payload(
            snapshot, slices.block.grid, (p,), pool.seq_len, step, mean_ci_floor
        )
        name = slice_filename(slices.block.operation, p)
        (grid_dir / name).write_text(f"window.registerABGridSlice({json.dumps(payload)});")
        counts[name] = int(sum(ids.size for ids in snapshot.saved.values()))
    return counts


def write_applet(grid_dir: Path, pool: ArithmeticPool, title: str) -> None:
    """`index.html` plus the manifest naming every slice file."""
    grid_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": title,
        "ops": [block.grid.symbol for block in pool.blocks],
        "positions": list(range(pool.seq_len)),
        "position_labels": list(pool.position_labels),
        "files": {
            f"{block.grid.symbol}:{p}": slice_filename(block.operation, p)
            for block in pool.blocks
            for p in range(pool.seq_len)
        },
    }
    (grid_dir / "manifest.js").write_text(f"window.AB_GRID_SLICES = {json.dumps(manifest)};\n")
    (grid_dir / "index.html").write_bytes(APPLET.read_bytes())
