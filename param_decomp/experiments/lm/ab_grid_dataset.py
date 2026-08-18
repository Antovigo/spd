"""Slow-eval `(a, b)`-grid snapshots of per-component CI and inner activations, plus the
`file://`-openable applet that browses them. See `param_decomp/CLAUDE.md`.
"""

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jaxtyping import Array, Float, Int

from param_decomp.core.ci_fn import CIRole, ci_preactivations, lower_leaky_hard_sigmoid
from param_decomp.core.components import ComponentStacks
from param_decomp.core.model import CaptureKeys, DecomposedModel, prepare_compute_weights
from param_decomp.experiments.lm.arithmetic_probe import ArithmeticGrid

AB_GRIDS_DIR = "ab_grids"
APPLET_FILENAME = "ab_grids_app.html"
MANIFEST_VAR = "AB_GRIDS_MANIFEST"

# The saved-column gather is a fresh jit shape every time the saved count moves, so the
# index is padded up to a multiple of this before the gather and trimmed on the host: a
# handful of traces over a run instead of one per snapshot.
GATHER_INDEX_MULTIPLE = 64

ABGridStep = Callable[
    [DecomposedModel, ComponentStacks, Any, Int[Array, "n_pad T"], Array],
    tuple[dict[CIRole, dict[str, Array]], dict[str, Array], dict[CIRole, dict[str, Array]]],
]
"""`(model, components, ci_fn, tokens, n_valid_rows) -> ({site: CI}, {site: inner}, {site:
summed CI})`. CI and inner activations are `(n_pad, n_pos, C)` at the recorded positions,
batch-sharded ON DEVICE (never fully host-gathered — see `collect_ab_grid_snapshot`); the
summed CI is `(n_pos, C)`, over the REAL (`< n_valid_rows`) rows only. `model`
(frozen-weight-bearing) is the jit ARG. `n_valid_rows` is a TRACED scalar, not static, so
every chunk of a chunked pass shares one compiled step even though the last chunk carries
fewer real rows."""


def make_ab_grid_step[PreparedT](
    model_static: DecomposedModel[PreparedT],
    ci_capture_keys: CaptureKeys,
    positions: tuple[int, ...],
    roles: tuple[CIRole, ...] = ("output",),
) -> ABGridStep:
    """Build the jit'd step returning, at each recorded position with the batch axis KEPT as
    the grid, BOTH per-component lower-leaky CI (from the CI fn) and the normalized inner
    activation `(x · V_c) / ‖V_c‖`, plus the per-position CI sum over the real rows
    (`n_valid_rows` masks the sharding-pad tail — garbage prompts must not move a mean).

    ONE frozen forward serves both: `component_activation_forward` returns the CI fn's taps
    and every site's `x @ V` from the same pass, so the snapshot costs no masked forward."""
    site_names = model_static.site_names
    position_index = np.asarray(positions, dtype=np.int32)

    # HLO-baking rule: read STATIC config (site_names, positions) off the closed-over
    # `model_static`; all array access goes through the traced `model` arg.
    @eqx.filter_jit
    def step(
        model: DecomposedModel[PreparedT],
        components: ComponentStacks,
        ci_fn: Any,
        tokens: Int[Array, "n_pad T"],
        n_valid_rows: Array,
    ) -> tuple[dict[CIRole, dict[str, Array]], dict[str, Array], dict[CIRole, dict[str, Array]]]:
        prepared_weights = prepare_compute_weights(model, components)
        clean_forward_result, component_activations = model.component_activation_forward(
            prepared_weights, tokens, capture_keys=ci_capture_keys
        )
        # Every role off the ONE frozen forward: the CI fn's trunk runs once and the heads
        # are separate readouts of it (S36), so a dual snapshot costs no extra forward — only
        # the second head's `[d_model, C]` matmul and its grids.
        preactivations_by_role: dict[CIRole, dict[str, Array]] = {
            role: ci_preactivations(ci_fn, clean_forward_result.captures, remat=False, role=role)
            for role in roles
        }
        first = preactivations_by_role[roles[0]]
        assert first[site_names[0]].ndim == 3, (
            f"the ab grid is LM-only ((n_prompts, T, C)); got {first[site_names[0]].shape}"
        )
        recorded = jnp.asarray(position_index)
        ci: dict[CIRole, dict[str, Array]] = {
            role: {
                site: lower_leaky_hard_sigmoid(preactivations[site])[:, recorded, :].astype(
                    jnp.float32
                )
                for site in site_names
            }
            for role, preactivations in preactivations_by_role.items()
        }
        inner = {}
        for site in site_names:
            v_norm = jnp.linalg.norm(components.site(site).V.astype(jnp.float32), axis=0)
            inner[site] = component_activations[site][:, recorded, :].astype(
                jnp.float32
            ) / jnp.maximum(v_norm, 1e-12)
        valid_rows = (jnp.arange(tokens.shape[0]) < n_valid_rows)[:, None, None]
        ci_sum: dict[CIRole, dict[str, Array]] = {
            role: {
                site: jnp.where(valid_rows, role_ci[site], 0.0).sum(axis=0, dtype=jnp.float32)
                for site in site_names
            }
            for role, role_ci in ci.items()
        }
        return ci, inner, ci_sum

    return step


@eqx.filter_jit
def _take_columns(per_prompt: Float[Array, "n_pad n_pos C"], idx: Int[Array, " k"]) -> Array:
    return jnp.take(per_prompt, idx, axis=2)


def saved_indices(mean_ci: np.ndarray, mean_ci_floor: float) -> np.ndarray:
    """Components whose prompt-mean CI reaches the floor at SOME recorded position. The
    mean-CI vector is saved for every component either way, so the cut stays visible in the
    applet's threshold slider."""
    assert mean_ci.ndim == 2, mean_ci.shape
    return np.nonzero(mean_ci.max(axis=0) >= mean_ci_floor)[0]


@dataclass(frozen=True)
class ABGridSnapshot:
    """Host-side result of one grid pass: the full per-position mean-CI vectors, and the
    per-prompt columns of the components that cleared the floor."""

    mean_ci: dict[CIRole, dict[str, np.ndarray]]
    """`{role: {site: (n_pos, C)}}` fp32, the prompt-mean lower-leaky CI per readout head."""
    saved: dict[str, np.ndarray]
    """`{site: component ids whose grids were gathered}`, ascending — ONE index set shared by
    every role. The floor is applied to the MAX over roles, so a subcomponent only the hidden
    head cares about is not filtered away (the applet indexes both roles' grids by this list)."""
    ci_columns: dict[CIRole, dict[str, np.ndarray]]
    """`{role: {site: (n_prompts, n_pos, len(saved[site]))}}` fp32, row-major `(a, b)` order."""
    inner_columns: dict[str, np.ndarray]
    """Role-INDEPENDENT: the inner activation is `(x · V_c) / ‖V_c‖`, which no CI head enters."""


def collect_ab_grid_snapshot(
    step: ABGridStep,
    model: DecomposedModel,
    components: ComponentStacks,
    ci_fn: Any,
    chunks: tuple[tuple[Int[Array, "n_pad T"], int], ...],
    n_prompts: int,
    mean_ci_floor: float,
) -> ABGridSnapshot:
    """Two-phase device->host pull, sized to what the snapshot stores — NEVER the full
    `(n_prompts, n_pos, C)` grids (they scale as n_prompts x C per site). Phase 1: the
    step's per-position CI sums come to host (an `(n_pos, C)` array per site) and drive the
    floor cut, identically on every rank. Phase 2: only the saved columns are host-gathered
    (the index padded up to a `GATHER_INDEX_MULTIPLE` boundary so the gather retraces
    rarely; each chunk's real-row count trims its sharding pad off the END). Both the step
    and the column gather are COLLECTIVE — all ranks join, and every rank walks the same
    chunks in the same order.

    `chunks` is `((tokens, n_valid_rows), ...)`: the grid split so ONE forward never carries
    the whole operand sweep. The per-chunk CI / inner grids stay on device between the
    phases (a few tens of MB total, unlike the forward that produced them), so chunking
    costs no extra forward. The sums add over chunks and the columns concatenate in chunk
    order, so WHICH components are saved is chunk-count-invariant exactly; the gathered
    values match a single-chunk pass up to float reassociation (SPEC D4)."""
    per_chunk: list[tuple[dict[CIRole, dict[str, Array]], dict[str, Array], int]] = []
    ci_totals: dict[CIRole, dict[str, np.ndarray]] = {}
    for chunk_tokens, chunk_valid_rows in chunks:
        ci, inner, chunk_sum = step(
            model, components, ci_fn, chunk_tokens, jnp.asarray(chunk_valid_rows)
        )
        per_chunk.append((ci, inner, chunk_valid_rows))
        for role, role_sum in chunk_sum.items():
            totals = ci_totals.setdefault(role, {})
            for site, value in role_sum.items():
                summed = np.asarray(value)
                totals[site] = summed if site not in totals else totals[site] + summed

    roles: tuple[CIRole, ...] = tuple(ci_totals)
    mean_ci: dict[CIRole, dict[str, np.ndarray]] = {
        role: {site: total / n_prompts for site, total in totals.items()}
        for role, totals in ci_totals.items()
    }
    # ONE index set for every role, cut on the MAX across them: a subcomponent that only the
    # hidden head cares about must not be filtered away, and the applet indexes both roles'
    # grids by this same list.
    sites = tuple(mean_ci[roles[0]])
    saved = {
        site: saved_indices(
            np.maximum.reduce([mean_ci[role][site] for role in roles]), mean_ci_floor
        )
        for site in sites
    }

    ci_columns: dict[CIRole, dict[str, np.ndarray]] = {role: {} for role in roles}
    inner_columns: dict[str, np.ndarray] = {}
    n_pos = next(iter(mean_ci[roles[0]].values())).shape[0]
    empty = {site for site, idx in saved.items() if idx.size == 0}
    for site in empty:
        for role in roles:
            ci_columns[role][site] = np.zeros((n_prompts, n_pos, 0), np.float32)
        inner_columns[site] = np.zeros((n_prompts, n_pos, 0), np.float32)

    live = [site for site in sites if site not in empty]
    ci_parts: dict[CIRole, dict[str, list[np.ndarray]]] = {
        role: {site: [] for site in live} for role in roles
    }
    inner_parts: dict[str, list[np.ndarray]] = {site: [] for site in live}
    for ci, inner, chunk_valid_rows in per_chunk:
        to_gather: dict[str, tuple[tuple[Array, ...], Array]] = {}
        for site in live:
            idx = saved[site]
            width = -(-idx.size // GATHER_INDEX_MULTIPLE) * GATHER_INDEX_MULTIPLE
            padded_idx = jnp.asarray(np.pad(idx, (0, width - idx.size), mode="edge"))
            to_gather[site] = (
                tuple(_take_columns(ci[role][site], padded_idx) for role in roles),
                _take_columns(inner[site], padded_idx),
            )
        if not to_gather:
            break
        gathered = multihost_utils.process_allgather(to_gather, tiled=True)
        for site, (ci_cols_per_role, inner_cols) in gathered.items():
            k = saved[site].size
            for role, ci_cols in zip(roles, ci_cols_per_role, strict=True):
                ci_parts[role][site].append(np.asarray(ci_cols)[:chunk_valid_rows, :, :k])
            inner_parts[site].append(np.asarray(inner_cols)[:chunk_valid_rows, :, :k])
    for site in live:
        for role in roles:
            ci_columns[role][site] = np.concatenate(ci_parts[role][site], axis=0)[:n_prompts]
        inner_columns[site] = np.concatenate(inner_parts[site], axis=0)[:n_prompts]
    return ABGridSnapshot(
        mean_ci=mean_ci, saved=saved, ci_columns=ci_columns, inner_columns=inner_columns
    )


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def encode_ci_u8(ci: np.ndarray) -> np.ndarray:
    return np.round(np.clip(ci, 0.0, 1.0) * 255.0).astype(np.uint8)


def _comp_major(columns: np.ndarray, grid: ArithmeticGrid) -> np.ndarray:
    """`(n_prompts, n_pos, k)` row-major `(a, b)` -> the applet's `[comp, pos, op, a, b]`.

    The op axis is length 1: one metric records one operation, while the payload layout
    (and the applet's op selector) carries the axis a multi-op pool would fill."""
    n_prompts, n_pos, k = columns.shape
    assert n_prompts == grid.n_a * grid.n_b, (columns.shape, grid.n_a, grid.n_b)
    per_cell = columns.reshape(grid.n_a, grid.n_b, n_pos, k)
    return np.transpose(per_cell, (3, 2, 0, 1))[:, :, None]


def ab_grid_payload(
    snapshot: ABGridSnapshot,
    grid: ArithmeticGrid,
    positions: tuple[int, ...],
    seq_len: int,
    step: int,
    mean_ci_floor: float,
) -> dict[str, Any]:
    """The applet's snapshot document. CI grids are quantized to u8 (1/255 steps) and inner
    activations to f16; the mean-CI vectors stay fp32."""
    # The applet keys the OUTPUT role's arrays on the historical names (`mean_ci`, `ci`) and
    # the hidden role's on `*_hidden`, so a single-role payload stays byte-compatible with
    # every snapshot written before S36 and the applet's own pre-dual fallback still applies.
    roles = tuple(snapshot.mean_ci)
    output_mean = snapshot.mean_ci["output"]
    hidden_mean = snapshot.mean_ci.get("hidden")
    modules_payload: list[dict[str, Any]] = []
    for site, mean_ci in output_mean.items():
        saved = snapshot.saved[site]
        entry: dict[str, Any] = {
            "name": site,
            "C": int(mean_ci.shape[1]),
            "saved": [int(c) for c in saved],
            "mean_ci": _b64(mean_ci.astype(np.float32)),
        }
        if hidden_mean is not None:
            entry["mean_ci_hidden"] = _b64(hidden_mean[site].astype(np.float32))
        if saved.size > 0:
            entry["ci"] = _b64(encode_ci_u8(_comp_major(snapshot.ci_columns["output"][site], grid)))
            if hidden_mean is not None:
                entry["ci_hidden"] = _b64(
                    encode_ci_u8(_comp_major(snapshot.ci_columns["hidden"][site], grid))
                )
            entry["inner"] = _b64(
                _comp_major(snapshot.inner_columns[site], grid).astype(np.float16)
            )
        modules_payload.append(entry)
    return {
        "step": step,
        "positions": list(positions),
        "seq_len": seq_len,
        "ops": [grid.symbol],
        "a_min": grid.a_values[0],
        "n_a": grid.n_a,
        "b_min": grid.b_values[0],
        "n_b": grid.n_b,
        "mean_ci_floor": mean_ci_floor,
        "ci_roles": list(roles),
        "modules": modules_payload,
    }


def write_ab_grid_snapshot(run_dir: Path, step: int, payload: dict[str, Any]) -> None:
    """Write `<run_dir>/ab_grids/step_<n>.js` next to the applet, regenerating `manifest.js`
    so a `file://`-opened `index.html` discovers every snapshot written so far. Process-0
    only — the caller owns that gate."""
    target = run_dir / AB_GRIDS_DIR
    target.mkdir(parents=True, exist_ok=True)
    (target / f"step_{step}.js").write_text(f"window.registerABGrids({json.dumps(payload)});")
    (target / "index.html").write_bytes((Path(__file__).parent / APPLET_FILENAME).read_bytes())
    snapshots = sorted(target.glob("step_*.js"), key=lambda p: int(p.stem.removeprefix("step_")))
    listing = json.dumps([p.name for p in snapshots])
    (target / "manifest.js").write_text(f"window.{MANIFEST_VAR} = {listing};\n")
