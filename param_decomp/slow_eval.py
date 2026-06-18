"""JAX-native slow (plot-type) eval metrics, the offline counterpart of `eval.py`.

`eval.py` runs the FAST scalar tier in-loop (CE/KL, CI-L0, the fresh-PGD probe). The
SLOW tier is the heavy plot metrics deferred to this out-of-loop pass:
`CIHistograms`, `ComponentActivationDensity`, `CIMeanPerComponent` (the torch eval-
metric classes of the same names). Every one of them is a reduction over the per-site
causal-importance arrays from a masked-free forward, then a numpy/matplotlib plot. The
forward + reduction is JAX; the plotting is framework-agnostic (it mirrors the torch
`param_decomp_lab/eval_metrics/plotting.py` reductions on numpy arrays, no torch).

This runs as an OFFLINE pass over an on-disk checkpoint (`pd-slow-eval <run_dir>`):
rebuild the JAX target from the run's
config, restore the `TrainState`, accumulate the reductions over `n_steps` eval batches,
render the figures, and log them under `slow_eval/*` into the run's wandb (the dedicated
`slow_eval/step` axis — the live run's `_step` has advanced, so an explicit `step=` write
would be dropped). No torch, no export round-trip.

Cross-batch reductions are exact under micro-batching: density/mean accumulate
SUM-over-positions + a position count, divided once at the end (token-weighted mean,
uniform `(B, T)` makes it the plain mean). `CIHistograms` caps its raw-value sample at
`n_batches_accum` batches, matching the torch metric's `n_batches_accum` early-stop.

It also computes the two SCALAR hidden-acts recon eval metrics (`CIHiddenActsReconLoss`,
`StochasticHiddenActsReconLoss`) natively — per decomposed site, the summed MSE between
the masked-model and target-model site OUTPUT activations, divided once by the element
count (`hidden_acts_eval.py`). Those ride the `masked_site_outputs` model seam (SPEC S31,
amended 2026-06-16 from keep-on-bridge) and are emitted as scalars under the torch log
keys (`<ClassName>/<site>` + a combined `<ClassName>`).

The three CONFIG-GATED permutation metrics (`PermutedCIPlots`, `UVPlots`,
`IdentityCIError`) are recomputed natively too, off the run's `eval.metrics` block
(re-validated from `config.yaml`, since the trainer's `EvalConfig` drops the raw metric
list). They share one column permutation per site — identity (scipy
`linear_sum_assignment` on `-CI`) or dense (by column mass) — derived from the batch-mean
`(position, C)` upper-leaky CI matrix (`make_position_ci_step` / `accumulate_position_ci`):
`PermutedCIPlots` heatmaps the permuted lower/upper CI, `UVPlots` reorders the V/U columns
by the same permutation, and `IdentityCIError` reports the discrete CI-vs-target distance
(generalizing the toy `tms`/`resid_mlp` `identity_ci_error`). All three are LM-only (they
need the `(B, T, C)` position axis) and empty unless their config names them.
"""

import argparse
import fnmatch
import io
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from param_decomp.checkpoint import make_checkpoint_manager, restore_step
from param_decomp.ci_fn import lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from param_decomp.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    load_run_dir_config,
)
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.hidden_acts_eval import (
    accumulate_hidden_acts,
    hidden_acts_log_entries,
    make_ci_hidden_acts_step,
    make_stochastic_hidden_acts_step,
)
from param_decomp.lm import DecomposedModel
from param_decomp.load_run import build_target
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.sharding import dp_mesh
from param_decomp_config.eval_metrics import (
    DenseCITargetSpec,
    IdentityCIErrorConfig,
    IdentityCITargetSpec,
    PermutedCIPlotsConfig,
    UVPlotsConfig,
)
from param_decomp_config.routing import SamplingType

IDENTITY_CI_ERROR_TOLERANCE = 0.1
"""Torch `IdentityCIPattern.distance_from` / `compute_target_metrics` default tolerance —
avoids sensitivity to small CI values from inactive components."""


@dataclass(frozen=True)
class SiteReduction:
    """Per-site accumulators across the eval pass (all `(C,)` or scalar / capped sample).

    `density_counts[c]` = #(positions where `lower_leaky > threshold`); `ci_sums[c]` =
    Σ positions `lower_leaky`; `n_positions` = total positions seen (shared count for
    both means). `lower_sample` / `logits_sample` are flattened raw values from the first
    `n_batches_accum` batches, for the two `CIHistograms` histograms."""

    density_counts: np.ndarray
    ci_sums: np.ndarray
    n_positions: int
    lower_sample: np.ndarray
    logits_sample: np.ndarray


SlowEvalStep = Callable[
    [Any, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]],
]
"""`(ci_fn, frozen, residual) -> (density_counts, ci_sums, n_positions, flat_lower,
flat_logits)` — the per-batch reduction, pre-reduced over positions. The slow plot
metrics read only the CI arrays, so V/U (`components`) is not an input."""


def make_slow_eval_step(lm: DecomposedModel, ci_alive_threshold: float) -> SlowEvalStep:
    """Build the jit'd per-batch reduction `slow_eval_step(ci_fn, frozen, residual) ->
    ({site: density_counts}, {site: ci_sums}, n_positions, {site: flat lower},
    {site: flat logits})`. `lower`/`logits` are returned whole (the host caps the
    histogram sample); counts/sums are pre-reduced over positions."""
    site_names = lm.site_names

    @jax.jit
    def slow_eval_step(
        ci_fn: Any, frozen: Any, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array, dict[str, Array], dict[str, Array]]:
        # CI fn stays fp32 (its master dtype): torch offline-eval keeps V/U + CI fn fp32,
        # casting only the frozen target to bf16. The slow plot metrics are a
        # fp32-CI-fn readout, so we don't take eval.py's bf16-compute path here.
        site_inputs = {
            s: x.astype(jnp.float32) for s, x in lm.site_inputs(frozen, residual).items()
        }
        logits = ci_fn.site_logits(site_inputs)
        lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in site_names}

        density_counts = {
            s: (lower[s] > ci_alive_threshold)
            .astype(jnp.float32)
            .reshape(-1, lower[s].shape[-1])
            .sum(0)
            for s in site_names
        }
        ci_sums = {s: lower[s].reshape(-1, lower[s].shape[-1]).sum(0) for s in site_names}
        first = lower[site_names[0]]
        n_positions = jnp.asarray(math.prod(first.shape[:-1]), jnp.int32)
        flat_lower = {s: lower[s].reshape(-1) for s in site_names}
        flat_logits = {s: logits[s].reshape(-1) for s in site_names}
        return density_counts, ci_sums, n_positions, flat_lower, flat_logits

    return slow_eval_step


def accumulate_site_reductions(
    slow_eval_step: SlowEvalStep,
    ci_fn: Any,
    frozen: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    n_batches_accum: int | None,
) -> dict[str, SiteReduction]:
    """Drive `slow_eval_step` over the eval batches and fold the per-batch reductions
    into one `SiteReduction` per site. `n_batches_accum` caps how many batches feed the
    `CIHistograms` raw-value sample (torch `n_batches_accum`); None keeps all."""
    assert residual_batches, "slow eval needs at least one batch"
    density: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    lower_chunks: dict[str, list[np.ndarray]] = {}
    logits_chunks: dict[str, list[np.ndarray]] = {}
    total_positions = 0
    for batch_idx, residual in enumerate(residual_batches):
        d, s, n_pos, flat_lower, flat_logits = slow_eval_step(ci_fn, frozen, residual)
        total_positions += int(n_pos)
        keep_sample = n_batches_accum is None or batch_idx < n_batches_accum
        for site in d:
            counts, ci_sum = np.asarray(d[site]), np.asarray(s[site])
            density[site] = counts if batch_idx == 0 else density[site] + counts
            sums[site] = ci_sum if batch_idx == 0 else sums[site] + ci_sum
            if keep_sample:
                lower_chunks.setdefault(site, []).append(np.asarray(flat_lower[site]))
                logits_chunks.setdefault(site, []).append(np.asarray(flat_logits[site]))

    return {
        site: SiteReduction(
            density_counts=density[site],
            ci_sums=sums[site],
            n_positions=total_positions,
            lower_sample=np.concatenate(lower_chunks[site]),
            logits_sample=np.concatenate(logits_chunks[site]),
        )
        for site in density
    }


PositionCIStep = Callable[
    [Any, Any, Float[Array, "*leading d"]],
    tuple[dict[str, Array], dict[str, Array], Array],
]
"""`(ci_fn, frozen, residual) -> ({site: lower (T, C)}, {site: upper (T, C)}, n_batch)` —
the per-batch CI summed over the batch leading axis, position axis kept. Pairs with
`accumulate_position_ci` to form a batch-mean `(T, C)` CI matrix per site."""


def make_position_ci_step(lm: DecomposedModel) -> PositionCIStep:
    """Per-batch CI reduction that KEEPS the position axis (the `(T, C)` matrix the
    permutation/heatmap metrics plot), summing only over the batch leading axis. LM-only:
    the residual is `(B, T, d)` and CI is `(B, T, C)`."""
    site_names = lm.site_names

    @jax.jit
    def position_ci_step(
        ci_fn: Any, frozen: Any, residual: Float[Array, "*leading d"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        site_inputs = {
            s: x.astype(jnp.float32) for s, x in lm.site_inputs(frozen, residual).items()
        }
        logits = ci_fn.site_logits(site_inputs)
        lower = {s: lower_leaky_hard_sigmoid(logits[s]) for s in site_names}
        upper = {s: upper_leaky_hard_sigmoid(logits[s]) for s in site_names}
        first = lower[site_names[0]]
        assert first.ndim == 3, f"position CI metrics are LM-only ((B, T, C)); got {first.shape}"
        n_batch = jnp.asarray(first.shape[0], jnp.int32)
        lower_sum = {s: lower[s].sum(0) for s in site_names}  # (T, C)
        upper_sum = {s: upper[s].sum(0) for s in site_names}
        return lower_sum, upper_sum, n_batch

    return position_ci_step


@dataclass(frozen=True)
class PositionCI:
    """Batch-mean CI matrices for one site, position axis kept (`(T, C)`)."""

    lower: np.ndarray
    upper: np.ndarray


def accumulate_position_ci(
    position_ci_step: PositionCIStep,
    ci_fn: Any,
    frozen: Any,
    residual_batches: list[Float[Array, "*leading d"]],
) -> dict[str, PositionCI]:
    """Fold `position_ci_step` over the eval batches into a batch-mean `(T, C)` CI matrix
    per site (token-weighted mean over batch elements; uniform batch makes it the plain
    mean). All batches must share one `(B, T)` shape."""
    assert residual_batches, "position CI accumulation needs at least one batch"
    lower: dict[str, np.ndarray] = {}
    upper: dict[str, np.ndarray] = {}
    total_batch = 0
    for batch_idx, residual in enumerate(residual_batches):
        lo, hi, n_batch = position_ci_step(ci_fn, frozen, residual)
        total_batch += int(n_batch)
        for site in lo:
            lo_np, hi_np = np.asarray(lo[site]), np.asarray(hi[site])
            lower[site] = lo_np if batch_idx == 0 else lower[site] + lo_np
            upper[site] = hi_np if batch_idx == 0 else upper[site] + hi_np
    assert total_batch > 0
    return {
        site: PositionCI(lower=lower[site] / total_batch, upper=upper[site] / total_batch)
        for site in lower
    }


def permute_to_identity(ci_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column permutation toward identity via Hungarian on `-ci` over the `min(shape)`
    square block, with unassigned columns appended in order. Returns
    `(permuted (rows, C), perm_indices (C,))`. Mirrors torch `permute_to_identity_hungarian`
    / the toy `identity_ci_error`'s permutation."""
    from scipy.optimize import linear_sum_assignment

    assert ci_vals.ndim == 2, ci_vals.shape
    rows, C = ci_vals.shape
    size = min(rows, C)
    _, col_indices = linear_sum_assignment(-ci_vals[:size])
    assigned = set(col_indices.tolist())
    remaining = [c for c in range(C) if c not in assigned]
    perm = np.array(list(col_indices) + remaining, dtype=np.int64)
    return ci_vals[:, perm], perm


def permute_to_dense(ci_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column permutation by total mass, densest first (torch `permute_to_dense`).
    Returns `(permuted (rows, C), perm_indices (C,))`."""
    assert ci_vals.ndim == 2, ci_vals.shape
    perm = np.argsort(-ci_vals.sum(axis=0))
    return ci_vals[:, perm], perm


def identity_ci_error(ci_vals: np.ndarray, tolerance: float) -> int:
    """Discrete identity-CI distance (torch `IdentityCIPattern.distance_from`,
    generalizing the toy `tms`/`resid_mlp` `identity_ci_error`): permute columns toward
    identity, then over the `min(shape)` square block count off-diagonal entries
    `> tolerance` plus on-diagonal entries `< 1 - tolerance`."""
    ci = ci_vals.astype(np.float64)
    permuted, _ = permute_to_identity(ci)
    size = min(permuted.shape)
    block = permuted[:size, :size]
    off_diag = ~np.eye(size, dtype=bool)
    off_diag_errors = int((block[off_diag] > tolerance).sum())
    on_diag_errors = int((np.diagonal(block) < (1 - tolerance)).sum())
    return off_diag_errors + on_diag_errors


def dense_ci_error(ci_vals: np.ndarray, k: int, tolerance: float, min_entries: int = 1) -> int:
    """Discrete dense-CI distance (torch `DenseCIPattern.distance_from`): sort columns by
    total mass, then over the first `k` columns count one error per column with fewer than
    `min_entries` strong activations (`>= 1 - tolerance`), and over the rest one error per
    weak activation (`> tolerance`)."""
    ci = ci_vals.astype(np.float64)
    C = ci.shape[1]
    assert k <= C, f"expected at least {k} columns, got {C}"
    sorted_ci, _ = permute_to_dense(ci)
    strong = (sorted_ci >= 1 - tolerance).sum(axis=0)
    missing_strong = np.clip(min_entries - strong, a_min=0, a_max=None)
    first_k_error = int(missing_strong[:k].sum())
    weak = (sorted_ci > tolerance).sum(axis=0)
    inactive_error = int(weak[k:].sum())
    return first_k_error + inactive_error


@dataclass(frozen=True)
class PermutationMetricSpec:
    """The permutation-plot / identity-error metrics resolved against the run's sites.

    `permutation` records, per matched site, which target shape (`identity` / `dense`)
    governs its column permutation — driving both the `PermutedCIPlots` heatmaps and the
    `UVPlots` V/U column reorder. `identity_targets` / `dense_targets` add the
    `IdentityCIError` discrete distances (per-site, by fnmatch pattern over site names).
    Empty maps mean the corresponding metric is not configured."""

    permutation: dict[str, "Literal['identity', 'dense']"]
    identity_targets: dict[str, int]
    dense_targets: dict[str, int]
    want_uv_plots: bool

    @property
    def any_plots(self) -> bool:
        return bool(self.permutation)

    @property
    def any_identity_error(self) -> bool:
        return bool(self.identity_targets) or bool(self.dense_targets)


def _resolve_permutation(
    site_names: tuple[str, ...],
    identity_patterns: list[str] | None,
    dense_patterns: list[str] | None,
) -> dict[str, "Literal['identity', 'dense']"]:
    """Map each site to its permutation target (torch `plot_causal_importance_vals`:
    identity patterns win, then dense, else default identity)."""
    resolved: dict[str, Literal["identity", "dense"]] = {}
    for name in site_names:
        if identity_patterns and any(fnmatch.fnmatch(name, p) for p in identity_patterns):
            resolved[name] = "identity"
        elif dense_patterns and any(fnmatch.fnmatch(name, p) for p in dense_patterns):
            resolved[name] = "dense"
        else:
            resolved[name] = "identity"
    return resolved


def resolve_permutation_metrics(
    site_names: tuple[str, ...], metrics: list[Any]
) -> PermutationMetricSpec:
    """Build the `PermutationMetricSpec` from the run config's typed `eval.metrics` entries
    (`UVPlots` / `PermutedCIPlots` / `IdentityCIError`). The two plot metrics share one
    column permutation; `UVPlots` additionally reorders V/U. Permutation is only computed
    when at least one plot metric is configured (both reuse it)."""
    plot_cfgs = [m for m in metrics if isinstance(m, (PermutedCIPlotsConfig, UVPlotsConfig))]
    want_uv = any(isinstance(m, UVPlotsConfig) for m in metrics)
    permutation: dict[str, Literal["identity", "dense"]] = {}
    if plot_cfgs:
        identity_patterns: list[str] = []
        dense_patterns: list[str] = []
        for cfg in plot_cfgs:
            identity_patterns += cfg.identity_patterns or []
            dense_patterns += cfg.dense_patterns or []
        permutation = _resolve_permutation(site_names, identity_patterns, dense_patterns)

    identity_targets: dict[str, int] = {}
    dense_targets: dict[str, int] = {}
    for metric in metrics:
        if not isinstance(metric, IdentityCIErrorConfig):
            continue
        for spec in metric.identity_ci or []:
            assert isinstance(spec, IdentityCITargetSpec)
            for name in site_names:
                if fnmatch.fnmatch(name, spec.layer_pattern):
                    identity_targets[name] = spec.n_features
        for spec in metric.dense_ci or []:
            assert isinstance(spec, DenseCITargetSpec)
            for name in site_names:
                if fnmatch.fnmatch(name, spec.layer_pattern):
                    dense_targets[name] = spec.k
    return PermutationMetricSpec(
        permutation=permutation,
        identity_targets=identity_targets,
        dense_targets=dense_targets,
        want_uv_plots=want_uv,
    )


def _render_figure(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _grid_dims(n: int, max_rows: int = 6) -> tuple[int, int]:
    n_cols = (n + max_rows - 1) // max_rows
    n_rows = min(n, max_rows)
    return n_rows, n_cols


def plot_ci_value_histograms(samples: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of flattened CI values (torch `plot_ci_values_histograms`)."""
    n_rows, n_cols = _grid_dims(len(samples))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(samples) :]:
        ax.set_visible(False)
    for ax, (name, values) in zip(flat_axes, samples.items(), strict=False):
        ax.hist(values, bins=bins)
        ax.set_yscale("log")
        ax.set_title(f"Causal importances for {name.replace('.', '_')}")
        ax.set_xlabel("Causal importance value")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_component_activation_density(densities: dict[str, np.ndarray], bins: int = 100) -> bytes:
    """Per-site histogram of per-component activation density (torch
    `plot_component_activation_density`)."""
    n_rows, n_cols = _grid_dims(len(densities))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    flat_axes = axs.T.ravel()
    for ax in flat_axes[len(densities) :]:
        ax.set_visible(False)
    for ax, (name, density) in zip(flat_axes, densities.items(), strict=False):
        ax.hist(density, bins=bins)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_figure(fig)


def plot_mean_component_cis_both_scales(
    mean_cis: dict[str, np.ndarray],
) -> tuple[bytes, bytes]:
    """Sorted-descending mean-CI scatter, linear and log y (torch
    `plot_mean_component_cis_both_scales`)."""
    sorted_data = {name: np.sort(v)[::-1] for name, v in mean_cis.items()}
    n_rows, n_cols = _grid_dims(len(sorted_data))
    images: list[bytes] = []
    for log_y in (False, True):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)
        flat_axes = axs.T.ravel()
        for ax in flat_axes[len(sorted_data) :]:
            ax.set_visible(False)
        for ax, (name, sorted_components) in zip(flat_axes, sorted_data.items(), strict=False):
            if log_y:
                ax.set_yscale("log")
            ax.scatter(range(len(sorted_components)), sorted_components, marker="x", s=10)
            ax.set_xlabel("Component")
            ax.set_ylabel("mean CI")
            ax.set_title(name, fontsize=10)
        fig.tight_layout()
        images.append(_render_figure(fig))
    return images[0], images[1]


def _plot_ci_matrices(matrices: dict[str, np.ndarray], colormap: str, title_prefix: str) -> bytes:
    """Per-site `(rows, C)` CI heatmaps stacked vertically with a shared colorbar (torch
    `_plot_causal_importances_figure`). `rows` is the position axis for the LM path."""
    n = len(matrices)
    fig, axs = plt.subplots(n, 1, figsize=(5, 5 * n), constrained_layout=True, squeeze=False)
    flat_axes = axs[:, 0]
    vmin = min(float(m.min()) for m in matrices.values())
    vmax = max(float(m.max()) for m in matrices.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    images = []
    for ax, (name, matrix) in zip(flat_axes, matrices.items(), strict=True):
        im = ax.matshow(matrix, aspect="auto", cmap=colormap, norm=norm)
        images.append(im)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")
        ax.set_xlabel("Subcomponent index")
        ax.set_ylabel("Position index")
        ax.set_title(name)
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    fig.suptitle(title_prefix)
    return _render_figure(fig)


def plot_permuted_ci_heatmaps(
    position_ci: dict[str, PositionCI], permutation: dict[str, "Literal['identity', 'dense']"]
) -> tuple[bytes, bytes]:
    """The `PermutedCIPlots` figures: per-site `(position, C)` CI heatmaps with columns
    permuted toward each site's target shape (identity / dense). Lower-leaky (`Blues`) and
    upper-leaky (`Reds`) views of the SAME permutation (derived from upper-leaky, as in
    torch). Returns `(lower_png, upper_png)`."""
    assert set(permutation) <= set(position_ci), "permutation sites must be a subset of CI sites"
    lower_permuted: dict[str, np.ndarray] = {}
    upper_permuted: dict[str, np.ndarray] = {}
    for name, target in permutation.items():
        pci = position_ci[name]
        permute = permute_to_identity if target == "identity" else permute_to_dense
        _, perm = permute(pci.upper)
        lower_permuted[name] = pci.lower[:, perm]
        upper_permuted[name] = pci.upper[:, perm]
    lower_png = _plot_ci_matrices(lower_permuted, "Blues", "Importance values lower leaky relu")
    upper_png = _plot_ci_matrices(upper_permuted, "Reds", "Importance values")
    return lower_png, upper_png


def plot_uv_matrices(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    permutation: dict[str, "Literal['identity', 'dense']"],
    position_ci: dict[str, PositionCI],
) -> bytes:
    """The `UVPlots` figure: per-site V `(d_in, C)` and U `(C, d_out)` heatmaps with the
    component axis reordered by the same identity/dense permutation the CI plots use (torch
    `plot_UV_matrices`). One row per site, V left / U right, shared colorbar."""
    names = sorted(components)
    perms = {}
    for name in names:
        permute = permute_to_identity if permutation[name] == "identity" else permute_to_dense
        perms[name] = permute(position_ci[name].upper)[1]
    n = len(names)
    fig, axs = plt.subplots(n, 2, figsize=(10, 5 * n), constrained_layout=True, squeeze=False)
    all_vals = [m for name in names for m in components[name]]
    norm = plt.Normalize(
        vmin=min(float(m.min()) for m in all_vals), vmax=max(float(m.max()) for m in all_vals)
    )
    images = []
    for row, name in enumerate(names):
        V, U = components[name]
        v_im = axs[row, 0].matshow(V[:, perms[name]], aspect="auto", cmap="coolwarm", norm=norm)
        axs[row, 0].set_ylabel("d_in index")
        axs[row, 0].set_xlabel("Component index")
        axs[row, 0].set_title(f"{name} (V matrix)")
        u_im = axs[row, 1].matshow(U[perms[name], :], aspect="auto", cmap="coolwarm", norm=norm)
        axs[row, 1].set_ylabel("Component index")
        axs[row, 1].set_xlabel("d_out index")
        axs[row, 1].set_title(f"{name} (U matrix)")
        images += [v_im, u_im]
    fig.colorbar(images[0], ax=axs.ravel().tolist())
    return _render_figure(fig)


def render_permutation_figures(
    spec: PermutationMetricSpec,
    position_ci: dict[str, PositionCI],
    components: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, bytes]:
    """The config-driven permutation plots (`PermutedCIPlots`, `UVPlots`) as
    `{figures/<key>: png}`, keyed as torch logs them under `slow_eval/`. Empty when neither
    plot metric is configured."""
    figures: dict[str, bytes] = {}
    if not spec.any_plots:
        return figures
    lower_png, upper_png = plot_permuted_ci_heatmaps(position_ci, spec.permutation)
    figures["figures/causal_importances"] = lower_png
    figures["figures/causal_importances_upper_leaky"] = upper_png
    if spec.want_uv_plots:
        present = {name: components[name] for name in spec.permutation}
        figures["figures/uv_matrices"] = plot_uv_matrices(present, spec.permutation, position_ci)
    return figures


def compute_identity_ci_errors(
    spec: PermutationMetricSpec, position_ci: dict[str, PositionCI], tolerance: float
) -> dict[str, float]:
    """The `IdentityCIError` discrete distances per configured site (torch
    `compute_target_metrics`), keyed `IdentityCIError/<site>` plus a summed
    `IdentityCIError` total. Empty when not configured. Operates on the batch-mean
    upper-leaky `(position, C)` CI matrix."""
    if not spec.any_identity_error:
        return {}
    per_site: dict[str, float] = {}
    for name, n_features in spec.identity_targets.items():
        matrix = position_ci[name].upper
        assert matrix.shape[1] >= n_features, (
            f"{name}: IdentityCIError expects >= {n_features} components, got {matrix.shape[1]}"
        )
        per_site[f"IdentityCIError/{name}"] = float(identity_ci_error(matrix, tolerance))
    for name, k in spec.dense_targets.items():
        per_site[f"IdentityCIError/{name}"] = float(
            dense_ci_error(position_ci[name].upper, k, tolerance)
        )
    per_site["IdentityCIError"] = float(sum(per_site.values()))
    return per_site


def render_slow_eval_figures(
    reductions: dict[str, SiteReduction],
) -> dict[str, bytes]:
    """The three slow plot metrics as `{log_key: png_bytes}`, keyed exactly as torch
    logs them under `slow_eval/` (`figures/<key>` from each metric's `compute()`)."""
    lower_hist = plot_ci_value_histograms({s: r.lower_sample for s, r in reductions.items()})
    logits_hist = plot_ci_value_histograms({s: r.logits_sample for s, r in reductions.items()})
    assert all(r.n_positions > 0 for r in reductions.values())
    densities = {s: r.density_counts / r.n_positions for s, r in reductions.items()}
    mean_cis = {s: r.ci_sums / r.n_positions for s, r in reductions.items()}
    density_fig = plot_component_activation_density(densities)
    mean_linear, mean_log = plot_mean_component_cis_both_scales(mean_cis)
    return {
        "figures/causal_importance_values": lower_hist,
        "figures/causal_importance_values_pre_sigmoid": logits_hist,
        "figures/component_activation_density": density_fig,
        "figures/ci_mean_per_component": mean_linear,
        "figures/ci_mean_per_component_log": mean_log,
    }


@dataclass(frozen=True)
class SlowEvalOutput:
    """The offline slow-eval payload: plot `figures` ({log_key: png}), scalar
    `hidden_acts` metrics ({torch_log_key: mse}), and scalar `identity_ci_errors`
    ({IdentityCIError[/<site>]: distance}, empty when unconfigured)."""

    figures: dict[str, bytes]
    hidden_acts: dict[str, float]
    identity_ci_errors: dict[str, float]


def _eval_config(cfg: ExperimentConfig) -> EvalConfig:
    assert cfg.eval is not None, f"{cfg.run_id}: no eval block — nothing to slow-eval"
    return cfg.eval


def compute_hidden_acts_metrics(
    lm: DecomposedModel,
    state: Any,
    frozen: Any,
    residual_batches: list[Float[Array, "*leading d"]],
    n_mask_samples: int,
    sampling: SamplingType,
    base_key: Array,
) -> dict[str, float]:
    """Both hidden-acts recon eval metrics over the eval batches, keyed by the torch
    `<ClassName>[/<site>]` log keys. `state.components`/`state.ci_fn` are the restored
    trajectory; `base_key` seeds the stochastic variant's per-batch draws."""
    ci_key, stoch_key = random.split(base_key)
    ci_step = make_ci_hidden_acts_step(lm)
    ci_reductions = accumulate_hidden_acts(
        ci_step, state.components, state.ci_fn, frozen, residual_batches, ci_key
    )
    stoch_step = make_stochastic_hidden_acts_step(lm, n_mask_samples, sampling)
    stoch_reductions = accumulate_hidden_acts(
        stoch_step, state.components, state.ci_fn, frozen, residual_batches, stoch_key
    )
    return {
        **hidden_acts_log_entries("CIHiddenActsReconLoss", ci_reductions),
        **hidden_acts_log_entries("StochasticHiddenActsReconLoss", stoch_reductions),
    }


def run_offline_slow_eval(run_dir: Path, cfg: ExperimentConfig, step: int) -> SlowEvalOutput:
    """Restore checkpoint `step` from `run_dir`'s ckpts, render the slow figures, and
    compute the scalar hidden-acts recon metrics. `run_dir` is the on-disk dir (the
    exporter takes it the same way); `cfg.run_dir` can differ when a run dir is read from
    a relocated copy. CPU-OK."""
    eval_cfg = _eval_config(cfg)
    mesh = dp_mesh()
    lm, frozen, prefix, prefix_residual_fn, _vocab_size = build_target(cfg, mesh)

    opt_vu, opt_ci, _schedules = build_optimizers(cfg)
    init_key, src_key, _run_key = random.split(random.PRNGKey(cfg.seed), 3)
    reference = init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, mesh)
    manager = make_checkpoint_manager(run_dir / "ckpts", cfg.cadence.keep_last)
    state = restore_step(manager, reference, step)

    data_cfg = cfg.data
    assert isinstance(data_cfg, DataConfig), "slow eval reads the LM parquet data path"
    schedule = BatchSchedule(scan_shards(data_cfg.dir), eval_cfg.batch_size, cfg.seed + 1)
    server = ShardServer(schedule, data_cfg.seq_len, jax.process_index(), jax.process_count())
    to_residual = jax.jit(prefix_residual_fn)
    residual_batches = [
        to_residual(prefix, jnp.asarray(server.local_batch(j))) for j in range(eval_cfg.n_steps)
    ]

    slow_eval_step = make_slow_eval_step(lm, eval_cfg.ci_alive_threshold)
    reductions = accumulate_site_reductions(
        slow_eval_step, state.ci_fn, frozen, residual_batches, _n_batches_accum(run_dir)
    )
    hidden_acts = compute_hidden_acts_metrics(
        lm, state, frozen, residual_batches, cfg.n_mask_samples, cfg.sampling,
        random.fold_in(random.PRNGKey(cfg.seed), step),
    )  # fmt: skip

    perm_spec = resolve_permutation_metrics(lm.site_names, _eval_metrics(run_dir))
    figures = render_slow_eval_figures(reductions)
    identity_ci_errors: dict[str, float] = {}
    if perm_spec.any_plots or perm_spec.any_identity_error:
        position_ci = accumulate_position_ci(
            make_position_ci_step(lm), state.ci_fn, frozen, residual_batches
        )
        components = {
            name: (np.asarray(V), np.asarray(U)) for name, (V, U) in state.components.vu.items()
        }
        figures |= render_permutation_figures(perm_spec, position_ci, components)
        identity_ci_errors = compute_identity_ci_errors(
            perm_spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE
        )
    return SlowEvalOutput(
        figures=figures, hidden_acts=hidden_acts, identity_ci_errors=identity_ci_errors
    )


def _eval_metrics(run_dir: Path) -> list[Any]:
    """The typed `eval.metrics` configs from the run's `config.yaml`. The trainer's
    `EvalConfig` keeps only scalar-tier fields, so the plot/permutation metric configs are
    re-validated here from the raw block (same source-of-truth read as `_n_batches_accum`)."""
    import yaml
    from pydantic import TypeAdapter

    from param_decomp_config.eval_metrics import AnyEvalMetricConfig

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    adapter = TypeAdapter(AnyEvalMetricConfig)
    return [adapter.validate_python(m) for m in raw["eval"]["metrics"]]


def _n_batches_accum(run_dir: Path) -> int | None:
    """The torch `CIHistograms.n_batches_accum` from the raw eval block (it's dropped by
    `EvalConfig`, which keeps only scalar-tier fields). None caps nothing."""
    import yaml

    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    for metric in raw["eval"]["metrics"]:
        if metric.get("type") == "CIHistograms":
            return metric.get("n_batches_accum")
    return None


def _write_output(output: SlowEvalOutput, out_dir: Path, step: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, png in output.figures.items():
        path = out_dir / f"{key.replace('/', '__')}_step{step}.png"
        path.write_bytes(png)
        print(f"wrote {path}", flush=True)
    scalars_path = out_dir / f"hidden_acts_recon_step{step}.json"
    scalars_path.write_text(json.dumps(output.hidden_acts, indent=2, sort_keys=True))
    print(f"wrote {scalars_path}", flush=True)
    for key in ("CIHiddenActsReconLoss", "StochasticHiddenActsReconLoss"):
        print(f"  {key} = {output.hidden_acts[key]:.6g}", flush=True)
    if output.identity_ci_errors:
        errors_path = out_dir / f"identity_ci_errors_step{step}.json"
        errors_path.write_text(json.dumps(output.identity_ci_errors, indent=2, sort_keys=True))
        print(f"wrote {errors_path}", flush=True)
        print(f"  IdentityCIError = {output.identity_ci_errors['IdentityCIError']:.6g}", flush=True)


def _log_to_wandb(cfg: ExperimentConfig, output: SlowEvalOutput, step: int) -> None:
    import wandb
    from PIL import Image

    assert cfg.wandb is not None, "no wandb config — pass --no-wandb to skip logging"
    wandb.init(
        id=cfg.run_id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow",
    )
    wandb.define_metric("slow_eval/step")
    wandb.define_metric("slow_eval/*", step_metric="slow_eval/step")
    payload: dict[str, Any] = {
        f"slow_eval/{k}": wandb.Image(Image.open(io.BytesIO(v))) for k, v in output.figures.items()
    }
    payload.update({f"slow_eval/loss/{k}": v for k, v in output.hidden_acts.items()})
    payload.update({f"slow_eval/{k}": v for k, v in output.identity_ci_errors.items()})
    payload["slow_eval/step"] = step
    wandb.log(payload)
    wandb.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--no-wandb", action="store_true", help="write PNGs to disk only")
    args = ap.parse_args()
    jax.config.update("jax_platforms", "cpu")

    cfg = load_run_dir_config(args.run_dir)
    manager = make_checkpoint_manager(args.run_dir / "ckpts", cfg.cadence.keep_last)
    step = args.step if args.step is not None else manager.latest_step()
    assert step is not None, f"no checkpoints under {args.run_dir / 'ckpts'}"

    output = run_offline_slow_eval(args.run_dir, cfg, step)
    _write_output(output, args.run_dir / "slow_eval", step)
    if not args.no_wandb:
        _log_to_wandb(cfg, output, step)


if __name__ == "__main__":
    main()
