"""Subspace-filtering battery for a TMS-5-2 decomposition — per-sample MSE vs the target.

Same question as `collect_filtered_kl_fulldata.py` (do the selected subcomponents'
subspaces match the original matrices' causally-used subspaces?) for a TMS toy
decomposition. Differences from the LM batteries:

- Model: `SavedTMSRun` — `out = ReLU(linear2(linear1(x)))`, activations are `[b, d]`
  (no sequence dim), and `linear2` carries a frozen target bias which the circuit hook
  re-adds (asserted, not assumed).
- Data: one batch of `--n-samples` draws from the run's own `SparseFeatureDataset`
  config (seeded).
- Readout: per-sample MSE between the variant's output and the *original* model's
  output (not the data labels): `((out_var - out_orig) ** 2).mean(-1)`.
- Selected sets are per-sample: subcomponents with lower-leaky CI > `--ci-thr`.
- `mu` for centered/bias flavors: mean over the batch per site, under the model variant
  being intervened on (original for `orig_*`; an actual circuit forward for `circuit_*`).

CPU-only (the model is tiny) — no `--slurm`.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_filtering.collect_filtered_mse_tms \
        <model_path> [--n-samples=4096] [--ci-thr=0.01] [--seed=0] [--output-dir=PATH]

Outputs (default `<run>/analysis/datasets/subspace_filtering/`):
- `mse_tms.tsv` — long format: experiment, flavor, sample, mse.
- `nci_tms.tsv` — per-sample active counts: sample, n_ci_linear1, n_ci_linear2.
- `meta_tms.json` — span-rank stats vs d per site, matrix-space ranks, wiring check.
- `<run>/analysis/subspace_filtering/boxplot_tms.png` — summary boxplot (log-y).
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import fire
import matplotlib
import numpy as np
import torch
from torch import Tensor, nn

from param_decomp.log import logger
from param_decomp_lab.experiments.tms.run import SavedTMSRun, build_tms_loader
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import analysis_datasets_dir, analysis_dir
from param_decomp_lab.scripts.validation.subspace_filtering.lib import (
    FLAVORS,
    apply_flavor,
    orthonormal_basis,
    project_onto,
    project_rows,
)
from param_decomp_lab.seed import set_seed

_BASELINE_KEY = "circuit_baseline"

Side = Literal["in", "out"]


@dataclass(frozen=True)
class _Intervention:
    key: str
    circuit: bool
    module: str
    side: Side
    space: Literal["span", "matrix"]


def _pre_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward pre-hook applying `transform` to the whole `[b, d]` module input."""

    def hook(_mod: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (transform(args[0].float()), *args[1:])

    return hook


def _out_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward hook applying `transform` to the whole `[b, d]` module output."""

    def hook(_mod: Any, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        return transform(output.float())

    return hook


def _circuit_hook(v: Tensor, u: Tensor, mask: Tensor, bias: Tensor | None) -> Callable[..., Any]:
    """Replace the module's `[b, d]` output with the masked subcomponent sum (+ frozen bias).

    `v [d_in, C]`, `u [C, d_out]` float32; `mask [b, C]`; `bias [d_out]` or `None`. The
    delta component stays off.
    """

    def hook(_mod: Any, args: tuple[Any, ...], _output: Tensor) -> Tensor:
        acts = (args[0].float() @ v) * mask
        out = acts @ u
        return out if bias is None else out + bias

    return hook


def _sample_bases(vecs: Tensor, support: np.ndarray) -> tuple[list[Tensor | None], np.ndarray]:
    """Per-sample span bases from `vecs [d, C]` and `support [b, C]`; empty set = zero span."""
    bases: list[Tensor | None] = []
    for i in range(support.shape[0]):
        idx = np.flatnonzero(support[i])
        bases.append(orthonormal_basis(vecs[:, torch.from_numpy(idx)]) if len(idx) else None)
    ranks = np.array([0 if q is None else q.shape[1] for q in bases], np.int32)
    return bases, ranks


def collect_filtered_mse_tms(
    model_path: ModelPath,
    n_samples: int = 4096,
    ci_thr: float = 0.01,
    seed: int = 0,
    output_dir: str | None = None,
) -> Path:
    """Write the long-format per-sample MSE TSV + counts + meta + boxplot; returns the dataset dir."""
    run = SavedTMSRun.from_path(model_path)
    model = run.load_model()
    model.eval()
    run_dir = run.checkpoint_path.parent

    modules = sorted(model.components)
    assert modules == ["linear1", "linear2"], f"expected TMS linears, got {modules}"
    linears = {m: model.target_model.get_submodule(m) for m in modules}
    for m in modules:
        assert isinstance(linears[m], nn.Linear), f"{m} is not nn.Linear"
    assert linears["linear1"].bias is None, "linear1 unexpectedly has a bias"
    assert linears["linear2"].bias is not None, "linear2 unexpectedly has no bias"
    biases = {m: cast(Tensor | None, linears[m].bias) for m in modules}
    biases = {m: None if b is None else b.detach().float() for m, b in biases.items()}
    w = {m: cast(Tensor, linears[m].weight).detach().float() for m in modules}  # [d_out, d_in]
    v = {m: model.components[m].V.detach().float() for m in modules}
    u = {m: model.components[m].U.detach().float() for m in modules}
    c_per_module = {m: int(v[m].shape[1]) for m in modules}

    interventions = [_Intervention(f"orig_in_span:{m}", False, m, "in", "span") for m in modules]
    interventions += [_Intervention(f"orig_out_span:{m}", False, m, "out", "span") for m in modules]
    interventions += [
        _Intervention(f"circuit_in_row:{m}", True, m, "in", "matrix")
        for m in modules
        if w[m].shape[0] < w[m].shape[1]
    ]
    interventions += [
        _Intervention(f"circuit_out_col:{m}", True, m, "out", "matrix")
        for m in modules
        if w[m].shape[1] < w[m].shape[0]
    ]
    matrix_keys = [iv.key for iv in interventions if iv.space == "matrix"]
    assert matrix_keys == ["circuit_in_row:linear1", "circuit_out_col:linear2"], matrix_keys

    matrix_q = {
        (iv.module, iv.side): orthonormal_basis(w[iv.module].T if iv.side == "in" else w[iv.module])
        for iv in interventions
        if iv.space == "matrix"
    }

    set_seed(seed)
    loader = build_tms_loader(
        run.cfg.target, run.cfg.data, split="eval", device="cpu", batch_size=n_samples
    )
    batch = next(iter(loader))
    x = batch[0]
    assert x.shape == (n_samples, w["linear1"].shape[1]), f"got {tuple(x.shape)}"

    with torch.no_grad():
        cached = model(batch, cache_type="input")
        ci = model.calc_causal_importances(cached.cache, sampling="continuous")
        supports = {}
        for m in modules:
            ci_m = ci.lower_leaky[m]
            assert ci_m.shape == (n_samples, c_per_module[m]), f"{m}: {tuple(ci_m.shape)}"
            supports[m] = (ci_m > ci_thr).cpu().numpy()
        masks = {m: torch.from_numpy(supports[m]).float() for m in modules}
        circuit_hooks = {m: _circuit_hook(v[m], u[m], masks[m], biases[m]) for m in modules}

        def run_forward(hooks: list[tuple[nn.Module, str, Callable[..., Any]]]) -> Tensor:
            handles = []
            try:
                for module, kind, hook in hooks:
                    reg = (
                        module.register_forward_pre_hook
                        if kind == "pre"
                        else module.register_forward_hook
                    )
                    handles.append(reg(hook))
                out = model.target_model(x)
                assert out.shape == x.shape
                return out.float()
            finally:
                for h in handles:
                    h.remove()

        def variant_hooks(
            iv: _Intervention, transform: Callable[[Tensor], Tensor]
        ) -> list[tuple[nn.Module, str, Callable[..., Any]]]:
            hooks: list[tuple[nn.Module, str, Callable[..., Any]]] = []
            if iv.circuit:
                hooks += [(linears[m], "out", circuit_hooks[m]) for m in modules]
            kind = "pre" if iv.side == "in" else "out"
            hook = _pre_hook(transform) if iv.side == "in" else _out_hook(transform)
            hooks.append((linears[iv.module], kind, hook))
            return hooks

        out_orig = run_forward([])

        # Per-site batch means under each intervened model variant (orig and circuit).
        mu: dict[tuple[str, str, Side], Tensor] = {}
        for kind, circuit in (("orig", False), ("circuit", True)):
            site_cache: dict[tuple[str, Side], Tensor] = {}

            def cache_in(m: str, hits: dict[tuple[str, Side], Tensor] = site_cache) -> Any:
                def hook(_mod: Any, args: tuple[Any, ...]) -> None:
                    hits[(m, "in")] = args[0]

                return hook

            def cache_out(m: str, hits: dict[tuple[str, Side], Tensor] = site_cache) -> Any:
                def hook(_mod: Any, _args: tuple[Any, ...], output: Tensor) -> None:
                    hits[(m, "out")] = output

                return hook

            hooks = [(linears[m], "out", circuit_hooks[m]) for m in modules] if circuit else []
            # Out-caching hooks registered after the circuit hooks so they see the
            # replaced output; pre-hooks see the (upstream-)circuited input.
            for m in modules:
                hooks.append((linears[m], "pre", cache_in(m)))
                hooks.append((linears[m], "out", cache_out(m)))
            run_forward(hooks)
            for (m, side), acts in site_cache.items():
                mu[(kind, m, side)] = acts.float().mean(dim=0)

        # Wiring check: original weights + raw row-space projection must be a no-op.
        q_chk = matrix_q[("linear1", "in")]
        chk_out = run_forward(
            [(linears["linear1"], "pre", _pre_hook(lambda t, q=q_chk: project_onto(t, q)))]
        )
        wiring_mse = float(((chk_out - out_orig) ** 2).mean().item())
        logger.info(f"wiring check (orig + raw row-space proj on linear1 input): {wiring_mse:.3g}")
        assert wiring_mse < 1e-6, f"row-space no-op check failed: MSE {wiring_mse}"

        exp_rows: list[tuple[str, str]] = [
            (iv.key, flavor) for iv in interventions for flavor in FLAVORS
        ]
        exp_rows.append((_BASELINE_KEY, "none"))
        mse = np.zeros((len(exp_rows), n_samples), np.float32)
        row_of = {key: i for i, key in enumerate(exp_rows)}

        def per_sample_mse(out: Tensor) -> np.ndarray:
            return ((out - out_orig) ** 2).mean(dim=-1).cpu().numpy()

        mse[row_of[(_BASELINE_KEY, "none")]] = per_sample_mse(
            run_forward([(linears[m], "out", circuit_hooks[m]) for m in modules])
        )

        rank_stats: dict[str, dict[str, float | int]] = {}
        for iv in interventions:
            if iv.space == "span":
                vecs = v[iv.module] if iv.side == "in" else u[iv.module].T
                bases, ranks = _sample_bases(vecs, supports[iv.module])
                rank_stats[f"{iv.module}.{iv.side}"] = {
                    "d": int(vecs.shape[0]),
                    "rank_mean": float(ranks.mean()),
                    "rank_min": int(ranks.min()),
                    "rank_max": int(ranks.max()),
                }
                proj_fn: Callable[[Tensor], Tensor] = lambda t, b=bases: project_rows(t, b)
            else:
                q_fixed = matrix_q[(iv.module, iv.side)]
                proj_fn = lambda t, q=q_fixed: project_onto(t, q)
            kind = "circuit" if iv.circuit else "orig"
            mu_site = mu[(kind, iv.module, iv.side)]
            for flavor in FLAVORS:
                transform = lambda t, f=flavor, m_=mu_site, p=proj_fn: apply_flavor(t, m_, f, p)
                mse[row_of[(iv.key, flavor)]] = per_sample_mse(
                    run_forward(variant_hooks(iv, transform))
                )
            logger.info(f"done {iv.key}")

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_datasets_dir(run_dir) / "subspace_filtering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mse_path = out_dir / "mse_tms.tsv"
    with mse_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["experiment", "flavor", "sample", "mse"])
        for ei, (exp_name, flavor) in enumerate(exp_rows):
            for si in range(n_samples):
                writer.writerow([exp_name, flavor, si, f"{mse[ei, si]:.6g}"])
    logger.info(f"wrote {mse_path}")

    nci_path = out_dir / "nci_tms.tsv"
    with nci_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample", "n_ci_linear1", "n_ci_linear2"])
        n1, n2 = supports["linear1"].sum(axis=1), supports["linear2"].sum(axis=1)
        for si in range(n_samples):
            writer.writerow([si, int(n1[si]), int(n2[si])])

    meta = {
        "run_id": run_dir.name,
        "n_samples": n_samples,
        "seed": seed,
        "ci_thr": ci_thr,
        "C": c_per_module,
        "set_scope": "per sample: lower-leaky CI > ci_thr",
        "mu": "mean over the n_samples batch, per site, under the intervened model variant",
        "span_rank_stats": rank_stats,
        "matrix_space_rank": {f"{m}.{s}": int(q.shape[1]) for (m, s), q in matrix_q.items()},
        "wiring_check_mse": wiring_mse,
        "experiments": [iv.key for iv in interventions] + [_BASELINE_KEY],
        "flavors": list(FLAVORS),
        "readout": "per-sample MSE vs the ORIGINAL model output (not data labels)",
    }
    meta_path = out_dir / "meta_tms.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"wrote {meta_path}")

    fig_dir = analysis_dir(run_dir) / "subspace_filtering"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _write_boxplot(mse, [iv.key for iv in interventions], row_of, fig_dir / "boxplot_tms.png")
    logger.info(f"done ({n_samples} samples)")
    return out_dir


def _write_boxplot(
    mse: np.ndarray,
    intervention_keys: list[str],
    row_of: dict[tuple[str, str], int],
    path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = 1e-12
    flierprops = dict(marker=".", markersize=2, alpha=0.25)
    fig, ax = plt.subplots(figsize=(1.8 * len(intervention_keys) + 2, 5))
    colors = {"raw": "tab:red", "centered": "tab:blue", "bias": "tab:green"}
    for gi, key in enumerate(intervention_keys):
        for fi, flavor in enumerate(FLAVORS):
            vals = np.maximum(mse[row_of[(key, flavor)]], floor)
            box = ax.boxplot(
                [vals],
                positions=[gi + (fi - 1) * 0.25],
                widths=0.2,
                whis=(5, 95),
                flierprops=flierprops,
                patch_artist=True,
            )
            box["boxes"][0].set_facecolor(colors[flavor])
    base_vals = np.maximum(mse[row_of[(_BASELINE_KEY, "none")]], floor)
    box = ax.boxplot(
        [base_vals],
        positions=[len(intervention_keys)],
        widths=0.2,
        whis=(5, 95),
        flierprops=flierprops,
        patch_artist=True,
    )
    box["boxes"][0].set_facecolor("tab:gray")
    ax.set_yscale("log")
    ax.set_xticks(list(range(len(intervention_keys) + 1)))
    ax.set_xticklabels([*intervention_keys, _BASELINE_KEY], rotation=20, ha="right")
    ax.set_ylabel("per-sample MSE vs original output")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for c in colors.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="tab:gray"))
    ax.legend(handles, [*colors.keys(), "baseline"], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    fire.Fire(collect_filtered_mse_tms)
