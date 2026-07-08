"""Does the alive subspace carry everything the `=` read needs, or just the circuit?

With only the alive subcomponents enabled, the last-position output is close to the
target's — yet the hidden activations span a much smaller subspace than the original
model's. If the decomposition found the *causally relevant* subspace, then projecting the
original activations onto it while keeping the full target weights should also preserve
the output: the discarded orthogonal complement should not matter downstream. This
measures that, per prompt of each `1..100 × 1..100` operand grid, as
`KL(P_target || P_variant)` of the last-position next-token distribution for three
variants of the decomposed layer's MLP. Every intervention acts at the **last (`=`)
position only** — the subcomponent set is defined there — and leaves decomposed attention
and all earlier positions untouched:

- `projected_inputs`  — the MLP input (post-RMSNorm residual read) is projected onto the
  span of the set's gate_proj + up_proj `V` columns, then fed to the *unchanged* MLP.
- `projected_outputs` — the MLP output (the residual-stream write) is projected onto the
  span of the set's down_proj `U` rows.
- `alive_only` / `ci_only` — the MLP weight is replaced by the set's subcomponent sum:
  masks 1 on the set / 0 off it and the weight-delta off at the last position (all
  components + delta on at earlier positions, i.e. exact reconstruction there).

Two ways to define the subcomponent set:

- default — the static **alive** set from `alive_components.tsv`
  (`find_alive_components` output, filtered to the MLP matrices); one subspace shared by
  every prompt; the circuit variant is `alive_only`.
- `--ci-thr=X` — **per prompt**: the subcomponents whose lower-leaky CI (continuous
  sampling) at the last position exceeds `X` on that prompt. Each prompt gets its own
  subspaces (per-row SVD bases) and its own circuit mask; the circuit variant is
  `ci_only` and outputs land in `projection_kl_ci<X>/` instead of `projection_kl/`.

The reference is the raw target model.

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.collect_projection_kl <model_path> \
        [--ops=add,sub] [--batch-size=512] [--ci-thr=0.01] [--alive-tsv=PATH] \
        [--output-dir=PATH] [--output-fig-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Outputs (`<name>` = `projection_kl` or `projection_kl_ci<X>`):
- `<run>/analysis/datasets/<name>/data_<op>.npz` — per op: `kl` [3, N, N] (variant order
  in `variants`), `token` [3, N, N] argmax under each variant, `orig_token` / `orig_prob`
  [N, N]; with `--ci-thr` also `n_ci_in` / `n_ci_out` [N, N] (per-prompt set sizes).
- `<run>/analysis/datasets/<name>/summary.tsv` — one row per (op, variant):
  mean/median/q95/max KL + argmax agreement.
- `<run>/analysis/datasets/<name>/meta.json` — set definition, subspace sizes/ranks (or
  `ci_thr`), KL direction, token decode map.
- `<run>/analysis/<name>/kl_heatmaps_<op>.png` — the three KL(a, b) heatmaps on a shared
  log colour scale.
"""

import csv
import json
from pathlib import Path
from typing import Any, cast

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from torch import Tensor  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp.masks import make_mask_infos  # noqa: E402
from param_decomp.torch_helpers import bf16_autocast  # noqa: E402
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset  # noqa: E402
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    MLP_MATRICES,
    SlurmOptions,
    analysis_datasets_dir,
    analysis_dir,
    load_lm_run,
    op_prompts_file,
    parse_module_name,
    parse_operands,
    read_alive_components,
    square_grid_size,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.collect_projection_kl"
_SUMMARY_FIELDS = ["op", "variant", "mean_kl", "median_kl", "q95_kl", "max_kl", "argmax_agree"]


def _orthonormal_basis(vectors: Tensor) -> Tensor:
    """Orthonormal basis `Q [d, r]` of the span of the columns of `vectors [d, n]`."""
    left, sv, _ = torch.linalg.svd(vectors, full_matrices=False)
    keep = sv > sv[0] * 1e-5
    return left[:, keep].contiguous()


def _project_last(x: Tensor, basis: Tensor) -> Tensor:
    """Project the last position of `x [b, seq, d]` onto span(`basis [d, r]`)."""
    assert x.ndim == 3 and x.shape[-1] == basis.shape[0], f"{x.shape} vs basis {basis.shape}"
    x = x.clone()
    x[:, -1] = ((x[:, -1].float() @ basis) @ basis.T).to(x.dtype)
    return x


def _project_last_per_row(x: Tensor, bases: list[Tensor | None]) -> Tensor:
    """Project each row's last position onto that row's own basis (`None` = empty span)."""
    assert x.ndim == 3 and len(bases) == x.shape[0], f"{x.shape} vs {len(bases)} bases"
    x = x.clone()
    last = x[:, -1].float()
    for i, basis in enumerate(bases):
        last[i] = 0.0 if basis is None else (last[i] @ basis) @ basis.T
    x[:, -1] = last.to(x.dtype)
    return x


def _last_pos_kl_and_token(out: Tensor, logP: Tensor, P: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """`(kl [b], argmax_token [b])` of the last-position distribution vs the reference."""
    logQ = torch.log_softmax(out[:, -1].float(), dim=-1)
    kl = (P * (logP - logQ)).sum(dim=-1).cpu().numpy()
    return kl, logQ.argmax(dim=-1).cpu().numpy()


def _plot_heatmaps(
    kl: np.ndarray, variants: tuple[str, ...], n: int, op: str, run_id: str, fig_path: Path
) -> None:
    floor = 1e-6  # log colour floor: KL rounds to exactly 0 in float32 on easy prompts
    clipped = np.maximum(kl, floor)
    fig, axes = plt.subplots(1, len(variants), figsize=(4.8 * len(variants), 4.4))
    norm = LogNorm(vmin=floor, vmax=clipped.max())
    im = None
    for ax, variant, grid, raw in zip(axes, variants, clipped, kl, strict=True):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=(0.5, n + 0.5, 0.5, n + 0.5),
            norm=norm,
            cmap="viridis",
        )
        ax.set_title(f"{variant}\nmean KL {raw.mean():.4g}")
        ax.set_xlabel("b")
    axes[0].set_ylabel("a")
    assert im is not None
    fig.colorbar(im, ax=list(axes), label="KL(target ‖ variant) at `=`", fraction=0.03)
    fig.suptitle(f"{op}: last-position KL vs target — {run_id}")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_projection_kl(
    model_path: ModelPath,
    ops: str | tuple[str, ...] = "add,sub",
    batch_size: int = 512,
    ci_thr: float | None = None,
    alive_tsv: str | None = None,
    output_dir: str | None = None,
    output_fig_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> tuple[Path, Path] | None:
    """Write per-op KL grids + summary + heatmaps for the three subspace variants.

    Returns `(output_dir, fig_dir)`, or `None` when `--slurm` submits the job instead.
    """
    op_list = [o.strip() for o in (ops.split(",") if isinstance(ops, str) else ops)]
    assert ci_thr is None or alive_tsv is None, "--ci-thr replaces the alive TSV; pass only one"
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--ops={','.join(op_list)}",
            f"--batch-size={batch_size}",
        ]
        if ci_thr is not None:
            argv.append(f"--ci-thr={ci_thr}")
        for flag, val in [
            ("--alive-tsv", alive_tsv),
            ("--output-dir", output_dir),
            ("--output-fig-dir", output_fig_dir),
        ]:
            if val is not None:
                argv.append(f"{flag}={Path(val).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-collect-projection-kl")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    variants = (
        "projected_inputs",
        "projected_outputs",
        "alive_only" if ci_thr is None else "ci_only",
    )

    mlp_layers = {
        parse_module_name(m)[0]
        for m in model.components
        if parse_module_name(m)[1].startswith("mlp.")
    }
    assert len(mlp_layers) == 1, f"expected one decomposed MLP layer, got {sorted(mlp_layers)}"
    layer = mlp_layers.pop()
    mlp_name = f"model.layers.{layer}.mlp"
    modules = [f"{mlp_name}.{proj}" for proj in MLP_MATRICES]
    assert set(modules) <= set(model.components), (
        f"run decomposes {sorted(model.components)}, missing some of {modules}"
    )
    mlp_module = model.target_model.get_submodule(mlp_name)
    n_comp = {m: model.components[m].V.shape[1] for m in modules}
    weight_deltas = model.calc_weight_deltas()
    dtype = next(model.parameters()).dtype

    # Float32 copies of the set-defining vectors: V columns of the two read matrices
    # (input subspace), U rows of the write matrix (output subspace).
    v_read = {
        proj: model.components[f"{mlp_name}.{proj}"].V.detach().float()
        for proj in ("gate_proj", "up_proj")
    }
    u_write = model.components[f"{mlp_name}.down_proj"].U.detach().float()

    tsv: Path | None = None
    alive_idx: dict[str, list[int]] = {}
    q_in = q_out = None
    alive_vec: dict[str, Tensor] = {}
    row_bases: dict[str, list[Tensor | None]] = {"in": [], "out": []}
    if ci_thr is None:
        tsv = (
            Path(alive_tsv).expanduser()
            if alive_tsv
            else analysis_datasets_dir(run.run_dir) / "alive_components.tsv"
        )
        alive = read_alive_components(tsv, keep_projs=MLP_MATRICES)
        assert {a.layer for a in alive} == {layer}, f"alive TSV layer != decomposed layer {layer}"
        alive_idx = {
            proj: sorted({a.component for a in alive if a.proj == proj}) for proj in MLP_MATRICES
        }
        for proj in MLP_MATRICES:
            assert alive_idx[proj], f"no alive {proj} subcomponents in {tsv}"
        in_vecs = torch.cat(
            [v_read[proj][:, alive_idx[proj]] for proj in ("gate_proj", "up_proj")], dim=1
        )
        out_vecs = u_write[alive_idx["down_proj"]].T
        q_in, q_out = _orthonormal_basis(in_vecs), _orthonormal_basis(out_vecs)
        logger.info(
            f"input subspace: {in_vecs.shape[1]} V columns, rank {q_in.shape[1]} of "
            f"d={in_vecs.shape[0]}; output subspace: {out_vecs.shape[1]} U rows, "
            f"rank {q_out.shape[1]}"
        )
        for proj in MLP_MATRICES:
            m = f"{mlp_name}.{proj}"
            vec = torch.zeros(n_comp[m], device=device, dtype=dtype)
            vec[alive_idx[proj]] = 1.0
            alive_vec[m] = vec
    else:
        logger.info(f"per-prompt sets: lower-leaky CI > {ci_thr} at the last position")

    def project_input_hook(_mod: Any, args: tuple[Any, ...]) -> tuple[Tensor]:
        if ci_thr is None:
            assert q_in is not None
            return (_project_last(args[0], q_in),)
        return (_project_last_per_row(args[0], row_bases["in"]),)

    def project_output_hook(_mod: Any, _args: tuple[Any, ...], output: Any) -> Tensor:
        if ci_thr is None:
            assert q_out is not None
            return _project_last(output, q_out)
        return _project_last_per_row(output, row_bases["out"])

    name = "projection_kl" if ci_thr is None else f"projection_kl_ci{ci_thr:g}"
    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_datasets_dir(run.run_dir) / name
    )
    fig_dir = (
        Path(output_fig_dir).expanduser() if output_fig_dir else analysis_dir(run.run_dir) / name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    token_ids: set[int] = set()
    wiring_checked = False
    for op in op_list:
        prompts_path = op_prompts_file(op)
        prompt_texts = [ln.strip() for ln in prompts_path.read_text().splitlines() if ln.strip()]
        pool = load_prompts_dataset(str(prompts_path), cast(Any, tokenizer)).to(device)
        assert pool.shape[0] == len(prompt_texts)
        seq = pool.shape[1]
        ab = [parse_operands(t, op) for t in prompt_texts]
        n = square_grid_size(ab)

        kl = np.zeros((len(variants), n, n), np.float32)
        var_token = np.zeros((len(variants), n, n), np.int32)
        orig_token = np.zeros((n, n), np.int32)
        orig_prob = np.zeros((n, n), np.float32)
        n_ci_in = np.zeros((n, n), np.int32)
        n_ci_out = np.zeros((n, n), np.int32)

        with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
            for start in range(0, pool.shape[0], batch_size):
                chunk = pool[start : start + batch_size]
                b = chunk.shape[0]
                grid_rows = [a_ - 1 for a_, _ in ab[start : start + b]]
                grid_cols = [b_ - 1 for _, b_ in ab[start : start + b]]

                # Reference forward; in CI mode it also feeds the CI fn (which needs every
                # decomposed module's input) to define this chunk's per-prompt sets.
                sup: dict[str, Tensor] = {}
                if ci_thr is None:
                    ref_out = model(chunk)
                else:
                    cached = model(chunk, cache_type="input")
                    ref_out = cached.output
                    ci = model.calc_causal_importances(cached.cache, sampling="continuous")
                    sup = {m: ci.lower_leaky[m][:, -1] > ci_thr for m in modules}  # [b, C] bool
                    row_bases["in"] = []
                    row_bases["out"] = []
                    for i in range(b):
                        cols = torch.cat(
                            [
                                v_read[proj][:, sup[f"{mlp_name}.{proj}"][i]]
                                for proj in ("gate_proj", "up_proj")
                            ],
                            dim=1,
                        )
                        rows_u = u_write[sup[f"{mlp_name}.down_proj"][i]]
                        row_bases["in"].append(_orthonormal_basis(cols) if cols.shape[1] else None)
                        row_bases["out"].append(
                            _orthonormal_basis(rows_u.T) if rows_u.shape[0] else None
                        )
                    n_ci_in[grid_rows, grid_cols] = (
                        (sup[f"{mlp_name}.gate_proj"].sum(1) + sup[f"{mlp_name}.up_proj"].sum(1))
                        .cpu()
                        .numpy()
                    )
                    n_ci_out[grid_rows, grid_cols] = (
                        sup[f"{mlp_name}.down_proj"].sum(1).cpu().numpy()
                    )

                logP = torch.log_softmax(ref_out[:, -1].float(), dim=-1)  # [b, vocab]
                P = logP.exp()
                orig_token[grid_rows, grid_cols] = logP.argmax(dim=-1).cpu().numpy()
                orig_prob[grid_rows, grid_cols] = logP.max(dim=-1).values.exp().cpu().numpy()

                if not wiring_checked:
                    # All-on + delta-on for the MLP modules must reproduce the raw target;
                    # otherwise the mask/delta wiring is wrong and every circuit KL would
                    # be measured against a bad reference.
                    full = {
                        m: torch.ones(b, seq, n_comp[m], device=device, dtype=dtype)
                        for m in modules
                    }
                    ones_delta = torch.ones(b, seq, device=device, dtype=dtype)
                    infos = make_mask_infos(
                        full,
                        weight_deltas_and_masks={
                            m: (weight_deltas[m], ones_delta) for m in modules
                        },
                    )
                    rec, _ = _last_pos_kl_and_token(model(chunk, mask_infos=infos), logP, P)
                    logger.info(f"reconstruction check: mean KL(target||all_on)={rec.mean():.4g}")
                    assert rec.mean() < 0.5, (
                        f"all-on+delta != target (KL {rec.mean()}); mask/delta wiring bug"
                    )
                    wiring_checked = True

                handle = mlp_module.register_forward_pre_hook(project_input_hook)
                try:
                    kl_pi, tok_pi = _last_pos_kl_and_token(model(chunk), logP, P)
                finally:
                    handle.remove()

                handle = mlp_module.register_forward_hook(project_output_hook)
                try:
                    kl_po, tok_po = _last_pos_kl_and_token(model(chunk), logP, P)
                finally:
                    handle.remove()

                # Circuit: set masks on / rest off / delta off at the last position;
                # everything on + delta at earlier positions (exact reconstruction there).
                masks = {}
                for m in modules:
                    full_mask = torch.ones(b, seq, n_comp[m], device=device, dtype=dtype)
                    full_mask[:, -1] = alive_vec[m] if ci_thr is None else sup[m].to(dtype)
                    masks[m] = full_mask
                delta_mask = torch.ones(b, seq, device=device, dtype=dtype)
                delta_mask[:, -1] = 0.0
                infos = make_mask_infos(
                    masks,
                    weight_deltas_and_masks={m: (weight_deltas[m], delta_mask) for m in modules},
                )
                kl_ci, tok_ci = _last_pos_kl_and_token(model(chunk, mask_infos=infos), logP, P)

                for vi, (kl_v, tok_v) in enumerate(
                    ((kl_pi, tok_pi), (kl_po, tok_po), (kl_ci, tok_ci))
                ):
                    kl[vi, grid_rows, grid_cols] = kl_v
                    var_token[vi, grid_rows, grid_cols] = tok_v

                logger.info(f"{op}: prompts {start}..{start + b} / {pool.shape[0]}")

        arrays: dict[str, Any] = {
            "kl": kl,
            "token": var_token,
            "orig_token": orig_token,
            "orig_prob": orig_prob,
            "variants": np.array(variants),
        }
        if ci_thr is not None:
            arrays |= {"n_ci_in": n_ci_in, "n_ci_out": n_ci_out}
        np.savez_compressed(out_dir / f"data_{op}.npz", **arrays)
        for vi, variant in enumerate(variants):
            summary_rows.append(
                {
                    "op": op,
                    "variant": variant,
                    "mean_kl": round(float(kl[vi].mean()), 6),
                    "median_kl": round(float(np.median(kl[vi])), 6),
                    "q95_kl": round(float(np.percentile(kl[vi], 95)), 6),
                    "max_kl": round(float(kl[vi].max()), 6),
                    "argmax_agree": round(float((var_token[vi] == orig_token).mean()), 4),
                }
            )
        if ci_thr is not None:
            logger.info(
                f"{op}: per-prompt set sizes — input mean {n_ci_in.mean():.1f} "
                f"(max {n_ci_in.max()}), output mean {n_ci_out.mean():.1f} (max {n_ci_out.max()})"
            )
        token_ids |= {int(t) for t in np.unique(orig_token)}
        token_ids |= {int(t) for t in np.unique(var_token)}
        _plot_heatmaps(kl, variants, n, op, run.run_dir.name, fig_dir / f"kl_heatmaps_{op}.png")

    with (out_dir / "summary.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    meta: dict[str, Any] = {
        "run_id": run.run_dir.name,
        "layer": layer,
        "ops": op_list,
        "variants": list(variants),
        "kl_direction": "KL(P_target || P_variant)",
        "position": "last (= answer) only; earlier positions and attention untouched",
        "circuit": f"{variants[2]}: set masks on, rest off, delta off at last position; "
        "all components + delta on elsewhere",
        "token_decode": {t: tokenizer.decode([t]) for t in sorted(token_ids)},
    }
    if ci_thr is None:
        assert tsv is not None and q_in is not None and q_out is not None
        meta |= {
            "alive_tsv": str(tsv),
            "n_alive": {proj: len(alive_idx[proj]) for proj in MLP_MATRICES},
            "input_subspace": {
                "n_vectors": sum(len(alive_idx[p]) for p in ("gate_proj", "up_proj")),
                "rank": int(q_in.shape[1]),
            },
            "output_subspace": {
                "n_vectors": len(alive_idx["down_proj"]),
                "rank": int(q_out.shape[1]),
            },
        }
    else:
        meta |= {"ci_thr": ci_thr, "ci": "lower-leaky, continuous sampling, last position only"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for row in summary_rows:
        logger.info(
            f"{row['op']}/{row['variant']}: mean KL {row['mean_kl']}, "
            f"argmax agree {row['argmax_agree']}"
        )
    logger.info(f"wrote grids + summary to {out_dir}, heatmaps to {fig_dir}")
    return out_dir, fig_dir


if __name__ == "__main__":
    fire.Fire(collect_projection_kl)
