"""Subspace-filtering battery for one decomposed block — implements `plan.md`.

For the L18 MLP or attention block, measures last-position `KL(P_target || P_variant)`
over the `1..100 × 1..100` operand grids under activation projections that test whether
the selected subcomponents' subspaces match the original matrices' causally-used
subspaces. Two selected-set modes: `--subset=alive` (static, `alive_subcomponents.tsv`)
and `--subset=active` (per prompt: lower-leaky CI > `--ci-thr` at `=`).

Experiments (each site is its own intervention; everything at the `=` position only):

- `orig_in_span:<m>`  — original weights, matrix `m`'s input projected onto the span of
  the selected `V` columns of `m`.
- `orig_out_span:<m>` — original weights, `m`'s output projected onto the span of the
  selected `U` rows of `m`.
- `circuit_in_row:<m>`  — circuit (selected subcomponents only, delta off, block
  replaced at `=`), `m`'s input projected onto `row(W_m)`; only where `d_out < d_in`.
- `circuit_out_col:<m>` — circuit, `m`'s output projected onto `col(W_m)`; only where
  `d_in < d_out`.
- `circuit_baseline` — circuit, no projection.

Every projection runs in three flavors, using the per-op grid mean `μ` of that site's
activation at `=` *under the model variant being intervened on* (original for exp 1,
circuit for exp 2): `raw` `P x`, `centered` `μ + P(x−μ)`, `bias` `x − (I−P)μ`. An empty
selected set projects to the zero span. The circuit is implemented with plain forward
hooks (`((x·V)⊙mask)·U` at `=`, original weights elsewhere — identical to all-on+delta),
so output projections chain after the replacement; the delta component is never used.
Projection/circuit math runs in float32 with autocast disabled inside the hooks. A
first-batch wiring check asserts that projecting an input onto `row(W)` while running
the original `W` is a no-op.

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_filtering.collect_filtered_kl \
        <model_path> <block:mlp|attn> [--subset=alive|active] [--ci-thr=0.01] \
        [--ops=add,sub] [--batch-size=512] [--alive-tsv=PATH] [--output-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=4:00:00 --slurm-mem=...]]

Outputs (default `<run>/analysis/datasets/subspace_filtering/`):
- `kl_<block>_<subset>_<op>.tsv` — long format: experiment, flavor, a, b, kl.
- `nci_<block>_<op>.tsv` (active only) — a, b, per-matrix selected count + total.
- `meta_<block>_<subset>.json` — selected counts, span/matrix-space ranks, per-site
  `‖μ‖` and `‖(I−P)μ‖/‖μ‖` diagnostics, wiring-check value, KL direction.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    analysis_datasets_dir,
    load_lm_run,
    op_prompts_file,
    parse_module_name,
    parse_operands,
    read_alive_components,
    square_grid_size,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.subspace_filtering.lib import (
    FLAVORS,
    apply_flavor,
    circuit_hook,
    orthonormal_basis,
    project_onto,
    project_rows,
    projection_out_hook,
    projection_pre_hook,
)

_MODULE = "param_decomp_lab.scripts.validation.subspace_filtering.collect_filtered_kl"
_BLOCK_MATRICES = {
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}
_BLOCK_PARENT = {"mlp": "mlp", "attn": "self_attn"}
_BASELINE_KEY = "circuit_baseline"

Side = Literal["in", "out"]
MeanKey = tuple[str, str, Side]  # (kind: orig|circuit, matrix, side)


@dataclass(frozen=True)
class _Intervention:
    key: str
    circuit: bool
    matrix: str
    side: Side
    space: Literal["span", "matrix"]


def _build_interventions(mats: tuple[str, ...], w: dict[str, Tensor]) -> list[_Intervention]:
    ivs = [_Intervention(f"orig_in_span:{m}", False, m, "in", "span") for m in mats]
    ivs += [_Intervention(f"orig_out_span:{m}", False, m, "out", "span") for m in mats]
    ivs += [
        _Intervention(f"circuit_in_row:{m}", True, m, "in", "matrix")
        for m in mats
        if w[m].shape[0] < w[m].shape[1]
    ]
    ivs += [
        _Intervention(f"circuit_out_col:{m}", True, m, "out", "matrix")
        for m in mats
        if w[m].shape[1] < w[m].shape[0]
    ]
    return ivs


def _last_pos_kl(out: Tensor, logP: Tensor, P: Tensor) -> np.ndarray:
    logQ = torch.log_softmax(out[:, -1].float(), dim=-1)
    return (P * (logP - logQ)).sum(dim=-1).cpu().numpy()


def collect_filtered_kl(
    model_path: ModelPath,
    block: str,
    subset: str = "alive",
    ci_thr: float = 0.01,
    ops: str | tuple[str, ...] = "add,sub",
    batch_size: int = 512,
    alive_tsv: str | None = None,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "4:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Write the per-op long-format KL TSVs + meta for one (block, subset) pair.

    Returns the output dir, or `None` when `--slurm` submits the job instead.
    """
    assert block in _BLOCK_MATRICES, f"block must be one of {list(_BLOCK_MATRICES)}"
    assert subset in ("alive", "active"), "subset must be 'alive' or 'active'"
    op_list = [o.strip() for o in (ops.split(",") if isinstance(ops, str) else ops)]
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            block,
            f"--subset={subset}",
            f"--ci-thr={ci_thr}",
            f"--ops={','.join(op_list)}",
            f"--batch-size={batch_size}",
        ]
        for flag, val in [("--alive-tsv", alive_tsv), ("--output-dir", output_dir)]:
            if val is not None:
                argv.append(f"{flag}={Path(val).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-subfilt-{block}-{subset}")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    parent, mats = _BLOCK_PARENT[block], _BLOCK_MATRICES[block]

    layers = {
        parse_module_name(m)[0]
        for m in model.components
        if parse_module_name(m)[1].startswith(f"{parent}.")
    }
    assert len(layers) == 1, f"expected one decomposed {parent} layer, got {sorted(layers)}"
    layer = layers.pop()
    full = {m: f"model.layers.{layer}.{parent}.{m}" for m in mats}
    assert set(full.values()) <= set(model.components), (
        f"run decomposes {sorted(model.components)}, missing some of {sorted(full.values())}"
    )
    linears = {m: model.target_model.get_submodule(full[m]) for m in mats}
    w = {m: cast(Tensor, linears[m].weight).detach().float() for m in mats}  # [d_out, d_in]
    v = {m: model.components[full[m]].V.detach().float() for m in mats}  # [d_in, C]
    u = {m: model.components[full[m]].U.detach().float() for m in mats}  # [C, d_out]
    n_comp = {m: v[m].shape[1] for m in mats}

    interventions = _build_interventions(mats, w)
    experiments = [(iv, flavor) for iv in interventions for flavor in FLAVORS]
    exp_keys = [f"{iv.key}|{flavor}" for iv, flavor in experiments] + [_BASELINE_KEY]

    # Static selected sets (alive mode) and their span bases.
    sel: dict[str, list[int]] = {}
    span_q: dict[tuple[str, Side], Tensor | None] = {}
    tsv: Path | None = None
    if subset == "alive":
        tsv = (
            Path(alive_tsv).expanduser()
            if alive_tsv
            else analysis_datasets_dir(run.run_dir) / "alive_subcomponents.tsv"
        )
        alive = read_alive_components(tsv, keep_projs=None)
        sel = {
            m: sorted({a.component for a in alive if a.proj == m and a.layer == layer})
            for m in mats
        }
        for m in mats:
            if not sel[m]:
                logger.info(f"NOTE: no alive subcomponents for {m} — empty span (projects to 0)")
            span_q[(m, "in")] = orthonormal_basis(v[m][:, sel[m]]) if sel[m] else None
            span_q[(m, "out")] = orthonormal_basis(u[m][sel[m]].T) if sel[m] else None
        static_mask = {}
        for m in mats:
            mask = torch.zeros(n_comp[m], device=device)
            if sel[m]:
                mask[torch.tensor(sel[m], device=device)] = 1.0
            static_mask[m] = mask
    else:
        static_mask = {}
        logger.info(f"per-prompt selected sets: lower-leaky CI > {ci_thr} at the last position")

    # Matrix-space bases for exp 2 (independent of the subset).
    matrix_q: dict[tuple[str, Side], Tensor] = {}
    for iv in interventions:
        if iv.space != "matrix":
            continue
        vectors = w[iv.matrix].T if iv.side == "in" else w[iv.matrix]
        matrix_q[(iv.matrix, iv.side)] = orthonormal_basis(vectors)
    logger.info(
        f"{block}: {len(interventions)} interventions, matrix-space ranks "
        f"{ {f'{m}.{s}': int(q.shape[1]) for (m, s), q in matrix_q.items()} }"
    )

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_datasets_dir(run.run_dir) / "subspace_filtering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_ops: dict[str, Any] = {}
    wiring_kl: float | None = None
    for op in op_list:
        prompts_path = op_prompts_file(op)
        prompt_texts = [ln.strip() for ln in prompts_path.read_text().splitlines() if ln.strip()]
        pool = load_prompts_dataset(str(prompts_path), cast(Any, tokenizer)).to(device)
        assert pool.shape[0] == len(prompt_texts)
        n_prompts = pool.shape[0]
        ab = [parse_operands(t, op) for t in prompt_texts]
        n = square_grid_size(ab)

        mean_keys: set[MeanKey] = {
            ("circuit" if iv.circuit else "orig", iv.matrix, iv.side) for iv in interventions
        }
        sums: dict[MeanKey, Tensor] = {}
        supports: dict[str, np.ndarray] = {
            m: np.zeros((n_prompts, n_comp[m]), dtype=bool) for m in mats
        }

        # ---- Pass A: per-site means at `=` (+ per-prompt CI supports in active mode).
        with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
            for start in range(0, n_prompts, batch_size):
                chunk = pool[start : start + batch_size]
                b = chunk.shape[0]
                cached = model(chunk, cache_type="input")
                x_in = {m: cached.cache[full[m]][:, -1].float() for m in mats}

                if subset == "active":
                    ci = model.calc_causal_importances(cached.cache, sampling="continuous")
                    for m in mats:
                        supports[m][start : start + b] = (
                            (ci.lower_leaky[full[m]][:, -1] > ci_thr).cpu().numpy()
                        )
                    mask_b = {
                        m: torch.from_numpy(supports[m][start : start + b]).to(device).float()
                        for m in mats
                    }
                else:
                    mask_b = {m: static_mask[m] for m in mats}

                need_h = block == "mlp" and ("circuit", "down_proj", "in") in mean_keys
                with torch.autocast("cuda", enabled=False):
                    circuit_out = {
                        m: ((x_in[m] @ v[m]) * mask_b[m]) @ u[m]
                        for m in mats
                        if ("circuit", m, "out") in mean_keys
                        or (need_h and m in ("gate_proj", "up_proj"))
                    }
                    for key in mean_keys:
                        kind, m, side = key
                        if kind == "orig":
                            val = x_in[m] if side == "in" else x_in[m] @ w[m].T
                        elif side == "out":
                            val = circuit_out[m]
                        elif block == "mlp" and m == "down_proj":
                            val = F.silu(circuit_out["gate_proj"]) * circuit_out["up_proj"]
                        else:
                            # attn circuit k/v inputs: upstream of the block's matrices,
                            # identical to the original input.
                            val = x_in[m]
                        sums[key] = sums.get(key, torch.zeros_like(val[0])) + val.sum(dim=0)
        mu = {key: (s / n_prompts) for key, s in sums.items()}

        # ---- Pass B: reference + interventions.
        kl = np.zeros((len(exp_keys), n, n), np.float32)
        overlap_sums: dict[tuple[str, Side], float] = {}
        with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
            for start in range(0, n_prompts, batch_size):
                chunk = pool[start : start + batch_size]
                b = chunk.shape[0]
                grid_rows = [a_ - 1 for a_, _ in ab[start : start + b]]
                grid_cols = [b_ - 1 for _, b_ in ab[start : start + b]]

                logP = torch.log_softmax(model(chunk)[:, -1].float(), dim=-1)
                P = logP.exp()

                if subset == "active":
                    mask_b = {
                        m: torch.from_numpy(supports[m][start : start + b]).to(device).float()
                        for m in mats
                    }
                    row_bases: dict[tuple[str, Side], list[Tensor | None]] = {}
                    sides: tuple[Side, ...] = ("in", "out")
                    for m in mats:
                        sup_b = supports[m][start : start + b]
                        for side in sides:
                            vecs = v[m] if side == "in" else u[m].T
                            bases: list[Tensor | None] = []
                            key_mu = mu[("orig", m, side)]
                            mu_norm = float(key_mu.norm()) or 1.0
                            ov = 0.0
                            for i in range(b):
                                idx = np.flatnonzero(sup_b[i])
                                q = (
                                    orthonormal_basis(vecs[:, torch.from_numpy(idx).to(device)])
                                    if len(idx)
                                    else None
                                )
                                bases.append(q)
                                ov += (
                                    float((key_mu - project_onto(key_mu[None], q)[0]).norm())
                                    / mu_norm
                                )
                            row_bases[(m, side)] = bases
                            overlap_sums[(m, side)] = overlap_sums.get((m, side), 0.0) + ov
                else:
                    mask_b = {m: static_mask[m] for m in mats}
                    row_bases = {}

                circuit_hooks = {m: circuit_hook(v[m], u[m], mask_b[m]) for m in mats}

                def run_variant(
                    circuit: bool,
                    proj_hooks: list[tuple[Any, str, Callable[..., Any]]],
                    _chunk: Tensor = chunk,
                    _logP: Tensor = logP,
                    _P: Tensor = P,
                    _circuit_hooks: dict[str, Callable[..., Any]] = circuit_hooks,
                ) -> np.ndarray:
                    handles = []
                    try:
                        if circuit:
                            for m in mats:
                                handles.append(linears[m].register_forward_hook(_circuit_hooks[m]))
                        for module, kind, hook in proj_hooks:
                            reg = (
                                module.register_forward_pre_hook
                                if kind == "pre"
                                else module.register_forward_hook
                            )
                            handles.append(reg(hook))
                        return _last_pos_kl(model(_chunk), _logP, _P)
                    finally:
                        for h in handles:
                            h.remove()

                if wiring_kl is None:
                    # Projecting an input onto row(W) while running the original W must
                    # be a no-op; validates projector construction + hook plumbing.
                    m_chk, q_chk = next(iter(matrix_q.items()))
                    assert m_chk[1] == "in"
                    chk = run_variant(
                        False,
                        [
                            (
                                linears[m_chk[0]],
                                "pre",
                                projection_pre_hook(lambda x, q=q_chk: project_onto(x, q)),
                            )
                        ],
                    )
                    wiring_kl = float(chk.mean())
                    logger.info(
                        f"wiring check (orig + row-space proj on {m_chk[0]}): "
                        f"mean KL {wiring_kl:.3g}"
                    )
                    assert wiring_kl < 0.05, f"row-space no-op check failed: KL {wiring_kl}"

                kl[len(experiments), grid_rows, grid_cols] = run_variant(True, [])

                for ei, (iv, flavor) in enumerate(experiments):
                    kind = "circuit" if iv.circuit else "orig"
                    mu_site = mu[(kind, iv.matrix, iv.side)]
                    if iv.space == "matrix":
                        q_fixed = matrix_q[(iv.matrix, iv.side)]
                        proj_fn = lambda x, q=q_fixed: project_onto(x, q)
                    elif subset == "alive":
                        q_span = span_q[(iv.matrix, iv.side)]
                        proj_fn = lambda x, q=q_span: project_onto(x, q)
                    else:
                        bases_site = row_bases[(iv.matrix, iv.side)]
                        proj_fn = lambda x, bs=bases_site: project_rows(x, bs)
                    transform = lambda x, f=flavor, m_=mu_site, p=proj_fn: apply_flavor(x, m_, f, p)
                    hook = (
                        (linears[iv.matrix], "pre", projection_pre_hook(transform))
                        if iv.side == "in"
                        else (linears[iv.matrix], "out", projection_out_hook(transform))
                    )
                    kl[ei, grid_rows, grid_cols] = run_variant(iv.circuit, [hook])

                logger.info(f"{op}: prompts {start}..{start + b} / {n_prompts}")

        # ---- Outputs for this op.
        kl_path = out_dir / f"kl_{block}_{subset}_{op}.tsv"
        with kl_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["experiment", "flavor", "a", "b", "kl"])
            for ei, key in enumerate(exp_keys):
                exp_name, _, flavor = key.partition("|")
                for ai in range(n):
                    for bi in range(n):
                        writer.writerow(
                            [exp_name, flavor or "none", ai + 1, bi + 1, f"{kl[ei, ai, bi]:.6g}"]
                        )
        logger.info(f"wrote {kl_path}")

        if subset == "active":
            nci_path = out_dir / f"nci_{block}_{op}.tsv"
            with nci_path.open("w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["a", "b", *(f"n_ci_{m}" for m in mats), "n_ci_total"])
                counts = {m: supports[m].sum(axis=1) for m in mats}
                for i, (a_, b_) in enumerate(ab):
                    per = [int(counts[m][i]) for m in mats]
                    writer.writerow([a_, b_, *per, sum(per)])
            logger.info(f"wrote {nci_path}")

        op_meta: dict[str, Any] = {
            "mu_norm": {
                f"{k[0]}.{k[1]}.{k[2]}": round(float(m_.norm()), 4) for k, m_ in mu.items()
            },
        }
        if subset == "alive":
            op_meta["mu_overlap_residual"] = {
                f"{m}.{s}": round(
                    float(
                        (
                            mu[("orig", m, s)]
                            - project_onto(mu[("orig", m, s)][None], span_q[(m, s)])[0]
                        ).norm()
                        / mu[("orig", m, s)].norm().clamp_min(1e-12)
                    ),
                    4,
                )
                for m in mats
                for s in cast(tuple[Side, ...], ("in", "out"))
            }
        else:
            op_meta["mu_overlap_residual_mean"] = {
                f"{m}.{s}": round(val / n_prompts, 4) for (m, s), val in overlap_sums.items()
            }
            op_meta["n_ci_mean"] = {
                m: round(float(supports[m].sum(axis=1).mean()), 2) for m in mats
            }
        meta_ops[op] = op_meta

    meta = {
        "run_id": run.run_dir.name,
        "block": block,
        "subset": subset,
        "alive_tsv": str(tsv) if subset == "alive" else None,
        "ci_thr": ci_thr if subset == "active" else None,
        "layer": layer,
        "ops": op_list,
        "experiments": [iv.key for iv in interventions] + [_BASELINE_KEY],
        "flavors": list(FLAVORS),
        "n_selected": {m: len(sel[m]) for m in mats} if subset == "alive" else "per-prompt",
        "span_rank": {
            f"{m}.{s}": (int(q.shape[1]) if q is not None else 0) for (m, s), q in span_q.items()
        }
        if subset == "alive"
        else "per-prompt",
        "matrix_space_rank": {f"{m}.{s}": int(q.shape[1]) for (m, s), q in matrix_q.items()},
        "wiring_check_kl": wiring_kl,
        "kl_direction": "KL(P_target || P_variant)",
        "position": "last (= answer) only; delta component never used",
        "per_op": meta_ops,
    }
    meta_path = out_dir / f"meta_{block}_{subset}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"wrote {meta_path}; done ({block}/{subset}, ops {op_list})")
    return out_dir


if __name__ == "__main__":
    fire.Fire(collect_filtered_kl)
