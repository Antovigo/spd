"""Subspace-filtering battery for a full-data decomposition — every layer, every position.

Same question as `collect_filtered_kl.py` (do the selected subcomponents' subspaces match
the original matrices' causally-used subspaces?) for a decomposition trained on its
model's own pretraining distribution, e.g. the 4-block Pile run `s-55ea3f9b`. Differences
from the targeted battery:

- Data: `--n-prompts` sequences drawn from the run's own training stream (seeded),
  trimmed to `--seq-len` tokens; token ids are saved for reproducibility.
- Positions: interventions apply at **every** position, and KL(P_target || P_variant) is
  recorded **per position**.
- Layers: each intervention acts on the same matrix type across **all** decomposed layers
  simultaneously (per-layer spans/means), e.g. `orig_in_span:c_fc` projects every
  `h.<L>.mlp.c_fc` input; the circuit replaces the whole block type at every layer.
- Selected sets are **per (prompt, position)**: the subcomponents with lower-leaky
  CI > `--ci-thr` at that position. The circuit mask is the per-position CI mask, and
  each position is projected onto its own span (padded batched bases; an empty set is
  the zero span).
- `mu` for centered/bias flavors: pooled mean over all sampled tokens (prompts x
  positions) per site, under the model variant being intervened on; circuit-site means
  come from an actual circuit forward (no closed-form dataflow assumptions).

The block's matrices and layers are derived from `model.components`. All target linears
must be bias-free (asserted). Blocks whose matrices are all square get no matrix-space
(experiment 2) interventions and skip the row-space wiring check.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_filtering.collect_filtered_kl_fulldata \
        <model_path> <block:mlp|attn> [--n-prompts=100] [--seq-len=32] [--ci-thr=0.01] \
        [--batch-size=25] [--seed=0] [--output-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Outputs (default `<run>/analysis/datasets/subspace_filtering/`):
- `kl_fulldata_<block>.tsv` — long format: experiment, flavor, prompt, pos, kl.
- `counts_fulldata_<block>.npz` — per-position active counts [n_prompts, seq] per module.
- `prompts_fulldata.npz` — the sampled token ids [n_prompts, seq].
- `meta_fulldata_<block>.json` — per-position span-rank stats vs d (the vacuousness
  diagnostic), matrix-space ranks, wiring check, sampling settings.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import fire
import numpy as np
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.data import create_lm_data_loader
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    analysis_datasets_dir,
    load_lm_run,
    parse_module_name,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.subspace_filtering.lib import (
    FLAVORS,
    apply_flavor,
    full_circuit_hook,
    full_projection_out_hook,
    full_projection_pre_hook,
    orthonormal_basis,
    project_onto,
    project_pairs,
)

_MODULE = "param_decomp_lab.scripts.validation.subspace_filtering.collect_filtered_kl_fulldata"
_BLOCK_PREFIX = {"mlp": "mlp.", "attn": ("attn.", "self_attn.")}
_BASELINE_KEY = "circuit_baseline"

Side = Literal["in", "out"]
LM = tuple[int, str]  # (layer, bare matrix name)


@dataclass(frozen=True)
class _Intervention:
    key: str
    circuit: bool
    matrix: str  # bare matrix name; applies at every layer
    side: Side
    space: Literal["span", "matrix"]


def _kl_all_positions(out: Tensor, logP: Tensor, P: Tensor) -> np.ndarray:
    """`[b, seq]` KL of the per-position next-token distributions vs the reference."""
    logQ = torch.log_softmax(out.float(), dim=-1)
    return (P * (logP - logQ)).sum(dim=-1).cpu().numpy()


def _pair_bases(vecs: Tensor, sup: np.ndarray, device: torch.device) -> tuple[Tensor, np.ndarray]:
    """Per-(prompt, position) padded span bases from `vecs [d, C]` and `sup [b, seq, C]`.

    Returns `(q_pad [b*seq, d, r_max], ranks [b*seq])`; empty sets give all-zero slices
    (the zero span).
    """
    flat = sup.reshape(-1, sup.shape[-1])
    bases: list[Tensor | None] = []
    for i in range(flat.shape[0]):
        idx = np.flatnonzero(flat[i])
        bases.append(
            orthonormal_basis(vecs[:, torch.from_numpy(idx).to(device)]) if len(idx) else None
        )
    ranks = np.array([0 if q is None else q.shape[1] for q in bases], np.int32)
    r_max = max(int(ranks.max()), 1)
    q_pad = torch.zeros(flat.shape[0], vecs.shape[0], r_max, device=device)
    for i, q in enumerate(bases):
        if q is not None:
            q_pad[i, :, : q.shape[1]] = q
    return q_pad, ranks


def collect_filtered_kl_fulldata(
    model_path: ModelPath,
    block: str,
    n_prompts: int = 100,
    seq_len: int = 32,
    ci_thr: float = 0.01,
    batch_size: int = 25,
    seed: int = 0,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    """Write the long-format per-position KL TSV + meta for one block.

    Returns the output dir, or `None` when `--slurm` submits the job instead.
    """
    assert block in _BLOCK_PREFIX, f"block must be one of {list(_BLOCK_PREFIX)}"
    if slurm:
        argv = [
            str(Path(model_path).expanduser().resolve()),
            block,
            f"--n-prompts={n_prompts}",
            f"--seq-len={seq_len}",
            f"--ci-thr={ci_thr}",
            f"--batch-size={batch_size}",
            f"--seed={seed}",
        ]
        if output_dir is not None:
            argv.append(f"--output-dir={Path(output_dir).expanduser().resolve()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-subfilt-fd-{block}")
        return None

    run = load_lm_run(model_path)
    model, cfg, device = run.model, run.cfg, run.device

    # Derive the block's decomposed (layer, matrix) set from the components themselves.
    prefixes = _BLOCK_PREFIX[block]
    block_modules: dict[LM, str] = {}
    for module in model.components:
        layer, matrix = parse_module_name(module)
        if matrix.startswith(prefixes):
            block_modules[(layer, matrix.split(".")[-1])] = module
    layers = sorted({lm[0] for lm in block_modules})
    mats = tuple(sorted({lm[1] for lm in block_modules}))
    assert layers and mats, f"no decomposed {block} modules in {sorted(model.components)}"
    assert set(block_modules) == {(la, m) for la in layers for m in mats}, (
        f"ragged decomposition: {sorted(block_modules)}"
    )
    lms: list[LM] = [(la, m) for la in layers for m in mats]

    linears = {lm: model.target_model.get_submodule(block_modules[lm]) for lm in lms}
    for lm in lms:
        assert getattr(linears[lm], "bias", None) is None, f"{block_modules[lm]} has a bias"
    w = {lm: cast(Tensor, linears[lm].weight).detach().float() for lm in lms}  # [d_out, d_in]
    v = {lm: model.components[block_modules[lm]].V.detach().float() for lm in lms}
    u = {lm: model.components[block_modules[lm]].U.detach().float() for lm in lms}
    for m in mats:
        shapes = {w[(la, m)].shape for la in layers}
        assert len(shapes) == 1, f"{m}: inconsistent shapes across layers {shapes}"

    shape_of = {m: w[(layers[0], m)].shape for m in mats}
    interventions = [_Intervention(f"orig_in_span:{m}", False, m, "in", "span") for m in mats]
    interventions += [_Intervention(f"orig_out_span:{m}", False, m, "out", "span") for m in mats]
    interventions += [
        _Intervention(f"circuit_in_row:{m}", True, m, "in", "matrix")
        for m in mats
        if shape_of[m][0] < shape_of[m][1]
    ]
    interventions += [
        _Intervention(f"circuit_out_col:{m}", True, m, "out", "matrix")
        for m in mats
        if shape_of[m][1] < shape_of[m][0]
    ]
    exp_rows = [(iv.key, flavor) for iv in interventions for flavor in FLAVORS]
    exp_rows.append((_BASELINE_KEY, "none"))
    row_of = {key: i for i, key in enumerate(exp_rows)}

    matrix_q: dict[tuple[LM, Side], Tensor] = {}
    for iv in interventions:
        if iv.space != "matrix":
            continue
        for la in layers:
            lm = (la, iv.matrix)
            vectors = w[lm].T if iv.side == "in" else w[lm]
            matrix_q[(lm, iv.side)] = orthonormal_basis(vectors)
    logger.info(
        f"{block}: layers {layers}, matrices {mats}, {len(interventions)} interventions, "
        f"matrix-space sites {sorted((f'h.{la}.{m}.{s}') for (la, m), s in matrix_q)}"
    )

    # ---- Sample the evaluation sequences from the run's own training stream.
    loader, _tok = create_lm_data_loader(
        cfg.data, split=cfg.data.train_split, batch_size=n_prompts, seed=seed
    )
    batch = next(iter(loader))
    tokens = (batch["input_ids"] if isinstance(batch, dict) else batch)[:, :seq_len]
    assert tokens.shape == (n_prompts, seq_len), f"got {tuple(tokens.shape)}"
    pool = tokens.to(device)
    seq = seq_len
    n_tokens = n_prompts * seq

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_datasets_dir(run.run_dir) / "subspace_filtering"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prompts_fulldata.npz", tokens=tokens.cpu().numpy())

    mean_keys: set[tuple[str, LM, Side]] = set()
    for iv in interventions:
        for la in layers:
            mean_keys.add(("circuit" if iv.circuit else "orig", (la, iv.matrix), iv.side))

    # Per-(prompt, position) active sets, bit-packed nowhere — plain bool [n, seq, C].
    supports: dict[LM, np.ndarray] = {
        lm: np.zeros((n_prompts, seq, v[lm].shape[1]), dtype=bool) for lm in lms
    }
    sums: dict[tuple[str, LM, Side], Tensor] = {}

    def accumulate(key: tuple[str, LM, Side], val: Tensor) -> None:
        sums[key] = sums.get(key, torch.zeros_like(val[0, 0])) + val.float().sum(dim=(0, 1))

    def batch_masks(sl: slice) -> dict[LM, Tensor]:
        return {lm: torch.from_numpy(supports[lm][sl]).to(device).float() for lm in lms}

    # ---- Pass A: per-position CI sets + pooled per-site means (orig and circuit).
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            sl = slice(start, start + chunk.shape[0])
            cached = model(chunk, cache_type="input")
            ci = model.calc_causal_importances(cached.cache, sampling="continuous")
            for lm in lms:
                supports[lm][sl] = (ci.lower_leaky[block_modules[lm]] > ci_thr).cpu().numpy()
            for key in mean_keys:
                kind, lm, side = key
                if kind != "orig":
                    continue
                x = cached.cache[block_modules[lm]]
                accumulate(key, x if side == "in" else x.float() @ w[lm].T)

            circuit_keys = [k for k in mean_keys if k[0] == "circuit"]
            if circuit_keys:
                mask_b = batch_masks(sl)
                handles = []
                cache_hits: dict[tuple[str, LM, Side], Tensor] = {}

                def cache_in_hook(
                    key: tuple[str, LM, Side],
                    hits: dict[tuple[str, LM, Side], Tensor] = cache_hits,
                ) -> Any:
                    def hook(_mod: Any, args: tuple[Any, ...]) -> None:
                        hits[key] = args[0]

                    return hook

                def cache_out_hook(
                    key: tuple[str, LM, Side],
                    hits: dict[tuple[str, LM, Side], Tensor] = cache_hits,
                ) -> Any:
                    def hook(_mod: Any, _args: tuple[Any, ...], output: Tensor) -> None:
                        hits[key] = output

                    return hook

                try:
                    for lm in lms:
                        handles.append(
                            linears[lm].register_forward_hook(
                                full_circuit_hook(v[lm], u[lm], mask_b[lm])
                            )
                        )
                    # Caching hooks registered after the circuit hooks: pre-hooks see the
                    # (upstream-)circuited input; out-hooks chain after the replacement.
                    for key in circuit_keys:
                        _, lm, side = key
                        if side == "in":
                            handles.append(
                                linears[lm].register_forward_pre_hook(cache_in_hook(key))
                            )
                        else:
                            handles.append(linears[lm].register_forward_hook(cache_out_hook(key)))
                    model(chunk)
                finally:
                    for h in handles:
                        h.remove()
                for key in circuit_keys:
                    accumulate(key, cache_hits[key])
            logger.info(f"pass A: prompts {start}..{start + chunk.shape[0]} / {n_prompts}")
    mu = {key: s / n_tokens for key, s in sums.items()}

    # ---- Pass B: reference + interventions, KL at every position.
    kl = np.zeros((len(exp_rows), n_prompts, seq), np.float32)
    rank_sums: dict[str, list[int]] = {}
    wiring_kl: float | None = None
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, n_prompts, batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            sl = slice(start, start + b)

            logP = torch.log_softmax(model(chunk).float(), dim=-1)  # [b, seq, vocab]
            P = logP.exp()

            mask_b = batch_masks(sl)
            circuit_hooks = {lm: full_circuit_hook(v[lm], u[lm], mask_b[lm]) for lm in lms}

            def run_variant(
                circuit: bool,
                proj_hooks: list[tuple[Any, str, Any]],
                _chunk: Tensor = chunk,
                _logP: Tensor = logP,
                _P: Tensor = P,
                _circuit_hooks: dict[LM, Any] = circuit_hooks,
            ) -> np.ndarray:
                handles = []
                try:
                    if circuit:
                        for lm in lms:
                            handles.append(linears[lm].register_forward_hook(_circuit_hooks[lm]))
                    for module, kind, hook in proj_hooks:
                        reg = (
                            module.register_forward_pre_hook
                            if kind == "pre"
                            else module.register_forward_hook
                        )
                        handles.append(reg(hook))
                    return _kl_all_positions(model(_chunk), _logP, _P)
                finally:
                    for h in handles:
                        h.remove()

            if wiring_kl is None:
                in_sites = [k for k in matrix_q if k[1] == "in"]
                if in_sites:
                    lm_chk, side_chk = in_sites[0]
                    q_chk = matrix_q[(lm_chk, side_chk)]
                    chk = run_variant(
                        False,
                        [
                            (
                                linears[lm_chk],
                                "pre",
                                full_projection_pre_hook(lambda x, q=q_chk: project_onto(x, q)),
                            )
                        ],
                    )
                    wiring_kl = float(chk.mean())
                    logger.info(
                        f"wiring check (orig + row-space proj on h.{lm_chk[0]}.{lm_chk[1]}): "
                        f"mean KL {wiring_kl:.3g}"
                    )
                    assert wiring_kl < 0.05, f"row-space no-op check failed: KL {wiring_kl}"
                else:
                    wiring_kl = float("nan")
                    logger.info("no rectangular matrix in block; row-space wiring check skipped")

            kl[row_of[(_BASELINE_KEY, "none")], sl] = run_variant(True, [])

            for iv in interventions:
                kind = "circuit" if iv.circuit else "orig"
                # Per-(prompt, position) bases for this intervention's span sites, shared
                # by the three flavors; matrix-space sites use their fixed basis.
                pair_q: dict[LM, Tensor] = {}
                if iv.space == "span":
                    for la in layers:
                        lm = (la, iv.matrix)
                        vecs = v[lm] if iv.side == "in" else u[lm].T
                        q_pad, ranks = _pair_bases(vecs, supports[lm][sl], device)
                        pair_q[lm] = q_pad
                        rank_sums.setdefault(f"h.{la}.{iv.matrix}.{iv.side}", []).extend(
                            ranks.tolist()
                        )
                for flavor in FLAVORS:
                    hooks = []
                    for la in layers:
                        lm = (la, iv.matrix)
                        if iv.space == "matrix":
                            q_fixed = matrix_q[(lm, iv.side)]
                            proj_fn: Any = lambda x, q=q_fixed: project_onto(x, q)
                        else:
                            proj_fn = lambda x, q=pair_q[lm]: project_pairs(x, q)
                        mu_site = mu[(kind, lm, iv.side)]
                        transform = lambda x, f=flavor, m_=mu_site, p=proj_fn: apply_flavor(
                            x, m_, f, p
                        )
                        hooks.append(
                            (linears[lm], "pre", full_projection_pre_hook(transform))
                            if iv.side == "in"
                            else (linears[lm], "out", full_projection_out_hook(transform))
                        )
                    kl[row_of[(iv.key, flavor)], sl] = run_variant(iv.circuit, hooks)
                del pair_q

            logger.info(f"pass B: prompts {start}..{start + b} / {n_prompts}")

    # ---- Outputs.
    kl_path = out_dir / f"kl_fulldata_{block}.tsv"
    with kl_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["experiment", "flavor", "prompt", "pos", "kl"])
        for ei, (exp_name, flavor) in enumerate(exp_rows):
            for pi in range(n_prompts):
                for pos in range(seq):
                    writer.writerow([exp_name, flavor, pi, pos, f"{kl[ei, pi, pos]:.6g}"])
    logger.info(f"wrote {kl_path}")

    count_arrays: dict[str, Any] = {
        f"h{la}_{m}": supports[(la, m)].sum(axis=-1).astype(np.int32) for la, m in lms
    }
    np.savez_compressed(out_dir / f"counts_fulldata_{block}.npz", **count_arrays)

    rank_stats = {
        site: {
            "rank_mean": float(np.mean(r)),
            "rank_min": int(np.min(r)),
            "rank_max": int(np.max(r)),
        }
        for site, r in sorted(rank_sums.items())
    }
    meta = {
        "run_id": run.run_dir.name,
        "block": block,
        "layers": layers,
        "matrices": list(mats),
        "n_prompts": n_prompts,
        "seq_len": seq_len,
        "seed": seed,
        "ci_thr": ci_thr,
        "set_scope": "per (prompt, position): lower-leaky CI > ci_thr at that position",
        "mu": "pooled mean over prompts x positions, per site, under the intervened model",
        "d_model_side": {
            f"{m}.{s}": int(shape_of[m][1 if s == "in" else 0]) for m in mats for s in ("in", "out")
        },
        "experiments": [iv.key for iv in interventions] + [_BASELINE_KEY],
        "flavors": list(FLAVORS),
        "span_rank_stats": rank_stats,
        "matrix_space_rank": {
            f"h.{lm[0]}.{lm[1]}.{s}": int(q.shape[1]) for (lm, s), q in matrix_q.items()
        },
        "wiring_check_kl": wiring_kl,
        "kl_direction": "KL(P_target || P_variant), every position",
        "data": f"{cfg.data.dataset_name} split={cfg.data.train_split} (decomposition training data)",
    }
    meta_path = out_dir / f"meta_fulldata_{block}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"wrote {meta_path}; done ({block}, {n_prompts} prompts x {seq_len} positions)")
    return out_dir


if __name__ == "__main__":
    fire.Fire(collect_filtered_kl_fulldata)
