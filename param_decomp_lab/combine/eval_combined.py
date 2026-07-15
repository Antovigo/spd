"""Evaluate assembled multi-block decompositions without training.

For the requested runs, builds the combined `ComponentModel` (and optionally each run
alone, through the identical code path) and reports targeted recon metrics on the
target-prompt distribution plus nontarget recon on the broad distribution.
"""

import json
from pathlib import Path
from typing import Any

import fire
import torch

from param_decomp.batch_and_loss_fns import ReconstructionLoss, move_batch_to_device
from param_decomp.ci_fns import GroupedGlobalCiConfig
from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.metrics.base import Metric
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.targeted import delta_override
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.combine.assembly import (
    SourceRun,
    build_combined_component_model,
    load_source_runs,
    resolve_run_dir,
)
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.eval_metrics.ci_l0 import CI_L0, CI_L0Config
from param_decomp_lab.eval_metrics.targeted_recon_loss import (
    NontargetReconLoss,
    NontargetReconLossConfig,
    TargetReconLoss,
    TargetReconLossConfig,
)
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    SavedLMRun,
    build_lm_loader,
    build_target,
    make_reconstruction_loss,
    make_run_batch,
)
from param_decomp_lab.seed import set_seed


def _build_ctx(
    batch: Any,
    *,
    model: ComponentModel,
    pd_cfg: PDConfig,
    reconstruction_loss: ReconstructionLoss,
    weight_deltas: dict[str, torch.Tensor],
    device: str,
) -> MetricContext:
    batch = move_batch_to_device(batch, device)
    output_with_cache = model(batch, cache_type="input")
    ci = model.calc_causal_importances(
        pre_weight_acts=output_with_cache.cache,
        detach_inputs=False,
        sampling=pd_cfg.sampling,
    )
    return MetricContext(
        model=model,
        batch=batch,
        target_out=output_with_cache.output,
        pre_weight_acts=output_with_cache.cache,
        ci=ci,
        weight_deltas=weight_deltas,
        step=0,
        total_steps=1,
        use_delta_component=pd_cfg.use_delta_component,
        sampling=pd_cfg.sampling,
        n_mask_samples=pd_cfg.n_mask_samples,
        reconstruction_loss=reconstruction_loss,
        is_eval=True,
    )


def _per_batch_update(metrics: list[Metric[Any]], ctx: MetricContext) -> dict[str, float]:
    for m in metrics:
        m.update(ctx)
    outputs = collect_metric_outputs(metrics)
    for m in metrics:
        m.reset()
    return {k: float(v) for k, v in outputs.items() if isinstance(v, (int, float, torch.Tensor))}


def _evaluate_model(
    model: ComponentModel,
    ref_cfg: LMExperimentConfig,
    l0_groups: dict[str, list[str]],
    *,
    ci_thr: float,
    ci_alive_thr: float,
    n_steps: int,
    batch_size: int,
    nontarget_batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    pd_cfg = ref_cfg.pd
    assert ref_cfg.nontarget is not None, "expected targeted runs (nontarget block)"
    weight_deltas = model.calc_weight_deltas()

    target_loader = build_lm_loader(
        ref_cfg.target,
        ref_cfg.data,
        split="eval",
        device=device,
        batch_size=batch_size,
        seed=pd_cfg.seed,
    )
    nontarget_loader = build_lm_loader(
        ref_cfg.target,
        ref_cfg.nontarget.data,
        split="eval",
        device=device,
        batch_size=nontarget_batch_size,
        seed=pd_cfg.seed,
    )

    eval_cfg = ref_cfg.eval
    assert eval_cfg is not None
    pgd_configs = [m for m in eval_cfg.metrics if m.type == "PGDReconLoss"]
    assert len(pgd_configs) == 1, f"expected one PGDReconLoss eval entry, got {pgd_configs}"
    target_metrics: list[Metric[Any]] = [
        TargetReconLoss(
            TargetReconLossConfig(rounding_threshold=ci_thr, ci_alive_threshold=ci_alive_thr)
        ),
        EVAL_METRIC_CLASSES["PGDReconLoss"](pgd_configs[0]),
        CI_L0(CI_L0Config(groups=l0_groups, ci_alive_threshold=ci_alive_thr)),
    ]
    nontarget_metrics: list[Metric[Any]] = [
        NontargetReconLoss(
            NontargetReconLossConfig(rounding_threshold=ci_thr, ci_alive_threshold=ci_alive_thr)
        ),
    ]
    for m in target_metrics + nontarget_metrics:
        m.bind(model=model, device=device)

    per_batch: dict[str, list[float]] = {}

    def record(batch_values: dict[str, float]) -> None:
        for key, value in batch_values.items():
            per_batch.setdefault(key, []).append(value)

    target_recon_loss = make_reconstruction_loss(ref_cfg.target)
    with torch.no_grad(), bf16_autocast(enabled=ref_cfg.runtime.autocast_bf16):
        target_iter = loop_dataloader(target_loader)
        for _ in range(n_steps):
            ctx = _build_ctx(
                next(target_iter),
                model=model,
                pd_cfg=pd_cfg,
                reconstruction_loss=target_recon_loss,
                weight_deltas=weight_deltas,
                device=device,
            )
            record(_per_batch_update(target_metrics, ctx))
            del ctx
        torch.cuda.empty_cache()

        nontarget_iter = loop_dataloader(nontarget_loader)
        with delta_override(1.0):
            for _ in range(n_steps):
                ctx = _build_ctx(
                    next(nontarget_iter),
                    model=model,
                    pd_cfg=pd_cfg,
                    reconstruction_loss=recon_loss_kl,
                    weight_deltas=weight_deltas,
                    device=device,
                )
                record(_per_batch_update(nontarget_metrics, ctx))
                del ctx

    del weight_deltas
    torch.cuda.empty_cache()
    return per_batch


def _evaluate_assembled(
    sources: list[SourceRun],
    *,
    target_model: torch.nn.Module,
    seed: int,
    device: str,
    **kwargs: Any,
) -> dict[str, list[float]]:
    set_seed(seed)
    ref_cfg = sources[0].cfg
    model = build_combined_component_model(
        sources, target_model, make_run_batch(ref_cfg.target), device
    )
    l0_groups = {s.group: sorted(s.module_patterns) for s in sources}
    result = _evaluate_model(model, ref_cfg, l0_groups, device=device, **kwargs)
    del model
    torch.cuda.empty_cache()
    return result


def _evaluate_finetuned(
    run: str,
    *,
    target_model: torch.nn.Module,
    seed: int,
    device: str,
    **kwargs: Any,
) -> dict[str, list[float]]:
    set_seed(seed)
    saved = SavedLMRun.from_path(resolve_run_dir(run))
    ci_config = saved.cfg.pd.ci_config
    assert isinstance(ci_config, GroupedGlobalCiConfig), (
        f"{run}: expected a fine-tuned combined run (grouped_global CI), got {ci_config.mode}"
    )
    model = load_component_model(
        pd_config=saved.cfg.pd,
        checkpoint_path=saved.checkpoint_path,
        target_model=target_model,
        run_batch=make_run_batch(saved.cfg.target),
    ).to(device)
    result = _evaluate_model(model, saved.cfg, dict(ci_config.groups), device=device, **kwargs)
    del model
    torch.cuda.empty_cache()
    return result


def main(
    runs: str,
    out: str,
    finetuned: str | None = None,
    ci_thr: float = 0.01,
    ci_alive_thr: float = 0.1,
    n_steps: int = 10,
    batch_size: int | None = None,
    nontarget_batch_size: int = 64,
    include_singles: bool = True,
    include_combined: bool = True,
    device: str = "cuda:0",
    seed: int = 0,
) -> None:
    """Evaluate the combined decomposition of `runs` (comma-separated ids or paths).

    Args:
        runs: Comma-separated single-block run ids (under PARAM_DECOMP_OUT_DIR/runs)
            or paths, assembled on the fly.
        out: Output JSON path.
        finetuned: Comma-separated ids of fine-tuned combined runs (grouped_global CI)
            to evaluate as additional subjects, loaded from their own checkpoints.
        ci_thr: Rounding threshold for the rounded recon mask (CI > thr -> 1).
        ci_alive_thr: Aliveness threshold for L0 counts.
        n_steps: Eval batches per subject and distribution.
        batch_size: Target-distribution eval batch size; defaults to the runs' eval batch size.
        nontarget_batch_size: Nontarget eval batch size. Default 64: the fp32 KL over
            the 128k-token vocab at seq len 64 OOMs a 44 GiB L40 at batch 128 when the
            model carries several blocks' weight deltas.
        include_singles: Also evaluate each run alone through the identical code path.
        include_combined: Evaluate the raw (non-fine-tuned) assembly of all `runs`.
        device: Torch device.
        seed: Seed, re-applied before each subject so stochastic parts are comparable.
    """
    run_list = [r.strip() for r in runs.split(",") if r.strip()]
    assert run_list, "no runs given"
    sources = load_source_runs(run_list)
    ref_cfg = sources[0].cfg
    eval_cfg = ref_cfg.eval
    assert eval_cfg is not None
    effective_batch_size = batch_size if batch_size is not None else eval_cfg.batch_size
    finetuned_list = (
        [r.strip() for r in finetuned.split(",") if r.strip()] if finetuned is not None else []
    )

    target_model = build_target(ref_cfg.target)
    eval_kwargs: dict[str, Any] = {
        "ci_thr": ci_thr,
        "ci_alive_thr": ci_alive_thr,
        "n_steps": n_steps,
        "batch_size": effective_batch_size,
        "nontarget_batch_size": nontarget_batch_size,
    }

    subjects: list[tuple[str, list[SourceRun] | str]] = []
    if include_singles:
        subjects += [(s.name, [s]) for s in sources]
    if include_combined:
        subjects.append(("combined", sources))
    subjects += [(run, run) for run in finetuned_list]

    results: dict[str, dict[str, Any]] = {}
    for subject_name, subject in subjects:
        print(f"=== evaluating {subject_name} ===", flush=True)
        if isinstance(subject, str):
            per_batch = _evaluate_finetuned(
                subject, target_model=target_model, seed=seed, device=device, **eval_kwargs
            )
        else:
            per_batch = _evaluate_assembled(
                subject, target_model=target_model, seed=seed, device=device, **eval_kwargs
            )
        means = {k: sum(v) / len(v) for k, v in per_batch.items()}
        results[subject_name] = {"mean": means, "per_batch": per_batch}
        for key in sorted(means):
            print(f"  {key}: {means[key]:.6g}", flush=True)

    payload = {
        "meta": {
            "runs": {s.name: str(s.checkpoint_path) for s in sources},
            "finetuned": finetuned_list,
            "ci_thr": ci_thr,
            "ci_alive_thr": ci_alive_thr,
            "n_steps": n_steps,
            "batch_size": effective_batch_size,
            "nontarget_batch_size": nontarget_batch_size,
            "seed": seed,
        },
        "results": results,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
