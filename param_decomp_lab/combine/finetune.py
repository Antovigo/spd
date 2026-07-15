"""Fine-tune an assembled multi-block decomposition (objectives 2 and 3).

Builds a combined `LMExperimentConfig` from the source runs (union of decomposition
targets, per-block `grouped_global` CI fns or a fresh single global CI fn), loads the
assembled weights into a fresh `Trainer`, and trains with the full targeted setup
(nontarget pass included). The importance-minimality schedules are pinned to their
end-of-training values (final p, base coeff) so fine-tuning continues the converged
objective rather than replaying the anneal.

Run via torchrun for DDP:
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.combine.finetune ...`
"""

import json
from typing import Literal

import fire

from param_decomp.ci_fns import GlobalCiConfig
from param_decomp.configs import OptimizerConfig, PDConfig
from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.optimize import Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.combine.assembly import (
    SourceRun,
    combined_ci_config,
    load_combined_state,
    load_source_runs,
)
from param_decomp_lab.distributed import get_device, init_distributed, with_distributed_cleanup
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    _build_eval_loop,
    _build_nontarget_pass,
    build_lm_loader,
    build_target,
    make_reconstruction_loss,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import init_pd_run
from param_decomp_lab.seed import set_seed

CiFnMode = Literal["grouped", "global_fresh"]

# Effective freeze: schedules require start_val > 0, and a hard requires_grad_(False)
# after DDP construction would stall the reducer. Adam's step size is lr regardless of
# gradient scale, so 1e-12 over any realistic run moves parameters by ~1e-8 total.
FROZEN_LR = 1e-12


def _end_state_impmin(
    cfg: ImportanceMinimalityLossConfig, coeff: float
) -> ImportanceMinimalityLossConfig:
    final_p = (
        cfg.p_anneal_final_p
        if cfg.p_anneal_final_p is not None and cfg.p_anneal_start_frac < 1.0
        else cfg.pnorm
    )
    return cfg.model_copy(
        update={
            "coeff": coeff,
            "pnorm": final_p,
            "p_anneal_final_p": None,
            "p_anneal_start_frac": 1.0,
            "p_anneal_end_frac": 1.0,
            "coeff_warmup_frac": 0.0,
            "coeff_peak_multiplier": 1.0,
            "coeff_anneal_start_frac": 1.0,
            "coeff_anneal_end_frac": 1.0,
        }
    )


def make_combined_config(
    sources: list[SourceRun],
    *,
    label: str,
    steps: int,
    batch_size: int,
    components_lr: float,
    ci_fn_lr: float,
    impmin_coeff: float,
    ci_fn_mode: CiFnMode,
    ci_d_model: int | None,
    ci_n_blocks: int | None,
    save_every: int | None,
    nontarget_batch_size: int | None,
    eval_batch_size: int | None,
    wandb: bool,
) -> LMExperimentConfig:
    ref = sources[0].cfg
    ref_pd = ref.pd

    match ci_fn_mode:
        case "grouped":
            ci_config = combined_ci_config(sources)
            assert ci_d_model is None and ci_n_blocks is None, (
                "ci_d_model/ci_n_blocks only apply to ci_fn_mode='global_fresh'"
            )
        case "global_fresh":
            ref_ci = ref_pd.ci_config
            assert isinstance(ref_ci, GlobalCiConfig)
            transformer_cfg = ref_ci.simple_transformer_ci_cfg
            assert transformer_cfg is not None
            transformer_cfg = transformer_cfg.model_copy(
                update={
                    "d_model": ci_d_model if ci_d_model is not None else transformer_cfg.d_model,
                    "n_blocks": ci_n_blocks
                    if ci_n_blocks is not None
                    else transformer_cfg.n_blocks,
                }
            )
            ci_config = ref_ci.model_copy(update={"simple_transformer_ci_cfg": transformer_cfg})

    loss_metrics = [
        _end_state_impmin(m, impmin_coeff) if isinstance(m, ImportanceMinimalityLossConfig) else m
        for m in ref_pd.loss_metrics
    ]

    def with_lr(optimizer_cfg: OptimizerConfig, lr: float) -> OptimizerConfig:
        return optimizer_cfg.model_copy(
            update={"lr_schedule": optimizer_cfg.lr_schedule.model_copy(update={"start_val": lr})}
        )

    pd = ref_pd.model_copy(
        update={
            "decomposition_targets": [t for s in sources for t in s.cfg.pd.decomposition_targets],
            "ci_config": ci_config,
            "steps": steps,
            "batch_size": batch_size,
            "loss_metrics": loss_metrics,
            "components_optimizer": with_lr(ref_pd.components_optimizer, components_lr),
            "ci_fn_optimizer": with_lr(ref_pd.ci_fn_optimizer, ci_fn_lr),
        }
    )
    pd = PDConfig.model_validate(pd.model_dump())

    cadence = ref.cadence
    if save_every is not None:
        cadence = cadence.model_copy(update={"save_every": save_every})
    nontarget = ref.nontarget
    if nontarget_batch_size is not None:
        assert nontarget is not None
        nontarget = nontarget.model_copy(
            update={"batch_size": nontarget_batch_size, "eval_batch_size": nontarget_batch_size}
        )
    eval_cfg = ref.eval
    if eval_batch_size is not None:
        assert eval_cfg is not None
        eval_cfg = eval_cfg.model_copy(update={"batch_size": eval_batch_size})
    return ref.model_copy(
        update={
            "pd": pd,
            "label": label,
            "cadence": cadence,
            "nontarget": nontarget,
            "eval": eval_cfg,
            "wandb": ref.wandb if wandb else None,
        }
    )


@with_distributed_cleanup
def main(
    runs: str,
    label: str,
    steps: int = 2000,
    batch_size: int | None = None,
    components_lr: float = 1e-4,
    ci_fn_lr: float = 5e-5,
    freeze_ci_fns: bool = False,
    freeze_components: bool = False,
    impmin_coeff: float | None = None,
    ci_fn_mode: CiFnMode = "grouped",
    ci_d_model: int | None = None,
    ci_n_blocks: int | None = None,
    save_every: int | None = None,
    nontarget_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    wandb: bool = True,
    group: str | None = None,
    tags: str | None = None,
    run_id: str | None = None,
) -> None:
    """Fine-tune the assembled decomposition of `runs` (comma-separated ids or paths).

    Args:
        runs: Source runs; each contributes one block's components (+ CI fn if grouped).
        label: Run label (also the wandb name).
        steps: Fine-tuning steps.
        batch_size: Global batch size; defaults to the sources' batch size.
        components_lr: Components LR (cosine, same shape as the sources' schedule).
        ci_fn_lr: CI-fn LR; ignored when `freeze_ci_fns`.
        freeze_ci_fns: Pin the CI-fn LR to 0 (objective 2a).
        freeze_components: Pin the components LR to 0.
        impmin_coeff: Importance-minimality coeff; defaults to min over sources of
            their base (end-of-anneal) coeff.
        ci_fn_mode: 'grouped' loads each source's CI fn as a per-block group;
            'global_fresh' trains a single randomly-initialised global CI fn over all
            blocks (objective 3).
        ci_d_model / ci_n_blocks: Architecture overrides for `global_fresh`.
        save_every: Checkpoint period override.
        nontarget_batch_size: Override for the nontarget train/eval batch size.
        eval_batch_size: Override for the target eval batch size (the ref 128 is the
            single-process memory peak; the sources only ever ran 64 per rank).
        wandb: Pass false for throwaway probes (memory checks etc.).
        group / tags / run_id: Passed to the sink (wandb).
    """
    sources = load_source_runs([r.strip() for r in runs.split(",") if r.strip()])
    ref = sources[0].cfg

    effective_impmin = (
        impmin_coeff
        if impmin_coeff is not None
        else min(
            m.coeff
            for s in sources
            for m in s.cfg.pd.loss_metrics
            if isinstance(m, ImportanceMinimalityLossConfig) and m.coeff is not None
        )
    )
    cfg = make_combined_config(
        sources,
        label=label,
        steps=steps,
        batch_size=batch_size if batch_size is not None else ref.pd.batch_size,
        components_lr=FROZEN_LR if freeze_components else components_lr,
        ci_fn_lr=FROZEN_LR if freeze_ci_fns else ci_fn_lr,
        impmin_coeff=effective_impmin,
        ci_fn_mode=ci_fn_mode,
        ci_d_model=ci_d_model,
        ci_n_blocks=ci_n_blocks,
        save_every=save_every,
        nontarget_batch_size=nontarget_batch_size,
        eval_batch_size=eval_batch_size,
        wandb=wandb,
    )

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(
            f"Sources: {[s.name for s in sources]}, impmin_coeff={effective_impmin}, "
            f"ci_fn_mode={ci_fn_mode}, freeze_ci_fns={freeze_ci_fns}, "
            f"freeze_components={freeze_components}"
        )
    set_seed(cfg.pd.seed)
    device = get_device()
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            )
        }
    )

    target_model = build_target(cfg.target)
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    eval_loop = _build_eval_loop(cfg, device, dist_state)
    nontarget = _build_nontarget_pass(cfg, device, dist_state)

    sink = init_pd_run(cfg, group=group, tags=tags, run_id=run_id)
    if is_main_process() and sink.out_dir is not None:
        provenance = {
            "sources": {s.name: str(s.checkpoint_path) for s in sources},
            "ci_fn_mode": ci_fn_mode,
            "freeze_ci_fns": freeze_ci_fns,
            "freeze_components": freeze_components,
        }
        (sink.out_dir / "combine_provenance.json").write_text(json.dumps(provenance, indent=2))

    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=make_reconstruction_loss(cfg.target),
            nontarget_reconstruction_loss=recon_loss_kl,
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        load_combined_state(trainer.component_model, sources, load_ci_fns=ci_fn_mode == "grouped")
        trainer.run(train_loader, sink, cfg.cadence, eval_loop, nontarget=nontarget)
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
