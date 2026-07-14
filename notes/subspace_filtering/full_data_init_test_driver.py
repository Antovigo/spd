"""Load-once driver: coupled vs kaiming init on the full-data pile_llama 4L decomposition.

Runs 6 conditions ({coupled, kaiming} x seeds {0, 1, 2}) in a single process so the
target model and the data pipeline are built exactly once and shared across every
condition. Each condition is measured at three points:

  (a) init        raw init, before warmup   -> a `warmup=0, steps=1` pass ("raw")
  (b) post-warmup after 100 warmup steps     -> the `train` pass, eval @ step 0
  (c) trained     after 200 main steps        -> the `train` pass, eval @ step 200

The `raw` pass takes one optimizer step at step 0 (steps must be a PositiveInt), but the
step is bounded by `grad_clip_norm=0.01` (||delta|| <= lr*0.01 = 5e-7) so the step-0 eval
is effectively raw init. The exact raw-init weight faithfulness is `train/loss/FaithfulnessLoss`
@ step 0 of the raw pass — it is computed on the pre-step weights.

Data is held constant across all conditions (fixed DATA_SEED); only `pd.seed` varies, so
the three seeds are three draws of the initialization / stochastic training, on identical
data. Weight init and warmup are the only knobs that change what is being compared.

Launch (8xH100, single node), from the repo root:

    torchrun --standalone --nproc_per_node=8 \
        notes/subspace_filtering/full_data_init_test_driver.py
"""

import gc

import torch

from param_decomp.distributed import is_main_process
from param_decomp.log import logger
from param_decomp.optimize import EvalLoop, Trainer
from param_decomp_lab.distributed import get_device, init_distributed
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    build_lm_loader,
    build_target,
    make_reconstruction_loss,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import init_pd_run
from param_decomp_lab.infra.settings import REPO_ROOT
from param_decomp_lab.seed import set_seed

BASE_CONFIG_PATH = REPO_ROOT / "param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml"

SCHEMES = ["kaiming", "coupled"]
SEEDS = [0, 1, 2]

# Data is identical for every condition: init variance is the only thing seeds vary.
DATA_SEED = 0

# Measurement-point config. `eval every == 200` fires eval at step 0 and step 200.
WARMUP_STEPS = 100
TRAIN_STEPS = 200
EVAL_EVERY = 200


def build_eval_loop(cfg: LMExperimentConfig, eval_loader) -> EvalLoop:
    """EvalLoop wrapping the shared eval loader with fresh metric instances."""
    assert cfg.eval is not None
    metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
    return EvalLoop(
        loader=eval_loader,
        metrics=metrics,
        n_steps=cfg.eval.n_steps,
        every=EVAL_EVERY,
        slow_every=EVAL_EVERY,
        slow_on_first_step=True,
        nontarget=None,
    )


def _skip_checkpoint(_snapshot: object) -> None:
    """No-op: this experiment needs only metrics.jsonl, not the multi-GB, optimizer-bearing
    model_/training_.pth snapshots Trainer writes unconditionally at the final step."""


def run_condition(
    *,
    base_cfg: LMExperimentConfig,
    target_model,
    train_loader,
    eval_loader,
    scheme: str,
    seed: int,
    phase: str,
) -> None:
    """One (scheme, seed, phase) pass. `phase` is 'raw' (init) or 'train'."""
    warmup_steps = 0 if phase == "raw" else WARMUP_STEPS
    steps = 1 if phase == "raw" else TRAIN_STEPS

    pd = base_cfg.pd.model_copy(
        update={
            "weight_init": scheme,
            "seed": seed,
            "faithfulness_warmup_steps": warmup_steps,
            "steps": steps,
        }
    )
    cadence = base_cfg.cadence.model_copy(update={"train_log_every": EVAL_EVERY})
    cfg = base_cfg.model_copy(update={"pd": pd, "cadence": cadence})

    set_seed(seed)
    eval_loop = build_eval_loop(cfg, eval_loader)
    run_id = f"cinit-{scheme}-s{seed}-{phase}"
    # wandb disabled (cfg.wandb=None) -> RunSink.local, metrics go to metrics.jsonl only.
    sink = init_pd_run(cfg, group=None, tags=None, run_id=run_id)
    # Trainer checkpoints unconditionally at the final step; skip the (multi-GB) write so
    # the pod only needs disk for metrics.jsonl. RunSink is frozen -> object.__setattr__.
    object.__setattr__(sink, "checkpoint", _skip_checkpoint)

    if is_main_process():
        logger.info(f"=== condition {run_id}: warmup={warmup_steps} steps={steps} ===")

    try:
        trainer = Trainer(
            target_model=target_model,
            run_batch=make_run_batch(cfg.target),
            reconstruction_loss=make_reconstruction_loss(cfg.target),
            pd_config=cfg.pd,
            runtime_config=cfg.runtime,
        )
        trainer.run(train_loader, sink, cfg.cadence, eval_loop)
    finally:
        sink.finish()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    base_cfg = LMExperimentConfig.from_file(BASE_CONFIG_PATH)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    device = get_device()
    base_cfg = base_cfg.model_copy(
        update={
            "wandb": None,  # local-only logging: metrics.jsonl per run, no wandb sync
            "runtime": base_cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            ),
        }
    )
    assert base_cfg.eval is not None, "the pile config must define an eval block"

    # --- load target + data ONCE, share across all 12 passes --- #
    target_model = build_target(base_cfg.target)
    train_loader = build_lm_loader(
        base_cfg.target,
        base_cfg.data,
        split="train",
        device=device,
        batch_size=base_cfg.pd.batch_size,
        dist_state=dist_state,
        seed=DATA_SEED,
    )
    eval_loader = build_lm_loader(
        base_cfg.target,
        base_cfg.data,
        split="eval",
        device=device,
        batch_size=base_cfg.eval.batch_size,
        dist_state=dist_state,
        seed=DATA_SEED,
    )

    for scheme in SCHEMES:
        for seed in SEEDS:
            for phase in ("raw", "train"):
                run_condition(
                    base_cfg=base_cfg,
                    target_model=target_model,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    scheme=scheme,
                    seed=seed,
                    phase=phase,
                )

    if is_main_process():
        logger.info("All conditions finished.")


if __name__ == "__main__":
    main()
