"""The targeted-PD (tPD) composition root: a dual-stream LM decomposition.

    python -m param_decomp_lab.experiments.lm_targeted.run <wrapper.yaml>  # via pd-lm-targeted

Parallels `param_decomp_lab.experiments.lm.run` (the "full VPD" composition root), adding a
second data stream. It builds:
  - the TARGET loader — a fixed prompt pool (`data.TargetPromptServer`), and
  - the NON-TARGET loader — the normal parquet `ShardServer`,
then calls the engine with the non-target pass, which runs a second recon grid under
`delta_override=1.0` (delta forced fully on) and accumulates its gradient into the same
optimizer step. See `notes/targeted_jax_plan.md` and SPEC-tPD.

This is a SCAFFOLD: the structure + seams are in place; the marked TODOs are the fill-in.
"""

from pathlib import Path

import fire


def train_targeted(config: Path, run_id: str) -> None:
    """Build both loaders + the non-target pass and call `run_decomposition_training`.

    TODO(tPD): mirror `lm.run.train`, then additionally:
      1. Build the TARGET `sample_batch` from `data.TargetPromptServer` (fixed prompt pool).
      2. Build the NON-TARGET `sample_batch` from the parquet `ShardServer` at
         `cfg.nontarget.batch_size`.
      3. Build the non-target loss set (`config.build_nontarget_loss_metrics`).
      4. Pass a `NontargetPass(sample_batch_nontarget, loss_metrics, impmin_coeff_ratio)`
         into `run_decomposition_training` (new optional engine arg, default None — see plan
         Phase 1), which threads `delta_override=1.0` into the non-target recon grid.
      5. Add the tPD eval metrics (Target/NontargetReconLoss, TargetedCIHeatmap,
         WeightMagnitude) to the `eval_fn` (plan Phase 5).
    """
    _ = (config, run_id)  # pending implementation
    raise NotImplementedError("tPD composition root — see targeted_jax_plan.md Phase 4")


def main(config: Path, run_id: str) -> None:
    """Process setup mirrors `lm.run.main` (sigterm flag, init_distributed, XLA cache,
    HF hardening, mesh, config pinning), then `train_targeted`.

    TODO(tPD): factor the shared process-setup out of `lm.run.main` or duplicate it here
    (per the repo's additive-merge preference), then call `train_targeted(config, run_id)`.
    """
    _ = (config, run_id)  # pending implementation
    raise NotImplementedError("tPD composition root — see targeted_jax_plan.md Phase 4")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
