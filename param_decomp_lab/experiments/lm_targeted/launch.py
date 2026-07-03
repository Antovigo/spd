"""`pd-lm-targeted` launcher: snapshot + shared-FS workspace + sbatch, config-driven via
`runtime.dp` — the tPD sibling of `param_decomp_lab.experiments.lm.launch`.

Identical in shape to the full-VPD launcher; the ONLY difference is the srun command runs
`python -m param_decomp_lab.experiments.lm_targeted.run <config> --run-id <id>` instead of
`...experiments.lm.run`, and it validates the `LMTargetedExperimentConfig` schema.

    pd-lm-targeted <config.yaml>            # dp=N -> sbatch across N//8 nodes
    pd-lm-targeted <config.yaml>            # dp=null -> run inline (smoke)

See `notes/targeted_jax_plan.md` Phase 6.
"""

import fire


def main(
    config_path: str,
    time: str = "12:00:00",
    qos: str | None = None,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    comment: str | None = None,
) -> None:
    """TODO(tPD): reuse `experiments.lm.launch` almost verbatim — validate against
    `LMTargetedExperimentConfig` (not `LMExperimentConfig`), and point the rank command at
    `experiments.lm_targeted.run`. Prefer importing/parametrizing the lm launcher's helpers
    (`_build_workspace`, `_stamp_config`, `_render_rank_env`, `_run_local`) over copying, if
    the module-run target can be parametrized cleanly; else duplicate per the additive-merge
    preference. Default `--time` follows the cluster 12h convention (not lm.launch's).
    """
    _ = (config_path, time, qos, run_id, group, tags, comment)  # pending implementation
    raise NotImplementedError("tPD launcher — see targeted_jax_plan.md Phase 6")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
