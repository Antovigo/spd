"""`pd-tms`: run a TMS (Toy Model of Superposition) parameter decomposition on CPU.

The toy domains live lab-side and call the generic core engine
(`param_decomp.run.run_decomposition_training`) as a library — the core itself carries
zero toy-specific code. A TMS run pretrains its tiny target from scratch in-process (the
Anthropic `mean((|x|-out)^2)` objective), then decomposes it through the same engine the
LM uses, validating via the ground-truth identity-CI metric logged every train-log step.

These toys train in seconds; `pd-tms` runs synchronously on CPU in the main venv (no
SLURM / `param_decomp.run` / CUDA). It mints its own `p-<8hex>` run id (toys do not go through
`pd-lm`).
"""

from pathlib import Path
from typing import Any

import fire
import jax
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.config import BuiltRun
from param_decomp.lm import SiteC
from param_decomp.recon import build_recon_terms
from param_decomp.run import run_decomposition_training
from param_decomp.sharding import dp_mesh
from param_decomp.train import TrainState
from param_decomp_lab.experiments import toy_uv_eval
from param_decomp_lab.experiments.config import (
    assert_canonical_algorithm_config,
    layerwise_mlp_ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.tms import model as tms
from param_decomp_lab.experiments.tms.config import TMSExperimentConfig
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


def build_tms_built_run(cfg: TMSExperimentConfig) -> BuiltRun:
    """Convert the canonical TMS schema to the engine's `BuiltRun` bundle via the shared
    helpers. TMS validates via the in-loop target-CI metric (not the LM CEandKLLosses scalar
    pass), so `eval` is `None`. The schema's `eval.metrics` list is still read at run time
    for the config-gated `UVPlots` figure (`toy_uv_eval`)."""
    site_cs = tms.canonical_site_cs(
        tuple(SiteC(t.module_pattern, t.C) for t in cfg.pd.decomposition_targets)
    )
    assert_canonical_algorithm_config(cfg)
    build_recon_terms(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
        cfg.pd.n_mask_samples,
        cfg.pd.sampling,
    )
    target = tms.TMSTargetConfig(
        n_features=cfg.target.n_features,
        n_hidden=cfg.target.n_hidden,
        n_hidden_layers=cfg.target.n_hidden_layers,
        hidden_layer_init=cfg.target.hidden_layer_init,
        init_bias_to_zero=cfg.target.init_bias_to_zero,
        sites=site_cs,
        pretrain_steps=cfg.target.pretrain.steps,
        pretrain_batch_size=cfg.target.pretrain.batch_size,
        pretrain_lr=cfg.target.pretrain.lr,
        pretrain_seed=cfg.target.pretrain.seed,
        feature_probability=cfg.data.feature_probability,
        data_generation_type=cfg.data.data_generation_type,
        global_batch=cfg.pd.batch_size,
    )
    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg),
        target=target,
        data=None,
        ci_fn=layerwise_mlp_ci_arch(cfg),
        eval=None,
    )


def run_tms_decomposition(built: BuiltRun, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build + pretrain the TMS target, then decompose it through the generic engine.

    The residual entering the decomposed model IS the raw input `x` (no prefix). The
    `eval_fn` reads the `lower_leaky` CI of the single-feature probe and logs the
    ground-truth `IdentityCIError` per site every train-log step (TMS has no separate eval
    cadence — `eval_every = cadence.train_log_every`)."""
    target_cfg = built.target
    assert isinstance(target_cfg, tms.TMSTargetConfig)
    is_main = jax.process_index() == 0

    tms_cfg = tms.TMSConfig(n_features=target_cfg.n_features, n_hidden=target_cfg.n_hidden)
    lm = tms.tms_decomposed_model(tms_cfg, tms.site_specs(tms_cfg, target_cfg.sites))
    if is_main:
        print(f"pretraining TMS target ({target_cfg.pretrain_steps} steps)...", flush=True)
    frozen = tms.replicate_target(
        tms.pretrain_tms_target(
            tms_cfg,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
            target_cfg.pretrain_steps,
            target_cfg.pretrain_batch_size,
            target_cfg.pretrain_lr,
            target_cfg.pretrain_seed,
        ),
        mesh,
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    @jax.jit
    def sample_residual(step_key: jax.Array) -> jax.Array:
        x = tms.sample_sparse_features(
            step_key,
            target_cfg.global_batch,
            target_cfg.n_features,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
        )
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("dp")))

    def sample_batch(step: int) -> jax.Array:
        return sample_residual(random.fold_in(data_key, step))

    @jax.jit
    def single_feature_ci(ci_fn: Any) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        probe = tms.single_feature_probe(target_cfg.n_features)
        ci = ci_fn(lm.read_activations(frozen, probe, ci_fn.input_names))
        return ci.lower, ci.upper

    uv_spec = toy_uv_eval.toy_uv_spec(lm, raw_cfg)

    def eval_fn(state: TrainState, now_step: int) -> dict[str, float]:
        ci_lower, ci_upper = single_feature_ci(state.ci_fn)
        toy_uv_eval.log_uv_figure(
            uv_spec,
            state.components.vu,
            ci_upper,
            now_step,
            wandb_active=built.run.wandb is not None,
        )
        return {
            f"eval/identity_ci_error/{site}": float(tms.identity_ci_error(ci, tolerance=0.1))
            for site, ci in ci_lower.items()
        }

    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        raw_cfg=raw_cfg,
        lm=lm,
        frozen=frozen,
        ci_fn=built.ci_fn,
        data=built.data,
        remat_recon_forwards=built.runtime.remat_recon_forwards,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=built.cadence.train_log_every,
        perf_tokens_per_step=None,
        mesh=mesh,
    )


def main(config: str, group: str | None = None, tags: str | None = None) -> None:
    schema_raw = yaml.safe_load(Path(config).read_text())
    if schema_raw.get("run_id") is None:
        schema_raw["run_id"] = generate_run_id("param_decomp")
    if schema_raw.get("out_dir") is None:
        schema_raw["out_dir"] = str(PARAM_DECOMP_OUT_DIR / "runs")
    if group is not None or tags is not None:
        wandb_cfg = dict(schema_raw.get("wandb") or {})
        if group is not None:
            wandb_cfg["group"] = group
        if tags is not None:
            wandb_cfg["tags"] = tags.split(",")
        schema_raw["wandb"] = wandb_cfg
    built = build_tms_built_run(TMSExperimentConfig(**schema_raw))
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    (built.run.run_dir / "config.yaml").write_text(yaml.safe_dump(schema_raw, sort_keys=False))
    mesh = dp_mesh()
    run_tms_decomposition(built, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
