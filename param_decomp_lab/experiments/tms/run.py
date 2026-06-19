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

from param_decomp.config import (
    ExperimentConfig,
    SharedAlgorithmConfig,
    convert_shared_algorithm_config,
    layerwise_mlp_ci_arch,
    run_instance,
)
from param_decomp.lm import SiteC
from param_decomp.recon import build_recon_terms
from param_decomp.run import run_decomposition_training
from param_decomp.sharding import dp_mesh
from param_decomp.train import TrainState
from param_decomp_config.tms import TMSExperimentConfig
from param_decomp_lab.experiments.tms import model as tms
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


def build_tms_experiment_config(cfg: TMSExperimentConfig) -> ExperimentConfig:
    """Convert the canonical TMS schema to the core `ExperimentConfig` via the shared
    algorithm-config helpers. TMS validates via the in-loop target-CI metric, not the LM
    CEandKLLosses pass, so `eval` must be omitted."""
    assert cfg.pd.identity_decomposition_targets is None, "identity targets unsupported"
    assert cfg.eval is None, (
        "TMS in-loop eval is the standalone target-CI metric (the lab provider), not the LM "
        "CEandKLLosses pass; omit the eval: block"
    )
    site_cs = tms.canonical_site_cs(
        tuple(SiteC(t.module_pattern, t.C) for t in cfg.pd.decomposition_targets)
    )
    shared = convert_shared_algorithm_config(cfg)
    loss_metrics = tuple(cfg.pd.loss_metrics)
    build_recon_terms(
        loss_metrics, tuple(sc.name for sc in site_cs), cfg.pd.n_mask_samples, cfg.pd.sampling
    )
    run_name, run_id, out_dir = run_instance(cfg)
    return _assemble(cfg, site_cs, shared, loss_metrics, run_name, run_id, out_dir)


def _assemble(
    cfg: TMSExperimentConfig,
    site_cs: tuple[SiteC, ...],
    shared: SharedAlgorithmConfig,
    loss_metrics: tuple[Any, ...],
    run_name: str,
    run_id: str,
    out_dir: Path,
) -> ExperimentConfig:
    return ExperimentConfig(
        run_name=run_name,
        run_id=run_id,
        out_dir=out_dir,
        seed=cfg.pd.seed,
        steps=cfg.pd.steps,
        target=tms.TMSTargetConfig(
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
        ),
        data=None,
        loss_metrics=loss_metrics,
        n_mask_samples=cfg.pd.n_mask_samples,
        sampling=cfg.pd.sampling,
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        vu_optimizer=shared.vu_optimizer,
        ci_optimizer=shared.ci_optimizer,
        ci_fn=layerwise_mlp_ci_arch(cfg),
        faith_warmup=shared.faith_warmup,
        cadence=shared.cadence,
        eval=None,
        wandb=cfg.wandb,
        resume_provenance=None,
    )


def run_tms_decomposition(cfg: ExperimentConfig, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build + pretrain the TMS target, then decompose it through the generic engine.

    The residual entering the decomposed model IS the raw input `x` (no prefix). The
    `eval_fn` reads the `lower_leaky` CI of the single-feature probe and logs the
    ground-truth `IdentityCIError` per site every train-log step (TMS has no separate eval
    cadence — `eval_every = cadence.log_every`)."""
    target_cfg = cfg.target
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

    data_key = random.fold_in(random.PRNGKey(cfg.seed), 17)

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
    def single_feature_ci(ci_fn: Any) -> dict[str, jax.Array]:
        probe = tms.single_feature_probe(target_cfg.n_features)
        return ci_fn(lm.site_inputs(frozen, probe)).lower

    def eval_fn(state: TrainState, _now_step: int) -> dict[str, float]:
        ci_lower = single_feature_ci(state.ci_fn)
        return {
            f"eval/identity_ci_error/{site}": float(tms.identity_ci_error(ci, tolerance=0.1))
            for site, ci in ci_lower.items()
        }

    run_decomposition_training(
        cfg=cfg,
        raw_cfg=raw_cfg,
        lm=lm,
        frozen=frozen,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=cfg.cadence.log_every,
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
    cfg = build_tms_experiment_config(TMSExperimentConfig(**schema_raw))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(schema_raw, sort_keys=False))
    mesh = dp_mesh()
    run_tms_decomposition(cfg, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
