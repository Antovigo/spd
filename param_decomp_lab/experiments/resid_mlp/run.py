"""`pd-resid-mlp`: run a ResidualMLP parameter decomposition on CPU.

The SPD/APD residual-stream toy lives lab-side and calls the generic core engine
(`jax_single_pool.run.run_decomposition_training`) as a library. The target pretrains from
scratch in-process (the `act_fn(coeffs·x) + x` read-off objective), then decomposes through
the same engine the LM uses, validating via the ground-truth identity-CI metric.

These toys train in seconds; `pd-resid-mlp` runs synchronously on CPU in the main venv
(no SLURM / `jsp-train` / CUDA). It mints its own `p-<8hex>` run id.
"""

from pathlib import Path
from typing import Any

import fire
import jax
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_single_pool.config import (
    ExperimentConfig,
    SharedAlgorithmConfig,
    convert_shared_algorithm_config,
    run_instance,
    toy_ci_arch,
)
from jax_single_pool.lm import SiteC
from jax_single_pool.recon import build_recon_terms
from jax_single_pool.run import run_decomposition_training
from jax_single_pool.sharding import dp_mesh
from jax_single_pool.train import TrainState

from param_decomp_config.resid_mlp import ResidMLPExperimentConfig
from param_decomp_lab.experiments.resid_mlp import model as resid_mlp
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


def build_resid_mlp_experiment_config(cfg: ResidMLPExperimentConfig) -> ExperimentConfig:
    """Convert the canonical ResidMLP schema to the core `ExperimentConfig` via the shared
    algorithm-config helpers. ResidMLP validates via the in-loop target-CI metric, so
    `eval` must be omitted."""
    assert cfg.pd.identity_decomposition_targets is None, "identity targets unsupported"
    assert cfg.eval is None, (
        "ResidMLP in-loop eval is the standalone target-CI metric (the lab provider), not the "
        "LM CEandKLLosses pass; omit the eval: block"
    )
    site_cs = resid_mlp.canonical_site_cs(
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
    cfg: ResidMLPExperimentConfig,
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
        target=resid_mlp.ResidMLPTargetConfig(
            n_features=cfg.target.n_features,
            d_embed=cfg.target.d_embed,
            d_mlp=cfg.target.d_mlp,
            n_layers=cfg.target.n_layers,
            act_fn_name=cfg.target.act_fn_name,
            in_bias=cfg.target.in_bias,
            out_bias=cfg.target.out_bias,
            fixed_identity_embedding=cfg.target.fixed_identity_embedding,
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
        ci_fn=toy_ci_arch(cfg),
        faith_warmup=shared.faith_warmup,
        cadence=shared.cadence,
        eval=None,
        wandb=cfg.wandb,
        resume_provenance=None,
    )


def run_resid_mlp_decomposition(cfg: ExperimentConfig, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build + pretrain the ResidMLP target, then decompose it through the generic engine.

    The residual entering the decomposed model is `x @ W_E` (the prefix `W_E` is carried
    inside the frozen target). The `eval_fn` reads the `lower_leaky` CI of the
    single-feature probe (embedded through `W_E`) and logs the ground-truth `IdentityCIError`
    per site every train-log step (`eval_every = cadence.log_every`)."""
    target_cfg = cfg.target
    assert isinstance(target_cfg, resid_mlp.ResidMLPTargetConfig)
    is_main = jax.process_index() == 0

    resid_cfg = resid_mlp.ResidMLPConfig(
        n_features=target_cfg.n_features,
        d_embed=target_cfg.d_embed,
        d_mlp=target_cfg.d_mlp,
        n_layers=target_cfg.n_layers,
        act_fn_name=target_cfg.act_fn_name,
        in_bias=target_cfg.in_bias,
        out_bias=target_cfg.out_bias,
        fixed_identity_embedding=target_cfg.fixed_identity_embedding,
    )
    lm = resid_mlp.resid_mlp_decomposed_model(
        resid_cfg, resid_mlp.site_specs(resid_cfg, target_cfg.sites)
    )
    if is_main:
        print(f"pretraining ResidMLP target ({target_cfg.pretrain_steps} steps)...", flush=True)
    frozen = resid_mlp.replicate_target(
        resid_mlp.pretrain_resid_mlp_target(
            resid_cfg,
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
        x = resid_mlp.sample_sparse_features(
            step_key,
            target_cfg.global_batch,
            target_cfg.n_features,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
        )
        residual = resid_mlp.resid_mlp_input_residual(frozen, x)
        return jax.lax.with_sharding_constraint(residual, NamedSharding(mesh, P("dp")))

    def sample_batch(step: int) -> jax.Array:
        return sample_residual(random.fold_in(data_key, step))

    @jax.jit
    def single_feature_ci(ci_fn: Any) -> dict[str, jax.Array]:
        resid = resid_mlp.single_feature_probe(target_cfg.n_features) @ frozen.W_E
        return ci_fn(lm.site_inputs(frozen, resid)).lower

    def eval_fn(state: TrainState, _now_step: int) -> dict[str, float]:
        ci_lower = single_feature_ci(state.ci_fn)
        return {
            f"eval/identity_ci_error/{site}": float(resid_mlp.identity_ci_error(ci, tolerance=0.1))
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
    cfg = build_resid_mlp_experiment_config(ResidMLPExperimentConfig(**schema_raw))
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.yaml").write_text(yaml.safe_dump(schema_raw, sort_keys=False))
    mesh = dp_mesh()
    run_resid_mlp_decomposition(cfg, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
