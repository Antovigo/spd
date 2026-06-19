"""The LM decomposition composition root: wrapper YAML -> full SPEC-compliant run on a
vendored target.

    python -m param_decomp_lab.experiments.lm.run <wrapper.yaml>   # normally via pd-lm,
        # which stamps run_id into the workspace copy; re-running resumes in place

This is the LM I/O layer over the generic core engine
(`param_decomp.run.run_decomposition_training`): read the run YAML, build the target +
prefix, harvest the residual from the frozen prefix over a parquet token batch
(`sample_batch`), build the CEandKL / CI-L0 / PGD / attn-patterns / slow `eval_fn`, then
call the engine. Process setup (`init_distributed`, the SIGTERM flag, the persistent XLA
compilation cache, HF http hardening), config pinning, and SLURM-requeue shutdown all live
here. The toy domains mirror this file under `experiments/{tms,resid_mlp}/run.py`.

Multi-process: launched one process per GPU under SLURM (`init_distributed`); every
process computes the same global schedule and contributes its local batch slice.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from param_decomp import llama_simple_mlp
from param_decomp.attn_patterns_eval import (
    accumulate_attn_patterns,
    attn_pattern_for,
    attn_patterns_log_entries,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.config import DataConfig, ExperimentConfig
from param_decomp.configs import ResumeProvenance
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.eval import make_eval_step
from param_decomp.hf_http import configure_hf_http_retries
from param_decomp.llama8b import (
    first_decomposed_layer,
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    load_prefix_from_hf,
    load_target_from_hf,
    prefix_residual,
)
from param_decomp.llama8b_sharding import replicate_target
from param_decomp.lm import DecomposedModel
from param_decomp.run import (
    SlowEvalRenderer,
    install_sigterm_flag,
    run_decomposition_training,
    sigterm_received,
    slow_eval_due,
)
from param_decomp.sharding import dp_mesh, init_distributed
from param_decomp.slow_eval import (
    IDENTITY_CI_ERROR_TOLERANCE,
    PositionCI,
    accumulate_position_ci,
    accumulate_site_reductions,
    compute_hidden_acts_metrics,
    compute_identity_ci_errors,
    eval_metrics_from_run_dir,
    make_position_ci_step,
    make_slow_eval_step,
    resolve_permutation_metrics,
)
from param_decomp.target_aliases import AnyFrozenTarget, AnyPrefix
from param_decomp.train import TrainState
from param_decomp_lab.experiments.lm.config import (
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    load_config,
    load_run_dir_config,
)


def _enable_persistent_compilation_cache(out_dir: Path) -> Path:
    """Cache compiled XLA executables to a shared-FS dir reused across runs/requeues.

    The ~24-min compile of the chunkwise step is keyed by HLO + backend + topology +
    jax/xla version, so a matching re-compile (requeue, or a fresh run at the same
    config+topology) loads from disk in seconds. The dir is a SIBLING of `runs/` (not
    per-run, not inside the immutable per-run workspace) so every run shares it; all 8N
    ranks point at the same shared-FS path. Only process 0 writes (jax gates the write on
    `process_id == 0` to avoid shared-FS write contention); every rank reads. Must run
    after `init_distributed` (the rank gate reads the distributed state) and before the
    first compile."""
    cache_dir = out_dir.parent / "xla_compilation_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 60.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir


def _global_token_batch(local: np.ndarray, mesh: Mesh, global_batch: int) -> jax.Array:
    sharding = NamedSharding(mesh, P("dp"))
    return jax.make_array_from_process_local_data(sharding, local, (global_batch, local.shape[1]))


def assert_finetune_structural_compat(cfg: ExperimentConfig, prov: ResumeProvenance) -> None:
    """Fine-tune requires the parent's decomposition STRUCTURE to match the new config's:
    same sites (names + C) and same ci-fn arch. A changed C / layers / target / ci-fn is a
    different-shaped decomposition and is NOT a fine-tune (the parent's V/U + ci_fn would
    not load onto the new reference). Only LR / coeffs / eps / seq / batch / steps may
    change. Read from the parent's pinned `config.yaml` so the failure is a readable config
    diff, not an opaque orbax tree mismatch."""
    parent_cfg = load_run_dir_config(prov.parent_run_dir)
    parent_sites = tuple((s.name, s.C) for s in parent_cfg.target.sites)
    new_sites = tuple((s.name, s.C) for s in cfg.target.sites)
    assert parent_sites == new_sites, (
        f"fine-tune sites mismatch: parent {parent_sites} != new {new_sites}"
    )
    assert parent_cfg.ci_fn == cfg.ci_fn, (
        f"fine-tune ci-fn arch mismatch: parent {parent_cfg.ci_fn} != new {cfg.ci_fn}"
    )


def train(
    cfg: ExperimentConfig,
    raw_cfg: dict[str, object],
    lm: DecomposedModel,
    frozen: AnyFrozenTarget,
    prefix: AnyPrefix,
    prefix_residual_fn: Callable[[Any, Any], jax.Array],
    mesh: Mesh,
) -> None:
    """The LM composition over the generic engine: a parquet `sample_batch` (harvest the
    residual from the frozen prefix) and the CEandKL / CI-L0 / PGD / attn-patterns
    `eval_fn`."""
    data = cfg.data
    assert isinstance(data, DataConfig), "train() is the LM (parquet) data path"
    n_proc = jax.process_count()
    ndev = mesh.devices.size
    assert data.global_batch % ndev == 0, (data.global_batch, ndev)
    is_main = jax.process_index() == 0

    key = random.PRNGKey(cfg.seed)
    _, _, run_key = random.split(key, 3)

    def _harvest(prefix_weights: Any, inputs: Any) -> jax.Array:
        residual = prefix_residual_fn(prefix_weights, inputs)
        return jax.lax.with_sharding_constraint(residual, NamedSharding(mesh, P("dp")))

    harvest = jax.jit(_harvest)

    schedule = BatchSchedule(scan_shards(data.dir), data.global_batch, cfg.seed)
    server = ShardServer(schedule, data.seq_len, jax.process_index(), n_proc)
    assert server.per_process % jax.local_device_count() == 0, (
        server.per_process, jax.local_device_count(),
    )  # fmt: skip

    def sample_batch(step: int) -> jax.Array:
        token_ids = _global_token_batch(server.local_batch(step), mesh, data.global_batch)
        return harvest(prefix, token_ids)

    eval_fn = None
    eval_every = cfg.steps + 1  # unreachable cadence when eval is disabled
    if cfg.eval is not None:
        assert cfg.eval.every % cfg.cadence.log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        assert cfg.eval.slow_every % cfg.eval.every == 0, (
            "slow_every must be a multiple of every: the slow tier reuses the fast eval "
            "pass's batches, so it can only fire on a fast-eval step"
        )
        eval_every = cfg.eval.every
        eval_fn = _make_lm_eval_fn(cfg, lm, frozen, prefix, harvest, run_key, mesh, n_proc, is_main)

    run_decomposition_training(
        cfg=cfg,
        raw_cfg=raw_cfg,
        lm=lm,
        frozen=frozen,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=eval_every,
        perf_tokens_per_step=data.global_batch * data.seq_len,
        mesh=mesh,
    )


def _make_lm_eval_fn(
    cfg: ExperimentConfig,
    lm: DecomposedModel,
    frozen: AnyFrozenTarget,
    prefix: AnyPrefix,
    harvest: Callable[[Any, Any], jax.Array],
    run_key: PRNGKeyArray,
    mesh: Mesh,
    n_proc: int,
    is_main: bool,
) -> Callable[[TrainState, int], dict[str, float]]:
    """The LM in-loop eval pass closure (CEandKL / CI-L0 / PGD / attn-patterns), keyed
    deterministically off `(run_key, now_step)` so it is bit-identical to the pre-engine
    inline loop. Mirrors the torch `eval_split: train` stream: an independent reader over
    the SAME corpus (own seed), advanced one block of `n_steps` batches per eval pass."""
    assert cfg.eval is not None
    data = cfg.data
    assert isinstance(data, DataConfig)
    eval_schedule = BatchSchedule(scan_shards(data.dir), cfg.eval.batch_size, cfg.seed + 1)
    eval_server = ShardServer(eval_schedule, data.seq_len, jax.process_index(), n_proc)
    assert eval_server.per_process % jax.local_device_count() == 0, (
        eval_server.per_process, jax.local_device_count(),
    )  # fmt: skip
    eval_pgd = (cfg.eval.pgd.n_steps, cfg.eval.pgd.step_size) if cfg.eval.pgd else None
    eval_step_fn = make_eval_step(
        lm,
        cfg.eval.rounding_threshold,
        cfg.eval.ci_alive_threshold,
        cfg.eval.l0_groups,
        eval_pgd,
        mesh,
    )
    attn_steps: dict[str, Any] = {}
    if cfg.eval.attn_patterns is not None:
        pattern_fn = attn_pattern_for(frozen)
        if cfg.eval.attn_patterns.ci_masked:
            attn_steps["CIMaskedAttnPatternsReconLoss"] = make_ci_attn_patterns_step(lm, pattern_fn)
        if cfg.eval.attn_patterns.stochastic:
            attn_steps["StochasticAttnPatternsReconLoss"] = make_stochastic_attn_patterns_step(
                lm, pattern_fn, cfg.n_mask_samples, cfg.sampling
            )

    slow_eval_step = make_slow_eval_step(lm, cfg.eval.ci_alive_threshold)
    slow_renderer = SlowEvalRenderer(is_main)
    # The CI-heatmap / permutation / UV / identity-error metrics read off the run's typed
    # `eval.metrics` (re-validated from the pinned config.yaml: the trainer's `EvalConfig`
    # drops the raw metric list). config.yaml is pinned before train().
    perm_spec = resolve_permutation_metrics(lm.site_names, eval_metrics_from_run_dir(cfg.run_dir))
    want_position_ci = perm_spec.any_plots or perm_spec.any_identity_error
    position_ci_step = make_position_ci_step(lm) if want_position_ci else None

    def eval_fn(state: TrainState, now_step: int) -> dict[str, float]:
        assert cfg.eval is not None
        eval_pass_index = now_step // cfg.eval.every
        # uniform-average of per-batch scalars; mean-safe vs torch's accumulate-then-
        # compute() ONLY because every emitted key is a per-batch reduction that torch also
        # averages across batches AND eval batches are uniform (B, T). See eval.py's module
        # docstring for the per-key parity argument (cites SPEC S8/D2).
        metric_sums: dict[str, jax.Array] = {}
        eval_residuals: list[jax.Array] = []
        for j in range(cfg.eval.n_steps):
            if sigterm_received():
                break
            eval_tokens = _global_token_batch(
                eval_server.local_batch(eval_pass_index * cfg.eval.n_steps + j),
                mesh,
                cfg.eval.batch_size,
            )
            eval_residual = harvest(prefix, eval_tokens)
            eval_residuals.append(eval_residual)
            # fold values >= cfg.steps never collide with the train step keys
            eval_key = random.fold_in(run_key, cfg.steps + eval_pass_index * cfg.eval.n_steps + j)
            eval_metrics = eval_step_fn(
                state.components, state.ci_fn, frozen, eval_tokens, eval_residual, eval_key
            )
            for k, v in eval_metrics.items():
                metric_sums[k] = metric_sums.get(k, jnp.zeros(())) + v
        eval_record = {f"eval/{k}": float(v) / cfg.eval.n_steps for k, v in metric_sums.items()}
        for class_name, attn_step in attn_steps.items():
            # token-weighted (Σ sum_kl / Σ n), NOT the uniform per-batch average above — KL
            # is summed over distributions, divided by their count.
            attn_key = random.fold_in(run_key, 2 * cfg.steps + eval_pass_index)
            reductions = accumulate_attn_patterns(
                attn_step, state.components, state.ci_fn, frozen, eval_residuals, attn_key
            )
            eval_record |= {
                f"eval/loss/{k}": v
                for k, v in attn_patterns_log_entries(class_name, reductions).items()
            }
        slow_due = slow_eval_due(
            now_step, cfg.eval.every, cfg.eval.slow_every, cfg.eval.slow_on_first_step
        )
        if eval_residuals and slow_due and not sigterm_received():
            # SLOW/PLOT TIER (SPEC S28/S29). The COLLECTIVE part runs in lockstep on every
            # rank — `accumulate_site_reductions` / `compute_hidden_acts_metrics` pull
            # C-sharded reductions to numpy, whose `np.asarray` triggers the all-gather all
            # ranks must join. It reuses the eval batches already loaded above. The
            # hidden-acts scalars ride the live `_step` axis through `eval_record`; the
            # figures' pure-host render + wandb.log happen OFF the loop on rank 0.
            site_reductions = accumulate_site_reductions(
                slow_eval_step, state.ci_fn, frozen, eval_residuals, cfg.eval.slow_n_batches_accum
            )
            hidden_acts_key = random.fold_in(run_key, 3 * cfg.steps + eval_pass_index)
            hidden_acts = compute_hidden_acts_metrics(
                lm, state, frozen, eval_residuals, cfg.n_mask_samples, cfg.sampling, hidden_acts_key
            )
            eval_record |= {f"eval/slow/loss/{k}": v for k, v in hidden_acts.items()}
            # The position-CI all-gather is ALSO collective (every rank joins it), gated on
            # the config naming a CI-heatmap / permutation / identity-error metric. The
            # heatmap FIGURES render off-loop on rank 0; the IdentityCIError SCALARS log
            # synchronously on the live `_step` (cheap + must stay `_step`-monotonic).
            position_ci: dict[str, PositionCI] | None = None
            if position_ci_step is not None:
                position_ci = accumulate_position_ci(
                    position_ci_step, state.ci_fn, frozen, eval_residuals
                )
                identity_ci_errors = compute_identity_ci_errors(
                    perm_spec, position_ci, IDENTITY_CI_ERROR_TOLERANCE
                )
                eval_record |= {f"eval/slow/{k}": v for k, v in identity_ci_errors.items()}
            # `UVPlots` needs the C-sharded V/U gathered to host (collective `np.asarray`).
            # This NAIVE gather is small-scale-only — it OOMs / breaks at production C BY
            # DESIGN (per Oli); gated on the config naming UVPlots so it costs nothing
            # otherwise. The component column order reuses the position-CI permutation.
            components: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
            if perm_spec.want_uv_plots:
                components = {
                    name: (np.asarray(V), np.asarray(U))
                    for name, (V, U) in state.components.vu.items()
                }
            slow_renderer.submit(site_reductions, perm_spec, position_ci, components, now_step)
        if is_main:
            headline = {
                k: eval_record[f"eval/{k}"]
                for k in ("ce_kl/kl_ci_masked", "ce_kl/ce_unrecovered_ci_masked")
            }
            print(f"[eval @ {now_step}] {headline}", flush=True)
        return eval_record

    return eval_fn


def _pin_config_copy(run_dir: Path, name: str, source: Path) -> None:
    """First run copies `source` into the run dir; resumes byte-compare against it."""
    copy = run_dir / name
    if copy.exists():
        assert copy.read_text() == source.read_text(), (
            f"{copy} differs from {source} — refusing to resume with a changed config"
        )
    else:
        copy.write_text(source.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    install_sigterm_flag()
    init_distributed()
    # Harden the cold-cache HF weight load against the 8N-rank startup burst before any
    # per-rank Hub call (no-op when huggingface_hub is absent / cache is pre-warmed).
    configure_hf_http_retries()
    mesh = dp_mesh()

    cfg, raw_cfg = load_config(args.config)
    if cfg.resume_provenance is not None:
        assert_finetune_structural_compat(cfg, cfg.resume_provenance)

    cache_dir = _enable_persistent_compilation_cache(cfg.out_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        _pin_config_copy(cfg.run_dir, "config.yaml", args.config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        site_summary = " ".join(f"{s.name}:C{s.C}" for s in cfg.target.sites)
        assert isinstance(cfg.data, DataConfig)
        print(
            f"run {cfg.run_name} | {mesh.devices.size} GPU / {jax.process_count()} proc | "
            f"B={cfg.data.global_batch} seq={cfg.data.seq_len} "
            f"sites=[{site_summary}] steps={cfg.steps}",
            flush=True,
        )

    frozen: AnyFrozenTarget
    prefix: AnyPrefix
    prefix_residual_fn: Callable[[Any, Any], jax.Array]
    match cfg.target:
        case TargetConfig():
            llama_cfg = llama31_8b_config()
            lm = llama_decomposed_lm(llama_cfg, llama_site_specs(llama_cfg, cfg.target.sites))
            first_layer = first_decomposed_layer(lm.site_names)
            frozen = replicate_target(
                load_target_from_hf(cfg.target.model_name, llama_cfg, first_layer), mesh
            )
            prefix = jax.device_put(
                load_prefix_from_hf(cfg.target.model_name, llama_cfg, first_layer),
                NamedSharding(mesh, P()),
            )
            prefix_residual_fn = prefix_residual
        case LlamaSimpleMLPTargetConfig():
            cache_dir = llama_simple_mlp.pretrain_cache_dir(cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            lm = llama_simple_mlp.llama_simple_mlp_decomposed_lm(
                simple_cfg, llama_simple_mlp.site_specs(simple_cfg, cfg.target.sites)
            )
            first_layer = llama_simple_mlp.first_decomposed_layer(lm.site_names)
            frozen = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_target_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            prefix = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_prefix_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            prefix_residual_fn = llama_simple_mlp.prefix_residual
        case _:
            raise AssertionError(
                f"the LM composition is LM-only; got target {type(cfg.target).__name__}"
            )

    train(cfg, raw_cfg, lm, frozen, prefix, prefix_residual_fn, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
