"""Run targeted language-model parameter decomposition from one config.

Each step trains once on the fixed target-prompt pool and once on the broader corpus. This
module prepares both streams and reuses the process setup, checkpoint, and shutdown
behavior from ordinary LM training."""

from pathlib import Path
from typing import cast

import jax
import yaml
from jax import random
from jax.sharding import Mesh

from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.core.configs import ResumeProvenance
from param_decomp.core.model import PlacedModel, Positioned
from param_decomp.core.run import (
    MetricsSink,
    install_sigterm_flag,
    run_targeted_decomposition_training,
)
from param_decomp.core.sharding import (
    data_parallel_size,
    hsdp_mesh,
    initialize_topology,
    local_data_parallel_size,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.arithmetic_probe import PromptEncoder
from param_decomp.experiments.lm.config import (
    LMTargetedExperimentConfig,
    build_targeted_experiment_config,
)
from param_decomp.experiments.lm.eval_operations import global_token_batch, make_lm_evaluation
from param_decomp.experiments.lm.load_run import build_target
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    LMTargetedRun,
    TargetConfig,
)
from param_decomp.experiments.lm.targeted_data import build_prompt_pool, pool_batch
from param_decomp.experiments.lm.training import (
    enable_hlo_dump,
    enable_persistent_compilation_cache,
    engine_profiling,
    pin_config_copy,
)
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.infra.run_files import generate_run_id
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards
from param_decomp.targets.glu_transformer import hf_snapshot_dir


def _pool_tokenizer(target: AnyLMTargetConfig, dataset_tokenizer_name: str) -> PromptEncoder:
    """The tokenizer the prompt pool encodes with — necessarily the SAME vocabulary the
    model and the broad stream use. An HF-family target reads its local snapshot (the
    weights load already staged it); the lab-pretrained target names its tokenizer via
    the dataset's own meta."""
    from transformers import AutoTokenizer

    match target:
        case TargetConfig():
            loaded = AutoTokenizer.from_pretrained(
                str(hf_snapshot_dir(target.model_name)), local_files_only=True
            )
        case LlamaSimpleMLPTargetConfig():
            loaded = AutoTokenizer.from_pretrained(dataset_tokenizer_name)
    return cast(PromptEncoder, cast(object, loaded))


def assert_targeted_finetune_structural_compat(
    built: LMTargetedRun, prov: ResumeProvenance, data_root: Path
) -> None:
    """The targeted twin of `training.assert_finetune_structural_compat` (S33): same
    sites (names + C) and ci-fn arch, read from the parent's pinned launch config. The
    parent must itself be a TARGETED run — its pinned config is parsed under
    `LMTargetedExperimentConfig`, so a plain parent fails the parse loudly instead of
    loading a decomposition trained under a different objective shape."""
    raw = yaml.safe_load((prov.parent_run_dir / LAUNCH_CONFIG_FILENAME).read_text())
    parent_cfg = LMTargetedExperimentConfig.model_validate(raw)
    parent = build_targeted_experiment_config(parent_cfg, prov.parent_run_dir.name, data_root)
    parent_sites = tuple((s.name, s.C) for s in parent.target.sites)
    new_sites = tuple((s.name, s.C) for s in built.target.sites)
    assert parent_sites == new_sites, (
        f"fine-tune sites mismatch: parent {parent_sites} != new {new_sites}"
    )
    assert parent.ci_fn == built.ci_fn, (
        f"fine-tune ci-fn arch mismatch: parent {parent.ci_fn} != new {built.ci_fn}"
    )


def train_targeted(
    built: LMTargetedRun,
    cfg: LMTargetedExperimentConfig,
    eval_config: EvalConfig | None,
    model: PlacedModel,
    mesh: Mesh,
) -> None:
    """The targeted LM composition over the engine: the prompt-pool TARGET seam, the
    parquet NON-TARGET seam, and the same domain-bound eval operations as the plain root
    (forward-only diagnostics on the broad eval split)."""
    data = built.data
    train_meta = read_dataset_meta(data.dir)
    eval_meta = read_dataset_meta(data.eval_dir)
    assert train_meta == eval_meta, (
        f"train and eval datasets disagree — {data.dir} is {train_meta}, {data.eval_dir} "
        f"is {eval_meta}. A holdout tokenized differently or at another seq_len makes "
        "every eval number incomparable to the training loss it is read against."
    )
    seq_len = train_meta.seq_len
    n_proc = jax.process_count()
    n_data = data_parallel_size(mesh)
    is_main = jax.process_index() == 0

    # Both streams shard over the data axes and replicate each shard over TP.
    target_batch = built.pd.batch_size
    nontarget_batch = cfg.nontarget.batch_size
    for name, batch in (("pd.batch_size", target_batch), ("nontarget.batch_size", nontarget_batch)):
        assert batch % n_data == 0 and batch >= n_data, (
            f"{name} {batch} must be a positive multiple of data-parallel size {n_data}"
        )

    tokenizer = _pool_tokenizer(built.target, train_meta.tokenizer_name)
    pool = build_prompt_pool(cfg.prompts, tokenizer)
    n_prompts, prompt_len = pool.tokens.shape
    if is_main:
        print(f"target prompt pool: {n_prompts} prompts x {prompt_len} positions", flush=True)

    key = random.PRNGKey(built.pd.seed)
    _, _, run_key = random.split(key, 3)

    schedule = BatchSchedule(scan_shards(data.dir), nontarget_batch, built.pd.seed)
    server = ShardServer(schedule, seq_len, jax.process_index(), n_proc)
    local_data = local_data_parallel_size(mesh)
    assert server.per_process % local_data == 0, (
        server.per_process,
        local_data,
    )

    def pool_global_batch(seed: int, step: int, batch: int) -> jax.Array:
        """One global batch drawn from the prompt pool, sliced to this process's share.

        ONE copy of the process-slice arithmetic: training and eval differ only in
        `(seed, batch)`, and a skew between two hand-written copies would be silently
        wrong data rather than a crash."""
        per_process = batch // n_proc
        rows = pool_batch(pool, seed, step, batch)
        local = rows[jax.process_index() * per_process :][:per_process]
        return global_token_batch(local, mesh, batch)

    def sample_target_batch(step: int) -> jax.Array:
        return pool_global_batch(built.pd.seed, step, target_batch)

    def sample_nontarget_batch(step: int) -> jax.Array:
        return global_token_batch(server.local_batch(step), mesh, nontarget_batch)

    sink = MetricsSink.for_run(built.run, is_main)
    evaluation = None
    if eval_config is not None:
        assert eval_config.every % built.cadence.train_log_every == 0, (
            "eval must land on a train-log step: the tok/s window resets after eval, so a "
            "mid-window eval would corrupt the next step-time estimate"
        )
        eval_target_batch = eval_config.batch_size

        def eval_target_pool_batches(pass_index: int) -> list[jax.Array]:
            """The eval pass's target stream: training's pure `(seed, step)` pool sampler on
            the `seed + 1` stream, so an eval never scores the rows the step just trained."""
            n_batches = eval_config.n_steps
            return [
                pool_global_batch(built.pd.seed + 1, pass_index * n_batches + j, eval_target_batch)
                for j in range(n_batches)
            ]

        evaluation = make_lm_evaluation(
            built,
            eval_config,
            model,
            run_key,
            mesh,
            n_proc,
            sink,
            cfg.runtime.resolved_compiler_options,
            target_pool_batches_for=eval_target_pool_batches,
        )

    run_targeted_decomposition_training(
        pd=built.pd,
        nontarget=cfg.nontarget,
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn=built.ci_fn,
        positions=Positioned(n_positions=prompt_len),
        nontarget_positions=Positioned(n_positions=seq_len),
        remat_recon_forwards=cfg.runtime.remat_recon_forwards,
        remat_ci_fn=cfg.runtime.remat_ci_fn,
        ascend_replicate=cfg.runtime.ascend_replicate,
        sequential_passes=cfg.runtime.sequential_passes,
        compiler_options=cfg.runtime.resolved_compiler_options,
        sample_target_batch=sample_target_batch,
        sample_nontarget_batch=sample_nontarget_batch,
        evaluation=evaluation,
        sink=sink,
        profiling=engine_profiling(cfg.runtime.profiling),
    )


def main(
    config: Path,
    data_root: Path,
    local_device_count: int,
    run_id: str | None = None,
) -> None:
    config = Path(config)
    data_root = Path(data_root)
    if run_id is None:
        # Ad-hoc run-here invocation: mint a fresh identity; `pin_config_copy` below
        # stages the config into the run dir exactly as the launcher would.
        run_id = generate_run_id("param_decomp")
    raw = yaml.safe_load(config.read_text())
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    built = build_targeted_experiment_config(cfg, run_id, data_root)
    runtime = cfg.runtime

    install_sigterm_flag()
    enable_hlo_dump(built.run.run_dir)
    initialize_topology(runtime.world_size, local_device_count)
    mesh = hsdp_mesh(runtime.replicate, runtime.fsdp, runtime.tp)

    if built.run.resume_provenance is not None:
        assert_targeted_finetune_structural_compat(built, built.run.resume_provenance, data_root)

    cache_dir = enable_persistent_compilation_cache(runtime.compilation_cache_dir)

    is_main = jax.process_index() == 0
    if is_main:
        cache_dir.mkdir(parents=True, exist_ok=True)
        built.run.run_dir.mkdir(parents=True, exist_ok=True)
        pin_config_copy(built.run.run_dir, LAUNCH_CONFIG_FILENAME, config)
        print(f"persistent XLA compilation cache: {cache_dir}", flush=True)
        print(
            f"targeted run {built.run.run_name} | {mesh.devices.size} GPU / "
            f"{jax.process_count()} proc | target B={built.pd.batch_size} "
            f"nontarget B={cfg.nontarget.batch_size} "
            f"seq={read_dataset_meta(built.data.dir).seq_len} "
            f"sites={len(built.target.sites)} steps={built.pd.steps}",
            flush=True,
        )

    model = build_target(built.target, mesh, data_root, runtime.sharding)

    train_targeted(built, cfg, cfg.eval, model, mesh)

    if jax.process_count() > 1:
        import jax.experimental.multihost_utils as mhu

        mhu.sync_global_devices("train_done")
        jax.distributed.shutdown()
