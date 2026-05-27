"""LM PD experiment: YAML -> `Trainer` glue + `SavedLMRun` reload + resumption.

Both the fresh-run path (`main`) and the reload path (`SavedLMRun`) share the
module-level `build_target` / `build_lm_loader` / `make_run_batch`. The resume
path (`main --resume <yaml>`) reads a parent run's `run_meta.yaml` plus
`training_<step>.pth`, rebuilds a `Trainer` via `Trainer.from_snapshot`, and
continues training.

Run via `pd-lm path/to/config.yaml` (fresh), `pd-lm --resume path/to/resume.yaml`
(resume), or `pd-lm --eval-only --resume <run_path> [--step N]` (one-shot eval
against a saved checkpoint; logs into the parent's wandb run at step N). Pass
`--dp N` to submit a DDP SLURM job (single-node for N <= 8, multi-node for N > 8
— N must then be a multiple of 8). For local DDP, invoke directly via
`torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run config.yaml`.
"""

import gc
import importlib
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import fire
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from pydantic import Discriminator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.output import collect_metric_outputs
from param_decomp.optimize import EvalLoop, Trainer, _build_metric_context
from param_decomp.three_pool import ThreePoolConfig, ThreePoolTrainer
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader
from param_decomp.training_state import ThreePoolTrainingState, TrainingState
from param_decomp.two_pool import TwoPoolConfig, TwoPoolTrainer
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.component_model_io import load_component_model
from param_decomp_lab.distributed import (
    ensure_cached_and_call,
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import (
    RUN_META_FILENAME,
    ExperimentConfig,
    init_pd_run,
)
from param_decomp_lab.infra.ddp_launch import build_ddp_launch
from param_decomp_lab.infra.git import create_git_snapshot
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id, resolve_run_files
from param_decomp_lab.infra.settings import (
    DEFAULT_PARTITION_NAME,
    PARAM_DECOMP_OUT_DIR,
    REPO_ROOT,
)
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job
from param_decomp_lab.infra.wandb import (
    get_wandb_entity,
    parse_wandb_run_path,
    try_wandb,
)
from param_decomp_lab.resumption import (
    ResumeConfig,
    ResumeProvenance,
    read_training_snapshot,
    resolve_step,
    write_provenance,
)
from param_decomp_lab.run_sink import OnePoolSink, ThreePoolSink
from param_decomp_lab.seed import set_seed


def _resolve_class(fqn: str) -> type:
    """Load a class from a fully-qualified name, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = fqn.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model.

    `run_path` accepts any form `PretrainRunInfo.from_path` does — compact W&B
    (`entity/project/runId`), full W&B (`entity/project/runs/runId`), or a local
    checkpoint path.
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: ModelPath


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into a vendored `param_decomp_lab.experiments.lm.pretrain.models.*`
    architecture via `<class>.from_hf_pretrained(<hub_id>)`.

    Useful when the decomposition target needs structural changes vs HF — e.g.
    `GPT2Simple`'s separate q/k/v projections vs HF's fused `c_attn`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must expose `from_hf_pretrained`
    model_name: str  # HF hub id


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """Config for the LM target model and how to extract the prediction tensor.

    `output_extract` (passed to `make_run_batch`) pulls the prediction tensor out of the
    model's forward output (default `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"
    activation_checkpointing: bool = False
    """If True and the target exposes `enable_activation_checkpointing()`, turn on
    per-block gradient checkpointing on the frozen target forward. Trades ~33% extra
    compute for ~10–15x less stored activation memory under 2-pool — the main lever for
    raising `b_per_rank` on deep targets."""


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    two_pool: TwoPoolConfig | None = None
    """When set, training runs under the 2-pool strategy
    (:class:`param_decomp.two_pool.TwoPoolTrainer`) instead of single-process
    :class:`param_decomp.optimize.Trainer`. Mutually exclusive with ``three_pool``."""

    three_pool: ThreePoolConfig | None = None
    """When set, training runs under the 3-pool strategy
    (:class:`param_decomp.three_pool.ThreePoolTrainer`). Mutually exclusive with
    ``two_pool``."""


def build_target(target_cfg: LMTargetConfig) -> nn.Module:
    """Load the LM target model in eval mode, dispatching on `target_cfg.spec.kind`."""
    spec = target_cfg.spec
    cls = _resolve_class(spec.model_class)
    match spec:
        case HFTarget():
            target_model = ensure_cached_and_call(cls.from_pretrained, spec.model_name)
        case PretrainedTarget():
            from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo

            run_info = ensure_cached_and_call(PretrainRunInfo.from_path, spec.run_path)
            # Older PretrainRunInfo objects predate model_type; default it from the model class.
            if "model_type" not in run_info.model_config_dict:
                run_info.model_config_dict["model_type"] = spec.model_class.rsplit(".", 1)[-1]
            target_model = cls.from_run_info(run_info)
        case HFWeightsInVendored():
            assert hasattr(cls, "from_hf_pretrained"), (
                f"HFWeightsInVendored target requires {spec.model_class!r} to expose a "
                "`from_hf_pretrained` classmethod"
            )
            target_model = ensure_cached_and_call(cls.from_hf_pretrained, spec.model_name)
    if target_cfg.activation_checkpointing:
        assert hasattr(target_model, "enable_activation_checkpointing"), (
            f"activation_checkpointing=True but {type(target_model).__name__} has no "
            "`enable_activation_checkpointing()` method"
        )
        target_model.enable_activation_checkpointing()
    target_model.eval()
    return target_model


def build_lm_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    split: Literal["train", "eval"],
    device: str,
    batch_size: int,
    dist_state: DistributedState | None = None,
    seed: int | None = None,
) -> DataLoader[Any]:
    """LM `DataLoader` for the requested split.

    The eval seed is offset by 1 so eval shuffles differently from train when both come
    from the same `pd_config.seed`.
    """
    del target_cfg, device
    effective_seed = (seed or 0) + (1 if split == "eval" else 0)
    split_name = data_cfg.eval_split if split == "eval" else data_cfg.train_split
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=split_name,
        batch_size=rank_batch_size(batch_size, dist_state, label=f"{split}_batch_size"),
        seed=effective_seed,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def make_run_batch(target_cfg: LMTargetConfig) -> RunBatch:
    return _make_run_batch(target_cfg.output_extract)


@dataclass(frozen=True)
class SavedLMRun:
    """Handle to a completed LM PD run on disk or in W&B."""

    cfg: LMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedLMRun":
        """Resolve a run directory or W&B path into a fully-validated `SavedLMRun`."""
        files = resolve_run_files(
            path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return cls(
            cfg=LMExperimentConfig.from_file(files.config_path),
            checkpoint_path=files.checkpoint_path,
        )

    def load_model(self) -> ComponentModel:
        return load_component_model(
            pd_config=self.cfg.pd,
            checkpoint_path=self.checkpoint_path,
            target_model=build_target(self.cfg.target),
            run_batch=make_run_batch(self.cfg.target),
        )


@with_distributed_cleanup
def main(
    config_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    eval_only: bool = False,
    step: int | None = None,
    eval_config: str | Path | None = None,
    group: str | None = None,
    tags: str | None = None,
    dp: int | None = None,
    partition: str | None = DEFAULT_PARTITION_NAME,
    time: str = "72:00:00",
    job_name: str = "pd-lm",
    no_snapshot: bool = False,
    run_id: str | None = None,
) -> None:
    """Run an LM PD experiment end-to-end.

    Args:
        config_path: YAML for a fresh run. Required when not resuming and not eval-only.
        resume: Path to a `ResumeConfig` YAML pointing at a prior run. When set,
            the parent's `run_meta.yaml` is the source of cfg truth; a new
            `run_id` + sibling `resume_provenance.yaml` are written.

            When combined with `--eval-only`, this is instead a `SavedLMRun` path
            (wandb URL / `entity/project/runId` / `p-xxxxxxxx` / local dir) — the
            checkpoint is loaded, one eval pass is run, and results are logged
            into the parent run's wandb timeline at `step` (default: latest).
        eval_only: If True, skip training and run a one-shot eval pass against
            the saved checkpoint pointed at by `--resume`.
        step: Which checkpoint to evaluate when `--eval-only`. Default: latest.
        group / tags: wandb-only (no-ops without `wandb:`).
        dp / partition / time / job_name / no_snapshot / run_id: SLURM submission
            knobs. Passing `--dp N` outside torchrun submits a SLURM job: single-node
            for N <= 8, multi-node for N > 8 (N must be a multiple of 8). For local
            DDP, invoke directly via
            `torchrun --standalone --nproc_per_node=N -m param_decomp_lab.experiments.lm.run`.
    """
    if eval_only:
        assert resume is not None, "--eval-only requires --resume <run_path>"
        assert config_path is None, "--eval-only and config_path are mutually exclusive"
        if dp is not None and os.environ.get("WORLD_SIZE") is None:
            _submit_slurm_eval_only(
                resume,
                step=step,
                eval_config=eval_config,
                dp=dp,
                group=group,
                tags=tags,
                partition=partition,
                time=time,
                job_name=job_name,
                no_snapshot=no_snapshot,
            )
            return
        _eval_only_main(resume, step=step, eval_config_override=eval_config, group=group, tags=tags)
        return

    if dp is not None and os.environ.get("WORLD_SIZE") is None:
        assert config_path is not None, "--dp SLURM submission requires a config_path"
        _submit_slurm(
            config_path,
            dp=dp,
            group=group,
            tags=tags,
            partition=partition,
            time=time,
            job_name=job_name,
            no_snapshot=no_snapshot,
            run_id=run_id,
        )
        return

    if resume is not None:
        assert config_path is None, "pass either config_path or --resume, not both"
        _resume_main(Path(resume), group=group, tags=tags, run_id=run_id)
    else:
        assert config_path is not None, "must provide either config_path or --resume"
        _fresh_main(Path(config_path), group=group, tags=tags, run_id=run_id)


def _agree_on_run_id_three_pool(run_id: str | None, dist_state: DistributedState | None) -> str:
    """Broadcast (or generate-then-broadcast) a single run id across all ranks.

    3-pool snapshot uses a file-based gather under
    ``PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/.snapshot_scratch/``; every
    rank must compute the same path.
    """
    if is_main_process() and run_id is None:
        run_id = generate_run_id("param_decomp")
    if dist_state is not None:
        objs: list[str | None] = [run_id]
        dist.broadcast_object_list(objs, src=0)
        run_id = objs[0]
    assert run_id is not None
    return run_id


def _fresh_main(
    config_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Fresh-run path: parse YAML, dispatch on pool config, train from step 0."""
    cfg = LMExperimentConfig.from_file(config_path)
    assert not (cfg.two_pool is not None and cfg.three_pool is not None), (
        "two_pool and three_pool are mutually exclusive; set only one."
    )

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
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
    is_multi_pool = cfg.two_pool is not None or cfg.three_pool is not None
    train_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        # Multi-pool requires the full global batch on every rank — each pool
        # slices it locally. Single-pool uses standard DistributedSampler.
        dist_state=None if is_multi_pool else dist_state,
        seed=cfg.pd.seed,
    )

    if cfg.three_pool is not None:
        run_id = _agree_on_run_id_three_pool(run_id, dist_state)
        scratch_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id / ".snapshot_scratch"
        three_sink = init_pd_run(
            cfg, sink_class=ThreePoolSink, group=group, tags=tags, run_id=run_id
        )
        # Multi-pool eval mirrors the train data-handling contract: full eval
        # batch on every rank, sliced internally by each pool. So we pass
        # dist_state=None.
        three_eval_loop = _build_eval_loop(cfg, device, dist_state=None)
        try:
            three_trainer = ThreePoolTrainer(
                target_model=target_model,
                run_batch=make_run_batch(cfg.target),
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
                three_pool_config=cfg.three_pool,
            )
            three_trainer.run(
                train_loader,
                three_sink,
                cfg.cadence,
                scratch_dir=scratch_dir,
                eval_loop=three_eval_loop,
            )
        finally:
            three_sink.finish()
        return

    one_sink = init_pd_run(cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id)
    eval_loop = _build_eval_loop(cfg, device, dist_state) if cfg.two_pool is None else None
    try:
        if cfg.two_pool is not None:
            two_trainer = TwoPoolTrainer(
                target_model=target_model,
                run_batch=make_run_batch(cfg.target),
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
                two_pool_config=cfg.two_pool,
                cadence=cfg.cadence,
            )
            two_trainer.run(train_loader, one_sink, cfg.cadence)
        else:
            trainer = Trainer(
                target_model=target_model,
                run_batch=make_run_batch(cfg.target),
                reconstruction_loss=recon_loss_kl,
                pd_config=cfg.pd,
                runtime_config=cfg.runtime,
            )
            trainer.run(train_loader, one_sink, cfg.cadence, eval_loop)
    finally:
        one_sink.finish()


def _resume_main(
    resume_cfg_path: Path,
    *,
    group: str | None,
    tags: str | None,
    run_id: str | None,
) -> None:
    """Resume-run path: read parent `run_meta.yaml` + `training_<step>.pth`,
    rebuild trainer via `from_snapshot`, continue training."""
    resume_cfg = ResumeConfig.from_file(resume_cfg_path)
    parent_cfg = LMExperimentConfig.from_file(resume_cfg.from_run / RUN_META_FILENAME)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Resuming from {resume_cfg.from_run} @ step {resume_cfg.step}")
    set_seed(parent_cfg.pd.seed)
    device = get_device()

    effective_cfg = parent_cfg.model_copy(
        update={
            "runtime": parent_cfg.runtime.model_copy(
                update={
                    "device": device,
                    "dp": dist_state.world_size if dist_state is not None else None,
                }
            ),
        }
    )

    resolved_step = resolve_step(resume_cfg.from_run, resume_cfg.step)
    snapshot = read_training_snapshot(resume_cfg.from_run, resolved_step)
    snapshot.runtime_config["device"] = device

    target_model = build_target(effective_cfg.target)
    is_multi_pool = effective_cfg.two_pool is not None or effective_cfg.three_pool is not None
    train_loader = build_lm_loader(
        effective_cfg.target,
        effective_cfg.data,
        split="train",
        device=device,
        batch_size=effective_cfg.pd.batch_size,
        dist_state=None if is_multi_pool else dist_state,
        seed=effective_cfg.pd.seed,
    )
    run_batch = make_run_batch(effective_cfg.target)

    if effective_cfg.three_pool is not None:
        run_id = _agree_on_run_id_three_pool(run_id, dist_state)
        scratch_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id / ".snapshot_scratch"
        three_sink = init_pd_run(
            effective_cfg, sink_class=ThreePoolSink, group=group, tags=tags, run_id=run_id
        )
        if three_sink.out_dir is not None:
            write_provenance(
                three_sink.out_dir,
                ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
            )
        three_eval_loop = _build_eval_loop(effective_cfg, device, dist_state=None)
        try:
            assert isinstance(snapshot, ThreePoolTrainingState), (
                f"3-pool resume needs ThreePoolTrainingState; got {type(snapshot).__name__}"
            )
            three_trainer = ThreePoolTrainer.from_snapshot(
                snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
            )
            three_trainer.run(
                train_loader,
                three_sink,
                effective_cfg.cadence,
                scratch_dir=scratch_dir,
                eval_loop=three_eval_loop,
            )
        finally:
            three_sink.finish()
        return

    one_sink = init_pd_run(
        effective_cfg, sink_class=OnePoolSink, group=group, tags=tags, run_id=run_id
    )
    if one_sink.out_dir is not None:
        write_provenance(
            one_sink.out_dir,
            ResumeProvenance(parent_run_dir=resume_cfg.from_run, parent_step=resolved_step),
        )
    eval_loop = (
        _build_eval_loop(effective_cfg, device, dist_state)
        if effective_cfg.two_pool is None
        else None
    )
    try:
        assert isinstance(snapshot, TrainingState), (
            f"1-pool / 2-pool resume needs TrainingState; got {type(snapshot).__name__}"
        )
        if effective_cfg.two_pool is not None:
            two_trainer = TwoPoolTrainer.from_snapshot(
                snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
                cadence=effective_cfg.cadence,
            )
            two_trainer.run(train_loader, one_sink, effective_cfg.cadence)
        else:
            trainer = Trainer.from_snapshot(
                snapshot,
                target_model=target_model,
                run_batch=run_batch,
                reconstruction_loss=recon_loss_kl,
            )
            trainer.run(train_loader, one_sink, effective_cfg.cadence, eval_loop)
    finally:
        one_sink.finish()


def _split_metrics_by_slow(
    metrics: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Split an `EvalConfig.metrics` list into `(slow, fast)`.

    Slowness is read from the metric class's `slow` class-attr — we look it up via
    ``EVAL_METRIC_CLASSES[m.type]``. Filtering at launch time means the parent
    YAML stays the single source of truth: training filters to fast, async eval
    filters to slow, both reading the same `cfg.eval.metrics`.
    """
    slow_metrics: list[Any] = []
    fast_metrics: list[Any] = []
    for m in metrics:
        cls = EVAL_METRIC_CLASSES[m.type]
        if getattr(cls, "slow", False):
            slow_metrics.append(m)
        else:
            fast_metrics.append(m)
    return slow_metrics, fast_metrics


def _resolve_train_run_id(run_path: str | Path) -> str:
    """Extract the parent run id from a `SavedLMRun.from_path`-compatible reference.

    Accepts wandb URL / `entity/project/runId` / bare `p-xxxxxxxx` / local directory
    whose final name is the run id (i.e. `PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/`).
    """
    s = str(run_path)
    try:
        _, _, run_id = parse_wandb_run_path(s)
        return run_id
    except ValueError:
        pass
    p = Path(s)
    return (p if p.is_dir() else p.parent).name


def _resolve_eval_checkpoint_path(run_path: str | Path, step: int | None) -> Path:
    """Locate the `model_<step>.pth` on disk, downloading from W&B if needed."""
    if step is None:
        files = resolve_run_files(
            run_path, config_filename=RUN_META_FILENAME, checkpoint_prefix="model"
        )
        return files.checkpoint_path
    filename = f"model_{step}.pth"
    files = resolve_run_files(
        run_path, config_filename=RUN_META_FILENAME, checkpoint_filename=filename
    )
    return files.checkpoint_path


def _step_from_checkpoint_name(filename: str) -> int:
    """Parse the step number out of a `model_<step>.pth` filename."""
    assert filename.startswith("model_") and filename.endswith(".pth"), (
        f"expected `model_<step>.pth`, got {filename!r}"
    )
    return int(filename.removeprefix("model_").removesuffix(".pth"))


def _eval_only_main(
    run_path: str | Path,
    *,
    step: int | None,
    eval_config_override: str | Path | None = None,
    group: str | None,
    tags: str | None,
) -> None:
    """Run one eval pass against a saved checkpoint; log results into the parent's wandb run.

    Bypasses the training loop entirely. The `LMExperimentConfig` is reloaded from the
    parent's `run_meta.yaml`; the eval loader + eval metrics are built from `cfg.eval`
    by default. If ``eval_config_override`` is given, it is parsed as an `EvalConfig`
    YAML and replaces ``cfg.eval`` — useful for running slow metrics async on a
    checkpoint when the parent training only ran fast metrics. Each metric value is
    logged to wandb as `eval/<metric_key>` at the resolved step.
    """
    from param_decomp_lab.experiments.utils import EvalConfig

    assert run_path is not None
    pd_run = SavedLMRun.from_path(run_path)
    if eval_config_override is not None:
        eval_cfg = EvalConfig.from_file(Path(eval_config_override))
    else:
        assert pd_run.cfg.eval is not None, (
            f"eval-only requires either an --eval-config override or the parent "
            f"config to declare an `eval:` block ({run_path})"
        )
        eval_cfg = pd_run.cfg.eval

    checkpoint_path = _resolve_eval_checkpoint_path(run_path, step)
    resolved_step = _step_from_checkpoint_name(checkpoint_path.name)
    train_run_id = _resolve_train_run_id(run_path)

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
        logger.info(f"Eval-only: {run_path} @ step {resolved_step} (train run_id={train_run_id})")
    set_seed(pd_run.cfg.pd.seed)
    device = get_device()

    target_model = build_target(pd_run.cfg.target)
    component_model = load_component_model(
        pd_config=pd_run.cfg.pd,
        checkpoint_path=checkpoint_path,
        target_model=target_model,
        run_batch=make_run_batch(pd_run.cfg.target),
    )
    component_model.to(device)

    eval_loader = build_lm_loader(
        pd_run.cfg.target,
        pd_run.cfg.data,
        split="eval",
        device=device,
        batch_size=eval_cfg.batch_size,
        dist_state=dist_state,
        seed=pd_run.cfg.pd.seed,
    )
    eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in eval_cfg.metrics]
    for m in eval_metrics:
        m.bind(model=component_model, device=device)

    results = _run_eval_pass(
        component_model=component_model,
        eval_loader=eval_loader,
        eval_metrics=eval_metrics,
        n_steps=eval_cfg.n_steps,
        device=device,
        step=resolved_step,
        pd_config=pd_run.cfg,
    )

    if is_main_process():
        _log_eval_to_wandb(
            results,
            cfg=pd_run.cfg,
            train_run_id=train_run_id,
            step=resolved_step,
            group=group,
            tags=tags,
        )


def _run_eval_pass(
    *,
    component_model: ComponentModel,
    eval_loader: DataLoader[Any],
    eval_metrics: list[Metric[Any]],
    n_steps: int,
    device: str,
    step: int,
    pd_config: "LMExperimentConfig",
) -> dict[str, Any]:
    """One full eval pass; returns the flattened metric output dict."""
    assert n_steps >= 1, f"n_steps must be at least 1, got {n_steps}"
    eval_iterator = loop_dataloader(eval_loader)
    with torch.no_grad(), bf16_autocast(enabled=pd_config.runtime.autocast_bf16):
        for m in eval_metrics:
            m.reset()
        for _ in range(n_steps):
            ctx = _build_metric_context(
                next(eval_iterator),
                step=step,
                is_eval=True,
                device=device,
                wrapped_model=component_model,
                component_model=component_model,
                config=pd_config.pd,
                reconstruction_loss=recon_loss_kl,
            )
            for m in eval_metrics:
                m.update(ctx)
        results = collect_metric_outputs(eval_metrics)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


def _log_eval_to_wandb(
    results: dict[str, Any],
    *,
    cfg: "LMExperimentConfig",
    train_run_id: str,
    step: int,
    group: str | None,
    tags: str | None,
) -> None:
    """Resume the parent's wandb run and log `eval/<k>` for each result at `step`."""
    if cfg.wandb is None:
        logger.info("No wandb config on parent run; skipping wandb log of eval results.")
        return
    parsed_tags = [s.strip() for s in tags.split(",") if s.strip()] if tags else None
    wandb.init(
        id=train_run_id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or get_wandb_entity(),
        resume="must",
        group=group,
        tags=parsed_tags,
    )
    payload = {f"eval/{k}": v for k, v in results.items()}
    try_wandb(wandb.log, payload, step=step)
    wandb.finish()


def _build_eval_loop(
    cfg: LMExperimentConfig,
    device: str,
    dist_state: DistributedState | None,
) -> EvalLoop | None:
    """Build the `EvalLoop` from `cfg.eval`, or `None` when eval is disabled.

    Slow metrics (class-attr ``slow=True``) are filtered out — in-train eval is
    fast-only. Slow metrics are picked up later by the async eval-only job (which
    receives the slow subset via a temp ``EvalConfig`` YAML, see
    :func:`_submit_slurm_async_slow_eval`).
    """
    if cfg.eval is None:
        return None
    _slow_metrics, fast_metrics = _split_metrics_by_slow(cfg.eval.metrics)
    eval_loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="eval",
        device=device,
        batch_size=cfg.eval.batch_size,
        dist_state=dist_state,
        seed=cfg.pd.seed,
    )
    return EvalLoop(
        loader=eval_loader,
        metrics=[EVAL_METRIC_CLASSES[m.type](m) for m in fast_metrics],
        n_steps=cfg.eval.n_steps,
        every=cfg.eval.every,
        slow_every=cfg.eval.slow_every,
        slow_on_first_step=cfg.eval.slow_on_first_step,
    )


def _submit_slurm(
    config_path: str | Path,
    *,
    dp: int,
    group: str | None,
    tags: str | None,
    partition: str | None,
    time: str,
    job_name: str,
    no_snapshot: bool,
    run_id: str | None,
) -> None:
    run_id = run_id or generate_run_id("param_decomp")
    snapshot_ref: str | None = None
    commit_hash = "no-snapshot"
    if not no_snapshot:
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")

    # If the config is an absolute path inside REPO_ROOT, rewrite to repo-relative so
    # the SLURM job picks up the snapshot's copy rather than the live worktree.
    path = Path(config_path)
    if path.is_absolute() and path.is_relative_to(REPO_ROOT):
        config_arg = path.relative_to(REPO_ROOT).as_posix()
    else:
        config_arg = str(config_path)

    base_parts = ["-m", "param_decomp_lab.experiments.lm.run", config_arg, "--run_id", run_id]
    if group is not None:
        base_parts += ["--group", group]
    if tags is not None:
        base_parts += ["--tags", tags]
    base_command = shlex.join(base_parts)

    launch = build_ddp_launch(
        base_command,
        dp=dp,
        job_name=job_name,
        snapshot_ref=snapshot_ref,
        port_seed=run_id,
    )
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time=time,
        snapshot_ref=snapshot_ref,
        comment=run_id,
    )
    script = generate_script(slurm_config, launch.command, env=launch.env)
    result = submit_slurm_job(script, "lm")

    wandb_url = _wandb_url_for_config(config_path, run_id)

    logger.section("LM PD job submitted!")
    summary: dict[str, str | None] = {
        "Run ID": run_id,
        "Job ID": result.job_id,
        "Log file": result.log_pattern,
        "Script": str(result.script_path),
        "Snapshot": f"{snapshot_ref} ({commit_hash[:8]})" if snapshot_ref else "(none)",
    }
    if wandb_url is not None:
        summary["WandB run URL"] = wandb_url
    logger.values(summary)


def _wandb_url_for_config(config_path: str | Path, run_id: str) -> str | None:
    cfg = LMExperimentConfig.from_file(config_path)
    if cfg.wandb is None:
        return None
    entity = cfg.wandb.entity or get_wandb_entity()
    return f"https://wandb.ai/{entity}/{cfg.wandb.project}/runs/{run_id}"


def submit_slurm_async_slow_eval(
    run_path: str | Path,
    *,
    step: int,
    parent_cfg: "LMExperimentConfig",
    dp: int = 8,
    time: str = "2:00:00",
    partition: str | None = DEFAULT_PARTITION_NAME,
    job_name: str = "pd-lm-eval-slow",
    group: str | None = None,
    tags: str | None = None,
) -> None:
    """Filter the parent's slow eval metrics into a temp YAML, submit a SLURM eval-only.

    Designed to be called from inside training right after a save: emits an async
    SLURM job that runs *only* the slow metrics against ``model_<step>.pth``,
    logging into the parent's wandb run at the same step. No-op when the parent
    has no eval block or no slow metrics.
    """
    from param_decomp_lab.experiments.utils import EvalConfig

    if parent_cfg.eval is None:
        return
    slow_metrics, _fast = _split_metrics_by_slow(parent_cfg.eval.metrics)
    if not slow_metrics:
        return

    slow_eval_cfg = EvalConfig(
        batch_size=parent_cfg.eval.batch_size,
        n_steps=parent_cfg.eval.n_steps,
        every=parent_cfg.eval.every,
        slow_every=parent_cfg.eval.every,  # any value; not consumed by eval-only
        slow_on_first_step=True,
        metrics=slow_metrics,
    )
    run_id = _resolve_train_run_id(run_path)
    scratch = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id / ".async_eval_configs"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg_path = scratch / f"slow_eval_step_{step}.yaml"
    slow_eval_cfg.to_file(cfg_path)

    _submit_slurm_eval_only(
        run_path,
        step=step,
        eval_config=cfg_path,
        dp=dp,
        group=group,
        tags=tags,
        partition=partition,
        time=time,
        job_name=job_name,
        no_snapshot=False,
    )


def _submit_slurm_eval_only(
    run_path: str | Path,
    *,
    step: int | None,
    eval_config: str | Path | None = None,
    dp: int,
    group: str | None,
    tags: str | None,
    partition: str | None,
    time: str,
    job_name: str,
    no_snapshot: bool,
) -> None:
    """Submit a SLURM job that runs `_eval_only_main` against `run_path`.

    Each invocation gets its own git snapshot so the eval job's code matches the
    invoking environment. The child job's `run_id` is unused (eval results log into
    the parent's wandb run via `id=<train_run_id>, resume="must"` inside
    `_eval_only_main`), but we still allocate one so the snapshot ref is unique.
    """
    eval_run_id = generate_run_id("param_decomp")
    snapshot_ref: str | None = None
    commit_hash = "no-snapshot"
    if not no_snapshot:
        snapshot_ref, commit_hash = create_git_snapshot(snapshot_id=eval_run_id)
        logger.info(f"Created git snapshot: {snapshot_ref} ({commit_hash[:8]})")

    base_parts = [
        "-m",
        "param_decomp_lab.experiments.lm.run",
        "--eval-only",
        "--resume",
        str(run_path),
    ]
    if step is not None:
        base_parts += ["--step", str(step)]
    if eval_config is not None:
        base_parts += ["--eval-config", str(eval_config)]
    if group is not None:
        base_parts += ["--group", group]
    if tags is not None:
        base_parts += ["--tags", tags]
    base_command = shlex.join(base_parts)

    launch = build_ddp_launch(
        base_command,
        dp=dp,
        job_name=job_name,
        snapshot_ref=snapshot_ref,
        port_seed=eval_run_id,
    )
    slurm_config = SlurmConfig(
        job_name=job_name,
        partition=partition,
        n_gpus=launch.gpus_per_node,
        n_nodes=launch.n_nodes,
        time=time,
        snapshot_ref=snapshot_ref,
        comment=f"eval-only:{_resolve_train_run_id(run_path)}",
    )
    script = generate_script(slurm_config, launch.command, env=launch.env)
    result = submit_slurm_job(script, "lm")

    print("\n" + "=" * 50)
    print("LM PD eval-only job submitted!")
    print("=" * 50 + "\n")
    print(f"  Parent run    : {run_path}")
    print(f"  Step          : {step if step is not None else 'latest'}")
    print(f"  Job ID        : {result.job_id}")
    print(f"  Log file      : {result.log_pattern}")
    print(f"  Script        : {result.script_path}")
    print(
        f"  Snapshot      : {snapshot_ref} ({commit_hash[:8]})"
        if snapshot_ref
        else "  Snapshot      : (none)"
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
