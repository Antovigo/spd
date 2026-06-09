"""The PD trainer.

:class:`Trainer` is the one entry point: construction sets up the
`ComponentModel`, the two optimizers, and the loss-metric instances from
``pd_config``; :meth:`Trainer.run` advances the training loop from
``self.step`` to ``pd_config.steps``; :meth:`Trainer.snapshot` and
:meth:`Trainer.from_snapshot` round-trip an atomic :class:`TrainingState`
that lets a caller persist and restore the full training state (resumption).
"""

from collections import defaultdict
from typing import Any, Self, cast

import torch
import torch.nn as nn
import torch.nn.parallel
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_decomp.batch_and_loss_fns import ReconstructionLoss, RunBatch
from param_decomp.component_model import ComponentModel, component_grad_norms
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import (
    insert_identity_operations_,
    resolve_decomposition_targets,
)
from param_decomp.distributed import (
    avg_metrics_across_ranks,
    get_distributed_state,
    is_main_process,
    seed_all_ranks,
    seed_per_rank,
    sync_across_processes,
)
from param_decomp.faithfulness_warmup import run_faithfulness_warmup
from param_decomp.log import logger
from param_decomp.metrics.base import Metric
from param_decomp.metrics.dispatch import instantiate_metrics
from param_decomp.metrics.persistent_pgd_recon import validate_pgd_scope
from param_decomp.run_sink import OnePoolRunSink
from param_decomp.torch_helpers import loop_dataloader
from param_decomp.train_step import (
    EvalLoop,
    _assert_ctx_invariants,
    _build_metric_context,
    _install_sigterm_flag,
    empty_cuda_cache_and_collect,
    run_eval_pass,
    run_loss_step,
    scheduled_lrs,
)
from param_decomp.training_state import TrainingState

__all__ = [
    "EvalLoop",
    "Trainer",
    "_assert_ctx_invariants",
    "_build_metric_context",
    "load_optimizer_state_by_name",
    "optimizer_state_by_name",
    "tie_component_weights",
]


def tie_component_weights(
    component_model: ComponentModel, tied_weights: list[tuple[str, str]]
) -> None:
    for src_name, tgt_name in tied_weights:
        tgt = component_model.components[tgt_name]
        src = component_model.components[src_name]
        assert tgt is not None and src is not None, (
            f"Cannot tie weights between {src_name} and {tgt_name} - one or both are None"
        )
        tgt.U.data = src.V.data.T
        tgt.V.data = src.U.data.T


def optimizer_state_by_name(
    optimizer: torch.optim.Optimizer,
    named_params: list[tuple[str, nn.Parameter]],
) -> dict[str, dict[str, Any]]:
    """Convert ``optimizer.state_dict()["state"]`` from integer-indexed to name-keyed.

    PyTorch optimizers key their internal per-parameter state (Adam moments, step
    counter, etc.) by the integer position of each parameter in
    ``param_groups[*]["params"]``. That position is topology-dependent: a rank
    holding a subset of sites indexes them 0..N for *its own* sites, in the
    order they were added. To make optimizer state survive a topology change,
    the caller passes the ``(name, param)`` pairs in the same order they were
    added to the optimizer, and this returns ``{name: state_entry}``.

    Names that have no optimizer state yet (e.g. fresh param, no step taken)
    are simply omitted.
    """
    raw_state: dict[int, dict[str, Any]] = optimizer.state_dict()["state"]
    by_name: dict[str, dict[str, Any]] = {}
    for i, (name, _) in enumerate(named_params):
        if i in raw_state:
            by_name[name] = raw_state[i]
    return by_name


def load_optimizer_state_by_name(
    optimizer: torch.optim.Optimizer,
    named_params: list[tuple[str, nn.Parameter]],
    by_name: dict[str, dict[str, Any]],
) -> None:
    """Inverse of :func:`optimizer_state_by_name`.

    Each ``(name, param)`` pair in ``named_params`` matches its position in
    the live optimizer's ``param_groups[*]["params"]``. For each name present
    in ``by_name``, install the saved entry under the matching integer index
    in ``optimizer.state``. Param groups (lr, betas, etc.) are taken from the
    live optimizer — they're hyperparameters configured at construction time,
    not state to be round-tripped.
    """
    current = optimizer.state_dict()
    new_state: dict[int, dict[str, Any]] = {}
    for i, (name, _) in enumerate(named_params):
        if name in by_name:
            new_state[i] = by_name[name]
    current["state"] = new_state
    optimizer.load_state_dict(current)


class Trainer:
    """Stateful PD trainer.

    Construction wires up the `ComponentModel`, both AdamW optimizers, and the
    loss-metric instances declared in ``pd_config.loss_metrics``. :meth:`run`
    advances the training loop from ``self.step`` to ``pd_config.steps``.
    :meth:`snapshot` and :meth:`from_snapshot` round-trip a
    :class:`~param_decomp.trainer_snapshot.TrainerSnapshot` that a caller can
    persist and restore.

    All ranks construct a `Trainer`. Sink output and the loader-replay skip on
    resume are governed by ``self.step`` (advanced by :meth:`run`,
    overwritten by :meth:`_load_state`).
    """

    pd_config: PDConfig
    runtime_config: RuntimeConfig
    reconstruction_loss: ReconstructionLoss
    component_model: ComponentModel
    components_optimizer: optim.Optimizer
    ci_fn_optimizer: optim.Optimizer
    loss_metrics: dict[str, Metric[Any]]
    step: int

    def __init__(
        self,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
        pd_config: PDConfig,
        runtime_config: RuntimeConfig,
    ) -> None:
        self.pd_config = pd_config
        self.runtime_config = runtime_config
        self.reconstruction_loss = reconstruction_loss
        self.step = 0

        dist_state = get_distributed_state()
        device = runtime_config.device
        validate_pgd_scope(
            pd_config.loss_metrics,
            batch_size=pd_config.batch_size,
            world_size=dist_state.world_size if dist_state is not None else 1,
        )

        if pd_config.identity_decomposition_targets is not None:
            insert_identity_operations_(
                target_model,
                identity_decomposition_targets=pd_config.identity_decomposition_targets,
            )

        target_model.requires_grad_(False)
        target_model.eval()
        decomposition_targets = resolve_decomposition_targets(
            target_model, pd_config.all_decomposition_target_configs
        )

        seed_all_ranks(pd_config.seed)
        model = ComponentModel(
            target_model=target_model,
            run_batch=run_batch,
            decomposition_targets=decomposition_targets,
            ci_config=pd_config.ci_config,
            sigmoid_type=pd_config.sigmoid_type,
        )
        model.to(device)

        # Diverge global RNG per rank so stochastic masks/sources differ across DP workers.
        seed_per_rank(pd_config.seed)

        if dist_state is not None:
            if dist_state.backend == "nccl":
                device_id = dist_state.local_rank
                self._wrapped_model: nn.Module = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[device_id], output_device=device_id
                )
            else:
                self._wrapped_model = torch.nn.parallel.DistributedDataParallel(model)
            component_model = cast(ComponentModel, self._wrapped_model.module)
        else:
            self._wrapped_model = model
            component_model = model
        assert isinstance(component_model, ComponentModel)
        self.component_model = component_model

        if pd_config.tied_weights is not None:
            tie_component_weights(component_model, pd_config.tied_weights)

        self._component_params: list[torch.nn.Parameter] = []
        for name in component_model.target_module_paths:
            self._component_params.extend(component_model.components[name].parameters())
        assert component_model.ci_fn is not None, (
            "single-pool Trainer assumes a ComponentModel with the CI fn intact"
        )
        self._ci_fn_params = list(component_model.ci_fn.parameters())
        assert len(self._component_params) > 0, "No parameters found in components to optimize"

        self.components_optimizer = optim.AdamW(
            self._component_params,
            lr=pd_config.components_optimizer.lr_schedule.start_val,
            betas=pd_config.components_optimizer.betas,
            weight_decay=pd_config.components_optimizer.weight_decay,
        )
        self.ci_fn_optimizer = optim.AdamW(
            self._ci_fn_params,
            lr=pd_config.ci_fn_optimizer.lr_schedule.start_val,
            betas=pd_config.ci_fn_optimizer.betas,
            weight_decay=pd_config.ci_fn_optimizer.weight_decay,
        )

        self.loss_metrics, _ = instantiate_metrics(pd_config, component_model, device)

    # ============================ Named-param accessors for optimizer state ============================

    def _components_optimizer_named_params(self) -> list[tuple[str, nn.Parameter]]:
        """The ``(name, param)`` pairs in the order they were added to ``components_optimizer``.

        Names follow ``components.<module_path>.<param_name_inside_component>`` so they're
        topology-independent (a sharded trainer holding only a subset of sites produces
        the same names for those sites as a 1-pool trainer holding all of them).
        """
        out: list[tuple[str, nn.Parameter]] = []
        for module_path in self.component_model.target_module_paths:
            for pname, p in self.component_model.components[module_path].named_parameters():
                out.append((f"components.{module_path}.{pname}", p))
        return out

    def _ci_fn_optimizer_named_params(self) -> list[tuple[str, nn.Parameter]]:
        """The ``(name, param)`` pairs for ``ci_fn_optimizer``."""
        assert self.component_model.ci_fn is not None
        return [(f"ci_fn.{n}", p) for n, p in self.component_model.ci_fn.named_parameters()]

    def _build_all_metric_instances(
        self,
        eval_loop: "EvalLoop | None",
        device: str,
    ) -> dict[str, "Metric[Any]"]:
        """Merge loss + eval-only metric instances keyed by class name.

        Binds each eval-only metric, rejects duplicate names within eval_loop, and
        rejects overlap between eval-only and loss metrics (loss metrics are
        auto-evaluated; duplicating them as eval metrics is a config error since
        ``evaluate()`` keys by class name).
        """
        eval_only_instances: dict[str, Metric[Any]] = {}
        if eval_loop is not None:
            for m in eval_loop.metrics:
                m.bind(model=self.component_model, device=device)
                metric_name = type(m).__name__
                assert metric_name not in eval_only_instances, (
                    f"duplicate eval metric {metric_name!r}"
                )
                eval_only_instances[metric_name] = m
            overlap = sorted(set(self.loss_metrics) & set(eval_only_instances))
            assert not overlap, (
                f"eval_loop.metrics overlap with pd_config.loss_metrics: {overlap}. Loss "
                "metrics are automatically evaluated; remove the duplicates from "
                "eval_loop.metrics."
            )
        return {**self.loss_metrics, **eval_only_instances}

    # ============================ Atomic cfg + state ============================

    def snapshot(self) -> TrainingState:
        """Canonical point-in-time view of this trainer.

        Returns a :class:`TrainingState` carrying configs (model_dump'd), model
        state dict, both optimizer states (keyed by parameter NAME so they
        survive topology changes), and every loss metric's ``state_dict()``.
        For 1-pool DDP this state is identical across ranks (the model and
        optimizers are replicated); the lab sink writes from rank 0 only.
        """
        return TrainingState(
            step=self.step,
            pd_config=self.pd_config.model_dump(),
            runtime_config=self.runtime_config.model_dump(),
            component_model=self.component_model.state_dict(),
            components_optimizer=optimizer_state_by_name(
                self.components_optimizer, self._components_optimizer_named_params()
            ),
            ci_fn_optimizer=optimizer_state_by_name(
                self.ci_fn_optimizer, self._ci_fn_optimizer_named_params()
            ),
            loss_metrics={n: m.state_dict() for n, m in self.loss_metrics.items()},
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TrainingState,
        *,
        target_model: nn.Module,
        run_batch: RunBatch,
        reconstruction_loss: ReconstructionLoss,
    ) -> Self:
        """Reconstruct a Trainer from a :class:`TrainingState`.

        For mid-trajectory edits to the saved config (e.g. extending ``steps``
        on a finished run), mutate ``snapshot.pd_config`` in place before this
        call.
        """
        pd_config = PDConfig.model_validate(snapshot.pd_config)
        runtime_config = RuntimeConfig.model_validate(snapshot.runtime_config)
        trainer = cls(
            target_model=target_model,
            run_batch=run_batch,
            reconstruction_loss=reconstruction_loss,
            pd_config=pd_config,
            runtime_config=runtime_config,
        )
        trainer._load_state(snapshot)
        return trainer

    def _load_state(self, state: TrainingState) -> None:
        """In-place load of the trainer's runtime state. Caller's responsibility
        to have constructed self with a matching cfg (use :meth:`from_snapshot`
        to guarantee this).
        """
        self.step = state.step
        self.component_model.load_state_dict(state.component_model)
        load_optimizer_state_by_name(
            self.components_optimizer,
            self._components_optimizer_named_params(),
            state.components_optimizer,
        )
        load_optimizer_state_by_name(
            self.ci_fn_optimizer,
            self._ci_fn_optimizer_named_params(),
            state.ci_fn_optimizer,
        )
        for name, m in self.loss_metrics.items():
            m.load_state_dict(state.loss_metrics[name])

    # ============================ Training loop ============================

    def run(
        self,
        train_loader: DataLoader[Any],
        sink: OnePoolRunSink,
        cadence: Cadence,
        eval_loop: EvalLoop | None = None,
    ) -> None:
        """Advance training from ``self.step`` to ``self.pd_config.steps``.

        When ``self.step == 0`` and ``pd_config.faithfulness_warmup_steps > 0``,
        faithfulness warmup runs once before the loop. When ``self.step > 0``
        (i.e. resumed mid-trajectory), warmup is skipped and the train loader is
        skip-advanced by ``self.step`` batches to reproduce the corresponding
        position in the data stream.
        """
        pd_config = self.pd_config
        runtime_config = self.runtime_config
        device = runtime_config.device

        train_iterator = loop_dataloader(train_loader)
        eval_iterator = loop_dataloader(eval_loop.loader) if eval_loop is not None else None

        # Loader replay: if we're starting from non-zero step, advance the iterator to
        # the matching position. Deterministic given the loader's seed.
        for _ in range(self.step):
            next(train_iterator)

        if self.step == 0 and pd_config.faithfulness_warmup_steps > 0:
            run_faithfulness_warmup(self.component_model, self._component_params, pd_config)

        all_instances = self._build_all_metric_instances(eval_loop, device)
        sigterm = _install_sigterm_flag()

        for step in tqdm(
            range(self.step, pd_config.steps + 1), ncols=0, disable=not is_main_process()
        ):
            self.step = step
            self.components_optimizer.zero_grad()
            self.ci_fn_optimizer.zero_grad()

            components_lr, ci_fn_lr = scheduled_lrs(
                step, total_steps=pd_config.steps, config=pd_config
            )
            for group in self.components_optimizer.param_groups:
                group["lr"] = components_lr
            for group in self.ci_fn_optimizer.param_groups:
                group["lr"] = ci_fn_lr

            _, batch_log_data = run_loss_step(
                batch=next(train_iterator),
                step=step,
                device=device,
                wrapped_model=self._wrapped_model,
                component_model=self.component_model,
                loss_metrics=self.loss_metrics,
                config=pd_config,
                reconstruction_loss=self.reconstruction_loss,
                autocast_bf16=runtime_config.autocast_bf16,
            )

            # --- Train Logging --- #
            if cadence.should_log_train(step):
                avg_metrics = avg_metrics_across_ranks(batch_log_data, device=device)
                batch_log_data = cast(defaultdict[str, float], avg_metrics)

                grad_norms = component_grad_norms(self.component_model, device)
                grad_norm_log_data = {f"grad_norms/{k}": v for k, v in grad_norms.items()}
                assert not set(batch_log_data) & set(grad_norm_log_data)
                batch_log_data.update(grad_norm_log_data)
                batch_log_data["schedules/lr/components"] = components_lr
                batch_log_data["schedules/lr/ci_fn"] = ci_fn_lr

                sink.console(
                    f"--- Step {step} ---",
                    f"LR[components]: {components_lr:.6f}",
                    f"LR[ci_fn]: {ci_fn_lr:.6f}",
                    *(f"train/{name}: {value:.15f}" for name, value in batch_log_data.items()),
                )
                sink.log({f"train/{k}": v for k, v in batch_log_data.items()}, step=step)

            # --- Evaluation --- #
            if eval_loop is not None and eval_loop.should_eval(step):
                assert eval_iterator is not None
                fast_metrics, slow_metrics = run_eval_pass(
                    eval_iterator=eval_iterator,
                    n_steps=eval_loop.n_steps,
                    slow_step=eval_loop.should_run_slow_eval(step),
                    all_instances=all_instances,
                    step=step,
                    device=device,
                    wrapped_model=self._wrapped_model,
                    component_model=self.component_model,
                    config=pd_config,
                    reconstruction_loss=self.reconstruction_loss,
                    autocast_bf16=runtime_config.autocast_bf16,
                )
                # Fast metrics use the `eval/` namespace + default step axis; slow metrics
                # use `slow_eval/` + a `slow_eval/step` axis, so this matches the 3-pool
                # async slow-eval keys (see experiments.lm.async_eval) and overlays.
                sink.console(*(f"eval/{k}: {v}" for k, v in fast_metrics.items()))
                sink.log({f"eval/{k}": v for k, v in fast_metrics.items()}, step=step)
                if slow_metrics:
                    sink.console(*(f"slow_eval/{k}: {v}" for k, v in slow_metrics.items()))
                    slow_payload = {f"slow_eval/{k}": v for k, v in slow_metrics.items()}
                    slow_payload["slow_eval/step"] = step
                    sink.log(slow_payload, step=step)
                empty_cuda_cache_and_collect()

            # --- Saving Checkpoint --- #
            if step == pd_config.steps or cadence.should_save(step) or sigterm.received:
                is_final = step == pd_config.steps or sigterm.received
                sink.checkpoint(self.snapshot(), final=is_final)
            if sigterm.received:
                if is_main_process():
                    logger.info(
                        f"SIGTERM received; saved checkpoint at step {step}, exiting train loop"
                    )
                break

            # Skip gradient step at the very last step (last step is just for plotting/logging).
            if step != pd_config.steps:
                sync_across_processes()
                if pd_config.components_optimizer.grad_clip_norm is not None:
                    clip_grad_norm_(
                        self._component_params, pd_config.components_optimizer.grad_clip_norm
                    )
                if pd_config.ci_fn_optimizer.grad_clip_norm is not None:
                    clip_grad_norm_(self._ci_fn_params, pd_config.ci_fn_optimizer.grad_clip_norm)
                self.components_optimizer.step()
                self.ci_fn_optimizer.step()

        if is_main_process():
            logger.info("Finished training loop.")
