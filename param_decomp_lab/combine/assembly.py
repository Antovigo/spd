"""Assemble several single-block PD runs into one combined `ComponentModel`."""

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import GlobalCiConfig, GroupedGlobalCiConfig
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import resolve_decomposition_targets
from param_decomp_lab.experiments.lm.run import LMExperimentConfig, SavedLMRun
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


@dataclass(frozen=True)
class SourceRun:
    """One single-block run to be merged into the combined model."""

    name: str
    group: str
    layer_idx: int
    cfg: LMExperimentConfig
    checkpoint_path: Path

    @property
    def module_patterns(self) -> list[str]:
        return [t.module_pattern for t in self.cfg.pd.decomposition_targets]


def resolve_run_dir(run: str) -> Path:
    path = Path(run)
    if not path.exists():
        path = PARAM_DECOMP_OUT_DIR / "runs" / run
    assert path.exists(), f"run not found: {run}"
    return path


def load_source_runs(runs: list[str]) -> list[SourceRun]:
    """Load run configs and derive one CI-fn group per run; assert they are combinable."""
    sources: list[SourceRun] = []
    for run in runs:
        run_dir = resolve_run_dir(run)
        saved = SavedLMRun.from_path(run_dir)
        assert saved.cfg.pd.identity_decomposition_targets is None
        assert saved.cfg.pd.tied_weights is None
        patterns = [t.module_pattern for t in saved.cfg.pd.decomposition_targets]
        layer_indices = {
            int(m.group(1))
            for p in patterns
            if (m := re.match(r"model\.layers\.(\d+)\.", p)) is not None
        }
        assert len(layer_indices) == 1, (
            f"{run_dir.name}: expected a single-block run, got layer indices {layer_indices}"
        )
        layer_idx = layer_indices.pop()
        sources.append(
            SourceRun(
                name=run_dir.name,
                group=f"layers{layer_idx}",
                layer_idx=layer_idx,
                cfg=saved.cfg,
                checkpoint_path=saved.checkpoint_path,
            )
        )

    ref = sources[0].cfg
    for s in sources[1:]:
        assert s.cfg.target == ref.target, f"{s.name}: target differs from {sources[0].name}"
        assert s.cfg.pd.ci_config == ref.pd.ci_config, f"{s.name}: ci_config differs"
        assert s.cfg.pd.sigmoid_type == ref.pd.sigmoid_type, f"{s.name}: sigmoid_type differs"
        assert s.cfg.pd.use_delta_component == ref.pd.use_delta_component
        assert s.cfg.pd.sampling == ref.pd.sampling, f"{s.name}: sampling differs"
    assert len({s.group for s in sources}) == len(sources), "duplicate layer across runs"
    return sorted(sources, key=lambda s: s.layer_idx)


def combined_ci_config(sources: list[SourceRun]) -> GroupedGlobalCiConfig:
    """Grouped CI config with one group per source run, matching its exact targets."""
    ref = sources[0].cfg.pd.ci_config
    assert isinstance(ref, GlobalCiConfig), f"expected a global CI config, got {type(ref)}"
    return GroupedGlobalCiConfig(
        fn_type=ref.fn_type,
        hidden_dims=ref.hidden_dims,
        simple_transformer_ci_cfg=ref.simple_transformer_ci_cfg,
        groups={s.group: sorted(s.module_patterns) for s in sources},
    )


CI_FN_CHECKPOINT_PREFIX = "ci_fn._global_ci_fn."


def combined_state_dict(sources: list[SourceRun], load_ci_fns: bool) -> dict[str, torch.Tensor]:
    """Slice `_components.*` (and optionally the CI fns) out of each source checkpoint.

    Each source's single `_global_ci_fn` is remapped to the group CI fn named after its
    layer (`ci_fn._group_ci_fns.layers<N>.*`). Checkpoints are mmap-loaded so their
    embedded `target_model.*` copies are never materialised.
    """
    state: dict[str, torch.Tensor] = {}
    for s in sources:
        checkpoint = torch.load(s.checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        for key, value in checkpoint.items():
            if key.startswith("_components."):
                new_key = key
            elif load_ci_fns and key.startswith(CI_FN_CHECKPOINT_PREFIX):
                suffix = key.removeprefix(CI_FN_CHECKPOINT_PREFIX)
                new_key = f"ci_fn._group_ci_fns.{s.group}.{suffix}"
            else:
                continue
            assert new_key not in state, f"{s.name}: duplicate state key {new_key}"
            state[new_key] = value
    return state


def load_combined_state(
    model: ComponentModel, sources: list[SourceRun], *, load_ci_fns: bool
) -> None:
    """Load the sources' assembled weights into `model`; the target model and (when
    `load_ci_fns` is false) the CI fn keep their existing weights."""
    state = combined_state_dict(sources, load_ci_fns)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys when loading combined state: {unexpected[:5]}"
    allowed_prefixes = ("target_model.",) if load_ci_fns else ("target_model.", "ci_fn.")
    bad_missing = [k for k in missing if not k.startswith(allowed_prefixes)]
    assert not bad_missing, f"missing keys not covered by the sources: {bad_missing[:5]}"


def build_combined_component_model(
    sources: list[SourceRun],
    target_model: nn.Module,
    run_batch: RunBatch,
    device: str,
) -> ComponentModel:
    """Union `ComponentModel` over all sources' targets, weights loaded per source.

    The target model is the caller's (already-loaded) copy; checkpoints'
    `target_model.*` keys are never read.
    """
    target_model.eval()
    target_model.requires_grad_(False)

    all_target_cfgs = [t for s in sources for t in s.cfg.pd.decomposition_targets]
    resolved_targets = resolve_decomposition_targets(target_model, all_target_cfgs)
    model = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=resolved_targets,
        ci_config=combined_ci_config(sources),
        sigmoid_type=sources[0].cfg.pd.sigmoid_type,
    )
    load_combined_state(model, sources, load_ci_fns=True)
    return model.to(device)
