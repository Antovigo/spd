"""Targeted-PD (tPD) LM experiment config schema + the tPD YAML->`BuiltRun` conversion.

The LM analog of the toy targeted runs (`experiments/{tms,resid_mlp}`): the TARGET stream is
a fixed prompt pool (`data.prompts_file`), the NON-TARGET stream is the normal parquet path
(`nontarget.data.dataset_name == "parquet"`), and the delta component is forced on over the
non-target pass (SPEC §11). Reuses the plain-LM target / eval / ci-fn resolution
(`experiments.lm.config`) verbatim; the only tPD-specific build is the `data` seam — the
engine sizes the persistent-PGD source off `data.seq_len` / `data.global_batch` on the
TARGET pass, so `BuiltRun.data` carries the TARGET prompt length + batch, not the parquet
stream.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator

from param_decomp.built_run import BuiltRun, DataConfig
from param_decomp_lab.experiments.config import (
    NontargetConfig,
    assert_canonical_algorithm_config,
    assert_targeted_faithfulness_off,
    ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.lm.config import (
    LMDataConfig,
    LMExperimentConfig,
    _assert_losses_supported,
    _eval,
    _resolve_chunkwise_ci_arch,
    _resolve_target,
    assert_supported_weights_dtype,
)


class LMTargetedExperimentConfig(LMExperimentConfig):
    """Targeted-PD LM run: `data` is the fixed prompt-pool TARGET stream, `nontarget.data` is
    the broad parquet NON-TARGET stream."""

    nontarget: NontargetConfig[LMDataConfig]

    @model_validator(mode="after")
    def _assert_targeted_invariants(self) -> "LMTargetedExperimentConfig":
        assert_targeted_faithfulness_off(self.pd)
        assert self.data.prompts_file is not None, (
            "targeted LM target stream must be a prompt pool (set data.prompts_file)"
        )
        assert self.nontarget.data.dataset_name is not None, (
            "targeted LM non-target stream must be parquet (set nontarget.data.dataset_name)"
        )
        return self


def _nontarget_parquet_dir(cfg: LMTargetedExperimentConfig) -> Path:
    """The non-target parquet shard dir, resolved from `nontarget.data` the same way
    `experiments.lm.config._data` resolves the plain-LM parquet dir."""
    data = cfg.nontarget.data
    assert data.is_tokenized and not data.streaming, (
        "JAX trainer reads pre-tokenized parquet shards; tokenize offline first"
    )
    assert data.dataset_name == "parquet" and data.column_name == "input_ids", data
    assert data.data_files is not None
    shard_glob = Path(data.data_files)
    assert shard_glob.name == "*.parquet", f"expected a *.parquet glob, got {data.data_files}"
    return shard_glob.parent


def _targeted_data(cfg: LMTargetedExperimentConfig) -> DataConfig:
    """The tPD `data` seam: `seq_len` = the TARGET prompt length, `global_batch` matches the
    plain-LM `_data` (`pd.batch_size`), `dir` = the NON-target parquet dir. The engine reads
    only `seq_len` / `global_batch` for persistent-PGD source sizing on the TARGET pass; `dir`
    is unused by the engine but must be a valid path."""
    return DataConfig(
        dir=_nontarget_parquet_dir(cfg),
        seq_len=cfg.data.max_seq_len,
        global_batch=cfg.pd.batch_size,
    )


def build_targeted(cfg: LMTargetedExperimentConfig, run_id: str) -> BuiltRun:
    """Convert the tPD schema to the engine's `BuiltRun`. Reuses the plain-LM target / eval /
    ci-fn resolution (`build_experiment_config` verbatim minus its `_data` call — the prompt
    `data` config asserts parquet); only `data` is tPD-specific (`_targeted_data`)."""
    target = _resolve_target(cfg)
    assert_canonical_algorithm_config(cfg)
    _assert_losses_supported(cfg, tuple(sc.name for sc in target.sites))
    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id),
        target=target,
        data=_targeted_data(cfg),
        ci_fn=ci_arch(cfg.pd.ci_config, lambda ci: _resolve_chunkwise_ci_arch(target, ci)),
        eval=_eval(cfg),
    )


def build_targeted_from_schema(schema_raw: dict[str, Any], run_id: str) -> BuiltRun:
    """Validate a self-contained tPD LM run config (`LMTargetedExperimentConfig`) and convert
    it to the engine's `BuiltRun`. `run_id` is the minted run identity."""
    cfg = LMTargetedExperimentConfig(**schema_raw)
    assert_supported_weights_dtype(cfg)
    return build_targeted(cfg, run_id)


def load_targeted_config(config_path: Path, run_id: str) -> tuple[BuiltRun, dict[str, Any]]:
    """Parse a tPD LM run YAML -> (built run, raw dict for wandb logging)."""
    schema_raw = yaml.safe_load(config_path.read_text())
    return build_targeted_from_schema(schema_raw, run_id), schema_raw
