"""Targeted PD (tPD) config schema — the dual-stream extension of the LM schema.

See `notes/targeted_jax_plan.md` for the full design and the paper
`Targeted Recovery of Weight-Space Mechanisms From Neural Networks` for the method.

tPD trains on TWO parallel streams accumulated into one optimizer step:
  - a narrow TARGET stream (fixed prompts; delta component adversarially ablated), and
  - a broad NON-TARGET stream (the training distribution; delta forced fully on, so
    `components + Δ` reconstruct the target exactly, `delta_override=1.0`).

This module carries only the lab-side schema + build; the engine's two-pass wiring lives
in `param_decomp/run.py` + `param_decomp/train.py` behind the `delta_override` seams
(`adversary.source_masks` / `train.stochastic_entry_masks`). See SPEC-tPD.
"""

from typing import Any

from pydantic import NonNegativeFloat, PositiveInt, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.built_run import BuiltRun
from param_decomp.configs import (
    FaithfulnessLossConfig,
    LossMetricConfig,
    PersistentPGDReconLossConfig,
    StochasticHiddenActsReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp_lab.experiments.config import ExperimentConfig
from param_decomp_lab.experiments.lm.config import LMDataConfig, LMTargetConfig

# Losses dropped from the NON-TARGET pass: PPGD is a stateful per-step adversary the delta
# override deliberately does not drive; unmasked recon is meaningless with a forced-on
# delta; hidden-acts recon is a target-only diagnostic. (Torch parity: the tPD plan's
# "losses excluded from the nontarget pass".)
EXCLUDED_NONTARGET_LOSS_CONFIGS: tuple[type[LossMetricConfig], ...] = (
    PersistentPGDReconLossConfig,
    UnmaskedReconLossConfig,
    StochasticHiddenActsReconLossConfig,
)


class TargetPromptsDataConfig(BaseConfig):
    """The narrow TARGET stream: a file of prompts (one per non-empty line) tokenized once
    at build time into a fixed pool (see `data.py`). Contrast `LMDataConfig`, which streams
    pre-tokenized parquet shards for the broad NON-TARGET stream."""

    prompts_file: str
    tokenizer_name: str
    max_seq_len: PositiveInt


class NontargetConfig(BaseConfig):
    """The broad NON-TARGET stream + its per-pass settings. `data` reuses the LM parquet
    schema. `impmin_coeff_ratio` scales the importance-minimality coeff on the non-target
    pass relative to the target pass (the paper doubles it, ~2.0), accounting for fewer
    active components off-target. The non-target pass forces the delta fully on."""

    data: LMDataConfig
    batch_size: PositiveInt
    eval_batch_size: PositiveInt
    impmin_coeff_ratio: NonNegativeFloat = 1.0


class LMTargetedExperimentConfig(ExperimentConfig[LMTargetConfig, TargetPromptsDataConfig]):
    """tPD run schema: `data` is the target prompts; `nontarget` is the broad stream.

    Parallels `LMExperimentConfig`; consumed by the `pd-lm-targeted` composition root
    (`run.py`)."""

    nontarget: NontargetConfig

    @model_validator(mode="after")
    def _assert_targeted_invariants(self) -> "LMTargetedExperimentConfig":
        # The delta component is always on in the JAX trainer, so no `use_delta_component`
        # check is needed (it was removed from PDConfig). tPD needs the faithfulness pressure
        # OFF (it drives the delta -> 0, but the delta must stay nonzero to carry non-target
        # behavior). The JAX engine (`recon.build_loss_terms`) nonetheless requires exactly
        # one FaithfulnessLoss term, so tPD keeps it present but INERT (coeff 0.0).
        assert self.pd.faithfulness_warmup_steps == 0, (
            "targeted PD needs a nonzero delta (it carries non-target behavior); a "
            "faithfulness warmup drives delta -> 0. Set pd.faithfulness_warmup_steps: 0."
        )
        faith = [m for m in self.pd.loss_metrics if isinstance(m, FaithfulnessLossConfig)]
        assert len(faith) == 1 and faith[0].coeff == 0.0, (
            "targeted PD requires exactly one FaithfulnessLoss with coeff: 0.0 — the engine "
            "needs the term (build_loss_terms asserts it), tPD needs it inert so the delta "
            f"stays nonzero. Got {[(type(m).__name__, m.coeff) for m in faith]}."
        )
        return self


def build_nontarget_loss_metrics(cfg: LMTargetedExperimentConfig) -> list[LossMetricConfig]:
    """Derive the NON-TARGET-pass loss set from the target-pass `pd.loss_metrics`: drop the
    excluded losses (`EXCLUDED_NONTARGET_LOSS_CONFIGS`) and scale the importance-minimality
    coeff by `nontarget.impmin_coeff_ratio`.

    TODO(tPD): implement per `notes/targeted_jax_plan.md` Phase 2. Filter out the excluded
    types; for `ImportanceMinimalityLossConfig` return a `model_copy` with
    `coeff *= impmin_coeff_ratio`; keep a full-model stochastic recon loss so the
    non-target pass grads every component.
    """
    _ = cfg  # pending implementation
    raise NotImplementedError("tPD non-target loss-set derivation — see targeted_jax_plan.md")


def build_targeted_from_schema(schema_raw: dict[str, Any], run_id: str) -> BuiltRun:
    """Validate an `LMTargetedExperimentConfig` and build the engine bundle.

    TODO(tPD): mirror `lm.config.build_from_schema`, additionally (a) building the
    non-target loss set via `build_nontarget_loss_metrics`, and (b) carrying the non-target
    data + pass settings so the composition root (`run.py`) can build the second loader and
    thread a `NontargetPass` into the engine. See `notes/targeted_jax_plan.md` Phase 2-4.
    Note the `BuiltRun` bundle may need a `nontarget` field, or the composition root passes
    the non-target pass to `run_decomposition_training` directly (design decision in the plan).
    """
    _ = (schema_raw, run_id)  # pending implementation
    raise NotImplementedError("tPD build — see targeted_jax_plan.md")
