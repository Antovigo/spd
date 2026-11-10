"""The dual-objective (SPEC S37/T12) LM config surface.

The run config itself is NOT committed — a launched run's provenance already lives in its
pinned `launch_config.yaml`, and the repo ships no targeted LM seat. So these build the dual
shape on top of the targeted variant of the canonical Llama recipe (`l18_mlp_raw` minus
faithfulness plus `prompts`/`nontarget`, as `test_lm_targeted` derives it), which pins the
machinery the run depends on without adding a seat: that variant plus the `hidden` blocks
the derivation adds.

The staged run lives at `~/pd_scratch/dual_obj_jax/` (config, sbatch, and the script that
derives the config from `addsub-L18-jax-mirror-01`); see `notes/dual_objective/README.md`.
"""

from typing import Any

import jax.numpy as jnp
import pytest
from pydantic import ValidationError

from param_decomp.core.components import DenseFactorization, SiteSpec
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.runtime_schedule import RuntimeSchedule
from param_decomp.experiments.lm.config import (
    ChunkwiseTransformerCIFnConfig,
    LMTargetedExperimentConfig,
)
from param_decomp.tests.lm_configs import l18_mlp_raw

# Stand-in site list — full resolution needs the HF snapshot.
_SITES = tuple(
    SiteSpec(name=name, factorization=DenseFactorization(d_in=4, d_out=4, C=4), group="g")
    for name in ("layers.18.mlp.gate_proj", "layers.18.mlp.up_proj", "layers.18.mlp.down_proj")
)

# Every MLP output from the decomposed layer to the end of the network — what the staged run
# reconstructs. `<site>.out` is already a per-block physical tap, captured whether or not that
# block is decomposed, so layers 19-31 need no new tap vocabulary.
POINTS = [f"layers.{layer}.mlp.down_proj.out" for layer in range(18, 32)]


def _at_start(coeff: RuntimeSchedule) -> float:
    return float(coeff.at(jnp.asarray(0.0, jnp.float32)))


def _targeted_raw() -> dict[str, Any]:
    """The single-objective targeted variant of the canonical Llama recipe."""
    raw = l18_mlp_raw()
    raw["target"]["attention_implementation"] = "xla"
    raw["decomposition"]["ci"]["attention"]["implementation"] = "xla"
    raw["pd"]["loss_metrics"] = [
        loss for loss in raw["pd"]["loss_metrics"] if loss["type"] != "FaithfulnessLoss"
    ]
    for field in (
        "faithfulness_warmup_lr",
        "faithfulness_warmup_steps",
        "faithfulness_warmup_weight_decay",
    ):
        del raw["pd"][field]
    raw["prompts"] = {
        "kind": "arithmetic_grid",
        "operation": "add",
        "a_range": [1, 100],
        "b_range": [1, 100],
    }
    raw["nontarget"] = {
        "batch_size": 256,
        "impmin_coeff": 1e-5,
        "recon": [
            {
                "type": "StochasticReconSubsetLoss",
                "coeff": 1.0,
                "routing": {"type": "uniform_k_subset"},
            }
        ],
    }
    return raw


def _dual_raw() -> dict[str, Any]:
    """The targeted variant plus exactly the dual-objective derivation."""
    raw = _targeted_raw()
    raw["decomposition"]["ci"]["dual"] = True
    raw["pd"]["hidden"] = {
        "points": POINTS,
        "impmin_coeff": 5.0e-05,
        "recon": [
            {
                "type": "StochasticReconSubsetLoss",
                "name": "HiddenStochasticReconSubset",
                "coeff": 2.0,
                "routing": {"type": "uniform_k_subset"},
            }
        ],
    }
    raw["nontarget"]["hidden"] = {
        "impmin_coeff": 1.0e-4,
        "recon": [
            {
                "type": "StochasticReconSubsetLoss",
                "name": "NontargetHiddenStochasticReconSubset",
                "coeff": 1.0,
                "routing": {"type": "uniform_k_subset"},
            }
        ],
    }
    raw["runtime"]["sequential_passes"] = True
    return raw


def test_dual_shape_builds_a_four_pass_objective():
    cfg = LMTargetedExperimentConfig.model_validate(_dual_raw())
    assert isinstance(cfg.decomposition.ci, ChunkwiseTransformerCIFnConfig)
    assert cfg.decomposition.ci.dual and cfg.runtime.sequential_passes
    assert cfg.pd.hidden is not None
    objective = build_targeted_objective(
        cfg.pd.loss_metrics, cfg.nontarget, _SITES, hidden=cfg.pd.hidden
    )
    assert objective.hidden is not None and objective.nontarget_hidden is not None
    assert _at_start(objective.hidden.minimality.activity_coeff) == pytest.approx(5e-5)
    assert _at_start(objective.nontarget_hidden.minimality.activity_coeff) == pytest.approx(1e-4)
    assert {t.name: _at_start(t.coeff) for t in objective.hidden.recon} == {
        "HiddenStochasticReconSubset": 2.0
    }
    # Both hidden passes reconstruct the SAME activations: what is reconstructed is a property
    # of the experiment, not of a stream (T12).
    assert objective.hidden.points == tuple(POINTS)
    assert objective.nontarget_hidden.points == tuple(POINTS)


def test_a_single_objective_targeted_run_is_unchanged():
    """The single-objective targeted variant carries no hidden pass and must keep building the
    two-pass objective — T12 is byte-inert when absent."""
    cfg = LMTargetedExperimentConfig.model_validate(_targeted_raw())
    assert isinstance(cfg.decomposition.ci, ChunkwiseTransformerCIFnConfig)
    assert cfg.pd.hidden is None and not cfg.decomposition.ci.dual
    objective = build_targeted_objective(cfg.pd.loss_metrics, cfg.nontarget, _SITES)
    assert objective.hidden is None and objective.nontarget_hidden is None


def test_every_loss_identity_is_unique_across_passes():
    """The target and hidden passes share one metric namespace, so a hidden term reusing a
    target term's default name would silently overwrite its curve. Refused at parse."""
    raw = _dual_raw()
    target_recon = next(
        loss["type"] for loss in raw["pd"]["loss_metrics"] if "Recon" in loss["type"]
    )
    raw["pd"]["hidden"]["recon"][0]["name"] = target_recon
    with pytest.raises(ValidationError, match="collides with a target-pass loss identity"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_hidden_pass_and_second_head_must_be_set_together():
    """Either alone is a misconfiguration: a hidden pass with no head has nothing to mask with,
    and a second head with no hidden pass trains against nothing while costing its parameters
    and its share of every step. Refused at parse."""
    raw = _dual_raw()
    raw["decomposition"]["ci"]["dual"] = False
    with pytest.raises(ValidationError, match="must be set together"):
        LMTargetedExperimentConfig.model_validate(raw)

    raw = _dual_raw()
    raw["pd"].pop("hidden")
    raw["nontarget"].pop("hidden")
    with pytest.raises(ValidationError, match="must be set together"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_nontarget_hidden_pass_cannot_spell_an_adversary():
    """T7 is untouched by T12: adversaries are target-stream only, and the non-target hidden
    pass is authored from the closed non-target vocabulary, which has no adversarial member."""
    raw = _dual_raw()
    raw["nontarget"]["hidden"]["recon"].append(
        {"type": "PersistentPGDReconLoss", "name": "Nope", "coeff": 1.0}
    )
    with pytest.raises(ValidationError, match="PersistentPGDReconLoss"):
        LMTargetedExperimentConfig.model_validate(raw)


def test_nontarget_hidden_pass_needs_the_target_one():
    """It measures at the target-stream hidden pass's `points`, so it cannot exist alone.
    The second head goes too, so the parse-time head/pass pairing check is satisfied and the
    refusal under test is the objective builder's."""
    raw = _dual_raw()
    raw["pd"].pop("hidden")
    raw["decomposition"]["ci"]["dual"] = False
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    with pytest.raises(AssertionError, match="nontarget.hidden needs pd.hidden"):
        build_targeted_objective(cfg.pd.loss_metrics, cfg.nontarget, _SITES)
