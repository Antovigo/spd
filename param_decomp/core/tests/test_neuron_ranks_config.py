"""`weight_init: neuron_aligned_targeted` at the config boundary (SPEC T13): targeted-only,
and its artifact reference is required by — and only by — that init."""

import pytest
from pydantic import ValidationError

from param_decomp.core.configs import NamedNeuronRanks, NeuronRanksDir, PDConfig, TargetedPDConfig

_TARGETED_BASE = {
    "loss_metrics": [
        {"type": "ImportanceMinimalityLoss", "coeff": 3e-3, "pnorm": 1.0, "eps": 1e-12},
        {"type": "StochasticReconLoss", "coeff": 1.0},
    ],
    "components_optimizer": {"lr_schedule": 1e-3},
    "ci_fn_optimizer": {"lr_schedule": 1e-3},
    "steps": 10,
    "batch_size": 8,
}


def test_targeted_neuron_aligned_requires_its_artifact_and_only_then():
    with pytest.raises(ValidationError, match="neuron_ranks"):
        TargetedPDConfig.model_validate(
            {**_TARGETED_BASE, "weight_init": "neuron_aligned_targeted"}
        )
    with pytest.raises(ValidationError, match="neuron_ranks"):
        TargetedPDConfig.model_validate(
            {
                **_TARGETED_BASE,
                "weight_init": "coupled",
                "neuron_ranks": {"kind": "name", "name": "x"},
            }
        )
    named = TargetedPDConfig.model_validate(
        {
            **_TARGETED_BASE,
            "weight_init": "neuron_aligned_targeted",
            "neuron_ranks": {"kind": "name", "name": "addsub-l31-8b"},
        }
    )
    assert isinstance(named.neuron_ranks, NamedNeuronRanks)
    explicit = TargetedPDConfig.model_validate(
        {
            **_TARGETED_BASE,
            "weight_init": "neuron_aligned_targeted",
            "neuron_ranks": {"kind": "dir", "dir": "/abs/ranks"},
        }
    )
    assert isinstance(explicit.neuron_ranks, NeuronRanksDir)
    assert TargetedPDConfig.model_validate(_TARGETED_BASE).neuron_ranks is None
    wrap = TargetedPDConfig.model_validate(
        {
            **_TARGETED_BASE,
            "weight_init": "neuron_aligned_wrap",
            "neuron_ranks": {"kind": "name", "name": "addsub-l31-8b"},
        }
    )
    assert wrap.weight_init == "neuron_aligned_wrap"
    with pytest.raises(ValidationError, match="neuron_ranks"):
        TargetedPDConfig.model_validate({**_TARGETED_BASE, "weight_init": "neuron_aligned_wrap"})


def test_neuron_ranks_refs_are_flat_names_or_absolute_dirs():
    with pytest.raises(ValidationError, match="flat"):
        NamedNeuronRanks.model_validate({"kind": "name", "name": "a/b"})
    with pytest.raises(ValidationError, match="absolute"):
        NeuronRanksDir.model_validate({"kind": "dir", "dir": "relative/ranks"})


def test_plain_pd_refuses_the_targeted_init():
    plain = {
        "loss_metrics": [
            {"type": "FaithfulnessLoss", "coeff": 1.0},
            {"type": "ImportanceMinimalityLoss", "coeff": 3e-3, "pnorm": 1.0, "eps": 1e-12},
            {"type": "StochasticReconLoss", "coeff": 1.0},
        ],
        "components_optimizer": {"lr_schedule": 1e-3},
        "ci_fn_optimizer": {"lr_schedule": 1e-3},
        "steps": 10,
        "batch_size": 8,
    }
    assert PDConfig.model_validate(plain).weight_init == "default"
    for init in ("neuron_aligned_targeted", "neuron_aligned_wrap"):
        with pytest.raises(ValidationError, match="targeted-run"):
            PDConfig.model_validate({**plain, "weight_init": init})
