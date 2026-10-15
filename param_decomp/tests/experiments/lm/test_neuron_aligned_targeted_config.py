"""`initialization: neuron_aligned_targeted` at the config boundary (SPEC T13): its
artifact reference is required by — and only by — that init, and a plain run refuses it."""

from typing import Any

import pytest
from pydantic import ValidationError

from param_decomp.core.configs import NamedNeuronRanks, NeuronRanksDir
from param_decomp.experiments.lm.config import (
    GluTransformerCSpec,
    LMExperimentConfig,
    LMTargetedExperimentConfig,
)
from param_decomp.tests.experiments.lm.test_lm_targeted import _targeted_raw
from param_decomp.tests.lm_configs import l18_mlp_raw

_SITES: dict[str, Any] = {
    "kind": "glu_transformer",
    "layers": {"kind": "list", "indices": [18]},
    "cs": {"gate": 8, "q": 4},
}


def test_the_artifact_is_required_by_and_only_by_the_targeted_init():
    with pytest.raises(ValidationError, match="neuron_ranks"):
        GluTransformerCSpec.model_validate({**_SITES, "initialization": "neuron_aligned_targeted"})
    with pytest.raises(ValidationError, match="neuron_ranks"):
        GluTransformerCSpec.model_validate(
            {**_SITES, "initialization": "random", "neuron_ranks": {"kind": "name", "name": "x"}}
        )
    named = GluTransformerCSpec.model_validate(
        {
            **_SITES,
            "initialization": "neuron_aligned_targeted",
            "neuron_ranks": {"kind": "name", "name": "addsub-l18-8b"},
        }
    )
    assert isinstance(named.neuron_ranks, NamedNeuronRanks)
    explicit = GluTransformerCSpec.model_validate(
        {
            **_SITES,
            "initialization": "neuron_aligned_targeted",
            "neuron_ranks": {"kind": "dir", "dir": "/abs/ranks"},
        }
    )
    assert isinstance(explicit.neuron_ranks, NeuronRanksDir)
    assert GluTransformerCSpec.model_validate(_SITES).neuron_ranks is None
    assert GluTransformerCSpec.model_validate(_SITES).initialization == "random"


def test_neuron_ranks_refs_are_flat_names_or_absolute_dirs():
    with pytest.raises(ValidationError, match="flat"):
        NamedNeuronRanks.model_validate({"kind": "name", "name": "a/b"})
    with pytest.raises(ValidationError, match="absolute"):
        NeuronRanksDir.model_validate({"kind": "dir", "dir": "relative/ranks"})


def test_a_plain_run_refuses_the_targeted_init():
    raw = l18_mlp_raw()
    raw["decomposition"]["sites"]["initialization"] = "neuron_aligned_targeted"
    raw["decomposition"]["sites"]["neuron_ranks"] = {"kind": "name", "name": "x"}
    with pytest.raises(ValidationError, match="targeted-run"):
        LMExperimentConfig.model_validate(raw)


def test_a_targeted_run_admits_the_targeted_init():
    raw = _targeted_raw()
    raw["decomposition"]["sites"]["initialization"] = "neuron_aligned_targeted"
    raw["decomposition"]["sites"]["neuron_ranks"] = {"kind": "name", "name": "x"}
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    assert cfg.decomposition.sites.initialization == "neuron_aligned_targeted"
