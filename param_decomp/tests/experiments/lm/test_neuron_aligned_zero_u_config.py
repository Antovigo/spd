"""`initialization: neuron_aligned_zero_u` at the config boundary: it reads the SAME
harvested artifact as `neuron_aligned_targeted`, so the `neuron_ranks` reference is required
by — and only by — the two of them, and a plain (non-targeted) run refuses it."""

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.configs import NamedNeuronRanks
from param_decomp.experiments.lm.config import (
    GluTransformerCSpec,
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    SimpleMlpCSpec,
)
from param_decomp.experiments.lm.resolved import ALIGNED_INITIALIZATIONS
from param_decomp.tests.experiments.test_repo_configs_parse import LM_CONFIG_PATHS

_SITES: dict[str, Any] = {
    "kind": "glu_transformer",
    "layers": {"kind": "list", "indices": [18]},
    "cs": {"gate": 8, "q": 4},
}
_MLP_SITES: dict[str, Any] = {
    "kind": "simple_mlp",
    "layers": {"kind": "list", "indices": [18]},
    "cs": {"c_fc": 8},
}


def test_the_artifact_is_required_by_the_zero_u_variant_too():
    with pytest.raises(ValidationError, match="neuron_ranks"):
        GluTransformerCSpec.model_validate({**_SITES, "initialization": "neuron_aligned_zero_u"})
    spec = GluTransformerCSpec.model_validate(
        {
            **_SITES,
            "initialization": "neuron_aligned_zero_u",
            "neuron_ranks": {"kind": "name", "name": "addsub-l18-8b"},
        }
    )
    assert isinstance(spec.neuron_ranks, NamedNeuronRanks)
    assert spec.initialization == "neuron_aligned_zero_u"


def test_the_simple_mlp_spec_agrees_with_the_glu_spec():
    """Both site specs carry their own copy of the validator; neither may drift."""
    with pytest.raises(ValidationError, match="neuron_ranks"):
        SimpleMlpCSpec.model_validate({**_MLP_SITES, "initialization": "neuron_aligned_zero_u"})
    spec = SimpleMlpCSpec.model_validate(
        {
            **_MLP_SITES,
            "initialization": "neuron_aligned_zero_u",
            "neuron_ranks": {"kind": "name", "name": "x"},
        }
    )
    assert spec.neuron_ranks is not None


def _seat(name: str) -> dict[str, Any]:
    [path] = [p for p in LM_CONFIG_PATHS if p.stem == name]
    return yaml.safe_load(path.read_text())


def test_a_plain_run_refuses_the_zero_u_variant():
    raw = _seat("llama8b_l18_C49k_200k")
    raw["decomposition"]["sites"]["initialization"] = "neuron_aligned_zero_u"
    raw["decomposition"]["sites"]["neuron_ranks"] = {"kind": "name", "name": "x"}
    with pytest.raises(ValidationError, match="targeted-run"):
        LMExperimentConfig.model_validate(raw)


def test_a_targeted_run_admits_the_zero_u_variant_and_resolves_its_initializer():
    raw = _seat("llama8b_l18_arith_targeted")
    raw["decomposition"]["sites"]["initialization"] = "neuron_aligned_zero_u"
    raw["decomposition"]["sites"]["neuron_ranks"] = {"kind": "name", "name": "x"}
    cfg = LMTargetedExperimentConfig.model_validate(raw)
    assert cfg.decomposition.sites.initialization == "neuron_aligned_zero_u"


def test_the_new_init_is_inside_the_aligned_set():
    """Every consumer keys off `ALIGNED_INITIALIZATIONS` — the config validators, the
    resolver's alignment assert, and the targeted trainer's artifact load. Membership is
    what wires the new init into all three, so it must not become a special case beside."""
    assert {"neuron_aligned_targeted", "neuron_aligned_zero_u"} == ALIGNED_INITIALIZATIONS
