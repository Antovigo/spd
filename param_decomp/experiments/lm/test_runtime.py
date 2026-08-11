"""The `runtime:` section: derived topology, the placement-preset vocabulary, and the
serialised shape every already-written `launch_config.yaml` was pinned with."""

import typing
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.placement import PRESET_NAMES
from param_decomp.experiments.lm.config import LMExperimentConfig
from param_decomp.experiments.lm.runtime import RuntimeConfig

CONFIGS = Path(__file__).parent / "configs"
SEAT = CONFIGS / "llama8b_l18_C49k_200k.yaml"


def test_topology_is_derived_and_fails_closed():
    def runtime(**overrides: Any) -> RuntimeConfig:
        return RuntimeConfig.model_validate({"sharding": "zero1", **overrides})

    # a sub-node world is one process over exactly dp local devices
    for dp in (1, 2, 8):
        assert not runtime(dp=dp).distributed

    # a multi-node world is one process per whole node; the node shape is itself config
    assert runtime(dp=32).distributed
    for bad_dp in (12, 20):
        with pytest.raises(ValidationError, match="multiple of gpus_per_node"):
            runtime(dp=bad_dp)
    assert runtime(dp=8, gpus_per_node=4).distributed
    assert not runtime(dp=4, gpus_per_node=4).distributed

    # dp is REQUIRED, and `dp: null` is not a mode
    with pytest.raises(ValidationError):
        runtime()
    with pytest.raises(ValidationError):
        runtime(dp=None)

    # Removed launch intent is not part of the live authoring schema.
    with pytest.raises(ValidationError):
        runtime(launch="slurm", dp=32)


def test_preset_names_match_placement_presets():
    """The authored preset vocabulary and the engine's resolver must name the same set —
    a preset that parses but has no rules (or vice versa) is a config that dies at build."""
    ann = RuntimeConfig.model_fields["sharding"].annotation
    literals = [a for a in typing.get_args(ann) if typing.get_origin(a) is typing.Literal]
    assert literals, ann
    assert set(typing.get_args(literals[0])) == set(PRESET_NAMES)


def test_sharding_accepts_an_explicit_placement_table():
    """The non-preset arm of `sharding` is reachable through the section, not just as a
    standalone type: an authored table lands as `PlacementTableConfig`."""
    table = {
        "params": {
            "persist": {"stack": "replicate", "d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
            "zero1": {"d_in": ["fsdp", "replicate"], "d_out": ["fsdp", "replicate"], "C": "tp"},
            "forward": {"d_in": "fsdp", "d_out": "fsdp", "C": "tp"},
        },
        "activations": {"batch": ["replicate", "fsdp"], "C": "tp"},
    }
    runtime = RuntimeConfig.model_validate({"dp": 1, "sharding": table})
    assert isinstance(runtime.sharding, PlacementTableConfig)
    assert runtime.sharding.params.zero1 is not None


_PINNED_LAUNCH_ENV = {
    "xla_python_client_mem_fraction": 0.85,
    "xla_python_client_allocator": "platform",
    "xla_pjrt_gpu_host_memory_limit_gb": 512,
    "nccl_debug": "INFO",
    "malloc_arena_max": 4,
    "env": {"SOME_ONE_OFF_VAR": "1"},
}


def test_a_pinned_launch_config_still_round_trips(tmp_path: Path):
    """Old runs are replayed by re-reading the `launch_config.yaml` they were pinned with
    (`adapters/pd.py`, `load_run.py`), and `extra="forbid"` makes any drift in the section
    layout a hard parse failure. `runtime:` is a section of the experiment config and
    `runtime.launch_env:` nests inside it — moving either would strand every stored run."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["runtime"]["launch_env"] = _PINNED_LAUNCH_ENV
    pinned = tmp_path / "launch_config.yaml"
    pinned.write_text(yaml.safe_dump(raw, sort_keys=False))

    cfg = LMExperimentConfig.from_file(pinned)

    assert cfg.runtime.dp == 32 and cfg.runtime.sharding == "owner+zero1"
    assert cfg.runtime.launch_env.as_env() == {
        "NCCL_DEBUG": "INFO",
        "MALLOC_ARENA_MAX": "4",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": "512",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "SOME_ONE_OFF_VAR": "1",
    }
    dumped = cfg.model_dump(mode="json")
    assert dumped["runtime"]["launch_env"] == _PINNED_LAUNCH_ENV
    assert "launch_env" not in dumped and "runtime" not in dumped["pd"]


def test_the_launch_env_block_has_no_second_home(tmp_path: Path):
    """Anti-vacuity for the test above: the nesting is load-bearing, not incidental —
    hoisting the block anywhere else is refused, so a silent migration cannot pass."""
    raw = yaml.safe_load(SEAT.read_text())
    raw["launch_env"] = _PINNED_LAUNCH_ENV
    hoisted = tmp_path / "launch_config.yaml"
    hoisted.write_text(yaml.safe_dump(raw, sort_keys=False))

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LMExperimentConfig.from_file(hoisted)
