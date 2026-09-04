"""Bootstrap the process environment, then harvest the neuron-ranking artifact
(`neuron_ranks_harvest.harvest`; SPEC T13) — the `run_targeted` pattern: the run YAML's
`runtime.launch_env` must be in place before anything imports jax."""

import os
from pathlib import Path

import fire
import yaml

from param_decomp.experiments.lm.runtime import RuntimeConfig


def main(
    config: Path,
    data_root: Path,
    out_dir: Path,
    local_device_count: int,
    layers: str | int | list[int] = "all",
    batch_size: int = 128,
) -> None:
    runtime = RuntimeConfig.model_validate(yaml.safe_load(Path(config).read_text())["runtime"])
    os.environ.update(runtime.launch_env.as_env())
    from param_decomp.experiments.lm.neuron_ranks_harvest import harvest

    harvest(
        Path(config),
        Path(data_root),
        Path(out_dir),
        local_device_count,
        layers=layers,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    fire.Fire(main)
