"""Configuration of the integer-only-adv CI filter (plan: `notes/integer_only/plan_adv.md`).

The filter fine-tunes a finished run's CI fn (its components frozen) on the integer-only
pipeline's prompt pool against `KL_int` at the last position under the run's own merged
stochastic-subset persistent-PGD term with the weight delta off. Every loss coefficient is the
run's own at its final training step (`coefficients.py`); this config holds only the settings
the run does not fix."""

from typing import Literal

from param_decomp_formats.config import BaseConfig
from pydantic import PositiveFloat, PositiveInt, model_validator


class IntegerOnlyAdvConfig(BaseConfig):
    step: int
    """The decomposition checkpoint step the filter starts from (CI fn, components and, with
    `adversary_init: run`, the run's target-stream persistent adversary)."""
    threshold: PositiveFloat = 0.1
    """A component is ON at position `t` of a prompt when its fine-tuned CI there exceeds
    this; the integer-only-adv components are those ON at some position of some pool prompt."""
    seed: int = 0
    steps: PositiveInt = 2500
    batch: PositiveInt = 4096
    """Prompts per CI-fn Adam step, drawn from `P \\ Q` (integer-only stage 2's sets)."""
    microbatch: PositiveInt = 256
    """Prompts per forward/backward; `batch` is a multiple of it. Hardware-only."""
    accumulate_on_host: bool = False
    """Hold the running gradient sum in host memory (`CIFilterConfig.accumulate_on_host`)."""
    lr: PositiveFloat = 8e-5
    """The CI fn's learning rate, constant over every step."""
    adversary_init: Literal["run", "fresh"] = "run"
    """`run`: the persistent adversary starts from the run's target-stream adversary at `step`
    (sources and Adam moments); `fresh`: `U[0,1]` sources and zero moments."""
    remat: bool = True
    placement: Literal["ddp", "zero1"] = "ddp"
    """How the `train` stage places the restored run on its mesh (hardware-only): `ddp` keeps
    the target, the components and the CI fn whole on every device (no weight gathers in the
    loop; the batch is split over the devices), `zero1` shards them (`restore_jax_run`'s
    default; every forward gathers the weights)."""
    save_every: PositiveInt = 250
    """Training steps between resumable state saves."""
    log_every: PositiveInt = 10
    ci_batch: PositiveInt = 256
    """Prompts per forward of the CI passes (alive list and collection)."""

    @model_validator(mode="after")
    def validate_batching(self) -> "IntegerOnlyAdvConfig":
        assert self.batch % self.microbatch == 0, (self.batch, self.microbatch)
        return self


MUTABLE = {"microbatch", "accumulate_on_host", "save_every", "log_every", "ci_batch", "placement"}
"""Fields a later stage or a resume may change: hardware-only settings."""
