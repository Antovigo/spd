"""Configuration of one integer-only global-mask pipeline (plan: `notes/integer_only/plan.md`).

Notation: a run decomposes every linear site of the target into components `c`; `ℓ_c` (real)
is the learned logit of candidate `c`, `σ` the logistic function, `T` (positive real) the
surrogate temperature and `λ` (non-negative real, nats per kept component) the sparsity
weight. The defaults are the settings of the addsub-08 learned mask (`p-ba8b0c08`, sweep
`kl_sampled`), with the reference KL measured on that mask."""

from pathlib import Path
from typing import Literal, Self

from param_decomp_formats.config import BaseConfig
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

from param_decomp.integer_only.layout import lam_name

type Precision = Literal["fp32", "tf32", "bf16"]
type Operation = Literal["add", "sub"]


class IntegerOnlyConfig(BaseConfig):
    step: int
    """The decomposition checkpoint step the pipeline reads."""
    prompts: Path
    """The prompt pool: one prompt per nonempty line, every line one of `templates` filled
    with two integer operands `a`, `b`; relative paths resolve against the working directory."""
    templates: dict[Operation, str]
    """Per operation, the prompt text with `{a}` and `{b}` placeholders (e.g. `"Answer: {a}
    plus {b} is "`). The answer is the true result `r` of the operation on `(a, b)`."""
    threshold: PositiveFloat = 0.1
    """A component is ON at position `t` of a prompt when its original output-head CI there
    exceeds this; candidates are the components on at some position of some prompt."""
    seed: int = 0
    check_size: PositiveInt = 32768
    """`|Q|`, the fixed random check set drawn from `P` (the prompts with `r >= 0`)."""

    lambdas: tuple[NonNegativeFloat, ...] = (5e-5, 1e-4, 2e-4, 5e-4)
    """The sparsity weights `λ` of the sweep, one mask each."""
    steps: PositiveInt = 3000
    batch: PositiveInt = 512
    """Prompts of `P \\ Q` per Adam step."""
    microbatch: PositiveInt = 512
    """Prompts per gradient pass; `batch` is a multiple of it."""
    lr: PositiveFloat = 0.01
    schedule: Literal["constant", "cosine"] = "cosine"
    """`cosine`: the learning rate at 0-based step `k` is `lr · (1 + cos(π k / steps)) / 2`."""
    init: float = 4.59512
    """Initial `ℓ_c` of every candidate: `ln 99`, so `σ(ℓ_c / T) = 0.99` at `T = 1`."""
    temperature: PositiveFloat = 1.0
    eval_every: PositiveInt = 100
    eval_size: PositiveInt = 4096
    """Prompts of `Q` (its first ones in `Q`'s order) of the periodic probe."""
    precision: Precision = "bf16"
    """Site and head matmuls of the factored model while learning and scoring."""
    target_precision: Precision = "bf16"
    """Site and head matmuls of the dense target (the reference distribution)."""
    collect_precision: Precision = "fp32"
    """Site and head matmuls of the integer-only model in the collection."""

    ci_batch: PositiveInt = 256
    """Prompts per forward of the CI passes (stages 1 and 5)."""
    score_batch: PositiveInt = 1024
    """Prompts per forward-only pass of the factored model or the target."""

    reference_kl_int: PositiveFloat = 0.16479748860001564
    """The λ rule's bar: the highest `λ` whose mask has mean `KL_int` to the target on `Q`
    below this is chosen. Default: the addsub-08 integer-only mask's value."""
    reference_source: str = (
        "p-ba8b0c08 analysis/pruning/step_40000/addsub-08-prune/learn/kl_table.json "
        "kl_sampled/lam_5e-05.kl_int_llama"
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        assert self.templates, "no templates"
        for op, text in self.templates.items():
            assert text.count("{a}") == 1 and text.count("{b}") == 1, (op, text)
        assert self.batch % self.microbatch == 0, (self.batch, self.microbatch)
        names = [lam_name(lam) for lam in self.lambdas]
        assert len(set(names)) == len(names), f"λ values share an output name: {names}"
        assert self.eval_size <= self.check_size, (self.eval_size, self.check_size)
        return self


LEARN_FIELDS = (
    "step", "threshold", "seed", "check_size", "steps", "batch", "lr", "schedule", "init",
    "temperature", "eval_every", "eval_size", "precision", "target_precision",
)  # fmt: skip
"""Fields that shape one `λ`'s outputs: pinned per `λ` in `learn_config.json`, so a resume under
other settings fails."""
