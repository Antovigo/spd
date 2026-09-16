"""Configuration and summary schemas of one CI filter run."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Discriminator, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.schedule import Knot, ScheduleConfig

# ----------------------------------- references -----------------------------------


class RunIdRef(BaseConfig):
    """A decomposition run by id, resolved as `<data_root>/runs/<id>`."""

    kind: Literal["id"] = "id"
    id: str


class RunDirRef(BaseConfig):
    """Escape arm: a run directory location (e.g. a backup outside `<data_root>/runs`)."""

    kind: Literal["dir"] = "dir"
    dir: Path


RunRef = Annotated[RunIdRef | RunDirRef, Discriminator("kind")]


def resolve_run_dir(ref: RunIdRef | RunDirRef, data_root: Path) -> Path:
    match ref:
        case RunIdRef(id=run_id):
            return data_root / "runs" / run_id
        case RunDirRef(dir=run_dir):
            return run_dir.expanduser()


class InitFromRun(BaseConfig):
    """Start from the CI fn of the decomposition's own checkpoint."""

    kind: Literal["run"] = "run"


class InitFromCIFilter(BaseConfig):
    """Start from the trained CI fn of an earlier CI filter of the SAME run and step
    (`<run_dir>/analysis/ci_filter/step_<step>/<id>/`)."""

    kind: Literal["ci_filter"] = "ci_filter"
    id: str


CIInit = Annotated[InitFromRun | InitFromCIFilter, Discriminator("kind")]


# ----------------------------------- objective -----------------------------------


Operation = Literal["add", "sub"]


class ArithmeticPoolConfig(BaseConfig):
    """The `operations x [a_range] x [b_range]` pool of `"<a><op><b>="` prompts, in
    operation order, each operation's block row-major `(a, b)`."""

    operations: tuple[Operation, ...] = ("add", "sub")
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        assert self.operations and len(set(self.operations)) == len(self.operations), (
            self.operations
        )
        for lo, hi in (self.a_range, self.b_range):
            assert lo <= hi, (self.a_range, self.b_range)
        return self


class LastPositionKL(BaseConfig):
    """`KL(clean || masked)` of the full next-token distribution at the last prompt position."""

    kind: Literal["last_position_kl"] = "last_position_kl"


class LastPositionIntegerKL(BaseConfig):
    """`KL(clean || masked)` at the last prompt position between the two distributions
    RESTRICTED to the answer tokens and renormalized over them: every all-digit token
    (`0`-`999` for Llama-3) plus the minus sign that starts a negative answer."""

    kind: Literal["last_position_integer_kl"] = "last_position_integer_kl"
    include_minus: bool = True


Objective = Annotated[LastPositionKL | LastPositionIntegerKL, Discriminator("kind")]


class ImpMinConfig(BaseConfig):
    """Geman-McClure smooth-L0 imp-min on the upper-leaky output CI (the decomposition's
    own term, `core.losses.importance_minimality_terms`), plus its frequency penalty."""

    coeff: ScheduleConfig
    gamma: ScheduleConfig
    normalize_at_one: bool = True
    frequency_coeff: ScheduleConfig | None = None
    reference_datapoint_count: PositiveInt = 640
    """`a'` of the frequency penalty, held FIXED so the curvature at a given firing rate does
    not move with the (micro)batch size. 640 = the decomposition's own target batch
    (128 prompts x 5 tokens), which reproduces its pressure per firing rate."""

    @model_validator(mode="after")
    def validate_gamma(self) -> Self:
        assert all(knot.frac > 0.0 for knot in self.gamma.points), "gamma must stay > 0"
        return self


class GridConfig(BaseConfig):
    """The `(a, b)`-grid applet written at the end: one lazily loaded file per
    (operation, position)."""

    mean_ci_floor: Probability = 0.05
    """A component's per-prompt grid is saved for a slice iff its prompt-mean CI there
    reaches this floor (the mean-CI vectors are saved for every component)."""
    chunk_prompts: PositiveInt = 1000


class PGDEvalConfig(BaseConfig):
    """The decomposition's own fresh sign-PGD reconstruction eval (`PGDReconLoss`, as
    addsub-all-layers-4xh100-05 logged it): per-component sources shared by the batch
    (`source_shape: c`), random init, `n_steps` sign-gradient ascents of `step_size` on the
    masks `ci + (1 - ci) * source` AND the weight-delta channel, scored by the output KL over
    every position. Scores the starting and the final CI fn on the same fixed
    `n_batches x batch_size` pool prompts, both at the END of the run (the starting fn waits
    in host memory); reported as the batch mean."""

    n_steps: PositiveInt = 20
    step_size: PositiveFloat = 0.1
    n_batches: PositiveInt = 4
    batch_size: PositiveInt = 128


class WandbConfig(BaseConfig):
    """Live wandb logging; the wandb run id is the filter id."""

    project: str
    entity: str | None = None
    """`None` resolves `WANDB_ENTITY` / the logged-in default."""


def _constant_then_cosine(max_val: float) -> ScheduleConfig:
    return ScheduleConfig(
        max_val=max_val, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.1, interp="cosine"))
    )


class CIFilterConfig(BaseConfig):
    """Fine-tune the output-role CI fn of a FROZEN decomposition against a narrower output
    objective, so components that objective does not need are switched off.

    The forward is masked by the deterministic lower-leaky CI with the weight delta OFF
    (a component whose CI is 0 contributes nothing); only the prompt pool's target stream
    is used, with no hidden-activation or non-target pass."""

    run: RunRef
    step: NonNegativeInt | None = None
    """Checkpoint step; `None` is the latest."""
    init: CIInit = InitFromRun()
    pool: ArithmeticPoolConfig = ArithmeticPoolConfig()
    objective: Objective

    steps: PositiveInt = 5000
    batch_size: PositiveInt = 1024
    microbatch_size: PositiveInt | None = None
    """Prompts per forward/backward; gradients are summed over `batch_size / microbatch_size`
    microbatches. `None` runs the whole batch at once. Hardware-only: the objective is a
    per-prompt mean and the frequency penalty's `a'` is fixed, so it does not change the
    optimization beyond the frequency term's per-microbatch estimate."""
    accumulate_on_host: bool = False
    """Hold the running gradient sum in host memory rather than on device: every microbatch
    but the last is copied off device, and the sum comes back once per step. Hardware-only
    (the summed gradient is the same up to float reassociation): it frees one CI-fn-sized
    gradient tree (~3.9 GB for the 8B run) at the cost of two PCIe transfers per step."""
    recompute_ci_inputs: bool = False
    """Split each microbatch's backward in two: the loss's cotangent on the CI values first,
    then a second clean forward recaptures the CI fn's inputs to pull it back through the CI
    fn. Hardware-only (same gradient): the captured inputs (~8.5 MB per prompt for the 8B run's
    `all_block_taps`) no longer sit in memory through the masked backward, at the cost of one
    extra frozen forward per microbatch."""
    eval_batch_size: PositiveInt = 1000
    seed: int = 0
    remat: bool = True

    recon_coeff: PositiveFloat = 1.5
    """Weight of the objective's KL; 1.5 is the decomposition's output-recon coefficient."""
    imp_min: ImpMinConfig
    lr_schedule: ScheduleConfig = _constant_then_cosine(4.0e-5)
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: PositiveFloat | None = None

    alive_threshold: Probability = 0.01
    """A component is alive iff its output CI exceeds this at any position of any prompt;
    the same cut rounds CI to binary masks in the evaluation."""
    eval_every: PositiveInt | None = 1000
    """Pool evaluation cadence in steps (always run before the first and after the last step)."""
    log_every: PositiveInt = 10
    pgd_eval: PGDEvalConfig | None = PGDEvalConfig()
    wandb: WandbConfig | None = None
    grid: GridConfig = GridConfig()
    compilation_cache_dir: Path | None = None

    @model_validator(mode="after")
    def validate_batching(self) -> Self:
        if self.microbatch_size is not None:
            assert self.batch_size % self.microbatch_size == 0, (
                f"batch_size {self.batch_size} must be a multiple of microbatch_size "
                f"{self.microbatch_size}"
            )
        return self

    @property
    def n_microbatches(self) -> int:
        return 1 if self.microbatch_size is None else self.batch_size // self.microbatch_size


# ------------------------------------ summaries ------------------------------------


class MaskScores(BaseConfig):
    """Pool means of one masking of the pool (every prompt, scored at its last position)."""

    kl: float
    """Full-vocabulary KL."""
    integer_kl: float
    """KL restricted to (and renormalized over) the answer tokens."""
    top1: float
    """Agreement of the full-vocabulary argmax with the clean model's."""
    integer_top1: float
    """Agreement of the argmax over answer tokens."""
    max_prompt_objective: float
    """The worst single prompt's value of the run's training objective."""


class PoolEval(BaseConfig):
    step: int
    ci: MaskScores
    """Continuous lower-leaky CI masks (what training sees)."""
    rounded: MaskScores
    """Per-prompt, per-position binary masks `CI > alive_threshold`."""
    l0_per_token: float
    """Mean number of components with `CI > alive_threshold` per prompt position."""
    l0_per_token_last: float
    """The same at the last position only."""
    n_alive: int
    alive_per_site: dict[str, int]
    alive_set: MaskScores | None = None
    """Every alive component on for every prompt and position (a global mask)."""
    all_on: MaskScores | None = None
    """Every component on (reference: the decomposition with the delta off)."""
    pgd_recon: float | None = None
    """`PGDEvalConfig`'s adversarial reconstruction KL (final evaluation only; the starting
    CI fn's value is in `eval/pgd_recon.json`)."""
