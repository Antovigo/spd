"""Configuration and summary schemas of one CI filter run."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Discriminator, NonNegativeInt, PositiveFloat, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.configs import AdamPGDConfig
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


class LastPositionAnswerCE(BaseConfig):
    """Cross-entropy of the TRUE arithmetic result at the last prompt position, over the
    answer tokens only (restricted logits, renormalized): `-log p(first token of a+b)`.

    Unlike the KL objectives this does not score faithfulness to the target model — it scores
    getting the arithmetic RIGHT, so the surviving components are the ones that compute the
    answer rather than the ones the model happens to use, and a filter may end up MORE accurate
    than the model it decomposes."""

    kind: Literal["last_position_answer_ce"] = "last_position_answer_ce"
    include_minus: bool = True


Objective = Annotated[
    LastPositionKL | LastPositionIntegerKL | LastPositionAnswerCE, Discriminator("kind")
]


class CIMaskedRecon(BaseConfig):
    """The objective under the deterministic masks `lower-leaky CI`, weight delta OFF."""

    kind: Literal["ci_masked"] = "ci_masked"
    coeff: PositiveFloat = 1.5
    """1.5 is the decomposition's output-recon coefficient."""


def _ramp(max_val: float, ramp_until: float) -> ScheduleConfig:
    return ScheduleConfig(
        max_val=max_val,
        points=(Knot(at=0.0, frac=0.0), Knot(at=ramp_until, frac=1.0), Knot(at=1.0, frac=1.0)),
    )


class MergedPPGDRecon(BaseConfig):
    """The decomposition's own `MergedStochasticSubsetPPGDReconLoss` (SPEC S34), scored by the
    filter's objective. Per step and batch element, `adv_fraction` of the rows take the
    persistent adversary's sources routed through every site, the rest fresh `U[0,1]` sources
    routed per a uniform-k site subset; masks are `ci + (1 - ci) * source` and the WEIGHT
    DELTA IS ON (its mask is the source's delta channel). The adversary keeps one source per
    (batch slot, position, component + delta), Adam-ascended `n_warmup_steps` times per step
    against the all-adversarial forward with the CI detached, then once more from the main
    backward. Defaults are addsub-all-layers-4xh100-05's target-pass values, except
    `n_warmup_steps` (-05: 2): the arithmetic pool repeats, so the final ascent alone may keep
    the adversary current."""

    kind: Literal["merged_stochastic_ppgd"] = "merged_stochastic_ppgd"
    coeff: PositiveFloat = 1.5
    adv_fraction: ScheduleConfig = _ramp(0.3333333, 0.05)
    n_warmup_steps: NonNegativeInt = 0
    optimizer: AdamPGDConfig = AdamPGDConfig(
        beta1=0.5, beta2=0.99, eps=1.0e-8, lr_schedule=_ramp(0.01, 0.0125)
    )

    @model_validator(mode="after")
    def validate_adv_fraction(self) -> Self:
        assert self.adv_fraction.max_val <= 1.0, self.adv_fraction
        return self


Recon = Annotated[CIMaskedRecon | MergedPPGDRecon, Discriminator("kind")]


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
    """The decomposition's fresh sign-PGD reconstruction eval (`PGDReconLoss`) scored like the
    filter trains: per-component sources shared by the batch (`source_shape: c`), random init,
    `n_steps` sign-gradient ascents of `step_size` on the masks `ci + (1 - ci) * source` AND the
    weight-delta channel, scored by the filter's objective at the LAST position (the
    decomposition's own eval averaged the full KL over every position). Scores the starting CI fn
    before the first step and the final one after the grid, on the same fixed
    `n_batches x batch_size` pool prompts; reported as the batch mean."""

    n_steps: PositiveInt = 20
    step_size: PositiveFloat = 0.1
    n_batches: PositiveInt = 4
    batch_size: PositiveInt = 128


class WandbConfig(BaseConfig):
    """Live wandb logging; the wandb run is named by the filter id (its wandb id is fresh)."""

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
    ci_ceiling: bool = False
    """CI can only decrease from the starting point's: per prompt, position and component, the
    CI every consumer reads (training masks, imp-min, evaluations, alive list, grid) is the
    minimum of the trained CI fn's and the STARTING CI's, so the narrower objective can only
    switch components off. The starting CI is the decomposition's CI fn for `init: run`, and the
    source filter's effective CI for `init: ci_filter` (its CI fn, capped by its own constraints
    when it had `ci_ceiling`). Each ceiling holds one frozen compute-precision copy of a CI fn
    and costs one extra CI forward (no backward) wherever CI is read. Over the cap only a
    gradient that lowers the CI reaches the trained fn."""
    prune_dead: bool = False
    """Remove, before the first step, every component whose STARTING CI (with the start's own
    constraints) never exceeds `alive_threshold` at any position of any pool prompt: its CI is
    forced to 0 and its U and V are zeroed, so it cannot come back, and with the weight delta on
    (PPGD) its weight lies in the delta `W - UV` as if deleted. Removal is inherited: a filter
    initialized from one that removed components keeps them removed. Shapes are unchanged
    (removal is by zeroing), so this does not make the step cheaper."""
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
    eval_batch_size: PositiveInt = 1000
    seed: int = 0
    remat: bool = True

    recon: Recon = CIMaskedRecon()
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
    answer_ce: float
    """Cross-entropy of the TRUE result over the answer tokens (what `last_position_answer_ce`
    trains); scored for every objective."""
    accuracy: float
    """Fraction of prompts whose restricted argmax IS the true result."""
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
    """Every component on (reference: the decomposition with the delta off); every KEPT one
    when components were removed."""
    n_kept: int | None = None
    """Components not removed (`prune_dead`, here or upstream); `None` when none were. A
    `prune_dead` filter logs two step-0 evaluations: before removal (`None`, jsonl only) and
    after."""
    pgd_recon: float | None = None
    """`PGDEvalConfig`'s adversarial reconstruction KL (final evaluation only; the starting
    CI fn's value is in `eval/pgd_recon.json`)."""
