"""The run's target-stream loss at its FINAL training step, held constant for the filter.

A run of `S` steps (integer, `S >= 2`) evaluates every schedule at the run fraction
`t = k / (S - 1)` of its 0-based step `k` (`core.schedule`), so its last update (`k = S - 1`)
reads every schedule at `t = 1`. `resolve_final` reads the run's pinned `pd` section and
returns each coefficient there, from the terms that survive without the hidden and non-target
passes: the merged stochastic-subset persistent-PGD reconstruction (`coeff`, `adv_fraction`,
warmup ascents, source Adam settings) and the importance minimality (activity coefficient,
`gamma`, frequency coefficient and its reference datapoint count `a'`). Every other term either
does not depend on the CI fn (its gradient there is zero with the components frozen) or belongs
to a pass the filter does not run; `FinalCoefficients.dropped` lists each with the reason.

`a'` (integer) normalizes the frequency penalty `f log2(1 + a' f)` (`f` in [0, 1], a
component's firing frequency over one call's `B·T` CI entries). The run writes it `auto`: each
call's own `B·T`, here the target batch `B` (integer) times the prompt length `T` (integer).
The filter holds `a'` FIXED at that value, so its penalty per firing rate is the run's whatever
the filter's (micro)batch."""

from dataclasses import asdict, dataclass
from pathlib import Path

from param_decomp.ci_filter.config import (
    CIFilterConfig,
    ImpMinConfig,
    LastPositionIntegerKL,
    MergedPPGDRecon,
    RunDirRef,
)
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    TargetedPDConfig,
    UniformKSubsetRoutingConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.schedule import ScheduleConfig, get_scheduled_value
from param_decomp.integer_only.adv.config import IntegerOnlyAdvConfig

TARGET_TERM = "MergedStochasticSubsetPPGDReconLoss"
"""The run's target-stream merged term's name, which keys its adversary in the checkpoint."""


def value_at_end(coeff: LossCoeff, steps: int) -> float:
    """A loss coefficient (a float constant or a schedule) at the run's last step `S - 1`."""
    match coeff:
        case float() | int():
            return float(coeff)
        case ScheduleConfig():
            return get_scheduled_value(steps - 1, steps, coeff)


@dataclass(frozen=True)
class FinalCoefficients:
    """Every coefficient the filter holds constant, at the run's last step."""

    run_steps: int
    """`S`: the run's training steps; the values are at step `S - 1` (run fraction 1)."""
    run_batch: int
    """The run's target batch `B`: rows of its persistent adversary."""
    recon_coeff: float
    adv_fraction: float
    n_warmup_steps: int
    source_beta1: float
    source_beta2: float
    source_eps: float
    source_lr: float
    impmin_coeff: float
    gamma: float
    normalize_at_one: bool
    frequency_coeff: float | None
    reference_datapoint_count: int | None
    """`a'`; None when the run has no frequency penalty."""
    ci_betas: tuple[float, float]
    ci_grad_clip_norm: float | None
    dropped: dict[str, str]
    """Each loss term or setting of the run the filter does not carry, with the reason."""

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def resolve_final(pd: TargetedPDConfig, seq_len: int) -> FinalCoefficients:
    """The final-step coefficients of the run's target stream; `seq_len` is the prompt length
    `T` (every pool prompt's token count)."""
    steps = pd.steps
    dropped: dict[str, str] = {}
    merged: list[MergedStochasticSubsetPPGDReconLossConfig] = []
    minimality: list[ImportanceMinimalityLossConfig] = []
    for term in pd.loss_metrics:
        match term:
            case MergedStochasticSubsetPPGDReconLossConfig():
                merged.append(term)
            case ImportanceMinimalityLossConfig():
                minimality.append(term)
            case UnmaskedReconLossConfig() | NonlinearityLocalityLossConfig():
                dropped[term.name or term.type] = (
                    "independent of the CI fn (every component on / a prior on U alone): zero "
                    "gradient with the components frozen"
                )
            case other:
                raise AssertionError(f"target-stream term {other.type!r} has no filter mapping")
    assert len(merged) == 1 and len(minimality) == 1, (merged, minimality)
    (recon,), (imp,) = merged, minimality
    assert (recon.name or recon.type) == TARGET_TERM, recon.name
    assert isinstance(recon.optimizer, AdamPGDConfig), recon.optimizer
    assert isinstance(recon.routing, UniformKSubsetRoutingConfig), recon.routing
    assert recon.source_shape == "bsc" and recon.source_dtype == "float32", recon
    assert not recon.auxiliaries, "reconstruction auxiliaries have no filter mapping"
    frequency = imp.frequency
    if frequency is not None:
        assert frequency.ema_halflife_steps is None, "the filter's frequency penalty is per batch"
        a_prime = frequency.reference_datapoint_count
        a_prime = pd.batch_size * seq_len if a_prime == "auto" else a_prime
    else:
        a_prime = None
    ci_opt = pd.ci_fn_optimizer
    assert isinstance(ci_opt, AdamWOptimizerConfig), ci_opt
    assert ci_opt.weight_decay == 0.0, ci_opt
    if pd.hidden is not None:
        dropped["hidden"] = "the hidden-activation pass (its recon, rider and imp-min) is not run"
    dropped["nontarget"] = "the non-target pass is not run"
    if pd.ci_scaled_weight_decay is not None:
        dropped["ci_scaled_weight_decay"] = "decays the components, which are frozen"
    dropped["components_optimizer"] = "the components are frozen"
    dropped["ci_fn_optimizer.lr_schedule"] = "replaced by the filter's constant learning rate"
    return FinalCoefficients(
        run_steps=steps,
        run_batch=pd.batch_size,
        recon_coeff=value_at_end(recon.coeff, steps),
        adv_fraction=value_at_end(recon.adv_fraction, steps),
        n_warmup_steps=recon.n_warmup_steps,
        source_beta1=recon.optimizer.beta1,
        source_beta2=recon.optimizer.beta2,
        source_eps=recon.optimizer.eps,
        source_lr=value_at_end(recon.optimizer.lr_schedule, steps),
        impmin_coeff=value_at_end(imp.coeff, steps),
        gamma=value_at_end(imp.gamma, steps),
        normalize_at_one=imp.normalize_at_one,
        frequency_coeff=None if frequency is None else value_at_end(frequency.coeff, steps),
        reference_datapoint_count=a_prime,
        ci_betas=ci_opt.betas,
        ci_grad_clip_norm=ci_opt.grad_clip_norm,
        dropped=dropped,
    )


def filter_config(
    coefficients: FinalCoefficients, config: IntegerOnlyAdvConfig, run_dir: Path
) -> CIFilterConfig:
    """The `CIFilterConfig` the training step runs: every schedule constant at its final-step
    value, `KL_int` at the last position (the restricted KL over the answer ids the caller
    passes: the 1000 integer tokens), the merged term with the weight delta adversarially
    ablated as in the run's target stream, the ceiling on
    and the prune
    as `config.prune` says (at `config.threshold`). Its `pool` is unused (the pool is the prompts file's)."""
    c = coefficients
    return CIFilterConfig(
        run=RunDirRef(dir=run_dir),
        step=config.step,
        objective=LastPositionIntegerKL(include_minus=False),
        steps=config.steps,
        batch_size=config.batch,
        microbatch_size=config.microbatch,
        accumulate_on_host=config.accumulate_on_host,
        seed=config.seed,
        remat=config.remat,
        recon=MergedPPGDRecon(
            coeff=c.recon_coeff,
            adv_fraction=ScheduleConfig.constant(c.adv_fraction),
            n_warmup_steps=c.n_warmup_steps,
            optimizer=AdamPGDConfig(
                beta1=c.source_beta1,
                beta2=c.source_beta2,
                eps=c.source_eps,
                lr_schedule=ScheduleConfig.constant(c.source_lr),
            ),
            weight_delta="on",
        ),
        imp_min=ImpMinConfig(
            coeff=ScheduleConfig.constant(c.impmin_coeff),
            gamma=ScheduleConfig.constant(c.gamma),
            normalize_at_one=c.normalize_at_one,
            frequency_coeff=None
            if c.frequency_coeff is None
            else ScheduleConfig.constant(c.frequency_coeff),
            **(
                {}
                if c.reference_datapoint_count is None
                else {"reference_datapoint_count": c.reference_datapoint_count}
            ),
        ),
        lr_schedule=ScheduleConfig.constant(config.lr),
        betas=c.ci_betas,
        grad_clip_norm=c.ci_grad_clip_norm,
        prune=config.prune,
        alive_threshold=config.threshold,
        eval_every=None,
        log_every=config.log_every,
        pgd_eval=None,
        wandb=None,
    )
