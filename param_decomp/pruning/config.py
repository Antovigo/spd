"""Configuration of one pruning run (plan: `notes/pruning/plan.md`)."""

from pathlib import Path
from typing import Literal

from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from param_decomp.ci_filter.config import ArithmeticPoolConfig, RunRef
from param_decomp.core.base_config import BaseConfig

type PrecisionName = Literal["fp32", "tf32", "bf16"]


class PruneConfig(BaseConfig):
    run: RunRef
    step: NonNegativeInt
    filter_id: str
    """The ceiling CI filter whose alive set is `S₀` (`<run_dir>/<filter_id>-<step>/alive/`) and
    whose CI fn picks each component's top-CI prompts `H_c`."""
    pool: ArithmeticPoolConfig
    """The prompt grid; only prompts whose true answer is ≥ 1 enter `P`."""
    seed: int = 0

    sample_size: PositiveInt = 4096
    """`|R|`, the shared uniform sample of `P`."""
    check_size: PositiveInt = 32768
    """`|Q|`, the fixed random check set of the rounds' exact `CE` checks and mask gradients (an
    all-`P` pass is ~14 min on one A100, dominated by the full-vocabulary unembedding)."""
    top_ci: PositiveInt = 256
    """`|H_c|`, each component's highest-CI prompts."""
    batch_size: PositiveInt = 4096
    """Rows per forward of the screen and of the full-`P` passes."""
    hc_components_per_batch: PositiveInt = 16
    """Components scored together in one `H_c` forward (`batch_size // top_ci` at most)."""
    select_batch_size: PositiveInt = 500
    """Prompts per CI-fn evaluation in the `H_c` selection."""
    grad_batch_size: PositiveInt = 1024
    """Prompts per microbatch of the mask-gradient pass."""

    site_precision: PrecisionName = "fp32"
    """Site matmuls of the screen and of every re-ablation (plan §2.1)."""
    head_precision: PrecisionName = "fp32"
    """Unembedding of the screen and of every re-ablation."""
    full_precision: PrecisionName = "bf16"
    """Site and head matmuls of the full-`P` passes (exact `CE`, gradient, reports)."""
    target_precision: PrecisionName = "bf16"
    round_precision: PrecisionName | None = None
    """Site and head matmuls of the blockwise re-screens (`None`: the screen's)."""
    """The dense target (Llama) reference: its weights are bf16."""

    objective: Literal["ce", "ce_int"] = "ce"
    """The rounds' criterion: full-vocabulary `CE` of the answer, or `ce_int`, the `CE` among the
    integer tokens (the answer's log-probability renormalized over `V_int`)."""
    mode: Literal["global", "blockwise"] = "global"
    """`global`: rounds over every component with gradient-guided re-measurement;
    `blockwise`: block by block from L0, every decision by direct ablation (`prune.py`)."""
    z: NonNegativeFloat = 1.0
    epsilon: NonNegativeFloat | None = None
    """Per-component negligible effect (nats); required by the prune stage, set from the
    screen's `SE_c` distribution."""
    theta: PositiveFloat = 1e-3
    round_tolerance: PositiveFloat = 0.01
    global_budget: PositiveFloat = 0.03
    max_batch_fraction: PositiveFloat = 0.1
    disagreement_sigma: PositiveFloat = 3.0
    max_disagreements: NonNegativeInt = 2000
    max_rounds: PositiveInt = 30
    max_fresh_screens: NonNegativeInt = 3

    noise_check_components: NonNegativeInt = 100
    estimator_check_components: NonNegativeInt = 20
    validation_prompts: PositiveInt = 1024
    temperature_prompts: PositiveInt = 1024
    """Prompts of `R` (its first ones) the `temperature` stage fits each component on."""
    compilation_cache_dir: Path | None = None

    learn_lambdas: tuple[PositiveFloat, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    """The `learn` stage's sparsity weights `λ` (nats per component), one mask each (`learn.py`)."""
    learn_steps: PositiveInt = 600
    learn_batch: PositiveInt = 512
    """Prompts of `P \\ Q` per Adam step (in `grad_batch_size` microbatches)."""
    learn_lr: PositiveFloat = 0.02
    learn_objective: Literal["ce_int", "kl_int"] = "ce_int"
    """What the `learn` stage minimizes besides the sparsity term: `ce_int`, or `kl_int`, the KL
    from `S₀` (the alive-only model) to the masked model over the integer tokens."""
    learn_schedule: Literal["constant", "cosine"] = "constant"
    """`cosine`: the learning rate at step `k` (1-based) is `learn_lr · (1 + cos(π (k−1) /
    learn_steps)) / 2`, decaying to ~0 at the last step."""
    learn_name: str = ""
    """Subdirectory of `learn/` this sweep writes into (`""`: `learn/` itself), so sweeps with
    other settings sit beside earlier ones."""
    learn_init: float = 1.0
    """Initial logit of every component (`S₀` all on)."""
    learn_temperature: PositiveFloat = 1.0
    """`T` of the straight-through surrogate `σ(ℓ / T)`."""
    learn_eval_every: PositiveInt = 50
    learn_eval_size: PositiveInt = 4096
    """Prompts of `Q` (its first ones) of the periodic `ce_int` probe."""
