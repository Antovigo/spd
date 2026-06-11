"""The trainer's internal experiment config — built ONLY by `torch_config.py`.

The yaml surface is the shared torch schema (`LMExperimentConfig` via the wrapper
route); this module holds the converted form the trainer consumes. Loss/adversary
configs are the SHARED pydantic types passed through verbatim; the dataclasses here
carry only the jax-runtime knobs that have no torch-schema home (chunking, remat,
checkpoint cadence, the CI-fn architecture extraction).
"""

from dataclasses import dataclass
from pathlib import Path

from jax_single_pool.ci_fn import CIArch
from jax_single_pool.lm import SiteC
from jax_single_pool.train import AdversaryConfig
from param_decomp_config.experiment import WandbConfig
from param_decomp_config.losses import ImportanceMinimalityLossConfig


@dataclass(frozen=True)
class TargetConfig:
    """The Llama-3.1-8B HF target (`llama8b.py`)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` pile-pretrained target (`llama_simple_mlp.py`); weights
    from the torch pretrain cache resolved from `pretrain_run_path`."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""


AnyTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig


@dataclass(frozen=True)
class DataConfig:
    dir: Path
    seq_len: int
    global_batch: int


@dataclass(frozen=True)
class ReconConfig:
    sites_per_chunk: int
    n_samples: int
    remat_forwards: bool


@dataclass(frozen=True)
class VUOptimizerConfig:
    lr: float
    grad_clip_norm: float


@dataclass(frozen=True)
class CIOptimizerConfig:
    lr: float


@dataclass(frozen=True)
class FaithWarmupConfig:
    steps: int
    lr: float


@dataclass(frozen=True)
class DenseLogPhase:
    every: int
    until_step: int


@dataclass(frozen=True)
class CadenceConfig:
    log_every: int
    save_every: int
    keep_last: int
    dense_log_phase: DenseLogPhase | None


@dataclass(frozen=True)
class EvalPGDConfig:
    """Fresh sign-PGD recon probe (torch eval `PGDReconLoss`: init random, source
    shared across batch and positions)."""

    n_steps: int
    step_size: float


@dataclass(frozen=True)
class EvalConfig:
    """In-loop eval pass (torch `EvalLoop` analog, scalar metrics only — plots ride the
    offline export path). `rounding_threshold` binarises CI for the CE/KL
    `rounded_masked` variant; `ci_alive_threshold` is the CI-L0 aliveness cutoff."""

    batch_size: int
    every: int
    n_steps: int
    rounding_threshold: float
    ci_alive_threshold: float
    l0_groups: dict[str, tuple[str, ...]] | None
    """torch CI_L0 `groups`: fnmatch site patterns whose member L0s sum into a
    group-named key. None = per-site keys only."""
    pgd: EvalPGDConfig | None


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name) — the torch
    `generate_run_id` convention, making the run a first-class citizen of the
    `runs/<id>/` postprocess world. Minted at submit time by `pd-jax-lm`."""
    out_dir: Path
    seed: int
    steps: int
    target: AnyTargetConfig
    data: DataConfig
    faith_coeff: float
    stoch_coeff: float
    imp_min: ImportanceMinimalityLossConfig
    adversary: AdversaryConfig
    """Recon adversary: persistent source-Adam (PPGD) or fresh per-batch sign-PGD —
    the shared torch loss config, subset-asserted in `make_train_step`."""
    recon: ReconConfig
    vu_optimizer: VUOptimizerConfig
    ci_optimizer: CIOptimizerConfig
    ci_fn: CIArch
    faith_warmup: FaithWarmupConfig
    cadence: CadenceConfig
    eval: EvalConfig | None
    wandb: WandbConfig | None

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.run_id
