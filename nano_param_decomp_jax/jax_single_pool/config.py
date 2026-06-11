"""Typed experiment config for the generic trainer, parsed from YAML.

Every field is explicit in the YAML — no defaults here (single source of truth: the
config file). Unknown keys raise. Field names mirror the torch production yamls where
the concepts map (`llama8b_l18_b512_2pool_lr_mid.yaml`), restricted to what this
trainer supports (SPEC §2 constants are a valid instantiation).
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from jax_single_pool import llama_simple_mlp
from jax_single_pool.ci_fn import CIArch
from jax_single_pool.llama8b import mlp_family_site_cs
from jax_single_pool.lm import SiteC
from jax_single_pool.train import AdversaryConfig, ImpMinConfig, LossCoeffs, SourceAdamConfig


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
class WandbConfig:
    project: str
    entity: str | None
    """None = the API key's default entity (the torch schema's `entity: null`)."""


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str | None
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name) — the torch
    `generate_run_id` convention, making the run a first-class citizen of the
    `runs/<id>/` postprocess world. None ONLY for runs launched before the id
    scheme (the live C49k run's pinned wrapper) — remove the None arm once that
    run finishes and migrates."""
    out_dir: Path
    seed: int
    steps: int
    target: AnyTargetConfig
    data: DataConfig
    losses: LossCoeffs
    imp_min: ImpMinConfig
    adversary: AdversaryConfig
    """Recon adversary: persistent source-Adam (PPGD) or fresh per-batch sign-PGD.
    The native yaml key stays `ppgd:` (always the persistent variant there); fresh-PGD
    arrives via the torch-config route only."""
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
        return self.out_dir / (self.run_id if self.run_id is not None else self.run_name)

    @property
    def wandb_id(self) -> str:
        return self.run_id if self.run_id is not None else self.run_name


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    names = (
        {f.name for f in fields(cls)} if hasattr(cls, "__dataclass_fields__") else set(cls._fields)
    )  # type: ignore[attr-defined]
    unknown = set(raw) - names
    assert not unknown, f"{where}: unknown keys {sorted(unknown)} (expected {sorted(names)})"
    missing = names - set(raw)
    assert not missing, f"{where}: missing keys {sorted(missing)}"
    return cls(**raw)


def load_config(path: Path) -> ExperimentConfig:
    assert path.exists(), f"config not found: {path}"
    raw = yaml.safe_load(path.read_text())
    top = {
        "run_name",
        "out_dir",
        "seed",
        "steps",
        "target",
        "data",
        "losses",
        "imp_min",
        "ppgd",
        "recon",
        "vu_optimizer",
        "ci_optimizer",
        "ci_fn",
        "faith_warmup",
        "cadence",
        "eval",
        "wandb",
    }
    unknown = set(raw) - top
    assert not unknown, f"{path}: unknown top-level keys {sorted(unknown)}"
    missing = top - set(raw) - {"eval", "wandb"}
    assert not missing, f"{path}: missing top-level keys {sorted(missing)}"

    target_raw = raw["target"]
    target: AnyTargetConfig
    if "pretrain_run_path" in target_raw:
        assert set(target_raw) == {"pretrain_run_path", "sites"}, (
            f"target: unknown keys {sorted(target_raw)}"
        )
        target = LlamaSimpleMLPTargetConfig(
            pretrain_run_path=target_raw["pretrain_run_path"],
            sites=llama_simple_mlp.canonical_site_cs(
                tuple(SiteC(site["name"], site["C"]) for site in target_raw["sites"])
            ),
        )
    else:
        assert set(target_raw) == {"model_name", "first_layer", "last_layer", "C"}, (
            f"target: unknown keys {sorted(target_raw)}"
        )
        target = TargetConfig(
            model_name=target_raw["model_name"],
            sites=mlp_family_site_cs(
                target_raw["first_layer"], target_raw["last_layer"], target_raw["C"]
            ),
        )

    data_raw = dict(raw["data"], dir=Path(raw["data"]["dir"]))
    cfg = ExperimentConfig(
        run_name=raw["run_name"],
        run_id=None,
        out_dir=Path(raw["out_dir"]),
        seed=raw["seed"],
        steps=raw["steps"],
        target=target,
        data=_build(DataConfig, data_raw, "data"),
        losses=_build(LossCoeffs, raw["losses"], "losses"),
        imp_min=_build(ImpMinConfig, raw["imp_min"], "imp_min"),
        adversary=_build(SourceAdamConfig, raw["ppgd"], "ppgd"),
        recon=_build(ReconConfig, raw["recon"], "recon"),
        vu_optimizer=_build(VUOptimizerConfig, raw["vu_optimizer"], "vu_optimizer"),
        ci_optimizer=_build(CIOptimizerConfig, raw["ci_optimizer"], "ci_optimizer"),
        ci_fn=_build(CIArch, raw["ci_fn"], "ci_fn"),
        faith_warmup=_build(FaithWarmupConfig, raw["faith_warmup"], "faith_warmup"),
        cadence=CadenceConfig(
            **{k: v for k, v in raw["cadence"].items() if k != "dense_log_phase"},
            dense_log_phase=(
                _build(DenseLogPhase, raw["cadence"]["dense_log_phase"], "dense_log_phase")
                if raw["cadence"].get("dense_log_phase")
                else None
            ),
        ),
        eval=(
            _build(
                EvalConfig,
                dict(
                    {k: v for k, v in raw["eval"].items() if k not in ("pgd", "l0_groups")},
                    l0_groups=(
                        {g: tuple(pats) for g, pats in raw["eval"]["l0_groups"].items()}
                        if raw["eval"].get("l0_groups")
                        else None
                    ),
                    pgd=(
                        _build(EvalPGDConfig, raw["eval"]["pgd"], "eval.pgd")
                        if raw["eval"].get("pgd")
                        else None
                    ),
                ),
                "eval",
            )
            if raw.get("eval")
            else None
        ),
        wandb=_build(WandbConfig, raw["wandb"], "wandb") if raw.get("wandb") else None,
    )
    n_sites = len(cfg.target.sites)
    assert n_sites % cfg.recon.sites_per_chunk == 0, (n_sites, cfg.recon.sites_per_chunk)
    assert cfg.cadence.save_every % cfg.cadence.log_every == 0
    return cfg
