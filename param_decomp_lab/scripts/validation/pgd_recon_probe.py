"""How much reconstruction error does a trained decomposition *permit* under attack?

Measures, for a saved checkpoint, the recon error under the CI mask against the error a
fresh `n_steps`-step sign-PGD adversary can reach from it — on both distributions of a
targeted run (target and nontarget) and in both modalities (final output and per-site
hidden activations). Four cells, so the gap between "these CI values happen to work" and
"these CI values really are safe to mask" is readable per cell.

The two distributions differ in how the weight delta is handled, exactly as in training:
on target data the CI-masked probe pins it off (the components must do the work) while the
adversary owns its mask; on nontarget data it is pinned fully on throughout, so the
adversary can only ablate components downward from CI — the CI-masked number is its floor
there, and the interesting quantity is how far above it PGD gets.

An 8B forward pass needs a GPU; pass `--slurm` to submit as a single-GPU job. PGD retains
the graph back to its adversarial sources, so a single GPU may need `--batch-size` /
`--nontarget-batch-size` below the run's own eval batch sizes (raise `--n-batches` to keep
the sample count).

Usage:
    python -m param_decomp_lab.scripts.validation.pgd_recon_probe <model_path> \
        [--n-steps=20] [--n-batches=10] [--step-size=0.1] [--site-patterns=...] \
        [--batch-size=N] [--nontarget-batch-size=N] \
        [--output-tsv=PATH] [--output-fig=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=1:00:00 --slurm-mem=...]]

Outputs (default in the run's `analysis/pgd_recon_probe/`):
- `pgd_recon_probe.tsv` — one row per (distribution, modality, mask) with its value.
- `pgd_recon_probe.png` — one panel per modality, CI-masked vs stochastic vs PGD.
"""

import csv
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypedDict

import fire
import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from param_decomp.ci_fns import CIRole  # noqa: E402
from param_decomp.log import logger  # noqa: E402
from param_decomp.metrics.base import Metric  # noqa: E402
from param_decomp.metrics.pgd_hidden_acts_recon import (  # noqa: E402
    NontargetPGDHiddenActsReconLoss,
    NontargetPGDHiddenActsReconLossConfig,
    PGDHiddenActsReconLoss,
    PGDHiddenActsReconLossConfig,
)
from param_decomp.metrics.pgd_masked_recon import (  # noqa: E402
    NontargetPGDReconLoss,
    NontargetPGDReconLossConfig,
    PGDReconLoss,
    PGDReconLossConfig,
)
from param_decomp.metrics.pgd_utils import MaskScope, PGDInitStrategy  # noqa: E402
from param_decomp.optimize import build_metric_context  # noqa: E402
from param_decomp.targeted import delta_override  # noqa: E402
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader  # noqa: E402
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl  # noqa: E402
from param_decomp_lab.eval_metrics.ci_hidden_acts_recon_loss import (  # noqa: E402
    CIHiddenActsReconLoss,
    CIHiddenActsReconLossConfig,
    NontargetCIHiddenActsReconLoss,
    NontargetCIHiddenActsReconLossConfig,
)
from param_decomp_lab.eval_metrics.targeted_recon_loss import (  # noqa: E402
    NontargetReconLoss,
    NontargetReconLossConfig,
    TargetReconLoss,
    TargetReconLossConfig,
)
from param_decomp_lab.experiments.lm.run import (  # noqa: E402
    LMExperimentConfig,
    build_lm_loader,
    make_reconstruction_loss,
)
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    LoadedRun,
    SlurmOptions,
    analysis_dir,
    load_lm_run,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.pgd_recon_probe"
# The two masking strategies of the multi-strategy recon metric this probe reports; it also
# computes `rounded` and `delta_only`, which belong to other questions.
_REPORTED_RECON_STRATEGIES = ("ci_masked", "stochastic")


@dataclass(frozen=True)
class ProbeRow:
    distribution: str
    modality: str
    mask: str
    value: float


@dataclass(frozen=True)
class ProbeSettings:
    n_steps: int
    step_size: float
    ci_role: CIRole
    site_patterns: list[str] | None

    @property
    def pgd_mask_label(self) -> str:
        return f"pgd_{self.n_steps}step"


@dataclass(frozen=True)
class ProbeCells:
    """The four metrics measured on one distribution, named rather than positional."""

    output_recon: Metric[Any]
    output_pgd: Metric[Any]
    hidden_ci: Metric[Any]
    hidden_pgd: Metric[Any]

    def all(self) -> list[Metric[Any]]:
        return [self.output_recon, self.output_pgd, self.hidden_ci, self.hidden_pgd]


class _PGDKwargs(TypedDict):
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


class _HiddenKwargs(TypedDict):
    ci_role: CIRole
    site_patterns: list[str] | None


def _build_cells(s: ProbeSettings, *, nontarget: bool) -> ProbeCells:
    pgd = _PGDKwargs(
        init="random",
        step_size=s.step_size,
        n_steps=s.n_steps,
        mask_scope="shared_across_batch",
    )
    hidden = _HiddenKwargs(ci_role=s.ci_role, site_patterns=s.site_patterns)
    if nontarget:
        return ProbeCells(
            output_recon=NontargetReconLoss(NontargetReconLossConfig(rounding_threshold=0.01)),
            output_pgd=NontargetPGDReconLoss(NontargetPGDReconLossConfig(**pgd)),
            hidden_ci=NontargetCIHiddenActsReconLoss(
                NontargetCIHiddenActsReconLossConfig(**hidden)
            ),
            hidden_pgd=NontargetPGDHiddenActsReconLoss(
                NontargetPGDHiddenActsReconLossConfig(**hidden, **pgd)
            ),
        )
    return ProbeCells(
        output_recon=TargetReconLoss(TargetReconLossConfig(rounding_threshold=0.01)),
        output_pgd=PGDReconLoss(PGDReconLossConfig(**pgd)),
        hidden_ci=CIHiddenActsReconLoss(CIHiddenActsReconLossConfig(**hidden)),
        hidden_pgd=PGDHiddenActsReconLoss(PGDHiddenActsReconLossConfig(**hidden, **pgd)),
    )


def _run_cells_over_batches(
    run: LoadedRun,
    cells: ProbeCells,
    loader: DataLoader[Any],
    n_batches: int,
    weight_deltas: dict[str, torch.Tensor],
    *,
    nontarget: bool,
) -> None:
    """Drive `cells` over `n_batches` of `loader`, mirroring the trainer's eval pass."""
    metrics = cells.all()
    batches = loop_dataloader(loader)
    for m in metrics:
        m.bind(model=run.model, device=str(run.device))
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        for _ in range(n_batches):
            ctx = build_metric_context(
                next(batches),
                step=run.cfg.pd.steps,
                is_eval=True,
                device=str(run.device),
                wrapped_model=run.model,
                component_model=run.model,
                config=run.cfg.pd,
                reconstruction_loss=(
                    recon_loss_kl if nontarget else make_reconstruction_loss(run.cfg.target)
                ),
                weight_deltas=weight_deltas,
            )
            with delta_override(1.0) if nontarget else nullcontext():
                for m in metrics:
                    m.update(ctx)


def _rows_from_cells(cells: ProbeCells, distribution: str, s: ProbeSettings) -> Iterator[ProbeRow]:
    recon_result = cells.output_recon.compute()
    assert isinstance(recon_result, dict)
    for strategy in _REPORTED_RECON_STRATEGIES:
        yield ProbeRow(distribution, "output", strategy, recon_result[strategy].item())
    output_pgd = cells.output_pgd.compute()
    assert isinstance(output_pgd, torch.Tensor)
    yield ProbeRow(distribution, "output", s.pgd_mask_label, output_pgd.item())
    yield ProbeRow(distribution, "hidden_acts", "ci_masked", _mean_site_error(cells.hidden_ci))
    yield ProbeRow(
        distribution, "hidden_acts", s.pgd_mask_label, _mean_site_error(cells.hidden_pgd)
    )


def _mean_site_error(metric: Metric[Any]) -> float:
    """The site-mean relative error a hidden-acts metric keys under its own instance key."""
    result = metric.compute()
    assert isinstance(result, dict)
    return result[metric.instance_key].item()


def _plot(rows: Sequence[ProbeRow], run_label: str, s: ProbeSettings, output_fig: Path) -> None:
    # Per modality, only the masks that modality actually reports: the hidden-acts probes
    # have no stochastic member.
    panels = [
        ("output", "Output recon (KL)", ["ci_masked", "stochastic", s.pgd_mask_label]),
        ("hidden_acts", "Hidden-acts relative error", ["ci_masked", s.pgd_mask_label]),
    ]
    distributions = ["target", "nontarget"]
    by_key = {(r.distribution, r.modality, r.mask): r.value for r in rows}

    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.5))
    for ax, (modality, title, masks) in zip(axes, panels, strict=True):
        width = 0.8 / len(masks)
        for i, mask in enumerate(masks):
            positions = [x + (i - (len(masks) - 1) / 2) * width for x in range(len(distributions))]
            values = [by_key[(d, modality, mask)] for d in distributions]
            bars = ax.bar(positions, values, width=width, label=mask, zorder=3)
            ax.bar_label(bars, fmt="%.4g", fontsize=7, padding=2)
        ax.set_yscale("log")
        ax.set_xticks(range(len(distributions)), distributions)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=8)
    fig.suptitle(f"{run_label}: CI-masked vs {s.n_steps}-step PGD reconstruction")
    fig.tight_layout()
    fig.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_tsv(rows: Sequence[ProbeRow], output_tsv: Path) -> None:
    with output_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(ProbeRow)], delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) | {"value": f"{row.value:.8g}"})


def _eval_loader(
    cfg: LMExperimentConfig, device: str, batch_size: int | None, *, nontarget: bool
) -> DataLoader[Any]:
    assert cfg.eval is not None and cfg.nontarget is not None
    data_cfg = cfg.nontarget.data if nontarget else cfg.data
    default_batch_size = cfg.nontarget.eval_batch_size if nontarget else cfg.eval.batch_size
    return build_lm_loader(
        cfg.target,
        data_cfg,
        split="eval",
        device=device,
        batch_size=batch_size if batch_size is not None else default_batch_size,
        seed=cfg.pd.seed,
    )


def pgd_recon_probe(
    model_path: ModelPath,
    n_steps: int = 20,
    n_batches: int = 10,
    step_size: float = 0.1,
    site_patterns: list[str] | None = None,
    batch_size: int | None = None,
    nontarget_batch_size: int | None = None,
    output_tsv: str | Path | None = None,
    output_fig: str | Path | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> tuple[Path, Path] | None:
    if slurm:
        argv = [str(Path(model_path).expanduser().resolve())]
        argv += [f"--n-steps={n_steps}", f"--n-batches={n_batches}", f"--step-size={step_size}"]
        if site_patterns is not None:
            argv.append(f"--site-patterns={','.join(site_patterns)}")
        if batch_size is not None:
            argv.append(f"--batch-size={batch_size}")
        if nontarget_batch_size is not None:
            argv.append(f"--nontarget-batch-size={nontarget_batch_size}")
        if output_tsv is not None:
            argv.append(f"--output-tsv={Path(output_tsv).resolve()}")
        if output_fig is not None:
            argv.append(f"--output-fig={Path(output_fig).resolve()}")
        submit_self_to_slurm(
            _MODULE,
            argv,
            SlurmOptions(
                partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
            ),
            job_name="val-pgd-recon-probe",
        )
        return None

    run = load_lm_run(model_path)
    assert run.cfg.nontarget is not None, (
        "pgd_recon_probe compares the target and nontarget distributions; this run has no "
        "`nontarget` config"
    )
    settings = ProbeSettings(
        n_steps=n_steps,
        step_size=step_size,
        ci_role="hidden" if run.cfg.pd.dual_hidden_ci else "output",
        site_patterns=site_patterns,
    )
    weight_deltas = run.model.calc_weight_deltas()

    rows: list[ProbeRow] = []
    for nontarget in (False, True):
        cells = _build_cells(settings, nontarget=nontarget)
        _run_cells_over_batches(
            run,
            cells,
            _eval_loader(
                run.cfg,
                str(run.device),
                nontarget_batch_size if nontarget else batch_size,
                nontarget=nontarget,
            ),
            n_batches,
            weight_deltas,
            nontarget=nontarget,
        )
        rows.extend(_rows_from_cells(cells, "nontarget" if nontarget else "target", settings))

    out_dir = analysis_dir(run.run_dir) / "pgd_recon_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = Path(output_tsv) if output_tsv is not None else out_dir / "pgd_recon_probe.tsv"
    fig_path = Path(output_fig) if output_fig is not None else out_dir / "pgd_recon_probe.png"
    _write_tsv(rows, tsv_path)
    _plot(rows, run.run_dir.name, settings, fig_path)

    logger.section(f"{n_steps}-step PGD recon probe — {run.run_dir.name}")
    logger.values({f"{r.distribution}/{r.modality}/{r.mask}": f"{r.value:.6g}" for r in rows})
    logger.info(f"Wrote {tsv_path} and {fig_path}")
    return tsv_path, fig_path


if __name__ == "__main__":
    fire.Fire(pgd_recon_probe)
