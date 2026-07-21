"""Score how cleanly each used subcomponent's inner activations isolate a single operand period.

Offline twin of the `PeriodSeparation` eval metric (`param_decomp_lab/eval_metrics/
period_separation.py`), for any saved checkpoint. For every subcomponent with mean CI >
`--ci-gate` over the full `a+b=` grid at the answer position, the V-column-normalised inner
activation `x·V_c/‖V_c‖` is laid out on the (a, b) grid, 2D-FFT'd, and decomposed into
canonical period classes T ∈ {2, 4, 5, 10, 20, 25, 50, 100} (`period_orbits.
period_class_shares` — the bins a linear read of period-T Fourier features can produce).
A class is *present* when its bin power is `--snr-thr`× the median of the same quantity
over random unit read-directions through the same module input (absolute detection against
the only meaningful null — the input is itself periodic, so a spectral-floor test fires on
everything and a share-relative test misses small genuine periods); `n_periods` = number of
present classes: 0 =
aperiodic (fine), 1 = clean single period (fine — a both-operand blob grid is one period),
≥ 2 = mixing, the failure mode being measured. `mixed_frac` / `excess_periods` aggregate
over periodic components only; the summary reports mixed_frac at λ ∈ {10, 20, 100} plus
the threshold-free `secondary_share` (mean share of the second-strongest class).

Usage:
    python -m param_decomp_lab.scripts.validation.score_period_separation <model_path> \
        [--ci-gate=0.1] [--snr-thr=20] [--module-grep=mlp] [--batch-size=512] \
        [--top-k-plot=20] [--output=PATH] [--output-summary=PATH] [--output-fig=PATH] \
        [--slurm ...]

Output: `analysis/datasets/period_separation.tsv` (one row per scored subcomponent:
per-class SNRs + shares, present periods, n_periods, secondary_share),
`analysis/datasets/period_separation_summary.tsv` (per matrix + pooled: n_active,
periodic_frac, mixed_frac at the three λs, excess_periods, secondary_share, census), and
`analysis/inner_acts_period_panel.png` (AB-heatmap-style panel of the top `--top-k-plot`
inner-activation grids per matrix by mean CI). GPU (one forward pass over the grid) —
use `--slurm` from the login node.
"""

import csv
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")

from param_decomp.log import logger  # noqa: E402
from param_decomp.torch_helpers import bf16_autocast  # noqa: E402
from param_decomp_lab.eval_metrics.period_separation import (  # noqa: E402
    ComponentPeriods,
    PeriodSeparation,
    PeriodSeparationConfig,
)
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME  # noqa: E402
from param_decomp_lab.period_orbits import CANONICAL_PERIODS, count_periods  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    SlurmOptions,
    analysis_datasets_dir,
    analysis_dir,
    load_lm_run,
    op_prompts_file,
    parse_module_name,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.score_period_separation"
_SNR_THRS = (10.0, 20.0, 100.0)


def _summary_row(scored: "list[ComponentPeriods]", snr_thr: float) -> dict[str, object]:
    n_periods = [count_periods(s.snr, snr_thr, CANONICAL_PERIODS) for s in scored]
    periodic = [n for n in n_periods if n >= 1]
    row: dict[str, object] = {
        "n_active": len(scored),
        "periodic_frac": round(len(periodic) / len(scored), 4) if scored else "",
    }
    for side_thr in _SNR_THRS:
        nps = [count_periods(s.snr, side_thr, CANONICAL_PERIODS) for s in scored]
        per = [n for n in nps if n >= 1]
        row[f"mixed_frac_snr{int(side_thr)}"] = (
            round(sum(1 for n in per if n >= 2) / len(per), 4) if per else ""
        )
    row["excess_periods"] = (
        round(sum(n - 1 for n in periodic) / len(periodic), 4) if periodic else ""
    )
    row["secondary_share"] = (
        round(sum(sorted(s.shares.values())[-2] for s in scored) / len(scored), 4) if scored else ""
    )
    census: dict[int, int] = {}
    for s in scored:
        for period, v in s.snr.items():
            if v >= snr_thr:
                census[period] = census.get(period, 0) + 1
    row["census"] = " ".join(f"T{p}={c}" for p, c in sorted(census.items()))
    return row


def score_period_separation(
    model_path: str,
    ci_gate: float = 0.1,
    snr_thr: float = 20.0,
    module_grep: str = "mlp",
    batch_size: int = 512,
    top_k_plot: int = 20,
    output: str | None = None,
    output_summary: str | None = None,
    output_fig: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = None,
) -> tuple[Path, Path, Path] | None:
    """Write per-subcomponent period-class rows, a per-matrix summary, and the
    inner-activation panel. Returns the three paths (None on the `--slurm` submit path)."""
    if slurm:
        argv = [str(Path(model_path).expanduser().resolve())]
        argv += [f"--ci-gate={ci_gate}", f"--snr-thr={snr_thr}", f"--module-grep={module_grep}"]
        argv += [f"--batch-size={batch_size}", f"--top-k-plot={top_k_plot}"]
        for flag, val in (("output", output), ("output-summary", output_summary),
                          ("output-fig", output_fig)):  # fmt: skip
            if val is not None:
                argv.append(f"--{flag}={Path(val).expanduser().resolve()}")
        submit_self_to_slurm(
            _MODULE,
            argv,
            SlurmOptions(
                partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
            ),
            job_name="val-period-separation",
        )
        return None

    run = load_lm_run(model_path)
    metric = PeriodSeparation(
        PeriodSeparationConfig(
            prompts_file=str(op_prompts_file("add")),
            tokenizer_name=run.cfg.data.tokenizer_name,
            ci_gate=ci_gate,
            snr_thr=snr_thr,
            module_grep=module_grep,
            batch_size=batch_size,
            top_k_plot=top_k_plot,
        )
    )
    metric.bind(model=run.model, device=str(run.device))
    metric.reset()
    metric.sampling = run.cfg.pd.sampling
    with bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        scored = metric.scored_components()
    assert scored, f"no subcomponent passed the CI gate ({ci_gate}) — nothing to score"

    datasets = analysis_datasets_dir(run.run_dir)
    datasets.mkdir(parents=True, exist_ok=True)
    out_path = Path(output).expanduser() if output else datasets / "period_separation.tsv"
    out_summary_path = (
        Path(output_summary).expanduser()
        if output_summary
        else datasets / "period_separation_summary.tsv"
    )
    out_fig_path = (
        Path(output_fig).expanduser()
        if output_fig
        else analysis_dir(run.run_dir) / "inner_acts_period_panel.png"
    )

    periods = sorted(scored[0].shares)
    fields = ["layer", "matrix", "component", "mean_ci", "n_periods", "present_periods",
              "extra_periods"]  # fmt: skip
    fields += [f"snr_T{p}" for p in periods] + [f"share_T{p}" for p in periods]
    fields += ["secondary_share"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for s in sorted(scored, key=lambda s: (parse_module_name(s.module), s.component)):
            layer, matrix = parse_module_name(s.module)
            detected = [p for p in periods if s.snr[p] >= snr_thr]
            present = sorted((p for p in detected if p in CANONICAL_PERIODS), reverse=True)
            extra = sorted((p for p in detected if p not in CANONICAL_PERIODS), reverse=True)
            row: dict[str, object] = {
                "layer": layer, "matrix": matrix, "component": s.component,
                "mean_ci": round(s.mean_ci, 4), "n_periods": len(present),
                "present_periods": " ".join(str(p) for p in present),
                "extra_periods": " ".join(str(p) for p in extra),
                "secondary_share": round(sorted(s.shares.values())[-2], 4),
            }  # fmt: skip
            for p in periods:
                row[f"snr_T{p}"] = round(s.snr[p], 1)
                row[f"share_T{p}"] = round(s.shares[p], 4)
            writer.writerow(row)

    matrices = sorted({parse_module_name(s.module) for s in scored})
    summary_fields = ["layer", "matrix", "n_active", "periodic_frac"]
    summary_fields += [f"mixed_frac_snr{int(t)}" for t in _SNR_THRS]
    summary_fields += ["excess_periods", "secondary_share", "census"]
    with open(out_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, delimiter="\t")
        writer.writeheader()
        for layer, matrix in matrices:
            group = [s for s in scored if parse_module_name(s.module) == (layer, matrix)]
            writer.writerow({"layer": layer, "matrix": matrix, **_summary_row(group, snr_thr)})
        writer.writerow({"layer": "", "matrix": "all", **_summary_row(scored, snr_thr)})

    fig = metric.plot_panel(scored)
    fig.savefig(out_fig_path, dpi=250)

    pooled = _summary_row(scored, snr_thr)
    logger.info(
        f"{len(scored)} subcomponents scored (CI > {ci_gate}) — "
        f"mixed_frac@snr{int(snr_thr)}={pooled[f'mixed_frac_snr{int(snr_thr)}']}, "
        f"secondary_share={pooled['secondary_share']} → {out_path}, {out_summary_path}, "
        f"{out_fig_path}"
    )
    return out_path, out_summary_path, out_fig_path


if __name__ == "__main__":
    fire.Fire(score_period_separation)
