#!/usr/bin/env python
"""Output-head metrics of the dual objective (-05) against its outputs-only twin, by step and
by wall time on the SAME hardware. One figure per stream:

    python plot_outputs_only_vs_dual.py     # writes plots/outputs_only_vs_dual/
                                            #   outputs_only_vs_dual_{target,nontarget}_{by_step,by_time}.png

Three output-head metrics per stream: CI-masked KL (`ce_kl/kl_ci_masked`, the arm
CIMaskedReconLoss emits in a single-role run), the fresh-PGD adversarial recon eval
(`loss/PGDReconLoss`) and the alive count (`l0/0.0_total`). The imp-min loss was dropped
(Antoine, 2026-09-23); L0 carries the sparsity story here. The TARGET stream is the addsub prompt pool; the
NON-TARGET stream is fineweb, the same keys under `nontarget_data/`. They get separate
figures because the two streams tell different stories and share no scale.

THE WALL-TIME AXIS IS A100 TIME FOR BOTH RUNS. -05 itself ran on 4x H100 (~2.55 s/step);
outputs-only runs on 4x A100. Plotting each on its own clock would credit -05 with a ~2.3x
hardware speed-up it did not earn from the objective. So -05's metrics are placed on the
MEASURED A100 clock of -07 (p-ba7a0c07), which runs -05's exact compute graph on the same
pod: its changes from -05 are schedule constants and the decay's bundle selection, none of
which change per-step cost. Compile time is excluded from both clocks (a one-off ~2 h,
cached on reruns); slow-eval and checkpoint overhead are included, since `elapsed_s`
deltas carry them.
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
BACKUP = Path.home() / "out" / "pod-backup"

# Reference palette (dataviz skill), validated: CVD dE 24.7, normal-vision dE 33.6.
DUAL, OUT_ONLY = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

PGD = "loss/PGDReconLoss"
# Per stream: (file suffix, headline, key prefix per phase, [(metric, title, log y)]).
# `eval/` metrics take `eval/<prefix>`, train metrics `train/<prefix>`.
STREAMS = [
    (
        "target",
        "target stream (addsub prompts)",
        "",
        [
            ("ce_kl/kl_ci_masked", "CI-masked KL", True),
            (PGD, "Adversarial recon (fresh PGD)", True),
            ("l0/0.0_total", "Alive components (L0, output head)", False),
        ],
    ),
    (
        "nontarget",
        "non-target stream (fineweb)",
        "nontarget_data/",
        [
            ("ce_kl/kl_ci_masked", "CI-masked KL", True),
            (PGD, "Adversarial recon (fresh PGD)", True),
            ("l0/0.0_total", "Alive components (L0, output head)", False),
        ],
    ),
]


def load(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.open():
        d = json.loads(line)
        if "step" in d:
            rows.setdefault(d["step"], {}).update(d)
    return rows


def a100_clock(rows: dict[int, dict[str, Any]]) -> dict[int, float]:
    """Training hours at each logged step, compile excluded, resumes stitched. A segment
    starts wherever `elapsed_s` goes backwards (a resume restarts the counter); its first
    row's `step_time_s` still carries the compile, so that one gap is costed at the
    segment's MEDIAN step time and every other gap is the measured `elapsed_s` delta."""
    pts = sorted(
        (s, r["train/perf/elapsed_s"], r["train/perf/step_time_s"])
        for s, r in rows.items()
        if "train/perf/elapsed_s" in r
    )
    segments, cur = [], [pts[0]]
    for p in pts[1:]:
        if p[1] < cur[-1][1]:
            segments.append(cur)
            cur = [p]
        else:
            cur.append(p)
    segments.append(cur)
    clock, acc, prev_step = {}, 0.0, 0
    for seg in segments:
        median_st = statistics.median(st for _, _, st in seg)
        acc += (seg[0][0] - prev_step) * median_st
        clock[seg[0][0]] = acc
        for (_, e0, _), (s1, e1, _) in zip(seg, seg[1:], strict=False):
            acc += e1 - e0
            clock[s1] = acc
        prev_step = seg[-1][0]
    return {s: t / 3600 for s, t in clock.items()}


def series(rows: dict[int, dict[str, Any]], key: str) -> list[tuple[int, float]]:
    return sorted((s, r[key]) for s, r in rows.items() if key in r)


def at_time(clock: dict[int, float], step: int) -> float | None:
    """Clock value at `step`, linearly interpolated between logged steps (step 0 -> 0)."""
    if step in clock:
        return clock[step]
    known = sorted(clock)
    if step <= 0:
        return 0.0
    if step > known[-1]:
        return None
    lo = max((s for s in known if s < step), default=0)
    hi = min(s for s in known if s > step)
    t_lo = clock.get(lo, 0.0)
    return t_lo + (clock[hi] - t_lo) * (step - lo) / (hi - lo)


def render(
    headline: str,
    prefix: str,
    metrics: list[tuple[str, str, bool]],
    runs: list[tuple[str, dict[int, dict[str, Any]], dict[int, float], str]],
    oo_last: int,
    out: Path,
    by_time: bool,
) -> None:
    """One figure, one x-axis: training step or A100 wall time (step and time live in
    separate files so either can be shown on its own)."""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6.5, 9))
    for row, (metric, title, logy) in enumerate(metrics):
        key = f"eval/{prefix}{metric}"
        sparse = metric == PGD
        ax = axes[row]
        for name, rows, clk, color in runs:
            pts = series(rows, key)
            if sparse and prefix:
                # Step-0 non-target PGD is ~0 by construction (zero-U init: the
                # components carry nothing, so no mask can break the recon). On a log
                # axis it reads as -inf and stretches the row down to 1e-3.
                pts = [(s, v) for s, v in pts if s > 0]
            if by_time:
                pts = [(t, v) for s, v in pts if (t := at_time(clk, s)) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs,
                ys,
                color=color,
                lw=1.5,
                label=name,
                marker="o" if sparse else None,
                ms=5,
                mec=SURFACE,
                mew=1.5,
            )
        if logy:
            ax.set_yscale("log")
            # Labelled ticks at 1-2-5: several rows span under a decade, where the
            # default locator labels a single power of ten.
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold")
        if not by_time:
            for step, what in ((10_000, "imp-min ramp ends"), (30_000, "γ anneal starts")):
                ax.axvline(step, color=INK_2, lw=0.6, ls=":", zorder=0)
                if row == 0:
                    ax.annotate(
                        what,
                        (step, 1),
                        xycoords=("data", "axes fraction"),
                        xytext=(3, -2),
                        textcoords="offset points",
                        va="top",
                        fontsize=7.5,
                        color=INK_2,
                    )
        if not by_time:
            axes[row].set_xlim(0, 40_000)
    axes[-1].set_xlabel(
        "Training wall time, hours on 4x A100 (compile excluded)" if by_time else "Training step"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.945),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        f"Output-head metrics, {headline}\nwith vs without the hidden-activation recon, "
        f"by {'wall time' if by_time else 'step'}",
        x=0.1,
        y=0.998,
        ha="left",
        color=INK,
        fontsize=11,
        fontweight="semibold",
    )
    note = f"Outputs-only (p-ba5a0c0f) to step {oo_last:,} of 40,000; dual = -05 (p-ba5a0c05).\n"
    note += (
        "The axis is -07's measured A100 clock for the dual run (same compute graph, same pod):\n"
        "-05 itself ran on H100s, whose clock would credit it with a hardware speed-up.\n"
        if by_time
        else ""
    )
    note += "Log scale on the top two rows."
    if prefix:
        note += "\nPGD omits step 0, where it is ~0 by construction (zero-U init: components carry nothing yet)."
    fig.text(0.1, 0.005, note, fontsize=7.5, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.925))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dual", type=Path, default=BACKUP / "p-ba5a0c05" / "metrics.jsonl")
    ap.add_argument("--dual-a100-clock", type=Path, default=BACKUP / "p-ba7a0c07" / "metrics.jsonl")
    ap.add_argument("--outputs-only", type=Path, default=BACKUP / "p-ba5a0c0f" / "metrics.jsonl")
    ap.add_argument("-o", "--out-dir", type=Path, default=HERE / "plots" / "outputs_only_vs_dual")
    a = ap.parse_args()

    dual, oo = load(a.dual), load(a.outputs_only)
    clk_dual, clk_oo = a100_clock(load(a.dual_a100_clock)), a100_clock(oo)
    oo_last = max(s for s, r in oo.items() if "train/perf/elapsed_s" in r)
    runs = [
        ("Dual objective (-05)", dual, clk_dual, DUAL),
        ("Outputs-only", oo, clk_oo, OUT_ONLY),
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for suffix, headline, prefix, metrics in STREAMS:
        for by_time, axis in ((False, "by_step"), (True, "by_time")):
            out = a.out_dir / f"outputs_only_vs_dual_{suffix}_{axis}.png"
            render(headline, prefix, metrics, runs, oo_last, out, by_time)
    print(
        f"outputs-only through step {oo_last}; A100 h at 40k: dual {clk_dual[40000]:.1f}, "
        f"outputs-only {clk_oo[oo_last] * 40000 / oo_last:.1f} projected"
    )


if __name__ == "__main__":
    main()
