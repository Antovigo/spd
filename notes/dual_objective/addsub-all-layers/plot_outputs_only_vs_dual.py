#!/usr/bin/env python
"""Output-head metrics of the dual objective (-05) against its outputs-only twin, by step and
by wall time on the SAME hardware.

    python plot_outputs_only_vs_dual.py            # reads ~/out/pod-backup, writes plots/

Four output-head metrics, target stream: CI-masked KL (`eval/ce_kl/kl_ci_masked`, the arm
CIMaskedReconLoss emits in a single-role run), the fresh-PGD adversarial recon eval
(`eval/loss/PGDReconLoss`), the output head's imp-min loss (`train/loss/
ImportanceMinimalityLoss`) and the alive count (`eval/l0/0.0_total`).

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

HERE = Path(__file__).resolve().parent
BACKUP = Path.home() / "out" / "pod-backup"

# Reference palette (dataviz skill), validated: CVD dE 24.7, normal-vision dE 33.6.
DUAL, OUT_ONLY = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

METRICS = [
    ("eval/ce_kl/kl_ci_masked", "CI-masked KL", True),
    ("eval/loss/PGDReconLoss", "Adversarial recon (fresh PGD)", True),
    ("train/loss/ImportanceMinimalityLoss", "Imp-min loss (output head)", False),
    ("eval/l0/0.0_total", "Alive components (L0, output head)", False),
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dual", type=Path, default=BACKUP / "p-ba5a0c05" / "metrics.jsonl")
    ap.add_argument("--dual-a100-clock", type=Path, default=BACKUP / "p-ba7a0c07" / "metrics.jsonl")
    ap.add_argument("--outputs-only", type=Path, default=BACKUP / "p-ba5a0c0f" / "metrics.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "plots" / "outputs_only_vs_dual.png")
    a = ap.parse_args()

    dual, oo = load(a.dual), load(a.outputs_only)
    clk_dual, clk_oo = a100_clock(load(a.dual_a100_clock)), a100_clock(oo)
    oo_last = max(s for s, r in oo.items() if "train/perf/elapsed_s" in r)

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
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(11, 12), sharey="row")
    runs = [("Dual objective (-05)", dual, clk_dual, DUAL), ("Outputs-only", oo, clk_oo, OUT_ONLY)]

    for row, (key, title, logy) in enumerate(METRICS):
        sparse = key == "eval/loss/PGDReconLoss"
        for col in range(2):
            ax = axes[row, col]
            for name, rows, clk, color in runs:
                pts = series(rows, key)
                if col == 1:
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
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold")
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
        axes[row, 0].set_xlim(0, 40_000)
    for ax in axes[-1]:
        ax.set_xlabel("")
    axes[-1, 0].set_xlabel("Training step")
    axes[-1, 1].set_xlabel("Training wall time, hours on 4x A100 (compile excluded)")
    axes[0, 0].set_title("vs step", loc="right", color=INK_2, fontsize=9)
    axes[0, 1].set_title("vs wall time", loc="right", color=INK_2, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.975),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.suptitle(
        "Output-head metrics, target stream: with vs without the hidden-activation recon",
        x=0.06,
        y=0.995,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.005,
        f"Outputs-only (p-ba5a0c0f) shown through step {oo_last:,} of 40,000. Dual = -05 (p-ba5a0c05); its "
        "wall-time axis uses -07's measured A100 clock (same compute graph, same pod), because -05 itself\n"
        "ran on H100s and its own clock would credit it with a hardware speed-up. Log scale on the two KL/recon rows.",
        fontsize=7.5,
        color=INK_2,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(
        f"wrote {a.out}  (outputs-only through step {oo_last}; "
        f"A100 h at 40k: dual {clk_dual[40000]:.1f}, outputs-only "
        f"{clk_oo[oo_last] * 40000 / oo_last:.1f} projected)"
    )


if __name__ == "__main__":
    main()
