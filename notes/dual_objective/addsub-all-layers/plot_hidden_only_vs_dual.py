#!/usr/bin/env python
"""The hidden-only arm against the dual objective (-05) and against the hybrid that switches
to the dual objective at step 28000, by step and by wall time on the SAME hardware.

    python plot_hidden_only_vs_dual.py   # writes plots/hidden_only_vs_dual/
                                         #   hidden_only_vs_dual_{output,hidden}_head_
                                         #     {target,nontarget}_{by_step,by_time}.png

THE THREE RUNS. `-05` (p-ba5a0c05) is the dual objective, 40000 steps. `hidden-only`
(p-ba5a0c0d) deletes both output-recon passes and was stopped at 36900, so its curves simply
end there. `hidden-then-dual` (p-ba5a0c0e) resumes -05's full dual objective from
hidden-only's step-28000 checkpoint, with the output head grafted as a copy of the hidden head,
and runs to 40000; its curves therefore START at 28000.

ONE FIGURE PER (HEAD, STREAM, AXIS), because the two CI heads answer different questions and the
two streams share no scale. The HIDDEN-head figures carry all three runs: every run trains that
head, so the series are directly comparable. The OUTPUT-head figures omit hidden-only, whose
output head is FROZEN at its `zero_init_readout` value (CI 0.5 for every subcomponent, hence
L0 = 124,928 and imp-min exactly 0 -- it is not a measurement of anything, and it would flatten
every other series on the axis). hidden-then-dual's output head is live only from 28000 on.

THE WALL-TIME AXIS IS 4x H100 TIME FOR ALL THREE, each on its own measured clock -- unlike the
outputs-only comparison, no cross-hardware stitching is needed, since all three ran on the same
pod. hidden-then-dual's clock is OFFSET by hidden-only's time to reach step 28000: those first
28000 steps were bought by the hidden-only run, and the point of the hybrid is what the whole
trajectory costs. Compile time is excluded (a one-off ~1-2 h, cached on reruns); slow-eval and
checkpoint overhead are included, since `elapsed_s` deltas carry them.

FOUR ROWS. CI-masked KL (`ce_kl/kl_ci_masked`, deterministic faithfulness), the fresh-PGD
adversarial recon eval (`loss/PGDReconLoss`, worst-case faithfulness), the alive count
(`l0/0.0_total`) and the importance-minimality loss -- the last two together are the sparsity
story, since imp-min is the pressure and L0 is what it buys.
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

# Reference palette (dataviz skill): blue / orange / green, distinguishable under CVD.
DUAL, HID_ONLY, HYBRID = "#2a78d6", "#eb6834", "#0f9d58"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

SWITCH_STEP = 28_000

# (metric suffix, title, log y). The imp-min key is not an `eval/` metric, so it is resolved
# separately in `metric_key`.
METRICS = [
    ("ce_kl/kl_ci_masked", "CI-masked KL (deterministic)", True),
    ("loss/PGDReconLoss", "Adversarial recon (fresh PGD)", True),
    ("l0/0.0_total", "Alive components (L0)", False),
    ("IMPMIN", "Importance-minimality loss", False),
]
HEADS = [
    ("output", "output head", ""),
    ("hidden", "hidden head", "hidden_ci/"),
]
STREAMS = [
    ("target", "target stream (addsub prompts)", ""),
    ("nontarget", "non-target stream (fineweb)", "nontarget_data/"),
]


def metric_key(metric: str, stream_prefix: str, head_prefix: str) -> str:
    """The logged key for one (metric, stream, head). Imp-min is a TRAIN metric and its key is
    irregular: the target-hidden pass logs it bare (`hidden_ci/loss/...`) while every other
    pass sits under `train/`."""
    if metric != "IMPMIN":
        return f"eval/{stream_prefix}{head_prefix}{metric}"
    tail = f"{stream_prefix}{head_prefix}loss/ImportanceMinimalityLoss"
    return tail if tail.startswith("hidden_ci/") else f"train/{tail}"


def load(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.open():
        d = json.loads(line)
        if "step" in d:
            rows.setdefault(d["step"], {}).update(d)
    return rows


def h100_clock(rows: dict[int, dict[str, Any]], step0: int = 0) -> dict[int, float]:
    """Training hours at each logged step, compile excluded, resumes stitched. A segment starts
    wherever `elapsed_s` goes backwards (a resume restarts the counter); its first row's
    `step_time_s` still carries the compile, so that one gap is costed at the segment's MEDIAN
    step time and every other gap is the measured `elapsed_s` delta. `step0` is the step this
    run began at, so a resumed run is not charged for steps it never ran."""
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
    clock, acc, prev_step = {}, 0.0, step0
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


def at_time(clock: dict[int, float], step: int, offset: float) -> float | None:
    """Clock value at `step` plus `offset`, linearly interpolated between logged steps."""
    known = sorted(clock)
    if not known:
        return None
    if step in clock:
        return clock[step] + offset
    if step <= known[0]:
        return offset
    if step > known[-1]:
        return None
    lo = max(s for s in known if s < step)
    hi = min(s for s in known if s > step)
    return clock[lo] + (clock[hi] - clock[lo]) * (step - lo) / (hi - lo) + offset


def render(
    head: tuple[str, str, str],
    stream: tuple[str, str, str],
    runs: list[tuple[str, dict[int, dict[str, Any]], dict[int, float], float, str, Any]],
    notes: list[str],
    out: Path,
    by_time: bool,
) -> None:
    _, head_name, head_prefix = head
    _, stream_name, stream_prefix = stream
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(6.5, 11))
    for row, (metric, title, logy) in enumerate(METRICS):
        key = metric_key(metric, stream_prefix, head_prefix)
        sparse = metric == "loss/PGDReconLoss"
        ax = axes[row]
        for name, rows, clk, offset, color, ancestor in runs:
            pts = series(rows, key)
            if sparse and stream_prefix:
                # Step-0 non-target PGD is ~0 by construction (zero-U init: the components
                # carry nothing, so no mask can break the recon); on a log axis it reads -inf.
                pts = [(s, v) for s, v in pts if s > 0]
            # A resumed run inherits its ancestor's trajectory up to the branch: those steps ARE
            # its own history, the same weights on the same head, so the prefix is drawn DASHED
            # in the run's colour rather than left as a gap. The dash also spans the branch-to-
            # first-eval interval, which nothing measured. Only where the ancestor's series is
            # the same quantity -- see `ancestor` at the call site.
            prefix: list[tuple[float, float]] = []
            if ancestor is not None:
                a_rows, a_clk, a_cut = ancestor
                prefix = [(s, v) for s, v in series(a_rows, key) if s <= a_cut]
                if sparse and stream_prefix:
                    prefix = [(s, v) for s, v in prefix if s > 0]
                if by_time:
                    prefix = [
                        (t, v) for s, v in prefix if (t := at_time(a_clk, s, 0.0)) is not None
                    ]
            if by_time:
                pts = [(t, v) for s, v in pts if (t := at_time(clk, s, offset)) is not None]
            if prefix and pts:
                bx, by = zip(*(prefix + pts[:1]), strict=True)
                ax.plot(bx, by, color=color, lw=1.1, ls=(0, (4, 2)), zorder=3)
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            ax.plot(
                xs, ys, color=color, lw=1.5, label=name,
                marker="o" if sparse else None, ms=5, mec=SURFACE, mew=1.5,
            )  # fmt: skip
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold")
        if not by_time:
            marks = (
                (10_000, "imp-min ramp ends"),
                (SWITCH_STEP, "dual objective resumes"),
                (30_000, "γ anneal starts"),
            )
            for step, what in marks:
                ax.axvline(step, color=INK_2, lw=0.6, ls=":", zorder=0)
                if row == 0:
                    ax.annotate(
                        what, (step, 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -2), textcoords="offset points",
                        va="top", fontsize=7, color=INK_2,
                    )  # fmt: skip
            ax.set_xlim(0, 40_000)
    axes[-1].set_xlabel(
        "Training wall time, hours on 4x H100 (compile excluded)" if by_time else "Training step"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.955),
        ncol=1, frameon=False, fontsize=9, labelcolor=INK,
    )  # fmt: skip
    fig.suptitle(
        f"{head_name.capitalize()} metrics, {stream_name}\n"
        f"hidden-only vs dual vs hidden-then-dual, by {'wall time' if by_time else 'step'}",
        x=0.1, y=0.998, ha="left", color=INK, fontsize=11, fontweight="semibold",
    )  # fmt: skip
    body = list(notes)
    if by_time:
        body.append(
            "hidden-then-dual's clock includes hidden-only's time to step 28,000: that prefix is\n"
            "what the hybrid actually costs. All three ran on the same 4x H100 pod."
        )
    body.append("Log scale on the top two rows.")
    fig.text(0.1, 0.005, "\n".join(body), fontsize=7.5, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dual", type=Path, default=BACKUP / "p-ba5a0c05" / "metrics.jsonl")
    ap.add_argument("--hidden-only", type=Path, default=BACKUP / "p-ba5a0c0d" / "metrics.jsonl")
    ap.add_argument("--hybrid", type=Path, default=BACKUP / "p-ba5a0c0e" / "metrics.jsonl")
    ap.add_argument("-o", "--out-dir", type=Path, default=HERE / "plots" / "hidden_only_vs_dual")
    a = ap.parse_args()

    dual, hid, hyb = load(a.dual), load(a.hidden_only), load(a.hybrid)
    clk_dual, clk_hid = h100_clock(dual), h100_clock(hid)
    clk_hyb = h100_clock(hyb, step0=SWITCH_STEP)
    hid_last = max(s for s, r in hid.items() if "train/perf/elapsed_s" in r)
    # The hybrid inherits hidden-only's cost up to the switch.
    offset = at_time(clk_hid, SWITCH_STEP, 0.0) or 0.0

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
    base = (
        f"Dual = -05 (p-ba5a0c05, 40,000 steps). hidden-only (p-ba5a0c0d) stopped at "
        f"{hid_last:,}.\nhidden-then-dual (p-ba5a0c0e) resumes the dual objective at "
        f"{SWITCH_STEP:,} from hidden-only's checkpoint,\nwith the output head grafted as a copy "
        f"of the hidden head, so its curves start there."
    )
    for head in HEADS:
        for stream in STREAMS:
            # The hybrid IS the hidden-only run up to step 28000, so on hidden-head figures its
            # curve is continued back through that shared history. NOT on output-head figures:
            # there the ancestor's series is its FROZEN head (L0 124,928, imp-min 0, PGD ~46),
            # a different quantity, and the hybrid's output head only exists from the graft on.
            ancestor = (hid, clk_hid, SWITCH_STEP) if head[0] == "hidden" else None
            runs = [
                ("Dual objective (-05)", dual, clk_dual, 0.0, DUAL, None),
                ("Hidden-only", hid, clk_hid, 0.0, HID_ONLY, None),
                ("Hidden-only → dual at 28k", hyb, clk_hyb, offset, HYBRID, ancestor),
            ]
            notes = [base]
            if head[0] == "hidden":
                notes.append(
                    "The hybrid's DASHED prefix is its inherited history: identical weights to\n"
                    "hidden-only before the switch, so it traces that curve, and the dash also\n"
                    "spans 28,000 → 32,000, where the resumed run logged no slow eval."
                )
            if head[0] == "output":
                runs = [r for r in runs if r[0] != "Hidden-only"]
                notes.append(
                    "hidden-only is OMITTED here: nothing trains its output head, which sits at\n"
                    "zero_init_readout (CI 0.5 everywhere, so L0 = 124,928 and imp-min = 0 exactly).\n"
                    "The hybrid's output head exists only from the 28,000 graft, so it is not\n"
                    "continued backwards."
                )
            if stream[0] == "nontarget" and head[0] == "output":
                notes.append(
                    "PGD omits step 0, where it is ~0 by construction (zero-U init: components\n"
                    "carry nothing yet)."
                )
            for by_time, axis in ((False, "by_step"), (True, "by_time")):
                out = a.out_dir / f"hidden_only_vs_dual_{head[0]}_head_{stream[0]}_{axis}.png"
                render(head, stream, runs, notes, out, by_time)
    print(
        f"hidden-only through {hid_last}; H100 h: dual to 40k {clk_dual[40000]:.1f}, "
        f"hidden-only to {hid_last} {clk_hid[hid_last]:.1f}, "
        f"switch at 28k {offset:.1f}, hybrid total {clk_hyb[40000] + offset:.1f}"
    )


if __name__ == "__main__":
    main()
