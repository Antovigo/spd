"""Plot coupled-vs-kaiming init results from the local metrics.jsonl files.

Run on the machine that has the retrieved run dirs (e.g. your laptop), NOT via wandb.
Expects a directory of run folders `cinit-<scheme>-s<seed>-<raw|train|nofaith>/`, each with
a `metrics.jsonl`. Reconstructs the measurement points per condition:

  init            raw run,     step 0
  post-warmup     train run,   step 0
  trained         train run,   step 200
  trained-nofaith nofaith run, step 200   (no warmup, no faithfulness loss)

Plotting conventions (matching the scratch init-study scripts):
  * All recon-loss panels share ONE log y-axis; all L0 panels share ANOTHER — each
    group's limits are pow10 bounds (10**floor(log10(min>0)) .. 10**ceil(log10(max)))
    pooled over the group, so every panel in a group is directly comparable.
  * Weight faithfulness (train/loss/FaithfulnessLoss) gets its own log panel.
  * kaiming = control (grey #555555); coupled = idea-under-test (red #d62728).
  * mean line across seeds + min-max ribbon.

Two comparison SERIES, each its own figure set (`<series>_<group>.png`):
  faith:    init -> post-warmup -> trained          (faithfulness warmup + loss)
  nofaith:  init -> trained-nofaith                 (no warmup, no faithfulness loss)

Grouping is EXPLICIT, keyed off the verified metric-key formats this config emits:
  recon:                      eval/loss/PGDReconLoss (aggregate recon scalar; the
                              hidden-acts recon metrics and per-module keys are excluded)
  l0:                         eval/l0/<threshold>_<group> for the config's aggregate
                              groups only — per-block (layer_N) + total; the per-matrix
                              layer keys are excluded
  faithfulness:               coalesced 'faithfulness' key = train/loss/FaithfulnessLoss
                              (faith phases) or eval/loss/FaithfulnessLoss (nofaith probe)
  impmin:                     train/loss/ImportanceMinimalityLoss (a training loss in
                              every phase, logged at step 0 and the final step)
  ce/kl headline:             eval/ce_kl/{kl,ce_*}_ci_masked

Writes one figure per (series, group) + a tidy summary.csv (all scalar keys, all phases).

    python full_data_init_test_plots.py <runs_dir> [--out <figures_dir>] [--train-steps 200]
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMES = ["kaiming", "coupled"]
SCHEME_STYLE = {"kaiming": ("kaiming", "#555555"), "coupled": ("coupled", "#d62728")}
ALL_PHASES = ["init", "post-warmup", "trained", "trained-nofaith"]
# Two comparison series -> separate figure sets (`<series>_<group>.png`), sharing the
# 'init' start. Each series is self-scaled: shared log limits are pooled over its own phases.
SERIES = {
    "faith": ["init", "post-warmup", "trained"],
    "nofaith": ["init", "trained-nofaith"],
}

FAITHFULNESS_KEY = "faithfulness"  # synthetic, injected by _inject_faithfulness
IMPMIN_KEY = "train/loss/ImportanceMinimalityLoss"
CE_KL_HEADLINE = [
    "eval/ce_kl/kl_ci_masked",
    "eval/ce_kl/ce_unrecovered_ci_masked",
    "eval/ce_kl/ce_difference_ci_masked",
]

# Aggregate L0 keys only: eval/l0/<threshold>_(layer_<n>|total), the CI_L0 `groups` from
# the config. Per-matrix keys (eval/l0/<threshold>_h.<n>.<module>...) are excluded.
L0_AGGREGATE_RE = re.compile(r"eval/l0/[\d.]+_(layer_\d+|total)")

LOG_GROUPS = {"recon", "l0", "faithfulness", "impmin"}
SHARED_GROUPS = {"recon", "l0"}  # every panel in these groups uses identical y-limits
GROUP_ORDER = ["faithfulness", "impmin", "recon", "l0", "ce_kl"]


def is_recon_aggregate(key: str) -> bool:
    """`eval/loss/<Class>` recon scalar — excludes per-module `.../<module>` (3+ slashes)
    and the hidden-acts recon metrics (so only PGDReconLoss remains)."""
    return (
        key.startswith("eval/loss/")
        and "recon" in key.lower()
        and key.count("/") == 2
        and "hiddenacts" not in key.lower()
    )


def group_keys(keys: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "faithfulness": [k for k in keys if k == FAITHFULNESS_KEY],
        "impmin": [k for k in keys if k == IMPMIN_KEY],
        "recon": sorted(k for k in keys if is_recon_aggregate(k)),
        "l0": sorted(k for k in keys if L0_AGGREGATE_RE.fullmatch(k)),
        "ce_kl": [k for k in CE_KL_HEADLINE if k in keys],
    }
    assert grouped["faithfulness"], f"{FAITHFULNESS_KEY} not found — was faithfulness logged?"
    assert grouped["impmin"], f"{IMPMIN_KEY} not found — was the train loss logged?"
    assert grouped["recon"], (
        "no recon aggregate keys (eval/loss/<Class> containing 'Recon') found — the "
        "metric-key format changed; update is_recon_aggregate"
    )
    assert grouped["l0"], (
        "no aggregate L0 keys (eval/l0/<threshold>_(layer_<n>|total)) found — the "
        "metric-key format or the CI_L0 group names changed; update L0_AGGREGATE_RE"
    )
    return grouped


def load_steps(run_dir: Path) -> dict[int, dict[str, float]]:
    """Merge all metrics.jsonl lines into {step: {key: scalar}} (train + eval lines)."""
    merged: dict[int, dict[str, float]] = defaultdict(dict)
    path = run_dir / "metrics.jsonl"
    assert path.exists(), f"missing {path}"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.pop("step")
            for k, v in row.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue  # skip non-scalar (images/charts already excluded on write)
                merged[step][k] = float(v)
    return dict(merged)


def at_step(steps: dict[int, dict[str, float]], step: int, run_name: str) -> dict[str, float]:
    assert step in steps, f"{run_name}: no metrics logged at step {step} (have {sorted(steps)})"
    return steps[step]


def _inject_faithfulness(data: dict[tuple[str, int, str], dict[str, float]]) -> None:
    """Coalesce loss-side (train/loss/FaithfulnessLoss, faith phases) and eval-probe
    (eval/loss/FaithfulnessLoss, nofaith phase) weight faithfulness into one 'faithfulness'
    key, so a single panel spans every phase."""
    for d in data.values():
        tf = d.get("train/loss/FaithfulnessLoss")
        ef = d.get("eval/loss/FaithfulnessLoss")
        v = tf if tf is not None else ef
        if v is not None:
            d["faithfulness"] = v


def collect(runs_dir: Path, train_steps: int) -> dict[tuple[str, int, str], dict[str, float]]:
    """(scheme, seed, phase) -> {metric_key: value}."""
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for run_dir in sorted(runs_dir.glob("cinit-*")):
        parts = run_dir.name.split("-")  # cinit-<scheme>-s<seed>-<phase>
        assert len(parts) == 4 and parts[0] == "cinit", f"unexpected run name {run_dir.name}"
        scheme, seed_str, kind = parts[1], parts[2], parts[3]
        seed = int(seed_str.lstrip("s"))
        steps = load_steps(run_dir)
        if kind == "raw":
            out[(scheme, seed, "init")] = at_step(steps, 0, run_dir.name)
        elif kind == "train":
            out[(scheme, seed, "post-warmup")] = at_step(steps, 0, run_dir.name)
            out[(scheme, seed, "trained")] = at_step(steps, train_steps, run_dir.name)
        elif kind == "nofaith":
            out[(scheme, seed, "trained-nofaith")] = at_step(steps, train_steps, run_dir.name)
        else:
            raise AssertionError(f"unknown phase {kind!r}")
    _inject_faithfulness(out)
    return out


def metric_keys(data) -> list[str]:
    keys: set[str] = set()
    for d in data.values():
        keys.update(d)
    return sorted(keys)


def seeds_present(data) -> list[int]:
    return sorted({seed for (_, seed, _) in data})


def pow10_limits(vals: list[float]) -> tuple[float, float] | None:
    """Highest power of ten below the min, lowest power of ten above the max."""
    pos = [v for v in vals if v > 0]
    if not pos:
        return None
    lo = 10.0 ** math.floor(math.log10(min(pos)))
    hi = 10.0 ** math.ceil(math.log10(max(vals)))
    return lo, (hi if hi > lo else lo * 10)


def group_values(data, keys, seeds, phases) -> list[float]:
    out: list[float] = []
    for key in keys:
        for scheme in SCHEMES:
            for seed in seeds:
                for phase in phases:
                    d = data.get((scheme, seed, phase))
                    if d is not None and key in d:
                        out.append(d[key])
    return out


def plot_group(data, group, keys, seeds, phases, out_dir, prefix) -> None:
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.6), squeeze=False)
    axes = axes[0]

    shared = (
        pow10_limits(group_values(data, keys, seeds, phases)) if group in SHARED_GROUPS else None
    )
    xs = list(range(len(phases)))
    for ax, key in zip(axes, keys, strict=True):
        for scheme in SCHEMES:
            label, color = SCHEME_STYLE[scheme]
            means, lows, highs, seed_pts = [], [], [], []
            for phase in phases:
                vals = [
                    data[(scheme, seed, phase)][key]
                    for seed in seeds
                    if (scheme, seed, phase) in data and key in data[(scheme, seed, phase)]
                ]
                means.append(sum(vals) / len(vals) if vals else math.nan)
                lows.append(min(vals) if vals else math.nan)
                highs.append(max(vals) if vals else math.nan)
                seed_pts.append(vals)
            ax.fill_between(xs, lows, highs, color=color, alpha=0.18, lw=0)
            ax.plot(xs, means, "-o", color=color, lw=2, label=label)
            for x, vals in zip(xs, seed_pts, strict=True):
                ax.scatter([x] * len(vals), vals, color=color, alpha=0.35, s=16, zorder=3)

        if group in LOG_GROUPS:
            ax.set_yscale("log")
            lim = (
                shared
                if shared is not None
                else pow10_limits(group_values(data, [key], seeds, phases))
            )
            if lim is not None:
                ax.set_ylim(*lim)
        ax.set_title(key.split("/")[-1], fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(phases, rotation=20, ha="right")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{prefix}: {group}" + (" (shared log axis)" if group in SHARED_GROUPS else ""))
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{group}.png", dpi=150)
    plt.close(fig)


def write_summary_csv(data, keys, seeds, out_dir: Path) -> None:
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "seed", "phase", *keys])
        for scheme in SCHEMES:
            for seed in seeds:
                for phase in ALL_PHASES:
                    d = data.get((scheme, seed, phase))
                    if d is None:
                        continue
                    w.writerow([scheme, seed, phase, *[d.get(k, "") for k in keys]])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs_dir", type=Path, help="dir containing cinit-* run folders")
    p.add_argument(
        "--out", type=Path, default=None, help="figures dir (default: <runs_dir>/figures)"
    )
    p.add_argument("--train-steps", type=int, default=200)
    args = p.parse_args()

    out_dir = args.out or (args.runs_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(args.runs_dir, args.train_steps)
    assert data, f"no cinit-* runs found under {args.runs_dir}"
    seeds = seeds_present(data)
    keys = metric_keys(data)
    grouped = group_keys(keys)

    print(f"conditions: {len(data)} | seeds: {seeds} | metric keys: {len(keys)}")
    for group in GROUP_ORDER:
        if grouped.get(group):
            print(f"  {group}: {grouped[group]}")
    for series_name, phases in SERIES.items():
        for group in GROUP_ORDER:
            if grouped.get(group):
                plot_group(data, group, grouped[group], seeds, phases, out_dir, series_name)
    write_summary_csv(data, keys, seeds, out_dir)
    print(f"wrote {out_dir}/{{{','.join(SERIES)}}}_*.png + summary.csv")


if __name__ == "__main__":
    main()
