"""Score how cleanly each alive subcomponent's CI pattern isolates a single operand period.

The addition/subtraction subcomponents activate periodically in the operands (periods 2, 5,
10, 20, 50, 100 — the model's Fourier features store integers modulo these). A *cleanly
separated* decomposition assigns each subcomponent a single such period; a mixed one spreads
several periods across one subcomponent. This script quantifies that from the
`find_alive_subcomponents` per-position CI JSON (CPU-only, no model forward):

For each (position, module, subcomponent) with mean CI over the displayed prompts above
`--min-mass`, the `[b, a]` CI grid is DC-removed and 2D-FFT'd. Power is grouped into
conjugate-symmetric frequency orbits `{(ka, kb), (-ka, -kb)}`; each orbit is labelled by its
direction — `a` (kb=0), `b` (ka=0), `a+b` (ka=kb), `a-b` (ka=-kb), else `mixed2d` — and its
period `N/gcd(ka, kb)` along that direction. Reported per subcomponent:

- `purity`: the dominant orbit's share of total non-DC power (1.0 = one clean sinusoid).
- `band_purity`: share of power in the top-1 orbit AND its harmonics (same direction,
  frequency an integer multiple) — a near-binary CI stripe of period 10 spreads power
  across period 10/5/3.3/... harmonic orbits yet is one clean pattern; this pools them.
  The headline per-component cleanliness number; `> 0.5` counts as clean.
- `n_orbits_50` / `n_orbits_90`: orbits needed to reach 50% / 90% of the power (1 = clean,
  higher = mixture). The 90% variant is dominated by the broadband speckle-noise floor on
  real CI grids; prefer the 50% one.
- `top{1,2,3}_{label,period,share}`: the three dominant orbits.

Always-on subcomponents (CI ≈ 1 everywhere, std of the grid < 0.05) carry no periodic
signal; they get `top1_label = flat`, empty score cells, and are excluded from the summary
aggregates (counted in `n_flat`).

Subtraction prompts only cover a triangle of the (a, b) grid, so missing cells would alias;
rows are computed per op and for `-` the grid is masked to displayed cells and analysed via
its two 1D marginals (nan-mean over the displayed cells) instead of the 2D FFT; `label` is
then the marginal axis (`a`/`b`). Run-level comparison should lean on the `+` rows.

The summary TSV aggregates per (op, position, module): median / mass-weighted
`band_purity`, mean `n_orbits_50`, the clean count (`band_purity > 0.5`), `n_flat`, and
the per-period component counts — the numbers to compare across runs/recipes.

Usage:
    python -m param_decomp_lab.scripts.validation.score_period_separation <per_position_json> \
        [--min-mass=0.01] [--output=PATH] [--output-summary=PATH]

Output (beside the input, in `analysis/datasets/`): `period_separation.tsv` (one row per
op × position × subcomponent) and `period_separation_summary.tsv`. A JSON named
`..._step<k>.json` yields `period_separation_step<k>.tsv` etc., so multi-checkpoint scores
coexist.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import (
    PerPosition,
    build_ci_grids,
    parse_ab_prompts,
    parse_module_name,
)

_OPS = ("+", "-")
_CLEAN_BAND = 0.5
_FLAT_CONTRAST = 0.05

_FIELDS = [
    "op", "pos", "layer", "matrix", "component", "mass", "contrast", "purity", "band_purity",
    "n_orbits_50", "n_orbits_90",
    "top1_label", "top1_period", "top1_share",
    "top2_label", "top2_period", "top2_share",
    "top3_label", "top3_period", "top3_share",
]  # fmt: skip
_SUMMARY_FIELDS = [
    "op", "pos", "layer", "matrix", "n_scored", "n_flat", "n_clean", "median_band_purity",
    "mass_weighted_band_purity", "mean_n_orbits_50", "period_counts",
]  # fmt: skip


def _orbit_label(ka: int, kb: int, n: int) -> tuple[str, int]:
    """Direction label and integer period of the frequency orbit `(ka, kb)` on an n×n grid.

    `ka`/`kb` are the a-axis / b-axis frequencies with `kb` folded into `(-n/2, n/2]`.
    """
    if kb == 0:
        return "a", round(n / ka)
    if ka == 0:
        return "b", round(n / abs(kb))
    if ka == kb:
        return "a+b", round(n / ka)
    if ka == -kb:
        return "a-b", round(n / ka)
    return "mixed2d", round(n / math.gcd(ka, abs(kb)))


class Orbit(NamedTuple):
    """One conjugate frequency orbit: human label + integer period + power share, plus the
    reduced direction `direction` and fundamental frequency `freq` (`(ka, kb) = freq *
    direction`) that let harmonics of a common fundamental be pooled."""

    label: str
    period: int
    share: float
    direction: tuple[str | int, int]
    freq: int

    def is_harmonic_of(self, other: "Orbit") -> bool:
        return self.direction == other.direction and self.freq % other.freq == 0


def _orbit_powers_2d(grid: NDArray[np.floating]) -> list[Orbit]:
    """Non-DC power per conjugate orbit of a square grid, descending by share.
    `grid` is `[b, a]`, so axis 1 carries the a-frequency."""
    n_b, n_a = grid.shape
    assert n_a == n_b, f"period labels assume a square grid, got {grid.shape}"
    n = n_a
    f = np.fft.fft2(grid - grid.mean())
    power = np.abs(f) ** 2
    orbits: dict[tuple[int, int], float] = defaultdict(float)
    for kb_raw in range(n):
        for ka_raw in range(n):
            if ka_raw == 0 and kb_raw == 0:
                continue
            ka = ka_raw - n if ka_raw > n // 2 else ka_raw
            kb = kb_raw - n if kb_raw > n // 2 else kb_raw
            key = (ka, kb) if (ka > 0 or (ka == 0 and kb > 0)) else (-ka, -kb)
            orbits[key] += float(power[kb_raw, ka_raw])
    total = sum(orbits.values())
    if total < 1e-12:
        return []
    ranked = sorted(orbits.items(), key=lambda kv: kv[1], reverse=True)
    return [
        Orbit(*_orbit_label(ka, kb, n), p / total, (ka // g, kb // g), g)
        for (ka, kb), p in ranked
        for g in (math.gcd(ka, abs(kb)),)
    ]


def _orbit_powers_marginals(
    grid: NDArray[np.floating], displayed: NDArray[np.bool_]
) -> list[Orbit]:
    """1D-FFT power orbits of the two displayed-cell marginals, pooled and normalised.

    Fallback for ops whose prompts don't cover the full grid (subtraction's triangle):
    `f(a)` = nan-mean of CI over the displayed b's, and vice versa. Diagonal (`a±b`)
    structure is invisible to marginals — `+` rows are the comparable ones.
    """
    masked = np.where(displayed, grid, np.nan)
    with np.errstate(invalid="ignore"):
        marginals = (("a", np.nanmean(masked, axis=0)), ("b", np.nanmean(masked, axis=1)))
    powers: list[Orbit] = []
    for axis_label, marginal in marginals:
        marginal = marginal[~np.isnan(marginal)]
        if marginal.size < 4:
            continue
        centered = marginal - marginal.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2
        for k in range(1, power.shape[0]):
            powers.append(
                Orbit(axis_label, round(marginal.size / k), float(power[k]), (axis_label, 0), k)
            )
    total = sum(o.share for o in powers)
    if total < 1e-12:
        return []
    return [
        o._replace(share=o.share / total)
        for o in sorted(powers, key=lambda o: o.share, reverse=True)
    ]


def score_period_separation(
    json_path: str,
    min_mass: float = 0.01,
    output: str | None = None,
    output_summary: str | None = None,
) -> tuple[Path, Path]:
    """Write per-subcomponent period-purity rows and a per-matrix summary. Returns both paths."""
    json_file = Path(json_path).expanduser()
    data: PerPosition = json.loads(json_file.read_text())

    suffix = json_file.stem.removeprefix("alive_subcomponents_per_position")
    out_dir = json_file.parent
    out_path = Path(output).expanduser() if output else out_dir / f"period_separation{suffix}.tsv"
    out_summary_path = (
        Path(output_summary).expanduser()
        if output_summary
        else out_dir / f"period_separation_summary{suffix}.tsv"
    )

    rows: list[dict[str, object]] = []
    for op in _OPS:
        ab, a_max, b_max = parse_ab_prompts(data, op, grep=None)
        modules = sorted(
            {m for pp in data.values() for pm in pp.values() for m in pm}, key=parse_module_name
        )
        grids = build_ci_grids(data, ab, a_max, b_max, set(modules))
        displayed = np.zeros((b_max, a_max), dtype=bool)
        for a, b in ab.values():
            displayed[b - 1, a - 1] = True
        full_grid = bool(displayed.all())

        for pos in sorted(grids, key=int):
            for module in modules:
                for component, grid in sorted(grids[pos].get(module, {}).items()):
                    mass = float(grid[displayed].mean())
                    if mass <= min_mass:
                        continue
                    contrast = float(grid[displayed].std())
                    layer, matrix = parse_module_name(module)
                    if contrast < _FLAT_CONTRAST:
                        rows.append(
                            {
                                "op": op,
                                "pos": int(pos),
                                "layer": layer,
                                "matrix": matrix,
                                "component": component,
                                "mass": round(mass, 4),
                                "contrast": round(contrast, 4),
                                "purity": "",
                                "band_purity": "",
                                "n_orbits_50": "",
                                "n_orbits_90": "",
                                "top1_label": "flat",
                                "top1_period": 0,
                                "top1_share": 0.0,
                                "top2_label": "",
                                "top2_period": 0,
                                "top2_share": 0.0,
                                "top3_label": "",
                                "top3_period": 0,
                                "top3_share": 0.0,
                            }  # fmt: skip
                        )
                        continue
                    orbits = (
                        _orbit_powers_2d(grid)
                        if full_grid
                        else _orbit_powers_marginals(grid, displayed)
                    )
                    if not orbits:
                        continue
                    cum = np.cumsum([o.share for o in orbits])
                    band = sum(o.share for o in orbits if o.is_harmonic_of(orbits[0]))
                    row: dict[str, object] = {
                        "op": op, "pos": int(pos), "layer": layer, "matrix": matrix,
                        "component": component, "mass": round(mass, 4),
                        "contrast": round(contrast, 4),
                        "purity": round(float(orbits[0].share), 4),
                        "band_purity": round(band, 4),
                        "n_orbits_50": int(np.searchsorted(cum, 0.50) + 1),
                        "n_orbits_90": int(np.searchsorted(cum, 0.90) + 1),
                    }  # fmt: skip
                    for i in range(3):
                        in_range = i < len(orbits)
                        row[f"top{i + 1}_label"] = orbits[i].label if in_range else ""
                        row[f"top{i + 1}_period"] = orbits[i].period if in_range else 0
                        row[f"top{i + 1}_share"] = round(orbits[i].share, 4) if in_range else 0.0
                    rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    by_group: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[
            (str(row["op"]), int(str(row["pos"])), int(str(row["layer"])), str(row["matrix"]))
        ].append(row)
    for (op, pos, layer, matrix), group in sorted(by_group.items()):
        flat = [r for r in group if r["top1_label"] == "flat"]
        scored_rows = [r for r in group if r["top1_label"] != "flat"]
        if not scored_rows:
            continue
        masses = np.array([float(str(r["mass"])) for r in scored_rows])
        bands = np.array([float(str(r["band_purity"])) for r in scored_rows])
        period_counts: dict[str, int] = defaultdict(int)
        for r in scored_rows:
            period_counts[f"{r['top1_label']}:{r['top1_period']}"] += 1
        summary_rows.append(
            {
                "op": op,
                "pos": pos,
                "layer": layer,
                "matrix": matrix,
                "n_scored": len(scored_rows),
                "n_flat": len(flat),
                "n_clean": int((bands > _CLEAN_BAND).sum()),
                "median_band_purity": round(float(np.median(bands)), 4),
                "mass_weighted_band_purity": round(float((bands * masses).sum() / masses.sum()), 4),
                "mean_n_orbits_50": round(
                    float(np.mean([int(str(r["n_orbits_50"])) for r in scored_rows])), 2
                ),
                "period_counts": " ".join(
                    f"{k}={v}" for k, v in sorted(period_counts.items(), key=lambda kv: -kv[1])
                ),
            }  # fmt: skip
        )
    with open(out_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    scored = sum(1 for r in rows if r["top1_label"] != "flat")
    clean = sum(
        1 for r in rows if r["top1_label"] != "flat" and float(str(r["band_purity"])) > _CLEAN_BAND
    )
    logger.info(
        f"scored {scored} (op, pos, subcomponent) grids — {clean} clean "
        f"(band_purity>{_CLEAN_BAND}) → {out_path}, {out_summary_path}"
    )
    return out_path, out_summary_path


if __name__ == "__main__":
    fire.Fire(score_period_separation)
