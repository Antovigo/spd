"""One TSV row per component alive at the decomposition's endpoint: where it stays alive across
the filters, and what the attribution says about it.

    python -m param_decomp.ci_filter.scripts.component_table --run_dir <run> [--step 40000]

Writes `<run_dir>/analysis/ci_filter/step_<step>/components.tsv`. Columns:

- `site`, `component` — the component's address, and `layer`, `kind` split out for grouping;
- `alive_<filter>` / `max_ci_<filter>` — alive at the filter's end (max output CI over the
  20k-prompt pool > 0.01) and that max CI. `alive_start` is the decomposition's own alive set
  (this table's row set), read from the pruning filter's `alive/kept.npz`;
- `effect_<source>[_add|_sub]` — mean first-order effect of ABLATING it on `log p(correct
  answer)`, over every pool prompt and split per operation. Positive = ablating HELPS, i.e. the
  component impairs the answer;
- `impairs_frac_<source>[_add|_sub]` — the share of prompts on which ablating it helps, size
  ignored: "impairs on most prompts" is this column > 0.5;
- `active_frac_<source>` — share of pool prompts where its CI exceeds 0.01;
- `swept_d{kl,integer_kl,answer_ce,accuracy}_<source>` — the MEASURED change from ablating it
  alone over the sweep's fixed prompt subset (`ablation_sweep.tsv`). A NEGATIVE `answer_ce`
  delta means removing it improves the arithmetic;
- `ablated_dscore_<source>` / `ablated_dacc_<source>` — the same measurement from the earlier
  1000-prompt verification of the extremes, empty elsewhere.

Sources are the filters plus `run` (the decomposition's own CI fn, `attribution_run/`)."""

import argparse
import json
from pathlib import Path

import numpy as np

ATTRIBUTION = {"run": "attribution_run"}
"""Sources whose attribution does not live under a filter directory."""

FILTERS = {
    "lastpos": "addsub-05-filter-last-pos",
    "prune": "addsub-05-filter-last-pos-alive",
    "ceiling": "addsub-05-filter-last-pos-ceiling",
    "integers": "addsub-05-filter-integers",
    "ce": "addsub-05-filter-answer-ce",
}
SWEPT_METRICS = ("kl", "integer_kl", "answer_ce", "accuracy")
START_FROM = "prune"
"""Whose `alive/kept.npz` is the decomposition's own alive set (it pruned from the run)."""


def component_table(run_dir: Path, step: int) -> Path:
    root = run_dir / "analysis" / "ci_filter" / f"step_{step}"
    with np.load(root / FILTERS[START_FROM] / "alive" / "kept.npz") as kept:
        start = {site: kept[site] for site in kept.files}

    max_ci: dict[str, dict[str, np.ndarray]] = {}
    for name, filter_id in FILTERS.items():
        with np.load(root / filter_id / "alive" / "max_ci.npz") as saved:
            max_ci[name] = {site: saved[site] for site in saved.files}

    effects: dict[str, dict[str, np.ndarray]] = {}
    measured: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    swept: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    sources = {**{n: root / f for n, f in FILTERS.items()}, "run": root}
    for name, base in sources.items():
        attribution = base / ATTRIBUTION.get(name, "attribution")
        if not (attribution / "components.npz").exists():
            continue
        with np.load(attribution / "components.npz") as saved:
            effects[name] = {key: saved[key] for key in saved.files}
        verified = json.loads((attribution / "verified.json").read_text())
        measured[name] = {(r["site"], int(r["component"])): r for r in verified}
        sweep = attribution / "ablation_sweep.tsv"
        if sweep.exists():
            lines = sweep.read_text().splitlines()
            header = lines[0].split("\t")
            swept[name] = {}
            for line in lines[1:]:
                cells = line.split("\t")
                row = dict(zip(header, cells, strict=True))
                swept[name][(row["site"], int(row["component"]))] = row

    header = ["site", "layer", "kind", "component", "alive_start"]
    for name in FILTERS:
        header += [f"alive_{name}", f"max_ci_{name}"]
    for name in effects:
        header += [
            f"effect_{name}",
            f"effect_{name}_add",
            f"effect_{name}_sub",
            f"impairs_frac_{name}",
            f"impairs_frac_{name}_add",
            f"impairs_frac_{name}_sub",
            f"active_frac_{name}",
            f"ablated_dscore_{name}",
            f"ablated_dacc_{name}",
        ]
        if name in swept:
            header += [f"swept_d{metric}_{name}" for metric in SWEPT_METRICS]

    n_prompts = {
        name: _pool_size(sources[name] / ATTRIBUTION.get(name, "attribution")) for name in effects
    }
    rows: list[str] = ["\t".join(header)]
    for site in sorted(start):
        layer, kind = site.split(".", 2)[1], site.split(".", 2)[2]
        for component in np.nonzero(start[site])[0]:
            component = int(component)
            values: list[str] = [site, layer, kind, str(component), "1"]
            for name in FILTERS:
                ci = float(max_ci[name][site][component])
                values += [str(int(ci > 0.01)), f"{ci:.6g}"]
            for name, per_site in effects.items():
                row = measured[name].get((site, component))
                pool_size = {"all": n_prompts[name], "add": 0, "sub": 0}
                pool_size["add"] = pool_size["sub"] = n_prompts[name] // 2
                values += [
                    f"{float(per_site[f'{site}|all'][component]):.6g}",
                    f"{float(per_site[f'{site}|add'][component]):.6g}",
                    f"{float(per_site[f'{site}|sub'][component]):.6g}",
                ]
                values += [
                    (
                        f"{float(per_site[f'positive:{site}|{part}'][component]) / pool_size[part]:.4g}"
                        if f"positive:{site}|{part}" in per_site
                        else ""
                    )
                    for part in ("all", "add", "sub")
                ]
                values += [
                    f"{float(per_site[f'active:{site}|all'][component]) / n_prompts[name]:.4g}",
                    "" if row is None else f"{float(row['ablated_delta_score']):.6g}",
                    "" if row is None else f"{float(row['ablated_delta_accuracy']):.6g}",
                ]
                if name in swept:
                    measured_row = swept[name].get((site, component))
                    values += [
                        "" if measured_row is None else measured_row[f"d_{metric}"]
                        for metric in SWEPT_METRICS
                    ]
            rows.append("\t".join(values))

    out = root / "components.tsv"
    out.write_text("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} components x {len(header)} columns -> {out}")
    return out


def _pool_size(attribution: Path) -> int:
    return int(json.loads((attribution / "summary.json").read_text())["n_prompts"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=40000)
    args = ap.parse_args()
    component_table(args.run_dir.expanduser(), args.step)


if __name__ == "__main__":
    main()
