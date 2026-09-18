"""One TSV row per component alive at the decomposition's endpoint: where it stays alive across
the filters, and what the attribution says about it.

    python -m param_decomp.ci_filter.scripts.component_table --run_dir <run> [--step 40000]

Writes `<run_dir>/analysis/ci_filter/step_<step>/components.tsv`. Columns:

- `site`, `component` — the component's address, and `layer`, `kind` split out for grouping;
- `alive_<filter>` / `max_ci_<filter>` — alive at the filter's end (max output CI over the
  20k-prompt pool > 0.01) and that max CI. `alive_start` is the decomposition's own alive set
  (this table's row set), read from the pruning filter's `alive/kept.npz`;
- `effect_<filter>[_add|_sub]` — mean first-order effect of ABLATING it on `log p(correct
  answer)`, over every pool prompt and split per operation. Positive = interferes;
- `active_frac_<filter>` — share of pool prompts where its CI exceeds 0.01;
- `ablated_dscore_<filter>` / `ablated_dacc_<filter>` — the MEASURED effect of ablating it alone
  (log-prob and accuracy), empty where it was not in the verified extremes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FILTERS = {
    "lastpos": "addsub-05-filter-last-pos",
    "prune": "addsub-05-filter-last-pos-alive",
    "ceiling": "addsub-05-filter-last-pos-ceiling",
    "integers": "addsub-05-filter-integers",
    "ce": "addsub-05-filter-answer-ce",
}
START_FROM = "prune"
"""Whose `alive/kept.npz` is the decomposition's own alive set (it pruned from the run)."""


def component_table(run_dir: Path, step: int) -> Path:
    root = run_dir / "analysis" / "ci_filter" / f"step_{step}"
    with np.load(root / FILTERS[START_FROM] / "alive" / "kept.npz") as kept:
        start = {site: kept[site] for site in kept.files}

    max_ci: dict[str, dict[str, np.ndarray]] = {}
    effects: dict[str, dict[str, np.ndarray]] = {}
    measured: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for name, filter_id in FILTERS.items():
        with np.load(root / filter_id / "alive" / "max_ci.npz") as saved:
            max_ci[name] = {site: saved[site] for site in saved.files}
        attribution = root / filter_id / "attribution"
        if attribution.exists():
            with np.load(attribution / "components.npz") as saved:
                effects[name] = {key: saved[key] for key in saved.files}
            measured[name] = {
                (row["site"], int(row["component"])): row
                for row in json.loads((attribution / "verified.json").read_text())
            }

    header = ["site", "layer", "kind", "component", "alive_start"]
    for name in FILTERS:
        header += [f"alive_{name}", f"max_ci_{name}"]
    for name in effects:
        header += [
            f"effect_{name}",
            f"effect_{name}_add",
            f"effect_{name}_sub",
            f"active_frac_{name}",
            f"ablated_dscore_{name}",
            f"ablated_dacc_{name}",
        ]

    n_prompts = {name: _pool_size(root / FILTERS[name] / "attribution") for name in effects}
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
                values += [
                    f"{float(per_site[f'{site}|all'][component]):.6g}",
                    f"{float(per_site[f'{site}|add'][component]):.6g}",
                    f"{float(per_site[f'{site}|sub'][component]):.6g}",
                    f"{float(per_site[f'active:{site}|all'][component]) / n_prompts[name]:.4g}",
                    "" if row is None else f"{float(row['ablated_delta_score']):.6g}",
                    "" if row is None else f"{float(row['ablated_delta_accuracy']):.6g}",
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
