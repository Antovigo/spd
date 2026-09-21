"""One TSV row per component alive at the decomposition's endpoint: where it stays alive across
the filters, and what the attribution says about it.

    python -m param_decomp.ci_filter.scripts.component_table --run_dir <run> [--step 40000]

Writes `<run_dir>/analysis/ablations/step_<step>/components.tsv`, one row per component of
`decomposition_alive.npz` there. Filters are the ACTIVE runs under `ci_filter/step_<step>/`
(`Trash/` is ignored), named by their id minus the shared prefix. Columns:

- `site`, `component` — the component's address, and `layer`, `kind` split out for grouping;
- `alive_<filter>` / `max_ci_<filter>` — alive at the filter's end (max output CI over the
  20k-prompt pool > 0.01) and that max CI; `alive_start` is always 1 (the row set);
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

Sources are the filters plus `run` (the decomposition's own CI fn), each read from
`ablations/step_<step>/<source>/` when present."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from param_decomp.ci_filter.paths import TRASH, ablation_source_dir, ablations_dir, ci_filter_dir

SWEPT_METRICS = ("kl", "integer_kl", "answer_ce", "accuracy")


def _filters(run_dir: Path, step: int) -> dict[str, Path]:
    """`{short name: directory}` of the ACTIVE filtering runs (not `Trash/`), short names being
    the ids with their shared prefix dropped (`addsub-05-filter-integers` -> `integers`)."""
    root = ci_filter_dir(run_dir, step)
    found = sorted(
        d for d in root.iterdir() if d.name != TRASH and (d / "alive" / "max_ci.npz").exists()
    )
    prefix = os.path.commonprefix([d.name for d in found]) if len(found) > 1 else ""
    prefix = prefix[: prefix.rfind("-") + 1] if "-" in prefix else ""
    return {d.name.removeprefix(prefix) or d.name: d for d in found}


def component_table(run_dir: Path, step: int) -> Path:
    ablations = ablations_dir(run_dir, step)
    with np.load(ablations / "decomposition_alive.npz") as kept:
        start = {site: kept[site] for site in kept.files}
    filters = _filters(run_dir, step)

    max_ci: dict[str, dict[str, np.ndarray]] = {}
    for name, directory in filters.items():
        with np.load(directory / "alive" / "max_ci.npz") as saved:
            max_ci[name] = {site: saved[site] for site in saved.files}

    effects: dict[str, dict[str, np.ndarray]] = {}
    measured: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    swept: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    sources = {"run": "run", **{name: d.name for name, d in filters.items()}}
    for name, source in sources.items():
        attribution = ablation_source_dir(run_dir, step, source)
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
    for name in filters:
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
        name: _pool_size(ablation_source_dir(run_dir, step, sources[name])) for name in effects
    }
    rows: list[str] = ["\t".join(header)]
    for site in sorted(start):
        layer, kind = site.split(".", 2)[1], site.split(".", 2)[2]
        for component in np.nonzero(start[site])[0]:
            component = int(component)
            values: list[str] = [site, layer, kind, str(component), "1"]
            for name in filters:
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

    out = ablations / "components.tsv"
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
