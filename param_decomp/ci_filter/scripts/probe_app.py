"""Build the browsable probe page from `nontarget_probe`'s output.

    python -m param_decomp.ci_filter.scripts.probe_app --probe <.../nontarget_probe>

Copies `probe_app.html` next to a `data.js` holding, per probed component, its kept sequences
(tokens, per-token KL and CI, and both forwards' argmax) plus the summary row. Open
`<probe>/app/index.html` over `file://`."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[1] / "probe_app.html"


def probe_app(probe: Path) -> Path:
    rows: dict[str, list[dict[str, Any]]] = json.loads((probe / "heatmap_tokens.json").read_text())
    summary = {
        f"{row['site']}|{row['component']}": {
            key: float(value)
            for key, value in row.items()
            if key not in {"site", "kind", "layer", "component"} and value not in {"", None}
        }
        for row in csv.DictReader((probe / "summary.tsv").open(), delimiter="\t")
    }
    missing = set(rows) - set(summary)
    assert not missing, f"sequences without a summary row: {sorted(missing)[:3]}"

    out = probe / "app"
    out.mkdir(exist_ok=True)
    (out / "data.js").write_text(
        f"window.PROBE_DATA = {json.dumps(rows)};\nwindow.PROBE_SUMMARY = {json.dumps(summary)};\n"
    )
    shutil.copy(APP, out / "index.html")
    sequences = sum(len(value) for value in rows.values())
    print(f"{len(rows)} components, {sequences} sequences -> {out / 'index.html'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, required=True, help="the nontarget_probe directory")
    args = ap.parse_args()
    probe_app(args.probe.expanduser())


if __name__ == "__main__":
    main()
