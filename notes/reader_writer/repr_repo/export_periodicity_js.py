"""Write the periodicity index (periodicity.json, seed_periodicity.py) as `periodicity.js` for the AB-grids applet.

The applet (param_decomp/ci_filter/grids_app.html, branch experiment/comp_pruning) loads it next to its slice files
and offers a "period" filter. Layout:

    window.PERIODICITY = {
      taus:      [2, 4, 5, 10, 20, 25, 50, 100]   the exact periods tau (dividing 100) the index tests
      blocks:    [0, ..., 16]                     blocks whose gate / up readers are indexed
      positions: {"<p>": {"<site>": {"<c>": [tau, ...]}}}
    }

p is the applet's position index (1 = the token a, 3 = the token b), site a module name such as
"layers.0.mlp.gate_proj", c the component index, and [tau, ...] the periods of that reader at that position (empty:
indexed, no period found). Components absent from `positions` were not indexed.

    python export_periodicity_js.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repo import ROOT  # noqa: E402

OUT = Path(
    "/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/addsub-08-last-pos-40000/ab_grids/periodicity.js"
)


def main() -> None:
    doc = json.loads((ROOT / "periodicity.json").read_text())
    positions: dict[str, dict[str, dict[str, list[int]]]] = {}
    for rd in doc["readers"].values():
        site = f"layers.{rd['block']}.mlp.{rd['kind']}_proj"
        for t, entry in rd["positions"].items():
            positions.setdefault(t, {}).setdefault(site, {})[str(rd["component"])] = entry[
                "periods"
            ]
    js = {
        "taus": [int(tau) for tau in next(iter(doc["index"].values()))],
        "blocks": doc["blocks"],
        "source": str(ROOT / "periodicity.json"),
        "positions": positions,
    }
    OUT.write_text("window.PERIODICITY = " + json.dumps(js, separators=(",", ":")) + ";\n")
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
