"""Component ids for the variance results, as the AB-grid applet labels them.

Every array of `results/L<l>/{variance,J}_pos<tau>.npz` is indexed by COLUMN: position in block l's
alive list of that kind (`alive_only/L<l>/comps.npz`, gate / up / down in `alive/alive.json`
order). The component's id is its index within its site, `comps.npz["index"]`, which is the id the
AB-grid applet shows (`modules[].saved` in `ab_grids/*.js`, site `layers.<l>.mlp.<kind>_proj`).

    from ids import component_ids, label
    ids = component_ids(src, 0)        # {"gate": (G,), "up": (U,), "down": (D,)} int
    label(0, "down", 69, ids)          # "layers.0.mlp.down_proj:92"
"""

from pathlib import Path

import numpy as np

KINDS = ("gate", "up", "down")


def component_ids(src: Path, layer: int) -> dict[str, np.ndarray]:
    comps = np.load(Path(src) / "alive_only" / f"L{layer}" / "comps.npz")
    return {k: comps["index"][comps["kind"] == k].astype(np.int64) for k in KINDS}


def label(layer: int, kind: str, column: int, ids: dict[str, np.ndarray]) -> str:
    return f"layers.{layer}.mlp.{kind}_proj:{int(ids[kind][column])}"
