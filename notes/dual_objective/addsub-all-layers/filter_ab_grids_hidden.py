#!/usr/bin/env python
"""Rewrite ab_grid snapshots keeping ONLY the hidden head, filtered on hidden mean CI.

    python filter_ab_grids_hidden.py <src ab_grids dir> <dst dir> [--floor 0.05]

WHY: in `addsub-all-layers-4xh100-05-hidden-only` (p-ba5a0c0d) the output head is FROZEN at
its `zero_init_readout` value, CI 0.5 for every component. `saved_indices` takes the mean-CI
floor as a max OVER ROLES, so all 124,928 components clear it and every snapshot is 6.7 GB —
a single JS file no browser will load. -05's are ~0.2 GB.

The rewrite drops the output role's arrays entirely and re-applies the floor to the HIDDEN
head's `mean_ci_hidden`, which is the only head this run trains.

PAYLOAD NOTE: the applet keys the historical names (`mean_ci`, `ci`) to the OUTPUT role and
falls back to a single-role reading when `ci_roles` has no `hidden` (`ab_grids_app.html:114`).
So the hidden arrays are written into those historical slots and `ci_roles` stays
`["output"]` — the pre-dual format, which every view renders. The data is the HIDDEN head's;
`source_role` records that (the applet ignores unknown keys), as does the output directory's
name. Do not read the label "CI" in the applet as the output head here.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def _decode(text: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


def filter_payload(payload: dict, floor: float) -> tuple[dict, int, int]:
    n_pos = len(payload["positions"])
    kept_total = 0
    seen_total = 0
    modules = []
    for module in payload["modules"]:
        c_total = int(module["C"])
        saved = np.asarray(module["saved"], dtype=np.int64)
        seen_total += saved.size
        entry = {"name": module["name"], "C": c_total}
        if "mean_ci_hidden" not in module:
            raise SystemExit(f"{module['name']}: no mean_ci_hidden — not a dual snapshot")
        mean_hidden = _decode(module["mean_ci_hidden"], np.float32).reshape(n_pos, c_total)
        entry["mean_ci"] = _b64(mean_hidden)
        if saved.size:
            # Keep a saved component when its hidden mean CI clears the floor at ANY position
            # — the same "at some position" rule the writer applies, on one role instead of max.
            keep = np.flatnonzero(mean_hidden[:, saved].max(axis=0) >= floor)
            entry["saved"] = [int(c) for c in saved[keep]]
            kept_total += keep.size
            if keep.size:
                ci = _decode(module["ci_hidden"], np.uint8).reshape(saved.size, -1)
                inner = _decode(module["inner"], np.float16).reshape(saved.size, -1)
                entry["ci"] = _b64(ci[keep])
                entry["inner"] = _b64(inner[keep])
        else:
            entry["saved"] = []
        modules.append(entry)
    out = {k: v for k, v in payload.items() if k != "modules"}
    out["modules"] = modules
    out["ci_roles"] = ["output"]  # single-role payload; see the module docstring
    out["source_role"] = "hidden"
    out["mean_ci_floor"] = floor
    return out, kept_total, seen_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--floor", type=float, default=0.05)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    written = []
    for path in sorted(args.src.glob("step_*.js"), key=lambda p: int(p.stem.removeprefix("step_"))):
        text = path.read_text()
        start, end = text.index("(") + 1, text.rindex(")")
        payload = json.loads(text[start:end])
        del text
        out, kept, seen = filter_payload(payload, args.floor)
        del payload
        target = args.dst / path.name
        target.write_text(f"window.registerABGrids({json.dumps(out)});")
        written.append(path.name)
        print(
            f"{path.name}: {seen} -> {kept} components, "
            f"{path.stat().st_size / 1e9:.2f} GB -> {target.stat().st_size / 1e9:.3f} GB",
            flush=True,
        )

    (args.dst / "manifest.js").write_text(f"window.AB_GRIDS_MANIFEST = {json.dumps(written)};\n")
    applet = args.src / "index.html"
    if applet.exists():
        (args.dst / "index.html").write_bytes(applet.read_bytes())
    print(f"wrote {len(written)} snapshots to {args.dst}")


if __name__ == "__main__":
    main()
