"""Panoptic view of the RMSNorm divisors: one tab per model (plus the CI-masked / original ratio),
stream point p (0..64) down the rows, (position t, operation) across the 10 columns, each cell
the 400 x 400 (a, b) grid binned BIN x BIN (mean) to a thumbnail.

s = sqrt(r^2 + eps) as in `build_applet.py`; the ratio tab bins s_ci_masked / s_original per prompt.
Writes OUT/applet/panoptic.html (a copy of `panoptic.html`) and OUT/applet/data/panoptic_<tab>.js:
for every cell (p, t, op), in that order, the thumbnail quantised to uint16 between its own min and
max, base64 little-endian, with the cell's lo / hi. Needs `build_applet.py`'s meta.js.

    python build_panoptic.py [--src DIR] [--bin N]
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/resid_rms")
HUB = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots"
N_SIDE = 400
TABS = ("original", "ci_masked", "alive_only", "ratio")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--bin", type=int, default=4)
    args = ap.parse_args()
    assert N_SIDE % args.bin == 0, args.bin
    side = N_SIDE // args.bin
    eps = float(json.loads(sorted(HUB.glob("*/config.json"))[-1].read_text())["rms_norm_eps"])
    rms = np.load(args.src / "rms.npy", mmap_mode="r")
    n_model, n_point, _, n_pos = rms.shape
    out = args.src / "applet"
    assert (out / "meta.js").exists(), "run build_applet.py first"
    shutil.copy(Path(__file__).with_name("panoptic.html"), out / "panoptic.html")

    cells = {tab: ([], [], []) for tab in TABS}  # lo, hi, quantised thumbnails
    for p in range(n_point):
        s = np.sqrt(np.asarray(rms[:, p], np.float64) ** 2 + eps)  # (3, N, 5)
        s = s.reshape(n_model, 2, N_SIDE, N_SIDE, n_pos)
        values = {**{tab: s[m] for m, tab in enumerate(TABS[:3])}, "ratio": s[1] / s[0]}
        for tab, v in values.items():
            thumb = v.reshape(2, side, args.bin, side, args.bin, n_pos).mean(
                (2, 4)
            )  # (op, a, b, t)
            for t in range(n_pos):
                for op in range(2):
                    g = thumb[op, :, :, t]
                    lo, hi = float(g.min()), float(g.max())
                    span = hi - lo if hi > lo else 1.0
                    cells[tab][0].append(lo)
                    cells[tab][1].append(hi)
                    cells[tab][2].append(np.rint((g - lo) / span * 65535).astype("<u2"))
        print(f"point {p} done", flush=True)
    for tab, (lo, hi, q) in cells.items():
        payload = {"side": side, "bin": args.bin, "lo": lo, "hi": hi,
                   "q": base64.b64encode(np.stack(q).tobytes()).decode()}  # fmt: skip
        (out / "data" / f"panoptic_{tab}.js").write_text(
            f"window.PANOPTIC['{tab}']={json.dumps(payload)};\n"
        )
    print(f"wrote {out / 'panoptic.html'}", flush=True)


if __name__ == "__main__":
    main()
