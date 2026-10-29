"""Build the residual-RMS heatmap applet from `collect.py`'s output.

Writes OUT/applet/index.html (a copy of `app.html`), OUT/applet/meta.js, and one data file per
(stream point p, position t), OUT/applet/data/p<p>_t<t>.js, loaded by the page on demand. Each data
file holds the six 400 x 400 grids (model m in original, ci_masked, alive_only; operation o in add,
sub) of r(a, b), row a - 1, column b - 1, each quantised to uint16 between its own min and max
(exact to (max - min) / 65535), base64 little-endian.

    python build_applet.py [--src DIR]
"""

import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/resid_rms")
N_SIDE = 400
POSITIONS = ("<BOS>", "a", "op", "b", "=")


def point_labels(n_layer: int) -> list[str]:
    labels = ["embed"]
    for block in range(n_layer):
        labels += [f"L{block}.attn", f"L{block}.mlp"]
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    args = ap.parse_args()
    meta = json.loads((args.src / "meta.json").read_text())
    rms = np.load(args.src / "rms.npy", mmap_mode="r")
    n_model, n_point, n, n_pos = rms.shape
    assert (n_model, n, n_pos) == (3, 2 * N_SIDE * N_SIDE, len(POSITIONS)), rms.shape
    kl = np.load(args.src / "kl_last.npy")
    out = args.src / "applet"
    (out / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("app.html"), out / "index.html")

    for p in range(n_point):
        block = np.asarray(rms[:, p], np.float64)  # (3, N, 5)
        for t in range(n_pos):
            grids = block[:, :, t].reshape(n_model * 2, N_SIDE * N_SIDE)  # (model, op) row-major
            lo, hi = grids.min(1), grids.max(1)
            span = np.where(hi > lo, hi - lo, 1.0)
            q = np.rint((grids - lo[:, None]) / span[:, None] * 65535).astype("<u2")
            payload = {"lo": lo.tolist(), "hi": hi.tolist(),
                       "q": base64.b64encode(q.tobytes()).decode()}  # fmt: skip
            (out / "data" / f"p{p}_t{t}.js").write_text(
                f"window.RMS_DATA['p{p}_t{t}']={json.dumps(payload)};\n"
            )
        print(f"point {p} done", flush=True)

    app_meta = {
        "models": ["original", "ci_masked", "alive_only"],
        "ops": ["add", "sub"],
        "points": point_labels((n_point - 1) // 2),
        "positions": list(POSITIONS),
        "n_side": N_SIDE,
        "kl_mean": {"ci_masked": float(kl[0].mean()), "alive_only": float(kl[1].mean())},
        "source": {k: meta[k] for k in ("run_dir", "step", "filter", "threshold", "commit")},
    }
    (out / "meta.js").write_text(f"window.RMS_META={json.dumps(app_meta)};\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
