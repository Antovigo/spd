"""Figure of the notebook page "Mod 10 representations", section 6: each mod-10 reader next to the writers it feeds.

Writers per reader come from the representation repository (`writers` field of position 1, written by
repr_repo/seed_l0_mod10_writers.py from the variance split). Grids are raw inner activations
t_c(a, b) = x . V_c at position 1 (the token a) of the alive-only model, on the prompts a + b = with
integer operands a, b in 1..400 (`variance/alive_only/L0/inner.npy`). Nothing is centred: the colour
scale is symmetric around the activation zero. Readers are multiplied by their sign s_c (page
notation); writers are shown in their raw gauge.

    python mod10_writers_fig.py --out <notebook dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import SRC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repr_repo"))
from repo import Repo  # noqa: E402

N = 400
SIGN = {"L0.up.54": -1}  # s_c of the page: only u54 is flipped
# main residue r of each reader (page section 2); u31 is left out of this figure
RESIDUE = {
    "u794": 0,
    "g43": 1,
    "g69": 2,
    "u60": 3,
    "g62": 4,
    "u52": 5,
    "u54": 6,
    "g58": 7,
    "g59": 8,
    "g47": 9,
}


def short(rid: str) -> str:
    _, kind, cid = rid.split(".")
    return f"{kind[0]}{cid}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    repo = Repo()
    readers = [
        rd
        for rd in repo.readers.values()
        if rd["positions"]["1"] and "writers" in rd["positions"]["1"] and short(rd["id"]) in RESIDUE
    ]
    assert len(readers) == len(RESIDUE)
    main_res = RESIDUE
    readers.sort(key=lambda rd: (main_res[short(rd["id"])], rd["id"]))
    comps = np.load(SRC / "comps.npz")
    col = {
        (str(k), int(i)): j
        for j, (k, i) in enumerate(zip(comps["kind"], comps["index"], strict=True))
    }
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    plus = np.asarray(inner[: N * N, 0, :], np.float32)  # rows (a - 1) * 400 + (b - 1), op "+"
    spread = np.abs(plus.reshape(N, N, -1) - plus.reshape(N, N, -1)[:, :1]).max()
    print(f"max |t(a, b) - t(a, 1)| over b at position 1: {spread:.3g}")

    def grid(kind: str, cid: int) -> np.ndarray:
        return plus[:, col[(kind, cid)]].reshape(N, N)

    ncol = 1 + max(len(rd["positions"]["1"]["writers"]["entries"]) for rd in readers)
    fig, axes = plt.subplots(len(readers), ncol, figsize=(1.25 * ncol, 1.45 * len(readers)))
    for row, rd in zip(axes, readers, strict=True):
        for ax in row:
            ax.axis("off")
        g = SIGN.get(rd["id"], 1) * grid(rd["kind"], rd["component"])
        panels = [(g, f"{short(rd['id'])}  (r = {main_res[short(rd['id'])]})", "#2a5db0")]
        for e in rd["positions"]["1"]["writers"]["entries"]:
            panels.append(
                (grid("down", e["component"]),
                 f"d{e['component']}  {e['reader_fraction']:.2f} · φ {e['phi']:+.2f}", "#1d1d1b")
            )  # fmt: skip
        for ax, (z, title, colour) in zip(row, panels, strict=False):
            m = float(np.abs(z).max()) or 1.0
            ax.imshow(z, cmap="RdBu_r", vmin=-m, vmax=m, origin="lower", extent=(1, N, 1, N),
                      interpolation="nearest")  # fmt: skip
            ax.set_title(f"{title}\n|t| ≤ {m:.3g}", fontsize=6, color=colour)
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(colour)
    fig.supxlabel(
        "each panel: b = 1..400 horizontally, a = 1..400 vertically (op +, position 1)", fontsize=8
    )
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "mod10_writers_grids.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
