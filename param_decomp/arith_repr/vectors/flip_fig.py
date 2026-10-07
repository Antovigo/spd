"""Schematic of the b flip on b's Fourier planes, with the components' read and write directions
(alive-only model, position `=`).

    python -m param_decomp.arith_repr.vectors.flip_fig   # -> OUT/flip/flip_planes.png

One row per harmonic k (period T = 100 / gcd(k, 100)) and its flip neurons. Plane: b's add code at k,
F = F_{b,add}(k) (as in `flip_dirs`), with basis e1 = Re F / |Re F|, e2 = Im F made orthogonal to e1.
* Left, the stream entering L15's MLP: the b class means of each op (mean over the 100 prompts with
  that b, centred per op) projected on the plane (dots), their harmonic-k part (closed curve; the
  labels 1 and 2 mark b = 1, 2, so the direction of travel shows), and the read directions
  V * ln2 of the gate / up components feeding the row's neurons most (largest |U_c[n]| x rms of the
  component's inner): arrow length = alignment of the read direction with the plane (1 = inside it).
* Right, the stream entering L16's MLP: the same for both ops (sub is mirrored), the write directions
  U of the row's down components (arrows, length = alignment), and each down component's own
  contribution, its inner's (op, b) class means centred per op times U, projected (small dots).
Both panels share one scale: the larger curve's radius is 1."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from param_decomp.arith_repr.vectors.common import AUTOINTERP, RESID, comp_table
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_dirs import codes, plane
from param_decomp.arith_repr.vectors.flip_set import _site_order

L = 15
ROWS = (  # (k, neurons, down components writing them)
    (20, (7446, 11305), ("c16", "c64")),
    (1, (130,), ("c4",)),
    (10, (9205,), ("c21",)),
    (2, (6456, 12769), ("c19", "c35")),
)
ADD, SUB, READ_C, WRITE_C = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED = "#1a1a19", "#6b6a63"


def main() -> None:
    z = np.load(DIR / "eval_alive_base.npz")
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    ln2 = np.load(RESID / "norms.npz")["ln2"][L]
    comps = comp_table()
    lcols = np.flatnonzero(comps["layer"] == L)

    def site(kind: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        suffix = {"gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}[kind]
        key = f"layers.{L}.{suffix}"
        order = _site_order(L, suffix, uv[key + ".ids"])
        sel = np.flatnonzero(comps["site"][lcols] == key)
        names = [f"c{int(comps['cidx'][lcols[s]])}" for s in sel]
        return (
            names,
            uv[key + ".V"][:, order].astype(np.float64),
            uv[key + ".U"][order].astype(np.float64),
        )

    S = {k: site(k) for k in ("gate", "up", "down")}
    rms = {
        k: np.sqrt(z[f"q{L}.{k}"][:, 4].mean(0) / 10000.0) for k in ("gate", "up")
    }  # inner rms at `=`
    pts = {p: z[f"x{p}"][:200].reshape(2, 100, -1) / 100.0 for p in (15, 16)}
    pts = {p: v - v.mean(1, keepdims=True) for p, v in pts.items()}
    F = {p: codes(z[f"x{p}"])[1] for p in (15, 16)}
    Hd = z[f"h{L}.down"][4, :200] / 100.0  # (200, n_down) class means by (op, b)

    fig, axes = plt.subplots(len(ROWS), 2, figsize=(10, 4.6 * len(ROWS)))
    for row, (k, neurons, downs) in enumerate(ROWS):
        T = 100 // np.gcd(k, 100)
        th = 2 * np.pi * k * np.arange(0, 100, 0.05) / 100
        for col, p in enumerate((15, 16)):
            ax = axes[row, col]
            B = plane(F[p][("b", 0)][k - 1])  # (2, d)
            curves = []
            for o in (0, 1):
                c = B @ F[p][("b", o)][k - 1]  # (2,) complex
                curves.append(np.stack([2 * np.real(c[i] * np.exp(1j * th)) for i in (0, 1)]))
            scale = max(np.linalg.norm(cu, axis=0).max() for cu in curves)
            for o, colr, ls in ((0, ADD, "-"), (1, SUB, "--")):
                xy = (pts[p][o] @ B.T) / scale
                ax.scatter(xy[:, 0], xy[:, 1], s=6, color=colr, alpha=0.25, linewidths=0)
                cu = curves[o] / scale
                ax.plot(cu[0], cu[1], ls, color=colr, lw=2, label=("add", "sub")[o])
                c = B @ F[p][("b", o)][k - 1]
                for v in (1, 2):
                    q = (
                        np.array(
                            [2 * np.real(c[i] * np.exp(2j * np.pi * k * v / 100)) for i in (0, 1)]
                        )
                        / scale
                    )
                    ax.scatter(*q, s=40, color=colr, edgecolors="white", linewidths=1.5, zorder=4)
                    ax.annotate(
                        str(v), q, xytext=(5, 5), textcoords="offset points", color=INK, fontsize=9
                    )
            arrows = []
            if p == 15:
                for n in neurons:
                    for kind in ("gate", "up"):
                        names, V, Uc = S[kind]
                        imp = np.abs(Uc[:, n]) * rms[kind]
                        j = int(np.argmax(imp))
                        r = V[:, j] * ln2
                        arrows.append(
                            (
                                f"{kind} {names[j]} -> neuron {n}",
                                B @ (r / np.linalg.norm(r)),
                                READ_C,
                            )
                        )
            else:
                names, _, Ud = S["down"]
                for dn in downs:
                    j = names.index(dn)
                    u = Ud[j] / np.linalg.norm(Ud[j])
                    arrows.append((f"down {dn}", B @ u, WRITE_C))
                    hc = Hd[:, j].reshape(2, 100)
                    hc = hc - hc.mean(1, keepdims=True)
                    for o, colr in ((0, ADD), (1, SUB)):
                        xy = np.outer(hc[o], B @ Ud[j]) / scale
                        ax.scatter(
                            xy[:, 0],
                            xy[:, 1],
                            s=10,
                            color=colr,
                            marker="x",
                            alpha=0.6,
                            linewidths=1,
                        )
            key = []
            for i, (lab, vec, colr) in enumerate(arrows):
                tag = f"{'r' if p == 15 else 'w'}{i + 1}"
                ax.annotate(
                    "", vec, (0, 0), arrowprops={"arrowstyle": "-|>", "color": colr, "lw": 2}
                )
                ax.annotate(tag, vec, xytext=(3, 3), textcoords="offset points", color=INK,
                            fontsize=8, fontweight="bold")  # fmt: skip
                key.append(f"{tag}: {lab}, alignment {np.linalg.norm(vec):.2f}")
            ax.text(0.02, 0.98, "\n".join(key), transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.5, color=INK)  # fmt: skip
            ax.axhline(0, color=MUTED, lw=0.5)
            ax.axvline(0, color=MUTED, lw=0.5)
            ax.set_aspect("equal")
            lim = 1.35
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])
            where = "entering L15 MLP (read)" if p == 15 else "entering L16 MLP (write)"
            ax.set_title(
                f"b's plane, k = {k} (period {T}) — stream {where}", fontsize=10, color=INK
            )
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(0.5, 0.004, "dots: b class means (x: the down component's own contribution); curves: their harmonic-k part;"
             " 1, 2: b = 1, 2\narrows: read (left) / write (right) directions of components, length = alignment"
             " with the plane (1 = inside it)", ha="center", fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(DIR / "flip_planes.png", dpi=130)
    print("saved", DIR / "flip_planes.png")


if __name__ == "__main__":
    main()
