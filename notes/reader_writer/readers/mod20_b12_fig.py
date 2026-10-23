"""Block 12's h -> g step drawn in the planes it reads, for the original model's neurons and for the components.

Block 12's MLP input at position 1 (residual stream mid_12, a = 1..400; float64 rebuilds of track_build.py and
track_build_orig.py). Nothing is centred: the cross in every plane is the activation zero.

Notation (a = integer operand 1..400; s = a mod 20; r = a mod 10; h = floor(a/10) mod 2; g = floor((a+5)/10) mod 2):
  P20        the frequency-1 plane of a mod 20 of mid_12 (mod20_track.plane), turned so that the vector from residue
             10's mean to residue 0's mean points right
  P10        the frequency-1 plane of a mod 10 of mid_12 (track_analyze.basis), turned so that residue 0's mean points right
  read vector of a reader, in residual-stream coordinates: rho = gamma * w, with gamma the block's post-attention RMSNorm
             gain and w the reader's row (neuron: a row of gate_proj or up_proj; component: its read vector V_c), so that
             the reader's activation is rho . X / rms(X)
  arrow      the projection of rho / ||rho|| onto the plane, scaled by 2.5 times the mean distance of the residue means to their centroid; its length is the
             cosine-squared root ||P^T rho|| / ||rho|| (1 = the read vector lies in the plane), printed as "in-plane"
  neuron n   gate_n = gate pre-activation, up_n = up pre-activation, write coefficient silu(gate_n) * up_n
  components feeding neuron n: gate_n = sum_c t_c U_gate[c, n] over gate components c; up_n likewise over up components

    python mod20_b12_fig.py --out notebook_mod20/fig
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod20_track import OUT, R10, S20, means, plane
from track_analyze import basis
from track_build import OUT as AOUT
from track_build import load_blocks, silu
from track_build_orig import OUT as OOUT
from track_build_orig import load_block

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
QCM = [plt.get_cmap(c) for c in ("Blues", "Oranges", "Greens", "Purples")]
NEURONS = (5212, 8953)
GATES_5212 = (
    37,
    33,
)  # gate components whose share of gate_5212 changes most between r >= 5 and r < 5
COLORS = {"up": "#b5442d", "gate": "#1f4e9c", "gate2": "#2a9d8f", "gate3": "#7a7a7a"}


def scol(s: int):
    return QCM[s // 5](0.45 + 0.12 * (s % 5))


def rcol(r: int):
    return QCM[0 if r < 5 else 1](0.45 + 0.12 * (r % 5))


def turned(P: np.ndarray, X: np.ndarray, lab: np.ndarray, a: int, b: int | None) -> np.ndarray:
    """P rotated inside its span so that mean(lab == a) - mean(lab == b) (or mean(lab == a)) points right."""
    m = means(X @ P, lab, lab.max() + 1)
    v = m[a] - (m[b] if b is not None else 0)
    ang = np.arctan2(v[1], v[0])
    rot = np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return P @ rot


def draw_plane(ax, X, P, lab, ncls, col, arrows, title) -> list[dict]:
    z = X @ P
    m = means(z, lab, ncls)
    for c in range(ncls):
        ax.scatter(*z[lab == c].T, s=4, color=col(c), alpha=0.55, lw=0)
    for c in range(ncls):
        ax.text(*m[c], str(c), fontsize=6.5, ha="center", va="center", weight="bold",
                bbox={"boxstyle": "circle,pad=0.12", "fc": "white", "ec": col(c), "lw": 0.8})  # fmt: skip
    ax.plot([0], [0], "k+", ms=10, zorder=5)
    R = float(np.linalg.norm(m - m.mean(0), axis=1).mean()) * 2.5
    out = []
    for name, rho, color in arrows:
        p = P.T @ rho / np.linalg.norm(rho)
        frac = float(np.linalg.norm(p))
        d = p * R
        ax.annotate(
            "",
            xy=d,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2},
            zorder=6,
        )
        ax.plot([], [], color=color, lw=2.2, label=f"{name}  (in-plane {frac:.2f})")
        out.append(
            {
                "reader": name,
                "in_plane": frac,
                "direction_deg": float(np.degrees(np.arctan2(p[1], p[0]))),
            }
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=2, frameon=False)
    return out


def profile_panel(ax, curves, title) -> None:
    for lab, y, color, ls in curves:
        ax.plot(range(20), y, ls, color=color, marker="o", ms=3, label=lab)
    for e in (5, 10, 15):
        ax.axvline(e - 0.5, color="0.85", lw=0.8, zorder=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xticks(range(0, 20, 5))
    ax.set_xlabel("s = a mod 20")
    ax.set_title(title, fontsize=8.5)
    ax.legend(fontsize=6.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    nums: dict = {}

    # neurons, original model
    X = np.load(OOUT / "stream.npz")["mid_12"].astype(np.float64)
    blk = load_block(12)
    g, u = blk.readers(X)
    P20 = turned(plane(X), X, S20, 0, 10)
    P10 = turned(basis(X)[0][0], X, R10, 0, None)
    rho = {
        f"{k} {n}": blk.g2 * W[n] for n in NEURONS for k, W in (("up", blk.Wu), ("gate", blk.Wg))
    }
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.0))
    nums["neurons_P20"] = draw_plane(axes[0], X, P20, S20, 20, scol, [
        ("up 5212", rho["up 5212"], COLORS["up"]), ("gate 5212", rho["gate 5212"], COLORS["gate"]),
        ("gate 8953", rho["gate 8953"], COLORS["gate2"]), ("up 8953", rho["up 8953"], COLORS["gate3"])],
        "(a) a mod 20 plane P20 at block 12's MLP input,\nread directions of neurons 5212 and 8953")  # fmt: skip
    nums["neurons_P10"] = draw_plane(axes[1], X, P10, R10, 10, rcol, [
        ("gate 5212", rho["gate 5212"], COLORS["gate"]), ("up 5212", rho["up 5212"], COLORS["up"]),
        ("gate 8953", rho["gate 8953"], COLORS["gate2"])],
        "(b) a mod 10 frequency-1 plane P10, same input\n(blue r = 0–4, orange r = 5–9)")  # fmt: skip
    prof = {k: means(v[:, [n]], S20, 20)[:, 0] for n in NEURONS for k, v in
            ((f"up{n}", u), (f"gate{n}", silu(g)), (f"w{n}", silu(g) * u))}  # fmt: skip
    profile_panel(axes[2], [("up 5212", prof["up5212"], COLORS["up"], "-"),
                            ("silu(gate 5212)", prof["gate5212"], COLORS["gate"], "-"),
                            ("write coefficient silu(gate)·up", prof["w5212"], "k", "--")],
                  "(c) neuron 5212: up reads h, gate opens for r = 5–9")  # fmt: skip
    profile_panel(axes[3], [("up 8953", prof["up8953"], COLORS["gate3"], "-"),
                            ("silu(gate 8953)", prof["gate8953"], COLORS["gate2"], "-"),
                            ("write coefficient silu(gate)·up", prof["w8953"], "k", "--")],
                  "(d) neuron 8953: gate fires for s = 10–14, up constant")  # fmt: skip
    fig.tight_layout()
    fig.savefig(args.out / "mod20_b12_neurons.png")
    plt.close(fig)
    nums["neurons_profiles"] = {k: v.tolist() for k, v in prof.items()}
    del blk

    # components, alive-only model
    X = np.load(AOUT / "stream.npz")["mid_12"].astype(np.float64)
    blk = load_blocks()[0][12]
    tg, tu = blk.readers(X)
    gid, uid = list(blk.index[blk.kind == "gate"]), list(blk.index[blk.kind == "up"])
    P20 = turned(plane(X), X, S20, 0, 10)
    P10 = turned(basis(X)[0][0], X, R10, 0, None)

    def rv(kind: str, c: int) -> np.ndarray:
        V = blk.m["V_gate"][:, gid.index(c)] if kind == "g" else blk.m["V_up"][:, uid.index(c)]
        return blk.g2 * V

    comps = [("u57", rv("u", 57), COLORS["up"]), ("g37", rv("g", 37), COLORS["gate"]), ("g33", rv("g", 33), "#6c9bd1"),
             ("g11", rv("g", 11), COLORS["gate2"]), ("u1", rv("u", 1), COLORS["gate3"])]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.0))
    nums["components_P20"] = draw_plane(axes[0], X, P20, S20, 20, scol, comps,
        "(a) a mod 20 plane P20 at block 12's MLP input,\nread directions of the components")  # fmt: skip
    nums["components_P10"] = draw_plane(axes[1], X, P10, R10, 10, rcol, comps[:4],
        "(b) a mod 10 frequency-1 plane P10, same input\n(blue r = 0–4, orange r = 5–9)")  # fmt: skip
    Ug, Uu = blk.m["U_gate"], blk.m["U_up"]
    hg, hu = tg @ Ug, tu @ Uu
    cprof = {}
    for n in NEURONS:
        cprof[f"gate{n}"] = means(silu(hg[:, [n]]), S20, 20)[:, 0]
        cprof[f"up{n}"] = means(hu[:, [n]], S20, 20)[:, 0]
        cprof[f"w{n}"] = means(silu(hg[:, [n]]) * hu[:, [n]], S20, 20)[:, 0]
    for c in GATES_5212:
        cprof[f"g{c}->5212"] = means(
            (tg[:, gid.index(c)] * Ug[gid.index(c), 5212])[:, None], S20, 20
        )[:, 0]
    cprof["u57->5212"] = means((tu[:, uid.index(57)] * Uu[uid.index(57), 5212])[:, None], S20, 20)[
        :, 0
    ]
    cprof["g11->8953"] = means((tg[:, gid.index(11)] * Ug[gid.index(11), 8953])[:, None], S20, 20)[
        :, 0
    ]
    profile_panel(axes[2], [("u57 → up 5212", cprof["u57->5212"], COLORS["up"], "-"),
                            ("g37 → gate 5212", cprof["g37->5212"], COLORS["gate"], "-"),
                            ("g33 → gate 5212", cprof["g33->5212"], "#6c9bd1", "-"),
                            ("write coefficient of neuron 5212", cprof["w5212"], "k", "--")],
                  "(c) neuron 5212 in the components model")  # fmt: skip
    profile_panel(axes[3], [("g11 → gate 8953", cprof["g11->8953"], COLORS["gate2"], "-"),
                            ("up 8953 (mostly u1)", cprof["up8953"], COLORS["gate3"], "-"),
                            ("write coefficient of neuron 8953", cprof["w8953"], "k", "--")],
                  "(d) neuron 8953 in the components model")  # fmt: skip
    fig.tight_layout()
    fig.savefig(args.out / "mod20_b12_components.png")
    plt.close(fig)
    nums["components_profiles"] = {k: v.tolist() for k, v in cprof.items()}
    nums["U_up_57_5212"] = float(Uu[uid.index(57), 5212])
    nums["U_up_57_next"] = float(np.sort(np.abs(Uu[uid.index(57)]))[-2])
    nums["U_gate_11_8953"] = float(Ug[gid.index(11), 8953])
    nums["U_gate_11_next"] = float(np.sort(np.abs(Ug[gid.index(11)]))[-2])
    (OUT / "b12_fig.json").write_text(json.dumps(nums, indent=1))
    for k in ("neurons_P20", "neurons_P10", "components_P20", "components_P10"):
        print(
            k, [(d["reader"], round(d["in_plane"], 3), round(d["direction_deg"])) for d in nums[k]]
        )


if __name__ == "__main__":
    main()
