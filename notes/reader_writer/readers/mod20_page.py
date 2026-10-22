"""Figures, numbers and notebook entry (id "mod20") of the page "Mod 20 representation at a (position 1)".

Reads track_mod20/{alive,orig}/{analysis.json,readers_scan.npz} (mod20_track.py, mod20_readers.py) and the two float64
streams (track_build.py: alive-only model; track_build_orig.py: original Llama-3.1-8B). Writes notebook/fig/mod20_*.png,
track_mod20/page_numbers.json and the entry in <out>/entries.js. Nothing is centred: plot origins are the activation
zero. Notation as in mod20_track.py; in addition:
  o_s        = (m_s - m_{s+10}) / 2 in R^2, s = 0..9, and o_{s+10} = -o_s: the odd part of the residue means in the plane
  phi_s      angle of o_s (unwrapped along s = 0..19); step_s = phi_{s+1} - phi_s in degrees (s + 1 mod 20), 18 for an
             evenly spaced ring
  alpha      the alpha in [0, 1] minimising sum_s wrap(phi_s - c - 2 pi / 20 (alpha s + (1 - alpha) 5 floor(s / 5)))^2 over
             alpha and an offset angle c: 1 = evenly spaced ring, 0 = four points (one per quarter)

    python mod20_page.py --out notebook [--entries <entries.js> ...]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod20_track import OUT, S20, means, plane
from track_build import OUT as AOUT
from track_build_orig import OUT as OOUT

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
MODELS = {"alive": "components (alive-only model)", "orig": "neurons (original model)"}
STREAM = {"alive": AOUT, "orig": OOUT}
NAMES = [f"{t}_{b}" for b in range(17) for t in ("pre", "mid")] + ["pre_17"]
PRE = [f"pre_{b}" for b in range(18)]
QCM = [plt.get_cmap(c) for c in ("Blues", "Oranges", "Greens", "Purples")]


def scol(s: int):
    return QCM[s // 5](0.45 + 0.12 * (s % 5))


def ring(X: np.ndarray) -> dict:
    P = plane(X)
    m = means(X @ P, S20, 20)
    o = (m[:10] - m[10:]) / 2
    oc = np.concatenate([o[:, 0] + 1j * o[:, 1], -(o[:, 0] + 1j * o[:, 1])])
    th = np.unwrap(np.angle(oc * np.conj(oc[0])))
    th = th * np.sign(th[-1] - th[0])
    s = np.arange(20)
    best = (0.0, np.inf)
    for al in np.linspace(0, 1, 101):
        pred = 2 * np.pi / 20 * (al * s + (1 - al) * 5 * (s // 5))
        c = np.angle(np.exp(1j * (th - pred)).mean())
        e = (np.angle(np.exp(1j * (th - pred - c))) ** 2).sum()
        if e < best[1]:
            best = (float(al), float(e))
    steps = np.degrees(np.diff(np.concatenate([th, [th[0] + 2 * np.pi]])))
    return {
        "alpha": best[0],
        "alpha_rms_deg": float(np.degrees(np.sqrt(best[1] / 20))),
        "steps": steps.tolist(),
    }


def fig_planes(st: dict, path: Path) -> None:
    show = (0, 2, 5, 9, 12, 13, 15)
    fig, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.6))
    for row, mdl in enumerate(MODELS):
        for ax, b in zip(axes[row], show, strict=True):
            X = st[mdl][f"pre_{b}"]
            P = plane(X)
            z = X @ P
            c0 = (means(z, S20, 20)[:10] - means(z, S20, 20)[10:])[0]
            ang = np.arctan2(c0[1], c0[0])
            rot = np.asarray([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
            z = z @ rot.T
            m = means(z, S20, 20)
            for s in range(20):
                ax.scatter(*z[s == S20].T, s=4, color=scol(s), alpha=0.6, lw=0)
            for s in range(20):
                ax.text(*m[s], str(s), fontsize=6.5, ha="center", va="center", weight="bold",
                        bbox={"boxstyle": "circle,pad=0.12", "fc": "white", "ec": scol(s), "lw": 0.8})  # fmt: skip
            ax.plot([0], [0], "k+", ms=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                ("embedding" if b == 0 else f"entering block {b}")
                + ("" if row else "\ncomponents"),
                fontsize=8,
            )
            if b == 0:
                ax.set_ylabel(MODELS[mdl], fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fourier(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, mdl in zip(axes, MODELS, strict=True):
        M = np.asarray([np.asarray(res[mdl]["boundaries"][n]["amp"]) / np.asarray(res[mdl]["boundaries"][n]["null99"])
                        for n in PRE])  # fmt: skip
        im = ax.imshow(M.T, aspect="auto", cmap="viridis", vmin=0, vmax=3, origin="lower")
        ax.contour(M.T, levels=[1.0], colors="w", linewidths=0.8)
        ax.set_yticks(
            range(10), [f"j={j} (τ={20 // np.gcd(j, 20)})" for j in range(1, 11)], fontsize=7
        )
        ax.set_xticks(range(18), [str(b) for b in range(18)], fontsize=7)
        ax.set_xlabel("residual stream entering block (17 = leaving block 16)")
        ax.set_title(f"{MODELS[mdl]}: ‖Ψ_j‖ / null_j (white: 1)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ring(rings: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for ax, mdl in zip(axes[:2], MODELS, strict=True):
        M = np.asarray([rings[mdl][n]["steps"] for n in PRE])
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=60, origin="upper")
        for b in range(18):
            for s in range(20):
                ax.text(s, b, f"{M[b, s]:.0f}", fontsize=4.8, ha="center", va="center",
                        color="w" if M[b, s] < 35 else "k")  # fmt: skip
        ax.set_xticks(
            range(20), [f"{s}→{(s + 1) % 20}" for s in range(20)], rotation=90, fontsize=6
        )
        ax.set_yticks(range(18), ["emb"] + [str(b) for b in range(1, 18)], fontsize=7)
        ax.set_ylabel("entering block")
        ax.set_title(f"{MODELS[mdl]}: angular step along the ring (°)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[2]
    for mdl, ls in zip(MODELS, ("-", "--"), strict=True):
        ax.plot(
            range(18),
            [rings[mdl][n]["alpha"] for n in PRE],
            ls,
            color="k",
            marker="o",
            ms=3,
            label=MODELS[mdl],
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("entering block")
    ax.set_ylabel("α (1: even ring, 0: four quarters)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_clouds(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    x = np.arange(18)
    for mdl, ls in zip(MODELS, ("-", "--"), strict=True):
        B = [res[mdl]["boundaries"][n] for n in PRE]
        lab = "" if mdl == "alive" else " (neurons)"
        axes[0].plot(x, [b["eta"]["s"] for b in B], ls, color="C0", label="η_s" + lab)
        axes[0].plot(x, [b["eta"]["Q"] for b in B], ls, color="C1", label="η_Q" + lab)
        axes[1].plot(
            x,
            [np.median(b["sep_antipodal"]) for b in B],
            ls,
            color="C2",
            label="median sep(s, s+10)" + lab,
        )
        axes[1].plot(
            x, [b["sep_quarters_min"] for b in B], ls, color="C3", label="closest quarters" + lab
        )
        axes[2].plot(x, [b["acc"]["h_given_r"] for b in B], ls, color="C4", label="h given r" + lab)
        axes[2].plot(x, [b["acc"]["Q"] for b in B], ls, color="C5", label="Q" + lab)
        axes[2].plot(x, [b["acc"]["s"] for b in B], ls, color="C6", label="s" + lab)
        axes[3].plot(
            x, [b["probe_h"]["stream"] for b in B], ls, color="C7", label="whole stream" + lab
        )
        axes[3].plot(
            x, [b["probe_h"]["plane"] for b in B], ls, color="C8", label="plane only" + lab
        )
        axes[3].plot(
            x,
            [b["probe_h"]["without_plane"] for b in B],
            ls,
            color="C9",
            label="stream minus plane" + lab,
        )
    titles = ("(a) share of in-plane energy explained", "(b) separation in the plane", "(c) nearest-mean accuracy (held out)",
              "(d) held-out probe accuracy for h")  # fmt: skip
    for ax, t in zip(axes, titles, strict=True):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("entering block")
        ax.legend(fontsize=6)
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[3].set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sublayers(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    x = np.arange(17)
    for mdl, ls in zip(MODELS, ("-", "--"), strict=True):
        S = [res[mdl]["sublayers"][f"mlp_{b}"] for b in x]
        At = [res[mdl]["sublayers"][f"attn_{b}"] for b in x]
        lab = "" if mdl == "alive" else " (neurons)"
        axes[0].plot(x, [s["gain"] for s in S], ls, color="C0", label="MLP gain" + lab)
        axes[0].plot(x, [s["gain"] for s in At], ls, color="C7", label="attention gain" + lab)
        axes[1].plot(x, [s["share_from_P"] for s in S], ls, color="C1", label="share from P" + lab)
        axes[1].plot(x, [s["without_P"] for s in S], ls, color="C2", label="without P" + lab)
        axes[2].plot(x, [s["r2_lin"] for s in S], ls, color="C3", label="r² linear in z" + lab)
        axes[2].plot(
            x, [s["r2_gated"] for s in S], ls, color="C4", label="r² linear per a mod 10" + lab
        )
        axes[3].plot(
            x, [s["mechanism"]["r2_secant_sum"] for s in S], ls, color="C5", label="secant" + lab
        )
        axes[3].plot(
            x,
            [s["mechanism"]["r2_gated_secant_sum"] for s in S],
            ls,
            color="C6",
            label="gated secant" + lab,
        )
        axes[3].plot(
            x,
            [s["mechanism"]["r2_additivity"] for s in S],
            ls,
            color="C8",
            label="additivity" + lab,
        )
    titles = ("(a) period-20 output / input", "(b) where the MLP's period-20 output comes from",
              "(c) Δ fitted from the plane (held out)", "(d) reader decomposition of Δ")  # fmt: skip
    for ax, t in zip(axes, titles, strict=True):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("block")
        ax.legend(fontsize=6)
    axes[2].set_ylim(0, 1)
    axes[3].set_ylim(-0.3, 1.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_readers(scan: dict, path: Path, examples: list) -> dict:
    counts = {}
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 4)
    for col, mdl in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, 2 * col : 2 * col + 2])
        c = np.zeros((17, 3))
        for b in range(17):
            for kind in ("gate", "up"):
                z = scan[mdl]
                sel = (z[f"L{b}_{kind}_odd"] >= 0.05) & (z[f"L{b}_{kind}_step_r2"] >= 0.8)
                e = z[f"L{b}_{kind}_edge"][sel]
                c[b] += [(e == 0).sum(), (e == 5).sum(), ((e != 0) & (e != 5)).sum()]
        counts[mdl] = c.tolist()
        bot = np.zeros(17)
        for k, (lab, colr) in enumerate((("edge 0: h (tens parity)", "C0"), ("edge 5: g (rounded-tens parity)", "C3"),
                                          ("other edges", "0.6"))):  # fmt: skip
            ax.bar(range(17), c[:, k], bottom=bot, color=colr, label=lab)
            bot += c[:, k]
        ax.set_xticks(range(17))
        ax.set_xlabel("block")
        ax.set_ylabel("readers")
        ax.set_title(
            f"{MODELS[mdl]}: readers with odd share ≥ 0.05 and step r² ≥ 0.8, by best edge",
            fontsize=9,
        )
        ax.legend(fontsize=7)
    for k, (title, items) in enumerate(examples):
        ax = fig.add_subplot(gs[1, k])
        for lab, prof, ls in items:
            ax.plot(range(20), prof / np.abs(prof).max(), ls, marker="o", ms=3, label=lab)
        for e in (5, 10, 15):
            ax.axvline(e - 0.5, color="0.85", lw=0.8, zorder=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(range(0, 20, 5))
        ax.set_xlabel("s = a mod 20")
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return counts


def fig_gating(gate: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (title, items) in zip(axes, gate, strict=True):
        for lab, prof, ls in items:
            ax.plot(range(20), prof, ls, marker="o", ms=3, label=lab)
        for e in (5, 10, 15):
            ax.axvline(e - 0.5, color="0.85", lw=0.8, zorder=0)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_xticks(range(0, 20, 5))
        ax.set_xlabel("s = a mod 20")
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def profiles(st: dict) -> dict[str, np.ndarray]:
    """Class-mean profiles over s of the example readers (activation t_c), and of the per-prompt write slopes of the
    gated up readers: for a component up reader c, ||(silu(h_gate) * U_up[c]) V_down U_down|| (the norm of the MLP
    output's derivative in t_c); for a neuron i, silu(gate_i). Cached in track_mod20/profiles.npz."""
    path = OUT / "profiles.npz"
    if path.exists():
        return dict(np.load(path))
    from track_build import load_blocks, silu
    from track_build_orig import load_block

    out = {}
    blocks = load_blocks()[0]
    for b, items in ((3, [("up", 141)]), (12, [("gate", 11), ("up", 57)]), (13, [("gate", 58)]),
                     (14, [("gate", 73), ("up", 33)])):  # fmt: skip
        blk = blocks[b]
        tg, tu = blk.readers(st["alive"][f"mid_{b}"])
        sg = silu(tg @ blk.m["U_gate"])
        for kind, c in items:
            T = tg if kind == "gate" else tu
            j = list(blk.index[blk.kind == kind]).index(c)
            out[f"alive_L{b}_{kind}{c}"] = means(T[:, [j]], S20, 20)[:, 0]
            if kind == "up":
                eff = np.linalg.norm(
                    ((sg * blk.m["U_up"][j]) @ blk.m["V_down"]) @ blk.m["U_down"], axis=1
                )
                out[f"alive_L{b}_slope_up{c}"] = means(eff[:, None], S20, 20)[:, 0]
    for b, items in ((3, [("u", 13082)]), (12, [("g", 8953), ("u", 5212), ("g", 5212)]), (13, [("g", 10534)]),
                     (14, [("g", 3498), ("u", 6838), ("g", 6838)])):  # fmt: skip
        blk = load_block(b)
        g, u = blk.readers(st["orig"][f"mid_{b}"])
        for kind, i in items:
            T = g if kind == "g" else u
            out[f"orig_L{b}_{kind}{i}"] = means(T[:, [i]], S20, 20)[:, 0]
            if kind == "g":
                out[f"orig_L{b}_silu_g{i}"] = means(silu(T[:, [i]]), S20, 20)[:, 0]
        del blk
    np.savez(path, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--entries",
        type=Path,
        nargs="*",
        default=None,
        help="entries.js files to update (default <out>)",
    )
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    res = {m: json.loads((OUT / m / "analysis.json").read_text()) for m in MODELS}
    scan = {m: dict(np.load(OUT / m / "readers_scan.npz")) for m in MODELS}
    st = {
        m: {k: v.astype(np.float64) for k, v in np.load(STREAM[m] / "stream.npz").items()}
        for m in MODELS
    }
    rings = {m: {n: ring(st[m][n]) for n in PRE} for m in MODELS}
    pr = profiles(st)
    fig_planes(st, fig_dir / "mod20_planes.png")
    fig_fourier(res, fig_dir / "mod20_fourier.png")
    fig_ring(rings, fig_dir / "mod20_ring.png")
    fig_clouds(res, fig_dir / "mod20_clouds.png")
    fig_sublayers(res, fig_dir / "mod20_sublayers.png")
    examples = [
        ("block 3: reads h", [("component u141", pr["alive_L3_up141"], "-"), ("neuron up 13082", pr["orig_L3_u13082"], "--")]),
        ("block 12: detects s in 10..14", [("component g11", pr["alive_L12_gate11"], "-"),
                                          ("neuron gate 8953", pr["orig_L12_g8953"], "--")]),
        ("block 13: reads g", [("component g58", pr["alive_L13_gate58"], "-"), ("neuron gate 10534", pr["orig_L13_g10534"], "--")]),
        ("block 14: reads g", [("component g73", pr["alive_L14_gate73"], "-"), ("neuron gate 3498", pr["orig_L14_g3498"], "--")]),
    ]  # fmt: skip
    counts = fig_readers(scan, fig_dir / "mod20_readers.png", examples)
    gate = [
        ("block 12: write slope of the up reader", [("component u57 (slope norm)", pr["alive_L12_slope_up57"], "-"),
                                                    ("neuron 5212: silu(gate)", pr["orig_L12_silu_g5212"], "--")]),
        ("block 12: the up reader's activation", [("component u57", pr["alive_L12_up57"], "-"),
                                                  ("neuron up 5212", pr["orig_L12_u5212"], "--")]),
        ("block 14: write slope of the up reader", [("component u33 (slope norm)", pr["alive_L14_slope_up33"], "-"),
                                                    ("neuron 6838: silu(gate)", pr["orig_L14_silu_g6838"], "--")]),
        ("block 14: the up reader's activation", [("component u33", pr["alive_L14_up33"], "-"),
                                                  ("neuron up 6838", pr["orig_L14_u6838"], "--")]),
    ]  # fmt: skip
    fig_gating(gate, fig_dir / "mod20_gating.png")
    nums = {"rings": rings, "reader_counts": counts}
    (OUT / "page_numbers.json").write_text(json.dumps(nums, indent=1))
    html = page(res, rings, counts, scan, pr)
    for p in args.entries or [args.out / "entries.js"]:
        insert_entry(p, html)


def insert_entry(p: Path, html: str) -> None:
    s = p.read_text()
    entry = (
        '{\nid: "mod20",\ntitle: "Mod 20 representation at a (position 1)",\ndate: "2026-10-22",\nhtml: `'
        + html
        + "`\n},\n"
    )
    if 'id: "mod20"' in s:
        i = s.index('{\nid: "mod20"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)


def page(res, rings, counts, scan, pr) -> str:
    def bd(m: str, b: int) -> dict:
        return res[m]["boundaries"][f"pre_{b}"]

    def sl(m: str, b: int, k: str = "mlp") -> dict:
        return res[m]["sublayers"][f"{k}_{b}"]

    def f2(x: float) -> str:
        return f"{x:.2f}"

    def rng(vals) -> str:
        v = list(vals)
        return f"{min(v):.2f}–{max(v):.2f}"

    def j1null(m: str, b: int) -> float:
        x = bd(m, b)
        return x["amp"][0] / x["null99"][0]

    def oddmax(m: str, b: int) -> float:
        x = bd(m, b)
        return max(x["amp"][j] / x["null99"][j] for j in (2, 4, 6, 8))

    def alpha(m: str, b: int) -> float:
        return rings[m][f"pre_{b}"]["alpha"]

    def step(m: str, b: int, s_: int) -> float:
        return rings[m][f"pre_{b}"]["steps"][s_]

    def anti(m: str, b: int) -> float:
        return float(np.median(bd(m, b)["sep_antipodal"]))

    def mech(m: str, b: int, k: str) -> float:
        return sl(m, b)["mechanism"][k]

    att_gain = {m: max(sl(m, b, "attn")["gain"] for b in range(17)) for m in MODELS}
    cA, cO = np.asarray(counts["alive"]), np.asarray(counts["orig"])
    rows = []
    for b in range(17):
        cells = []
        for m in MODELS:
            x, y = sl(m, b), bd(m, b + 1)
            cells.append(
                f"<td>{x['gain']:.2f}</td><td>{x['share_from_P']:.2f}</td><td>{x['r2_lin']:.2f}</td><td>{x['r2_gated']:.2f}</td>"
                f"<td>{x['mechanism']['r2_secant_sum']:.2f}</td><td>{x['mechanism']['r2_gated_secant_sum']:.2f}</td>"
                f"<td>{x['mechanism']['r2_additivity']:.2f}</td><td>{alpha(m, b + 1):.2f}</td><td>{y['acc']['Q']:.2f}</td>"
                f"<td>{anti(m, b + 1):.2f}</td>"
            )
        rows.append(f"<tr><td>{b}</td>{cells[0]}<td class='sep'></td>{cells[1]}</tr>")
    head = "<th>gain</th><th>share from P</th><th>r²<sub>lin</sub></th><th>r²<sub>gated</sub></th><th>secant</th><th>gated secant</th><th>additivity</th><th>α after</th><th>acc<sub>Q</sub> after</th><th>sep(s, s+10) after</th>"
    table = (
        "<table class='tight'><tr><th rowspan=2>block</th><th colspan=10>components (alive-only model)</th><th class='sep'></th>"
        "<th colspan=10>neurons (original model)</th></tr><tr>"
        + head
        + "<th class='sep'></th>"
        + head
        + "</tr>"
        + "".join(rows)
        + "</table>"
    )
    return f"""
<p><b>Summary.</b> At the token <i>a</i>, the information that tells <i>a</i> mod 20 apart from <i>a</i> mod 10, that is,
whether the tens digit is even or odd, sits in a single plane of the residual stream: the frequency-1 Fourier plane of
<i>a</i> mod 20. No other frequency that separates <i>s</i> from <i>s</i> + 10 rises above a shuffled-label null at any block
from the embedding to block 16. In that plane the 20 residues lie on a ring around the activation zero, in order. At the
embedding the ring is nearly evenly spaced, with a larger gap where the tens digit changes (between 9 and 10, and
between 19 and 0). The MLP of block 12 turns it into four clumps of five residues, {{0–4}}, {{5–9}}, {{10–14}}, {{15–19}}: the
ring then carries two bits, the parity of the tens digit (<i>h</i>) and the parity of <i>a</i> rounded to the nearest ten
(<i>g</i>). The readers say the same thing. Up to block 12, the readers that respond to the plane read <i>h</i>; from
block 13 on, most read <i>g</i>. Block 12 holds the step between the two: its MLP multiplies a reading of the ring by a
gate that is open only when the units digit is 5–9, which is the product that converts <i>h</i> into <i>g</i>. In
blocks 0–11 every MLP moves the residue means within the plane by one 2 × 2 linear map, and in the components model it
does so through reader paths that are each linear in their own input. Blocks 12 and 14 are not linear in the plane; they
become close to linear once a separate linear map is allowed for each value of <i>a</i> mod 10. In the components model
the clouds blur in blocks 3–11 (their spread grows faster than the ring), and the MLPs of blocks 11–13 sharpen them
again. The same plane, the same ring-to-quarters change at block 12, the same switch of the readers from <i>h</i> to
<i>g</i> and the same gating are found in the original model's neuron basis, with matching reader profiles; the original
model blurs much less.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands <i>a</i>,
<i>b</i> ∈ {{1, …, 400}} and op ∈ {{+, −}}. Everything on this page is at position 1, the token <i>a</i>. Position 1 attends
only to &lt;BOS&gt; and to itself, so its residual stream depends on <i>a</i> alone, and 400 two-token forwards give
every value. Two models are compared, both rebuilt in float64 on the CPU through block 16:</p>
<ul>
<li><b>Components</b>: the alive-only model of section 8 of the "Mod 10 representations" page (every alive attention and MLP
component of the ceiling filter <code>addsub-08-last-pos</code> on, the weight delta off; <code>track_build.py</code>). A
<i>reader</i> is a kept gate or up component of a block's MLP; its activation is its read vector applied to the
MLP input.</li>
<li><b>Neurons</b>: the original Llama-3.1-8B, with no decomposition (<code>track_build_orig.py</code>). A reader is one row of
<code>gate_proj</code> ("gate neuron <i>i</i>") or of <code>up_proj</code> ("up neuron <i>i</i>"), <i>i</i> ∈ {{0, …, 14335}}.
Its residual stream has mean cosine 0.93–0.96 with the components model's at every block.</li>
</ul>
<p>Nothing is centred anywhere on this page: the network has no biases, so zero is the zero of the activation space, and
every plot's origin is that zero.</p>
<p><b>Variables of a.</b> <i>s</i> = <i>a</i> mod 20 (integer 0–19; 20 values of <i>a</i> per residue) and <i>r</i> = <i>a</i> mod 10
(integer 0–9). <i>h</i> = ⌊<i>a</i>/10⌋ mod 2 is the parity of the tens digit: 0 for <i>s</i> ∈ 0–9, 1 for <i>s</i> ∈ 10–19.
<i>g</i> = ⌊(<i>a</i> + 5)/10⌋ mod 2 is the parity of <i>a</i> rounded to the nearest ten (halves rounded up): 0 for <i>s</i> ∈ 0–4
and 15–19, 1 for <i>s</i> ∈ 5–14. <i>Q</i> = ⌊<i>s</i>/5⌋ (integer 0–3) is the quarter of the 20-cycle; the pair (<i>h</i>, <i>g</i>)
is (0, 0), (0, 1), (1, 1), (1, 0) for <i>Q</i> = 0, 1, 2, 3, so <i>Q</i> and (<i>h</i>, <i>g</i>) carry the same information. Note
<i>g</i> = <i>h</i> XOR [<i>r</i> ≥ 5], where [·] is 1 when the condition holds and 0 otherwise. More generally, for an integer
<i>e</i> ∈ 0–9, <i>B<sub>e</sub></i> = ⌊(<i>a</i> + 10 − <i>e</i>)/10⌋ mod 2 is the binary variable that flips at <i>s</i> = <i>e</i>
and <i>s</i> = <i>e</i> + 10 ("edge <i>e</i>"); <i>B</i><sub>0</sub> = <i>h</i> and <i>B</i><sub>5</sub> = <i>g</i>.</p>
<p><b>Stream and Fourier plane.</b> <i>X</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream at one boundary: entering
block <i>ℓ</i> (<i>ℓ</i> = 0 is the embedding; "entering block 17" is the stream leaving block 16) or after block <i>ℓ</i>'s
attention. <i>μ<sub>s</sub></i> ∈ ℝ<sup>4096</sup> is the mean of <i>X</i> over the 20 <i>a</i> with <i>a</i> mod 20 = <i>s</i>. For an integer
frequency <i>j</i> ∈ 1–10, Ψ<sub><i>j</i></sub> = Σ<sub><i>s</i>=0…19</sub> <i>μ<sub>s</sub></i> exp(−2π i <i>j s</i>/20), with i the
imaginary unit; the exact period of frequency <i>j</i> is <i>τ</i> = 20/gcd(<i>j</i>, 20). The odd frequencies <i>j</i> = 1, 3, 5, 7, 9 are
exactly the ones in which residues <i>s</i> and <i>s</i> + 10 differ, so they hold everything <i>a</i> mod 20 adds to
<i>a</i> mod 10 (the even frequencies are the <i>a</i> mod 10 representation of the mod-10 page). null<sub><i>j</i></sub> is the 99th
percentile of ‖Ψ<sub><i>j</i></sub>‖ over 100 random relabellings of the 400 values of <i>a</i>. <i>P</i> (a 4096 × 2 matrix with
orthonormal columns) spans Re Ψ<sub>1</sub> and Im Ψ<sub>1</sub>: the frequency-1 plane of <i>a</i> mod 20.
<i>z</i>(<i>a</i>) = <i>P</i><sup>T</sup><i>X</i>(<i>a</i>) ∈ ℝ<sup>2</sup> and <i>m<sub>s</sub></i> = <i>P</i><sup>T</sup><i>μ<sub>s</sub></i> are
a prompt's and a residue's coordinates in it, and Π<sub><i>P</i></sub> = <i>PP</i><sup>T</sup> is the projection onto it.</p>

<h3>1. One plane carries what a mod 20 adds to a mod 10</h3>
<p>Figure 1 compares the strength of each frequency with what shuffled labels give. Of the five odd frequencies, only
<i>j</i> = 1 is above its null: ‖Ψ<sub>1</sub>‖/null<sub>1</sub> is {rng(j1null("alive", b) for b in range(18))} (components) and
{rng(j1null("orig", b) for b in range(18))} (neurons) over the 18 boundaries, while <i>j</i> = 3, 5, 7, 9 stay at
{rng(oddmax("alive", b) for b in range(1, 16))} (components) and {rng(oddmax("orig", b) for b in range(1, 16))} (neurons) of
their nulls from block 1 to block 15. At the embedding they reach {f2(oddmax("alive", 0))}, and leaving block 16
{f2(oddmax("alive", 17))} (components) and {f2(oddmax("orig", 17))} (neurons). The period-20 information is therefore
in one plane, <i>P</i>.</p>
<p>A held-out linear probe checks that nothing linear is missed (figure 7d). An L2 logistic regression (inverse
regularisation 1) is trained to predict <i>h</i> on the prompts with <i>a</i> ≤ 200 and tested on <i>a</i> &gt; 200, and the
reverse; the plane used is fitted on the training half only. From the two coordinates <i>z</i> alone it reaches
{rng(bd("alive", b)["probe_h"]["plane"] for b in range(18))} accuracy (components) and
{rng(bd("orig", b)["probe_h"]["plane"] for b in range(18))} (neurons). From the whole 4096-dimensional stream it reaches
{rng(bd("alive", b)["probe_h"]["stream"] for b in range(18))} and {rng(bd("orig", b)["probe_h"]["stream"] for b in range(18))}.
From the stream with the plane projected out, <i>X</i> − Π<sub><i>P</i></sub><i>X</i>, it reaches only
{rng(bd("alive", b)["probe_h"]["without_plane"] for b in range(18))} and
{rng(bd("orig", b)["probe_h"]["without_plane"] for b in range(18))} (chance is 0.5). In the components model's blocks 5–12 the
whole stream beats the plane alone (for example {f2(bd("alive", 11)["probe_h"]["stream"])} against
{f2(bd("alive", 11)["probe_h"]["plane"])} entering block 11), while the stream without the plane stays at
{f2(bd("alive", 11)["probe_h"]["without_plane"])}: other directions do not carry <i>h</i> themselves but help the probe
remove the spread of the clouds in the plane (section 5).</p>
<figure><img src="notebook/fig/mod20_fourier.png"><figcaption>Figure 1. ‖Ψ<sub><i>j</i></sub>‖/null<sub><i>j</i></sub> for
every frequency <i>j</i> of <i>a</i> mod 20 (rows; <i>τ</i> its exact period) at the residual stream entering each block
(columns). White contour: 1, the 99th percentile of shuffled labels. Odd <i>j</i> (τ = 20 or 4) separate <i>s</i> from
<i>s</i> + 10; only <i>j</i> = 1 is above 1. Even <i>j</i> are the <i>a</i> mod 10, mod 5 and mod 2 representations.</figcaption></figure>

<h3>2. Geometry: a ring of the 20 residues, which turns into four clumps</h3>
<p>Figure 2 shows the 400 prompts in the plane at several blocks. The residues sit on a ring around the activation zero,
in the order 0, 1, …, 19, so that <i>s</i> and <i>s</i> + 10 are on opposite sides. Roundness (minor over major axis of the
ellipse traced by Re Ψ<sub>1</sub>, Im Ψ<sub>1</sub>) is {f2(bd("alive", 0)["roundness"])} at the embedding.</p>
<figure><img src="notebook/fig/mod20_planes.png"><figcaption>Figure 2. Residual stream at position 1 projected onto the
frequency-1 plane <i>P</i> of its own boundary (top: components; bottom: neurons). Each dot is one <i>a</i> (400), coloured by
<i>s</i> = <i>a</i> mod 20 in four colour families, one per quarter <i>Q</i> (blue 0–4, orange 5–9, green 10–14, purple 15–19);
labels are the residue means <i>m<sub>s</sub></i>. The axes are turned so that the vector from residue 10's mean to residue 0's
mean points right. The cross is the activation zero.</figcaption></figure>
<p><b>How evenly the residues are spaced.</b> Let <i>o<sub>s</sub></i> = (<i>m<sub>s</sub></i> − <i>m</i><sub><i>s</i>+10</sub>)/2 ∈ ℝ<sup>2</sup>
for <i>s</i> = 0–9 and <i>o</i><sub><i>s</i>+10</sub> = −<i>o<sub>s</sub></i>: half the vector from a residue's antipode to the residue,
which removes whatever <i>s</i> and <i>s</i> + 10 share (it is the frequency-1 part, not a subtraction of the overall mean).
<i>φ<sub>s</sub></i> is its angle, unwrapped along <i>s</i>, and the step from <i>s</i> to <i>s</i> + 1 is
<i>φ</i><sub><i>s</i>+1</sub> − <i>φ<sub>s</sub></i> in degrees. An evenly spaced ring has every step at 18°. The quantization index
<i>α</i> ∈ [0, 1] is the value minimising the squared angular error of <i>φ<sub>s</sub></i> ≈ <i>c</i> + (2π/20)(<i>α s</i> + (1 − <i>α</i>)·5⌊<i>s</i>/5⌋)
over <i>α</i> and an offset angle <i>c</i>: <i>α</i> = 1 is the even ring, <i>α</i> = 0 is four points, one per quarter.</p>
<p>At the embedding, <i>α</i> = {f2(alpha("alive", 0))}: steps are 14–17° inside each run of ten and
{step("alive", 0, 9):.0f}° from 9 to 10 (and from 19 to 0), with {step("alive", 0, 4):.0f}° from 4 to 5. The embedding already
marks the change of tens digit. Through blocks 1–11 the components model keeps <i>α</i> at
{rng(alpha("alive", b) for b in range(1, 12))} (neurons {rng(alpha("orig", b) for b in range(1, 13))} through block 12) and widens the
9 → 10 step to {step("alive", 11, 9):.0f}° (neurons {step("orig", 11, 9):.0f}°). The MLP of block 12 drops <i>α</i> from
{f2(alpha("alive", 12))} to {f2(alpha("alive", 13))} (neurons {f2(alpha("orig", 12))} → {f2(alpha("orig", 13))}), and it stays
there through block 13: entering block 14, <i>α</i> = {f2(alpha("alive", 14))} (neurons {f2(alpha("orig", 14))}), the 4 → 5 step is
{step("alive", 14, 4):.0f}° (neurons {step("orig", 14, 4):.0f}°), the 9 → 10 step {step("alive", 14, 9):.0f}°
({step("orig", 14, 9):.0f}°), and the steps inside a quarter shrink to 3–17°. Blocks 15–16 partly undo it (<i>α</i>
{f2(alpha("alive", 17))} and {f2(alpha("orig", 17))} leaving block 16). Figure 3 shows every step at every block.</p>
<figure><img src="notebook/fig/mod20_ring.png"><figcaption>Figure 3. Left and centre: the angular step <i>φ</i><sub><i>s</i>+1</sub> −
<i>φ<sub>s</sub></i> (degrees) from residue <i>s</i> to <i>s</i> + 1 (columns) at the stream entering each block (rows; 17 =
leaving block 16); 18° everywhere would be an evenly spaced ring. The quarter edges are the columns 4→5, 9→10, 14→15, 19→0.
Right: the quantization index <i>α</i>.</figcaption></figure>

<h3>3. What variable is stored: h early, g late</h3>
<p>The plane holds a two-dimensional code, so which binary variable a reader extracts depends on the direction it reads.
The readers therefore show what the network uses. For each reader <i>c</i> (component, or neuron) of each block,
<i>t<sub>c</sub></i>(<i>a</i>) is its activation at the MLP input, <i>m<sub>c</sub></i>(<i>s</i>) its mean over the 20 <i>a</i> with
<i>a</i> mod 20 = <i>s</i>, and <i>o<sub>c</sub></i>(<i>s</i>) = (<i>m<sub>c</sub></i>(<i>s</i>) − <i>m<sub>c</sub></i>(<i>s</i> + 10))/2 for <i>s</i> = 0–9 its
odd part. The <b>odd share</b> is 20 Σ<sub><i>s</i></sub> <i>o<sub>c</sub></i>(<i>s</i>)² / Σ<sub><i>a</i></sub> <i>t<sub>c</sub></i>(<i>a</i>)² (uncentred). The odd
part of a ±1-coded <i>B<sub>e</sub></i> is the step <i>σ<sub>e</sub></i>(<i>s</i>) = −1 for <i>s</i> &lt; <i>e</i>, +1 for <i>s</i> ≥ <i>e</i>;
the <b>step r²<sub><i>e</i></sub></b> is the uncentred r² of <i>o<sub>c</sub></i> ≈ β<i>σ<sub>e</sub></i> with one coefficient β, and the best
edge is the <i>e</i> with the largest one (<code>mod20_readers.py</code>).</p>
<p>Figure 4 (top) counts the readers with odd share ≥ 0.05 and best step r² ≥ 0.8, by best edge. Only edges 0 (<i>h</i>) and
5 (<i>g</i>) are common. Components: readers of <i>h</i> appear in blocks 3, 4 and 12 ({int(cA[12, 0])} in block 12), readers
of <i>g</i> from block 13 on ({int(cA[13, 1])}, {int(cA[14, 1])}, {int(cA[15, 1])}, {int(cA[16, 1])} in blocks 13–16), and
only {int(cA[:, 2].sum())} readers in all have another edge. Neurons: {int(cO[:13, 0].sum())} read <i>h</i> in blocks 0–12
against {int(cO[:13, 1].sum())} for <i>g</i>; in blocks 13–16 the counts are {int(cO[13:, 0].sum())} for <i>h</i> and
{int(cO[13:, 1].sum())} for <i>g</i>, {int(cO[14, 1])} and {int(cO[15, 1])} of them in blocks 14 and 15. The neuron basis has more
readers at every block (it is not sparse), but the same switch from <i>h</i> to <i>g</i> at block 13.</p>
<p>The bottom row of figure 4 pairs the largest component of a block with a neuron of the same block, chosen as the
largest contributor to that block's MLP write from the plane (section 4). Their profiles over <i>s</i> match. Block 3:
component u141 and up neuron 13082 are high for <i>s</i> ∈ 0–8 and low for 10–19 (<i>h</i>, with a smooth edge). Block 12:
component g11 and gate neuron 8953 fire for <i>s</i> ∈ 10–14 only, the quarter <i>Q</i> = 2 (<i>h</i> = 1 and <i>r</i> &lt; 5). Blocks 13
and 14: component g58 and gate neuron 10534, component g73 and gate neuron 3498, are steps at <i>s</i> = 5 and 15:
<i>g</i>.</p>
<figure><img src="notebook/fig/mod20_readers.png"><figcaption>Figure 4. Top: number of readers per block with odd share ≥
0.05 and best step r² ≥ 0.8, coloured by their best edge (left: kept gate and up components; right: all 28,672 gate and up
neurons). Bottom: mean activation over <i>s</i> of matched readers, each divided by its largest absolute value (solid:
component, dashed: neuron). Grey lines mark the quarter edges.</figcaption></figure>

<h3>4. How the blocks move the representation</h3>
<p><b>Per-sublayer quantities</b> (<code>mod20_track.py</code>). A sublayer <i>S</i> is one block's attention or MLP, including its
RMSNorm; it adds <i>S</i>(<i>X</i>) to the stream. Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>)(<i>a</i>)
is what <i>S</i> writes because its input contains the plane. The <b>gain</b> is ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖/‖Ψ<sub>1</sub>(<i>X</i>)‖,
with Ψ<sub>1</sub>(·) the frequency-1 Fourier vector of a field over <i>a</i>. <b>Share from <i>P</i></b> is
Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(Δ)⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖², the part of the period-20 output made from the plane;
<b>without <i>P</i></b> is ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>))‖/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖, the period-20 output the
sublayer still makes from other features of <i>a</i>. For r²(<i>Y</i>, <i>Ŷ</i>) = 1 − Σ<sub><i>a</i></sub>‖<i>Y</i> − <i>Ŷ</i>‖²/Σ<sub><i>a</i></sub>‖<i>Y</i>‖²
(uncentred): <b>r²<sub>lin</sub></b> is the held-out r² of Δ ≈ <i>Lz</i> with one 4096 × 2 matrix <i>L</i> (fitted on <i>a</i> ≤ 200,
tested on <i>a</i> &gt; 200, and the reverse); <b>r²<sub>gated</sub></b> is the same with one matrix <i>L<sub>r</sub></i> for each <i>r</i> = <i>a</i> mod 10.
For the MLP readers, <i>dt<sub>c</sub></i> = <i>t<sub>c</sub></i>(<i>X</i>) − <i>t<sub>c</sub></i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>) is the change of reader <i>c</i>'s
activation when the plane is removed, Δ<sub><i>c</i></sub> the change of the MLP output when only that reader changes, and
<i>κ<sub>c</sub></i> ∈ ℝ<sup>4096</sup> the least-squares slope of Δ<sub><i>c</i></sub> on <i>dt<sub>c</sub></i>; <i>κ<sub>c,r</sub></i> is the same slope within each
<i>r</i>. <b>Secant</b> is the r² of Δ ≈ Σ<sub><i>c</i></sub> <i>κ<sub>c</sub> dt<sub>c</sub></i> (every reader path linear in its own input),
<b>gated secant</b> the r² of Δ ≈ Σ<sub><i>c</i></sub> <i>κ</i><sub><i>c</i>, <i>a</i> mod 10</sub> <i>dt<sub>c</sub></i> (the slope may depend on the
units digit), and <b>additivity</b> the r² of Δ ≈ Σ<sub><i>c</i></sub> Δ<sub><i>c</i></sub> (no interaction between readers).</p>
<p><b>Attention does not act on it.</b> The attention gain is at most {f2(att_gain["alive"])} (components) and
{f2(att_gain["orig"])} (neurons) in every block, as for <i>a</i> mod 10.</p>
<p><b>Block 0's MLP enlarges the ring and turns the plane.</b> Its gain is {f2(sl("alive", 0)["gain"])} ({f2(sl("orig", 0)["gain"])}),
and the plane turns: the mean principal cosine between the plane entering block 1 and the embedding's is
{f2(float(np.mean(bd("alive", 1)["cos_emb"])))} ({f2(float(np.mean(bd("orig", 1)["cos_emb"])))}). Only
{f2(sl("alive", 0)["share_from_P"])} ({f2(sl("orig", 0)["share_from_P"])}) of its period-20 output comes from the plane;
the rest is made from other features of the token. The residue means still move by a near-linear map of the plane: the
odd parts <i>o<sub>s</sub></i> after the block are fitted from those before with one 2 × 2 matrix at r² {f2(sl("alive", 0)["r2_map"])}
({f2(sl("orig", 0)["r2_map"])}).</p>
<p><b>Blocks 1–11: small rewrites, linear within the plane.</b> MLP gains are {rng(sl("alive", b)["gain"] for b in range(1, 12))}
({rng(sl("orig", b)["gain"] for b in range(1, 12))}). The residue means' odd parts after each MLP are fitted from those
before by one 2 × 2 matrix at r² ≥ {min(sl(m, b)["r2_map"] for m in MODELS for b in range(1, 12)):.3f}. Per prompt, Δ is fitted
from <i>z</i> by one linear map at held-out r²<sub>lin</sub> {rng(sl("alive", b)["r2_lin"] for b in range(1, 12))}
({rng(sl("orig", b)["r2_lin"] for b in range(1, 12))}), and a map per units digit does not do better
(r²<sub>gated</sub> − r²<sub>lin</sub> ≤ {max(sl(m, b)["r2_gated"] - sl(m, b)["r2_lin"] for m in MODELS for b in range(1, 12)):.2f}). In the components model the reader paths are linear in their own input
(secant {rng(mech("alive", b, "r2_secant_sum") for b in range(1, 12))}, additivity
{rng(mech("alive", b, "r2_additivity") for b in range(1, 12))}), and allowing a slope per units digit adds at most
{max(mech("alive", b, "r2_gated_secant_sum") - mech("alive", b, "r2_secant_sum") for b in range(1, 12)):.2f}. In the neuron basis
the secant r² is lower ({rng(mech("orig", b, "r2_secant_sum") for b in range(1, 12))}) because each neuron's SiLU is curved
over the range the plane moves it, and the per-digit slope again adds little (at most
{max(mech("orig", b, "r2_gated_secant_sum") - mech("orig", b, "r2_secant_sum") for b in range(1, 12)):.2f}). The plane drifts
slowly: by the stream entering block 12 the mean principal cosine with the embedding's plane is
{f2(float(np.mean(bd("alive", 12)["cos_emb"])))} ({f2(float(np.mean(bd("orig", 12)["cos_emb"])))}).</p>
<p><b>Block 12: h times [units digit ≥ 5].</b> The MLP's gain rises to {f2(sl("alive", 12)["gain"])}
({f2(sl("orig", 12)["gain"])}), and {f2(sl("alive", 12)["share_from_P"])} ({f2(sl("orig", 12)["share_from_P"])}) of its
period-20 output comes from the plane. Its write is not linear in the plane, r²<sub>lin</sub> = {f2(sl("alive", 12)["r2_lin"])}
({f2(sl("orig", 12)["r2_lin"])}), but is close to linear for each units digit separately, r²<sub>gated</sub> =
{f2(sl("alive", 12)["r2_gated"])} ({f2(sl("orig", 12)["r2_gated"])}). The reader decomposition says the same: secant
{f2(mech("alive", 12, "r2_secant_sum"))} → gated secant {f2(mech("alive", 12, "r2_gated_secant_sum"))} (neurons
{f2(mech("orig", 12, "r2_secant_sum"))} → {f2(mech("orig", 12, "r2_gated_secant_sum"))}). In the components model three
readers carry 90% of Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²: g11 (the <i>Q</i> = 2 detector of figure 4), u57 and g7. In the neuron
basis gate neuron 8953 and up neuron 5212 carry the two largest shares. Figure 5 (left) shows how u57's write is gated.
Its activation follows the ring: highest for <i>s</i> ∈ 1–4, near zero for 5–9, lowest for 13–19 (the up neuron 5212:
positive for <i>s</i> ∈ 0–9, negative for 11–19). The size of its effect on the output, ‖(silu(<i>h</i><sub>gate</sub>) ⊙ <i>U</i><sub>up,57</sub>)<i>V</i><sub>down</sub><i>U</i><sub>down</sub>‖
(here <i>h</i><sub>gate</sub> ∈ ℝ<sup>14336</sup> is the vector of gate pre-activations, <i>U</i><sub>up,57</sub> component 57's write
into the 14,336 neurons, <i>V</i><sub>down</sub>, <i>U</i><sub>down</sub> the down components, ⊙ the elementwise product), is large when
<i>r</i> ∈ 5–9 and small when <i>r</i> ∈ 0–4. For the neuron, the slope is silu of its gate pre-activation: {rng(pr["orig_L12_silu_g5212"][np.arange(20) % 10 >= 5])}
for <i>r</i> ∈ 5–9 and {rng(pr["orig_L12_silu_g5212"][np.arange(20) % 10 < 5])} for <i>r</i> ∈ 0–4. A write proportional to (ring
reading) × [<i>r</i> ≥ 5] is what turns <i>h</i> into <i>g</i>: with ±1 coding, <i>g</i> = <i>h</i>·(1 − 2[<i>r</i> ≥ 5]).</p>
<p><b>Block 13: g is read, and the clouds tighten.</b> Across block 13, <i>α</i> barely moves, while the clouds
separate: held-out quarter accuracy {f2(bd("alive", 13)["acc"]["Q"])} → {f2(bd("alive", 14)["acc"]["Q"])}
({f2(bd("orig", 13)["acc"]["Q"])} → {f2(bd("orig", 14)["acc"]["Q"])}), closest quarters
{f2(bd("alive", 13)["sep_quarters_min"])} → {f2(bd("alive", 14)["sep_quarters_min"])}
({f2(bd("orig", 13)["sep_quarters_min"])} → {f2(bd("orig", 14)["sep_quarters_min"])}). Block 13's own write is not linear in the plane either:
r²<sub>lin</sub> {f2(sl("alive", 13)["r2_lin"])} ({f2(sl("orig", 13)["r2_lin"])}), and per-digit maps do not help
({f2(sl("alive", 13)["r2_gated"])}, {f2(sl("orig", 13)["r2_gated"])}). Its largest component, g58 ({mech("alive", 13, "top")[0]["share"] * 100:.0f}%
of the reader energy), reads <i>g</i> (figure 4); so does the largest neuron, 10534 (gate and up). In the neuron basis the
additivity of block 13 is {f2(mech("orig", 13, "r2_additivity"))}: removing the plane changes both the gate and the up
pre-activation of neuron 10534, and the product of the two changes, which belongs to neither reader alone, is large. Summed
per neuron instead of per reader, the decomposition is exact by construction.</p>
<p><b>Block 14: g read again, gated by a units-digit set.</b> r²<sub>lin</sub> {f2(sl("alive", 14)["r2_lin"])} →
r²<sub>gated</sub> {f2(sl("alive", 14)["r2_gated"])} (neurons {f2(sl("orig", 14)["r2_lin"])} → {f2(sl("orig", 14)["r2_gated"])}),
secant {f2(mech("alive", 14, "r2_secant_sum"))} → {f2(mech("alive", 14, "r2_gated_secant_sum"))}
({f2(mech("orig", 14, "r2_secant_sum"))} → {f2(mech("orig", 14, "r2_gated_secant_sum"))}). The largest component is up
reader u33 and the largest gated neuron path up neuron 6838; both activations are <i>g</i>-shaped (figure 5, right), and both
writes are gated by a function of the units digit that is largest for <i>r</i> ∈ {{2, 3, 8}}, smaller for 9 (component),
and largest for <i>r</i> ∈ {{2, 3, 8, 9}}, smaller for 5 (neuron).</p>
<figure><img src="notebook/fig/mod20_gating.png"><figcaption>Figure 5. The gated up readers of blocks 12 and 14. First and
third panels: per residue <i>s</i>, the mean size of the reader's write per unit of its activation (component: the norm of the
MLP output's derivative in <i>t<sub>c</sub></i>; neuron: silu of the same neuron's gate pre-activation). Second and fourth panels: the
reader's mean activation. Solid: component, dashed: neuron.</figcaption></figure>
<p><b>Blocks 15–16.</b> Gains {rng(sl("alive", b)["gain"] for b in (15, 16))} ({rng(sl("orig", b)["gain"] for b in (15, 16))}),
r²<sub>gated</sub> − r²<sub>lin</sub> = {rng(sl("alive", b)["r2_gated"] - sl("alive", b)["r2_lin"] for b in (15, 16))}
({rng(sl("orig", b)["r2_gated"] - sl("orig", b)["r2_lin"] for b in (15, 16))}); <i>α</i> rises back to
{f2(alpha("alive", 17))} ({f2(alpha("orig", 17))}) leaving block 16, and the ring moves away from the activation zero (the
norm of the mean of <i>z</i> grows from {bd("alive", 15)["offset_norm"]:.2f} to {bd("alive", 17)["offset_norm"]:.2f} while the mean
residue radius goes from {bd("alive", 15)["radius_mean"]:.2f} to {bd("alive", 17)["radius_mean"]:.2f}).</p>
<figure><img src="notebook/fig/mod20_sublayers.png"><figcaption>Figure 6. Per block (solid: components, dashed: neurons).
(a) Period-20 gain of the MLP and of attention. (b) Share of the MLP's period-20 output made from the plane, and the
period-20 output it makes without the plane. (c) Held-out r² of Δ from the plane's coordinates: one linear map, or one
per units digit. (d) Reader decomposition of Δ: secant, gated secant, additivity.</figcaption></figure>

<h3>5. Separation of the clouds</h3>
<p>Per boundary, in the plane: <i>c<sub>s</sub></i> is the mean of <i>z</i> over residue <i>s</i>, <i>sp<sub>s</sub></i> the root-mean-square
distance of those <i>z</i> to <i>c<sub>s</sub></i>, and the separation of residues <i>s</i>, <i>s′</i> is
‖<i>c<sub>s</sub></i> − <i>c<sub>s′</sub></i>‖/√((<i>sp<sub>s</sub></i>² + <i>sp<sub>s′</sub></i>²)/2); the same for the four quarters. η<sub><i>v</i></sub> is the
uncentred share of Σ<sub><i>a</i></sub>‖<i>z</i>(<i>a</i>)‖² explained by the class means of variable <i>v</i>. Accuracies are held out
(plane and class means fitted on one half of the <i>a</i>, tested on the other): <i>s</i> among 20 classes (chance 0.05), <i>Q</i>
among 4 (0.25), and <i>h</i> given <i>r</i> (choose between residues <i>r</i> and <i>r</i> + 10; 0.5).</p>
<p>Neighbouring residues are never well separated in this one plane (held-out accuracy for <i>s</i>
{rng(bd("alive", b)["acc"]["s"] for b in range(18))}); the units digit is carried by the mod-10 representation, and the plane
adds <i>h</i>. Antipodal residues are. At the embedding the median separation of <i>s</i> and <i>s</i> + 10 is
{f2(anti("alive", 0))}. In the components model it falls to {f2(anti("alive", 11))} entering block 11 (closest quarters
{f2(bd("alive", 11)["sep_quarters_min"])}; <i>h</i> given <i>r</i> {f2(bd("alive", 11)["acc"]["h_given_r"])}): the residue radius
grows {bd("alive", 0)["radius_mean"]:.3f} → {bd("alive", 11)["radius_mean"]:.3f} while the spread grows
{bd("alive", 0)["spread_mean"]:.3f} → {bd("alive", 11)["spread_mean"]:.3f}, and η<sub><i>s</i></sub> drops from
{f2(bd("alive", 0)["eta"]["s"])} to {f2(bd("alive", 9)["eta"]["s"])} (entering block 9). The neuron model blurs much less:
{f2(anti("orig", 11))} entering block 11, <i>h</i> given <i>r</i> {f2(bd("orig", 11)["acc"]["h_given_r"])}, η<sub><i>s</i></sub>
≥ {min(bd("orig", b)["eta"]["s"] for b in range(18)):.2f}. The MLPs of blocks 11–13 restore it in both (block 11 brings the
components model's antipodal separation to {f2(anti("alive", 12))}): entering block 14 the
antipodal separation is {f2(anti("alive", 14))} ({f2(anti("orig", 14))}), <i>h</i> given <i>r</i>
{f2(bd("alive", 14)["acc"]["h_given_r"])} ({f2(bd("orig", 14)["acc"]["h_given_r"])}), quarters
{f2(bd("alive", 14)["acc"]["Q"])} ({f2(bd("orig", 14)["acc"]["Q"])}).</p>
<figure><img src="notebook/fig/mod20_clouds.png"><figcaption>Figure 7. Per boundary (solid: components, dashed: neurons).
(a) η<sub><i>s</i></sub> and η<sub><i>Q</i></sub>. (b) Median antipodal separation and closest pair of quarters. (c) Held-out nearest-mean
accuracies. (d) Held-out probe accuracy for <i>h</i> from the whole stream, the plane only, and the stream minus the plane.</figcaption></figure>

<h3>6. Block by block</h3>
<p>Columns per model: MLP gain, share from <i>P</i>, r²<sub>lin</sub>, r²<sub>gated</sub>, secant, gated secant and additivity of
the block's MLP (section 4); then <i>α</i>, held-out quarter accuracy and median antipodal separation of the stream leaving the
block.</p>
{table}

<h3>7. Components against neurons</h3>
<ul>
<li>One plane, the frequency-1 plane of <i>a</i> mod 20, holds the period-20 information at every block: both models (figure 1,
probes in figure 7d).</li>
<li>A ring that turns into four quarter clumps in blocks 12–13: both (figures 2–3; <i>α</i> {f2(alpha("alive", 14))} vs
{f2(alpha("orig", 14))} entering block 14).</li>
<li>Readers of <i>h</i> up to block 12, of <i>g</i> from block 13: both (figure 4), with matched reader profiles in blocks 3,
12, 13 and 14.</li>
<li>Block 12 multiplies a ring reading by [<i>r</i> ≥ 5]; block 14 reads <i>g</i> gated by a units-digit set: both (figure 5).</li>
<li>Blocks 1–11 act through reader paths linear in their own input: components only (secant ≥
{min(mech("alive", b, "r2_secant_sum") for b in range(1, 12)):.2f}); single neurons are curved (secant
{rng(mech("orig", b, "r2_secant_sum") for b in range(1, 12))}).</li>
<li>Blur of the clouds in blocks 3–11: much stronger in the components model.</li>
</ul>
<p>Scripts: <code>track_build_orig.py</code>, <code>mod20_track.py</code>, <code>mod20_readers.py</code>, <code>mod20_page.py</code>;
data in <code>&lt;run&gt;/analysis/reader_writer/track_mod20/</code>.</p>
"""


if __name__ == "__main__":
    main()
