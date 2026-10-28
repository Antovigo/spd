"""Figures, numbers and notebook entry (id "mod20") of the page "Mod 20 representation at a (position 1)".

Page of the separate "Mod 20 representations" notebook (notebook_mod20/index.html).
Reads track_mod20/{alive,orig}/{analysis.json,readers_scan.npz} (mod20_track.py, mod20_readers.py) and the two float64
streams (track_build.py: alive-only model; track_build_orig.py: original Llama-3.1-8B). Writes notebook/fig/mod20_*.png,
track_mod20/page_numbers.json and the entry in <out>/entries.js. Nothing is centred: plot origins are the activation
zero. Notation as in mod20_track.py; in addition:
  o_s        = (m_s - m_{s+10}) / 2 in R^2, s = 0..9, and o_{s+10} = -o_s: the odd part of the residue means in the plane
  phi_s      angle of o_s (unwrapped along s = 0..19); step_s = phi_{s+1} - phi_s in degrees (s + 1 mod 20), 18 for an
             evenly spaced ring
  alpha      the alpha in [0, 1] minimising sum_s wrap(phi_s - c - 2 pi / 20 (alpha s + (1 - alpha) 5 floor(s / 5)))^2 over
             alpha and an offset angle c: 1 = evenly spaced ring, 0 = four points (one per quarter)

    python mod20_page.py --out notebook_mod20
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
    ap.add_argument("--text-only", action="store_true",
                    help="rewrite the page text from the stored numbers (page_numbers.json, profiles.npz); no figures")  # fmt: skip
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    res = {m: json.loads((OUT / m / "analysis.json").read_text()) for m in MODELS}
    scan = {m: dict(np.load(OUT / m / "readers_scan.npz")) for m in MODELS}
    if args.text_only:
        nums = json.loads((OUT / "page_numbers.json").read_text())
        html = page(
            res, nums["rings"], nums["reader_counts"], scan, dict(np.load(OUT / "profiles.npz"))
        )
        for p in args.entries or [args.out / "entries.js"]:
            insert_entry(p, html)
        return
    st = {
        m: {k: v.astype(np.float64) for k, v in np.load(STREAM[m] / "stream.npz").items()}
        for m in MODELS
    }
    rings = {m: {n: ring(st[m][n]) for n in PRE} for m in MODELS}
    pr = profiles(st)
    fig_planes(st, fig_dir / "mod20_planes.png")
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


def comp_label(r: dict) -> str:
    if r["odd_share"] < 0.05:
        return "little period-20 signal"
    if r["step_r2"] >= 0.8:
        return {0: "reads <i>h</i>", 5: "reads <i>g</i>"}.get(
            r["edge"], f"step at <i>s</i> = {r['edge']}"
        )
    if r["circle_r2"] >= 0.8:
        return "reads the ring"
    return "reads the plane"


def comp_text(c: dict, n_read: int = 4, n_write: int = 3) -> str:
    rd = ", ".join(
        f"{r['reader']} ({r['share'] * 100:.0f}%, {comp_label(r)})" for r in c["readers"][:n_read]
    )
    if not c["writers"]:
        return f"readers {rd}; writers: coming soon"
    wr = ", ".join(
        f"d{w['writer'][1:]} ({w['out_share'] * 100:+.0f}%)" for w in c["writers"][:n_write]
    )
    return f"readers {rd}; writers {wr}"


def readers_only(res: dict, scan: dict) -> dict:
    """The component lists without writers, from mod20_track.py's reader shares and mod20_readers.py's labels (used
    until mod20_components.py has written components.json)."""
    out = {}
    for b in range(17):
        rows = []
        for t in res["alive"]["sublayers"][f"mlp_{b}"]["mechanism"]["top"][:6]:
            kind = {"g": "gate", "u": "up"}[t["reader"][0]]
            j = list(scan["alive"][f"L{b}_{kind}_ids"]).index(int(t["reader"][1:]))
            rows.append(t | {k: float(scan["alive"][f"L{b}_{kind}_{k}"][j]) for k in ("odd", "step_r2", "circle_r2")}
                        | {"odd_share": float(scan["alive"][f"L{b}_{kind}_odd"][j]), "edge": int(scan["alive"][f"L{b}_{kind}_edge"][j])})  # fmt: skip
        out[str(b)] = {"readers": rows, "writers": []}
    return out


def sec4(res, comp, rings, pr, att_gain, f2, rng, sl, bd, mech, alpha) -> str:
    rows = "".join(
        f"<tr><td>{b}</td><td>{sl('alive', b)['gain']:.2f}</td><td>{sl('alive', b)['r2_lin']:.2f}</td>"
        f"<td>{sl('alive', b)['r2_gated']:.2f}</td><td>{comp_text(comp[str(b)])}</td></tr>"
        for b in range(17)
    )
    table = ("<table><tr><th>block</th><th>gain</th><th>r²<sub>lin</sub></th><th>r²<sub>gated</sub></th>"
             "<th>components (readers: share of the plane-driven write, what they read; writers: share of the period-20 output)</th></tr>"
             + rows + "</table>")  # fmt: skip
    c = {b: comp[str(b)] for b in range(17)}
    return f"""<h3>4. How the blocks move the representation</h3>
<p>Each block adds two things to the stream at the token <i>a</i>: its attention output and its MLP output. This section
goes through the blocks and asks four questions about each addition. Does it change the mod-20 plane <i>P</i> at all? Is the
change caused by the plane itself, or made from other information about <i>a</i>? Is it a simple linear map of the plane,
or does it depend on the units digit? And which components do it?</p>

<p><b>How each question is measured</b> (<code>mod20_track.py</code>, <code>mod20_components.py</code>). <i>X</i>(<i>a</i>) is the
stream entering the sublayer and <i>S</i>(<i>X</i>)(<i>a</i>) ∈ ℝ<sup>4096</sup> what the sublayer adds (<i>S</i> includes the block's
RMSNorm). For any quantity <i>f</i>(<i>a</i>) over the operands, Ψ<sub>1</sub>(<i>f</i>) is the frequency-1 Fourier coefficient of its 20
residue means (section 0.2): its period-20 content.</p>
<ul>
<li><b>Does it change the plane?</b> The <b>gain</b> ‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖/‖Ψ<sub>1</sub>(<i>X</i>)‖ compares the period-20 signal the
sublayer writes with the period-20 signal already in the stream. A gain of 0.5 means it writes half as much as is there.</li>
<li><b>Is the change caused by the plane?</b> Run the sublayer twice: on <i>X</i>, and on <i>X</i> − Π<sub><i>P</i></sub><i>X</i>, the same stream with
the plane removed. The difference Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>P</i></sub><i>X</i>)(<i>a</i>) is what the sublayer
writes because the plane is there. The <b>share from <i>P</i></b>, Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(Δ)⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖², is
the fraction of the sublayer's period-20 output that disappears when the plane is removed.</li>
<li><b>Is it linear in the plane?</b> Try to predict Δ(<i>a</i>) from the two plane coordinates <i>z</i>(<i>a</i>) with one fixed
4096 × 2 matrix, fitted on the operands <i>a</i> ≤ 200 and scored on <i>a</i> &gt; 200 (and the reverse). The score <b>r²<sub>lin</sub></b> is
1 − Σ<sub><i>a</i></sub>‖Δ − prediction‖² / Σ<sub><i>a</i></sub>‖Δ‖² (uncentred). <b>r²<sub>gated</sub></b> allows a different matrix for each units digit
<i>r</i> = <i>a</i> mod 10. When r²<sub>gated</sub> is clearly above r²<sub>lin</sub>, the block does something different to the plane for
different units digits.</li>
<li><b>Which components?</b> <i>Readers</i> (gate and up components): change one reader's activation as the plane's removal would,
keep the others fixed, and measure how much of Δ that produces; a reader's <b>share</b> is its part of the sum over readers of
Σ<sub><i>a</i></sub>‖Δ<sub>reader</sub>‖². Each listed reader is labelled with what it reads, from the scan of section 3.1: <i>h</i>
(tens parity), <i>g</i> (parity of <i>a</i> rounded to the nearest ten), "the ring" (a smooth period-20 profile), or "little period-20
signal" (odd share below 0.05). <i>Writers</i> (down components): writer <i>w</i> adds <i>t<sub>w</sub></i>(<i>a</i>) <i>U<sub>w</sub></i> to the stream, and its
<b>share of the period-20 output</b> is Re⟨Ψ<sub>1</sub>(<i>S</i>(<i>X</i>)), Ψ<sub>1</sub>(<i>t<sub>w</sub></i>) <i>U<sub>w</sub></i>⟩/‖Ψ<sub>1</sub>(<i>S</i>(<i>X</i>))‖²; the shares sum to
1 over writers and can be negative (a writer that shrinks the ring).</li>
<li><b>Reader-level linearity</b> (used in the table of section 2.3 and figure 6d). <b>Secant</b>: the r² of Δ when each reader's
contribution is replaced by a fixed multiple of its own activation change; high when every reader path is linear on its own.
<b>Gated secant</b>: the same with one multiple per units digit. <b>Additivity</b>: the r² of Δ by the plain sum of the single-reader
effects; below 1 when readers interact.</li>
</ul>
<p>Numbers in brackets are the original model's (neurons); the component lists are for the components model.</p>

<p><b>Attention leaves the plane alone.</b> In every block the attention's gain is at most {f2(att_gain["alive"])}
({f2(att_gain["orig"])}). Everything below is done by the MLPs.</p>

<p><b>Block 0: the ring grows and turns.</b> The MLP writes {f2(sl("alive", 0)["gain"])} times ({f2(sl("orig", 0)["gain"])}) the
period-20 signal it receives, and the plane turns: its mean principal cosine with the embedding's plane is
{f2(float(np.mean(bd("alive", 1)["cos_emb"])))} ({f2(float(np.mean(bd("orig", 1)["cos_emb"])))}) after the block. Only
{f2(sl("alive", 0)["share_from_P"])} ({f2(sl("orig", 0)["share_from_P"])}) of that output is caused by the plane: block 0 builds
most of the new ring from other features of the token. The work is spread thin: no reader carries more than
{c[0]["readers"][0]["share"] * 100:.0f}% of the plane-driven write, and {mech("alive", 0, "n_readers_90")} readers carry 90% of it. Components:
{comp_text(c[0])}.</p>

<p><b>Blocks 1–11: small linear touch-ups.</b> Each MLP writes {rng(sl("alive", b)["gain"] for b in range(1, 12))}
({rng(sl("orig", b)["gain"] for b in range(1, 12))}) of the period-20 signal it receives. Its effect is a linear map of the plane:
Δ is predicted from the two plane coordinates at held-out r² {rng(sl("alive", b)["r2_lin"] for b in range(1, 12))}
({rng(sl("orig", b)["r2_lin"] for b in range(1, 12))}), and one map per units digit does no better (at most
{max(sl(m, b)["r2_gated"] - sl(m, b)["r2_lin"] for m in MODELS for b in range(1, 12)):.2f} higher). In the components model each
reader path is itself linear (secant {rng(mech("alive", b, "r2_secant_sum") for b in range(1, 12))}) and the paths add up
(additivity ≥ {min(mech("alive", b, "r2_additivity") for b in range(1, 12)):.3f}); single neurons are more curved (secant
{rng(mech("orig", b, "r2_secant_sum") for b in range(1, 12))}). The plane drifts slowly: entering block 12, its mean principal
cosine with the embedding's plane is {f2(float(np.mean(bd("alive", 12)["cos_emb"])))} ({f2(float(np.mean(bd("orig", 12)["cos_emb"])))}).
The two largest writes are in blocks 3 and 11 (gains {f2(sl("alive", 3)["gain"])} and {f2(sl("alive", 11)["gain"])}). Block 3's is
almost one reader, u141 ({c[3]["readers"][0]["share"] * 100:.0f}%), which reads <i>h</i> (figure 4); its components: {comp_text(c[3])}.
Block 11's: {comp_text(c[11])}. The components of every other block are in the table below.</p>

<p><b>Block 12: the ring becomes four quarters.</b> This MLP turns the ring into the four clumps of section 2.1. Its gain is
{f2(sl("alive", 12)["gain"])} ({f2(sl("orig", 12)["gain"])}), and {f2(sl("alive", 12)["share_from_P"])}
({f2(sl("orig", 12)["share_from_P"])}) of its period-20 output is caused by the plane. The write is not one linear map of the plane
(r²<sub>lin</sub> {f2(sl("alive", 12)["r2_lin"])}, {f2(sl("orig", 12)["r2_lin"])}) but is close to one map per units digit
(r²<sub>gated</sub> {f2(sl("alive", 12)["r2_gated"])}, {f2(sl("orig", 12)["r2_gated"])}). Three components carry 90% of it:</p>
<ul>
<li>u57 ({c[12]["readers"][1]["share"] * 100:.0f}%) reads the plane (in-plane value 0.66, section 3.2) and feeds neuron 5212's up input. That neuron's gate opens only
for units digits 5–9, so u57's write is multiplied by [<i>r</i> ≥ 5]: figure 5, left, shows the size of its write per unit of
activation, large for <i>r</i> = 5–9 and small for <i>r</i> = 0–4. In the original model, the same neuron's silu(gate) is
{rng(pr["orig_L12_silu_g5212"][np.arange(20) % 10 >= 5])} for <i>r</i> = 5–9 and
{rng(pr["orig_L12_silu_g5212"][np.arange(20) % 10 < 5])} for <i>r</i> = 0–4. A ring reading times [<i>r</i> ≥ 5] is the
product that turns <i>h</i> into <i>g</i>: in ±1 coding, <i>g</i> = <i>h</i>·(1 − 2[<i>r</i> ≥ 5]).</li>
<li>g11 ({c[12]["readers"][0]["share"] * 100:.0f}%) fires only for <i>s</i> = 10–14 (figure 4) and drives neuron 8953's gate,
which writes a vector only for that quarter.</li>
<li>g7 ({c[12]["readers"][2]["share"] * 100:.0f}%).</li>
</ul>
<p>Sections 3.2 and 4.2 draw these readers in the plane. Writers: {", ".join(f"d{w['writer'][1:]} ({w['out_share'] * 100:+.0f}%)" for w in c[12]["writers"][:4]) or "coming soon"}.</p>

<p><b>Block 13: <i>g</i> is read and the clouds tighten.</b> The quarter structure stays (<i>α</i> barely moves), and the clouds
separate: held-out quarter accuracy goes from {f2(bd("alive", 13)["acc"]["Q"])} to {f2(bd("alive", 14)["acc"]["Q"])}
({f2(bd("orig", 13)["acc"]["Q"])} → {f2(bd("orig", 14)["acc"]["Q"])}). The write is not linear in the plane
(r²<sub>lin</sub> {f2(sl("alive", 13)["r2_lin"])}, {f2(sl("orig", 13)["r2_lin"])}) and a map per units digit does not help
({f2(sl("alive", 13)["r2_gated"])}, {f2(sl("orig", 13)["r2_gated"])}). One reader dominates: g58
({c[13]["readers"][0]["share"] * 100:.0f}%), which reads <i>g</i>; in the original model neuron 10534 (gate and up) does the same.
Removing the plane changes both the gate and the up input of neuron 10534, and the product of the two changes belongs to
neither reader alone, which is why the neuron-basis additivity is {f2(mech("orig", 13, "r2_additivity"))}. Components:
{comp_text(c[13])}.</p>

<p><b>Block 14: <i>g</i> read again, gated by the units digit.</b> r²<sub>lin</sub> {f2(sl("alive", 14)["r2_lin"])} →
r²<sub>gated</sub> {f2(sl("alive", 14)["r2_gated"])} ({f2(sl("orig", 14)["r2_lin"])} → {f2(sl("orig", 14)["r2_gated"])}). The
largest reader, u33 ({c[14]["readers"][0]["share"] * 100:.0f}%), and the largest neuron path, up neuron 6838, both read <i>g</i>
(figure 5, right); their writes are largest for units digits 2, 3 and 8 (component) and 2, 3, 8 and 9 (neuron). Components:
{comp_text(c[14])}.</p>
<figure><img src="fig/mod20_gating.png"><figcaption>Figure 5. The gated up readers of blocks 12 and 14. First and
third panels: per residue <i>s</i>, the mean size of the reader's write per unit of its activation (component: the norm of the
MLP output's derivative in <i>t<sub>c</sub></i>; neuron: silu of the same neuron's gate pre-activation). Second and fourth panels: the
reader's mean activation. Solid: component, dashed: neuron.</figcaption></figure>

<p><b>Blocks 15–16: the quarters loosen and the ring drifts off zero.</b> Gains {rng(sl("alive", b)["gain"] for b in (15, 16))}
({rng(sl("orig", b)["gain"] for b in (15, 16))}); the per-digit maps add {rng(sl("alive", b)["r2_gated"] - sl("alive", b)["r2_lin"] for b in (15, 16))}
({rng(sl("orig", b)["r2_gated"] - sl("orig", b)["r2_lin"] for b in (15, 16))}) to r². <i>α</i> rises back to
{f2(alpha("alive", 17))} ({f2(alpha("orig", 17))}) leaving block 16, and the ring moves away from the activation zero (the
distance of the plane coordinates' mean from zero grows from {bd("alive", 15)["offset_norm"]:.2f} to {bd("alive", 17)["offset_norm"]:.2f}
while the mean residue radius goes from {bd("alive", 15)["radius_mean"]:.2f} to {bd("alive", 17)["radius_mean"]:.2f}). The work is
spread over more readers ({mech("alive", 15, "n_readers_90")} and {mech("alive", 16, "n_readers_90")} for 90%). Block 15:
{comp_text(c[15])}. Block 16: {comp_text(c[16])}.</p>

<p><b>All blocks.</b> Gain, linearity and the main components of every MLP (components model):</p>
{table}
<figure><img src="fig/mod20_sublayers.png"><figcaption>Figure 6. Per block (solid: components, dashed: neurons).
(a) Period-20 gain of the MLP and of attention. (b) Share of the MLP's period-20 output caused by the plane, and the
period-20 output it makes without the plane. (c) Held-out r² of Δ from the plane's coordinates: one linear map, or one
per units digit. (d) Reader-level linearity: secant, gated secant, additivity.</figcaption></figure>"""


def axis_steps(odd: list) -> list[dict]:
    """Principal axes of the 10 x 2 odd parts; for each, the best step edge e (sigma_e(s) = -1 for s < e, +1 otherwise,
    s = 0..9) and its uncentred r2, and the axis' share of the odd-part energy."""
    o = np.asarray(odd)
    u, sv, vt = np.linalg.svd(o, full_matrices=False)
    ss = np.arange(10)
    out = []
    for i in range(2):
        prof = o @ vt[i]
        r2s = [
            float((np.where(ss < e, -1.0, 1.0) @ prof) ** 2 / 10 / (prof @ prof)) for e in range(10)
        ]
        e = int(np.argmax(r2s))
        out.append(
            {"edge": e, "step_r2": r2s[e], "energy_share": float(sv[i] ** 2 / (sv**2).sum())}
        )
    return out


def sec11(cp: dict, f2, rng) -> str:
    def g(m: str, w: str, n: str, k: str):
        return cp[f"{m}_{w}"][n][k]

    def amp(m: str, w: str, n: str, j: int) -> float:
        x = cp[f"{m}_{w}"][n]
        return x["amp"][j - 1] / x["null99"][j - 1]

    def gain(m: str, w: str, j: int) -> float:
        return cp[f"{m}_{w}"]["gain"][j - 1]

    def heads(m: str, w: str) -> str:
        h20, h10 = (
            np.asarray(cp[f"{m}_{w}"]["heads"]["p20"]),
            np.asarray(cp[f"{m}_{w}"]["heads"]["p10"]),
        )
        o = np.argsort(-h20)[:2]
        return ", ".join(
            f"head {int(h)} ({h20[h] * 100:.0f}% of the period-20 copy, {h10[h] * 100:.0f}% of the period-10 copy)"
            for h in o
        )

    def vlist(w: str, key: str, other: str) -> str:
        rows = cp[f"alive_{w}"]["vcomps" if key == "p20" else "vcomps_p10_top"][:4]
        return ", ".join(f"v{c['id']} ({c[key] * 100:.0f}%; {c[other] * 100:.0f}%)" for c in rows)

    def edge_name(e: int) -> str:
        return {0: "<i>h</i>", 5: "<i>g</i>"}.get(e, f"a step at <i>s</i> = {e}")

    def axes(m: str, w: str, n: str) -> str:
        if "odd" not in cp[f"{m}_{w}"][n]:
            return ""
        a = axis_steps(cp[f"{m}_{w}"][n]["odd"])
        return " and ".join(f"{edge_name(x['edge'])} on the {'long' if i == 0 else 'short'} axis ({x['energy_share'] * 100:.0f}% of the ring's energy, step r² {x['step_r2']:.2f})"
                            for i, x in enumerate(a))  # fmt: skip

    def rd(m: str, w: str, n: str) -> str:
        x = cp[f"{m}_{w}"][n]
        return f"{x['roundness']:.2f}" if "roundness" in x else "–"

    rows = ""
    for m, mn in (("alive", "components"), ("orig", "neurons")):
        for w, L in (("a", 16), ("b", 15)):
            for n, nn in (("source", "source position"), ("copy", 'copy written at "="'), ("eq_before", '"=" before the attention'),
                          ("eq_after", '"=" after the attention')):  # fmt: skip
                rows += (f"<tr><td>{mn}</td><td>{w} (block {L})</td><td>{nn}</td><td>{amp(m, w, n, 1):.2f}</td><td>{g(m, w, n, 'alpha'):.2f}</td>"
                         f"<td>{rd(m, w, n)}</td><td>{g(m, w, n, 'acc_h'):.2f}</td><td>{g(m, w, n, 'd_anti'):.1f}</td></tr>")  # fmt: skip
    table = ("<table><tr><th>model</th><th>copy</th><th>where</th><th>‖Ψ<sub>1</sub>‖/null</th><th>α</th><th>roundness</th>"
             "<th><i>h</i> given <i>r</i> (held out)</th><th>d(<i>s</i>, <i>s</i>+10)</th></tr>" + rows + "</table>")  # fmt: skip
    return f"""<h3>11. Copying the circle to "="</h3>
<p>The answer is written at the last token, "=". The operands reach it through attention: block 16 copies <i>a</i> from
position 1 and block 15 copies <i>b</i> from position 3 (the mod-10 notebook, part 5). This section asks whether the mod-20
circle is copied too, by which heads and value components, and what it looks like once it arrives (<code>mod20_copy.py</code>).</p>
<p><b>Setup.</b> The prompts are the two sets of the mod-10 copy analysis: for the copy of <i>a</i>, every <i>a</i> = 1…400 with
ten values of <i>b</i> and both operations (8,000 prompts); for the copy of <i>b</i>, ten values of <i>a</i> and every <i>b</i>. The
original model uses 2,000 of each set. In this section <i>s</i> = <i>o</i> mod 20, <i>r</i> = <i>o</i> mod 10 and <i>h</i> = ⌊<i>o</i>/10⌋ mod 2,
where <i>o</i> is the copied operand (<i>a</i> or <i>b</i>). Four places are compared:</p>
<ul>
<li><b>source position</b>: the attention input (after the block's input RMSNorm) at the operand's own position;</li>
<li><b>copy written at "="</b>: the part of the attention output at "=" that comes from the operand's position (all heads);</li>
<li><b>"=" before / after the attention</b>: the stream at "=" entering the block, and the same plus the whole attention output.</li>
</ul>
<p>For a quantity over the prompts, Ψ<sub><i>j</i></sub> is the Fourier coefficient at frequency <i>j</i> of its 20 residue means
(section 0.2), and "‖Ψ<sub>1</sub>‖/null" compares the circle's strength with the 99th percentile over 100 random relabellings of
the prompts. The <b>copy gain</b> at frequency <i>j</i> is ‖Ψ<sub><i>j</i></sub>(copy)‖/‖Ψ<sub><i>j</i></sub>(source)‖. α, the held-out accuracy
of <i>h</i> given <i>r</i> and the antipodal Cohen's d are as in sections 2.1 and 2.2, computed in the plane of the place in
question. <b>Roundness</b> is the smaller over the larger singular value of the ten odd parts <i>o<sub>s</sub></i> (1 for a round ring).
A head's or value component's <b>share</b> is its part of Σ‖Ψ<sub>1</sub>‖² over heads (or components) of the copy, set beside its
part of Σ<sub><i>k</i>=1…5</sub>‖Ψ<sub>2<i>k</i></sub>‖², the copy of the mod-10 representation.</p>

<p><b>The circle is copied, by the same head as the mod-10 representation.</b> The copy of <i>a</i> carries the circle with a
gain of {gain("alive", "a", 1):.2f} ({gain("orig", "a", 1):.2f}), within the range of the mod-10 frequencies'
gains, {rng(gain("alive", "a", j) for j in (2, 4, 6, 8, 10))} ({rng(gain("orig", "a", j) for j in (2, 4, 6, 8, 10))}). For <i>b</i> the
circle's gain is {gain("alive", "b", 1):.2f} ({gain("orig", "b", 1):.2f}) against {rng(gain("alive", "b", j) for j in (2, 4, 6, 8, 10))}
({rng(gain("orig", "b", j) for j in (2, 4, 6, 8, 10))}). The heads: for <i>a</i>, {heads("alive", "a")} (original model:
{heads("orig", "a")}); for <i>b</i>, {heads("alive", "b")} (original model: {heads("orig", "b")}). The same head carries both
periods in the same proportion.</p>
<p><b>But through different value components.</b> In the components model, the value components that carry the copied circle
are not the ones that carry the copied mod-10 representation (each listed with its share of the period-20 copy; then of the
period-10 copy). For <i>a</i>, period 20: {vlist("a", "p20", "p10")}; period 10: {vlist("a", "p10", "p20")}. For <i>b</i>,
period 20: {vlist("b", "p20", "p10")}; period 10: {vlist("b", "p10", "p20")}.</p>
<p><b>The copy is one linear map of the circle, which flattens it.</b> The ten odd parts of the copy are fitted from those at
the source by one 2 × 2 matrix at r² {cp["alive_a"]["map_r2"]:.3f} and {cp["alive_b"]["map_r2"]:.3f} (components; neurons
{cp["orig_a"]["map_r2"]:.3f} and {cp["orig_b"]["map_r2"]:.3f}). The map is not a rotation: it keeps one direction of the plane
and shrinks the other. Roundness goes from {rd("alive", "a", "source")} at the source to {rd("alive", "a", "copy")} in the copy of
<i>a</i> (neurons {rd("orig", "a", "source")} → {rd("orig", "a", "copy")}) and from {rd("alive", "b", "source")} to {rd("alive", "b", "copy")}
in the copy of <i>b</i> ({rd("orig", "b", "source")} → {rd("orig", "b", "copy")}); α falls with it ({g("alive", "a", "source", "alpha"):.2f} →
{g("alive", "a", "copy", "alpha"):.2f} and {g("alive", "b", "source", "alpha"):.2f} → {g("alive", "b", "copy", "alpha"):.2f}). The long axis is the rounded-tens parity <i>g</i>, the short one mostly the tens parity <i>h</i>.
In the components model the copied ring carries {axes("alive", "a", "copy")} for <i>a</i>,
and {axes("alive", "b", "copy")} for <i>b</i>. In the original model: {axes("orig", "a", "copy")} for <i>a</i>; {axes("orig", "b", "copy")}
for <i>b</i>.</p>
<p><b>What arrives at "=".</b> Before the attention, "=" holds no circle of the operand (‖Ψ<sub>1</sub>‖/null
{amp("alive", "a", "eq_before", 1):.2f} for <i>a</i>, {amp("alive", "b", "eq_before", 1):.2f} for <i>b</i>; <i>h</i> given <i>r</i> at chance). After
it, the circle is there: <i>h</i> given <i>r</i> is decoded at {g("alive", "a", "eq_after", "acc_h"):.2f} for <i>a</i> and
{g("alive", "b", "eq_after", "acc_h"):.2f} for <i>b</i> (neurons {g("orig", "a", "eq_after", "acc_h"):.2f} and
{g("orig", "b", "eq_after", "acc_h"):.2f}), and the plane at "=" is the copy's plane (principal cosines
{", ".join(f"{x:.3f}" for x in cp["alive_a"]["cos_copy_eq_after_planes"])} for <i>a</i>,
{", ".join(f"{x:.3f}" for x in cp["alive_b"]["cos_copy_eq_after_planes"])} for <i>b</i>).</p>
<p><b>A caveat on the null.</b> At the source position the stream depends on the operand alone, so the 400 prompts of one
residue share 20 identical copies of each value, and the shuffled-label null is weak. Here the odd frequencies <i>j</i> = 3…9
also exceed it (up to {max(amp("alive", "a", "source", j) for j in (3, 5, 7, 9)):.1f} times for <i>a</i> at the source), unlike in the
400-prompt analysis of section 1.1.</p>
{table}
<figure><img src="fig/mod20_copy_alive.png"><figcaption>Figure 16. Components model. Rows: the copy of <i>a</i> by block 16
and of <i>b</i> by block 15. First three panels: the circle's odd parts <i>o<sub>s</sub></i> and −<i>o<sub>s</sub></i> (<i>s</i> = 0…19, coloured by
quarter as in figure 2) at the source position, in the copy written at "=", and at "=" after the attention, each in its own
plane, turned so that residue 0 points right; the cross is the plane's zero. Titles give α, the held-out accuracy of <i>h</i>
given <i>r</i>, and the median antipodal Cohen's d. Last panel: each head's share of the copy's period-20 energy and of its
period-10 energy.</figcaption></figure>
<figure><img src="fig/mod20_copy_orig.png"><figcaption>Figure 17. As figure 16, original model (2,000 prompts per copy).</figcaption></figure>"""


QHEX = [["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"], ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#a63603"],
        ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c"], ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f"]]  # fmt: skip


def sec12(c3: dict) -> str:
    """Rotatable 3D views (plotly) of the quarter clumps at the source positions; data inlined."""
    data = json.dumps(c3, separators=(",", ":"))
    colors = json.dumps([QHEX[s // 5][s % 5] for s in range(20)])

    def sh(k: str, f: str) -> str:
        return f"{c3[k]['shares'][f]:.2f}"

    return f"""<h3>12. The four clumps in 3D, just before the copy</h3>
<p>After block 12 the 20 residues gather into four clumps (section 2), the quarters <i>Q</i> = ⌊<i>s</i>/5⌋, with
<i>s</i> = <i>o</i> mod 20 for the operand <i>o</i>. The four clump centres need three dimensions: two in the mod-20 plane and one in
the mod-10 representation (the singular values of their differences are in the ratio 1 : 0.5 : 0.4, and more than 99% of
those differences lie in the mod-20 plane or the mod-10 subspace; <code>mod20_clumps.py</code>). The two views below show those three dimensions at
the moment the operand is copied: <i>a</i> at position 1 entering block 16, and <i>b</i> at position 3 entering block 15, in the
attention input of the copying block (after its input RMSNorm). Drag to rotate, scroll to zoom, and use the buttons to
switch between the components model and the original model (<code>mod20_clumps3d.py</code>).</p>
<p><b>Axes.</b> <i>ν<sub>Q</sub></i> ∈ ℝ<sup>4096</sup> is the mean of the attention input over the prompts of quarter <i>Q</i>, and the
clump layout is spanned by the three differences <i>ν</i><sub>1</sub> − <i>ν</i><sub>0</sub>, <i>ν</i><sub>2</sub> − <i>ν</i><sub>0</sub>,
<i>ν</i><sub>3</sub> − <i>ν</i><sub>0</sub>. Axes 1 and 2 are an orthonormal basis of the part of that span that lies in the mod-20 plane
<i>P</i>, turned so that residue 0 is on the positive side of axis 1. Axis 3 is the unit vector along the rest, oriented so that
quarter 1 (units digit 5–9) lies above quarter 0. Every point is a prompt's projection onto the three axes; nothing is
centred, so the origin (the black cross) is the activation zero. Axis 3 lies in the mod-10 subspace: its share there is
{c3["alive_a"]["e3_in_D10"]:.2f} for <i>a</i> and {c3["alive_b"]["e3_in_D10"]:.2f} for <i>b</i> (components;
{c3["orig_a"]["e3_in_D10"]:.2f} and {c3["orig_b"]["e3_in_D10"]:.2f} in the original model). Of the clump layout's energy
(Σ‖<i>ν<sub>Q</sub></i> − <i>ν</i><sub>0</sub>‖²), the three axes hold {sh("alive_a", "G_in_view")} for <i>a</i> and {sh("alive_b", "G_in_view")}
for <i>b</i>; the mod-20 plane holds {sh("alive_a", "G_in_P")} and {sh("alive_b", "G_in_P")}, and the mod-10 subspace {sh("alive_a", "G_in_D10")}
and {sh("alive_b", "G_in_D10")} (the two overlap slightly).</p>
<p><b>Markers.</b> Small dots: prompts (400 for <i>a</i>, one per value; for <i>b</i>, 8,000 prompts in the components model and 2,000
in the original model, since position 3 also depends on <i>a</i> and the operation). Colour: residue <i>s</i>, one colour family
per quarter (blue 0–4, orange 5–9, green 10–14, purple 15–19), darker for larger <i>s</i> within the quarter. Labelled
circles: the 20 residue means. Diamonds: the four quarter centres. Click a legend entry to hide a residue.</p>
<div class="applet" id="clumps3d_a" style="width:100%;height:560px"></div>
<p class="note">Figure 18. <i>a</i> at position 1 entering block 16, projected onto the three axes above.</p>
<div class="applet" id="clumps3d_b" style="width:100%;height:560px"></div>
<p class="note">Figure 19. <i>b</i> at position 3 entering block 15, projected onto the three axes above.</p>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<script>
(function () {{
  const C3 = {data};
  const COL = {colors};
  const fg = getComputedStyle(document.body).color || "#222";
  function traces(key, visible) {{
    const d = C3[key], out = [];
    for (let s = 0; s < 20; s++) {{
      const idx = d.s.map((v, i) => v === s ? i : -1).filter(i => i >= 0);
      out.push({{type: "scatter3d", mode: "markers", name: "s = " + s, legendgroup: "s" + s, visible: visible,
        x: idx.map(i => d.points[i][0]), y: idx.map(i => d.points[i][1]), z: idx.map(i => d.points[i][2]),
        text: idx.map(i => "operand " + d.operand[i] + ", s = " + s), hoverinfo: "text",
        marker: {{size: 2.2, color: COL[s], opacity: 0.55}}}});
    }}
    out.push({{type: "scatter3d", mode: "markers+text", name: "residue means", visible: visible,
      x: d.residue_means.map(p => p[0]), y: d.residue_means.map(p => p[1]), z: d.residue_means.map(p => p[2]),
      text: d.residue_means.map((p, s) => String(s)), textfont: {{color: fg, size: 11}}, hoverinfo: "text",
      marker: {{size: 6, color: COL, line: {{color: fg, width: 1}}}}}});
    out.push({{type: "scatter3d", mode: "markers", name: "quarter centres", visible: visible,
      x: d.quarter_means.map(p => p[0]), y: d.quarter_means.map(p => p[1]), z: d.quarter_means.map(p => p[2]),
      text: ["Q = 0 (s 0–4)", "Q = 1 (s 5–9)", "Q = 2 (s 10–14)", "Q = 3 (s 15–19)"], hoverinfo: "text",
      marker: {{size: 9, symbol: "diamond", color: [COL[2], COL[7], COL[12], COL[17]], line: {{color: fg, width: 1.5}}}}}});
    out.push({{type: "scatter3d", mode: "markers", name: "activation zero", visible: visible, x: [0], y: [0], z: [0],
      hoverinfo: "name", marker: {{size: 5, symbol: "cross", color: fg}}}});
    return out;
  }}
  function draw(div, w) {{
    const a = traces("alive_" + w, true), o = traces("orig_" + w, false), n = a.length;
    const vis = (on) => Array(n).fill(on).concat(Array(n).fill(!on));
    const ax = (t) => ({{title: {{text: t, font: {{color: fg}}}}, color: fg, gridcolor: "rgba(128,128,128,0.3)", zerolinecolor: "rgba(128,128,128,0.6)"}});
    Plotly.newPlot(div, a.concat(o), {{
      paper_bgcolor: "rgba(0,0,0,0)", font: {{color: fg}}, margin: {{l: 0, r: 0, t: 30, b: 0}},
      legend: {{font: {{size: 10}}, itemsizing: "constant"}},
      scene: {{xaxis: ax("axis 1 (mod-20 plane)"), yaxis: ax("axis 2 (mod-20 plane)"), zaxis: ax("axis 3 (mod-10)"), aspectmode: "data"}},
      updatemenus: [{{type: "buttons", direction: "right", x: 0, y: 1.08, xanchor: "left", showactive: true,
        buttons: [{{label: "components", method: "restyle", args: ["visible", vis(true)]}},
                  {{label: "neurons (original model)", method: "restyle", args: ["visible", vis(false)]}}]}}]
    }}, {{responsive: true, displaylogo: false}});
  }}
  if (window.Plotly) {{ draw("clumps3d_a", "a"); draw("clumps3d_b", "b"); }}
}})();
</script>"""


def sec13(c3: dict) -> str:
    """Rotatable 3D views of the quarter clumps at "=" after the copy (copy alone, and the stream after the attention)."""
    data = json.dumps(c3, separators=(",", ":"))
    colors = json.dumps([QHEX[s // 5][s % 5] for s in range(20)])

    def sh(k: str, f: str) -> str:
        return f"{c3[k]['shares'][f]:.2f}"

    def row(k: str, name: str) -> str:
        x = c3[k]
        sv = np.asarray(x["G_sv"])
        return (f"<tr><td>{name}</td><td>{sh(k, 'G_in_P')}</td><td>{sh(k, 'G_in_D10')}</td><td>{sh(k, 'G_in_view')}</td>"
                f"<td>1 : {sv[1] / sv[0]:.2f} : {sv[2] / sv[0]:.2f}</td><td>{x['e3_in_D10']:.2f}</td></tr>")  # fmt: skip

    rows = "".join(row(f"{m}_{w}_{n}", f"{mn}, {w}, {nn}") for m, mn in (("alive", "components"), ("orig", "neurons"))
                   for w in ("a", "b") for n, nn in (("copy", "copy alone"), ("eq", '"=" after the attention')))  # fmt: skip
    table = ("<table><tr><th>view</th><th>clump layout in the mod-20 plane</th><th>in the mod-10 subspace</th>"
             "<th>in the three axes</th><th>singular values</th><th>axis 3 in the mod-10 subspace</th></tr>" + rows + "</table>")  # fmt: skip
    return f"""<h3>13. The four clumps in 3D, after the copy</h3>
<p>The same construction as section 12, at the last position "=" once the copying block's attention has written: block 16
for <i>a</i>, block 15 for <i>b</i>, with the prompt sets of section 11 (for <i>a</i>: every <i>a</i>, ten values of <i>b</i>, both
operations; for <i>b</i>: ten values of <i>a</i>, every <i>b</i>; 8,000 prompts in the components model, 2,000 in the original model).
Each view has four versions, chosen with the buttons: the <b>copy alone</b> (the part of the attention output at "=" that comes
from the operand's position, all heads), and the whole <b>stream at "=" after the attention</b> (the stream entering the block
plus the attention output from all positions), each in the components model and in the original model. The axes, the
markers and the colours are defined as in section 12, recomputed for each version from its own residue and quarter means.
Dots are coloured by the copied operand's residue; in the stream at "=" the other operand and the operation also vary, which
spreads each clump.</p>
<div class="tw">{table}</div>
<div class="applet" id="clumps3d_after_a" style="width:100%;height:560px"></div>
<p class="note">Figure 20. <i>a</i> at "=" after block 16's attention.</p>
<div class="applet" id="clumps3d_after_b" style="width:100%;height:560px"></div>
<p class="note">Figure 21. <i>b</i> at "=" after block 15's attention.</p>
<script>
(function () {{
  const C3 = {data};
  const COL = {colors};
  const fg = getComputedStyle(document.body).color || "#222";
  function traces(key, visible) {{
    const d = C3[key], out = [];
    for (let s = 0; s < 20; s++) {{
      const idx = d.s.map((v, i) => v === s ? i : -1).filter(i => i >= 0);
      out.push({{type: "scatter3d", mode: "markers", name: "s = " + s, legendgroup: "s" + s, visible: visible, showlegend: visible,
        x: idx.map(i => d.points[i][0]), y: idx.map(i => d.points[i][1]), z: idx.map(i => d.points[i][2]),
        text: idx.map(i => "operand " + d.operand[i] + ", s = " + s), hoverinfo: "text",
        marker: {{size: 2, color: COL[s], opacity: 0.5}}}});
    }}
    out.push({{type: "scatter3d", mode: "markers+text", name: "residue means", visible: visible, showlegend: visible,
      x: d.residue_means.map(p => p[0]), y: d.residue_means.map(p => p[1]), z: d.residue_means.map(p => p[2]),
      text: d.residue_means.map((p, s) => String(s)), textfont: {{color: fg, size: 11}}, hoverinfo: "text",
      marker: {{size: 6, color: COL, line: {{color: fg, width: 1}}}}}});
    out.push({{type: "scatter3d", mode: "markers", name: "quarter centres", visible: visible, showlegend: visible,
      x: d.quarter_means.map(p => p[0]), y: d.quarter_means.map(p => p[1]), z: d.quarter_means.map(p => p[2]),
      text: ["Q = 0 (s 0–4)", "Q = 1 (s 5–9)", "Q = 2 (s 10–14)", "Q = 3 (s 15–19)"], hoverinfo: "text",
      marker: {{size: 9, symbol: "diamond", color: [COL[2], COL[7], COL[12], COL[17]], line: {{color: fg, width: 1.5}}}}}});
    out.push({{type: "scatter3d", mode: "markers", name: "activation zero", visible: visible, showlegend: visible, x: [0], y: [0], z: [0],
      hoverinfo: "name", marker: {{size: 5, symbol: "cross", color: fg}}}});
    return out;
  }}
  function draw(div, w) {{
    const keys = ["alive_" + w + "_copy", "alive_" + w + "_eq", "orig_" + w + "_copy", "orig_" + w + "_eq"];
    const labels = ["components: copy", 'components: "=" after', "neurons: copy", 'neurons: "=" after'];
    const all = keys.map((k, i) => traces(k, i === 0)), n = all[0].length;
    const vis = (j) => all.flatMap((t, i) => Array(n).fill(i === j));
    const ax = (t) => ({{title: {{text: t, font: {{color: fg}}}}, color: fg, gridcolor: "rgba(128,128,128,0.3)", zerolinecolor: "rgba(128,128,128,0.6)"}});
    Plotly.newPlot(div, all.flat(), {{
      paper_bgcolor: "rgba(0,0,0,0)", font: {{color: fg}}, margin: {{l: 0, r: 0, t: 30, b: 0}},
      legend: {{font: {{size: 10}}, itemsizing: "constant"}},
      scene: {{xaxis: ax("axis 1 (mod-20 plane)"), yaxis: ax("axis 2 (mod-20 plane)"), zaxis: ax("axis 3 (mod-10)"), aspectmode: "data"}},
      updatemenus: [{{type: "buttons", direction: "right", x: 0, y: 1.08, xanchor: "left", showactive: true,
        buttons: labels.map((l, j) => ({{label: l, method: "restyle", args: [{{visible: vis(j), showlegend: vis(j)}}]}}))}}]
    }}, {{responsive: true, displaylogo: false}});
  }}
  if (window.Plotly) {{ draw("clumps3d_after_a", "a"); draw("clumps3d_after_b", "b"); }}
}})();
</script>"""


def sec14(eqp: dict, eo: dict) -> str:
    """The plane of the stream at "=" after the copy, with the attention's write and the pre-copy stream in it."""

    def st(m: str, w: str, part: str, k: str) -> float:
        v = eqp[f"{m}_{w}_{part}"][k]
        return float(np.linalg.norm(v)) if isinstance(v, list) else float(v)

    def fs(m: str, w: str, term: str, k: str) -> float:
        name = {"E": "E (stream before the attention)", "C": "C (copy from the operand)",
                "R": "R (attention from the other positions)", "E'": "E' (after the attention)"}[term]  # fmt: skip
        return float(eo[f"{m}_{w}"]["terms"][name][k])

    rows = ""
    for m, mn in (("alive", "components"), ("orig", "neurons")):
        for w in ("a", "b"):
            rows += (f"<tr><td>{mn}</td><td>{w}</td><td>{st(m, w, 'before', 'psi1_norm_in_plane'):.2f}</td>"
                     f"<td>{st(m, w, 'write', 'psi1_norm_in_plane'):.2f}</td><td>{st(m, w, 'after', 'psi1_norm_in_plane'):.2f}</td>"
                     f"<td>{st(m, w, 'before', 'rms_spread'):.2f}</td><td>{st(m, w, 'write', 'rms_spread'):.2f}</td>"
                     f"<td>{st(m, w, 'after', 'rms_spread'):.2f}</td><td>{st(m, w, 'before', 'mean'):.2f}</td></tr>")  # fmt: skip
    table = ("<table><tr><th>model</th><th>copy</th><th>‖Ψ<sub>1</sub>‖ before</th><th>‖Ψ<sub>1</sub>‖ write</th><th>‖Ψ<sub>1</sub>‖ after</th>"
             "<th>spread before</th><th>spread write</th><th>spread after</th><th>‖mean‖ before</th></tr>" + rows + "</table>")  # fmt: skip
    srows = ""
    for m, mn in (("alive", "components"), ("orig", "neurons")):
        for w in ("a", "b"):
            for t, tn in (
                ("E", "before the attention"),
                ("C", "copy from the operand"),
                ("R", "attention from other positions"),
            ):
                srows += (f"<tr><td>{mn}</td><td>{w}</td><td>{tn}</td><td>{fs(m, w, t, 'offset'):.2f}</td><td>{fs(m, w, t, 'own'):.2f}</td>"
                          f"<td>{fs(m, w, t, 'own_s'):.2f}</td><td>{fs(m, w, t, 'context'):.2f}</td><td>{fs(m, w, t, 'interaction'):.2f}</td></tr>")  # fmt: skip
    split = ("<table><tr><th>model</th><th>copy</th><th>term</th><th>constant</th><th>copied operand</th><th>of which its mod 20</th>"
             "<th>context (other operand, operation)</th><th>interaction</th></tr>" + srows + "</table>")  # fmt: skip
    return f"""<h3>14. The plane at "=" after the copy, and where its content comes from</h3>
<p>The stream at "=" after the copying attention, <i>E′</i>, is the sum of the stream before it, <i>E</i>, and of the attention's
whole write, <i>W</i> (all heads, all source positions): <i>E′</i> = <i>E</i> + <i>W</i>. Figures 22–25 find the mod-20 plane <i>P′</i> of
<i>E′</i> (the frequency-1 plane of the copied operand <i>o</i> mod 20, as in section 0.2, computed from <i>E′</i>'s residue means) and
project all three on it, in the order <i>E</i>, <i>W</i>, <i>E′</i>: each prompt's point in the third panel is the sum of its points in
the first two. Each operand value appears in 20 prompts (ten values of the other operand, two operations), which share a colour
but not a position in the first panel. Points are coloured by the raw value
of the copied operand <i>o</i> (1–400); labels are the residue means of <i>s</i> = <i>o</i> mod 20; the cross is the activation zero.
The prompt sets are those of section 11 (<code>mod20_eq_planes.py</code>).</p>
<figure><img src="fig/mod20_eq_planes_alive_a.png"><figcaption>Figure 22. Components model, copy of <i>a</i> by block 16: the
stream at "=" before the attention (<i>E</i>), the attention's write (<i>W</i>), and the stream after it (<i>E′</i>), all projected on the
mod-20 plane of <i>E′</i>, turned so that the vector from residue 10's mean to residue 0's mean points right. The three panels
share their axes.</figcaption></figure>
<figure><img src="fig/mod20_eq_planes_alive_b.png"><figcaption>Figure 23. As figure 22, components model, copy of <i>b</i> by
block 15.</figcaption></figure>
<figure><img src="fig/mod20_eq_planes_orig_a.png"><figcaption>Figure 24. As figure 22, original model, copy of <i>a</i> by
block 16.</figcaption></figure>
<figure><img src="fig/mod20_eq_planes_orig_b.png"><figcaption>Figure 25. As figure 22, original model, copy of <i>b</i> by
block 15.</figcaption></figure>
<p>In each panel, ‖Ψ<sub>1</sub>‖ is the norm of the frequency-1 coefficient of the 20 residue means in the plane (the size of the
circle), "spread" the root-mean-square distance of the points to their own mean, and ‖mean‖ the distance of that mean from
the activation zero:</p>
{table}
<p>The circle comes from the write: ‖Ψ<sub>1</sub>‖ of the stream before the attention is {st("alive", "a", "before", "psi1_norm_in_plane"):.2f}
and {st("alive", "b", "before", "psi1_norm_in_plane"):.2f} (components, <i>a</i> and <i>b</i>) against {st("alive", "a", "write", "psi1_norm_in_plane"):.2f} and
{st("alive", "b", "write", "psi1_norm_in_plane"):.2f} in the write. In the original model the stream before the attention already
holds a small circle ({st("orig", "a", "before", "psi1_norm_in_plane"):.2f} and {st("orig", "b", "before", "psi1_norm_in_plane"):.2f}, against
{st("orig", "a", "write", "psi1_norm_in_plane"):.2f} and {st("orig", "b", "write", "psi1_norm_in_plane"):.2f} in the write). What the stream already holds in the plane is a shift and a spread that
do not follow <i>s</i>: for <i>b</i> its spread is {st("alive", "b", "before", "rms_spread"):.2f} (components) and
{st("orig", "b", "before", "rms_spread"):.2f} (neurons), as large as the write's or larger.</p>
<p><b>What that content is.</b> The split below (<code>mod20_eq_origin.py</code>) takes each term's coordinates on the three clump
axes of section 13 and divides their energy (Σ over prompts of the squared coordinates, uncentred) into a constant, a part that
depends on the copied operand's value (and, within it, the part that depends on its residue mod 20 only), a part that depends
on the context (the other operand and the operation), and the rest. In the components model the prompt sets are complete
crosses of operand and context, and the shares add to 1. The original model's 2,000 prompts are not, so its shares overlap and
can add to more than 1.</p>
{split}
<p>Before the copy, "=" holds almost nothing of the copied operand's mod 20 (at most {max(fs(m, w, "E", "own_s") for m in ("alive",) for w in "ab"):.3f}
of its energy in the components model). It holds a constant and the context. For the copy of <i>a</i>, about half of the context
part is the operation; for the copy of <i>b</i>, it is mostly the size of <i>a</i> (the operation explains 3%). In the components model
the earlier sublayers that write the context part are mainly the MLPs and attention of blocks 12–15: block 15's MLP (32%) and the
attention of blocks 14 and 15 (15%, 13%) for <i>a</i>; the MLPs of blocks 12–14 and block 13's attention (11–25% each) for <i>b</i>.
The constant is the sum of large, partly cancelling writes of many blocks.</p>"""


def page(res, rings, counts, scan, pr) -> str:
    b12 = json.loads((OUT / "b12_fig.json").read_text())
    geo = json.loads((OUT / "geometry.json").read_text())
    gx, go, gv = geo["x"], geo["x_orig"], geo["x_V"]
    cp, npf = b12["components_profiles"], b12["neurons_profiles"]

    def inpl(key: str, name: str) -> float:
        return next(d["in_plane"] for d in b12[key] if d["reader"] == name)

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    def bd(m: str, b: int) -> dict:
        return res[m]["boundaries"][f"pre_{b}"]

    def sl(m: str, b: int, k: str = "mlp") -> dict:
        return res[m]["sublayers"][f"{k}_{b}"]

    def f2(x: float) -> str:
        return f"{x:.2f}"

    def rng(vals) -> str:
        v = list(vals)
        if min(v) < 0:
            return f"{min(v):.2f} to {max(v):.2f}".replace("-", "−")
        if f"{min(v):.2f}" == f"{max(v):.2f}":
            return f"{min(v):.2f}"
        return f"{min(v):.2f}–{max(v):.2f}"

    hj = json.loads((OUT / "harmonics.json").read_text())
    vj = json.loads((OUT / "vs_mod10.json").read_text())

    def hr(m: str, j: int, bs) -> list[float]:
        return [hj[m]["r2"][j - 1][b] for b in bs]

    def hrng(m: str, j: int, bs) -> str:
        return rng(max(v, 0.0) for v in hr(m, j, bs))

    def vc(m: str, k: int) -> list[float]:
        return [c[k] for c in vj[m]["cos"]]

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
    c3p = Path(__file__).resolve().parent / "notebook_mod20" / "mod20_clumps3d.json"
    sec12_html = sec12(json.loads(c3p.read_text())) if c3p.exists() else ""
    c3a = Path(__file__).resolve().parent / "notebook_mod20" / "mod20_clumps3d_after.json"
    sec13_html = sec13(json.loads(c3a.read_text())) if c3a.exists() else ""
    eqp_path, eo_path = OUT / "eq_planes.json", OUT / "eq_origin.json"
    sec14_html = (sec14(json.loads(eqp_path.read_text()), json.loads(eo_path.read_text()))
                  if eqp_path.exists() and eo_path.exists() else "")  # fmt: skip
    cpy = OUT / "copy.json"
    sec11_html = sec11(json.loads(cpy.read_text()), f2, rng) if cpy.exists() else ""
    cpath = OUT / "components.json"
    comp = json.loads(cpath.read_text()) if cpath.exists() else readers_only(res, scan)
    sec4_html = sec4(res, comp, rings, pr, att_gain, f2, rng, sl, bd, mech, alpha)
    return f"""
<p><b>Summary.</b> At the token <i>a</i>, the information that tells <i>a</i> mod 20 apart from <i>a</i> mod 10, that is,
whether the tens digit is even or odd, sits in a single plane of the residual stream: the frequency-1 Fourier plane of
<i>a</i> mod 20. Up to block 12, no other harmonic that separates <i>s</i> from <i>s</i> + 10 can be read linearly from
the MLP input on held-out values of <i>a</i>; harmonic 3 (period 20/3) becomes readable from block 13, when the ring has
turned into four clumps. In that plane the 20 residues lie on a ring around the activation zero, in order. At the
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
<li><b>Components</b>: the alive-only model of section 1.3 of the mod-10 notebook (every alive attention and MLP
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
imaginary unit; frequency (harmonic) <i>j</i> of <i>a</i> mod 20 has period 20/<i>j</i> in <i>a</i>, and for even <i>j</i> it is
harmonic <i>k</i> = <i>j</i>/2 of <i>a</i> mod 10, of the same period 10/<i>k</i>. The odd frequencies <i>j</i> = 1, 3, 5, 7, 9 are
exactly the ones in which residues <i>s</i> and <i>s</i> + 10 differ, so they hold everything <i>a</i> mod 20 adds to
<i>a</i> mod 10 (the even frequencies are the <i>a</i> mod 10 representation of the mod-10 page). null<sub><i>j</i></sub> is the 99th
percentile of ‖Ψ<sub><i>j</i></sub>‖ over 100 random relabellings of the 400 values of <i>a</i>. <i>P</i> (a 4096 × 2 matrix with
orthonormal columns) spans Re Ψ<sub>1</sub> and Im Ψ<sub>1</sub>: the frequency-1 plane of <i>a</i> mod 20.
<i>z</i>(<i>a</i>) = <i>P</i><sup>T</sup><i>X</i>(<i>a</i>) ∈ ℝ<sup>2</sup> and <i>m<sub>s</sub></i> = <i>P</i><sup>T</sup><i>μ<sub>s</sub></i> are
a prompt's and a residue's coordinates in it, and Π<sub><i>P</i></sub> = <i>PP</i><sup>T</sup> is the projection onto it.</p>
<p>Scripts: <code>track_build_orig.py</code>, <code>mod20_track.py</code>, <code>mod20_readers.py</code>,
<code>mod20_b12_fig.py</code>, <code>mod20_geometry.py</code>, <code>mod20_page.py</code>; data in
<code>&lt;run&gt;/analysis/reader_writer/track_mod20/</code>.</p>
<p><b>Ring and cloud measures</b> (used from chapter 1 on; illustrated in chapter 2).
<i>o<sub>s</sub></i> = (<i>m<sub>s</sub></i> − <i>m</i><sub><i>s</i>+10</sub>)/2 ∈ ℝ<sup>2</sup> for <i>s</i> = 0–9, and <i>o</i><sub><i>s</i>+10</sub> = −<i>o<sub>s</sub></i>;
<i>φ<sub>s</sub></i> is its angle, unwrapped along <i>s</i>, and the step from <i>s</i> to <i>s</i> + 1 is <i>φ</i><sub><i>s</i>+1</sub> − <i>φ<sub>s</sub></i>
in degrees (18° on an evenly spaced ring). The quantization index <i>α</i> ∈ [0, 1] minimises the squared angular error of
<i>φ<sub>s</sub></i> ≈ <i>c</i> + (2π/20)(<i>α s</i> + (1 − <i>α</i>)·5⌊<i>s</i>/5⌋) over <i>α</i> and an offset angle <i>c</i>: 1 is the even ring, 0 is
four points, one per quarter. In the plane, <i>c<sub>s</sub></i> is the mean of <i>z</i> over residue <i>s</i> and <i>sp<sub>s</sub></i> the
root-mean-square distance of those <i>z</i> to <i>c<sub>s</sub></i>; the separation of residues <i>s</i>, <i>s′</i> is
‖<i>c<sub>s</sub></i> − <i>c<sub>s′</sub></i>‖/√((<i>sp<sub>s</sub></i>² + <i>sp<sub>s′</sub></i>²)/2) (the "antipodal" separation is that of <i>s</i> and
<i>s</i> + 10), and the same for the four quarters ("closest quarters": the smallest of their pairwise separations). Held-out
accuracies fit the plane and the class means on the <i>a</i> ≤ 200 half and test on the other half, and the reverse: <i>Q</i>
among 4 classes (chance 0.25), <i>s</i> among 20 (0.05), and <i>h</i> given <i>r</i> (choose between residues <i>r</i> and <i>r</i> + 10; 0.5).</p>

<h3>1. One plane carries what a mod 20 adds to a mod 10</h3>
<p><b>Which harmonics can be read.</b> Figures 1a and 1b ask, for every harmonic <i>j</i> = 1–10 of <i>a</i> mod 20 and the MLP
of every block <i>ℓ</i> = 0–16, whether the harmonic can be read linearly from what that MLP sees, on values of <i>a</i> the
read never saw. The MLP input is <i>x</i>(<i>a</i>) = RMSNorm(<i>X</i>(<i>a</i>)) ∈ ℝ<sup>4096</sup>, with <i>X</i> the stream after
block <i>ℓ</i>'s attention and the norm's learned gain included. A ridge regression without intercept predicts the two numbers
cos(2π<i>j a</i>/20) and sin(2π<i>j a</i>/20) from <i>x</i>(<i>a</i>) (at <i>j</i> = 10 the sine is 0 for every integer <i>a</i> and is
dropped). The 400 values of <i>a</i> are split into 5 folds of 80; each fold holds 4 of the 20 values of every residue
<i>s</i>. Each fold is predicted by a regression fitted on the other 320 values, so every <i>a</i> is predicted once, by a fit
that never saw it. The ridge penalty is chosen for each fold, block and harmonic by a 4-fold cross-validation inside the
320 training values, on a grid from 10<sup>−12</sup> to 10<sup>2</sup> times the largest squared singular value of the training
inputs. The held-out r² is 1 − Σ(prediction − target)² / Σ target², summed over the 400 predictions and the two targets. It is
1 for a perfect read and 0 for predicting 0, and it is negative when the predictions are worse than 0.</p>
<p>The rows are periods: harmonic <i>j</i> of <i>a</i> mod 20 has period 20/<i>j</i>. The even harmonics <i>j</i> = 2<i>k</i> are harmonic
<i>k</i> of <i>a</i> mod 10 (periods 10, 5, 10/3, 5/2 and 2), and they take the same value at <i>s</i> and <i>s</i> + 10. The odd
harmonics (periods 20, 20/3, 4, 20/7 and 20/9) are the only ones that tell <i>s</i> from <i>s</i> + 10.</p>
<ul>
<li>Period 20 (<i>j</i> = 1): r² = {f2(hr("alive", 1, [0])[0])} at block 0, then {hrng("alive", 1, range(1, 17))} (components)
and {hrng("orig", 1, range(1, 17))} (neurons) in blocks 1–16.</li>
<li>Periods 20/3, 20/7 and 20/9 (<i>j</i> = 3, 7, 9): r² ≤ {f2(max(0.0, max(max(hr(m, j, range(13))) for m in ("alive", "orig") for j in (3, 7, 9))))}
in blocks 0–12 in both models. <i>j</i> = 3 then rises: {", ".join(f2(max(v, 0.0)) for v in hr("alive", 3, range(13, 17)))} in blocks
13–16 (components) and {", ".join(f2(max(v, 0.0)) for v in hr("orig", 3, range(13, 17)))} (neurons). <i>j</i> = 7 and 9 stay at 0
in every block.</li>
<li>Period 4 (<i>j</i> = 5): {f2(hr("alive", 5, [0])[0])} at block 0, {hrng("alive", 5, range(1, 15))} (components) and
{hrng("orig", 5, range(1, 15))} (neurons) in blocks 1–14, and {hrng("alive", 5, [15, 16])} and {hrng("orig", 5, [15, 16])} in
blocks 15–16.</li>
<li>The mod-10 harmonics (even <i>j</i>): periods 10, 5 and 2 stay at {hrng("alive", 2, range(1, 17))}, {hrng("alive", 4, range(1, 17))}
and {hrng("alive", 10, range(1, 17))} (components) from block 1. Periods 10/3 and 5/2 are at {hrng("alive", 6, range(1, 12))} and
{hrng("alive", 8, range(1, 12))} in blocks 1–11 and reach {f2(hr("alive", 6, [16])[0])} and {f2(hr("alive", 8, [16])[0])} at block 16.
In the original model these two fall to minima of {f2(min(hr("orig", 6, range(17))))} and {f2(min(hr("orig", 8, range(17))))} in blocks 8–11
and recover to {f2(hr("orig", 6, [16])[0])} and {f2(hr("orig", 8, [16])[0])} at block 16.</li>
</ul>
<p>Up to block 12, then, all of what <i>a</i> mod 20 adds to <i>a</i> mod 10 that a linear read can recover on new values of
<i>a</i> is in the period-20 sinusoid, apart from a weak period-4 signal (r² ≤ {f2(max(hr("alive", 5, range(1, 13)) + hr("orig", 5, range(1, 13))))}
after block 0).</p>
<p><b>Does the penalty cause false negatives?</b> With 320 training values and 4096 input dimensions, an unpenalised least-squares
fit is not unique, so the smallest penalty on the grid (10<sup>−12</sup>) stands in for "no penalty" (the minimum-norm fit). On every
cell where the cross-validated r² exceeds 0.3, the minimum-norm fit gives the same held-out r² to within
{f2(max(abs(a - b) for m in ("alive", "orig") for ra, rb in zip(hj[m]["r2"], hj[m]["r2_minnorm"], strict=True) for a, b in zip(ra, rb, strict=True) if a > 0.3))}.
On the cells near 0, the minimum-norm fit is negative, down to {f2(min(min(r) for m in ("alive", "orig") for r in hj[m]["r2_minnorm"])).replace("-", "−")}:
it fits the training values exactly and predicts new values worse than 0. There the inner cross-validation picks the
largest penalty, which predicts close to 0.</p>
<figure><img src="fig/mod20_harmonics_alive.png"><figcaption>Figure 1a. Components model. Held-out r² of cos and
sin(2π<i>j a</i>/20) read linearly from the input of block <i>ℓ</i>'s MLP (columns) at the token <i>a</i>, for every harmonic
<i>j</i> of <i>a</i> mod 20 (rows, labelled by the period 20/<i>j</i>; for even <i>j</i> also by <i>k</i> = <i>j</i>/2, the harmonic of
<i>a</i> mod 10). Negative values are coloured as 0.</figcaption></figure>
<figure><img src="fig/mod20_harmonics_orig.png"><figcaption>Figure 1b. As figure 1a, original model.</figcaption></figure>

<p><b>The mod-20 plane and the mod-10 subspace.</b> At the MLP input of each block, <i>P</i> is the frequency-1 plane of
<i>a</i> mod 20, built from the residue means of <i>x</i> as in section 0.2. <i>D</i><sub>10</sub> (a 4096 × 9 matrix with
orthonormal columns) spans the real and imaginary parts of Φ<sub><i>k</i></sub> = Σ<sub><i>r</i>=0…9</sub> <i>ν<sub>r</sub></i>
exp(−2π i <i>k r</i>/10) for <i>k</i> = 1–5, where <i>ν<sub>r</sub></i> is the mean of <i>x</i> over the 40 values of <i>a</i> with
<i>a</i> mod 10 = <i>r</i>; Im Φ<sub>5</sub> = 0, so the span has 9 dimensions. <i>D</i><sub>10</sub> is the mod-10 representation. Neither
<i>P</i> nor <i>D</i><sub>10</sub> contains the mean of <i>x</i> over all <i>a</i>. The overlap between them is measured by
their two principal cosines: the singular values of <i>P</i><sup>T</sup><i>D</i><sub>10</sub>. They are 1 for a direction of
<i>P</i> that lies in <i>D</i><sub>10</sub> and 0 for one orthogonal to it.</p>
<p>The overlap is small (figure 1c). The largest cosine grows from {f2(vc("alive", 0)[0])} at block 0 to
{f2(max(vc("alive", 0)))} at block {max(range(17), key=lambda b: vc("alive", 0)[b])} in the components model, then falls to
{rng(vc("alive", 0)[14:])} in blocks 14–16. In the original model it peaks at {f2(max(vc("orig", 0)))}. The second cosine stays
at or below {f2(max(vc("alive", 1)))} (components) and {f2(max(vc("orig", 1)))} (neurons). A random 9-dimensional subspace gives
a largest cosine of {f2(vj["null_cos1_99"])} (99th percentile of 200 draws). Removing <i>D</i><sub>10</sub>, that is, replacing
<i>x</i> by <i>x</i> − <i>D</i><sub>10</sub><i>D</i><sub>10</sub><sup>T</sup><i>x</i>, keeps
{rng(vj["alive"]["ring_kept"])} (components) and {rng(vj["orig"]["ring_kept"])} (neurons) of the spread of the 20 residue means in
<i>P</i> (their sum of squared distances to their own average). Figures 1d and 1e show the clouds in <i>P</i> before and after
the removal: the ring and, from block 13, the four clumps remain in the bottom row.</p>
<figure><img src="fig/mod20_vs_mod10_overlap.png"><figcaption>Figure 1c. Principal cosines between the mod-20 plane
<i>P</i> and the 9-dimensional mod-10 subspace <i>D</i><sub>10</sub> at the input of each block's MLP. Solid: components model;
dashed: original model. Dotted: the largest cosine of <i>P</i> with a random 9-dimensional subspace, 99th
percentile.</figcaption></figure>
<figure><img src="fig/mod20_vs_mod10_alive.png"><figcaption>Figure 1d. Components model. The 400 prompts at the input of
six blocks' MLPs projected on <i>P</i> (top: <i>P</i><sup>T</sup><i>x</i>) and with the mod-10 subspace removed first (bottom:
<i>P</i><sup>T</sup>(<i>x</i> − <i>D</i><sub>10</sub><i>D</i><sub>10</sub><sup>T</sup><i>x</i>)). Colour and numbers: <i>s</i> = <i>a</i> mod 20,
each number at its residue's mean. Same axes in both rows; the origin is the activation zero.</figcaption></figure>
<figure><img src="fig/mod20_vs_mod10_orig.png"><figcaption>Figure 1e. As figure 1d, original model.</figcaption></figure>

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

<h3>2. Geometry: a ring of the 20 residues, which turns into four clumps</h3>
<p>Figure 2 shows the 400 prompts in the plane at several blocks. The residues sit on a ring around the activation zero,
in the order 0, 1, …, 19, so that <i>s</i> and <i>s</i> + 10 are on opposite sides. Roundness (minor over major axis of the
ellipse traced by Re Ψ<sub>1</sub>, Im Ψ<sub>1</sub>) is {f2(bd("alive", 0)["roundness"])} at the embedding.</p>
<figure><img src="fig/mod20_planes.png"><figcaption>Figure 2. Residual stream at position 1 projected onto the
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
<figure><img src="fig/mod20_ring.png"><figcaption>Figure 3. Left and centre: the angular step <i>φ</i><sub><i>s</i>+1</sub> −
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
<figure><img src="fig/mod20_readers.png"><figcaption>Figure 4. Top: number of readers per block with odd share ≥
0.05 and best step r² ≥ 0.8, coloured by their best edge (left: kept gate and up components; right: all 28,672 gate and up
neurons). Bottom: mean activation over <i>s</i> of matched readers, each divided by its largest absolute value (solid:
component, dashed: neuron). Grey lines mark the quarter edges.</figcaption></figure>

{sec4_html}

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
<figure><img src="fig/mod20_clouds.png"><figcaption>Figure 7. Per boundary (solid: components, dashed: neurons).
(a) η<sub><i>s</i></sub> and η<sub><i>Q</i></sub>. (b) Median antipodal separation and closest pair of quarters. (c) Held-out nearest-mean
accuracies. (d) Held-out probe accuracy for <i>h</i> from the whole stream, the plane only, and the stream minus the plane.</figcaption></figure>

<h3>6. Block by block</h3>
<p>Columns per model: MLP gain, share from <i>P</i>, r²<sub>lin</sub>, r²<sub>gated</sub>, secant, gated secant and additivity of
the block's MLP (section 4); then <i>α</i>, held-out quarter accuracy and median antipodal separation of the stream leaving the
block.</p>
{table}

<h3>7. Components against neurons</h3>
<ul>
<li>One plane, the frequency-1 plane of <i>a</i> mod 20, holds the period-20 information up to block 12: both models (figures 1a–1b,
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
<h3>8. Block 12 in the planes: components</h3>
<p>Block 12's MLP turns the ring into quarters (section 2). The components model's MLP has the same 14,336 neurons as
the original (indices 0–14335); only the gate, up and down matrices are replaced by sums of components. In the components
model, two neurons carry the step, and each is fed by a few components (<code>mod20_b12_fig.py</code>). A component reader feeds the 14,336 neurons through its write row
<i>U</i><sub>up,<i>c</i></sub> or <i>U</i><sub>gate,<i>c</i></sub> ∈ ℝ<sup>14336</sup>; a neuron <i>n</i>'s gate pre-activation is
gate<sub><i>n</i></sub> = Σ<sub><i>c</i></sub> <i>t<sub>c</sub></i> <i>U</i><sub>gate,<i>c</i></sub>[<i>n</i>] over gate components, its up
pre-activation up<sub><i>n</i></sub> the same over up components, and its <b>write coefficient</b> is silu(gate<sub><i>n</i></sub>)·up<sub><i>n</i></sub>
(silu(<i>x</i>) = <i>x</i>/(1 + e<sup>−<i>x</i></sup>)). The read vector of a reader in residual-stream coordinates is
<i>ρ</i> = <i>γ</i> ⊙ <i>w</i>, with <i>γ</i> ∈ ℝ<sup>4096</sup> block 12's post-attention RMSNorm gain, ⊙ the elementwise product and
<i>w</i> the reader's read vector, so that its activation is <i>ρ</i> · <i>X</i>/rms(<i>X</i>). Its <b>in-plane</b> value is
‖<i>P</i><sup>T</sup><i>ρ</i>‖/‖<i>ρ</i>‖ (1 when <i>ρ</i> lies in the plane), and the arrows in figure 8 are <i>P</i><sup>T</sup><i>ρ</i>/‖<i>ρ</i>‖ drawn
from the activation zero. <i>P</i><sub>10</sub> is the frequency-1 plane of <i>a</i> mod 10 of the same stream.</p>
<ul>
<li><b>Neuron 5212, the h × [r ≥ 5] product.</b> Its up input is almost only component u57
(<i>U</i><sub>up,57</sub>[5212] = {fmt(b12["U_up_57_5212"])}; u57's next largest write to any neuron is {fmt(b12["U_up_57_next"])}).
u57 reads the ring (in-plane {fmt(inpl("components_P20", "u57"))}): its contribution to up<sub>5212</sub> is
{rng(cp["u57->5212"][1:4])} for <i>s</i> = 1–3 and {rng(cp["u57->5212"][15:18])} for <i>s</i> = 15–17. Its gate is set by
components that read the units digit, not the ring (in-plane in <i>P</i>: g37 {fmt(inpl("components_P20", "g37"))}, g33
{fmt(inpl("components_P20", "g33"))}; in <i>P</i><sub>10</sub>: {fmt(inpl("components_P10", "g37"))} and
{fmt(inpl("components_P10", "g33"))}). g37 lowers gate<sub>5212</sub> by {rng(cp["g37->5212"][2:5])} for <i>r</i> = 2–4, and g33
raises it by {rng(cp["g33->5212"][5:8])} for <i>r</i> = 5–7. silu(gate<sub>5212</sub>) is {rng(cp["gate5212"][5:10] + cp["gate5212"][15:20])}
for <i>r</i> = 5–9 and {rng(cp["gate5212"][0:5] + cp["gate5212"][10:15])} for <i>r</i> = 0–4. The write coefficient is
{rng(cp["w5212"][5:10])} for <i>s</i> = 5–9, {rng(cp["w5212"][15:20])} for <i>s</i> = 15–19 and between
{min(cp["w5212"][0:5] + cp["w5212"][10:15]):.2f} and {max(cp["w5212"][0:5] + cp["w5212"][10:15]):.2f} elsewhere: a sign set by the
ring's <i>h</i> side, nonzero only when <i>r</i> ≥ 5.</li>
<li><b>Neuron 8953, the quarter s = 10–14.</b> Its gate is mostly component g11 (<i>U</i><sub>gate,11</sub>[8953] =
{fmt(b12["U_gate_11_8953"])}; next largest {fmt(b12["U_gate_11_next"])}), which reads the ring (in-plane {fmt(inpl("components_P20", "g11"))})
and contributes {rng(cp["g11->8953"][10:15])} to gate<sub>8953</sub> for <i>s</i> = 10–14. Its up input is nearly constant
({rng(cp["up8953"])}, mostly component u1, in-plane {fmt(inpl("components_P20", "u1"))}). The write coefficient is
{rng(cp["w8953"][10:15])} for <i>s</i> = 10–14 and {rng(cp["w8953"][0:10] + cp["w8953"][15:20])} elsewhere.</li>
</ul>
<figure><img src="fig/mod20_b12_components.png"><figcaption>Figure 8. Components model, block 12's MLP input. (a) The
<i>a</i> mod 20 plane <i>P</i> (residues coloured by quarter, as in figure 2; the vector from residue 10's mean to residue 0's mean
points right) with the read directions of the components that feed neurons 5212 and 8953; the legend gives each one's
in-plane value. (b) The same readers in the <i>a</i> mod 10 frequency-1 plane <i>P</i><sub>10</sub> (blue <i>r</i> = 0–4, orange
<i>r</i> = 5–9; residue 0's mean points right). An arrow gives the sign of the component's activation; g37 enters gate<sub>5212</sub>
with a negative weight. (c) Per residue <i>s</i>: each component's contribution to neuron 5212's gate or up pre-activation, and
the neuron's write coefficient. (d) The same for neuron 8953.</figcaption></figure>

<h3>9. Block 12 in the planes: neurons</h3>
<p>The same two neurons of the original model, with quantities defined as in section 8 (here <i>w</i> is the row of
<code>gate_proj</code> or <code>up_proj</code>). <b>Neuron 5212</b>: the up row reads the ring (in-plane
{fmt(inpl("neurons_P20", "up 5212"))}; up<sub>5212</sub> is {rng(npf["up5212"][1:9])} for <i>s</i> = 1–8 and {rng(npf["up5212"][13:20])}
for <i>s</i> = 13–19). The gate row reads the units digit (in-plane in <i>P</i> {fmt(inpl("neurons_P20", "gate 5212"))}, in
<i>P</i><sub>10</sub> {fmt(inpl("neurons_P10", "gate 5212"))}): silu(gate<sub>5212</sub>) is {rng(npf["gate5212"][5:10] + npf["gate5212"][15:20])}
for <i>r</i> = 5–9 and {rng(npf["gate5212"][0:5] + npf["gate5212"][10:15])} for <i>r</i> = 0–4. The write coefficient is
{rng(npf["w5212"][5:10])} for <i>s</i> = 5–9, {rng(npf["w5212"][15:20])} for <i>s</i> = 15–19, and between
{min(npf["w5212"][0:5] + npf["w5212"][10:15]):.2f} and {max(npf["w5212"][0:5] + npf["w5212"][10:15]):.2f} elsewhere. <b>Neuron 8953</b>:
the gate row reads the ring (in-plane {fmt(inpl("neurons_P20", "gate 8953"))}) and silu(gate<sub>8953</sub>) is
{rng(npf["gate8953"][10:15])} for <i>s</i> = 10–14 against {rng(npf["gate8953"][0:10] + npf["gate8953"][15:20])} elsewhere; the up row
is nearly constant ({rng(npf["up8953"])}, in-plane {fmt(inpl("neurons_P20", "up 8953"))}). Its write coefficient is
{rng(npf["w8953"][10:15])} for <i>s</i> = 10–14 and {rng(npf["w8953"][0:10] + npf["w8953"][15:20])} elsewhere. The components model
routes the same two neurons through u57 (up 5212), g37 and g33 (gate 5212), g11 (gate 8953) and u1 (up 8953) (section 8).</p>
<figure><img src="fig/mod20_b12_neurons.png"><figcaption>Figure 9. Original model, block 12's MLP input, as figure 8 with the
gate and up rows of neurons 5212 and 8953 as readers. (c), (d): per residue <i>s</i>, the neuron's up pre-activation,
silu of its gate pre-activation, and their product (the write coefficient).</figcaption></figure>
<h3>10. The 20 residue means in the full MLP input: cosines and Cohen's d</h3>
<p>Sections 1 and 2 look only at the plane <i>P</i>. This section looks at the whole residue means, in the same way as the
mod-10 geometry page (<code>mod20_geometry.py</code>). <i>y</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is block <i>ℓ</i>'s MLP input (the
post-attention RMSNorm output) in one of three spaces: <i>x</i>, the full input of the components model; <i>x</i><sub>orig</sub>,
the full input of the original model; <i>x<sub>V</sub></i>, <i>x</i> projected onto the span of the block's kept readers' read
vectors (the space of the mod-10 geometry page). <i>ν<sub>s</sub></i> ∈ ℝ<sup>4096</sup> is the mean of <i>y</i> over the 20 operands with
<i>a</i> mod 20 = <i>s</i> (the residue's direction; nothing is centred). cos[<i>s</i>, <i>s′</i>] = <i>ν<sub>s</sub></i> · <i>ν<sub>s′</sub></i>/(‖<i>ν<sub>s</sub></i>‖ ‖<i>ν<sub>s′</sub></i>‖).
Cohen's d of residues <i>s</i>, <i>s′</i> projects both clouds of 20 operands onto <i>u</i> = (<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>)/‖<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>‖ and
divides ‖<i>ν<sub>s</sub></i> − <i>ν<sub>s′</sub></i>‖ by the pooled within-cloud standard deviation of the projections (unbiased variances, averaged
over the two clouds). Pairs are grouped as <b>antipodal</b> (<i>s′</i> = <i>s</i> + 10: same <i>r</i>, other <i>h</i>), <b>lag 1</b>
(<i>s′</i> = <i>s</i> ± 1 mod 20) and <b>other</b>.</p>
<p><b>Cosines.</b> The antipodal pairs are the most similar pairs at every block in <i>x</i> and <i>x</i><sub>orig</sub>: mean
cosine {gx["0"]["cos_mean"]["antipodal"]:.3f} at block 0 (lag 1 {gx["0"]["cos_mean"]["lag 1"]:.3f}, other
{gx["0"]["cos_mean"]["other"]:.3f}) and {rng(gx[b]["cos_mean"]["antipodal"] for b in gx if b != "0")} over blocks 1–16. From
block 1 on every cosine is at least {min(gx[b]["cos_min"] for b in gx if b != "0"):.2f}, because all 20 means share one large
direction; figure 10 gives each block its own colour range. The bright diagonal lines at <i>s′</i> = <i>s</i> ± 10 are the
mod-10 structure: residues that share <i>r</i> point almost the same way, and the part that tells <i>s</i> from <i>s</i> + 10 is small.
The original model gives the same picture ({rng(go[b]["cos_mean"]["antipodal"] for b in go if b != "0")} for antipodal pairs over
blocks 1–16).</p>
<p><b>Cohen's d.</b> The antipodal pairs are also the least separated pairs at every block, in all three spaces. In <i>x</i>,
their median d is {gx["0"]["d_median"]["antipodal"]:.1f} at block 0 (lag 1 {gx["0"]["d_median"]["lag 1"]:.1f}), falls to
{min(gx[b]["d_median"]["antipodal"] for b in gx):.1f} by block 11, and rises to {gx["15"]["d_median"]["antipodal"]:.1f} at
block 15 (lag 1 {gx["15"]["d_median"]["lag 1"]:.1f}). In <i>x</i><sub>orig</sub> the minimum over blocks is
{min(go[b]["d_median"]["antipodal"] for b in go):.1f} (block {min(go, key=lambda b: go[b]["d_median"]["antipodal"])}). In <i>x<sub>V</sub></i> the
separations are lower at several blocks (antipodal median {gv["1"]["d_median"]["antipodal"]:.1f} at block 1,
{gv["16"]["d_median"]["antipodal"]:.1f} at block 16, against {gx["1"]["d_median"]["antipodal"]:.1f} and
{gx["16"]["d_median"]["antipodal"]:.1f} in <i>x</i>): part of the difference between the residue clouds lies outside the span of
the kept readers.</p>
<figure><img src="fig/mod20_geom_cos_x.png"><figcaption>Figure 10. cos[<i>s</i>, <i>s′</i>] between the 20 residue means of the
components model's full MLP input, blocks 0–16 (row <i>s</i>, column <i>s′</i>; diagonal grey; white lines at the quarter
edges). Each panel has its own colour range. Titles: mean cosine of the antipodal and lag-1 pairs.</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x.png"><figcaption>Figure 11. Cohen's d between residue clouds <i>s</i> and <i>s′</i> along the
axis joining their means, components model, full MLP input; one logarithmic colour scale for all blocks. Titles: median d of
the antipodal and lag-1 pairs.</figcaption></figure>
<figure><img src="fig/mod20_geom_cos_x_orig.png"><figcaption>Figure 12. As figure 10, original model.</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x_orig.png"><figcaption>Figure 13. As figure 11, original model.</figcaption></figure>
<figure><img src="fig/mod20_geom_cos_x_V.png"><figcaption>Figure 14. As figure 10, in <i>x<sub>V</sub></i> (components model,
kept readers' span).</figcaption></figure>
<figure><img src="fig/mod20_geom_cohen_x_V.png"><figcaption>Figure 15. As figure 11, in <i>x<sub>V</sub></i>.</figcaption></figure>
{sec11_html}
{sec12_html}
{sec13_html}
{sec14_html}
"""


if __name__ == "__main__":
    main()
