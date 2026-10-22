"""Notebook page "Mod 10 representations at b (position 3)": comparison with position 1, figures, and the context
analysis (how the operation and a move the b mod 10 circle). Reads track_mod10/{analysis,residues,separability}.json
(position 1), track_mod10_p3/{analysis,L0}.json and stream.npz (position 3). Writes context.json, the figures
notebook/fig/p3_*.png and the page entry (id "mod10_p3") in notebook/entries.js. Nothing is centred.

    python p3_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from track_build import OUT as O1
from track_build_p3 import OUT as O3
from track_p3 import CTX, OP, RB, ST, basis, separable

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
CM = plt.get_cmap("tab10")
NAMES = [f"{t}_{b}" for b in range(17) for t in ("pre", "mid")] + ["pre_17"]


def context_analysis() -> dict:
    """Per boundary, in D coordinates: per-context residue means C[k, r]; their average over contexts;
    share of the between-context variation of C explained by one offset vector per context; op and a offsets."""
    ctxs = np.unique(CTX)
    op_of = np.asarray([OP[k == CTX][0] for k in ctxs])
    out = {}
    for n in NAMES:
        X = ST[n].astype(np.float64)
        _, D = basis(X)
        z = X @ D
        C = np.stack([[z[(r == RB) & (k == CTX)].mean(0) for r in range(10)] for k in ctxs])
        avg = C.mean(0)
        dev = C - avg[None]
        t = dev.mean(1, keepdims=True)
        out[n] = {
            "radius": float(np.linalg.norm(avg, axis=1).mean()),
            "offset_share": float(1 - ((dev - t) ** 2).sum() / (dev**2).sum())
            if (dev**2).sum() > 0
            else None,
            "op_offset": float(np.linalg.norm(t[op_of == 0, 0].mean(0) - t[op_of == 1, 0].mean(0))),
            "a_offset": float(
                np.mean(
                    [
                        np.linalg.norm(t[op_of == o, 0] - t[op_of == o, 0].mean(0), axis=1).mean()
                        for o in (0, 1)
                    ]
                )
            ),  # fmt: skip
        }
        if n.startswith("pre"):
            out[n]["per_context_ovr"] = [int(sum(separable(z[k == CTX], np.where(RB[k == CTX] == r, 1.0, -1.0), True)
                                                 for r in range(10))) for k in ctxs]  # fmt: skip
    return out


def figures(r1: dict, r3: dict, s1: dict, ctx: dict, fig_dir: Path) -> None:
    pre = [f"pre_{b}" for b in range(18)]
    x = np.arange(18)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for k, c in zip((0, 4), ("#2a5db0", "#e67e22"), strict=True):
        ax[0].plot(
            x,
            [r1["boundaries"][n]["amp"][k] for n in pre],
            "-o",
            ms=3,
            color=c,
            label=f"position 1, k = {k + 1}",
        )
        ax[0].plot(
            x,
            [r3["boundaries"][n]["amp"][k] for n in pre],
            "--s",
            ms=3,
            color=c,
            label=f"position 3, k = {k + 1}",
        )
    ax[0].plot(
        x,
        [r3["boundaries"][n]["a_amp"][0] for n in pre],
        ":^",
        ms=3,
        color="#c0392b",
        label="a mod 10 at position 3, k = 1",
    )
    ax[0].set_title("(a) ‖Ψ_k‖ of the operand's residue means")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(
        x, [r1["boundaries"][n].get("sep_min", 0) or 0 for n in pre], alpha=0
    )  # placeholder axis
    q1 = json.loads((O1 / "residues.json").read_text())["boundaries"]
    ax[1].plot(
        x,
        [q1[n]["sep_min"] for n in pre],
        "-o",
        ms=3,
        color="#1d1d1b",
        label="position 1 (a mod 10)",
    )
    ax[1].plot(
        x,
        [r3["boundaries"][n]["clouds"]["sep_min"] for n in pre],
        "--s",
        ms=3,
        color="#c0392b",
        label="position 3 (b mod 10)",
    )
    ax[1].set_title("(b) closest-pair separation in D")
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot(
        x,
        [sum(d["separable"] for d in s1[n]["D"]["affine"]["one_vs_rest"]) for n in pre],
        "-o",
        ms=3,
        color="#1d1d1b",
        label="position 1",
    )
    ax[2].plot(
        x,
        [r3["boundaries"][n]["sep_D_affine_ovr"] for n in pre],
        "--s",
        ms=3,
        color="#c0392b",
        label="position 3, all contexts pooled",
    )
    ax[2].plot(
        x,
        [min(ctx[n]["per_context_ovr"]) for n in pre],
        ":^",
        ms=3,
        color="#27ae60",
        label="position 3, worst single context",
    )
    ax[2].set_ylim(-0.5, 10.5)
    ax[2].set_title("(c) residues linearly separable from the other nine in D")
    ax[2].legend(frameon=False, fontsize=7)
    ax[3].plot(
        x,
        [ctx[n]["radius"] for n in pre],
        "-o",
        ms=3,
        color="#2a5db0",
        label="mean radius of the b circle",
    )
    ax[3].plot(
        x,
        [ctx[n]["op_offset"] for n in pre],
        "--s",
        ms=3,
        color="#c0392b",
        label="offset between + and − circles",
    )
    ax[3].plot(
        x,
        [ctx[n]["a_offset"] for n in pre],
        ":^",
        ms=3,
        color="#8e44ad",
        label="mean offset of an a-context",
    )
    ax[3].set_title("(d) position 3: the circle and its context offsets in D")
    ax[3].legend(frameon=False, fontsize=7)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([str(i) for i in range(18)], fontsize=7)
        a.set_xlabel("block (stream entering it; 17 = leaving block 16)")
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_overview.png")
    plt.close(fig)

    # the frequency-1 plane at position 3: pooled (coloured by b mod 10) and split by op
    show = (0, 2, 5, 8, 11, 13, 15, 17)
    fig, axes = plt.subplots(2, len(show), figsize=(2.6 * len(show), 5.6))
    for j, b in enumerate(show):
        X = ST[f"pre_{b}"].astype(np.float64)
        P = basis(X)[0][0]
        q = X @ P
        c0 = q[RB == 0].mean(0)
        ang = -np.arctan2(c0[1], c0[0])
        q = q @ np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]).T
        for r in range(10):
            sel = r == RB
            axes[0, j].scatter(q[sel, 0], q[sel, 1], s=1, color=CM(r), alpha=0.3, lw=0)
            axes[0, j].text(
                *q[sel].mean(0), str(r), fontsize=8, ha="center", va="center", weight="bold"
            )
        for o, c in ((0, "#2a5db0"), (1, "#c0392b")):
            sel = o == OP
            axes[1, j].scatter(
                q[sel, 0], q[sel, 1], s=1, color=c, alpha=0.25, lw=0, label="+" if o == 0 else "−"
            )
            for r in range(10):
                m = q[sel & (r == RB)].mean(0)
                axes[1, j].text(
                    m[0], m[1], str(r), fontsize=7, color=c, ha="center", va="center", weight="bold"
                )
        for a in axes[:, j]:
            a.plot([0], [0], "k+", ms=8)
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
        where = (
            "embedding" if b == 0 else ("leaving block 16" if b == 17 else f"entering block {b}")
        )
        axes[0, j].set_title(where, fontsize=8)
    axes[0, 0].set_ylabel("all prompts, colour = b mod 10", fontsize=8)
    axes[1, 0].set_ylabel("blue: a + b, red: a − b", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_k1.png", dpi=100)
    plt.close(fig)


def row(b: int, r1: dict, r3: dict, q1: dict, ctx: dict) -> str:
    def m(r: dict, key: str) -> str:
        mm = r["blocks"][str(b)]["mlp"]
        v = {
            "lin": mm["r2_lin"],
            "sec": mm["mechanism"]["r2_secant_sum"],
            "add": mm["mechanism"]["r2_sum_single"],
        }[key]
        return f"{v:.2f}"

    g1, g3 = r1["blocks"][str(b)]["mlp"]["out_gain"], r3["blocks"][str(b)]["mlp"]["out_gain"]
    a3 = r3["blocks"][str(b)]["attn"]["weight_by_position"]
    c3 = r3["boundaries"][f"pre_{b + 1}"]["clouds"]
    return (f"<tr><td>{b}</td><td>{min(g1):.2f}–{max(g1):.2f}</td><td>{min(g3):.2f}–{max(g3):.2f}</td>"
            f"<td>{m(r1, 'sec')} / {m(r3, 'sec')}</td><td>{m(r1, 'add')} / {m(r3, 'add')}</td>"
            f"<td>{q1[f'pre_{b + 1}']['sep_min']:.2f} / {c3['sep_min']:.2f}</td>"
            f"<td>{q1[f'pre_{b + 1}']['acc_loo']:.3f} / {c3['acc_loo']:.3f}</td>"
            f"<td>{r3['boundaries'][f'pre_{b + 1}']['sep_D_affine_ovr']} ({min(ctx[f'pre_{b + 1}']['per_context_ovr'])})</td>"
            f"<td>{ctx[f'pre_{b + 1}']['op_offset']:.2f} / {ctx[f'pre_{b + 1}']['radius']:.2f}</td>"
            f"<td>{a3[0]:.2f} / {a3[1]:.2f} / {a3[2]:.2f} / {a3[3]:.2f}</td></tr>")  # fmt: skip


def page(r1: dict, r3: dict, q1: dict, ctx: dict, l0: dict) -> str:
    rows = "".join(row(b, r1, r3, q1, ctx) for b in range(17))
    cos_min = min(min(c) for c in l0["cos_planes_p1_p3"])
    amp_ratio = [a3 / a1 for a1, a3 in zip(l0["amp_p1"], l0["amp_p3"], strict=True)]
    shares = [ctx[n]["offset_share"] for n in NAMES[4:] if ctx[n]["offset_share"] is not None]
    return f"""
<p><b>Summary.</b> At the token <i>b</i> (position 3), block 0 holds the same mod-10 representation as at the token
<i>a</i>: the same five Fourier planes (principal cosines ≥ {cos_min:.4f}), read by the same 11 readers, which feed the
same writers. Two things differ. (1) Its amplitude at the MLP input is {min(amp_ratio):.2f}–{max(amp_ratio):.2f} times
position 1's, because block-0 attention at position 3 adds a larger vector to the stream and the RMSNorm then divides
by a larger norm. (2) From block 0 on, attention brings the operation and <i>a</i> into position 3. Over blocks 2–10, attention and
the MLPs turn them into an offset of the whole <i>b</i> circle inside its own subspace, one offset per context (operation, <i>a</i>).
Pooled over contexts, the <i>b</i> clouds then overlap: by block 11 no residue is linearly separable from the others
(position 1: all ten at every block). Within a single context, at least 8 of the 10 still are, and all 10
up to block 7 and from block 13. The MLPs of blocks 12–14
shrink the operation offset and enlarge the circle, and separability comes back (9 of 10 from block 14).</p>

<h3>Setup</h3>
<p>Run p-ba8b0c08, step 40000, alive-only model, prompts <code>&lt;BOS&gt; a op b =</code> (tokenized without spaces, as
in the pool). The token <i>b</i> is the same token as the number <i>b</i> at position 1, so the two positions start from the
same embedding rows. Position 3 attends to &lt;BOS&gt;, <i>a</i>, the operation and itself, so its stream depends on
(<i>a</i>, op, <i>b</i>). The residue is <i>r</i> = <i>b</i> mod 10. Block-0 numbers use the stored activations of all 320,000 prompts
(<code>mod10_p3_L0.py</code>). The tracking through block 16 uses a float64 rebuild of positions 0–3 (<code>track_build_p3.py</code>;
max relative error against the stored activations 2.3%) on 8,000 prompts: the operands <i>a</i> = 7, 48, 89, 130, 171, 212,
253, 294, 335, 376 (one per residue of <i>a</i>), every <i>b</i> = 1 … 400 and both operations; a <b>context</b> is one
(operation, <i>a</i>) pair, 20 in all, and each residue class pools 800 prompts over the 20 contexts. All quantities are
defined as in the "Mod 10 representations" page (sections 1–8), with <i>a</i> mod 10 replaced by <i>b</i> mod 10. The
per-reader mechanism of each MLP runs on 1,600 of the prompts (<i>a</i> = 7 and 171). Nothing is centred.
Scripts: <code>track_p3.py</code>, <code>p3_page.py</code>.</p>

<h3>1. Block 0: the same representation, smaller</h3>
<ul>
<li><b>Same planes.</b> In the readers' span, the five planes of <i>b</i> mod 10 at position 3 and of <i>a</i> mod 10 at position 1
have principal cosines {cos_min:.4f}–1.0000. Circulant statistic: {l0["offdiag_p3"]:.3f} (position 1: {l0["offdiag_p1"]:.3f}).</li>
<li><b>Same readers.</b> The readers with r²<sub><i>D</i></sub> ≥ 0.5 are the same 11 (g69, g59, u60, g47, g58, u54, g62, u52, g43,
u794, u31), with r² {min(l0["r2_D_p3"][n] for n in l0["mod10_readers_p3"]):.2f}–{max(l0["r2_D_p3"][n] for n in l0["mod10_readers_p3"]):.2f}.
The position-1 subspace explains them equally well
({min(l0["r2_p3_explained_by_p1_subspace"].values()):.2f}–{max(l0["r2_p3_explained_by_p1_subspace"].values()):.2f}).
Their activation hardly depends on the context: the part that varies with (<i>a</i>, op) at fixed <i>b</i> holds
{max(v for k, v in l0["context_share_p3"].items() if k != "u31") * 100:.2f}% or less of Σ <i>f</i>² (u31: {l0["context_share_p3"]["u31"] * 100:.2f}%).</li>
<li><b>Same writers.</b> Each reader's main writer (variance split at position 3) is the same as at position 1, with
reader fractions 0.62–0.88: g43 → d43, g47 → d47, g58 → d58, g59 → d59, g62 → d62, g69 → d69, u52 → d52, u54 → d54,
u60 → d60, u794 → d41, u31 → d41.</li>
<li><b>Smaller amplitude, from the RMSNorm.</b> ‖Ψ<sub><i>k</i></sub>‖ in the readers' span is
{min(amp_ratio):.2f}–{max(amp_ratio):.2f} times position 1's. Block-0 attention adds a vector of mean norm 0.47 at position 3
against 0.22 at position 1, so the stream entering the MLP has mean norm 0.80 against 0.70. The same embedding structure is
therefore divided by a 1.16 times larger norm. Block-0 MLP's period-10 gain is correspondingly lower: 1.08–1.51 against
1.37–1.86.</li>
</ul>

<h3>2. Through blocks 0–16: what is the same</h3>
<ul>
<li>Attention never acts on the <i>b</i> representation itself (period-10 gain ≤ 0.07). Position 3 puts 79–93% of its attention
on &lt;BOS&gt;, 1–8% on <i>a</i>, 3–11% on the operation and 1–8% on itself.</li>
<li>The MLPs rewrite the representation as at position 1: their gains follow the same pattern (block 0, then blocks 3–4,
11–14), they are near-additive over readers (≥ 0.92 in blocks 0–12), and blocks 12–14 are the least linear (secant r²
0.80, 0.53, 0.67; additivity 0.92, 0.80, 0.81), as at position 1 (0.76, 0.51, 0.64).</li>
<li>The planes drift from the embedding's at the same pace (mean principal cosine leaving block 16: 0.23–0.37, against
0.20–0.34).</li>
</ul>

<h3>3. What differs: the context moves the <i>b</i> circle</h3>
<p><b><i>a</i> and the operation arrive at position 3.</b> The Fourier amplitude of the means over <i>a</i> mod 10 at position 3 is
0 in the embedding and grows from block 0's attention on (0.2 after block 0's attention, 1.2 after block 1's, 2.6 entering
block 3, 7.2 entering block 11, 9.0 leaving block 16),
larger than <i>b</i>'s own. At most 6% of it lies in <i>b</i>'s subspace <i>D</i>. The difference between the + and − means grows from 0.3
(after block-0 attention) to 4.9 entering block 11.</p>
<p><b>Inside <i>D</i>, the context translates the circle.</b> For each context <i>k</i>, let <i>C<sub>k,r</sub></i> be the mean of the
<i>D</i> coordinates over that context's prompts with <i>b</i> mod 10 = <i>r</i>. From block 2 to block 16, one offset vector
per context reproduces {min(shares):.2f}–{max(shares):.2f} of the variation of <i>C<sub>k,r</sub></i> across contexts. Each context
therefore has its own copy of the circle, shifted as a whole. Most of the shift is between the two operations
(panel d: the offset between the + and − circles reaches 1.10 entering block 11, against a circle radius of 0.89). The
shift builds up over blocks 2–10. Block-2 attention takes it from 0.04 to 0.16; the attention of blocks 4–9 adds 0.03–0.07
each; the MLPs add 0.02–0.07 each, except block 7's, which adds 0.36. The MLPs' period-10
output at position 3 is less tied to <i>D</i> than at position 1: in blocks 2–10, 0.77–1.00 of it comes from <i>D</i>
(position 1: 0.87–1.00).</p>
<p><b>Pooled clouds overlap; each context stays separable.</b> With the 20 contexts pooled, the closest-pair separation falls
to 0.73 entering block 11 (position 1: 0.95) and leave-one-out accuracy to 0.764 (position 1: 0.897). Linear separability
in <i>D</i>, which holds at every block at position 1, is lost: 9 residues separable entering block 5, 5 entering block 7, 0
entering block 11. Within a single context, the residues stay far more separable: the worst of the 20 contexts has all ten separable up to
block 7 and from block 13, and 8 or 9 in between (panel c, green).</p>
<p><b>Blocks 12–14 undo the shift.</b> The operation offset falls from 1.09 to 0.57 entering block 14 while the circle's
radius grows from 0.96 to 1.44. Separation returns to 2.52–3.07 entering blocks 14–15, and 9 of 10 residues are separable
again from block 14.</p>
<figure><img src="notebook/fig/p3_overview.png"><figcaption>(a) Fourier amplitudes ‖Ψ<sub><i>k</i></sub>‖ for <i>k</i> = 1, 5 of the operand's
residue means at position 1 (solid) and position 3 (dashed), and of <i>a</i> mod 10 at position 3 (dotted). (b) Closest-pair
separation of the residue clouds in <i>D</i>. (c) Number of residues linearly separable (affine) from the other nine in <i>D</i>:
position 1, position 3 with the 20 contexts pooled, and the worst single context at position 3. (d) Position 3: mean radius of
the <i>b</i> circle in <i>D</i> (averaged over contexts), norm of the offset between the + and − circles, and mean offset of an
<i>a</i>-context from its operation's average.</figcaption></figure>
<figure><img src="notebook/fig/p3_k1.png"><figcaption>Position 3, projected onto the frequency-1 plane of <i>b</i> mod 10 at each
boundary (axes turned so that residue 0's mean is on the positive horizontal axis; the cross is the activation zero). Top: all
8,000 prompts coloured by <i>b</i> mod 10, with the residue means. Bottom: the same points split by operation (blue +, red −), with
each operation's residue means.</figcaption></figure>

<h3>4. Block by block</h3>
<p>Each cell gives position 1 / position 3 where both are listed. MLP gain is the range over <i>k</i> = 1 … 5 of the MLP's period-10
output relative to its input. Separation and accuracy are measured on the stream leaving the block. "Separable" counts the
residues linearly separable (affine) from the other nine in <i>D</i> at position 3 with contexts pooled, and in brackets the
worst single context. "Op offset / radius" is position 3's offset between the + and − circles and the circle's radius in
<i>D</i>, leaving the block. The last column is position 3's mean attention weight on &lt;BOS&gt; / <i>a</i> / op / itself in the block.</p>
<div class="tw"><table><tr><th>block</th><th>MLP gain p1</th><th>MLP gain p3</th><th>secant r²</th><th>additivity r²</th>
<th>closest-pair sep</th><th>accuracy</th><th>separable p3 (worst context)</th><th>op offset / radius p3</th>
<th>attention weights p3</th></tr>{rows}</table></div>

<h3>5. Repository</h3>
<p>The representation repository now holds position 3 for blocks 0–11: the five block-0 planes <code>L0.p3.b_mod_10.k1</code>, …,
the residue-mean spans <code>L1.p3.b_mod_10.means</code> … <code>L11.p3.b_mod_10.means</code> with their circulant statistics, every
kept reader's position-3 response, the block-0 mod-10 readers' position-3 writers, and the transitions
<code>L0.p3.b_mod_10</code> … <code>L11.p3.b_mod_10</code> with this page's per-block measurements.</p>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r1 = json.loads((O1 / "analysis.json").read_text())
    r3 = json.loads((O3 / "analysis.json").read_text())
    q1 = json.loads((O1 / "residues.json").read_text())["boundaries"]
    s1 = json.loads((O1 / "separability.json").read_text())
    l0 = json.loads((O3 / "L0.json").read_text())
    cp = O3 / "context.json"
    if cp.exists():
        ctx = json.loads(cp.read_text())
    else:
        ctx = context_analysis()
        cp.write_text(json.dumps(ctx, indent=1))
    figures(r1, r3, s1, ctx, args.out / "fig")
    html = page(r1, r3, q1, ctx, l0)
    p = args.out / "entries.js"
    s = p.read_text()
    entry = (
        '{\nid: "mod10_p3",\ntitle: "Mod 10 representations at b (position 3)",\ndate: "2026-10-22",\nhtml: `'
        + html
        + "`\n},\n"
    )
    if 'id: "mod10_p3"' in s:
        i = s.index('{\nid: "mod10_p3"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)


if __name__ == "__main__":
    main()
