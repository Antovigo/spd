"""Section 8 of the notebook page "Mod 10 representations": the representation through blocks 0..16.

Reads track_analyze.py's analysis.json; writes notebook/fig/mod10_track_*.png and the section HTML between
the markers <!-- track:begin --> and <!-- track:end --> of notebook/entries.js (appended before the end of
the page's html if the markers are absent).

    python track_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from track_build import LAST, OUT

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
COL = ["#2a5db0", "#c0392b", "#27ae60", "#8e44ad", "#e67e22"]
BEGIN, END = "<!-- track:begin -->", "<!-- track:end -->"


def rng(v: list[float], d: int = 2) -> str:
    lo, hi = min(v), max(v)
    return f"{lo:.{d}f}" if f"{lo:.{d}f}" == f"{hi:.{d}f}" else f"{lo:.{d}f}–{hi:.{d}f}"


def cloud_figure(rq: dict, fig_dir: Path) -> None:
    names = [n for n in rq["boundaries"] if n.startswith("pre")]
    x = np.arange(len(names))
    cb = [rq["boundaries"][n] for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.4))
    ax[0].plot(x, [c["sep_min"] for c in cb], "-o", ms=3, color="#c0392b", label="closest pair")
    ax[0].plot(x, [c["sep_median"] for c in cb], "-o", ms=3, color="#1d1d1b", label="median pair")
    ax[0].set_title("(g) separation of residue clouds in D^X")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(
        x,
        [np.mean(c["radius"]) for c in cb],
        "-o",
        ms=3,
        color="#2a5db0",
        label="mean radius ‖c_r‖",
    )
    ax[1].plot(
        x, [np.mean(c["spread"]) for c in cb], "-o", ms=3, color="#e67e22", label="mean spread s_r"
    )
    ax[1].set_title("(h) radius and spread")
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot(x, [c["acc_loo"] for c in cb], "-o", ms=3, color="#1d1d1b")
    ax[2].set_ylim(0.85, 1.01)
    ax[2].set_title("(i) leave-one-out nearest-mean accuracy")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([n.split("_")[1] for n in names], fontsize=7)
        a.set_xlabel("block (residual stream entering it; 17 = leaving block 16)")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_track_clouds.png")
    plt.close(fig)


def figures(res: dict, fig_dir: Path) -> None:
    bnames = list(res["boundaries"])
    xb = np.arange(len(bnames))
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    for k in range(5):
        ax[0].plot(
            xb,
            [res["boundaries"][n]["amp"][k] for n in bnames],
            "-o",
            ms=2,
            color=COL[k],
            label=f"k = {k + 1}",
        )
        ax[1].plot(
            xb[1:],
            [np.mean(res["boundaries"][n]["cos_prev"][k]) for n in bnames[1:]],
            "-o",
            ms=2,
            color=COL[k],
        )
        ax[2].plot(
            xb,
            [np.mean(res["boundaries"][n]["cos_emb"][k]) for n in bnames],
            "-o",
            ms=2,
            color=COL[k],
        )
    ticks = [i for i, n in enumerate(bnames) if n.startswith("pre")]
    for a in ax:
        a.set_xticks(ticks)
        a.set_xticklabels([bnames[i].split("_")[1] for i in ticks], fontsize=7)
        a.set_xlabel("block (tick = residual stream entering the block)")
    ax[0].set_ylabel("‖Ψ_k‖")
    ax[0].set_title("(a) amplitude of each frequency")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_title("(b) mean principal cosine of P_k with the previous boundary")
    ax[2].set_title("(c) mean principal cosine of P_k with the embedding's P_k")
    ax[1].set_ylim(0, 1.02)
    ax[2].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_track_planes.png")
    plt.close(fig)

    blocks = [res["blocks"][str(b)] for b in range(LAST + 1)]
    xs = np.arange(LAST + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
    for k in range(5):
        ax[0].plot(
            xs,
            [b["mlp"]["out_gain"][k] for b in blocks],
            "-o",
            ms=3,
            color=COL[k],
            label=f"MLP, k = {k + 1}",
        )
        ax[0].plot(xs, [b["attn"]["out_gain"][k] for b in blocks], ":", color=COL[k], lw=1)
        ax[1].plot(
            xs,
            [b["mlp"]["out_share_from_D"][k] for b in blocks],
            "-o",
            ms=3,
            color=COL[k],
            label=f"k = {k + 1}",
        )
    ax[0].set_title("(d) ‖Ψ_k(S(X))‖ / ‖Ψ_k(X)‖ (solid: MLP, dotted: attention)")
    ax[0].legend(frameon=False, fontsize=7, ncol=2)
    ax[1].axhline(1, color="#999", lw=0.8)
    ax[1].set_title("(e) MLP: share of its period-10 output that comes from D^X")
    ax[2].plot(
        xs,
        [b["mlp"]["r2_lin"] for b in blocks],
        "-o",
        ms=3,
        color="#1d1d1b",
        label="MLP: Δ ≈ L z (r²_lin)",
    )
    ax[2].plot(xs, [b["mlp"]["mechanism"]["r2_secant_sum"] for b in blocks], "-s", ms=3, color="#2a5db0",
               label="MLP: Δ ≈ Σ_c κ_c dt_c (secant)")  # fmt: skip
    ax[2].plot(xs, [b["mlp"]["mechanism"]["r2_sum_single"] for b in blocks], "-^", ms=3, color="#27ae60",
               label="MLP: Δ ≈ Σ_c Δ_c (additivity)")  # fmt: skip
    ax[2].plot(
        xs,
        [b["attn"]["r2_lin"] for b in blocks],
        ":o",
        ms=3,
        color="#c0392b",
        label="attention: r²_lin",
    )
    ax[2].set_ylim(0, 1.02)
    ax[2].set_title("(f) linearity of what each sublayer does with D^X")
    ax[2].legend(frameon=False, fontsize=7, loc="lower left")
    for a in ax:
        a.set_xticks(xs)
        a.set_xlabel("block")
    fig.tight_layout()
    fig.savefig(fig_dir / "mod10_track_sublayers.png")
    plt.close(fig)


NOTES = {
    0: "The MLP is where the representation grows: frequency-k amplitude goes up 1.4–1.9 times, and the planes turn "
    "(mean cosine 0.62–0.78 with the embedding's). The step is linear (secant r² 0.93; section 7). Residue 0 is "
    "moved least: its write is 1.07 times its radius, against 1.74–2.07 for the other residues. The clouds widen faster "
    "than they move apart (spread 0.035 → 0.091, radius 0.157 → 0.381), so the closest pair (1, 2) drops from 5.93 to 4.83.",
    1: "The MLP writes little period-10 content (gain 0.04–0.10), and what it writes is linear in the readers (secant "
    "r² 0.91, additivity 1.00). The linear fit on z alone is lower (0.54) because Δ here is small (0.13 of the "
    "input's D norm). The planes do not move (cosine ≥ 0.99). Spread grows, and the smallest separation falls 4.83 → 3.59.",
    2: "Linear (secant r² 0.94). The MLP writes mostly to residues 0 and 1 (0.83 and 0.66 of their radius, against "
    "0.19–0.30 for the others), and roundness drops (k = 2: 0.90 → 0.77; k = 4: 0.57 → 0.43).",
    3: "Linear (secant r² 0.97). The largest write is to residue 0 (1.23 of its radius), then 2 and 5 (0.90, 0.98); "
    "the others get 0.57–0.85. Radius grows 0.52 → 0.78, and the closest pair rises slightly (2.58 → 2.73).",
    4: "Linear (0.97). Writes are spread over all residues (0.33–0.57 of their radius). Spread grows faster than radius "
    "(0.25 → 0.35 against 0.78 → 1.05), and the closest pair falls 2.72 → 2.22; accuracy stays 1.000.",
    5: "Linear (0.96). Uniform writes (0.36–0.49). Separation keeps falling (closest pair 2.22 → 1.70), and "
    "leave-one-out accuracy drops below 1 for the first time (0.985).",
    6: "Linear (0.99). Uniform writes (0.28–0.40); closest pair 1.70 → 1.36, accuracy 0.955.",
    7: "Linear (0.99). Residue 1 receives the largest write (0.97 of its radius, against 0.61–0.73); closest pair "
    "1.36 → 1.12, accuracy 0.917.",
    8: "Linear (0.98). Residue 1 again receives the largest write (0.48, against 0.30–0.37); closest pair 1.11 → 1.04, "
    "accuracy 0.907.",
    9: "Linear (0.99). Small, uniform writes (0.14–0.20); closest pair 1.04 → 1.01, accuracy 0.912.",
    10: "Linear (0.98). The smallest writes of all blocks (0.07–0.15), radius ratios 0.98–1.00. The closest pair reaches "
    "its lowest value (1.01 → 0.95), with accuracy 0.897.",
    11: "Still linear (secant 0.94, additivity 0.99), but the first block since block 0 with a large write (gain up to 0.81). "
    "Residue 1 is moved most (0.68 of its radius, against 0.19–0.38). Separation starts to recover: median 1.42 → 1.82, "
    "accuracy 0.897 → 0.945.",
    12: "The first block where the MLP's period-10 action is no longer a sum of independent linear reader paths: "
    "secant r² 0.76 and additivity 0.81, while the linear fit on z is still 0.92. The k = 3 plane turns most (cosine 0.70) "
    "and becomes round (0.76 → 0.98). The clouds tighten (spread 0.90 → 0.83) and separate (median 1.82 → 2.70; accuracy "
    "0.988). Residues 1 and 9 are moved most (0.49, 0.45, against 0.19–0.27).",
    13: "The most non-linear block: linear fit 0.70, secant 0.51, additivity 0.75; its main readers are less one-hot "
    "(0.12–0.59) and individually non-linear (secant r² 0.34–0.54). Planes k = 3, 4 turn (cosine 0.66, 0.65), and k = 4 "
    "becomes rounder (0.39 → 0.70). The clouds tighten (spread 0.83 → 0.67) and separate sharply: closest pair 1.45 → 3.06, "
    "median 2.70 → 4.21, accuracy 1.000. Residues 3, 7 and 8 are moved most (0.59, 0.65, 0.52, against 0.21–0.45).",
    14: "Non-linear (linear fit 0.73, secant 0.64, additivity 0.83). The writes split the residues into two groups: "
    "1, 2, 4, 6 and 9 receive 0.38–0.43 of their radius, and 0, 3, 5, 7 and 8 receive 0.17. Separation increases "
    "further (closest pair 3.06 → 4.12, median 4.22 → 5.09).",
    15: "Additive over readers again (0.97) but not linear per reader (secant 0.75, linear fit 0.74). Writes are "
    "small (0.08–0.20). Spread grows (0.64 → 0.71), and the closest pair falls slightly (4.12 → 3.89).",
    16: "Additive (0.95), secant 0.77, linear fit 0.79. Residue 1 is moved most (0.47, against 0.11–0.29). Spread "
    "grows (0.71 → 0.88) faster than radius (2.69 → 3.21), and the closest pair falls 3.88 → 3.52.",
}


def residue_rows(b: int, rq: dict) -> str:
    w = rq["sublayers"][f"pre_{b + 1}"]["write_into_output_rel"]
    rr = rq["sublayers"][f"pre_{b + 1}"]["radius_ratio"]
    c0, c1 = rq["boundaries"][f"mid_{b}"], rq["boundaries"][f"pre_{b + 1}"]
    head = "".join(f"<th>{r}</th>" for r in range(10))
    return f"""<div class="tw"><table><tr><th>MLP, per residue r</th>{head}</tr>
<tr><td>write / input radius</td>{"".join(f"<td>{v:.2f}</td>" for v in w)}</tr>
<tr><td>radius after / before</td>{"".join(f"<td>{v:.2f}</td>" for v in rr)}</tr>
<tr><td>spread after / mean spread</td>{"".join(f"<td>{v / np.mean(c1['spread']):.2f}</td>" for v in c1["spread"])}</tr>
</table></div>
<p>Clouds entering → leaving the MLP: closest pair sep {c0["sep_min"]:.2f} ({c0["sep_min_pair"][0]}, {c0["sep_min_pair"][1]}) →
{c1["sep_min"]:.2f} ({c1["sep_min_pair"][0]}, {c1["sep_min_pair"][1]}); median sep {c0["sep_median"]:.2f} → {c1["sep_median"]:.2f};
mean spread {np.mean(c0["spread"]):.3f} → {np.mean(c1["spread"]):.3f}; mean radius {np.mean(c0["radius"]):.3f} →
{np.mean(c1["radius"]):.3f}; leave-one-out accuracy {c0["acc_loo"]:.3f} → {c1["acc_loo"]:.3f}.</p>"""


def block_html(b: int, blk: dict, res: dict, rq: dict) -> str:
    at, ml = blk["attn"], blk["mlp"]
    mm = ml["mechanism"]
    pre, nxt = res["boundaries"][f"pre_{b}"], res["boundaries"][f"pre_{b + 1}"]
    top = mm["top90"]
    rows = "".join(
        f"<tr><td>{d['reader']}</td><td>{d['energy'] / sum(t['energy'] for t in top):.2f}</td>"
        f"<td>{d['main_residue']}</td><td>{d['onehot']:.2f}</td><td>{d['r2_secant']:.2f}</td></tr>"
        for d in top[:8]
    )
    more = f" ({len(top) - 8} more readers not listed)" if len(top) > 8 else ""
    heads = ", ".join(f"head {h} ({s:.2f})" for h, s in at["mechanism"]["top_heads"][:3])
    return f"""
<h4>Block {b}</h4>
<p>{NOTES[b]}</p>
<div class="tw"><table><tr><th></th><th>out gain ‖Ψ<sub>k</sub>(S)‖/‖Ψ<sub>k</sub>(X)‖, k = 1…5</th><th>share from D<sup>X</sup></th>
<th>Δ / ‖Π<sub>D</sub>X‖</th><th>r²<sub>lin</sub></th><th>mechanism</th></tr>
<tr><td>attention</td><td>{" / ".join(f"{v:.2f}" for v in at["out_gain"])}</td><td>{rng(at["out_share_from_D"])}</td>
<td>{at["delta_over_input_D"]:.3f}</td><td>{at["r2_lin"]:.2f}</td>
<td>value path r² {at["mechanism"]["r2_value_path"]:.2f}; weight on itself {at["mechanism"]["p_self_mean"]:.3f}</td></tr>
<tr><td>MLP</td><td>{" / ".join(f"{v:.2f}" for v in ml["out_gain"])}</td><td>{rng(ml["out_share_from_D"])}</td>
<td>{ml["delta_over_input_D"]:.3f}</td><td>{ml["r2_lin"]:.2f}</td>
<td>secant r² {mm["r2_secant_sum"]:.2f}; additivity r² {mm["r2_sum_single"]:.2f}</td></tr></table></div>
<p>Mean principal cosine of each plane (k = 1…5) across the attention:
{" / ".join(f"{np.mean(c):.2f}" for c in res["boundaries"][f"mid_{b}"]["cos_prev"])}; across the MLP:
{" / ".join(f"{np.mean(c):.2f}" for c in nxt["cos_prev"])}. Roundness (k = 1…4) entering the block:
{" / ".join(f"{v:.2f}" for v in pre["round"])}; leaving it: {" / ".join(f"{v:.2f}" for v in nxt["round"])}.
Largest attention heads in the value path: {heads}.</p>
<p>MLP readers carrying 90% of Σ<sub><i>c</i></sub> Σ<sub><i>a</i></sub>‖Δ<sub><i>c</i></sub>‖²{more}:</p>
<div class="tw"><table><tr><th>reader</th><th>share</th><th>main residue of dt<sub>c</sub></th><th>dt<sub>c</sub> one-hot share</th>
<th>secant r²</th></tr>{rows}</table></div>
{residue_rows(b, rq)}
<figure><img src="notebook/fig/mod10_track_b{b}.png"><figcaption>Block {b}: every prompt's coordinates in the five
planes P<sub>k</sub> of the residual stream entering the block (top) and leaving it (bottom), each in its own
boundary's planes, coloured by residue, with the residue means labelled. The cross is the activation zero. For k = 5 (a line)
the vertical axis is a.</figcaption></figure>"""


def section_html(res: dict, rq: dict) -> str:
    blocks = [res["blocks"][str(b)] for b in range(LAST + 1)]
    attn_gain = max(max(b["attn"]["out_gain"]) for b in blocks)
    bos_w = [1 - b["attn"]["mechanism"]["p_self_mean"] for b in blocks]
    intro = f"""
<h3>8. The representation through blocks 0–{LAST}</h3>
<p>Everything in this section is at position 1 (the token <i>a</i>), in the alive-only model. Position 1 attends
only to &lt;BOS&gt; and to itself, so the model was rebuilt in float64 from the checkpoint for those two
positions through block {LAST}. Every block's alive attention and MLP components were on, and the weight delta was off.
The rebuild matches the stored MLP reader and writer activations of every block to within 1.8% (max relative
error). Scripts: <code>track_build.py</code>, <code>track_analyze.py</code>, <code>track_page.py</code>.</p>
<ul>
<li><b>Boundaries and sublayers.</b> <i>X</i>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream at one boundary: entering block
<i>b</i>, or after its attention. A sublayer <i>S</i> is one block's attention or MLP, including its RMSNorm. It adds
<i>S</i>(<i>X</i>) to the stream.</li>
<li><b>Fourier planes.</b> Ψ<sub><i>k</i></sub>, <i>P<sub>k</sub></i> and roundness are as in sections 1 and 7, computed from
<i>X</i>'s residue means. <i>D<sup>X</sup></i> is the 9-dimensional span of <i>X</i>'s five planes, Π<sub><i>D</i></sub> the orthogonal
projection onto it, and <i>z</i>(<i>a</i>) ∈ ℝ<sup>9</sup> the coordinates of <i>X</i>(<i>a</i>) in an orthonormal basis of
<i>D<sup>X</sup></i>.</li>
<li><b>What a sublayer does with the representation.</b> Δ(<i>a</i>) = <i>S</i>(<i>X</i>)(<i>a</i>) − <i>S</i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>)(<i>a</i>).
<i>S</i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>) is what the sublayer writes when its input has no period-10 structure. Its own
period-10 content is therefore made from other features of <i>a</i>.
<b>Share from D<sup>X</sup></b> is Re⟨Ψ<sub><i>k</i></sub>(<i>S</i>(<i>X</i>)), Ψ<sub><i>k</i></sub>(Δ)⟩ / ‖Ψ<sub><i>k</i></sub>(<i>S</i>(<i>X</i>))‖², for
<i>k</i> = 1…5. It is 1 when the sublayer's period-10 output is entirely made from <i>D<sup>X</sup></i>, and 0 when none of it is.</li>
<li><b>Linearity tests.</b> r²<sub>lin</sub> is the uncentred r² of Δ ≈ <i>L z</i>, with one 4096 × 9 matrix <i>L</i> fitted by least squares
with no intercept, over the 400 prompts. For an MLP, the secant test generalises section 7. For each reader <i>c</i>,
dt<sub><i>c</i></sub> = <i>t<sub>c</sub></i>(<i>X</i>) − <i>t<sub>c</sub></i>(<i>X</i> − Π<sub><i>D</i></sub><i>X</i>). Δ<sub><i>c</i></sub> is the change in
the MLP's output when only reader <i>c</i>'s activation is changed that way, and κ<sub><i>c</i></sub> ∈ ℝ<sup>4096</sup> is the
least-squares secant of Δ<sub><i>c</i></sub> on dt<sub><i>c</i></sub>. "Secant r²" is the r² of Δ ≈ Σ<sub><i>c</i></sub> κ<sub><i>c</i></sub>
dt<sub><i>c</i></sub>, and "additivity r²" is the r² of Δ ≈ Σ<sub><i>c</i></sub> Δ<sub><i>c</i></sub>. For attention, the value
path keeps the attention pattern of <i>X</i> and takes the values from <i>X</i> − Π<sub><i>D</i></sub><i>X</i>.</li>
</ul>
<p><b>Attention does not act on the representation at position 1.</b> In every block, the attention output's
frequency-<i>k</i> amplitude is at most {attn_gain:.2f} times the input's. Averaged over heads and prompts, position 1 puts
{100 * min(bos_w):.1f}–{100 * max(bos_w):.1f}% of its attention weight on &lt;BOS&gt;.</p>
<p><b>Residue clouds.</b> At each boundary, <i>c<sub>r</sub></i> ∈ ℝ<sup>9</sup> is the mean of <i>z</i>(<i>a</i>) over the 40
values of <i>a</i> with <i>a</i> mod 10 = <i>r</i> (its norm is the residue's radius), and <i>s<sub>r</sub></i> is the root-mean-square
distance of those <i>z</i>(<i>a</i>) to <i>c<sub>r</sub></i> (the cloud's spread). The separation of two residues <i>r</i>, <i>r′</i> is
‖<i>c<sub>r</sub></i> − <i>c<sub>r′</sub></i>‖ / √((<i>s<sub>r</sub></i>² + <i>s<sub>r′</sub></i>²)/2). "Accuracy" is the
leave-one-out nearest-mean classification accuracy of <i>r</i> from <i>z</i>(<i>a</i>); chance is 0.1. Per residue, an MLP's
write is the mean of its output over the residue's prompts projected onto <i>D</i> of the stream leaving it, divided by
the residue's input radius ‖Π<sub><i>D</i></sub>μ<sub><i>r</i></sub>‖.</p>
<p><b>Summary of blocks 0–16.</b></p>
<ul>
<li><b>The MLPs' period-10 output comes from the representation itself.</b> In every block, 0.87–1.05 of each MLP's
frequency-<i>k</i> output comes from <i>D<sup>X</sup></i> (panel e). Little period-10 content is made from other features of
<i>a</i>.</li>
<li><b>Blocks 0–11 rewrite it linearly; blocks 12–16 do not.</b> In blocks 0–11, the MLP's action on
<i>D<sup>X</sup></i> is a sum of independent, linear reader paths (secant r² 0.91–0.99, additivity 0.94–1.00). In blocks
12–16, the secant r² falls to 0.51–0.77, and in blocks 12–14 additivity falls to 0.75–0.83: readers interact (panel f).</li>
<li><b>The clouds first blur, then sharpen.</b> From the embedding to the stream entering block 11, the clouds' spread
grows faster than their radius, and the closest-pair separation falls from 5.93 to 0.95 (accuracy 0.897). The MLPs of
blocks 11–14 (12–14 non-additive) tighten the clouds and push them apart: the closest pair reaches 4.12 and accuracy 1.000
(panels g–i). Blocks 15–16 widen the clouds again (closest pair 3.52).</li>
<li><b>The planes drift.</b> Each MLP turns the planes a little (cosines 0.62–1.00 per block). By the stream leaving
block 16, the mean principal cosine with the embedding's planes is 0.20–0.34 (panel c).</li>
<li><b>Some MLPs move a few residues more than others.</b> Residue 0 in blocks 2 and 3; residue 1 in blocks 7, 8, 11 and 16;
residues 1 and 9 in block 12; residues 3, 7 and 8 in block 13. In block 14, residues 1, 2, 4, 6 and 9 get more than twice
the write of 0, 3, 5, 7 and 8. Residue 0's cloud is the widest or second widest at every boundary (1.11–1.35 times the
mean spread), together with residue 1 up to block 14 and residue 5 after it.</li>
</ul>
<figure><img src="notebook/fig/mod10_track_clouds.png"><figcaption>Residue clouds in <i>D<sup>X</sup></i> at the residual
stream entering each block. (g) Closest-pair and median separation. (h) Mean radius ‖<i>c<sub>r</sub></i>‖ and mean spread
<i>s<sub>r</sub></i>. (i) Leave-one-out nearest-mean accuracy of the residue.</figcaption></figure>
<p><b>The blur and the sharpening, seen in one plane.</b> The figure below projects the stream onto its own
frequency-1 plane <i>P</i><sub>1</sub>, the plane that carries <i>a</i> mod 10 at one turn per ten residues. In that plane, the
closest-pair separation falls from 2.58 at the embedding to 0.19 entering block 11, and the median from 7.23 to 1.88.
Entering blocks 13–15 they are back to 1.00–1.86 and 5.71–6.96. From block 4 to block 12, the residue means' ring no longer
surrounds the activation zero. From block 6 to block 12 it is also squeezed into a flattened curve, and the clouds overlap. Entering blocks
13–16, they sit on a circle around zero again, with tight, separated clouds.</p>
<figure><img src="notebook/fig/mod10_track_k1.png"><figcaption>Residual stream at position 1, entering the block named
in each title (the last panel leaves block 16), projected onto that boundary's frequency-1 plane <i>P</i><sub>1</sub>. Within the
plane, the axes are turned so that residue 0's mean lies on the positive horizontal axis. Dots are the 400 values of
<i>a</i>, coloured by <i>r</i> = <i>a</i> mod 10, and labels are the residue means. The cross is the activation zero.
Each title gives the closest-pair and median separation in the plane. Script: <code>track_k1_fig.py</code>.</figcaption></figure>
<figure><img src="notebook/fig/mod10_track_planes.png"><figcaption>Residual stream at every boundary. (a) ‖Ψ<sub><i>k</i></sub>‖.
(b) Mean principal cosine between <i>P<sub>k</sub></i> and the same frequency's plane at the previous boundary. (c) The
same against the embedding's plane.</figcaption></figure>
<figure><img src="notebook/fig/mod10_track_sublayers.png"><figcaption>Per sublayer. (d) Period-10 output relative to
the input, per frequency (solid: MLP, dotted: attention). (e) Share of the MLP's period-10 output that comes from
<i>D<sup>X</sup></i>. (f) Linearity of Δ: the linear fit, the reader-secant reduction, and additivity over readers.</figcaption></figure>
"""
    return intro + "".join(block_html(b, res["blocks"][str(b)], res, rq) for b in range(LAST + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = json.loads((OUT / "analysis.json").read_text())
    figures(res, args.out / "fig")
    rq = json.loads((OUT / "residues.json").read_text())
    cloud_figure(rq, args.out / "fig")
    html = BEGIN + section_html(res, rq) + END
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        tail = "\n`\n},\n];"
        assert s.rstrip().endswith("`\n},\n];"), "unexpected end of entries.js"
        s = s.rstrip()[: -len(tail.strip()) - 1].rstrip() + "\n" + html + tail + "\n"
    p.write_text(s)


if __name__ == "__main__":
    main()
