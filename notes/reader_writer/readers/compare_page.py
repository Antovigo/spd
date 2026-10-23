"""Section 6 of the notebook page "Mod 10 representations at b (position 3)": a and b compared block by block.

Reads compare_p1p3/compare.json (compare_p1p3.py); writes notebook/fig/p3_compare.png and the section between
<!-- compare:begin --> and <!-- compare:end --> at the end of the mod10_p3 entry. Nothing is centred.

    python compare_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compare_p1p3 import OUT

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
BEGIN, END = "<!-- compare:begin -->", "<!-- compare:end -->"


def rng(v: list[float], d: int = 2) -> str:
    return f"{min(v):.{d}f}–{max(v):.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = json.loads((OUT / "compare.json").read_text())
    bl = sorted(int(b) for b in r)
    x = np.asarray(bl)
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for k, c in ((0, "#2a5db0"), (2, "#c0392b")):
        ax[0].plot(
            x,
            [r[str(b)]["R_res"][k] for b in bl],
            "-o",
            ms=3,
            color=c,
            label=f"residual stream, k = {k + 1}",
        )
        ax[0].plot(
            x,
            [r[str(b)]["R_norm"][k] for b in bl],
            "--s",
            ms=3,
            color=c,
            label=f"MLP input (after the norm), k = {k + 1}",
        )
    ax[0].plot(
        x,
        [1 / r[str(b)]["rho_ratio"] for b in bl],
        ":^",
        ms=3,
        color="#1d1d1b",
        label="1 / (RMS ratio)",
    )
    ax[0].plot(
        x,
        [r[str(b)]["reader_amp_ratio_median"] or np.nan for b in bl],
        "d",
        ms=5,
        color="#27ae60",
        label="mod-10 readers (median)",
    )
    ax[0].set_title("(a) amplitude at position 3 ÷ position 1")
    ax[0].legend(frameon=False, fontsize=6)
    for k, c in ((0, "#2a5db0"), (2, "#c0392b"), (3, "#8e44ad")):
        ax[1].plot(
            x,
            [
                r[str(b)]["compensated"][k] if r[str(b)]["compensated"][k] is not None else np.nan
                for b in bl
            ],
            "-o",
            ms=3,
            color=c,
            label=f"k = {k + 1}",
        )
        ax[2].plot(
            x,
            [r[str(b)]["cos_planes_res"][k] for b in bl],
            "-o",
            ms=3,
            color=c,
            label=f"k = {k + 1}",
        )
    ax[1].axhline(0, color="#999", lw=0.8)
    ax[1].set_title("(b) share of the shrinking undone by the norm")
    ax[1].legend(frameon=False, fontsize=7)
    ax[2].set_ylim(0.6, 1.02)
    ax[2].set_title("(c) planes of a (position 1) vs b (position 3)")
    ax[2].legend(frameon=False, fontsize=7)
    ax[3].plot(
        x,
        [r[str(b)]["clouds_actual"]["sep_min"] for b in bl],
        "-o",
        ms=3,
        color="#1d1d1b",
        label="actual",
    )
    ax[3].plot(x, [r[str(b)]["clouds_no_ctx_response"]["sep_min"] for b in bl], "--s", ms=3, color="#c0392b",
               label="MLP's response to the context removed")  # fmt: skip
    ax[3].set_title("(d) position 3: closest-pair separation leaving the block")
    ax[3].legend(frameon=False, fontsize=7)
    for a in ax:
        a.set_xticks(x)
        a.set_xlabel("block (MLP input)")
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "p3_compare.png")
    plt.close(fig)

    rows = ""
    for b in bl:
        v = r[str(b)]
        cs = v["ctx_share_mod_readers"]
        big = (
            ", ".join(
                f"{n} ({s:.2f})" for n, s in sorted(cs.items(), key=lambda t: -t[1]) if s > 0.3
            )
            or "–"
        )
        comp = [c for c in v["compensated"] if c is not None]
        sp = v["delta_ctx_split_in_Dout"]
        med = "–" if v["reader_amp_ratio_median"] is None else f"{v['reader_amp_ratio_median']:.2f}"
        rows += (f"<tr><td>{b}</td><td>{rng(v['R_res'])}</td><td>{v['rho_ratio']:.2f}</td><td>{rng(v['R_norm'])}</td>"
                 f"<td>{rng(comp) if comp else '–'}</td><td>{rng(v['cos_planes_res'])}</td><td>{v['n_mod_readers']}</td>"
                 f"<td>{med}</td>"
                 f"<td>{big}</td><td>{v['delta_ctx_in_Dout_over_mlp_in_Dout']:.2f} ({sp['offset']:.2f} / {sp['b_periodic']:.2f} / {sp['rest']:.2f})</td>"
                 f"<td>{v['clouds_no_ctx_response']['sep_min']:.2f} → {v['clouds_actual']['sep_min']:.2f}</td></tr>")  # fmt: skip
    rr = [v["rho_ratio"] for v in r.values()]
    html = f"""{BEGIN}
<h3>6. a and b compared, blocks 0–16</h3>
<p>This section compares the representation of <i>a</i> mod 10 at the token <i>a</i> (position 1, 400 prompts) with that of <i>b</i> mod 10
at the token <i>b</i> (position 3, the 8,000 position-3 prompts), at the MLP input of every block (<code>compare_p1p3.py</code>,
<code>compare_page.py</code>).</p>
<ul>
<li><b>Amplitudes.</b> <i>X</i> is the residual stream entering the MLP, ρ = √(mean of <i>X</i>² over the 4096 coordinates) its RMS, and
<i>x</i> = γ ⊙ <i>X</i> / ρ the MLP input, with γ the norm's weight. <i>A<sub>k</sub></i> is the frequency-<i>k</i> Fourier amplitude of the residue means
(<i>k</i> = 1 … 5). R<sub>res</sub> = <i>A<sub>k</sub></i>(<i>X</i> at position 3) / <i>A<sub>k</sub></i>(<i>X</i> at position 1). R<sub>norm</sub> is the same ratio for <i>x</i>, and the RMS
ratio is mean ρ at position 3 / mean ρ at position 1. "Undone by the norm" = 1 − log R<sub>norm</sub> / log R<sub>res</sub>: 0 when
the readers see the residual stream's shrinking in full, 1 when the norm cancels it, and negative when the norm adds to it.</li>
<li><b>Mod-10 readers</b> of a block are its gate and up components whose inner activation <i>t<sub>c</sub></i> = <i>x</i> · <i>V<sub>c</sub></i> is explained with uncentred
r² ≥ 0.5 by the input projected onto the period-10 subspace <i>D</i>, at either position. Their amplitude ratio is the norm of the
period-10 part of <i>t<sub>c</sub></i>'s residue means at position 3 divided by that at position 1.</li>
<li><b>Context.</b> <i>C</i> is the 10-dimensional span of the frequency 1–5 Fourier components of the position-3 stream's means over
<i>a</i> mod 10, plus the difference between the + and − means, with <i>b</i>'s period-10 subspace projected out. A reader's context
share is Σ(<i>t<sub>c</sub></i>(<i>x</i>) − <i>t<sub>c</sub></i>(<i>x<sub>C</sub></i>))² / Σ <i>t<sub>c</sub></i>(<i>x</i>)², where <i>x<sub>C</sub></i> = γ ⊙ (<i>X</i> − Π<sub><i>C</i></sub><i>X</i>) / ρ(<i>X</i>) is the
input without the context content and with the norm held fixed. It can exceed 1 when the context flips the activation's sign.
The MLP's response to the context is Δ<sub>ctx</sub> = MLP(<i>x</i>) − MLP(<i>x<sub>C</sub></i>). The table gives the size of its part in <i>D′</i>, <i>b</i>'s
subspace of the stream leaving the block, relative to the MLP's whole write into <i>D′</i>. That part is split into a per-context offset, a
part periodic in <i>b</i>, and the rest.</li>
</ul>
<div class="tw"><table><tr><th>block</th><th>R<sub>res</sub></th><th>RMS ratio</th><th>R<sub>norm</sub></th><th>undone by the norm</th>
<th>plane cosines a vs b</th><th>mod-10 readers</th><th>their amplitude ratio (median)</th><th>mod-10 readers with context share &gt; 0.3</th>
<th>Δ<sub>ctx</sub> in D′ / MLP in D′ (offset / b-periodic / rest)</th><th>closest pair: no Δ<sub>ctx</sub> → actual</th></tr>{rows}</table></div>
<p><b>Directions are conserved, amplitudes are not.</b> The planes of <i>a</i> at position 1 and of <i>b</i> at position 3 have mean
principal cosines {rng([min(v["cos_planes_res"]) for v in r.values()])} (lowest) to 1.00 per plane, with the full-residue plane
<i>k</i> = 3 and the mod-5 plane <i>k</i> = 4 drifting most (0.75–0.79 at blocks 9–12). In the residual stream, <i>b</i>'s amplitude is
{rng([min(v["R_res"]) for v in r.values()])} to {rng([max(v["R_res"]) for v in r.values()])} of <i>a</i>'s, frequency by frequency.
It is 1.00 at block 0, 0.86–0.88 at blocks 1–2 (block 0's MLP writes less at position 3), 0.77–0.84 at block 3, and
0.57–0.76 from block 4 on.</p>
<p><b>What the norm does.</b> The readers see R<sub>norm</sub>, not R<sub>res</sub>, and the two differ by the RMS ratio
({rng(rr)}). The norm undoes 0.06–0.42 of the shrinking in blocks 1, 4–9 and 12–15, where position 3's RMS is smaller (ratio
0.89–0.96). It undoes almost nothing in blocks 10, 11 and 16 (0.01–0.12; RMS ratio 0.97–0.98). It adds to the shrinking in blocks 0, 2 and 3, where position 3's
RMS is larger (1.04–1.14). At block 0 the stream's amplitudes are equal and the whole 0.87 comes from the norm. The mod-10 readers
follow R<sub>norm</sub>: their median amplitude ratio is 0.73–0.87 in every block that has them (block 10's single reader: 0.34).</p>
<p><b>Mod-10 readers rarely respond to the context, and its effect on <i>b</i>'s representation is mostly a shift.</b> Except in blocks
10–12 (0.15–0.40), the mod-10 readers' median context share is ≤ 0.13. The exceptions are a few readers in blocks 3, 4, 12 and 13, listed in the
table, whose activation at position 3 depends strongly on <i>a</i> or the operation. The MLP's response to the context lands in
<i>b</i>'s subspace with 0.26–0.74 of the size of its whole write there. From block 1 on it is mostly an offset per context
(0.57–0.95 of its energy) and hardly periodic in <i>b</i> (≤ 0.09). At block 0, 0.62 of it is periodic in <i>b</i> and 0.27 is an offset. The components
that write it are mostly not mod-10 readers. In blocks 0–11 and 14–16 it lowers the closest-pair separation (largest at block 7:
1.31 → 0.94). In blocks 12–13 it raises it (1.21 → 1.33 and 2.39 → 2.52); the largest contributors there include the mod-10
readers g46 and g14.</p>
<figure><img src="notebook/fig/p3_compare.png"><figcaption>(a) Frequency-<i>k</i> amplitude at position 3 ÷ position 1, in the residual stream
(solid) and at the MLP input (dashed), the inverse RMS ratio (dotted), and the mod-10 readers' median ratio (green). (b) Share of the
residual-stream shrinking undone by the norm. (c) Mean principal cosine between the planes of <i>a</i> at position 1 and of <i>b</i> at
position 3. (d) Closest-pair separation of the <i>b</i> clouds leaving the block, actual and with the MLP's response to the context
removed.</figcaption></figure>
{END}"""
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        i = s.index('id: "mod10_p3"')
        j = s.index("`\n},\n", i)
        s = s[:j] + html + "\n" + s[j:]
    p.write_text(s)


if __name__ == "__main__":
    main()
