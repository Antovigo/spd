"""Figures, numbers and page of the "Mod 50 representations" notebook (first page: census and the block-15 readers).

Data: the periodicity index (repr_repo/seed_periodicity.py, periodicity.json), the writer split of the block-15 gate
readers g19 and g29 (repr_repo/mod50_l15_writers.py, mod50_p3/l15_writers.json) and the float64 rebuild of the
alive-only model at position 3 (track_build_p3.py, track_mod10_p3/stream.npz). Nothing is centred or detrended: zero
is the zero of the activation space.

Notation (also defined on the page):
  b            integer operand 1..400 at position 3 (the token b); a the other operand
  f_c(x)       inner activation of reader c on prompt x divided by ||V_c|| (V_c its read vector)
  F_c(b)       mean of f_c over the 20 rebuild prompts with that b (10 values of a, 2 ops)
  P50, P|50    the exact-period-50 part of F_c, and its whole b mod 50 part (every non-constant frequency whose period
               divides 50), as functions of b (repr_repo/mod50_l15_writers.py)
  share_w      writer w's signed share of P50 F_c (sums to 1 over the writers)

    python mod50_page.py --out notebook_mod50
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import HUB, RUN, bf16_tensor

REPO = RUN / "analysis/representation_repository"
SPLIT = RUN / "analysis/reader_writer/mod50_p3/l15_writers.json"
STREAM = RUN / "analysis/reader_writer/track_mod10_p3/stream.npz"
L15 = RUN / "analysis/reader_writer/variance/alive_only/L15/comps.npz"
READERS = (19, 29)
K = np.arange(100)
TAU = np.asarray([100 // np.gcd(int(k), 100) for k in K])
COL = {"19": "#2a5db0", "29": "#d9730d"}
POS, NEG = "#2a5db0", "#c0392b"
plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def census() -> dict:
    """Per block and operand position: the components labelled with period 50, and how many of them have the period-50
    share as their largest exact-period share (tau > 1) and a smooth_r2 above 0.5."""
    d = json.loads((REPO / "periodicity.json").read_text())
    out: dict = {"threshold": d["threshold"], "rows": []}
    for blk in range(17):
        row = {"block": blk}
        for t, v in (("1", "a"), ("3", "b")):
            ids = [i for i in d["index"][t][v]["50"] if i.startswith(f"L{blk}.")]
            top = smooth = 0
            for i in ids:
                e = d["components"][i]["positions"][t][v]
                sh = {k: s for k, s in e["share"].items() if k != "1"}
                top += max(sh, key=sh.get) == "50"
                smooth += e["smooth_r2"] > 0.5
            row[t] = {"n": len(ids), "top50": top, "smooth": smooth, "ids": ids}
        out["rows"].append(row)
    for t in ("1", "3"):
        out[f"total_{t}"] = {k: sum(r[t][k] for r in out["rows"]) for k in ("n", "top50", "smooth")}
    out["l15"] = {}
    for c in READERS:
        e = d["components"][f"L15.gate.{c}"]["positions"]["3"]["b"]
        out["l15"][str(c)] = {
            k: e[k] for k in ("periods", "exact_periods", "share", "share_rest", "smooth_r2")
        }
    return out


def reader_curves() -> dict[str, np.ndarray]:
    """F_c(b), b = 1..400, for the two readers, from the rebuild (same definition as mod50_l15_writers.py)."""
    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.layers.15.post_attention_layernorm.weight"
    g2 = bf16_tensor(snap + wmap[name], name).astype(np.float64)
    z = np.load(L15)
    gi = {int(c): j for j, c in enumerate(z["index"][z["kind"] == "gate"])}
    V = z["V_gate"].astype(np.float64)[:, [gi[c] for c in READERS]]
    st = np.load(STREAM)
    X, b = st["mid_15"].astype(np.float64), st["b"]
    rho = np.sqrt((X**2).mean(-1) + 1e-5)
    f = (X @ (g2[:, None] * V / np.linalg.norm(V, axis=0))) / rho[:, None]
    Fb = np.zeros((400, 2))
    np.add.at(Fb, b - 1, f)
    Fb /= np.bincount(b - 1, minlength=400)[:, None]
    return {str(c): Fb[:, j] for j, c in enumerate(READERS)}


def part(F: np.ndarray, sel: np.ndarray) -> np.ndarray:
    bb = np.arange(1, 401)
    mr = np.stack([F[bb % 100 == r].mean() for r in range(100)])
    M = np.exp(-2j * np.pi * np.outer(K, np.arange(100)) / 100) @ mr
    rec = (np.exp(2j * np.pi * np.outer(np.arange(100), K[sel]) / 100) @ M[sel]).real / 100
    return rec[bb % 100]


def fig_curves(curves: dict[str, np.ndarray], path: Path) -> None:
    bb = np.arange(1, 401)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.4), gridspec_kw={"width_ratios": [2.2, 1]})
    for row, c in enumerate(curves):
        F = curves[c]
        const = part(F, K == 0)
        div50 = part(F, (50 % TAU == 0) & (K != 0))
        p50 = part(F, TAU == 50)
        ax = axs[row, 0]
        ax.plot(bb, F, ".", ms=2.5, color="0.55", label="F(b)")
        ax.plot(bb, const + div50, lw=1, color=COL[c], label="constant + b mod 50 part")
        ax.plot(
            bb, const + p50, lw=1, color="k", alpha=0.6, label="constant + exact-period-50 part"
        )
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xlabel("b")
        ax.set_ylabel(f"L15 g{c}: F(b)")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax = axs[row, 1]
        r50 = bb % 50
        for q in range(8):
            sel = (bb - 1) // 50 == q
            ax.plot(r50[sel], F[sel], ".", ms=3, color=COL[c], alpha=0.35)
        ax.plot(np.arange(50), [F[r50 == r].mean() for r in range(50)], "-", color="k", lw=1)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xlabel("b mod 50")
        ax.set_ylabel("F(b)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_shares(split: dict, path: Path, top: int = 12) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for row, c in enumerate(READERS):
        e = split["readers"][f"L15.gate.{c}"]["50"]
        ws = e["top"][:top][::-1]
        ax = axs[row, 0]
        ax.barh(range(len(ws)), [w["share"] for w in ws],
                color=[POS if w["share"] > 0 else NEG for w in ws])  # fmt: skip
        ax.set_yticks(range(len(ws)), [w["writer"] for w in ws], fontsize=7)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_xlabel(f"share of L15 g{c}'s exact-period-50 part")
        ax.set_title(f"g{c}: the {top} writers with the largest |share|", fontsize=9)
        subs = [(k, v) for k, v in e["by_sublayer"].items() if abs(v) >= 0.02]
        order = sorted(
            subs,
            key=lambda kv: (
                kv[0] != "emb",
                int(kv[0].split(".")[0][1:]) if kv[0] != "emb" else 0,
                kv[0],
            ),
        )
        ax = axs[row, 1]
        ax.bar(range(len(order)), [v for _, v in order],
               color=[POS if v > 0 else NEG for _, v in order])  # fmt: skip
        ax.set_xticks(range(len(order)), [k for k, _ in order], rotation=70, fontsize=7)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel("share (summed over the sublayer)")
        ax.set_title(f"g{c}: by sublayer (|share| ≥ 0.02)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(x: float, n: int = 2) -> str:
    return f"{x:+.{n}f}".replace("-", "−")


def page(cen: dict, split: dict) -> str:
    th = cen["threshold"]
    rows = "".join(
        f"<tr><td>{r['block']}</td><td>{r['1']['n']}</td><td>{r['1']['top50']}</td><td>{r['1']['smooth']}</td>"
        f"<td>{r['3']['n']}</td><td>{r['3']['top50']}</td><td>{r['3']['smooth']}</td></tr>"
        for r in cen["rows"]
    )
    t1, t3 = cen["total_1"], cen["total_3"]
    rows += (f"<tr><td><b>total</b></td><td>{t1['n']}</td><td>{t1['top50']}</td><td>{t1['smooth']}</td>"
             f"<td>{t3['n']}</td><td>{t3['top50']}</td><td>{t3['smooth']}</td></tr>")  # fmt: skip
    l15 = cen["l15"]

    def per(c: str) -> str:
        s = l15[c]["share"]
        return ", ".join(f"τ = {k}: {s[k]:.3f}" for k in ("1", "5", "25", "50", "100"))

    R = split["readers"]

    def wrow(c: int) -> str:
        e = R[f"L15.gate.{c}"]["50"]
        return "".join(
            f"<tr><td><code>{w['writer']}</code></td><td>{fmt(w['share'])}</td><td>{w['own']:.2f}</td></tr>"
            for w in e["top"][:10]
        )

    def sub(c: int, k: str) -> str:
        return fmt(R[f"L15.gate.{c}"]["50"]["by_sublayer"].get(k, 0.0))

    sub_rows = "".join(
        f"<tr><td>{k}</td><td>{sub(19, k)}</td><td>{sub(29, k)}</td></tr>"
        for k in ("emb", "L4.mlp", "L8.mlp", "L11.mlp", "L12.mlp", "L13.mlp", "L14.mlp", "L15.attn")
    )
    early = {
        c: sum(
            v
            for k, v in R[f"L15.gate.{c}"]["50"]["by_sublayer"].items()
            if k != "emb" and int(k.split(".")[0][1:]) <= 10
        )  # fmt: skip
        for c in READERS
    }
    top15 = {
        c: next(
            w["writer"] for w in R[f"L15.gate.{c}"]["50"]["top"] if w["writer"].startswith("L15.o.")
        )
        for c in READERS
    }
    return f"""
<p><b>Summary.</b> At the token <i>b</i>, the block-15 gate readers <code>g19</code> and <code>g29</code> read a
pattern of <i>b</i> mod 50. Splitting their input exactly into one term per upstream writer, the exact-period-50 part
of that pattern comes mostly from three MLP writers of blocks 11–12 (<code>L12.down.24</code>,
<code>L11.down.10</code>, <code>L12.down.14</code>), from block-13 writers, and from the embedding of the token
<i>b</i>. Block 15's own attention opposes it: its share is {fmt(R["L15.gate.19"]["50"]["by_sublayer"]["L15.attn"])}
for <code>g19</code> and {fmt(R["L15.gate.29"]["50"]["by_sublayer"]["L15.attn"])} for <code>g29</code>.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {{1, …, 400}} and op ∈ {{+, −}}. Positions: 1 is the token <i>a</i>, 3 the token <i>b</i>. The
model is the alive-only model: every component of the ceiling filter <code>addsub-08-last-pos</code> kept by the
mask “kl_int, sampled mask, λ = 5e-05” is on, and every other component and the weight delta are off.
<b>Nothing in this page is centred or detrended</b>: the network has no biases, so zero is the zero of the
activation space.</p>
<ul>
<li>A <b>component</b> <i>c</i> has a read vector <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup> (gate, up, q, k, v) or
ℝ<sup>14336</sup> (down) and a write vector <i>U<sub>c</sub></i>. Its id is <code>L&lt;block&gt;.&lt;kind&gt;.&lt;index&gt;</code>;
<code>g19</code> is <code>L15.gate.19</code>. A <b>writer</b> is a component that adds to the residual stream: an
attention <code>o</code> component or an MLP <code>down</code> component; the embedding of the token <i>b</i> counts
as one writer, “emb”.</li>
<li><i>f<sub>c</sub></i>(<i>x</i>): the inner activation of component <i>c</i> on prompt <i>x</i> divided by
‖<i>V<sub>c</sub></i>‖, a real number.</li>
<li><i>k</i> ∈ {{0, …, 99}}: a frequency of <i>b</i> mod 100 (integer); its exact period is
τ(<i>k</i>) = 100 / gcd(<i>k</i>, 100), a divisor of 100. The <b>exact-period-τ part</b> of a function of <i>b</i> is
its part made of the frequencies <i>k</i> with τ(<i>k</i>) = τ, computed from its means over the classes
<i>b</i> mod 100. Parts of different τ are orthogonal; a function of <i>b</i> mod 50 has parts with τ ∈ {{1, 2, 5,
10, 25, 50}} (τ = 1 is the constant).</li>
<li><b>share<sub>τ</sub></b>: the energy of the exact-period-τ part over the total energy Σ<sub><i>b</i></sub>
<i>f</i>(<i>b</i>)<sup>2</sup> (uncentred). A component is <b>labelled with period 50</b> when share<sub>50</sub> ≥
{th}, beats the 99th percentile of 1,000 random permutations of <i>b</i>, and beats the 99th percentile of 1,000
draws that rotate the hundreds of <i>b</i> independently; 50 is then in its <code>periods</code> list (definitions in
<code>repr_repo/seed_periodicity.py</code>). <b>smooth_r2</b> is the uncentred share of the energy explained by a
degree-3 polynomial in <i>b</i>; a slow trend in <i>b</i> also puts energy into τ = 50 and 100.</li>
</ul>

<h3>1. Components with period 50 at the operand positions</h3>
<p>Counts of the components of each block labelled with period 50 at position 1 (variable <i>a</i>) and at
position 3 (variable <i>b</i>), from <code>periodicity.json</code>. All kinds (gate, up, down, q, k, v, o) are counted.
“τ = 50 largest” counts those whose share<sub>50</sub> is the largest of their shares with τ &gt; 1. “smooth” counts
those with smooth_r2 &gt; 0.5.</p>
<table><tr><th>block</th><th>p1: labelled 50</th><th>p1: τ = 50 largest</th><th>p1: smooth</th>
<th>p3: labelled 50</th><th>p3: τ = 50 largest</th><th>p3: smooth</th></tr>{rows}</table>
<p>Block 15 has the most labelled components at both positions ({cen["rows"][15]["1"]["n"]} at position 1,
{cen["rows"][15]["3"]["n"]} at position 3), followed by block 16.</p>

<h3>2. The block-15 gate readers g19 and g29 and their writers</h3>
<p>Shares of <code>g19</code> at position 3 (<code>periodicity.json</code>, all 320,000 prompts): {per("19")}; its
<code>periods</code> list is {l15["19"]["periods"]}. Shares of <code>g29</code>: {per("29")}; its
<code>periods</code> list is {l15["29"]["periods"] or "empty"}: its share<sub>50</sub> is under the
{th} threshold, and most of its energy is a constant.</p>
<p><b>Data.</b> The float64 rebuild of the alive-only model at position 3 (<code>track_build_p3.py</code>): 8,000
prompts, op ∈ {{+, −}}, ten values of <i>a</i> (7 + 41<i>i</i>, <i>i</i> = 0, …, 9), every <i>b</i>. On it,
<i>F<sub>c</sub></i>(<i>b</i>) is the mean of <i>f<sub>c</sub></i> over the 20 prompts with that <i>b</i>; the
share<sub>50</sub> of <code>g19</code> measured on these 8,000 prompts is
{split["readers"]["L15.gate.19"]["period_shares"]["50"]:.3f} (against {l15["19"]["share"]["50"]:.3f} on all
prompts).</p>
<figure><img src="fig/mod50_l15_curves.png" alt="F(b) of g19 and g29 against b and against b mod 50">
<figcaption>Left: <i>F<sub>c</sub></i>(<i>b</i>) for <i>b</i> = 1, …, 400 (grey dots), with its constant plus its
<i>b</i> mod 50 part (colour) and its constant plus its exact-period-50 part (black). Right: the same values against
<i>b</i> mod 50, one dot per <i>b</i>; the black line is the mean over the 8 values of <i>b</i> in each class.
Top: <code>g19</code>; bottom: <code>g29</code>.</figcaption></figure>

<p><b>The split.</b> Let <i>X</i>(<i>x</i>) ∈ ℝ<sup>4096</sup> be the residual stream at position 3 after block 15's
attention, γ ∈ ℝ<sup>4096</sup> the gain of block 15's post-attention RMSNorm, and
ρ(<i>x</i>) = (mean<sub><i>j</i></sub> <i>X<sub>j</sub></i>(<i>x</i>)<sup>2</sup> + 10<sup>−5</sup>)<sup>1/2</sup>
its RMS (<i>j</i> ∈ {{1, …, 4096}} indexes the coordinates). Then
<i>f<sub>c</sub></i>(<i>x</i>) = (γ ⊙ <i>V<sub>c</sub></i>) · <i>X</i>(<i>x</i>) / (ρ(<i>x</i>) ‖<i>V<sub>c</sub></i>‖),
and <i>X</i> is the sum of the embedding of <i>b</i> and of <i>t<sub>w</sub></i>(<i>x</i>) <i>U<sub>w</sub></i> over
every alive <code>o</code> component of blocks 0–15 and every alive <code>down</code> component of blocks 0–14
(<i>t<sub>w</sub></i> the writer's activation at position 3). So
<i>f<sub>c</sub></i> = Σ<sub><i>w</i></sub> <i>e<sub>w</sub></i> exactly, with
<i>e<sub>w</sub></i>(<i>x</i>) = <i>t<sub>w</sub></i>(<i>x</i>) (γ ⊙ <i>V<sub>c</sub></i>) · <i>U<sub>w</sub></i> /
(ρ(<i>x</i>) ‖<i>V<sub>c</sub></i>‖) (the reconstruction error is below 10<sup>−3</sup> of the largest
|<i>f<sub>c</sub></i>|). Writing <i>P</i><sub>50</sub><i>G</i> for the exact-period-50 part of the per-<i>b</i> mean
<i>G</i> of a field, writer <i>w</i>'s <b>share</b> is
⟨<i>P</i><sub>50</sub><i>E<sub>w</sub></i>, <i>P</i><sub>50</sub><i>F<sub>c</sub></i>⟩ /
‖<i>P</i><sub>50</sub><i>F<sub>c</sub></i>‖<sup>2</sup> (inner products over <i>b</i> = 1, …, 400; <i>E<sub>w</sub></i>
the per-<i>b</i> mean of <i>e<sub>w</sub></i>). The shares are signed and sum to 1 over the writers; a negative share
opposes the reader's pattern. <b>own</b> = ‖<i>P</i><sub>50</sub><i>E<sub>w</sub></i>‖<sup>2</sup> /
‖<i>P</i><sub>50</sub><i>F<sub>c</sub></i>‖<sup>2</sup> is the size of the writer's own period-50 part next to the
reader's.</p>
<figure><img src="fig/mod50_l15_shares.png" alt="writer shares of g19 and g29's period-50 part">
<figcaption>Left: the 12 writers with the largest |share| of the reader's exact-period-50 part (blue positive, red
negative). Right: the shares summed over each sublayer (“L<i>l</i>.mlp” all down components of block <i>l</i>,
“L<i>l</i>.attn” all its <code>o</code> components), sublayers with |sum| ≥ 0.02. Top: <code>g19</code>; bottom:
<code>g29</code>.</figcaption></figure>
<table><tr><th>writer</th><th>share in g19</th><th>own</th></tr>{wrow(19)}</table>
<table><tr><th>writer</th><th>share in g29</th><th>own</th></tr>{wrow(29)}</table>
<p>Summed by sublayer:</p>
<table><tr><th>sublayer</th><th>g19</th><th>g29</th></tr>{sub_rows}</table>
<p>All MLP and attention sublayers of blocks 0–10 together: {fmt(early[19])} for <code>g19</code>,
{fmt(early[29])} for <code>g29</code>. The block-15 attention term with the largest |share| is
<code>{top15[19]}</code> for <code>g19</code> and <code>{top15[29]}</code> for <code>g29</code>.</p>
<p class="note">Scripts: <code>repr_repo/mod50_l15_writers.py</code> (split; output
<code>&lt;run&gt;/analysis/reader_writer/mod50_p3/l15_writers.json</code>, which also holds the split of the whole
<i>b</i> mod 50 part) and <code>readers/mod50_page.py</code> (this page).</p>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "notebook_mod50")
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cen = census()
    split = json.loads(SPLIT.read_text())
    fig_curves(reader_curves(), fig_dir / "mod50_l15_curves.png")
    fig_shares(split, fig_dir / "mod50_l15_shares.png")
    (args.out / "mod50.json").write_text(json.dumps(cen, indent=1))
    entry = ('{\nid: "mod50",\ntitle: "Mod 50: census and the block-15 readers of b",\ndate: "2026-10-27",\nhtml: `'
             + page(cen, split) + "`\n},\n")  # fmt: skip
    p = args.out / "entries.js"
    s = (
        p.read_text()
        if p.exists()
        else "// Notebook pages of the mod 50 report.\nwindow.NOTEBOOK = [\n];\n"
    )
    if 'id: "mod50"' in s:
        i = s.index('{\nid: "mod50"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)
    print(json.dumps({k: cen[k] for k in ("total_1", "total_3")}))


if __name__ == "__main__":
    main()
