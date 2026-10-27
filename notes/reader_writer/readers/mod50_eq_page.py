"""Page of the "Mod 50 representations" notebook: the block-15 gate readers g19 and g29 at "=" and where their input
comes from (results of mod50_eq.py).

Data: <run>/analysis/reader_writer/mod50_eq/{attribution.json, fig/*.png} (mod50_eq.py) and the periodicity index
(repr_repo/seed_periodicity.py, periodicity.json). Light: reads JSON and PNGs and draws one bar chart.

    python mod50_eq_page.py --out notebook_mod50
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
SRC = RUN / "analysis/reader_writer/mod50_eq"
PERIODICITY = RUN / "analysis/representation_repository/periodicity.json"
READERS = ("L15.gate.19", "L15.gate.29")
OPS = ("+", "-")
SOURCES = ("BOS", "a", "op", "b", "=")
FIGS = ("stream_add.png", "stream_sub.png", "writes15_add.png", "writes15_sub.png")
POS, NEG = "#2a5db0", "#c0392b"


def fmt(x: float, n: int = 2) -> str:
    return f"{x:+.{n}f}".replace("-", "−")


def short(c: str) -> str:
    return "g" + c.split(".")[-1]


def fig_groups(att: dict, path: Path) -> None:
    """Group shares per (reader, op): each sublayer of blocks 0..14, the embedding, block 15's attention by source."""
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    keys = ["emb"] + [f"L{b}.{s}" for b in range(15) for s in ("attn", "mlp")]
    keys += [f"L15.attn.{s}" for s in SOURCES]
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for ax, (c, op) in zip(axs, [(c, op) for c in READERS for op in OPS], strict=True):
        g = att["readers"][c][op]["groups"]
        v = [g[k]["share"] for k in keys]
        ax.bar(range(len(keys)), v, color=[POS if x > 0 else NEG for x in v])
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel("share")
        ax.set_title(f"{short(c)}, prompts a {op} b", fontsize=8)
    axs[-1].set_xticks(range(len(keys)), keys, rotation=80, fontsize=6)
    fig.savefig(path)
    plt.close(fig)


def page(att: dict, per: dict) -> str:
    R = att["readers"]
    ck = att["checks"]

    def g(c: str, op: str, k: str) -> float:
        return R[c][op]["groups"][k]["share"]

    def early(c: str, op: str) -> float:
        return sum(
            v["share"]
            for k, v in R[c][op]["groups"].items()
            if k != "emb" and not k.startswith("L15")
        )

    cols = [(c, op) for c in READERS for op in OPS]
    head = "".join(f"<th>{short(c)}, a {op} b</th>" for c, op in cols)
    rows = [("mean of f<sub>c</sub>", [fmt(R[c][op]["mean_f"]) for c, op in cols]),
            ("rms of f<sub>c</sub>", [f"{R[c][op]['rms_f']:.2f}" for c, op in cols])]  # fmt: skip
    for s in SOURCES:
        rows.append(
            (f"block-15 attention, source {s}", [fmt(g(c, op, f"L15.attn.{s}")) for c, op in cols])
        )
    rows.append(("blocks 0–14, every writer", [fmt(early(c, op)) for c, op in cols]))
    rows.append(("embedding of “=”", [fmt(g(c, op, "emb")) for c, op in cols]))
    rows.append(("r² of block-15 attention from b alone",
                 [f"{R[c][op]['groups']['L15.attn.b']['r2_alone']:.2f}" for c, op in cols]))  # fmt: skip
    for k in (1, 5, 10, 20, 60):
        rows.append((f"r² of the {k} writers of largest |share|",
                     [f"{R[c][op]['r2_topk'][k - 1]:.2f}" for c, op in cols]))  # fmt: skip
    table = "".join(
        f"<tr><td>{n}</td>{''.join(f'<td>{x}</td>' for x in xs)}</tr>" for n, xs in rows
    )

    def top(c: str, op: str, n: int = 15) -> str:
        return "".join(
            f"<tr><td><code>{t['writer']}</code></td><td>{fmt(t['share'])}</td><td>{t['own']:.2f}</td>"
            f"<td>{fmt(t['mean'])}</td></tr>"
            for t in R[c][op]["top"][:n]
        )

    tops = "".join(
        f"<h4>{short(c)}, prompts a {op} b</h4><table><tr><th>writer</th><th>share</th><th>own</th><th>mean</th></tr>"
        f"{top(c, op)}</table>"
        for c, op in cols
    )

    def pshares(c: str) -> str:
        e = per["components"][c]["positions"]["4"]["b"]
        s = e["share"]
        return (", ".join(f"τ = {k}: {s[k]:.2f}" for k in ("1", "5", "10", "25", "50", "100"))
                + f"; periods {e['periods']}")  # fmt: skip

    lo = min(g(c, op, "L15.attn.b") for c, op in cols)
    hi = max(g(c, op, "L15.attn.b") for c, op in cols)
    return f"""
<p><b>Summary.</b> At the token “=”, the block-15 gate readers <code>g19</code> and <code>g29</code> carry a
pattern of <i>b</i> mod 50. Their activation there splits exactly into one term per upstream writer. The block-15
attention reading from the token <i>b</i> supplies {lo:.2f}–{hi:.2f} of it, depending on the reader and the operator.
The other source positions of that attention supply almost nothing. The rest comes from many small writers of blocks
0–14; none of them reaches 0.13. On the read directions of the two readers, the stream at “=” stays in small
clouds up to block 15's attention. After that attention the points spread along a band about 0.5 long, with the
colours of <i>b</i> mod 50 sorted along it, and the source-<i>b</i> write alone draws that band.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {{1, …, 400}} and op ∈ {{+, −}}: 320,000 prompts, 160,000 per operator. Positions 0 to 4 are
<code>&lt;BOS&gt;</code>, <i>a</i>, op, <i>b</i> and “=”. The model is the alive-only model: every alive component
of the ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off.
It is rebuilt from its stored activations (float32, every alive component, positions 1–4); position 0 is the same on
every prompt. <b>Nothing in this notebook is centred or detrended</b>: the network has no biases, so zero is the
zero of the activation space.</p>
<ul>
<li>A <b>component</b> has a read vector and a write vector. Its id is
<code>L&lt;block&gt;.&lt;kind&gt;.&lt;index&gt;</code>; <code>g19</code> is <code>L15.gate.19</code>,
<code>g29</code> is <code>L15.gate.29</code>. <i>c</i> denotes one of these two readers, <i>V<sub>c</sub></i> ∈
ℝ<sup>4096</sup> its read vector.</li>
<li>γ ∈ ℝ<sup>4096</sup>: the gain of block 15's post-attention RMSNorm. <b>r<sub>c</sub></b> =
γ ⊙ <i>V<sub>c</sub></i> / ‖<i>V<sub>c</sub></i>‖ (⊙ the elementwise product): the reader's read direction with
the gain folded in.</li>
<li><i>x</i>: a prompt. <b>pre<sub><i>l</i></sub></b>(<i>x</i>), <b>mid<sub><i>l</i></sub></b>(<i>x</i>) ∈
ℝ<sup>4096</sup>: the residual stream at “=” entering block <i>l</i> and after block <i>l</i>'s attention (the input
of its MLP), <i>l</i> ∈ {{0, …, 16}} a block index.</li>
<li>ρ(<i>x</i>) = (mean<sub><i>j</i></sub> mid<sub>15,<i>j</i></sub>(<i>x</i>)<sup>2</sup> +
10<sup>−5</sup>)<sup>1/2</sup>, <i>j</i> ∈ {{1, …, 4096}} the coordinates: the RMS that block 15's RMSNorm divides
by. <b>f<sub>c</sub></b>(<i>x</i>) = r<sub>c</sub> · mid<sub>15</sub>(<i>x</i>) / ρ(<i>x</i>): the reader's inner
activation divided by ‖<i>V<sub>c</sub></i>‖.</li>
<li><b>Writers</b> <i>w</i>: the embedding of “=” (one term, the same on every prompt); every alive attention
<code>o</code> component and every alive MLP <code>down</code> component of blocks 0–14, at “=”; and every alive
<code>o</code> component of block 15 split by the source position <i>s</i> ∈ {{0, …, 4}} its heads read from, id
<code>L15.o.&lt;index&gt;@s&lt;s&gt;</code>. The split recomputes block 15's attention at “=” and keeps, in every
head, only the pattern weight on source <i>s</i> times the value at <i>s</i>; the five pieces add up to the
component's activation. {att["n_writers"]} writers in all. <i>t<sub>w</sub></i>(<i>x</i>) is the writer's
activation at “=”, <i>U<sub>w</sub></i> its write vector, and mid<sub>15</sub> = Σ<sub><i>w</i></sub>
<i>t<sub>w</sub></i> <i>U<sub>w</sub></i> exactly.</li>
<li><b>e<sub>w</sub></b>(<i>x</i>) = <i>t<sub>w</sub></i>(<i>x</i>) (r<sub>c</sub> · <i>U<sub>w</sub></i>) /
ρ(<i>x</i>): writer <i>w</i>'s term in the reader's activation, so f<sub>c</sub> = Σ<sub><i>w</i></sub>
e<sub>w</sub> exactly. ρ is the reader's own per-prompt RMS, shared by every term.</li>
<li>Sums over <i>x</i> run over the 160,000 prompts of one operator, uncentred.
<b>share<sub><i>w</i></sub></b> = Σ<sub><i>x</i></sub> e<sub>w</sub> f<sub>c</sub> / Σ<sub><i>x</i></sub>
f<sub>c</sub><sup>2</sup>: writer <i>w</i>'s signed share of the activation, constant part included; the shares
add up to 1 over the writers, and a negative share opposes the activation. <b>own<sub><i>w</i></sub></b> =
Σ<sub><i>x</i></sub> e<sub>w</sub><sup>2</sup> / Σ<sub><i>x</i></sub> f<sub>c</sub><sup>2</sup>: the size of the
term next to the activation. <b>mean<sub><i>w</i></sub></b>: the mean of e<sub>w</sub> over the prompts.
<b>r²</b>(<i>S</i>) = 1 − Σ<sub><i>x</i></sub> (f<sub>c</sub> − Σ<sub><i>w</i> ∈ <i>S</i></sub>
e<sub>w</sub>)<sup>2</sup> / Σ<sub><i>x</i></sub> f<sub>c</sub><sup>2</sup> for a set <i>S</i> of writers
(uncentred): how much of the activation the writers of <i>S</i> rebuild on their own.</li>
<li>Periodicity shares (from <code>periodicity.json</code>, addition prompts): share<sub>τ</sub> is the energy of
the part of f<sub>c</sub> that is a function of <i>b</i> mod 100 with exact period τ (τ a divisor of 100, τ = 1 the
constant), as a share of Σ f<sub>c</sub><sup>2</sup>.</li>
</ul>

<h3>1. The readers g19 and g29 at “=” and where their input comes from</h3>
<p>Periodicity shares in <i>b</i> at “=”: <code>g19</code> {pshares("L15.gate.19")}; <code>g29</code>
{pshares("L15.gate.29")}.</p>
<p><b>Checks.</b> The rebuilt activations of the block-15 <code>o</code> components (summed over the sources) match
the stored ones to {ck["o15_rel_max"]:.3f} of their largest value, and the rebuilt f<sub>c</sub> matches the stored
reader activations with r² = {min(ck["f_r2_vs_stored"]):.5f}. Σ e<sub>w</sub> = f<sub>c</sub> holds on every
prompt to float64 precision.</p>
<div class="tw"><table><tr><th></th>{head}</tr>{table}</table></div>
<figure><img src="fig/mod50_eq_groups.png" alt="shares by sublayer for g19 and g29, both operators">
<figcaption>Shares summed by group: the embedding of “=”, the attention and the MLP of each block 0–14, and
block 15's attention by source position (blue positive, red negative). One row per reader and operator.</figcaption>
</figure>
<p>The writers with the largest |share|:</p>
{tops}

<h3>2. The stream at “=” on the read directions of g19 and g29, blocks 0–15</h3>
<p>Each panel plots, for 20,000 random prompts of one operator, the two coordinates mid<sub><i>l</i></sub> ·
r<sub>19</sub> and mid<sub><i>l</i></sub> · r<sub>29</sub> (raw: not divided by the RMS of the stream), coloured by
<i>b</i> mod 50 on the viridis scale. Blocks 0–14 show mid<sub><i>l</i></sub>; the last two panels show
pre<sub>15</sub> (before block 15's attention) and mid<sub>15</sub> (what block 15's MLP reads, up to its RMS).
Each panel has its own axis limits.</p>
<figure><img src="fig/stream_add.png" alt="stream at = on r19 and r29, blocks 0 to 15, addition">
<figcaption>Prompts <i>a</i> + <i>b</i>.</figcaption></figure>
<figure><img src="fig/stream_sub.png" alt="stream at = on r19 and r29, blocks 0 to 15, subtraction">
<figcaption>Prompts <i>a</i> − <i>b</i>.</figcaption></figure>

<h3>3. What block 15 writes along these directions</h3>
<p>The same coordinates for block 15's writes: its attention write mid<sub>15</sub> − pre<sub>15</sub>, the same
split by source position (BOS, <i>a</i>, op, <i>b</i>, “=”), and its MLP write pre<sub>16</sub> − mid<sub>15</sub>.
The first panel repeats mid<sub>15</sub>. All panels of a figure share the same axis limits.</p>
<figure><img src="fig/writes15_add.png" alt="block 15 writes on r19 and r29, addition">
<figcaption>Prompts <i>a</i> + <i>b</i>.</figcaption></figure>
<figure><img src="fig/writes15_sub.png" alt="block 15 writes on r19 and r29, subtraction">
<figcaption>Prompts <i>a</i> − <i>b</i>.</figcaption></figure>
<p class="note">Scripts: <code>readers/mod50_eq.py</code> (rebuild, split and plots; outputs in
<code>&lt;run&gt;/analysis/reader_writer/mod50_eq/</code>) and <code>readers/mod50_eq_page.py</code> (this page).</p>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "notebook_mod50")
    args = ap.parse_args()
    (args.out / "fig").mkdir(parents=True, exist_ok=True)
    att = json.loads((SRC / "attribution.json").read_text())
    per = json.loads(PERIODICITY.read_text())
    for f in FIGS:
        shutil.copy(SRC / "fig" / f, args.out / "fig" / f)
    fig_groups(att, args.out / "fig/mod50_eq_groups.png")
    entry = ('{\nid: "mod50_eq",\ntitle: "Mod 50: the block-15 readers g19 and g29 at =",\ndate: "2026-10-27",\nhtml: `'
             + page(att, per) + "`\n},\n")  # fmt: skip
    p = args.out / "entries.js"
    s = (
        p.read_text()
        if p.exists()
        else "// Notebook pages of the mod 50 report.\nwindow.NOTEBOOK = [\n];\n"
    )
    if 'id: "mod50_eq"' in s:
        i = s.index('{\nid: "mod50_eq"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)


if __name__ == "__main__":
    main()
