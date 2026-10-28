"""Page of the "Mod 50 representations" notebook: the block-15 readers g19, g29 (gate) and u32 (up) at "=" and where
their input comes from (results of mod50_eq.py and mod50_chain.py).

Data: <run>/analysis/reader_writer/mod50_eq/{attribution.json, chain.json, fig/*.png} (mod50_eq.py, mod50_chain.py)
and the periodicity index
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
READERS = ("L15.gate.19", "L15.gate.29", "L15.up.32")
APPLET = "https://claude.ai/artifact/2SnrbErKDYntUN1pR8xib2"
OPS = ("+", "-")
SOURCES = ("BOS", "a", "op", "b", "=")
FIGS = (
    "stream_add.png",
    "stream_sub.png",
    "writes15_add.png",
    "writes15_sub.png",
    "chain_classmeans_add.png",
    "chain_classmeans_sub.png",
    "chain_pattern_add.png",
    "chain_pattern_sub.png",
)
POS, NEG = "#2a5db0", "#c0392b"


def fmt(x: float, n: int = 2) -> str:
    return f"{x:+.{n}f}".replace("-", "−")


def short(c: str) -> str:
    return {"gate": "g", "up": "u"}[c.split(".")[1]] + c.split(".")[-1]


def fig_groups(att: dict, path: Path) -> None:
    """Group shares per (reader, op): each sublayer of blocks 0..14, the embedding, block 15's attention by source."""
    plt.rcParams.update({"figure.dpi": 110, "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})  # fmt: skip
    keys = ["emb"] + [f"L{b}.{s}" for b in range(15) for s in ("attn", "mlp")]
    keys += [f"L15.attn.{s}" for s in SOURCES]
    fig, axs = plt.subplots(6, 1, figsize=(12, 13), sharex=True, layout="constrained")
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


def chain_section(chn: dict) -> str:
    C = chn["readers"]
    cols = [(c, op) for c in READERS for op in OPS]
    head = "".join(f"<th>{short(c)}, a {op} b</th>" for c, op in cols)
    keys = [("E (block-15 attention from b)", "E<sub>c</sub>: block-15 attention from <i>b</i>"),
            ("E const pattern", "E<sub>c</sub><sup>const</sup>: constant pattern"),
            ("E b-part of pattern", "E<sub>c</sub><sup>b</sup>: part of the pattern that depends on <i>b</i> alone"),
            ("E rest of pattern", "E<sub>c</sub><sup>rest</sup>: rest of the pattern"),
            ("pre_15 term", "(r<sub>c</sub> · pre<sub>15</sub>) / ρ: everything before block 15's attention")]  # fmt: skip
    rows = "".join(f"<tr><td>{lab}</td>{''.join(f'<td>{fmt(C[c][op]['share'][k])}</td>' for c, op in cols)}</tr>"
                   for k, lab in keys)  # fmt: skip
    span = "".join(
        f"<tr><td>{lab}</td>"
        + "".join(
            f"<td>{min(C[c][op]['classmean'][k]):+.2f} … {max(C[c][op]['classmean'][k]):+.2f}</td>"
            for c, op in cols
        )  # fmt: skip
        + "</tr>"
        for k, lab in [("f", "f<sub>c</sub>"), *keys]
    ).replace("-", "−")
    hd = chn["heads"]
    heads = sorted({x["head"] for c in READERS for op in OPS for x in C[c][op]["main_heads"]})
    hrows = "".join(
        f"<tr><td>{h}</td>" + "".join(
            f"<td>{hd[op]['pbar'][h]:.3f}</td><td>{hd[op]['rms_beta'][h]:.3f}</td><td>{hd[op]['rms_gamma'][h]:.3f}</td>"
            f"<td>{min(hd[op]['classmean_p'][h]):.3f} … {max(hd[op]['classmean_p'][h]):.3f}</td>" for op in OPS)
        + "".join(f"<td>{next((fmt(x['share_in_f']) for x in C[c]['+']['main_heads'] if x['head'] == h), '·')}</td>"
                  for c in READERS)
        + "</tr>"
        for h in heads
    )  # fmt: skip

    def otab(c: str, op: str) -> str:
        r = C[c][op]
        body = "".join(
            f"<tr><td><code>{x['o']}</code></td><td>{x['w']:+.3f}</td><td>{fmt(x['share_in_f'])}</td>"
            f"<td>{x['share_of_t_from_b']:.2f}</td><td>{x['r2_const_pattern']:.3f}</td>"
            f"<td>{', '.join(f'{v["v"][4:]} {fmt(v["share_in_t3"])}' for v in x['main_v'][:6])}</td></tr>"
            for x in r["main_o"][:10]
        ).replace("-", "−")
        vs = ", ".join(f"<code>{x['v']}</code> {fmt(x['share_in_f'])}" for x in r["main_v"][:10])
        return (f"<h4>{short(c)}, prompts a {op} b</h4><p>{len(r['main_o'])} main o components (the ten largest are"
                f" listed); {len(r['main_v'])} main v components, the ten largest by share of f<sub>c</sub>: {vs}."
                "</p><div class=\"tw\"><table><tr><th>o</th><th>w<sub>co</sub></th><th>share in f<sub>c</sub></th>"
                "<th>share of t<sub>o</sub> from <i>b</i></th><th>r² constant pattern</th>"
                "<th>largest v components of t<sub>o</sub><sup>3</sup> (share)</th></tr>" + body + "</table></div>")  # fmt: skip

    otabs = "".join(otab(c, op) for c, op in cols)
    hhead = "".join(
        f"<th>{lab}, a {op} b</th>" for op in OPS for lab in ("p̄", "rms β", "rms γ", "class means")
    )
    hhead += "".join(f"<th>share in {short(c)} (a + b)</th>" for c in READERS)
    return f"""
<h3>4. The attention term rewritten through block 15's o and v components</h3>
<p>This section follows the part of each reader's activation that block 15's attention reads from the token <i>b</i>
back to the activations of block 15's <code>v</code> components at position 3, and asks whether the attention
pattern's dependence on <i>b</i> carries any of the mod-50 signal. Notation, besides that of the setup:</p>
<ul>
<li><i>h</i> ∈ {{0, …, 31}}: a query head; kv(<i>h</i>) = ⌊<i>h</i>/4⌋ ∈ {{0, …, 7}} its key/value head;
<i>d</i> ∈ {{0, …, 127}} a coordinate of a head. <b>p<sub>h3</sub></b>(<i>x</i>): head <i>h</i>'s attention weight
from “=” to the token <i>b</i> (position 3).</li>
<li><i>o</i>: an alive <code>o</code> component of block 15, read vector <i>V<sub>o</sub></i> ∈ ℝ<sup>4096</sup>
(entries indexed by <i>h</i>·128 + <i>d</i>), write vector <i>U<sub>o</sub></i>. <b>t<sub>o</sub></b>(<i>x</i>):
its activation at “=”; t<sub>o</sub><sup>3</sup>: the part read from position 3. <b>w<sub>co</sub></b> =
r<sub>c</sub> · <i>U<sub>o</sub></i>, a constant.</li>
<li><i>k</i>: an alive <code>v</code> component of block 15, read vector <i>V<sub>k</sub></i> ∈ ℝ<sup>4096</sup>,
write vector <i>U<sub>k</sub></i> ∈ ℝ<sup>1024</sup> (entries indexed by kv·128 + <i>d</i>). <b>τ<sub>k</sub></b>(<i>x</i>)
= <i>V<sub>k</sub></i> · RMSNorm(pre<sub>15</sub> at position 3): its activation at the token <i>b</i>.</li>
<li><b>B<sub>ohk</sub></b> = Σ<sub><i>d</i></sub> <i>V<sub>o</sub></i>[<i>h</i>·128 + <i>d</i>]
<i>U<sub>k</sub></i>[kv(<i>h</i>)·128 + <i>d</i>]: the constant that carries <code>v</code> component <i>k</i> into
<code>o</code> component <i>o</i> through head <i>h</i>.</li>
<li><b>Pattern split</b>, per operator: p<sub>h3</sub>(<i>x</i>) = p̄<sub>h</sub> + β<sub>h</sub>(<i>b</i>) +
γ<sub>h</sub>(<i>x</i>). p̄<sub>h</sub> is the mean of p<sub>h3</sub> over the operator's 160,000 prompts;
β<sub>h</sub>(<i>b</i>) is its mean over the 400 prompts with that <i>b</i>, minus p̄<sub>h</sub> (the part that
depends on <i>b</i> alone); γ<sub>h</sub> is the rest (what depends on <i>a</i>).</li>
<li><b>Main</b> components: the largest |share| first, until their |shares| add up to at least 90% of the sum of all
|shares|. <b>Class mean</b> of a quantity: its mean over the 3,200 prompts of one operator with <i>b</i> mod 50 =
<i>r</i>, <i>r</i> ∈ {{0, …, 49}}.</li>
</ul>
<p><b>The rewrite.</b> Exactly, on every prompt:</p>
<p>f<sub>c</sub> = [ r<sub>c</sub> · pre<sub>15</sub> + Σ<sub><i>o</i></sub> w<sub>co</sub> t<sub>o</sub> ] / ρ,
&nbsp;&nbsp; t<sub>o</sub><sup>3</sup> = Σ<sub><i>h</i></sub> p<sub>h3</sub> Σ<sub><i>k</i></sub> B<sub>ohk</sub>
τ<sub>k</sub>,</p>
<p>so the source-<i>b</i> term E<sub>c</sub> = Σ<sub><i>o</i></sub> w<sub>co</sub> t<sub>o</sub><sup>3</sup> / ρ =
Σ<sub><i>h</i></sub> Σ<sub><i>k</i></sub> p<sub>h3</sub> τ<sub>k</sub> A<sub>chk</sub> / ρ with A<sub>chk</sub> =
Σ<sub><i>o</i></sub> w<sub>co</sub> B<sub>ohk</sub>. Replacing p<sub>h3</sub> by p̄<sub>h</sub>, β<sub>h</sub> or
γ<sub>h</sub> splits E<sub>c</sub> exactly into E<sub>c</sub><sup>const</sup> + E<sub>c</sub><sup>b</sup> +
E<sub>c</sub><sup>rest</sup>. With the constant pattern alone, t<sub>o</sub><sup>3</sup> ≈ Σ<sub><i>k</i></sub>
W<sub>ok</sub> τ<sub>k</sub>, W<sub>ok</sub> = Σ<sub><i>h</i></sub> p̄<sub>h</sub> B<sub>ohk</sub>: each o
component's activation from <i>b</i> is a fixed linear combination of the v activations at <i>b</i>. Checks: these
identities hold to {chn["checks"]["E_vs_proj_rel_max"]:.0e} of the largest term against the rebuild, and the rebuilt
τ<sub>k</sub> match the stored v activations to 2% of their largest value.</p>
<p><b>Shares of the reader's activation</b> (uncentred, Σ term · f<sub>c</sub> / Σ f<sub>c</sub><sup>2</sup>):</p>
<div class="tw"><table><tr><th></th>{head}</tr>{rows}</table></div>
<p><b>Range of the class means over <i>b</i> mod 50</b> (smallest … largest):</p>
<div class="tw"><table><tr><th></th>{head}</tr>{span}</table></div>
<figure><img src="fig/chain_classmeans_add.png" alt="class means over b mod 50 of the reader and its terms, addition">
<figcaption>Class means over <i>b</i> mod 50, prompts <i>a</i> + <i>b</i>: the reader f<sub>c</sub> (black), the
source-<i>b</i> term E<sub>c</sub> (blue) and its constant-pattern part (blue dashed), the parts carried by
β (orange) and γ (grey dotted), and f<sub>c</sub> − E<sub>c</sub> (green).</figcaption></figure>
<figure><img src="fig/chain_classmeans_sub.png" alt="class means over b mod 50, subtraction">
<figcaption>The same for prompts <i>a</i> − <i>b</i>.</figcaption></figure>
<p><b>The attention weight on <i>b</i></b>, for the main heads of the three readers: p̄<sub>h</sub>, the rms over
<i>b</i> = 1, …, 400 of β<sub>h</sub>, the rms over prompts of γ<sub>h</sub>, the range of the class means of
p<sub>h3</sub> over <i>b</i> mod 50, and the head's share in each reader (· where the head is not main for that
reader; head term E<sub>ch</sub> = p<sub>h3</sub> Σ<sub><i>k</i></sub> τ<sub>k</sub> A<sub>chk</sub> / ρ).</p>
<div class="tw"><table><tr><th>head</th>{hhead}</tr>{hrows}</table></div>
<figure><img src="fig/chain_pattern_add.png" alt="class means of the attention weight on b, addition">
<figcaption>Class means of p<sub>h3</sub> over <i>b</i> mod 50 for the main heads, prompts <i>a</i> + <i>b</i>.
</figcaption></figure>
<figure><img src="fig/chain_pattern_sub.png" alt="class means of the attention weight on b, subtraction">
<figcaption>The same for prompts <i>a</i> − <i>b</i>.</figcaption></figure>
<p><b>Main o components and their v components.</b> For each reader and operator: the main <code>o</code>
components (by share of f<sub>c</sub> of e<sub>co</sub> = w<sub>co</sub> t<sub>o</sub> / ρ, t<sub>o</sub> the whole
activation at “=”), the share of t<sub>o</sub> that comes from <i>b</i> (Σ t<sub>o</sub><sup>3</sup> t<sub>o</sub> /
Σ t<sub>o</sub><sup>2</sup>), the r² of the constant-pattern formula for t<sub>o</sub><sup>3</sup>, and its largest
<code>v</code> components by share of t<sub>o</sub><sup>3</sup> (term τ<sub>k</sub> Σ<sub><i>h</i></sub>
p<sub>h3</sub> B<sub>ohk</sub>). The main <code>v</code> components of the reader are ranked by their share of
f<sub>c</sub> (term τ<sub>k</sub> Σ<sub><i>h</i></sub> p<sub>h3</sub> A<sub>chk</sub> / ρ).</p>
{otabs}
<p class="note">Script: <code>readers/mod50_chain.py</code> (outputs <code>chain.json</code> and the figures above in
<code>&lt;run&gt;/analysis/reader_writer/mod50_eq/</code>).</p>
"""


def page(att: dict, per: dict, chn: dict) -> str:
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
<p><b>Summary.</b> At the token “=”, the block-15 readers <code>g19</code>, <code>g29</code> (gate) and
<code>u32</code> (up) carry a pattern of <i>b</i> mod 50. Their activation there splits exactly into one term per
upstream writer. The block-15
attention reading from the token <i>b</i> supplies {lo:.2f}–{hi:.2f} of it, depending on the reader and the operator.
The other source positions of that attention supply almost nothing. The rest comes from many small writers of blocks
0–14; none of them reaches 0.13. On the read directions of the two readers, the stream at “=” stays in small
clouds up to block 15's attention. After that attention the points spread along a band about 0.5 long, with the
colours of <i>b</i> mod 50 sorted along it, and the source-<i>b</i> write alone draws that band. Rewritten through
block 15's attention, the source-<i>b</i> term is a fixed linear combination of the <code>v</code> activations at the
token <i>b</i>, weighted by an attention pattern that is close to constant: head 13 carries most of it, with a
weight on <i>b</i> of 0.97–0.99 (<i>a</i> + <i>b</i>) and 0.81–0.87 (<i>a</i> − <i>b</i>) across the classes of
<i>b</i> mod 50. The pattern's dependence on <i>b</i> changes the class means by at most 0.07, against a span of
2.3 to 3.1 for the readers themselves. A rotatable 3D view of the three readers is at <a href="{APPLET}">Mod 50 reader
space</a>.</p>

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
<code>g29</code> is <code>L15.gate.29</code>, <code>u32</code> is <code>L15.up.32</code>. <i>c</i> denotes one of
these three readers, <i>V<sub>c</sub></i> ∈
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

<h3>1. The readers g19, g29 and u32 at “=” and where their input comes from</h3>
<p>Periodicity shares in <i>b</i> at “=”: <code>g19</code> {pshares("L15.gate.19")}; <code>g29</code>
{pshares("L15.gate.29")}; <code>u32</code> {pshares("L15.up.32")}.</p>
<p><b>Checks.</b> The rebuilt activations of the block-15 <code>o</code> components (summed over the sources) match
the stored ones to {ck["o15_rel_max"]:.3f} of their largest value, and the rebuilt f<sub>c</sub> matches the stored
reader activations with r² = {min(ck["f_r2_vs_stored"]):.5f}. Σ e<sub>w</sub> = f<sub>c</sub> holds on every
prompt to float64 precision.</p>
<div class="tw"><table><tr><th></th>{head}</tr>{table}</table></div>
<figure><img src="fig/mod50_eq_groups.png" alt="shares by sublayer for g19, g29 and u32, both operators">
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
<code>&lt;run&gt;/analysis/reader_writer/mod50_eq/</code>) and <code>readers/mod50_eq_page.py</code> (this page).
The 3D view of <code>g19</code>, <code>g29</code> and <code>u32</code> at the input of block 15's MLP is at
<a href="{APPLET}">Mod 50 reader space</a> (<code>readers/mod50_applet.py</code>).</p>
""" + chain_section(chn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "notebook_mod50")
    args = ap.parse_args()
    (args.out / "fig").mkdir(parents=True, exist_ok=True)
    att = json.loads((SRC / "attribution.json").read_text())
    per = json.loads(PERIODICITY.read_text())
    chn = json.loads((SRC / "chain.json").read_text())
    for f in FIGS:
        shutil.copy(SRC / "fig" / f, args.out / "fig" / f)
    fig_groups(att, args.out / "fig/mod50_eq_groups.png")
    entry = ('{\nid: "mod50_eq",\ntitle: "Mod 50: the block-15 readers g19, g29 and u32 at =",\ndate: "2026-10-28",\nhtml: `'
             + page(att, per, chn) + "`\n},\n")  # fmt: skip
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
