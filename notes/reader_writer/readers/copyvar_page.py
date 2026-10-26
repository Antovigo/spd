"""Section 12 of the notebook page "Mod 10 representations": why the copies of a and b at "=" vary in size (figures from
copyvar_figs.py; numbers from p1_magnitude.json, b_subclouds.json, weight_qk_split.json, p21_vary.json, q21_vary.json
and logit_writers.json).

Writes the section between <!-- copyvar:begin --> and <!-- copyvar:end -->, after section 11 of the mod10 entry.

    python copyvar_page.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

from copy_build import OUT

BEGIN, END = "<!-- copyvar:begin -->", "<!-- copyvar:end -->"


def f2(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    J = {k: json.loads((OUT / f"{k}.json").read_text())
         for k in ("p1_magnitude", "b_subclouds", "weight_qk_split", "p21_vary", "q21_vary", "logit_writers")}  # fmt: skip
    pm, bs, qk, pv, qv, lw = (J[k] for k in J)
    ca, co = pm["alive-only"]["copy_alone"], pm["original"]["copy_alone"]
    pa, po = pv["alive-only"], pv["original"]
    ba, bo = bs["alive-only"], bs["original"]
    sh = {k: v["share"] for k, v in qk.items()}
    r2 = {k: v["factor_r2_within"] for k, v in qk.items()}

    def cv(c):
        return f"{min(c['rho_within_cv_by_residue']):.2f}–{max(c['rho_within_cv_by_residue']):.2f}"

    def top(side, n):
        return ", ".join(
            f"block {t['writer'].split('.')[0][1:]} {'MLP down' if t['writer'].split('.')[1] == 'mlp' else 'attention output'} "
            f"component {t['writer'].split('.')[2]} ({f2(t['share'])})"
            for t in lw[f"{side}_top"][:n]
        )

    small_a = [v for k, v in ba["small_rate_by_a"].items() if k not in ("7", "48", "89")]
    small_o = [v for k, v in bo["small_rate_by_a"].items() if k not in ("7", "48", "89")]
    html = f"""{BEGIN}
<h3>12. Why the copies vary in size</h3>
<p>In the copy figure of section 10, the copy of <i>a</i> at "=" does not form ten round clouds in its first Fourier plane.
Each residue's points are drawn out into a streak that points away from the origin: the copies of one residue share a
direction but not a length. The copy of <i>b</i> looks different. Each residue has a big cloud and a much smaller one,
close to the origin. This section traces both back to the copying head's attention weight, and the attention weight
back to its query and keys. It uses the 2,000 prompts per set of section 11, in both the alive-only model and the original
model (<code>p1_magnitude.py</code>, <code>b_subclouds.py</code>, <code>weight_qk_split.py</code>,
<code>p21_vary.py</code>, <code>q21_vary.py</code>, <code>logit_writers.py</code>, <code>copyvar_figs.py</code>,
<code>copyvar_page.py</code>).</p>
<ul>
<li><b>Notation.</b> The <i>copy</i> <i>O</i> is the attention write at "=" (position 4) from the source position: block 16's write from
position 1 for <i>a</i>, block 15's write from position 3 for <i>b</i>. <i>P</i><sub>1</sub> is the <i>k</i> = 1 Fourier plane of the copy's
own residue means, and the residue <i>r</i> (0 … 9) is the copied operand mod 10. For a prompt of residue <i>r</i>, ρ (a real number) is
the copy's coordinate along the unit vector of the residue-<i>r</i> mean in <i>P</i><sub>1</sub>: its length along its residue's
direction. <i>p</i> (0 … 1) is the attention weight of the copying head at "=" on the source: head 21 for <i>a</i>, head 13 for
<i>b</i>. The copying head's own part of ρ is <i>p</i> × <i>c</i>, where <i>c</i> is ρ of that head's value write at weight 1. For
<i>a</i>, <i>c</i> depends on <i>a</i> only, because position 1 sees only &lt;BOS&gt; and <i>a</i>.</li>
<li><b>Within-residue shares.</b> All the spreads here are measured inside each residue's cloud: around the residue mean of the
quantity, pooled over the ten residues. This is the only centring in the section, and it is what "spread" means: the size
of each cloud, not the distance between clouds. A part's share is the covariance of its within-residue deviation with the
total's deviation, divided by the total's within-residue variance. The shares of a split sum to 1, and a share can be negative.</li>
</ul>
<figure><img src="notebook/fig/copyvar_clouds.png"><figcaption>The copies at "=" in their <i>k</i> = 1 plane, each point a prompt, coloured
by the copying head's weight <i>p</i> on the source. Top: copy of <i>a</i> (block 16, head 21). Bottom: copy of <i>b</i> (block 15,
head 13). Left: alive-only model; right: original model. Red digits mark the residue means; the cross is the activation
zero.</figcaption></figure>
<p><b>The streaks of <i>a</i> are lengths.</b> {f2(ca["radial_share_of_within_spread"])} of each cloud's spread in <i>P</i><sub>1</sub>
is along its residue's direction in the alive-only model, and {f2(co["radial_share_of_within_spread"])} in the original. ρ varies
by {cv(ca)} of its mean within a residue (standard deviation over mean) in the alive-only model, and by {cv(co)} in the
original. Head 21 carries {f2(ca["share_head21"])} of the spread of ρ (alive-only) and {f2(co["share_head21"])} (original).</p>
<p><b>Most of each streak is head 21's weight.</b> Head 21's part <i>p</i> × <i>c</i> splits to first order into the weight
(the deviation of <i>p</i> times the residue's mean <i>c</i>), the value (the mean <i>p</i> times the deviation of <i>c</i>) and the
product of the two deviations. The weight is {f2(ca["head21_weight_p21"])} of head 21's spread in the alive-only model and
{f2(co["head21_weight_p21"])} in the original. The value is {f2(ca["head21_value_c"])} and {f2(co["head21_value_c"])}, and the product
{f2(ca["head21_product"])} and {f2(co["head21_product"])}. The value <i>c</i> differs between values of <i>a</i> that share a residue by
{f2(ca["within_cv_c"])} (alive-only) and {f2(co["within_cv_c"])} (original) of its mean, and it is larger for smaller <i>a</i>
(within-residue correlation of <i>c</i> with <i>a</i>: {f2(ca["corr_within_c_a"])} and {f2(co["corr_within_c_a"])}).</p>
<p><b>The weight depends on the operator, on <i>b</i> and on <i>a</i>.</b> Head 21's weight on <i>a</i> is
{f2(pa["p1_mean_sd"][0])} ± {f2(pa["p1_mean_sd"][1])} (mean ± standard deviation over prompts) in the alive-only model and
{f2(po["p1_mean_sd"][0])} ± {f2(po["p1_mean_sd"][1])} in the original. With "+" it stays at
{min(v for k, v in pa["p1_mean_by_op_b"].items() if k.startswith("op0")):.2f}–{max(v for k, v in pa["p1_mean_by_op_b"].items() if k.startswith("op0")):.2f}
(alive-only) and {min(v for k, v in po["p1_mean_by_op_b"].items() if k.startswith("op0")):.2f}–{max(v for k, v in po["p1_mean_by_op_b"].items() if k.startswith("op0")):.2f}
(original) for every <i>b</i>. With "−" it falls as <i>b</i> grows: from {f2(pa["p1_mean_by_op_b"]["op1_b3"])} at <i>b</i> = 3 to
{f2(pa["p1_mean_by_op_b"]["op1_b331"])} at <i>b</i> = 331 (alive-only), and from {f2(po["p1_mean_by_op_b"]["op1_b3"])} to
{f2(po["p1_mean_by_op_b"]["op1_b331"])} (original). It also rises with <i>a</i>: {f2(pa["p1_mean_by_a_quintile"][0])} for <i>a</i> = 1–80 and
{f2(pa["p1_mean_by_a_quintile"][4])} for <i>a</i> = 321–400 (alive-only); {f2(po["p1_mean_by_a_quintile"][0])} and
{f2(po["p1_mean_by_a_quintile"][4])} (original). When the weight is low, it goes to <i>b</i> in the alive-only model: in the tenth of
prompts with the lowest weight, head 21 puts {f2(pa["pattern_lowest_decile_of_p1"][1])} on <i>a</i> and
{f2(pa["pattern_lowest_decile_of_p1"][3])} on <i>b</i>. In the original model it goes to <i>b</i> and &lt;BOS&gt; alike:
{f2(po["pattern_lowest_decile_of_p1"][1])} on <i>a</i>, {f2(po["pattern_lowest_decile_of_p1"][3])} on <i>b</i>,
{f2(po["pattern_lowest_decile_of_p1"][0])} on &lt;BOS&gt;.</p>
<p><b>The sub-clouds of <i>b</i> are the same effect, with two levels.</b> Split each residue's prompts into two groups by ρ over
the residue's mean ρ (a two-means split). The small cloud holds {f2(ba["fraction_small"])} of the prompts in the alive-only model and
{f2(bo["fraction_small"])} in the original. In it, head 13 puts {f2(ba["by_cloud"]["small"]["p13_mean"])} (alive-only) and
{f2(bo["by_cloud"]["small"]["p13_mean"])} (original) of its weight on <i>b</i> and {f2(ba["by_cloud"]["small"]["pattern_head13_mean"][0])} and
{f2(bo["by_cloud"]["small"]["pattern_head13_mean"][0])} on &lt;BOS&gt;. In the big cloud it puts {f2(ba["by_cloud"]["big"]["p13_mean"])} and
{f2(bo["by_cloud"]["big"]["p13_mean"])} on <i>b</i>. The small cloud is mostly "−" prompts and the three smallest values of <i>a</i>. With
"+", {f2(ba["small_rate_by_op"]["0"])} of the prompts fall in it (alive-only) and {f2(bo["small_rate_by_op"]["0"])} (original). With
"−", {f2(ba["small_rate_by_op"]["1"])} and {f2(bo["small_rate_by_op"]["1"])} do. For <i>a</i> = 7, 48 and 89 the rate is
{f2(ba["small_rate_by_a"]["7"])}, {f2(ba["small_rate_by_a"]["48"])} and {f2(ba["small_rate_by_a"]["89"])} (alive-only), and
{f2(bo["small_rate_by_a"]["7"])}, {f2(bo["small_rate_by_a"]["48"])} and {f2(bo["small_rate_by_a"]["89"])} (original). For the
seven other values of <i>a</i> (130 … 376) it is {min(small_a):.2f}–{max(small_a):.2f} and {min(small_o):.2f}–{max(small_o):.2f}.
Head 13's value write is also smaller in the small cloud, but by less than its weight: <i>c</i> over the residue's mean ρ is
{f2(ba["by_cloud"]["small"]["c_over_residue_mean_rho"])} against {f2(ba["by_cloud"]["big"]["c_over_residue_mean_rho"])} (alive-only), and
{f2(bo["by_cloud"]["small"]["c_over_residue_mean_rho"])} against {f2(bo["by_cloud"]["big"]["c_over_residue_mean_rho"])} (original).</p>
<p><b>Where the weight's variation enters: query or keys.</b> The weight <i>p</i> is a softmax over the scores of the head's query at
"=" against its keys at positions 0 … 4. Three factors can move it: the query, the key at the source, and the keys at the
other four positions. Recompute <i>p</i> with any subset of the factors replaced by its mean over prompts. The Shapley value of each
factor, per prompt, then gives an exact split of <i>p</i> into the three factors. Their shares of the within-residue variance of
<i>p</i> are:</p>
<table><tr><th>copying head</th><th>model</th><th>query at "="</th><th>key at the source</th><th>keys at the other positions</th></tr>
<tr><td>head 21 on <i>a</i> (block 16)</td><td>alive-only</td><td>{f2(sh["a/alive-only"]["Q"])}</td><td>{f2(sh["a/alive-only"]["K_src"])}</td><td>{f2(sh["a/alive-only"]["K_rest"])}</td></tr>
<tr><td>head 21 on <i>a</i> (block 16)</td><td>original</td><td>{f2(sh["a/original"]["Q"])}</td><td>{f2(sh["a/original"]["K_src"])}</td><td>{f2(sh["a/original"]["K_rest"])}</td></tr>
<tr><td>head 13 on <i>b</i> (block 15)</td><td>alive-only</td><td>{f2(sh["b/alive-only"]["Q"])}</td><td>{f2(sh["b/alive-only"]["K_src"])}</td><td>{f2(sh["b/alive-only"]["K_rest"])}</td></tr>
<tr><td>head 13 on <i>b</i> (block 15)</td><td>original</td><td>{f2(sh["b/original"]["Q"])}</td><td>{f2(sh["b/original"]["K_src"])}</td><td>{f2(sh["b/original"]["K_rest"])}</td></tr></table>
<figure><img src="notebook/fig/copyvar_split.png"><figcaption>Left: the within-residue radial spread of the copy of <i>a</i> in its
<i>k</i> = 1 plane, split into the heads other than 21 and head 21's weight, value and product terms. Right: the within-residue variance
of each copying head's weight on its source, split exactly (Shapley) between its query at "=", its key at the source and its
keys at the other positions.</figcaption></figure>
<ul>
<li><b>In the original model, the query decides.</b> The query carries {f2(sh["a/original"]["Q"])} of head 21's variation and
{f2(sh["b/original"]["Q"])} of head 13's. Its Shapley value follows the operator and the other operand: within-residue R² on the
(<i>b</i>, operator) pair {f2(r2["a/original"]["Q"]["other operand and op"])} for head 21, and on the (<i>a</i>, operator) pair
{f2(r2["b/original"]["Q"]["other operand and op"])} for head 13.</li>
<li><b>In the alive-only model, the keys decide.</b> For head 21, the keys at the other positions carry
{f2(sh["a/alive-only"]["K_rest"])}; their Shapley value follows (<i>b</i>, operator) with R²
{f2(r2["a/alive-only"]["K_rest"]["other operand and op"])}. For head 13, the key at <i>b</i>'s own position carries
{f2(sh["b/alive-only"]["K_src"])}; it follows (<i>a</i>, operator) with R² {f2(r2["b/alive-only"]["K_src"]["other operand and op"])},
since position 3 sees <i>a</i> and the operator as well as <i>b</i>.</li>
<li><b>The key at <i>a</i> plays a small part.</b> It carries {f2(sh["a/alive-only"]["K_src"])} (alive-only) and
{f2(sh["a/original"]["K_src"])} (original) of head 21's variation, and follows <i>a</i> itself (R²
{f2(r2["a/alive-only"]["K_src"]["copied operand"])} and {f2(r2["a/original"]["K_src"]["copied operand"])}).</li>
</ul>
<p><b>The query of head 21 in the alive-only model is one direction.</b> Of the 11 query components of block 16, one (component 139)
carries {qv["top_query_components"][0]["share_of_var_D13"]:.3f} of the variance of the score difference
<i>s</i><sub>1</sub> − <i>s</i><sub>3</sub>, where <i>s<sub>t</sub></i> is head 21's score at "=" on position <i>t</i>. Its inner activation (the
component's read of the normed stream at "=") is {qv["top_query_components"][0]["inner_act_mean_by_op"][0]:.1f} on "+" prompts and
{qv["top_query_components"][0]["inner_act_mean_by_op"][1]:.1f} on "−" prompts, and varies by 6% of its mean across prompts. Block 15
writes most of it: its MLP {f2(qv["top_query_components"][0]["inner_act_share_by_source"]["M15"])} and its attention
{f2(qv["top_query_components"][0]["inner_act_share_by_source"]["A15"])} of the variance, against
{f2(qv["top_query_components"][0]["inner_act_share_by_source"]["X15"])} from the stream entering block 15.</p>
<p><b>Which writers move head 21's score on <i>a</i>.</b> The score <i>s</i><sub>1</sub> itself (not the weight) is
{f2(lw["s1_mean_sd"][0])} ± {f2(lw["s1_mean_sd"][1])} in the alive-only model, and <i>a</i> explains {f2(lw["r2_s1"]["a"])} of its variance.
The stream entering block 16 at a position is a sum of writes: the token embedding plus each writer of blocks 0–15
(attention output components and MLP down components), each its inner activation times its write vector. Since the query
and the key are linear in the normed stream, <i>s</i><sub>1</sub> splits exactly into one term per writer, either at "=" (through the
query, with the actual key) or at <i>a</i> (through the key, with the actual query).</p>
<ul>
<li><b>Through the key.</b> Moving the key with the query at its mean gives {f2(lw["var_share_key_variation_at_mean_query"])} of
the variance of <i>s</i><sub>1</sub>. All the writers at position 1 depend on <i>a</i> only. The largest are {top("key", 4)}. By block:
MLP 15 {f2(lw["key_by_block"]["L15.mlp"])}, MLP 14 {f2(lw["key_by_block"]["L14.mlp"])}, MLP 13 {f2(lw["key_by_block"]["L13.mlp"])}.
{lw["key_n_writers_for_80pct_positive"]} writers make up 80%.</li>
<li><b>Through the query.</b> Moving the query with the key at its mean gives {f2(lw["var_share_query_variation_at_mean_key"])}.
The largest writers at "=" are {top("query", 3)}. Over prompts, the R² of their inner activations on the operator is
{", ".join(f2(t["y_r2"]["op"]) for t in lw["query_top"][:3])}, and on the (<i>b</i>, operator) pair
{", ".join(f2(t["y_r2"]["op_b"]) for t in lw["query_top"][:3])}. By block: MLP 15 {f2(lw["query_by_block"]["L15.mlp"])}, attention 15 {f2(lw["query_by_block"]["L15.attn"])},
MLP 14 {f2(lw["query_by_block"]["L14.mlp"])}. {lw["query_n_writers_for_80pct_positive"]} writers make up 80%.</li>
</ul>
<p>The score on <i>a</i> is only one of the five scores in head 21's softmax. In the alive-only model, head 21's score on <i>b</i>
varies more than its score on <i>a</i>: standard deviations {qv["s3_sd"]:.2f} and {qv["s1_sd"]:.2f} over prompts.</p>
{END}"""
    p = args.out / "entries.js"
    s = p.read_text()
    if BEGIN in s:
        s = s[: s.index(BEGIN)] + html + s[s.index(END) + len(END) :]
    else:
        anchor = "<!-- copyfig:end -->"
        assert anchor in s
        s = s.replace(anchor, anchor + "\n" + html, 1)
    p.write_text(s)


if __name__ == "__main__":
    main()
