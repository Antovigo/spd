"""Figures, numbers and page of the "Mod 2 representations" notebook (block-3 down component d189).

Data: the float64 rebuild of the alive-only model at position 1 (track_build.py, track_mod10/stream.npz), the
representation repository (repr_repo/repo.py) and the exact variance split of the λ=5e-05 model
(variance_kl5e-5/results/L3). Nothing is centred or detrended: zero is the zero of the activation space.

Notation (also defined on the page):
  a          integer operand 1..400 at position 1; r = a mod 10; even / odd: a mod 2 = 0 / 1
  pre_l, mid_l   residual stream at position 1 entering block l / after block l's attention add
  out_w(a)   in R^4096, what term w adds to the residual stream (embedding, one attention sublayer,
             or one down component: t_w(a) U_w)
  x(s)(a)    = RMSNorm(s(a)) with block 4's post-attention gain: the block-4 MLP input of residual state s
  m_e, m_o   means of x over even / odd a; u = (m_e - m_o) / ||m_e - m_o||
  d'         ||m_e - m_o|| / within-class standard deviation of x . u (pooled over the two classes)
  gap        (min over even a of x . u - max over odd a of x . u) / ||m_e - m_o||
  A5(f)      mean over a of f(a) (-1)^a: the mod-2 amplitude of a scalar f

    python mod2_page.py --out notebook_mod2
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mod10_L0 import RUN, A
from track_build import OUT, load_blocks, rmsnorm

REPO = RUN / "analysis/representation_repository"
SPLIT = RUN / "analysis/reader_writer/variance_kl5e-5/results/L3"
WRITER = 189
EVEN, ODD = "#2a5db0", "#d9730d"
plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "repr_repo"))
from repo import Repo  # noqa: E402

odd = A % 2 == 1
sign = (-1.0) ** A


def col(block, kind: str, i: int) -> int:
    return list(block.index[block.kind == kind]).index(i)


def kde(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid[:, None] - x[None]) / bw) ** 2).sum(1) / (
        len(x) * bw * np.sqrt(2 * np.pi)
    )


def separation(x: np.ndarray, B: np.ndarray | None = None) -> dict:
    Z = x if B is None else x @ B
    me, mo = Z[~odd].mean(0), Z[odd].mean(0)
    d = me - mo
    u = d / np.linalg.norm(d)
    Re, Ro = Z[~odd] - me, Z[odd] - mo
    sw = np.sqrt((np.sum((Re @ u) ** 2) + np.sum((Ro @ u) ** 2)) / (len(Z) - 2))
    pe, po = Z[~odd] @ u, Z[odd] @ u
    out = {"dist": float(np.linalg.norm(d)), "dprime": float(np.linalg.norm(d) / sw),
           "gap": float((pe.min() - po.max()) / np.linalg.norm(d))}  # fmt: skip
    if B is not None:
        Sw = (Re.T @ Re + Ro.T @ Ro) / (len(Z) - 2)
        out["lda"] = float(np.sqrt(d @ np.linalg.solve(Sw, d)))
    return out


def compute() -> tuple[dict, dict, dict, np.ndarray]:
    blocks, _, _ = load_blocks()
    S = np.load(OUT / "stream.npz")
    pre = {k: S[f"pre_{k}"].astype(np.float64) for k in range(5)}
    mid = {k: S[f"mid_{k}"].astype(np.float64) for k in range(5)}
    b3, b4 = blocks[3], blocks[4]
    outs: dict[str, np.ndarray] = {"embed": pre[0]}
    tds = {}
    for blk in range(4):
        b = blocks[blk]
        _, td = b.mlp_from_t(*b.readers(mid[blk]))
        tds[blk] = td
        for j, i in enumerate(b.index[b.kind == "down"]):
            outs[f"L{blk}.down.{i}"] = td[:, j : j + 1] * b.m["U_down"][j][None]
    for blk in range(5):
        outs[f"L{blk}.attn"] = mid[blk] - pre[blk]
    assert np.linalg.norm(sum(outs.values()) - mid[4]) / np.linalg.norm(mid[4]) < 1e-6
    jw = col(b3, "down", WRITER)
    t189 = tds[3][:, jw]
    U = b3.m["U_down"][jw]
    w = outs[f"L3.down.{WRITER}"]
    mlp3 = pre[4] - mid[3]
    states = {"before": mid[4] - mlp3, "after": mid[4], "after_no_d189": mid[4] - w}
    X = {k: rmsnorm(v, b4.g2) for k, v in states.items()}
    B4 = np.load(REPO / "vectors/L4.p1.a_mod_10.means.npz")["basis"]
    res: dict = {"t189_by_r": [float(t189[r == A % 10].mean()) for r in range(10)],
                 "t189_even_odd": [float(t189[~odd].mean()), float(t189[odd].mean())]}  # fmt: skip
    res["sep"] = {k: {"full": separation(x), "B4": separation(x, B4)} for k, x in X.items()}
    me, mo = X["after"][~odd].mean(0), X["after"][odd].mean(0)
    u_after = (me - mo) / np.linalg.norm(me - mo)
    proj_common = {k: x @ u_after for k, x in X.items()}
    proj_own = {}
    for k, x in X.items():
        d = x[~odd].mean(0) - x[odd].mean(0)
        proj_own[k] = x @ (d / np.linalg.norm(d))
    # readers of d189 (variance split)
    z = np.load(SPLIT / "variance_pos1.npz")
    wcol = list(z["ids_down"]).index(WRITER)
    phi = [(float(z["phi_Mg"][wcol, j]), f"g{i}") for j, i in enumerate(z["ids_gate"])]
    phi += [(float(z["phi_Mu"][wcol, j]), f"u{i}") for j, i in enumerate(z["ids_up"])]
    res["phi"] = sorted(phi, key=lambda x: -abs(x[0]))[:6]
    # the two main gate readers on block 3's mod-10 representation
    rep3 = np.load(REPO / "vectors/L3.p1.a_mod_10.means.npz")
    mu3 = rep3["mu"]
    res["gates"] = {}
    for g in (30, 145):
        V = b3.m["V_gate"][:, col(b3, "gate", g)]
        resp = mu3 @ V
        E = np.abs(np.fft.fft(resp)) ** 2
        E[1:5] *= 2
        res["gates"][g] = {"resp": resp.tolist(), "freq_share": (E[:6] / E[:6].sum()).tolist()}
    cosg = b3.m["V_gate"][:, col(b3, "gate", 30)] @ b3.m["V_gate"][:, col(b3, "gate", 145)]
    res["gates_cos"] = float(cosg / np.linalg.norm(b3.m["V_gate"][:, col(b3, "gate", 30)])
                             / np.linalg.norm(b3.m["V_gate"][:, col(b3, "gate", 145)]))  # fmt: skip
    # geometry of the write in block 4's input
    gU = b4.g2 * U
    V4 = np.concatenate([b4.m["V_gate"], b4.m["V_up"]], 1)
    Q, s, _ = np.linalg.svd(V4, full_matrices=False)
    Q = Q[:, s > 1e-9 * s.max()]
    pv = Q @ (Q.T @ gU)
    mu4 = np.load(REPO / "vectors/L4.p1.a_mod_10.means.npz")["mu"]
    psi5 = ((-1.0) ** np.arange(10)) @ mu4
    res["write"] = {"in_V4": float(np.linalg.norm(pv) / np.linalg.norm(gU)),
                    "in_B4": float(np.linalg.norm(B4.T @ pv) / np.linalg.norm(pv)),
                    "cos_k5": float(pv @ psi5 / np.linalg.norm(pv) / np.linalg.norm(psi5))}  # fmt: skip
    # block-4 readers: share removed by deleting d189's write
    rd = Repo().readers
    tg, tu = b4.readers(mid[4])
    tg_d, tu_d = b4.readers(mid[4] - w)
    rows = []
    for kind, t, td_ in (("gate", tg, tg_d), ("up", tu, tu_d)):
        for j, i in enumerate(b4.index[b4.kind == kind]):
            p = rd.get(f"L4.{kind}.{i}", {}).get("positions", {}).get("1")
            r2 = p["r2_by_rep"].get("L4.p1.a_mod_10.means", 0.0) if p else 0.0
            if r2 < 0.5:
                continue
            rows.append({"id": f"{kind[0]}{i}", "r2": float(r2),
                         "removed": float(np.sum((t[:, j] - td_[:, j]) ** 2) / np.sum(t[:, j] ** 2)),
                         "A5": float(A5(t[:, j])), "A5_no": float(A5(td_[:, j])),
                         "rms": float(np.sqrt(np.mean(t[:, j] ** 2)))})  # fmt: skip
    res["b4_readers"] = sorted(rows, key=lambda x: -x["removed"])
    # u147: sources of its mod-2 amplitude, linearised at block 4's RMS scale
    j = col(b4, "up", 147)
    V = b4.m["V_up"][:, j]
    r4 = np.sqrt((mid[4] ** 2).mean(-1) + 1e-5)
    contrib = {k: (v @ (b4.g2 * V)) / r4 for k, v in outs.items()}
    a5 = {k: float(A5(c)) for k, c in contrib.items()}
    l0 = sum(v for k, v in a5.items() if k.startswith("L0.down"))
    res["u147"] = {"A5": float(A5(tu[:, j])), "A5_no": float(A5(tu_d[:, j])),
                   "t_by_r": [float(tu[r == A % 10, j].mean()) for r in range(10)],
                   "top": sorted(a5.items(), key=lambda x: -abs(x[1]))[:6], "L0_down_total": float(l0),
                   "in_B4": float(np.linalg.norm(B4.T @ V) / np.linalg.norm(V)),
                   "cos_k5": float(V @ psi5 / np.linalg.norm(V) / np.linalg.norm(psi5))}  # fmt: skip
    return res, proj_common, proj_own, t189


def A5(f: np.ndarray) -> float:
    return float((f * sign).mean())


def fig_distributions(proj: dict, path: Path, title: str) -> None:
    allv = np.concatenate(list(proj.values()))
    grid = np.linspace(allv.min() - 0.3, allv.max() + 0.3, 600)
    bw = 0.15 * np.std(proj["after"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True, sharey=True)
    for ax, k, name in (
        (axs[0], "before", "before block 3's MLP"),
        (axs[1], "after", "after block 3's MLP"),
    ):
        for cls, c, lab in ((~odd, EVEN, "even a"), (odd, ODD, "odd a")):
            ax.plot(grid, kde(proj[k][cls], grid, bw), color=c, lw=1.6, label=lab)
            if k == "after":
                ax.plot(grid, kde(proj["after_no_d189"][cls], grid, bw), color=c, lw=1.3, ls="--",
                        label=f"{lab}, without d189")  # fmt: skip
        ax.set_title(name)
        ax.set_xlabel("x · u")
        ax.axvline(0, color="#999", lw=0.6)
    axs[0].set_ylabel("density over a")
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_residues(proj: dict, t189: np.ndarray, path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    r = A % 10
    for k, ls, lab in (("before", "-", "before block 3's MLP"), ("after", "-", "after"),
                       ("after_no_d189", "--", "after, without d189")):  # fmt: skip
        m = [proj[k][r == q].mean() for q in range(10)]
        s = [proj[k][r == q].std() for q in range(10)]
        axs[0].errorbar(np.arange(10) + {"before": -0.15, "after": 0, "after_no_d189": 0.15}[k], m, yerr=s, ls=ls,
                        marker="o", ms=3, capsize=2, lw=1, label=lab,
                        color={"before": "#888", "after": "#1d1d1b", "after_no_d189": "#1d1d1b"}[k])  # fmt: skip
    axs[0].set_xticks(range(10))
    axs[0].set_xlabel("r = a mod 10")
    axs[0].set_ylabel("x · u  (mean ± sd over a)")
    axs[0].axhline(0, color="#999", lw=0.6)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].scatter(A[~odd], t189[~odd], s=5, color=EVEN, label="even a")
    axs[1].scatter(A[odd], t189[odd], s=5, color=ODD, label="odd a")
    axs[1].set_xlabel("a")
    axs[1].set_ylabel("t_189(a)")
    axs[1].axhline(0, color="#999", lw=0.6)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def f3(x: float) -> str:
    return f"{x:.3f}"


def page(res: dict) -> str:
    sp = res["sep"]
    names = {
        "before": "before block 3's MLP",
        "after_no_d189": "after, without d189",
        "after": "after block 3's MLP",
    }
    sep_rows = "".join(
        f"<tr><td>{names[k]}</td><td>{sp[k]['full']['dist']:.2f}</td><td>{sp[k]['full']['dprime']:.2f}</td>"
        f"<td>{sp[k]['full']['gap']:+.2f}</td><td>{sp[k]['B4']['dist']:.2f}</td><td>{sp[k]['B4']['dprime']:.2f}</td>"
        f"<td>{sp[k]['B4']['gap']:+.2f}</td><td>{sp[k]['B4']['lda']:.2f}</td></tr>"
        for k in ("before", "after_no_d189", "after")
    )
    phi_rows = "".join(
        f"<tr><td><code>{n}</code></td><td>{v:+.3f}</td></tr>" for v, n in res["phi"]
    )
    g = res["gates"]
    gate_rows = "".join(
        f"<tr><td><code>g{c}</code></td>"
        + "".join(f"<td>{v:+.2f}</td>" for v in g[c]["resp"])
        + "".join(f"<td>{v:.2f}</td>" for v in g[c]["freq_share"])
        + "</tr>"
        for c in (30, 145)
    )
    b4_rows = "".join(
        f"<tr><td><code>{x['id']}</code></td><td>{x['r2']:.2f}</td><td>{x['rms']:.2f}</td><td>{x['removed']:.3f}</td>"
        f"<td>{x['A5']:+.3f}</td><td>{x['A5_no']:+.3f}</td></tr>"
        for x in res["b4_readers"]
    )
    u = res["u147"]
    top_rows = "".join(f"<tr><td><code>{k}</code></td><td>{v:+.3f}</td></tr>" for k, v in u["top"])
    t = res["t189_by_r"]
    wr = res["write"]
    sa, sb, sn = sp["after"]["full"], sp["before"]["full"], sp["after_no_d189"]["full"]
    return f"""
<p><b>Summary.</b> At the token <i>a</i>, the block-3 down component <code>d189</code> is one of the first
components whose activation is binary in <i>a</i>: about {res["t189_even_odd"][0]:.2f} for even <i>a</i> and
{res["t189_even_odd"][1]:.2f} for odd <i>a</i>. It is driven mostly by one gate reader, <code>g30</code>, which reads
the parity line of block 3's mod-10 representation. <code>d189</code> writes back into the same mod-10 representation,
along its parity line, not into a separate subspace. Only the block-4 readers that are themselves binary pick the
write up. It is the largest single source of the parity signal of the block-4 reader <code>u147</code>. Block 3's
MLP raises the even/odd separation d′ at the block-4 MLP input from {sb["dprime"]:.2f} to {sa["dprime"]:.2f}.
Without <code>d189</code> the rest of that MLP would lower it to {sn["dprime"]:.2f}.</p>

<h3>Setup and notation</h3>
<p>Run p-ba8b0c08, step 40000. The prompts are <code>&lt;BOS&gt; a op b =</code>, with integer operands
<i>a</i>, <i>b</i> ∈ {{1, …, 400}} and op ∈ {{+, −}}. The model is the alive-only model: every component of the
ceiling filter <code>addsub-08-last-pos</code> is on, and every other component and the weight delta are off. It is
rebuilt in float64 at position 1 (the token <i>a</i>), where everything depends on <i>a</i> alone.
<b>Nothing on this page is centred or detrended.</b> The network has no biases, so zero is the zero of the
activation space. Script: <code>notes/reader_writer/readers/mod2_page.py</code> (branch
experiment/arith_representations).</p>
<ul>
<li><b>Residues.</b> <i>r</i> = <i>a</i> mod 10 ∈ {{0, …, 9}}. "Even" and "odd" mean <i>a</i> mod 2 = 0 and 1
(200 values each).</li>
<li><b>Components.</b> <code>g&lt;id&gt;</code>, <code>u&lt;id&gt;</code> and <code>d&lt;id&gt;</code> are gate,
up and down components of the block named in the text; the ids are the AB-grid ids. <i>V<sub>c</sub></i> ∈ ℝ<sup>4096</sup>
is a reader's read vector and <i>U<sub>w</sub></i> ∈ ℝ<sup>4096</sup> a down component's write vector.
<i>t<sub>189</sub></i>(<i>a</i>) is the inner activation of <code>d189</code>, so it adds
<i>t<sub>189</sub></i>(<i>a</i>) <i>U<sub>189</sub></i> to the residual stream.</li>
<li><b>Mod-10 representation.</b> <code>L&lt;l&gt;.p1.a_mod_10.means</code> in the representation repository is
the span of the ten residue means μ<sub><i>r</i></sub> ∈ ℝ<sup>4096</sup> of block <i>l</i>'s MLP input
(10 dimensions, including the direction shared by all residues). Its parity line is the direction of
Σ<sub><i>r</i></sub> (−1)<sup><i>r</i></sup> μ<sub><i>r</i></sub>, which is Fourier frequency <i>k</i> = 5 in <i>r</i>.
<i>B</i><sub>4</sub> is an orthonormal basis (4096 × 10) of the block-4 span.</li>
<li><b>Residual states.</b> mid<sub>4</sub>(<i>a</i>) ∈ ℝ<sup>4096</sup> is the residual stream after block 4's
attention. "After block 3's MLP" is mid<sub>4</sub>. "Before" is mid<sub>4</sub> minus block 3's whole MLP output.
"Without d189" is mid<sub>4</sub> minus <i>t<sub>189</sub></i> <i>U<sub>189</sub></i>. Block 4's attention is held
fixed in all three. Re-running it on the edited stream changes none of the block-4 reader numbers in section 2 at the third decimal.</li>
<li><b>Block-4 MLP input.</b> <i>x</i>(<i>a</i>) = RMSNorm(state(<i>a</i>)) with block 4's post-attention gain.
This is what block 4's readers read.</li>
<li><b>Separation axis.</b> <i>m<sub>e</sub></i>, <i>m<sub>o</sub></i> ∈ ℝ<sup>4096</sup> are the means of
<i>x</i> over even and odd <i>a</i>. <i>u</i> = (<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>) / ‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖
is a unit vector.</li>
<li><b>Mod-2 amplitude.</b> For a scalar <i>f</i>(<i>a</i>), A5(<i>f</i>) = mean over <i>a</i> of
<i>f</i>(<i>a</i>) (−1)<sup><i>a</i></sup>.</li>
</ul>

<h3>1. What drives d189</h3>
<p>The exact variance split of block 3 credits each share φ of <code>d189</code>'s variance over the prompts to one
reader (φ sums to 1 over readers). It was run on the model with only the λ = 5e-05 mask on. At position 1 one gate
dominates:</p>
<table><tr><th>reader</th><th>φ</th></tr>{phi_rows}</table>
<p>Both main gates are explained by block 3's mod-10 representation (r² 0.96 for <code>g30</code>, 0.97 for
<code>g145</code>). Inside it they read different directions; the cosine of their read vectors is
{res["gates_cos"]:.2f}. The table gives each gate's pre-activation on the ten residue means,
μ<sub><i>r</i></sub> · <i>V<sub>c</sub></i>, and the share of that 10-vector's energy at each Fourier frequency
<i>k</i> in <i>r</i>:</p>
<table><tr><th></th>{"".join(f"<th>r={q}</th>" for q in range(10))}{"".join(f"<th>k={q}</th>" for q in range(6))}</tr>{gate_rows}</table>
<p><code>g30</code> is mostly the parity line (<i>k</i> = 5), with a <i>k</i> = 1 part that makes it positive at
<i>r</i> = 1 and 9. <code>g145</code> is mostly <i>k</i> = 2 and 4. It is high at both <i>r</i> = 0 and <i>r</i> = 5,
so it carries <i>a</i> mod 5 = 0 rather than parity. The resulting activation of <code>d189</code> by residue is
{", ".join(f"{v:.2f}" for v in t)} for <i>r</i> = 0, …, 9 (right panel of figure 2).</p>

<h3>2. Where d189 writes</h3>
<p>Block 4's readers see the write through block 4's norm gain. Of that gained write vector,
{wr["in_V4"] * 100:.0f}% of the norm lies in the span of block 4's read vectors. Of that part,
{wr["in_B4"] * 100:.1f}% lies in the block-4 mod-10 span, and its cosine with that span's parity line is
{wr["cos_k5"]:.2f}. So <code>d189</code> writes back into the mod-10 representation, along its parity line.</p>
<p>The table lists every block-4 reader that the mod-10 representation explains (r² ≥ 0.5). "Removed" is
‖<i>t</i> − <i>t</i><sub>no d189</sub>‖² / ‖<i>t</i>‖² over <i>a</i> = 1..400, where <i>t</i> is the reader's
pre-activation and <i>t</i><sub>no d189</sub> the same without <code>d189</code>'s write. "rms" is the root-mean-square
of <i>t</i>.</p>
<table><tr><th>reader</th><th>mod-10 r²</th><th>rms <i>t</i></th><th>removed</th><th>A5(<i>t</i>)</th><th>A5 without d189</th></tr>{b4_rows}</table>
<p>The large-amplitude mod-10 readers barely move. The binary readers <code>g182</code>, <code>u147</code> and
<code>u854</code> lose 9–19% of their pre-activation.</p>

<h3>3. Where u147 reads its parity from</h3>
<p><code>u147</code>'s read vector lies {u["in_B4"] * 100:.0f}% (by norm) in the block-4 mod-10 span. Its cosine with
that span's parity line is {u["cos_k5"]:.2f}. Its mean pre-activation by residue is
{", ".join(f"{v:.2f}" for v in u["t_by_r"])} for <i>r</i> = 0, …, 9, and A5 = {u["A5"]:.3f}. The residual stream
mid<sub>4</sub> is exactly the sum of the token embedding, the attention outputs of blocks 0–4 and every down
component's write in blocks 0–3. Holding block 4's RMS scale fixed, A5 of <code>u147</code> splits over these terms
(the terms sum to A5 exactly). The largest are:</p>
<table><tr><th>term</th><th>share of A5</th></tr>{top_rows}</table>
<p>All block-0 down components together give {u["L0_down_total"]:+.3f}. Deleting <code>d189</code>'s write lowers
A5 from {u["A5"]:.3f} to {u["A5_no"]:.3f}.</p>

<h3>4. Does d189 sharpen the parity split?</h3>
<p>Three metrics compare the states. All are computed on <i>x</i> projected onto that state's own <i>u</i>:</p>
<ul>
<li>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖, the distance between the class means.</li>
<li>d′ = ‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖ / σ<sub>w</sub>, where σ<sub>w</sub> is the standard
deviation of <i>x</i> · <i>u</i> within each class, pooled over the two classes.</li>
<li>gap = (min over even <i>a</i> of <i>x</i> · <i>u</i> − max over odd <i>a</i> of <i>x</i> · <i>u</i>) /
‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖. It is positive when no even and odd <i>a</i> overlap.</li>
</ul>
<p>σ<sub>w</sub> is a spread around each class mean; the activations themselves are not centred. The metrics are
given in the full 4096-dimensional space and inside the block-4 mod-10 span <i>x</i> <i>B</i><sub>4</sub>. In the span,
"LDA d′" uses the full 10 × 10 within-class covariance Σ<sub>w</sub>:
√((<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>)<sup>T</sup> Σ<sub>w</sub><sup>−1</sup> (<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>)).</p>
<table><tr><th>state</th><th>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖</th><th>d′</th><th>gap</th><th>‖<i>m<sub>e</sub></i> − <i>m<sub>o</sub></i>‖ in <i>B</i><sub>4</sub></th><th>d′ in <i>B</i><sub>4</sub></th><th>gap in <i>B</i><sub>4</sub></th><th>LDA d′ in <i>B</i><sub>4</sub></th></tr>{sep_rows}</table>
<p>Block 3's MLP moves the class means apart and turns the overlap into a margin. Without <code>d189</code> the same
MLP makes the classes overlap (negative gap).</p>
<figure><img src="fig/mod2_l3_dist.png" alt="Density of x·u for even and odd a, before and after block 3's MLP">
<figcaption>Figure 1. Density over <i>a</i> (Gaussian kernel) of <i>x</i>(<i>a</i>) · <i>u</i>, where <i>u</i> is the axis
of the "after" state. Blue: even <i>a</i>; orange: odd <i>a</i>. Left: before block 3's MLP. Right: after it (solid)
and after it without <code>d189</code>'s write (dashed). One axis for all states, so positions are comparable.</figcaption></figure>
<figure><img src="fig/mod2_l3_dist_own.png" alt="The same densities, each state on its own axis">
<figcaption>Figure 1b. As figure 1, but each state projected on its own axis <i>u</i> (the axis the table's metrics use).</figcaption></figure>
<figure><img src="fig/mod2_l3_residues.png" alt="Projection by residue and d189 activation">
<figcaption>Figure 2. Left: mean ± standard deviation over <i>a</i> of <i>x</i> · <i>u</i> (the "after" axis) for each residue
<i>r</i>, in the three states. Right: <i>t</i><sub>189</sub>(<i>a</i>) against <i>a</i>, coloured by parity.</figcaption></figure>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res, proj_common, proj_own, t189 = compute()
    (args.out / "mod2_l3.json").write_text(json.dumps(res, indent=1))
    fig_distributions(
        proj_common,
        fig_dir / "mod2_l3_dist.png",
        "projection on u of the state after block 3's MLP",
    )
    fig_distributions(proj_own, fig_dir / "mod2_l3_dist_own.png", "each state on its own u")
    fig_residues(proj_common, t189, fig_dir / "mod2_l3_residues.png")
    entry = ('{\nid: "mod2_l3",\ntitle: "Mod 2: block-3 d189 and the parity line",\ndate: "2026-10-22",\nhtml: `'
             + page(res) + "`\n},\n")  # fmt: skip
    p = args.out / "entries.js"
    s = (
        p.read_text()
        if p.exists()
        else "// Notebook pages of the mod 2 report (index.html).\nwindow.NOTEBOOK = [\n];\n"
    )
    if 'id: "mod2_l3"' in s:
        i = s.index('{\nid: "mod2_l3"')
        j = s.index("`\n},\n", i) + len("`\n},\n")
        s = s[:i] + entry + s[j:]
    else:
        assert s.rstrip().endswith("];")
        s = s.rstrip()[:-2] + entry + "];\n"
    p.write_text(s)
    print(json.dumps({k: res[k] for k in ("sep", "write", "t189_even_odd")}, indent=1))


if __name__ == "__main__":
    main()
