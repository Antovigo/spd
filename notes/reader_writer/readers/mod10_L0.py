"""Block-0 readers that respond to a mod 10, the directions they read, and what is left in the others.

Data: the alive-only model's inner activations (`../variance/extract_alive_only.py`) at position 1
(the token a; the prompt is `<BOS> a op b =`, so everything there is a function of a alone), for the
block-0 gate / up components kept by the learned mask "kl_int, sampled mask, λ=5e-05".

Notation (all also defined in the notebook entry):
  a          integer operand, 1..400
  c          a kept block-0 reader (gate or up component); K of them
  V_c        its read vector in R^4096 (column of V_<kind>); V = [V_c] (4096 x K)
  x(a)       the block-0 MLP input (post-RMSNorm) at position 1; t_c(a) = x(a) . V_c (stored)
  f_c(a)     = t_c(a) / ||V_c||, the plotted inner activation
  x_V(a)     orthogonal projection of x(a) onto span(V): V (V^T V)^+ t(a); exact for every reader
  trend      width-10 moving average over a (window a-5..a+4), which removes every period-10 part
  p_c(a)     f_c(a) - trend: the oscillating part, on A = {6..395} (39 full periods)
  m_c(r)     mean of p_c over a in A with a mod 10 = r, r in 0..9
  S10_c      sum_A m_c(a mod 10)^2 / sum_A (f_c - mean f_c)^2: share of f_c's variance that is
             periodic mod 10 (same definition with modulus q for the controls S_q)
  mu_r       in R^4096: mean over a in A, a mod 10 = r, of x_V(a) - trend; M = [mu_0..mu_9]

    python mod10_L0.py --out <notebook dir> [--tau 0.5]
"""

import argparse
import glob
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08")
SRC = RUN / "analysis/reader_writer/variance/alive_only/L0"
FILTER = RUN / "addsub-08-last-pos-40000"
HUB = (
    "/mnt/nw/home/a.vigouroux/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/*/"
)
N = 400
A = np.arange(1, N + 1)
VALID = (A >= 6) & (A <= 395)
CONTROLS = (7, 9, 11, 12, 13)
plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def kept_readers() -> tuple[np.ndarray, list[str], np.ndarray]:
    text = (FILTER / "ab_grids/pruned.js").read_text().strip()
    doc = json.loads(text[len("window.PRUNED = ") :].rstrip(";"))
    (mask,) = [m for m in doc["masks"] if m["name"] == "kl_sampled/lam_5e-05"]
    comps = np.load(SRC / "comps.npz")
    cols, names, vecs = [], [], []
    for kind in ("gate", "up"):
        sel = np.nonzero(comps["kind"] == kind)[0]
        drop = set(mask["pruned"].get(f"layers.0.mlp.{kind}_proj", []))
        for j, col in enumerate(sel):
            cid = int(comps["index"][col])
            if cid not in drop:
                cols.append(int(col))
                names.append(f"{kind[0]}{cid}")
                vecs.append(comps[f"V_{kind}"][:, j])
    return np.asarray(cols), names, np.stack(vecs, axis=1).astype(np.float64)


def moving_trend(f: np.ndarray, q: int) -> np.ndarray:
    """Width-q moving average along axis 0 (window a - q//2 .. a + q - q//2 - 1); NaN at the edges."""
    out = np.full(f.shape, np.nan)
    lo = q // 2
    cs = np.concatenate([np.zeros((1, *f.shape[1:])), np.cumsum(f, axis=0)])
    for i in range(lo, N - (q - lo) + 1):
        out[i] = (cs[i - lo + q] - cs[i - lo]) / q
    return out


def class_means(p: np.ndarray, q: int, valid: np.ndarray) -> np.ndarray:
    """(q, ...) mean of p over valid a with a mod q = r."""
    return np.stack([p[valid & (A % q == r)].mean(axis=0) for r in range(q)])


def periodic_share(f: np.ndarray, q: int) -> np.ndarray:
    valid = (q < A) & (N - q >= A)
    p = f - moving_trend(f, q)
    m = class_means(p, q, valid)
    num = (m[A[valid] % q] ** 2).sum(axis=0)
    den = ((f[valid] - f[valid].mean(axis=0)) ** 2).sum(axis=0)
    return num / den


def bf16_tensor(path: str, name: str, rows: np.ndarray | None = None) -> np.ndarray:
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        info = header[name]
        assert info["dtype"] == "BF16", info
        start, end = info["data_offsets"]
        shape = info["shape"]
        base = 8 + hlen + start
        if rows is None:
            fh.seek(base)
            raw = np.frombuffer(fh.read(end - start), np.uint16).reshape(shape)
        else:
            width = shape[1] * 2
            raw = np.empty((rows.size, shape[1]), np.uint16)
            for i, r in enumerate(rows):
                fh.seek(base + int(r) * width)
                raw[i] = np.frombuffer(fh.read(width), np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def embedding_inputs() -> np.ndarray:
    """gamma ⊙ e(a) / rms(e(a)) for a = 1..400: the MLP input if attention added nothing."""
    from transformers import AutoTokenizer

    (SNAP,) = glob.glob(HUB)

    tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
    ids = []
    for a in A:
        enc = tok.encode(str(a), add_special_tokens=False)
        assert len(enc) == 1, (a, enc)
        ids.append(enc[0])
    idx = json.loads(Path(SNAP, "model.safetensors.index.json").read_text())["weight_map"]
    e = bf16_tensor(
        SNAP + idx["model.embed_tokens.weight"], "model.embed_tokens.weight", np.asarray(ids)
    )
    gamma = bf16_tensor(
        SNAP + idx["model.layers.0.post_attention_layernorm.weight"],
        "model.layers.0.post_attention_layernorm.weight",
    )
    return gamma * e / np.sqrt((e**2).mean(axis=1, keepdims=True) + 1e-5)


def principal_cosines(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    qx, _ = np.linalg.qr(X)
    qy, _ = np.linalg.qr(Y)
    return np.linalg.svd(qx.T @ qy, compute_uv=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    fig_dir = args.out / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    cols, names, V = kept_readers()
    K = len(names)
    vnorm = np.linalg.norm(V, axis=0)
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    rows = (A - 1) * N  # op "+", b = 1
    T = np.asarray(inner[rows, 0, :][:, cols], np.float64)  # (400, K) raw t_c
    # position 1 must not depend on b or op
    dev = 0.0
    rng = np.random.default_rng(0)
    for b, op in zip(rng.integers(1, N + 1, 8), rng.integers(0, 2, 8), strict=True):
        other = np.asarray(inner[op * N * N + (A - 1) * N + b - 1, 0, :][:, cols], np.float64)
        dev = max(dev, float(np.abs(other - T).max() / np.abs(T).max()))
    res["max_rel_dev_over_b_op"] = dev
    F = T / vnorm

    # 1) mod-10 responders
    s10 = periodic_share(F, 10)
    ctrl = np.stack([periodic_share(F, q) for q in CONTROLS])
    s2, s5 = periodic_share(F, 2), periodic_share(F, 5)
    sel = np.nonzero(s10 >= args.tau)[0]
    res["K"] = K
    res["tau"] = args.tau
    res["control_max"] = {str(q): float(ctrl[i].max()) for i, q in enumerate(CONTROLS)}
    res["scores"] = {
        n: {
            "S10": float(s10[i]),
            "S5": float(s5[i]),
            "S2": float(s2[i]),
            "Sctrl_max": float(ctrl[:, i].max()),
        }
        for i, n in enumerate(names)
    }
    res["mod10"] = [names[i] for i in sel]
    order = np.argsort(-s10)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(K), s10[order], color="#2a5db0", label="S₁₀ (period 10)")
    ax.plot(
        np.arange(K),
        ctrl[:, order].max(axis=0),
        "k.",
        ms=3,
        label="max S_q, q ∈ {7, 9, 11, 12, 13}",
    )
    ax.axhline(args.tau, color="#c0392b", lw=1, ls="--", label=f"τ = {args.tau}")
    ax.set_xticks(np.arange(K)[: len(sel) + 10])
    ax.set_xticklabels([names[i] for i in order[: len(sel) + 10]], rotation=90, fontsize=6)
    ax.set_ylabel("share of variance periodic")
    ax.set_title(
        f"block-0 readers at position 1 (a), sorted by S₁₀; {len(sel)} of {K} have S₁₀ ≥ τ"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_s10_sorted.png")
    plt.close(fig)

    p = F - moving_trend(F, 10)
    m = class_means(p, 10, VALID)  # (10, K)
    msel = m[:, sel]
    prof = msel / np.abs(msel).max(axis=0)
    pk = np.argsort(np.argmax(msel, axis=0) * 10 + np.argmin(msel, axis=0))
    fig, ax = plt.subplots(figsize=(4.2, 0.18 * len(sel) + 1.2))
    im = ax.imshow(prof[:, pk].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([names[sel[i]] for i in pk], fontsize=7)
    ax.set_xticks(range(10))
    ax.set_xlabel("r = a mod 10")
    ax.set_title("m_c(r) / max_r |m_c(r)|")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_mod10_profiles.png")
    plt.close(fig)
    res["profiles"] = {names[i]: m[:, i].tolist() for i in sel}

    show = [n for n in ("u54", "u60") if n in names] + [
        names[i] for i in order[:4] if names[i] not in ("u54", "u60")
    ]
    fig, axs = plt.subplots(len(show), 1, figsize=(9, 1.5 * len(show)), sharex=True)
    for axx, n in zip(np.atleast_1d(axs), show, strict=True):
        i = names.index(n)
        axx.plot(A, F[:, i], lw=0.8, color="#2a5db0")
        axx.plot(A, moving_trend(F[:, [i]], 10)[:, 0], lw=0.8, color="#999")
        axx.set_ylabel(n)
        axx.text(
            0.995,
            0.9,
            f"S₁₀ = {s10[i]:.2f}",
            transform=axx.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    np.atleast_1d(axs)[-1].set_xlabel("a")
    fig.suptitle("f_c(a) (blue) and its width-10 moving average (grey)")
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_examples.png")
    plt.close(fig)

    # 2) directions
    G = V.T @ V
    W = V @ np.linalg.pinv(G)  # x_V(a) = W t(a)
    pT = T - moving_trend(T, 10)
    mu = W @ class_means(pT, 10, VALID).T  # (4096, 10)
    res["mu_mean_norm_ratio"] = float(
        np.linalg.norm(mu.mean(axis=1)) / np.linalg.norm(mu, axis=0).mean()
    )
    U, sv, _ = np.linalg.svd(mu - mu.mean(axis=1, keepdims=True), full_matrices=False)
    res["sv_mu"] = sv.tolist()
    # explained share of the selected readers' periodic and total variance by the top-k directions
    xc = (W @ T.T).T  # (400, 4096) x_V(a)
    xc = xc - xc.mean(axis=0)
    ps = p[VALID][:, sel]
    fs = F[VALID][:, sel] - F[VALID][:, sel].mean(axis=0)
    ev_p, ev_f = [], []
    for k in range(10):
        Uk = U[:, :k]
        proj = (xc @ Uk) @ (Uk.T @ V[:, sel]) / vnorm[sel]
        pp = proj - moving_trend(proj, 10)
        ev_p.append(float(1 - ((ps - pp[VALID]) ** 2).sum() / (ps**2).sum()))
        e = proj[VALID] - proj[VALID].mean(axis=0)
        ev_f.append(float(1 - ((fs - e) ** 2).sum() / (fs**2).sum()))
    res["ev_periodic_vs_k"] = ev_p
    res["ev_total_vs_k"] = ev_f
    kstar = next(k for k in range(10) if ev_p[k] >= 0.95)
    res["kstar"] = kstar
    cos = (mu / np.linalg.norm(mu, axis=0)).T @ (mu / np.linalg.norm(mu, axis=0))
    muc = mu - mu.mean(axis=1, keepdims=True)
    cosc = (muc / np.linalg.norm(muc, axis=0)).T @ (muc / np.linalg.norm(muc, axis=0))
    res["cos_mu"] = cos.tolist()
    res["cos_mu_centered"] = cosc.tolist()
    # DFT over r: Psi_k = sum_r mu_r exp(-2 pi i k r / 10)
    rr = np.arange(10)
    psi = {k: mu @ np.exp(-2j * np.pi * k * rr / 10) for k in range(1, 6)}
    energy = {k: float(np.linalg.norm(psi[k]) ** 2 * (1 if k == 5 else 2)) for k in psi}
    tot = sum(energy.values())
    res["fourier_energy_share"] = {str(k): energy[k] / tot for k in energy}
    planes = []
    for k in range(1, 5):
        re, im_ = psi[k].real, psi[k].imag
        planes.append(
            (
                k,
                float(np.dot(re, im_) / np.linalg.norm(re) / np.linalg.norm(im_)),
                float(np.linalg.norm(re) / np.linalg.norm(im_)),
            )
        )
    res["plane_re_im_cos_and_ratio"] = planes
    basis = np.stack(
        [psi[k].real for k in range(1, 6)] + [psi[k].imag for k in range(1, 5)], axis=1
    )
    bn = basis / np.linalg.norm(basis, axis=0)
    res["fourier_basis_cos"] = (bn.T @ bn).tolist()

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    axs[0].bar(range(1, sv.size + 1), sv**2 / (sv**2).sum(), color="#2a5db0")
    axs[0].set_xlabel("singular direction j")
    axs[0].set_ylabel("share of ‖M − mean‖²")
    axs[0].set_title("SVD of the residue means μ_r")
    axs[1].plot(range(10), ev_p, "o-", label="periodic part p_c")
    axs[1].plot(range(10), ev_f, "s-", label="whole f_c")
    axs[1].axhline(0.95, color="#999", lw=0.8, ls="--")
    axs[1].set_xlabel("n (top-n singular directions of M)")
    axs[1].set_ylabel("variance explained")
    axs[1].set_title(f"selected readers ({len(sel)}); n* = {kstar}")
    axs[1].legend(frameon=False)
    im = axs[2].imshow(cosc, cmap="RdBu_r", vmin=-1, vmax=1)
    axs[2].set_xticks(range(10))
    axs[2].set_yticks(range(10))
    axs[2].set_xlabel("r")
    axs[2].set_ylabel("r′")
    axs[2].set_title("cos(μ_r − μ̄, μ_r′ − μ̄)")
    fig.colorbar(im, ax=axs[2], shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_mu_svd_cos.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 6, figsize=(15, 2.8))
    ks = list(range(1, 6))
    axs[0].bar(ks, [res["fourier_energy_share"][str(k)] for k in ks], color="#2a5db0")
    axs[0].set_xlabel("frequency k (period 10/k)")
    axs[0].set_title("energy of Ψ_k")
    for axx, k in zip(axs[1:5], range(1, 5), strict=True):
        re, im_ = psi[k].real, psi[k].imag
        e1 = re / np.linalg.norm(re)
        e2 = im_ - np.dot(im_, e1) * e1
        e2 /= np.linalg.norm(e2)
        pts = np.stack([muc.T @ e1, muc.T @ e2], axis=1)
        axx.plot(*np.vstack([pts, pts[:1]]).T, "-", color="#ccc", lw=0.8)
        axx.scatter(*pts.T, c=rr, cmap="tab10", s=20)
        for r in rr:
            axx.annotate(str(r), pts[r], fontsize=7, xytext=(3, 3), textcoords="offset points")
        axx.set_aspect("equal")
        axx.set_title(f"plane of Ψ_{k}")
    imc = axs[5].imshow(np.abs(bn.T @ bn), cmap="Greys", vmin=0, vmax=1)
    lab = [f"Re{k}" for k in range(1, 6)] + [f"Im{k}" for k in range(1, 5)]
    axs[5].set_xticks(range(9))
    axs[5].set_xticklabels(lab, rotation=90, fontsize=6)
    axs[5].set_yticks(range(9))
    axs[5].set_yticklabels(lab, fontsize=6)
    axs[5].set_title("|cos| between Re/Im Ψ_k")
    fig.colorbar(imc, ax=axs[5], shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_fourier.png")
    plt.close(fig)

    Vs = V[:, sel]
    Pd = U[:, :9].T @ Vs
    res["sel_V_share_in_D"] = dict(
        zip([names[i] for i in sel], ((Pd**2).sum(0) / (Vs**2).sum(0)).tolist(), strict=True)
    )
    Pn = Pd / np.linalg.norm(Pd, axis=0)
    gram = Pn.T @ Pn
    res["sel_V_cos_in_D"] = gram.tolist()
    ordr = np.argsort([int(np.argmax(np.abs(msel[:, j]))) for j in range(len(sel))], kind="stable")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].bar(
        range(len(sel)), [res["sel_V_share_in_D"][names[sel[j]]] for j in ordr], color="#2a5db0"
    )
    axs[0].set_xticks(range(len(sel)))
    axs[0].set_xticklabels(
        [f"{names[sel[j]]} (r={int(np.argmax(np.abs(msel[:, j])))})" for j in ordr],
        rotation=90,
        fontsize=7,
    )
    axs[0].set_ylim(0, 1)
    axs[0].set_title("‖Π_D V_c‖² / ‖V_c‖²")
    im = axs[1].imshow(gram[np.ix_(ordr, ordr)], cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_xticks(range(len(sel)))
    axs[1].set_xticklabels([names[sel[j]] for j in ordr], rotation=90, fontsize=7)
    axs[1].set_yticks(range(len(sel)))
    axs[1].set_yticklabels([names[sel[j]] for j in ordr], fontsize=7)
    axs[1].set_title("cos(Π_D V_c, Π_D V_c′)")
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_reader_vectors.png")
    plt.close(fig)

    # embedding hypothesis
    xe = embedding_inputs()  # (400, 4096)
    pe = xe - moving_trend(xe, 10)
    mue = class_means(pe, 10, VALID).T  # (4096, 10)
    Pv = V @ np.linalg.pinv(G) @ V.T
    mue_v = Pv @ mue
    muec = mue - mue.mean(axis=1, keepdims=True)
    muevc = mue_v - mue_v.mean(axis=1, keepdims=True)
    res["principal_cos_data_vs_emb_projected"] = principal_cosines(muc, muevc).tolist()
    res["principal_cos_data_vs_emb_full"] = principal_cosines(muc, muec).tolist()
    res["share_of_emb_mod10_in_spanV"] = float(
        np.linalg.norm(muevc) ** 2 / np.linalg.norm(muec) ** 2
    )
    pred = (mue_v.T @ V[:, sel]) / vnorm[sel]  # (10, |sel|) embedding-only residue profiles
    pred = pred - pred.mean(axis=0)
    obs = msel - msel.mean(axis=0)
    per_c = [float(np.corrcoef(pred[:, j], obs[:, j])[0, 1]) for j in range(len(sel))]
    res["emb_profile_corr"] = dict(zip([names[i] for i in sel], per_c, strict=True))
    res["emb_profile_scale"] = dict(
        zip(
            [names[i] for i in sel],
            (np.linalg.norm(obs, axis=0) / np.linalg.norm(pred, axis=0)).tolist(),
            strict=True,
        )
    )
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    pc = res["principal_cos_data_vs_emb_projected"]
    axs[0].bar(range(1, len(pc) + 1), pc, color="#2a5db0")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("principal angle index")
    axs[0].set_ylabel("cos")
    axs[0].set_title("span{μ_r − μ̄} vs span{Π_V(ε_r − ε̄)}")
    axs[1].hist(per_c, bins=20, range=(-1, 1), color="#2a5db0")
    axs[1].set_xlabel("corr over r of m_c(r) with V_c·Π_V ε_r / ‖V_c‖")
    axs[1].set_title(f"selected readers ({len(sel)})")
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_embedding.png")
    plt.close(fig)

    # 3) other readers minus the k* mod-10 directions
    Uk = U[:, :kstar]
    others = np.setdiff1d(np.arange(K), sel)
    E = (xc @ Uk) @ (Uk.T @ V[:, others]) / vnorm[others]  # (400, |others|)
    Fo = F[:, others] - F[:, others].mean(axis=0)
    R = Fo - E
    change = np.linalg.norm(E, axis=0) / np.linalg.norm(Fo, axis=0)
    oo = np.argsort(-change)
    res["others_change"] = {names[others[j]]: float(change[j]) for j in oo}
    res["others_resid_S10"] = {
        names[others[j]]: float(periodic_share(R[:, [j]], 10)[0]) for j in oo
    }
    res["others_orig_S10"] = {names[others[j]]: float(s10[others[j]]) for j in oo}
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(range(others.size), change[oo], color="#2a5db0")
    ax.set_xticks(range(min(40, others.size)))
    ax.set_xticklabels([names[others[j]] for j in oo[:40]], rotation=90, fontsize=6)
    ax.set_ylabel("‖w_c‖ / ‖f_c − mean f_c‖")
    ax.set_title(
        f"other readers ({others.size}): size of the part along the n* = {kstar} mod-10 directions"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_others_change.png")
    plt.close(fig)
    top = oo[:24]
    fig, axs = plt.subplots(12, 2, figsize=(12, 15), sharex=True)
    for axx, j in zip(axs.T.ravel(), top, strict=True):
        axx.plot(A, Fo[:, j], lw=0.7, color="#bbb", label="f_c − mean f_c")
        axx.plot(A, R[:, j], lw=0.8, color="#2a5db0", label="residual ρ_c")
        axx.set_ylabel(names[others[j]], fontsize=8)
        axx.text(
            0.995,
            0.92,
            f"change {change[j]:.2f}",
            transform=axx.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
    axs[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    axs[-1, 0].set_xlabel("a")
    axs[-1, 1].set_xlabel("a")
    fig.suptitle(
        "other block-0 readers, ordered by change: f_c − mean f_c (grey) and ρ_c = f_c − mean f_c − w_c (blue)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "l0_others_residual.png")
    plt.close(fig)
    np.savez(
        args.out / "l0_mod10.npz",
        names=np.asarray(names),
        F=F,
        s10=s10,
        mu=mu,
        U=U,
        sv=sv,
        sel=sel,
        others=others,
        E=E,
        R=R,
        change=change,
        mue=mue,
    )
    (args.out / "l0_mod10.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k
                not in (
                    "scores",
                    "profiles",
                    "cos_mu",
                    "fourier_basis_cos",
                    "others_change",
                    "others_resid_S10",
                    "others_orig_S10",
                    "emb_profile_scale",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
