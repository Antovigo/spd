"""How attention copies the mod-10 representations to the last position "=" (alive-only model, copy_build.py streams).

Two copies: a mod 10 from position 1 by block 16 (prompt set "a"), b mod 10 from position 3 by block 15 (set "b").
Nothing is centred except where a spread around a class mean is the quantity measured. Notation, for one copy
(block L, source position s, residue r = a mod 10 or b mod 10):
  X_j        residual stream at position j entering block L; x_j = RMSNorm(X_j) (the attention's input norm)
  t_c(j)     inner activation x_j . V_c of value component c; v(j) = sum_c t_c(j) U_c (8 KV heads x 128)
  p_h        attention weight of head h (h = 0..31) at "=" on position s; kv(h) = h // 4 its KV head
  O_s        the attention output at "=" that comes from position s: sum_h p_h v_kv(h)(s) through the o projection
             (= sum over o components of (head outputs . V_o) U_o); O_s^h the part through head h
  Psi_k, P_k, D: Fourier components (k = 1..5), planes and 9-dim period-10 subspace of a field's residue means
  gains      g_k = sqrt(lambda_k(out) / lambda_k(in)), lambda_k the Fourier-diagonal of the Gram matrix of the residue
             means (k = 0 is the direction shared by all residues); offdiag = share of the frequency 1..5 Gram energy
             off the Fourier diagonal (0 = residues equally spaced on circles)
Writes <run>/analysis/reader_writer/copy_mod10/{copy.json, copy_geo.npz} (figures: copy_page.py).

    python copy_analyze.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy_build import OUT
from mlp0_mod10 import rope
from scipy.optimize import linprog
from track_build import Block, load_blocks, rmsnorm

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
F10 = np.exp(-2j * np.pi * np.outer(np.arange(10), np.arange(10)) / 10) / np.sqrt(10)
AB = Path("/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/addsub-08-last-pos-40000/ab_grids")


def means(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.stack([Z[lab == r].mean(0) for r in range(10)])


def psi(Z: np.ndarray, lab: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * np.outer(np.arange(1, 6), np.arange(10)) / 10) @ means(Z, lab)


def planes(Z: np.ndarray, lab: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    P = []
    for p in psi(Z, lab):
        u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
        P.append(u[:, s > 1e-9 * s.max()])
    return P, np.linalg.qr(np.concatenate(P, 1))[0]


def gram_fourier(mu: np.ndarray) -> np.ndarray:
    return F10 @ (mu @ mu.T) @ F10.conj().T


def offdiag(mu: np.ndarray) -> float:
    H = gram_fourier(mu)[1:, 1:]
    return float((np.abs(H - np.diag(np.diag(H))) ** 2).sum() / (np.abs(H) ** 2).sum())


def lam(mu: np.ndarray) -> np.ndarray:
    d = gram_fourier(mu).diagonal().real
    return np.asarray([d[0]] + [d[k] + d[10 - k] for k in range(1, 5)] + [d[5]])


def cos(P: np.ndarray, Q: np.ndarray) -> list[float]:
    return np.linalg.svd(P.T @ Q, compute_uv=False).tolist()


def loo_acc(z: np.ndarray, lab: np.ndarray) -> float:
    c = means(z, lab)
    n = np.bincount(lab, minlength=10)
    d = ((z[:, None, :] - c[None]) ** 2).sum(-1)
    own = (c[lab] * n[lab, None] - z) / (n[lab, None] - 1)
    d[np.arange(len(z)), lab] = ((z - own) ** 2).sum(-1)
    return float((d.argmin(1) == lab).mean())


def ovr_separable(z: np.ndarray, lab: np.ndarray) -> int:
    Z = np.concatenate([z, np.ones((len(z), 1))], 1)
    ok = 0
    for r in range(10):
        y = np.where(lab == r, 1.0, -1.0)
        res = linprog(np.zeros(Z.shape[1]), A_ub=-(y[:, None] * Z), b_ub=-np.ones(len(z)),
                      bounds=[(None, None)] * Z.shape[1], method="highs")  # fmt: skip
        ok += int(res.status == 0)
    return ok


def attention_parts(blk: Block, X: np.ndarray) -> dict:
    H, KV, dh = blk.cfg["num_attention_heads"], blk.cfg["num_key_value_heads"], 128
    n, T, _ = X.shape
    x = rmsnorm(X, blk.g1)
    rep = np.arange(H) // (H // KV)
    q = rope(blk.proj(x[:, 4], "q").reshape(n, H, dh), 4, blk.cfg)
    k = np.stack(
        [rope(blk.proj(x[:, t], "k").reshape(n, KV, dh), t, blk.cfg) for t in range(T)], 1
    )[:, :, rep]
    tv = np.stack([x[:, t] @ blk.a["V_v"] for t in range(T)], 1)  # (n, T, Cv)
    v = (tv @ blk.a["U_v"]).reshape(n, T, KV, dh)
    s = np.einsum("nhd,nthd->nht", q, k) / np.sqrt(dh)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return {"x": x, "p": p, "tv": tv, "v": v, "rep": rep}


def copied(
    blk: Block,
    parts: dict,
    s: int,
    heads: np.ndarray | None = None,
    vcomps: np.ndarray | None = None,
) -> np.ndarray:
    """O_s restricted to the given heads and value components."""
    p, tv, rep = parts["p"], parts["tv"], parts["rep"]
    n = p.shape[0]
    t = tv[:, s] if vcomps is None else tv[:, s] * np.isin(np.arange(tv.shape[2]), vcomps)
    v = (t @ blk.a["U_v"]).reshape(n, 8, 128)[:, rep]  # (n, 32, 128)
    z = p[:, :, s, None] * v
    if heads is not None:
        z = z * np.isin(np.arange(32), heads)[None, :, None]
    return blk.proj(z.reshape(n, 32 * 128), "o")


def per_frequency(mu_in: np.ndarray, mu_out: np.ndarray) -> dict:
    li, lo = lam(mu_in), lam(mu_out)
    return {"gain": np.sqrt(lo / li).tolist(), "offdiag_in": offdiag(mu_in), "offdiag_out": offdiag(mu_out),
            "sv_period10_in": np.linalg.svd((F10.conj().T @ np.diag(np.r_[0, np.ones(9)]) @ F10).real @ mu_in, compute_uv=False)[:9].tolist(),
            "sv_period10_out": np.linalg.svd((F10.conj().T @ np.diag(np.r_[0, np.ones(9)]) @ F10).real @ mu_out, compute_uv=False)[:9].tolist()}  # fmt: skip


def analyse(
    blk: Block, X: np.ndarray, lab: np.ndarray, other: np.ndarray, s: int, X_next4: np.ndarray
) -> tuple[dict, dict]:
    parts = attention_parts(blk, X)
    Ocp = copied(blk, parts, s)
    xs = parts["x"][:, s]
    Ps, Ds = planes(xs, lab)
    Po, Do = planes(Ocp, lab)
    P4, D4 = planes(X[:, 4], lab)
    Pn, Dn = planes(X_next4, lab)
    mu_s, mu_o = means(xs, lab), means(Ocp, lab)
    res = {
        "attention_weight_on_source": {
            "mean_over_heads": float(parts["p"][:, :, s].mean()),
            "by_head": parts["p"][:, :, s].mean(0).tolist(),
            "by_position_mean": parts["p"].mean((0, 1)).tolist(),
        },  # fmt: skip
        "source": {
            "amp": [float(np.linalg.norm(p)) for p in psi(xs, lab)],
            "offdiag": offdiag(mu_s),
            "acc": loo_acc(xs @ Ds, lab),
            "ovr": ovr_separable(xs @ Ds, lab),
        },
        "copy": {
            "amp": [float(np.linalg.norm(p)) for p in psi(Ocp, lab)],
            "offdiag": offdiag(mu_o),
            "acc": loo_acc(Ocp @ Do, lab),
            "ovr": ovr_separable(Ocp @ Do, lab),
            "norm_mean": float(np.linalg.norm(Ocp, axis=1).mean()),
            "period10_share": float(((Ocp @ Do) ** 2).sum() / (Ocp**2).sum()),
            "other_operand_amp": [float(np.linalg.norm(p)) for p in psi(Ocp, other)],
        },
        "eq_before": {
            "amp": [float(np.linalg.norm(p)) for p in psi(X[:, 4], lab)],
            "acc": loo_acc(X[:, 4] @ D4, lab),
        },
        "eq_after_block": {
            "amp": [float(np.linalg.norm(p)) for p in psi(X_next4, lab)],
            "acc": loo_acc(X_next4 @ Dn, lab),
            "ovr": ovr_separable(X_next4 @ Dn, lab),
        },
        "transfer": per_frequency(mu_s, mu_o),
        "cos_copy_vs_source_planes": [cos(Po[k], Ps[k]) for k in range(5)],
        "cos_copy_vs_eq_before_planes": [cos(Po[k], P4[k]) for k in range(5)],
        "cos_eq_after_vs_copy_planes": [cos(Pn[k], Po[k]) for k in range(5)],
    }
    # heads: period-10 energy of each head's part of O_s (in the copy's D)
    e_head = []
    for h in range(32):
        Oh = copied(blk, parts, s, heads=np.asarray([h]))
        e_head.append(float((np.abs(psi(Oh, lab)) ** 2).sum()))
    res["head_period10_energy"] = e_head
    # value components: residue profiles at the source, KV-head occupancy, contribution to the copy
    Cv = blk.a["V_v"].shape[1]
    ids_v = blk.a_ids["v"]
    tv = parts["tv"][:, s]
    prof = means(tv, lab)  # (10, Cv) raw residue means of each v component's inner activation
    spec = np.abs(np.fft.fft(prof, axis=0)) ** 2  # (10, Cv) energy per frequency 0..9
    spec_k = np.stack(
        [spec[0]] + [spec[k] + spec[10 - k] for k in range(1, 5)] + [spec[5]]
    )  # (6, Cv) k = 0..5
    occ = (blk.a["U_v"].reshape(Cv, 8, 128) ** 2).sum(-1)
    occ = occ / occ.sum(1, keepdims=True)
    contrib = []
    for c in range(Cv):
        Oc = copied(blk, parts, s, vcomps=np.asarray([c]))
        contrib.append(float((np.abs(psi(Oc, lab)) ** 2).sum()))
    contrib = np.asarray(contrib)
    order = np.argsort(-contrib)
    tot = contrib.sum()
    cum = np.cumsum(contrib[order]) / tot
    top = order[: int(np.searchsorted(cum, 0.9)) + 1]
    # do the top components together span the 9 period-10 dimensions of the source?
    P10 = (F10.conj().T @ np.diag(np.r_[0, np.ones(9)]) @ F10).real
    prof_top = P10 @ prof[:, top]
    res["vcomps"] = {
        "n_alive": int(Cv),
        "top90": [{"id": int(ids_v[c]), "share": float(contrib[c] / tot), "kv_occupancy": occ[c].round(3).tolist(),
                   "kv_main": int(occ[c].argmax()), "kv_main_share": float(occ[c].max()),
                   "freq_share": (spec_k[:, c] / spec_k[:, c].sum()).round(3).tolist(),
                   "profile": prof[:, c].tolist()} for c in top],  # fmt: skip
        "top_profiles_sv": np.linalg.svd(prof_top, compute_uv=False).tolist(),
        "top_profiles_rank_1e-2": int((np.linalg.svd(prof_top, compute_uv=False) > 1e-2 * np.linalg.svd(prof_top, compute_uv=False)[0]).sum()),
    }  # fmt: skip
    # o components: head occupancy of their read vectors and contribution
    Co = blk.a["V_o"].shape[1]
    occo = (blk.a["V_o"].reshape(32, 128, Co) ** 2).sum(1).T
    occo = occo / occo.sum(1, keepdims=True)
    head_out = (parts["p"][:, :, s, None] * parts["v"][:, s][:, parts["rep"]]).reshape(
        len(X), 32 * 128
    )
    so = head_out @ blk.a["V_o"]  # o components' inner activations from position s
    co = np.asarray(
        [
            float((np.abs(psi(np.outer(so[:, c], blk.a["U_o"][c]), lab)) ** 2).sum())
            for c in range(Co)
        ]
    )
    oo = np.argsort(-co)
    cumo = np.cumsum(co[oo]) / co.sum()
    topo = oo[: int(np.searchsorted(cumo, 0.9)) + 1]
    res["ocomps"] = {"n_alive": int(Co), "top90": [{"id": int(blk.a_ids["o"][c]), "share": float(co[c] / co.sum()),
                                                     "head_main": int(occo[c].argmax()), "head_main_share": float(occo[c].max()),
                                                     "n_heads_for_90": int(np.searchsorted(np.cumsum(np.sort(occo[c])[::-1]), 0.9) + 1)}
                                                    for c in topo]}  # fmt: skip
    geo = {"Ps": Ps, "Po": Po, "Pn": Pn, "Ds": Ds, "Do": Do, "Dn": Dn, "xs": xs, "O": Ocp, "X4n": X_next4, "prof": prof,
           "top": top, "ids_v": ids_v, "spec_k": spec_k, "e_head": np.asarray(e_head)}  # fmt: skip
    return res, geo


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    blocks, _, _ = load_blocks()
    import json as _j

    alive = _j.loads((AB.parent / "alive/alive.json").read_text())["components"]
    for blk in blocks:
        blk.a_ids = {k: np.asarray(alive[f"layers.{blk.b}.self_attn.{k}_proj"]) for k in "qkvo"}
    results, geos = {}, {}
    for which, L, s in (("a", 16, 1), ("b", 15, 3)):
        st = np.load(OUT / f"stream_{which}.npz")
        X = st[f"X{L}"].astype(np.float64)
        lab = (st[which] % 10).astype(int)
        other = (st["b" if which == "a" else "a"] % 10).astype(int)
        blk = blocks[L]
        from track_build_p3 import attn_seq

        Xn = X + np.concatenate([attn_seq(blk, X[i : i + 1000]) for i in range(0, len(X), 1000)])
        Xn4 = Xn[:, 4] + blk.mlp(Xn[:, 4])
        res, geo = analyse(blk, X, lab, other, s, Xn4)
        results[which] = res
        geos[which] = geo
        c = res["copy"]
        print(which, "weight on source", round(res["attention_weight_on_source"]["mean_over_heads"], 3),
              "| source amp", np.round(res["source"]["amp"], 2), "acc", res["source"]["acc"], "ovr", res["source"]["ovr"],
              "| copy amp", np.round(c["amp"], 2), "acc", c["acc"], "ovr", c["ovr"], "offdiag", round(c["offdiag"], 3),
              "| gains", np.round(res["transfer"]["gain"], 2), "| sv in", np.round(res["transfer"]["sv_period10_in"], 2),
              "out", np.round(res["transfer"]["sv_period10_out"], 2), flush=True)  # fmt: skip
        print(
            "   heads energy share",
            np.round(np.asarray(res["head_period10_energy"]) / sum(res["head_period10_energy"]), 2),
        )
        print(
            "   v top",
            [
                (
                    d["id"],
                    round(d["share"], 2),
                    d["kv_main"],
                    round(d["kv_main_share"], 2),
                    np.round(d["freq_share"], 2).tolist(),
                )
                for d in res["vcomps"]["top90"][:12]
            ],
        )
        print(
            "   v profiles rank",
            res["vcomps"]["top_profiles_rank_1e-2"],
            np.round(res["vcomps"]["top_profiles_sv"], 3),
        )
        print(
            "   o top",
            [
                (
                    d["id"],
                    round(d["share"], 2),
                    d["head_main"],
                    round(d["head_main_share"], 2),
                    d["n_heads_for_90"],
                )
                for d in res["ocomps"]["top90"][:12]
            ],
        )
        print("   cos copy/source", [np.round(c_, 2).tolist() for c_ in res["cos_copy_vs_source_planes"]], "cos eq_after/copy",
              [np.round(c_, 2).tolist() for c_ in res["cos_eq_after_vs_copy_planes"]], flush=True)  # fmt: skip
    # a vs b
    ga, gb = geos["a"], geos["b"]
    results["a_vs_b"] = {
        "cos_source_planes": [cos(ga["Ps"][k], gb["Ps"][k]) for k in range(5)],
        "cos_copy_planes": [cos(ga["Po"][k], gb["Po"][k]) for k in range(5)],
        "cos_eq_after_planes_a_L16_vs_b_L15": [cos(ga["Pn"][k], gb["Pn"][k]) for k in range(5)],
        "cos_D_source": cos(ga["Ds"], gb["Ds"]), "cos_D_copy": cos(ga["Do"], gb["Do"]),
    }  # fmt: skip
    Va = blocks[16].a["V_v"][:, ga["top"]]
    Vb = blocks[15].a["V_v"][:, gb["top"]]
    Ca = (Va / np.linalg.norm(Va, axis=0)).T @ (Vb / np.linalg.norm(Vb, axis=0))
    results["a_vs_b"]["vcomp_read_cos_max_per_a"] = np.abs(Ca).max(1).tolist()
    print("a vs b: cos source planes", [np.round(c, 2).tolist() for c in results["a_vs_b"]["cos_source_planes"]],
          "copy planes", [np.round(c, 2).tolist() for c in results["a_vs_b"]["cos_copy_planes"]],
          "D source", np.round(results["a_vs_b"]["cos_D_source"], 2), "D copy", np.round(results["a_vs_b"]["cos_D_copy"], 2),
          "v read cos max", np.round(results["a_vs_b"]["vcomp_read_cos_max_per_a"], 2))  # fmt: skip
    (OUT / "copy.json").write_text(json.dumps(results, indent=1))
    np.savez(OUT / "copy_geo.npz", **{f"{w}_{k}": (np.concatenate(v, 1) if isinstance(v, list) else v)
                                      for w in ("a", "b") for k, v in geos[w].items()})  # fmt: skip


if __name__ == "__main__":
    main()
