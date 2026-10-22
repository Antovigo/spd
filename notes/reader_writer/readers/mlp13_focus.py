"""Block 13's MLP at position 1: why its period-10 write separates the residue clouds (alive-only model).

Notation (a = integer operand 1..400, r = a mod 10; X the stream entering the MLP, D its 9-dim period-10 subspace):
  t_c(a)      inner activation of reader c (x . V_c, x = RMSNorm(X)); t_c^-(a) the same with X - Pi_D X; dt_c = t_c - t_c^-
  Delta_c(a)  change of the MLP output when only reader c's activation goes from t_c to t_c^- (R^4096)
  o_c(a)      Delta_c(a) . u_c, u_c the unit vector maximising sum_a (Delta_c . u)^2 (first singular vector)
  SNR of a scalar field f between residues r and r': |mean_r f - mean_r' f| / sqrt((var_r f + var_r' f)/2), with
              var_r the variance over the 40 a of residue r (a spread around the residue's own mean)
  focus       for each reader: SNR of o_c against SNR of dt_c, between the main residue r0 (largest mean |dt_c|) and
              every other residue; a linear path (o_c = kappa dt_c) has equal SNRs
Per neuron n reached by reader c: h_g,n and h_u,n its gate and up pre-activations, SiLU'(h) = d SiLU / dh.
Regime of the gate path: mean over prompts of the local slope SiLU'(h_g) at the main residue vs elsewhere.

    python mlp13_focus.py
"""

import json

import numpy as np
from mod10_L0 import A
from track_analyze import basis
from track_build import OUT, load_blocks

R = A % 10
B = 13


def snr(f: np.ndarray, r: int, q: int) -> float:
    a, b = f[r == R], f[q == R]
    return float(abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def main() -> None:
    st = np.load(OUT / "stream.npz")
    blk = load_blocks()[0][B]
    X = st[f"mid_{B}"].astype(np.float64)
    _, D = basis(X)
    Xd = X - (X @ D) @ D.T
    tg, tu = blk.readers(X)
    tgd, tud = blk.readers(Xd)
    hg, hu = tg @ blk.m["U_gate"], tu @ blk.m["U_up"]
    sg = hg / (1 + np.exp(-hg))
    sig = 1 / (1 + np.exp(-hg))
    dsilu = sig * (1 + hg * (1 - sig))
    o = sg * hu
    Vd, Ud = blk.m["V_down"], blk.m["U_down"]
    names = [f"{k[0]}{int(i)}" for k, i in zip(blk.kind, blk.index, strict=True) if k != "down"]
    G = tg.shape[1]
    rows = []
    for j, name in enumerate(names):
        if j < G:
            dt = tg[:, j] - tgd[:, j]
            h2 = hg - np.outer(dt, blk.m["U_gate"][j])
            dc = ((o - (h2 / (1 + np.exp(-h2))) * hu) @ Vd) @ Ud
            w = blk.m["U_gate"][j]
        else:
            dt = tu[:, j - G] - tud[:, j - G]
            dc = ((sg * np.outer(dt, blk.m["U_up"][j - G])) @ Vd) @ Ud
            w = blk.m["U_up"][j - G]
        rows.append((float((dc**2).sum()), j, name, dt, dc, w))
    rows.sort(key=lambda t: -t[0])
    tot = sum(r[0] for r in rows)
    out = []
    for en, j, name, dt, dc, w in rows[:8]:
        u = np.linalg.svd(dc, full_matrices=False)[2][0]
        oc = dc @ u
        prof_dt = np.asarray([dt[r == R].mean() for r in range(10)])
        r0 = int(np.argmax(np.abs(prof_dt)))
        if np.sign(prof_dt[r0]) != np.sign(oc[r0 == R].mean()):
            oc = -oc
        others = [q for q in range(10) if q != r0]
        nn = np.argsort(-np.abs(w))[:200]  # the neurons this reader writes to most
        wt = np.abs(w[nn])
        on, off = r0 == R, r0 != R
        gate = j < G
        rows_out = {
            "reader": name, "kind": "gate" if gate else "up", "energy_share": en / tot, "main_residue": r0,
            "dt_profile": prof_dt.tolist(), "o_profile": [float(oc[r == R].mean()) for r in range(10)],
            "dt_std_by_r": [float(dt[r == R].std()) for r in range(10)], "o_std_by_r": [float(oc[r == R].std()) for r in range(10)],
            "snr_dt": [snr(dt, r0, q) for q in others], "snr_o": [snr(oc, r0, q) for q in others],
            "snr_dt_offpairs_median": float(np.median([snr(dt, q, p) for q in others for p in others if p > q])),
            "snr_o_offpairs_median": float(np.median([snr(oc, q, p) for q in others for p in others if p > q])),
            "top_neurons_mean_hg_on": float((hg[np.ix_(on, nn)] * wt).sum() / (wt.sum() * on.sum())),
            "top_neurons_mean_hg_off": float((hg[np.ix_(off, nn)] * wt).sum() / (wt.sum() * off.sum())),
            "top_neurons_silu_slope_on": float((dsilu[np.ix_(on, nn)] * wt).sum() / (wt.sum() * on.sum())),
            "top_neurons_silu_slope_off": float((dsilu[np.ix_(off, nn)] * wt).sum() / (wt.sum() * off.sum())),
            "top_neurons_mean_hu_on": float((hu[np.ix_(on, nn)] * wt).sum() / (wt.sum() * on.sum())),
            "top_neurons_mean_hu_off": float((hu[np.ix_(off, nn)] * wt).sum() / (wt.sum() * off.sum())),
            "corr_dt_o_on": float(np.corrcoef(dt[on], oc[on])[0, 1]),
            "corr_dt_o_off": float(np.corrcoef(dt[off], oc[off])[0, 1]),
        }  # fmt: skip
        out.append(rows_out)
        print(f"{name} ({rows_out['kind']}, share {en / tot:.2f}, r0={r0}) "
              f"SNR dt vs o (main vs others, median): {np.median(rows_out['snr_dt']):.2f} -> {np.median(rows_out['snr_o']):.2f}; "
              f"off-off pairs {rows_out['snr_dt_offpairs_median']:.2f} -> {rows_out['snr_o_offpairs_median']:.2f}; "
              f"std off/on dt {np.mean([rows_out['dt_std_by_r'][q] for q in others]) / rows_out['dt_std_by_r'][r0]:.2f} "
              f"o {np.mean([rows_out['o_std_by_r'][q] for q in others]) / max(rows_out['o_std_by_r'][r0], 1e-12):.2f}; "
              f"h_g on/off {rows_out['top_neurons_mean_hg_on']:.2f}/{rows_out['top_neurons_mean_hg_off']:.2f} "
              f"slope on/off {rows_out['top_neurons_silu_slope_on']:.2f}/{rows_out['top_neurons_silu_slope_off']:.2f} "
              f"h_u on/off {rows_out['top_neurons_mean_hu_on']:.2f}/{rows_out['top_neurons_mean_hu_off']:.2f} "
              f"corr(dt,o) on/off {rows_out['corr_dt_o_on']:.2f}/{rows_out['corr_dt_o_off']:.2f}", flush=True)  # fmt: skip
        print("   dt profile", np.round(prof_dt / np.abs(prof_dt).max(), 2), " o profile",
              np.round(np.asarray(rows_out["o_profile"]) / np.abs(rows_out["o_profile"]).max(), 2), flush=True)  # fmt: skip
    (OUT / "mlp13_focus.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
