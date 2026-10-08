"""Extract the per-site data the quantity agent works on, for one token position (cluster job:
needs the full activation files).

Domain of position t (the distinct inputs activations can depend on, causal attention):
t = 1 (a): a = 1..100 (prompts op = +, b = 1); t = 2 (op): (op, a), 200 prompts (b = 1);
t = 3 (b) and t = 4 (=): (op, a, b), all 20000 prompts.
For every read site (block c, attention input l = 2c or MLP input l = 2c + 1) with readers active
at t (original-model CI > 0.01 somewhere at t), stores:
  <site>/Z   (D, k) float32: the alive-only stream at the site, in an orthonormal basis Q of the
             readers' gain-folded unit read directions (Z = X Q);
  <site>/W   (k, n) float32: readout, raw reads Y = Z W;
  <site>/CI  (D, n) float16: the readers' CI at t on the domain rows;
  <site>/cols (n,): dataset columns of the readers; <site>/Q (4096, k) float32;
  <site>/rho (D,) float32: the stream's RMS, sqrt(mean_j x_j^2 + EPS), which the norm divides by
             (reader activation = raw read / rho).
plus rows, op, a, b (D,) for the domain. Output: OUT/site_data_t<t>.npz.

    python qprep.py <t>
"""

import sys
from pathlib import Path

import numpy as np

RUN = Path.home() / "out/pod-backup/p-ba5a0c05"
DATASET = RUN / "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
VW = RUN / "analysis/virtual_weights"
OUT = RUN / "analysis/quantities/qagent"
READ = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}
EPS = 1e-5  # RMSNorm epsilon of Llama-3.1-8B


def domain_rows(t: int) -> np.ndarray:
    if t == 1:
        return np.arange(100) * 100
    if t == 2:
        return np.concatenate([op * 10000 + np.arange(100) * 100 for op in (0, 1)])
    return np.arange(20000)


def main() -> None:
    t = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    ix = np.load(DATASET / "index.npz")
    kind = np.array([k.split(".")[-1].replace("_proj", "") for k in ix["comp_kind"]])
    act = np.load(VW / "ci_positions.npz")["ci_max"][:, t] > 0.01
    vix = np.load(VW / "index.npz")
    rrow = {int(c): j for j, c in enumerate(vix["r_col"])}
    Vtil = np.load(VW / "vectors.npz")["V_til"]
    rows = domain_rows(t)
    resid = np.load(VW / "alive_only/resid.npy", mmap_mode="r")
    ci = np.asarray(np.load(DATASET / "original/ci.npy", mmap_mode="r")[:, t])[rows]  # (D, A)
    out = {"rows": rows, "op": ix["op"][rows], "a": ix["a"][rows], "b": ix["b"][rows]}
    for c in range(32):
        for pt in ("attn", "mlp"):
            cols = np.flatnonzero((ix["comp_layer"] == c) & np.isin(kind, READ[pt]) & act)
            if len(cols) == 0:
                continue
            V = Vtil[[rrow[int(x)] for x in cols]].astype(np.float64)  # (n, 4096)
            U, s, _ = np.linalg.svd(V.T, full_matrices=False)
            Q = U[:, s > 1e-6 * s[0]]
            lpos = 2 * c + (pt == "mlp")
            X = np.asarray(resid[lpos][:, t])[rows].astype(np.float64)  # (D, 4096)
            key = f"L{c}.{pt}"
            out[key + "/Z"] = (X @ Q).astype(np.float32)
            out[key + "/W"] = (Q.T @ V.T).astype(np.float32)
            out[key + "/CI"] = ci[:, cols].astype(np.float16)
            out[key + "/cols"] = cols
            out[key + "/Q"] = Q.astype(np.float32)
            out[key + "/rho"] = np.sqrt((X**2).mean(1) + EPS).astype(np.float32)
            print(f"t={t} {key}: n={len(cols)} k={Q.shape[1]}", flush=True)
    np.savez(OUT / f"site_data_t{t}.npz", **out)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
