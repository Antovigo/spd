"""Weight-level wiring of the alive components (no activations beyond residual norms).

    python -m param_decomp.arith_repr.autointerp.wiring --uv uv_alive.npz --dataset <filter>/dataset
        --resid <original-resid dir> --out wiring.npz

Every array is indexed by the dataset's component columns (`index.npz: comp_site/comp_index`).

* `dla` (A, 202): direct logit effect of a unit write along U_c into the final residual, on the
  tokens "0".."200" and "-": `W_U[t] . (g_final * U_c) / rms_final(=)` (only residual writers,
  o_proj/down_proj; zero rows elsewhere). `rms_final` is the mean RMS of the last residual at `=`.
* `head` (A, 32): share of ||U_c||^2 (q) / ||V_c||^2 (o) per query head, and per kv head for k/v
  (first 8 columns).
* `resid_vw` (n_writer, n_reader): U_w . (ln_r * V_r) for every residual writer (o/down) and
  reader (q/k/v/gate/up) downstream of it (zero elsewhere); `writers`, `readers` = column ids.
  `rms` (64, 5): mean RMS of the raw residual at every read point (`attn_in.l` = 2l,
  `mlp_in.l` = 2l+1) and position, so the effect on the reader's inner is
  `w_writer * resid_vw / rms`.
* `vo_<l>` (n_v, n_o, 32): per query head h, U_v[kv(h)] . V_o[h] (v -> o through head h).
* `mlp_<l>_{gate,up}` (n_gate|n_up, n_down): cosine of the gate/up write U with the down read V
  in the neuron basis.
"""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, number_token_ids
from param_decomp.arith_repr.resid import load_raw, raw_key

N_HEAD, N_KV, HD = 32, 8, 128
READ_KINDS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
)
WRITE_KINDS = ("self_attn.o_proj", "mlp.down_proj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    index = np.load(args.dataset / "index.npz")
    site, cidx, kind, layer = (
        index["comp_site"],
        index["comp_index"],
        index["comp_kind"],
        index["comp_layer"],
    )
    n_comp = site.size
    uv = np.load(args.uv)
    U: list[np.ndarray] = [np.empty(0)] * n_comp
    V: list[np.ndarray] = [np.empty(0)] * n_comp
    for s in np.unique(site):
        cols = np.flatnonzero(site == s)
        ids = uv[s + ".ids"]
        pos = {int(i): j for j, i in enumerate(ids)}
        Vs, Us = uv[s + ".V"], uv[s + ".U"]
        for c in cols:
            j = pos[int(cidx[c])]
            V[c], U[c] = Vs[:, j], Us[j]
    print("loaded U/V", flush=True)
    norms = np.load(args.resid / "norms.npz")
    ln1, ln2, g_final = norms["ln1"], norms["ln2"], norms["final"]

    # mean RMS of the raw residual at every read point / position, and at the final residual
    sub = np.arange(0, 20000, 10)
    rms = np.zeros((64, 5), np.float32)
    for li in range(32):
        for s_i, stream in enumerate(("attn_in", "mlp_in")):
            for p in range(5):
                x = load_raw(args.resid, raw_key(f"{stream}.{li}"), p)[sub]
                rms[2 * li + s_i, p] = np.sqrt((x**2).mean(-1)).mean()
    rms_final = np.array(
        [np.sqrt((load_raw(args.resid, "resid.32", p)[sub] ** 2).mean(-1)).mean() for p in range(5)]
    )
    print("rms done", rms_final, flush=True)

    w = Weights()
    num_ids, minus = number_token_ids(200)
    tok = np.concatenate([num_ids, [minus]])
    W_U = w.get_rows("lm_head.weight", tok)  # (202, 4096)
    writers = np.flatnonzero(np.isin(kind, WRITE_KINDS))
    readers = np.flatnonzero(np.isin(kind, READ_KINDS))
    dla = np.zeros((n_comp, tok.size), np.float32)
    for c in writers:
        dla[c] = W_U @ (g_final * U[c]) / rms_final[4]

    head = np.zeros((n_comp, N_HEAD), np.float32)
    for c in range(n_comp):
        k = kind[c]
        if k == "self_attn.q_proj":
            e = (U[c].reshape(N_HEAD, HD) ** 2).sum(1)
        elif k == "self_attn.o_proj":
            e = (V[c].reshape(N_HEAD, HD) ** 2).sum(1)
        elif k in ("self_attn.k_proj", "self_attn.v_proj"):
            e = np.zeros(N_HEAD)
            e[:N_KV] = (U[c].reshape(N_KV, HD) ** 2).sum(1)
        else:
            continue
        head[c] = e / e.sum()

    Umat = np.stack([U[c] for c in writers])  # (n_w, 4096)
    Vfold = np.stack(
        [(ln1 if kind[c].startswith("self_attn") else ln2)[layer[c]] * V[c] for c in readers]
    )  # (n_r, 4096)
    vw = (Umat @ Vfold.T).astype(np.float32)
    # causal mask: reader downstream of the writer (an MLP reads its own layer's attention write)
    lw, lr = layer[writers][:, None], layer[readers][None, :]
    w_is_o = (kind[writers] == "self_attn.o_proj")[:, None]
    r_is_mlp = np.char.startswith(kind[readers].astype(str), "mlp")[None, :]
    downstream = (lr > lw) | ((lr == lw) & w_is_o & r_is_mlp)
    vw *= downstream
    print("resid vw done", vw.shape, flush=True)

    out: dict[str, np.ndarray] = {
        "dla": dla, "head": head, "resid_vw": vw, "writers": writers, "readers": readers,
        "rms": rms, "rms_final": rms_final, "tokens": tok,
    }  # fmt: skip
    for li in range(32):
        vs = np.flatnonzero((layer == li) & (kind == "self_attn.v_proj"))
        os_ = np.flatnonzero((layer == li) & (kind == "self_attn.o_proj"))
        if vs.size and os_.size:
            Uv = np.stack([U[c].reshape(N_KV, HD) for c in vs])  # (n_v, 8, hd)
            Vo = np.stack([V[c].reshape(N_HEAD, HD) for c in os_])  # (n_o, 32, hd)
            Uv_q = np.repeat(Uv, N_HEAD // N_KV, axis=1)  # (n_v, 32, hd)
            out[f"vo_{li}"] = np.einsum("vhd,ohd->voh", Uv_q, Vo).astype(np.float32)
            out[f"vo_{li}_ids"] = np.concatenate([vs, os_])
        ds = np.flatnonzero((layer == li) & (kind == "mlp.down_proj"))
        Vd = np.stack([V[c] / np.linalg.norm(V[c]) for c in ds]) if ds.size else None
        for kname in ("gate", "up"):
            gs = np.flatnonzero((layer == li) & (kind == f"mlp.{kname}_proj"))
            if gs.size and Vd is not None:
                Ug = np.stack([U[c] / np.linalg.norm(U[c]) for c in gs])
                out[f"mlp_{li}_{kname}"] = (Ug @ Vd.T).astype(np.float32)
                out[f"mlp_{li}_{kname}_ids"] = gs
        if ds.size:
            out[f"mlp_{li}_down_ids"] = ds
    np.savez(args.out, **cast(dict[str, Any], out))
    print("saved", args.out)


if __name__ == "__main__":
    main()
