"""Virtual weights between the alive components of the -05 addsub decomposition (weights only).

Conventions (see README.md written next to the outputs):
* l in 0..64 indexes the residual stream: 0 after the embedding, 2b+1 after block b's attention
  write, 2b+2 after its MLP write (b in 0..31). Readers of block b read l = 2b (q, k, v) or
  l = 2b+1 (gate, up); writers write at m = 0 (embedding), 2b+1 (o) or 2b+2 (down).
* The embedding is one component per token tau seen in the target data: V = one-hot(tau),
  U = E[tau].
* Gauge: every component is rescaled so ||V|| = 1 (U absorbs ||V||, the inner activation is
  divided by it). The embedding's V is already a unit one-hot.
* M[w, r] = U_w . (gamma_l * V_r) with gamma_l the gain of the norm that reads stream l, for
  every writer w upstream of reader r (m_w <= l_r); NaN otherwise. Then, on prompt i, position t,
  A_r(i,t) = sum_w A_w(i,t) M[w, r] / rho_l(i,t), rho_l = sqrt(mean(x_l^2) + eps).

`check` verifies this identity on the activation dataset's decomposed run (per-prompt masks
1[original CI > 0.01], delta off): it rebuilds x_l from the writes and A_r from M.

    python compute.py build   # writes OUT/{M.npy, index.npz, vectors.npz}
    python compute.py check   # writes OUT/check.json
"""

import json
import sys

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy, for the safetensors reads)
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, snapshot
from param_decomp.arith_repr.vectors.common import AUTOINTERP, DATASET, RESID, RUN, point_names

OUT = RUN / "analysis/virtual_weights"
N_LAYER = 32
SITES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
         "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
         "down": "mlp.down_proj"}  # fmt: skip
READERS = ("q", "k", "v", "gate", "up")
WRITERS = ("o", "down")


def read_pos(b: int, kind: str) -> int:
    return 2 * b + (0 if kind in ("q", "k", "v") else 1)


def write_pos(b: int, kind: str) -> int:
    return 2 * b + (1 if kind == "o" else 2)


def dataset_cols() -> dict[tuple[str, int], int]:
    ix = np.load(DATASET / "index.npz")
    return {
        (str(s), int(c)): j
        for j, (s, c) in enumerate(zip(ix["comp_site"], ix["comp_index"], strict=True))
    }


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    norms = np.load(RESID / "norms.npz")
    gain = {0: norms["ln1"], 1: norms["ln2"]}  # (32, 4096) each: input_ / post_attention_layernorm
    cols = dataset_cols()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    vocab = np.unique(np.load(DATASET / "index.npz")["tokens"])
    emb = Weights().get_rows("model.embed_tokens.weight", vocab)

    W: dict[str, list] = {k: [] for k in ("kind", "layer", "cidx", "token", "pos", "col", "vnorm")}
    R: dict[str, list] = {k: [] for k in ("kind", "layer", "cidx", "pos", "col", "vnorm")}
    U_hat, V_til = [], []

    for j, t in enumerate(vocab):
        W["kind"].append("embed"), W["layer"].append(-1), W["cidx"].append(int(t))
        W["token"].append(tok.id_to_token(int(t))), W["pos"].append(0), W["col"].append(-1)
        W["vnorm"].append(1.0)
        U_hat.append(emb[j])

    for b in range(N_LAYER):
        for kind in ("q", "k", "v", "o", "gate", "up", "down"):
            site = f"layers.{b}.{SITES[kind]}"
            ids, V, U = (
                uv[site + ".ids"],
                uv[site + ".V"],
                uv[site + ".U"],
            )  # V (d_in, n), U (n, d_out)
            order = np.argsort(ids)
            vn = np.linalg.norm(V, axis=0)
            for j in order:
                c = int(ids[j])
                if kind in WRITERS:
                    W["kind"].append(kind), W["layer"].append(b), W["cidx"].append(c)
                    W["token"].append(""), W["pos"].append(write_pos(b, kind))
                    W["col"].append(cols[(site, c)]), W["vnorm"].append(float(vn[j]))
                    U_hat.append(U[j] * vn[j])
                else:
                    R["kind"].append(kind), R["layer"].append(b), R["cidx"].append(c)
                    R["pos"].append(read_pos(b, kind)), R["col"].append(cols[(site, c)])
                    R["vnorm"].append(float(vn[j]))
                    g = gain[0 if kind in ("q", "k", "v") else 1][b]
                    V_til.append(g * V[:, j] / vn[j])

    U_hat_a = np.asarray(U_hat, np.float32)
    V_til_a = np.asarray(V_til, np.float32)
    w_pos, r_pos = np.asarray(W["pos"]), np.asarray(R["pos"])
    M = U_hat_a.astype(np.float64) @ V_til_a.T.astype(np.float64)
    M[w_pos[:, None] > r_pos[None, :]] = np.nan
    np.save(OUT / "M.npy", M.astype(np.float32))

    np.savez(
        OUT / "index.npz",
        point_names=np.asarray(point_names()),
        **{"w_" + k: np.asarray(v) for k, v in W.items()},
        **{"r_" + k: np.asarray(v) for k, v in R.items()},
        w_unorm=np.linalg.norm(U_hat_a, axis=1),
        r_gvnorm=np.linalg.norm(V_til_a, axis=1),
    )
    np.savez(OUT / "vectors.npz", U_hat=U_hat_a, V_til=V_til_a)
    fin = np.isfinite(M)
    print(f"writers {len(w_pos)}, readers {len(r_pos)}, upstream pairs {fin.sum()}", flush=True)
    print(f"|M| quantiles (50/90/99/99.9/max): "
          f"{np.quantile(np.abs(M[fin]), [0.5, 0.9, 0.99, 0.999, 1.0])}", flush=True)  # fmt: skip


def check(n_prompts: int = 400) -> None:
    """Rebuild the decomposed run's x_l and reader activations from M on n_prompts prompts."""
    ix = np.load(OUT / "index.npz")
    M = np.load(OUT / "M.npy")
    U_hat = np.load(OUT / "vectors.npz")["U_hat"]
    rows = np.sort(np.random.default_rng(0).choice(20000, n_prompts, replace=False))
    tokens = np.load(DATASET / "index.npz")["tokens"][rows]  # (n, 5)
    inner = np.load(DATASET / "decomposed/inner.npy", mmap_mode="r")[rows]  # (n, 5, 11604)
    on = np.load(DATASET / "original/ci.npy", mmap_mode="r")[rows] > 0.01
    resid = np.load(DATASET / "decomposed/resid.npy", mmap_mode="r")

    w_kind, w_pos, w_col = ix["w_kind"], ix["w_pos"], ix["w_col"]
    emb = w_kind == "embed"
    # writer activations in the ||V|| = 1 gauge: embed = one-hot of the token; others = inner / ||V||, masked
    A_w = np.zeros((n_prompts, 5, len(w_kind)), np.float32)
    tok_col = {int(t): j for j, t in enumerate(ix["w_cidx"]) if w_kind[j] == "embed"}
    for i in range(n_prompts):
        for t in range(5):
            A_w[i, t, tok_col[int(tokens[i, t])]] = 1.0
    nc = ~emb
    A_w[:, :, nc] = inner[:, :, w_col[nc]] * on[:, :, w_col[nc]] / ix["w_vnorm"][nc]

    eps = float(np.load(RESID / "norms.npz")["eps"])
    out: dict[str, dict] = {}
    r_pos, r_col = ix["r_pos"], ix["r_col"]
    for pos in range(64):
        up = w_pos <= pos
        x = np.einsum("itw,wd->itd", A_w[:, :, up], U_hat[up])  # (n, 5, d)
        x_ds = resid[pos][rows].astype(np.float32)
        rho = np.sqrt((x**2).mean(-1) + eps)
        rd = r_pos == pos
        if not rd.any():
            continue
        pred = (
            np.einsum("itw,wr->itr", A_w[:, :, up], np.nan_to_num(M[np.ix_(up, rd)]))
            / rho[..., None]
        )
        true = inner[:, :, r_col[rd]] / ix["r_vnorm"][rd]
        out[point_names()[pos]] = {
            "x_rel_err": float(np.linalg.norm(x - x_ds) / np.linalg.norm(x_ds)),
            "A_rel_err": float(np.linalg.norm(pred - true) / np.linalg.norm(true)),
            "A_max_abs_err": float(np.abs(pred - true).max()),
        }
        print(pos, out[point_names()[pos]], flush=True)
    (OUT / "check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()
