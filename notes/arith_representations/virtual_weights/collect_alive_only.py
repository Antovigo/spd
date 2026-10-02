"""Activations of the ALIVE-ONLY model: every alive component on at 100%, every dead component
and the weight delta off, on all 20000 target prompts.

Uses `ComponentsModel` (the components-only JAX forward) with its per-prompt masks replaced by
ones. Before the full pass it reruns chunk 0 with the dataset's per-prompt masks and checks that
the raw streams match `dataset/decomposed/resid.npy` (validates the forward).

Outputs, in OUT (layouts as in the activation dataset):
* `resid.npy` (65, N, T, d) float16: raw stream x_l, l = 0 (embedding), 2b+1 (after block b's
  attention add), 2b+2 (after its MLP add); before any norm.
* `inner.npy` (N, T, A) float32: each alive component's inner activation x @ V (raw gauge, A in
  dataset column order; divide by `index.npz/comp_v_norm` for the ||V|| = 1 gauge).
* `last_top_ids.npy` int32, `last_top_logprobs.npy` float32 (N, 10); `kl_last.npy` (N,) float32:
  KL(original || alive-only) at the last position, the original from `dataset/original/resid.npy`.
* `meta.json`.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpy.lib.format import open_memmap

from param_decomp.arith_repr.isa import components_model as cm
from param_decomp.arith_repr.vectors.common import DATASET, RUN, comp_table

OUT = RUN / "analysis/virtual_weights/alive_only"
N, T, D, A = 20000, 5, 4096, 11604
POINTS = set(range(65))


class AliveOnly(cm.ComponentsModel):
    def __init__(self) -> None:
        super().__init__()
        self.all_on = False

    def masks(self, layer: int, rows: np.ndarray) -> np.ndarray:
        m = super().masks(layer, rows)
        return np.ones_like(m) if self.all_on else m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = AliveOnly()
    comps = comp_table()
    layer_cols = [np.flatnonzero(comps["layer"] == li) for li in range(32)]
    print(f"model loaded {time.time() - t0:.0f}s", flush=True)

    # validation: dataset masks reproduce the dataset's decomposed streams on chunk 0
    rows = np.arange(cm.CHUNK)
    _, caps = model._chunk(rows, rows, None, None, POINTS)
    ds = np.load(DATASET / "decomposed/resid.npy", mmap_mode="r")
    errs = [float(np.linalg.norm(caps[t].astype(np.float32) - ds[t][rows].astype(np.float32))
                  / np.linalg.norm(ds[t][rows].astype(np.float32))) for t in range(65)]  # fmt: skip
    print(f"validation (dataset masks) max rel err over 65 points: {max(errs):.2e}", flush=True)
    assert max(errs) < 1e-2, errs

    model.all_on = True
    resid = open_memmap(OUT / "resid.npy", "w+", np.float16, (65, N, T, D))
    inner_mm = open_memmap(OUT / "inner.npy", "w+", np.float32, (N, T, A))
    top_ids = np.zeros((N, 10), np.int32)
    top_lp = np.zeros((N, 10), np.float32)
    kl = np.zeros(N, np.float32)
    orig_last = np.load(DATASET / "original/resid.npy", mmap_mode="r")[64]

    for s in range(0, N, cm.CHUNK):
        rows = np.arange(s, s + cm.CHUNK)
        buf = np.zeros((cm.CHUNK, T, A), np.float32)

        def rec(
            li: int, kind: str, h: jax.Array, rms: jax.Array, idx: np.ndarray, buf: np.ndarray = buf
        ) -> jax.Array:
            buf[:, :, layer_cols[li][model.sites[li][kind].cols]] = np.asarray(h)
            return h

        lp, caps = model._chunk(rows, rows, None, rec, POINTS)
        for t in range(65):
            resid[t, rows] = caps[t]
        inner_mm[rows] = buf
        lp_o = np.asarray(
            cm._logprobs(
                jnp.asarray(orig_last[rows, -1], jnp.float32), model.final_j, model.unembed
            )
        )
        kl[rows] = (np.exp(lp_o) * (lp_o - lp)).sum(-1)
        top = np.argsort(-lp, -1)[:, :10]
        top_ids[rows], top_lp[rows] = top, np.take_along_axis(lp, top, -1)
        print(
            f"{s + cm.CHUNK}/{N} {time.time() - t0:.0f}s kl_mean so far {kl[: s + cm.CHUNK].mean():.4f}",
            flush=True,
        )

    resid.flush(), inner_mm.flush()
    np.save(OUT / "last_top_ids.npy", top_ids)
    np.save(OUT / "last_top_logprobs.npy", top_lp)
    np.save(OUT / "kl_last.npy", kl)
    acc = (top_ids[:, 0] == model.answer).mean()
    meta = {"n_prompts": N, "seq_len": T, "n_alive": A, "masks": "all alive = 1, dead = 0, delta off",
            "validation_max_rel_err_dataset_masks": max(errs), "kl_last_mean": float(kl.mean()),
            "kl_last_median": float(np.median(kl)), "top1_answer_acc": float(acc)}  # fmt: skip
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta, flush=True)


if __name__ == "__main__":
    main()
