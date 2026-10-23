"""Residual stream at every position (0..4: <BOS>, a, op, b, =) entering blocks 15 and 16 of the alive-only model.

Two prompt sets, 8,000 prompts each, tokenized as the pool does (f"{a}{op}{b}=", no spaces), op in (+, -):
  set "a"   every a = 1..400, b in B_SUB (ten values, one per residue b mod 10)   -> copying of a mod 10
  set "b"   a in A_SUB (ten values, one per residue a mod 10), every b = 1..400   -> copying of b mod 10
Writes <run>/analysis/reader_writer/copy_mod10/stream_<set>.npz with X15, X16 (8000, 5, 4096) float32 (the stream
entering block 15 / 16), and a, b, op (8000,). The rebuild (float64, alive components on, weight delta off) is
checked against the stored MLP inner activations at positions 1, 3, 4 (`variance/alive_only/L<l>/inner.npy`).

    python copy_build.py
"""

import json

import numpy as np
from track_build import SRC, load_blocks
from track_build_p3 import A_SUB, attn_seq

OUT = SRC.parent.parent / "copy_mod10"
B_SUB = np.asarray([3 + 41 * i for i in range(10)])  # residues 3, 4, ..., 2
LAST = 16


def prompts(which: str) -> dict[str, np.ndarray]:
    a_vals = np.arange(1, 401) if which == "a" else A_SUB
    b_vals = B_SUB if which == "a" else np.arange(1, 401)
    a, b, op = (g.ravel() for g in np.meshgrid(a_vals, b_vals, (0, 1), indexing="ij"))
    order = np.lexsort((b, a, op))
    return {"a": a[order], "b": b[order], "op": op[order]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, _, _ = load_blocks()
    import glob
    from pathlib import Path

    from mod10_L0 import HUB, bf16_tensor
    from transformers import AutoTokenizer

    (snap,) = glob.glob(HUB)
    wmap = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    checks = {}
    for which in ("a", "b"):
        meta = prompts(which)
        toks = np.asarray([tok.encode(f"{a}{'+-'[o]}{b}=", add_special_tokens=True)
                           for a, b, o in zip(meta["a"], meta["b"], meta["op"], strict=True)])  # fmt: skip
        assert toks.shape[1] == 5
        uniq, inv = np.unique(toks.ravel(), return_inverse=True)
        X = bf16_tensor(
            snap + wmap["model.embed_tokens.weight"], "model.embed_tokens.weight", uniq
        )[inv]
        X = X.reshape(*toks.shape, -1)
        rows = meta["op"] * 160000 + (meta["a"] - 1) * 400 + (meta["b"] - 1)
        save = dict(meta)
        for blk in blocks:
            if blk.b in (15, 16):
                save[f"X{blk.b}"] = X.astype(np.float32)
            X = X + np.concatenate([attn_seq(blk, X[i : i + 1000]) for i in range(0, len(X), 1000)])
            n, T, d = X.shape
            flat = X.reshape(n * T, d)
            tg, tu = blk.readers(flat)
            ref = np.asarray(
                np.load(SRC / f"L{blk.b}" / "inner.npy", mmap_mode="r")[np.sort(rows)], np.float64
            )
            ref = ref[np.argsort(np.argsort(rows))]  # (n, 4 positions, K)
            mine = np.concatenate([tg, tu], 1).reshape(n, T, -1)[:, 1:]
            r = ref[:, :, np.isin(blk.kind, ("gate", "up"))]
            checks[f"{which}_L{blk.b}"] = float(np.abs(mine - r).max() / np.abs(r).max())
            X = X + np.concatenate(
                [blk.mlp(flat[i : i + 2000]) for i in range(0, n * T, 2000)]
            ).reshape(n, T, d)
            print(which, blk.b, round(checks[f"{which}_L{blk.b}"], 4), flush=True)
            if blk.b == LAST:
                break
        np.savez(OUT / f"stream_{which}.npz", **save)
    (OUT / "build.json").write_text(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
