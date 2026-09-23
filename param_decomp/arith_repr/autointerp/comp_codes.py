"""Which Fourier codes every residual writer supplies, measured right after it writes.

    python -m param_decomp.arith_repr.autointerp.comp_codes --dataset <filter>/dataset
        --resid <original-resid dir> --uv uv_alive.npz --wiring wiring.npz --out comp_codes.npz

Same share as `code_attrib.py` (`Re <z(k), z_c(k)> / |z(k)|^2`, z the Fourier coefficient of the
class means of the post-norm read input over a quantity), but for EVERY writer, each at the read
point that first reads its write: a layer-l o component at `mlp_in.l`, a layer-l down component
at `attn_in.(l+1)` (`final`, the final norm's input, for l = 31).

* `share` (A, 5 positions, 2 ops, 3 quantities, 51 k) float16: quantities a, b, res (a+b on
  add, a-b on sub); zero rows for non-writers.
* `power` (65 read points, 5, 2, 3, 51): |z(k)|^2 (k and -k) over the read input's total
  variance at that position/op — how strong the code is at all (read points: attn_in.l = 2l,
  mlp_in.l = 2l+1, final = 64)."""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.resid import load_raw, load_read_input, rms_norm

K = 51


def read_input(resid: Path, point: int, pos: int, norms: Any) -> np.ndarray:
    if point == 64:
        return rms_norm(load_raw(resid, "resid.32", pos), norms["final"], float(norms["eps"]))
    li, s = divmod(point, 2)
    return load_read_input(resid, f"{('attn_in', 'mlp_in')[s]}.{li}", pos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--wiring", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ix = np.load(args.dataset / "index.npz")
    a, b, op = ix["a"], ix["b"], ix["op"]
    site, cidx, kind, layer = ix["comp_site"], ix["comp_index"], ix["comp_kind"], ix["comp_layer"]
    n_comp = site.size
    wz = np.load(args.wiring)
    writers, rms, rms_final = wz["writers"], wz["rms"], wz["rms_final"]
    uv = np.load(args.uv)
    U = np.zeros((writers.size, 4096), np.float32)
    for s in np.unique(site[writers]):
        ids = {int(i): j for j, i in enumerate(uv[s + ".ids"])}
        Us = uv[s + ".U"]
        for n, c in enumerate(writers):
            if site[c] == s:
                U[n] = Us[ids[int(cidx[c])]]
    is_o = kind[writers] == "self_attn.o_proj"
    point_of = np.where(is_o, 2 * layer[writers] + 1, 2 * layer[writers] + 2)  # down L31 -> 64
    norms = np.load(args.resid / "norms.npz")
    ci_mm = np.load(args.dataset / "original" / "ci.npy", mmap_mode="r")
    inner_mm = np.load(args.dataset / "original" / "inner.npy", mmap_mode="r")
    share = np.zeros((n_comp, 5, 2, 3, K), np.float16)
    power = np.zeros((65, 5, 2, 3, K), np.float32)
    ks = np.arange(K)
    quantities = {0: (a, b, a + b), 1: (a, b, a - b)}
    for pos in range(1, 5):
        W = np.asarray(inner_mm[:, pos, :])[:, writers] * (
            np.asarray(ci_mm[:, pos, :], np.float32)[:, writers] > 0.01
        )
        for point in range(65):
            sel = np.flatnonzero(point_of == point)
            x = read_input(args.resid, point, pos, norms)
            if point == 64:
                ln, rp = norms["final"], rms_final[pos]
            else:
                li, s = divmod(point, 2)
                ln, rp = (norms["ln1"], norms["ln2"])[s][li], rms[point, pos]
            u = U[sel] * ln[None] / rp  # (n_sel, d)
            for o in (0, 1):
                m = op == o
                xm = x[m]
                tot = float(((xm - xm.mean(0)) ** 2).sum(1).mean())
                for qi, qv in enumerate(quantities[o]):
                    ang = 2 * np.pi * np.outer(ks, qv[m]) / 100
                    cos = (np.cos(ang) / m.sum()).astype(np.float32)
                    sin = (np.sin(ang) / m.sum()).astype(np.float32)
                    zr, zi = cos @ xm, -(sin @ xm)  # (K, d)
                    norm2 = np.maximum((zr**2 + zi**2).sum(1), 1e-30)
                    mult = np.where((ks == 0) | (ks == 50), 1.0, 2.0)
                    power[point, pos, o, qi] = mult * norm2 / max(tot, 1e-30)
                    if sel.size == 0:
                        continue
                    Wm = W[m][:, sel]
                    sr, si = cos @ Wm, -(sin @ Wm)  # (K, n_sel)
                    Mr, Mi = zr @ u.T, -(zi @ u.T)  # conj(z) . u
                    sh = (sr * Mr - si * Mi) / norm2[:, None]
                    share[writers[sel], pos, o, qi] = sh.T.astype(np.float16)
            print("pos", pos, "point", point, "writers", sel.size, flush=True)
    np.savez(args.out, **cast(dict[str, Any], {"share": share, "power": power}))
    print("saved", args.out)


if __name__ == "__main__":
    main()
