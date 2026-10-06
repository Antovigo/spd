"""Is b's code at `=` mirrored on subtraction, with and without the seven L15 flip neurons, and who
writes the mirror?

    python -m param_decomp.arith_repr.vectors.flip_check run       # GPU -> OUT/flip/check_<cond>.npz
    python -m param_decomp.arith_repr.vectors.flip_check analyse   # CPU, prints the tables

Conditions (components-only model, all 20,000 prompts): `base`; `F0`, the seven flip neurons zeroed
at `=`; `mlp15_0`, every L15 neuron zeroed at `=`.

b-line code. For a quantity y at `=` and op o, F_o(k) = (1/100) sum_b ybar_o(b) e^{-2 pi i k b / 100},
k = 1..50, with ybar_o(b) the mean over the 100 values of a of the prompts (o, b), minus its mean over
b. b -> -b maps F(k) to conj(F(k)).

Readers' view at the stream point before block l's MLP: R_o(k) = G_l^T F_o(k), G_l = the layer's gate
and up V scaled by the norm gain (one complex number per reader). Measures, per k:
* same = |sum R_sub conj(R_add)| / (|R_add| |R_sub|): 1 when sub sees b as add does;
* refl = |sum R_sub R_add| / (|R_add| |R_sub|): 1 when sub sees -b (up to a global phase);
* odd = |O|^2 / (|E|^2 + |O|^2), E = (R_add + R_sub) / 2, O = (R_add - R_sub) / 2: share of b's
  code that changes with the op; |O| is also given relative to `base`.

Attribution. The stream at a point is the embedding plus every earlier sublayer write, so O at a read
point splits exactly over the writes; a write's share is Re(sum_r O_w,r conj(O_r)) / sum_r |O_r|^2
(shares sum to 1). Attention writes split exactly over heads; MLP writes are split over neurons with
the down map averaged over the prompts' masks (`approx` = sum of neuron shares vs the exact write)."""

import sys
from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.isa.atlas import ATLAS
from param_decomp.arith_repr.vectors.common import AUTOINTERP, RESID, comp_table
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_model import FLIP

CONDS = ("base", "F0", "mlp15_0")
KS = (1, 2, 3, 10, 20)
GATE_UP = ("mlp.gate_proj", "mlp.up_proj")


def run() -> None:
    import jax

    from param_decomp.arith_repr.vectors.flip_model import FlipModel

    print(jax.devices(), flush=True)
    M = FlipModel()
    rows = np.arange(20000)
    eqT = np.array([0, 0, 0, 0, 1], bool)
    inF = np.zeros(14336, bool)
    inF[FLIP] = True
    specs = {
        "base": None,
        "F0": (inF, eqT),
        "mlp15_0": (np.ones(14336, bool), eqT),
    }
    M.capture_on = True
    for cond, neurons in specs.items():
        M.spec = {"comps": [], "neurons": neurons}
        M.reset_capture()
        M.forward(rows)
        z = {
            k: np.asarray(v, np.float32) / 100.0 for k, v in M.cap.items()
        }  # 100 prompts per (op, b)
        np.savez(DIR / f"check_{cond}.npz", **cast(dict[str, Any], z))
        print(cond, "saved", len(z), "arrays", flush=True)


def bl(y: np.ndarray) -> np.ndarray:
    """(200, ...) group means over (op, b) -> (2, 50, ...) b-line coefficients per op."""
    out = []
    k = np.arange(1, 51).reshape((-1,) + (1,) * (y.ndim - 1))
    for o in (0, 1):
        yb = y[o * 100 : (o + 1) * 100]
        yb = yb - yb.mean(0)
        f = np.fft.fft(yb, axis=0)[1:51] / 100
        out.append(f * np.exp(-2j * np.pi * k / 100))
    return np.stack(out)


def site_uv(layer: int, suffix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V (d_in, n), U (n, d_out) and the mask columns of one site, in the components model's order."""
    comps = comp_table()
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    lcols = np.flatnonzero(comps["layer"] == layer)
    key = f"layers.{layer}.{suffix}"
    sel = np.flatnonzero(comps["site"][lcols] == key)
    ids = {int(i): j for j, i in enumerate(uv[key + ".ids"])}
    order = [ids[int(comps["cidx"][lcols[s]])] for s in sel]
    return (
        uv[key + ".V"][:, order].astype(np.float32),
        uv[key + ".U"][order].astype(np.float32),
        sel,
    )


def readers(layer: int) -> np.ndarray:
    ln2 = np.load(RESID / "norms.npz")["ln2"][layer]
    return np.concatenate([site_uv(layer, s)[0] for s in GATE_UP], 1) * ln2[:, None]


def measures(Ra: np.ndarray, Rs: np.ndarray) -> tuple[float, float, float]:
    den = np.linalg.norm(Ra) * np.linalg.norm(Rs)
    Ev, Od = (Ra + Rs) / 2, (Ra - Rs) / 2
    same = abs((Rs * np.conj(Ra)).sum()) / den
    refl = abs((Rs * Ra).sum()) / den
    odd = (np.abs(Od) ** 2).sum() / ((np.abs(Ev) ** 2).sum() + (np.abs(Od) ** 2).sum())
    return float(same), float(refl), float(odd)


def point_name(p: int) -> str:
    return "embed" if p < 0 else f"L{p // 2}.{'attn' if p % 2 == 0 else 'mlp'}"


def analyse() -> None:
    Z = {c: dict(np.load(DIR / f"check_{c}.npz")) for c in CONDS}
    BL = {c: {key: bl(v) for key, v in z.items() if key.startswith("x")} for c, z in Z.items()}
    layers = list(range(13, 32))
    G = {li: readers(li) for li in layers}

    print("1. b's code as the MLP readers of layer l see it at `=` (stream before block l's MLP):"
          " same / refl / odd, and |O| relative to base")  # fmt: skip
    Obase: dict[tuple[int, int], float] = {}
    for c in CONDS:
        print(f"\n  {c}")
        for li in layers:
            p = 2 * li + 1
            cells = []
            for k in KS:
                Ra, Rs = (BL[c][f"x{p}"][o, k - 1] @ G[li] for o in (0, 1))
                same, refl, odd = measures(Ra, Rs)
                On = float(np.linalg.norm((Ra - Rs) / 2))
                if c == "base":
                    Obase[(li, k)] = On
                cells.append(f"k{k} {same:.2f}/{refl:.2f}/{odd:.2f} ({On / Obase[(li, k)]:.2f})")
            print(f"    L{li:2d}: " + "  ".join(cells))

    print("\n2. Who writes the op-odd part O at the readers of layer l (shares of O; top writes)")
    names = ["embed"] + [point_name(p) for p in range(64)]
    for c in CONDS:
        print(f"\n  {c}")
        for li in (15, 16, 17, 18, 20, 24):
            p_read = 2 * li + 1
            for k in (2, 10, 20):
                Ot = (BL[c][f"x{p_read}"][0, k - 1] - BL[c][f"x{p_read}"][1, k - 1]) / 2 @ G[li]
                den = (np.abs(Ot) ** 2).sum()
                shares = []
                for p in range(p_read):
                    w = BL[c][f"x{p}"] if p == 0 else BL[c][f"x{p}"] - BL[c][f"x{p - 1}"]
                    Ow = (w[0, k - 1] - w[1, k - 1]) / 2 @ G[li]
                    shares.append(float((Ow * np.conj(Ot)).sum().real / den))
                # shares[p] belongs to the write that ends at point p (embed for p = 0)
                lab = ["embed"] + [names[p] for p in range(1, p_read)]
                order = np.argsort(-np.abs(shares))[:6]
                print(f"    L{li} readers k{k:2d} |O| {np.sqrt(den):.3g}: "
                      + ", ".join(f"{lab[i]} {shares[i]:+.2f}" for i in order))  # fmt: skip

    print("\n3. Inside the top sublayers: heads (exact) and neurons (mean-mask down map)")
    for c in CONDS:
        z = Z[c]
        print(f"\n  {c}")
        for li_read, k in ((16, 2), (16, 10), (16, 20), (18, 2), (18, 10), (18, 20)):
            Gr = G[li_read]
            p_read = 2 * li_read + 1
            Ot = (BL[c][f"x{p_read}"][0, k - 1] - BL[c][f"x{p_read}"][1, k - 1]) / 2 @ Gr
            den = (np.abs(Ot) ** 2).sum()
            for wl in range(13, li_read + 1):
                _, Uo, _ = site_uv(wl, "self_attn.o_proj")
                oh = bl(z[f"oh{wl}"])  # (2, 50, H, n_o)
                Oh = (oh[0, k - 1] - oh[1, k - 1]) / 2 @ (Uo @ Gr)  # (H, readers)
                hs = (Oh * np.conj(Ot)[None]).sum(1).real / den
                if wl < li_read:
                    Vd, Ud, dcols = site_uv(wl, "mlp.down_proj")
                    hd = bl(z[f"hd{wl}"])
                    exact = float(
                        (((hd[0, k - 1] - hd[1, k - 1]) / 2 @ (Ud @ Gr)) * np.conj(Ot)).sum().real
                        / den
                    )
                    act = bl(z[f"act{wl}"])  # (2, 50, 14336)
                    mk = np.load(ATLAS / "masks" / f"L{wl}.npy", mmap_mode="r")
                    mmean = np.asarray(mk[:, 4][:, dcols], np.float32).mean(0)
                    Wr = (Vd * mmean[None]) @ (Ud @ Gr)  # (14336, readers)
                    On = ((act[0, k - 1] - act[1, k - 1]) / 2)[:, None] * Wr
                    ns = (On * np.conj(Ot)[None]).sum(1).real / den
                    top = np.argsort(-np.abs(ns))[:5]
                    mlp_txt = (f" | L{wl}.mlp exact {exact:+.2f} approx {ns.sum():+.2f}, F {ns[FLIP].sum():+.2f}: "
                               + ", ".join(f"n{int(n)}{'*' if n in FLIP else ''} {ns[n]:+.2f}" for n in top))  # fmt: skip
                else:
                    mlp_txt, exact = "", 0.0
                if abs(hs.sum()) < 0.05 and (not mlp_txt or abs(exact) < 0.05):
                    continue
                topH = np.argsort(-np.abs(hs))[:3]
                print(f"    L{li_read} readers k{k:2d}: L{wl}.attn {hs.sum():+.2f} ("
                      + ", ".join(f"H{int(h)} {hs[h]:+.2f}" for h in topH) + ")" + mlp_txt)  # fmt: skip


if __name__ == "__main__":
    cast(Any, {"run": run, "analyse": analyse})[sys.argv[1]]()
