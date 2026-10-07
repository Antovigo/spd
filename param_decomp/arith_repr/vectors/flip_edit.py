"""Weight-space edits to the alive components that should swap addition and subtraction, solved in closed
form from the components' measured activations (no optimization), tested on the original model.

    python -m param_decomp.arith_repr.vectors.flip_edit diagnose  # CPU -> stdout: op shift = mean shift?
    python -m param_decomp.arith_repr.vectors.flip_edit run       # GPU -> OUT/flip/edits.json

Model. The original weights (dense, delta included): an edit replaces component c's U_c by U_c + dU_c,
i.e. adds the rank-one term dU_c (x V_c) to the site. The delta is assumed to hold no op mechanism.
The components-only model is not used: its masks come from the original model's CI and are tied to
the operation (e.g. L15 gate c72 is masked off on every addition prompt), so no weight edit can make a
sub-only component fire on addition there.

Mean swap of a site. Let hbar_c(o, p) be component c's mean inner activation (x V_c, original model)
on op o in {add, sub} at position p in {BOS, a, op, b, =}, and v_c its variance within (o, p)
(averaged). For targets Tgt(o, p) (vectors in the site's output space), the edit solves
    sum_c hbar_c(o, p) dU_c = Tgt(o, p)   for all 10 (o, p),
minimising sum_c v_c |dU_c|^2 (the variance the edit injects): dU = W^-1 H (H^T W^-1 H)^+ Tgt with
H (n_c, 10) = hbar, W = diag(v). The site's mean output at (o, p) then moves by Tgt(o, p).

Edits (prediction in brackets):
* `switch15`: L15 gate c9 (on add) <-> c72 (on sub) and up c34 (add) <-> c117 (sub) exchange their
  U, rescaled by the ratio of their mean inner at `=` on their own op. [The b mirror moves from sub
  to add at the L16 readers; the answers barely change, since removing the mirror did not change them.]
* `mlp15_in`: mean swap of the L15 gate and up sites, Tgt(o, p) = gbar(1-o, p) - gbar(o, p) (and the
  same for up) at p in {op, b, =}, 0 at BOS and a. [Every L15 neuron sees the other op's mean inputs;
  same prediction as switch15 plus whatever else L15 does with the op.]
* `stream<t>`: mean swap of the residual writers (o or down) of sublayer t, Tgt(o, p) = xbar(1-o, p)
  - xbar(o, p) at the stream point right after t, for p in {op, b, =} (0 at BOS, a). [If the
  operation reaches the answer only through the op-conditional MEAN of the stream after t, the
  edited model answers a - b on add prompts and a + b on sub prompts. `diagnose` measures how much of
  the op difference is a mean shift at each point: the swap can work only where that share is ~1.]

Scores per op, with each prompt (o, a, b) paired with its counterpart (1 - o, a, b): `swap` = top-1 of
the edited model equals the original model's top-1 on the counterpart; `same` = equals its own; `kl`
= KL(original on counterpart || edited)."""

import json
import sys
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import kl
from param_decomp.arith_repr.vectors.common import DATASET, comp_table
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_set import READ, Net, flip_stats

POS = ("BOS", "a", "op", "b", "=")
CUTS = ((0, "o"), (0, "down"), (1, "o"), (1, "down"), (2, "o"), (2, "down"), (3, "down"), (5, "down"),
        (8, "down"), (11, "down"), (14, "down"), (15, "o"), (15, "down"), (17, "down"), (20, "down"))  # fmt: skip
SUFFIX = {
    "o": "self_attn.o_proj",
    "down": "mlp.down_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
}


def write_point(layer: int, kind: str) -> int:
    return 1 + 2 * layer + (0 if kind == "o" else 1)


def site_stats(
    net: Net, sites: list[tuple[int, str]]
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Per site: hbar (n_c, 2, 5) and within-(op, position) variance (n_c,) of its alive components on
    the original model, columns in the components model's order. One pass over the inner dataset."""
    comps = comp_table()
    dcols = {}
    for li, kind in sites:
        lcols = np.flatnonzero(comps["layer"] == li)
        dcols[(li, kind)] = lcols[np.asarray(net.cm.sites[li][kind].cols)]
    allc = np.unique(np.concatenate(list(dcols.values())))
    mm = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    s1 = np.zeros((2, 5, len(allc)))
    s2 = np.zeros_like(s1)
    for r0 in range(0, 20000, 1000):
        h = np.asarray(mm[r0 : r0 + 1000], np.float64)[:, :, allc]  # (1000, 5, n)
        o = r0 // 10000
        s1[o] += h.sum(0)
        s2[o] += (h**2).sum(0)
    mean = s1 / 10000
    var = s2 / 10000 - mean**2
    out = {}
    for key, dc in dcols.items():
        j = np.searchsorted(allc, dc)
        out[key] = (np.moveaxis(mean[:, :, j], -1, 0), var[:, :, j].mean((0, 1)))
    return out


def stream_means(point: int) -> np.ndarray:
    """(2, 5, d) op-conditional means of the raw stream at a point (original model)."""
    x = np.asarray(np.load(DATASET / "original/resid.npy", mmap_mode="r")[point], np.float64)
    return x.reshape(2, 10000, 5, -1).mean(1)


def preact_means(layer: int, kind: str) -> np.ndarray:
    x = np.load(DATASET / f"original/mlp_{kind}.npy", mmap_mode="r")[layer]
    return np.asarray(x, np.float64).reshape(2, 10000, 5, -1).mean(1)


def solve(hbar: np.ndarray, var: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, float]:
    """hbar (n_c, 2, 5), var (n_c,), tgt (2, 5, d) -> dU (n_c, d) and the relative constraint residual."""
    H = hbar.reshape(len(hbar), 10)
    Tg = tgt.reshape(10, -1)
    winv = 1.0 / np.maximum(var, 1e-6 * var.max())
    A = H.T @ (winv[:, None] * H)
    dU = (winv[:, None] * H) @ np.linalg.pinv(A, rcond=1e-8) @ Tg
    res = np.linalg.norm(H.T @ dU - Tg) / max(np.linalg.norm(Tg), 1e-12)
    return dU, float(res)


def swap_targets(means: np.ndarray) -> np.ndarray:
    """(2, 5, d) means -> targets: the other op's mean minus this op's, 0 at BOS and a."""
    tgt = means[::-1] - means
    tgt[:, :2] = 0
    return tgt


def diagnose() -> None:
    """Share of the op difference that is a mean shift: paired prompts (add, a, b) / (sub, a, b),
    D = x_sub - x_add; share = |mean D|^2 / mean |D|^2, per stream point and position."""
    mm = np.load(DATASET / "original/resid.npy", mmap_mode="r")
    print(
        "share of the paired op difference that is the mean shift, positions op / b / =; |mean D| / rms(x)"
    )
    for layer, kind in CUTS:
        t = write_point(layer, kind)
        x = np.asarray(mm[t], np.float32)
        D = x[10000:] - x[:10000]
        row = []
        for p in (2, 3, 4):
            d = D[:, p].mean(0)
            share = float((d**2).sum() / (D[:, p] ** 2).sum(1).mean())
            rms = float(np.sqrt((x[:, p] ** 2).mean()))
            row.append(f"{POS[p]} {share:.2f} ({np.linalg.norm(d) / rms / np.sqrt(4096):.2f})")
        print(f"  after L{layer}.{kind}: " + "  ".join(row), flush=True)


def scores(base: np.ndarray, lp: np.ndarray) -> dict[str, float]:
    other = np.r_[np.arange(10000, 20000), np.arange(10000)]
    tb, te = base.argmax(-1), lp.argmax(-1)
    d = kl(base[other], lp)
    out = {}
    for o, nm in ((0, "add"), (1, "sub")):
        sl = slice(o * 10000, (o + 1) * 10000)
        out[f"swap_{nm}"] = float((te[sl] == tb[other][sl]).mean())
        out[f"same_{nm}"] = float((te[sl] == tb[sl]).mean())
        out[f"kl_{nm}"] = float(d[sl].mean())
    return out


def run() -> None:
    net = Net(dense=True)
    s = net.scales_one()
    rows = np.arange(20000)
    base = net.logprobs(s, rows)
    results: dict[str, Any] = {"identity": scores(base, base)}
    print("identity", results["identity"], flush=True)
    phi0 = {
        li: float(flip_stats(jnp.asarray(v))[0])
        for li, v in ((int(k[1:]), v) for k, v in net.light(s).items())
    }

    def apply(name: str, edits: dict[tuple[int, str], np.ndarray], note: str) -> None:
        net.P["dU"] = [{} for _ in range(32)]
        for (li, kind), dU in edits.items():
            j = jnp.arange(dU.shape[0])
            net.P["dU"][li][kind] = (j, jnp.asarray(dU, jnp.float32))
        lp = net.logprobs(s, rows)
        sc: dict[str, Any] = dict(scores(base, lp))
        light = net.light(s)
        sc["phi_rel"] = {
            li: float(flip_stats(jnp.asarray(light[f"r{li}"]))[0]) / phi0[li]
            for li in READ
            if li <= 20
        }
        sc["note"] = note
        results[name] = sc
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items() if k != "phi_rel"},
              "Phi rel L16-20:", {k: round(v, 2) for k, v in sc["phi_rel"].items()}, flush=True)  # fmt: skip
        net.P["dU"] = [{} for _ in range(32)]
        (DIR / "edits.json").write_text(json.dumps(results, indent=1))

    stats = site_stats(net, [(15, "gate"), (15, "up"), *CUTS])

    # switch15: exchange the op switches' U at L15, rescaled to the partner's mean inner at '='
    edits: dict[tuple[int, str], np.ndarray] = {}
    for kind, (ca, cs) in (("gate", ("c9", "c72")), ("up", ("c34", "c117"))):
        hbar, _ = stats[(15, kind)]
        names = net.cm.sites[15][kind].names
        ia, is_ = names.index(f"L15.{kind}.{ca}"), names.index(f"L15.{kind}.{cs}")
        U = np.asarray(net.cm.sites[15][kind].U, np.float64)
        dU = np.zeros_like(U)
        ha, hs = hbar[ia, 0, 4], hbar[is_, 1, 4]
        dU[ia] = U[is_] * hs / ha - U[ia]
        dU[is_] = U[ia] * ha / hs - U[is_]
        edits[(15, kind)] = dU
        print(
            f"switch15 {kind}: {ca} mean add {ha:.2f} (sub {hbar[ia, 1, 4]:.2f}), {cs} mean sub {hs:.2f} (add {hbar[is_, 0, 4]:.2f})"
        )
    apply("switch15", edits, "L15 gate c9<->c72, up c34<->c117")

    # mlp15_in: mean swap of the L15 gate and up pre-activations
    edits = {}
    for kind in ("gate", "up"):
        hbar, var = stats[(15, kind)]
        dU, res = solve(hbar, var, swap_targets(preact_means(15, kind)))
        edits[(15, kind)] = dU
        print(
            f"mlp15_in {kind}: residual {res:.3g}, |dU| / |U| {np.linalg.norm(dU) / np.linalg.norm(np.asarray(net.cm.sites[15][kind].U)):.2f}"
        )
    apply("mlp15_in", edits, "mean swap of L15 gate/up inputs")

    # stream<t>: mean swap of the stream after sublayer t
    for li, kind in CUTS:
        hbar, var = stats[(li, kind)]
        tgt = swap_targets(stream_means(write_point(li, kind)))
        dU, res = solve(hbar, var, tgt)
        noise = float((var[:, None] * dU**2).sum() / max((tgt[:, 2:] ** 2).sum() / 6, 1e-12))
        print(
            f"stream L{li}.{kind}: {len(hbar)} comps, residual {res:.3g}, injected variance / mean |Tgt|^2 {noise:.3g}"
        )
        apply(f"stream_L{li}.{kind}", {(li, kind): dU}, f"mean swap after L{li}.{kind}")


if __name__ == "__main__":
    cast(Any, {"diagnose": diagnose, "run": run})[sys.argv[1]]()
