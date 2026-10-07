"""Where does the minus sign change the computation (not the encoding of the operation)? Paired swaps of
each sublayer's op-dependent computation on the original model, keeping its op flag.

    python -m param_decomp.arith_repr.vectors.flip_patch run <mode>   # -> OUT/flip/patch_<mode>.json
    (mode: dense on GPU; comp / comp_all fit on CPU)

Pairs. NP random (a, b) pairs; each chunk holds the add prompts of CHUNK / 2 pairs then the sub prompts
of the same pairs, so a row's partner (same a, b, other operation) sits half a chunk away.

Op difference of a sublayer. For sublayer t (t = 2 l: block l's attention, 2 l + 1: its MLP) and
position p, D(a, b) = out_sub(a, b) - out_add(a, b). Its mean over the pairs is the op flag the
sublayer writes (a constant per operation); D - mean D is the op-dependent computation (op x operand).
The base run reports |mean D|^2 and mean |D - mean D|^2 per sublayer (at b and =).

Interaction swap of sublayer t at positions op, b, =: each row's output is replaced by its partner's,
shifted back to its own operation's mean:
    out'(o, a, b) = out(1 - o, a, b) - mu_t(1 - o, p) + mu_t(o, p),
mu_t the base model's op-conditional mean of the sublayer's output. The sublayer then writes its own
op flag but the other operation's op x operand computation. If the answer follows the partner, that
sublayer's op-dependent computation is what turns the operation into a different answer.

Scores per operation: `swap` = top-1 equals the base model's top-1 on the partner prompt, `same` =
equals its own base top-1, `kl` = KL(base on partner || patched).

Runs: base; every single sublayer; four-layer windows of attention / MLP / both; all sublayers from
layer l on, and all sublayers up to layer l."""

import json
import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.isa.components_model import CHUNK, _attention, kl
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_set import Net

NP = 2000
POSM = np.array([0, 0, 1, 1, 1], np.float32)  # op, b, =


MODES = ("dense", "comp", "comp_all")


class Patcher:
    """mode: `dense` the original weights; `comp` the alive components with the dataset's masks (no
    delta); `comp_all` every alive component always on (no masks, no delta)."""

    def __init__(self, mode: str) -> None:
        assert mode in MODES
        self.mode = mode
        self.target: tuple[int, str, int] | None = None  # (layer, kind, position) for `comps`
        self.net = Net(dense=mode == "dense")
        cm = self.net.cm
        rng = np.random.default_rng(0)
        self.pairs = np.sort(rng.choice(10000, NP, replace=False))  # add-row index of each pair
        h = CHUNK // 2
        self.chunks = [
            np.r_[self.pairs[i : i + h], self.pairs[i : i + h] + 10000] for i in range(0, NP, h)
        ]
        self.rows = np.concatenate(self.chunks)
        self.cm = cm
        self._fwd = jax.jit(self._run)

    def _run(
        self,
        P: dict[str, Any],
        x: jax.Array,
        m: list[jax.Array],
        patch: jax.Array,
        mu: jax.Array,
        rpatch: jax.Array,
        wpatch: jax.Array,
        cpatch: jax.Array,
        dpatch: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """x (B, T, d) embeddings of one chunk (add half, sub half); patch (64,) 0/1; mu (64, 2, T, d).
        Returns last-position log-probs, per-sublayer output sums by op (64, 2, T, d), sums of D
        (64, T, d) and of |D|^2 (64, T)."""
        cm = self.cm
        B, T, _ = x.shape
        h = B // 2
        perm = jnp.r_[jnp.arange(h, B), jnp.arange(h)]
        opv = jnp.r_[jnp.zeros(h, jnp.int32), jnp.ones(h, jnp.int32)]
        posm = jnp.asarray(POSM)[None, :, None]
        sums, sD, sD2 = [], [], []
        for li in range(32):

            def site(kind: str, xin: jax.Array) -> jax.Array:
                V, U, cols = P["sites"][li][kind]  # noqa: B023
                if self.mode == "dense":
                    y = xin @ P["W"][li][kind].astype(jnp.float32).T  # noqa: B023
                    if self.target is not None and self.target[:2] == (li, kind):  # noqa: B023
                        h = xin @ V
                        delta = y - h @ U
                        pm = jnp.zeros(T).at[self.target[2]].set(1.0)[None, :, None]
                        y = y + pm * ((cpatch * (h[perm] - h)) @ U + dpatch * (delta[perm] - delta))  # noqa: B023
                    return y
                h = xin @ V
                if self.mode == "comp":
                    h = h * m[li][:, :, cols]  # noqa: B023
                return h @ U

            for off in (0, 1):
                t = 2 * li + off
                x = x + rpatch[t][None, :, None] * (x[perm] - x)  # residual patch from the partner
                rms = jnp.sqrt((x * x).mean(-1) + cm.eps)
                xin = x / rms[..., None] * P["ln"][off][li]
                if off == 0:
                    q = site("q", xin).reshape(B, T, cm.n_head, cm.hd)
                    k = site("k", xin).reshape(B, T, cm.n_kv, cm.hd)
                    v = site("v", xin).reshape(B, T, cm.n_kv, cm.hd)
                    out = site("o", _attention(q, k, v, P["cos"], P["sin"]))
                else:
                    out = site("down", jax.nn.silu(site("gate", xin)) * site("up", xin))
                sums.append(jnp.stack([out[:h].sum(0), out[h:].sum(0)]))
                D = out[h:] - out[:h]
                sD.append(D.sum(0))
                sD2.append((D * D).sum((0, 2)))
                swapped = out[perm] - mu[t][1 - opv] + mu[t][opv]
                out = out + patch[t] * posm * (swapped - out)  # 1: swap, 0.5: op-symmetric
                out = out + wpatch[t][None, :, None] * (
                    out[perm] - out
                )  # whole write from the partner
                x = x + out
        x = x + rpatch[64][None, :, None] * (x[perm] - x)
        xl = x[:, -1]
        xf = xl / jnp.sqrt((xl * xl).mean(-1, keepdims=True) + cm.eps) * P["final"]
        lp = jax.nn.log_softmax(xf @ P["unembed"].T, -1)
        return lp, jnp.stack(sums), jnp.stack(sD), jnp.stack(sD2)

    def run(
        self,
        patch: np.ndarray,
        mu: np.ndarray,
        rpatch: np.ndarray | None = None,
        wpatch: np.ndarray | None = None,
        cpatch: np.ndarray | None = None,
        dpatch: float = 0.0,
    ) -> tuple[np.ndarray, ...]:
        rp = jnp.asarray(np.zeros((65, 5), np.float32) if rpatch is None else rpatch)
        wp = jnp.asarray(np.zeros((64, 5), np.float32) if wpatch is None else wpatch)
        cp = jnp.asarray(np.zeros(1, np.float32) if cpatch is None else cpatch, jnp.float32)
        lps, acc_s, acc_d, acc_d2 = [], 0.0, 0.0, 0.0
        P = self.net.P
        for rows in self.chunks:
            x0 = jnp.asarray(np.stack([[self.cm.embed[int(t)] for t in toks] for toks in self.cm.tokens[rows]]), jnp.float32)  # fmt: skip
            m = (
                [jnp.asarray(self.cm.masks(li, rows)) for li in range(32)]
                if self.mode == "comp"
                else []  # comp_all (the alive-only model) and dense need no masks
            )
            lp, s, sd, sd2 = self._fwd(
                P, x0, m, jnp.asarray(patch), jnp.asarray(mu), rp, wp, cp, jnp.float32(dpatch)
            )
            lps.append(np.asarray(lp))
            acc_s, acc_d, acc_d2 = (
                acc_s + np.asarray(s),
                acc_d + np.asarray(sd),
                acc_d2 + np.asarray(sd2),
            )
        return (
            np.concatenate(lps),
            np.asarray(acc_s) / NP,
            np.asarray(acc_d) / NP,
            np.asarray(acc_d2) / NP,
        )


def scores(
    base: np.ndarray,
    lp: np.ndarray,
    chunk_half: int,
    gt: np.ndarray,
    answer: np.ndarray | None = None,
) -> dict[str, float]:
    """Rows are chunked (add half, sub half); the partner of row r is r +- chunk_half. gt: a > b per
    row; scores are given on all pairs and on the pairs with a > b (where both answers are numbers)."""
    n = len(base)
    idx = np.arange(n)
    partner = np.where((idx % (2 * chunk_half)) < chunk_half, idx + chunk_half, idx - chunk_half)
    is_sub = (idx % (2 * chunk_half)) >= chunk_half
    tb, te = base.argmax(-1), lp.argmax(-1)
    d = kl(base[partner], lp)
    out = {}
    for nm, sel0 in (("add", ~is_sub), ("sub", is_sub)):
        for tag, sel in (("", sel0), ("_gt", sel0 & gt)):
            out[f"swap_{nm}{tag}"] = float((te[sel] == tb[partner][sel]).mean())
            out[f"same_{nm}{tag}"] = float((te[sel] == tb[sel]).mean())
            out[f"kl_{nm}{tag}"] = float(d[sel].mean())
            if answer is not None:
                out[f"acc_{nm}{tag}"] = float((te[sel] == answer[sel]).mean())
    return out


def run(mode: str) -> None:
    pt = Patcher(mode)
    out_file = DIR / f"patch_{mode}.json"
    h = CHUNK // 2
    zero = np.zeros(64, np.float32)
    mu0 = np.zeros((64, 2, 5, 4096), np.float32)
    base, mu, sD, sD2 = pt.run(zero, mu0)
    gt = pt.cm.a[pt.rows] > pt.cm.b[pt.rows]
    res: dict[str, Any] = {"identity": scores(base, base, h, gt)}
    print("identity (base vs its partner):", res["identity"], flush=True)
    md2 = (sD**2).sum(-1)
    inter = sD2 - md2
    print(
        "\nop difference written by each sublayer at b / =: |mean D|^2 (flag) and mean |D - mean D|^2 (op x operand)"
    )
    for t in range(64):
        nm = f"L{t // 2}.{'attn' if t % 2 == 0 else 'mlp'}"
        print(
            f"  {nm:9s} b: flag {md2[t, 3]:8.3g} op x operand {inter[t, 3]:8.3g} | =: flag {md2[t, 4]:8.3g} op x operand {inter[t, 4]:8.3g}"
        )
    res["diff"] = {"flag": md2.tolist(), "inter": inter.tolist()}

    def go(name: str, ts: list[int]) -> None:
        patch = zero.copy()
        patch[ts] = 1
        lp, _, _, _ = pt.run(patch, mu)
        sc = scores(base, lp, h, gt)
        res[name] = sc
        print(f"{name}: " + " ".join(f"{k} {v:.3f}" for k, v in sc.items()), flush=True)
        out_file.write_text(json.dumps(res, indent=1))

    print("\ninteraction swaps (positions op, b, =)")
    go("all", list(range(64)))
    for t in range(64):
        go(f"L{t // 2}.{'attn' if t % 2 == 0 else 'mlp'}", [t])
    for l0 in range(0, 32, 4):
        ls = range(l0, l0 + 4)
        go(f"attn L{l0}-{l0 + 3}", [2 * li for li in ls])
        go(f"mlp L{l0}-{l0 + 3}", [2 * li + 1 for li in ls])
        go(f"both L{l0}-{l0 + 3}", [t for li in ls for t in (2 * li, 2 * li + 1)])
    for l0 in range(0, 32, 2):
        go(f"from L{l0}", list(range(2 * l0, 64)))
        go(f"up to L{l0}", list(range(0, 2 * l0 + 2)))


def sym(mode: str) -> None:
    """Necessity of each sublayer's op-dependent computation: patch 0.5 makes it op-symmetric (the mean
    of the two operations' computations) while keeping its own op flag. Scores include `acc` (top-1 is
    the correct answer token; on sub with a < b the reference answer is the minus sign)."""
    pt = Patcher(mode)
    out_file = DIR / f"sym_{mode}.json"
    h = CHUNK // 2
    zero = np.zeros(64, np.float32)
    base, mu, _, _ = pt.run(zero, np.zeros((64, 2, 5, 4096), np.float32))
    gt = pt.cm.a[pt.rows] > pt.cm.b[pt.rows]
    ans = pt.cm.answer[pt.rows]
    res: dict[str, Any] = {"base": scores(base, base, h, gt, ans)}
    print(
        "base", {k: round(v, 3) for k, v in res["base"].items() if k.startswith("acc")}, flush=True
    )

    def go(name: str, ts: list[int]) -> None:
        patch = zero.copy()
        patch[ts] = 0.5
        lp, _, _, _ = pt.run(patch, mu)
        sc = scores(base, lp, h, gt, ans)
        res[name] = sc
        keys = (
            "acc_add_gt",
            "acc_sub_gt",
            "acc_sub",
            "same_add",
            "same_sub",
            "kl_add",
            "kl_sub",
            "swap_add_gt",
            "swap_sub_gt",
        )
        print(f"{name}: " + " ".join(f"{k} {sc[k]:.3f}" for k in keys), flush=True)
        out_file.write_text(json.dumps(res, indent=1))

    go("all", list(range(64)))
    for t in range(64):
        go(f"L{t // 2}.{'attn' if t % 2 == 0 else 'mlp'}", [t])
    for l0 in range(0, 32, 4):
        ls = range(l0, l0 + 4)
        go(f"attn L{l0}-{l0 + 3}", [2 * li for li in ls])
        go(f"mlp L{l0}-{l0 + 3}", [2 * li + 1 for li in ls])
        go(f"both L{l0}-{l0 + 3}", [t for li in ls for t in (2 * li, 2 * li + 1)])
    for l0 in range(0, 32, 2):
        go(f"from L{l0}", list(range(2 * l0, 64)))
        go(f"up to L{l0}", list(range(0, 2 * l0 + 2)))


def resid(mode: str) -> None:
    """Residual patching: the whole stream at point t (t = 2 l before block l's attention, 2 l + 1
    before its MLP, 64 before the final norm) and position(s) p is copied from the partner prompt
    (same a, b, other operation). `swap` = the answer follows the partner."""
    pt = Patcher(mode)
    out_file = DIR / f"resid_{mode}.json"
    h = CHUNK // 2
    zero = np.zeros(64, np.float32)
    mu0 = np.zeros((64, 2, 5, 4096), np.float32)
    base, _, _, _ = pt.run(zero, mu0)
    gt = pt.cm.a[pt.rows] > pt.cm.b[pt.rows]
    ans = pt.cm.answer[pt.rows]
    res: dict[str, Any] = {"base": scores(base, base, h, gt, ans)}
    for t in range(0, 65, 2):
        for pname, ps in (("op", [2]), ("b", [3]), ("=", [4]), ("op+b", [2, 3])):
            rp = np.zeros((65, 5), np.float32)
            rp[t, ps] = 1
            lp, _, _, _ = pt.run(zero, mu0, rp)
            sc = scores(base, lp, h, gt, ans)
            res[f"{t}|{pname}"] = sc
            print(f"point {t:2d} (L{t // 2} in) {pname:4s}: " + " ".join(f"{k} {sc[k]:.3f}" for k in
                  ("swap_add_gt", "swap_sub_gt", "same_add_gt", "same_sub_gt", "swap_add", "swap_sub")), flush=True)  # fmt: skip
        out_file.write_text(json.dumps(res, indent=1))


def writes(mode: str) -> None:
    """Which writes put the operation into `=` before L16: the whole output of the chosen sublayers at
    `=` only (flag and computation) is copied from the partner; everything else is the prompt's own.
    Also the same at position b only."""
    pt = Patcher(mode)
    out_file = DIR / f"writes_{mode}.json"
    h = CHUNK // 2
    zero = np.zeros(64, np.float32)
    mu0 = np.zeros((64, 2, 5, 4096), np.float32)
    base, _, _, _ = pt.run(zero, mu0)
    gt = pt.cm.a[pt.rows] > pt.cm.b[pt.rows]
    ans = pt.cm.answer[pt.rows]
    res: dict[str, Any] = {}

    def go(name: str, ts: list[int], pos: int = 4) -> None:
        wp = np.zeros((64, 5), np.float32)
        wp[ts, pos] = 1
        lp, _, _, _ = pt.run(zero, mu0, None, wp)
        sc = scores(base, lp, h, gt, ans)
        res[name] = sc
        keys = (
            "swap_add_gt",
            "swap_sub_gt",
            "same_add_gt",
            "same_sub_gt",
            "acc_add_gt",
            "acc_sub_gt",
        )
        print(f"{name}: " + " ".join(f"{k} {sc[k]:.3f}" for k in keys), flush=True)
        out_file.write_text(json.dumps(res, indent=1))

    for t in range(32):
        go(f"= L{t // 2}.{'attn' if t % 2 == 0 else 'mlp'}", [t])
    attn = [t for t in range(32) if t % 2 == 0]
    mlp = [t for t in range(32) if t % 2 == 1]
    go("= all attn L0-15", attn)
    go("= all mlp L0-15", mlp)
    go("= all L0-15", list(range(32)))
    go("= L15 attn+mlp", [30, 31])
    go("= L13-15", list(range(26, 32)))
    go("= L0-4", list(range(10)))
    for t in range(32):
        go(f"b L{t // 2}.{'attn' if t % 2 == 0 else 'mlp'}", [t], pos=3)
    go("b all L0-15", list(range(32)), pos=3)
    go("b mlp L0-15", mlp, pos=3)
    go("b attn L0-15", attn, pos=3)


def comps(mode: str, layer: str, kind: str, pos: str) -> None:
    """Component-level copy from the partner at one writer site (o or down) and position: each alive
    component's contribution (x V_c) U_c alone, all of them, and the remainder (delta plus the dead
    components) alone. Dense only."""
    assert mode == "dense"
    pt = Patcher(mode)
    li, p = int(layer), int(pos)
    pt.target = (li, kind, p)
    pt._fwd = jax.jit(pt._run)
    names = pt.net.cm.sites[li][kind].names
    n = len(names)
    out_file = DIR / f"comps_{mode}_L{li}{kind}_p{p}.json"
    h = CHUNK // 2
    zero = np.zeros(64, np.float32)
    mu0 = np.zeros((64, 2, 5, 4096), np.float32)
    base, _, _, _ = pt.run(zero, mu0, cpatch=np.zeros(n, np.float32))
    gt = pt.cm.a[pt.rows] > pt.cm.b[pt.rows]
    ans = pt.cm.answer[pt.rows]
    res: dict[str, Any] = {}

    def go(name: str, cp: np.ndarray, dp: float) -> None:
        lp, _, _, _ = pt.run(zero, mu0, cpatch=cp, dpatch=dp)
        sc = scores(base, lp, h, gt, ans)
        res[name] = sc
        print(
            f"{name}: "
            + " ".join(
                f"{k} {sc[k]:.3f}"
                for k in ("swap_add_gt", "swap_sub_gt", "same_add_gt", "same_sub_gt")
            ),
            flush=True,
        )
        out_file.write_text(json.dumps(res, indent=1))

    go("everything", np.ones(n, np.float32), 1.0)
    go("all components", np.ones(n, np.float32), 0.0)
    go("delta + dead", np.zeros(n, np.float32), 1.0)
    for c in range(n):
        e = np.zeros(n, np.float32)
        e[c] = 1
        go(names[c], e, 0.0)


if __name__ == "__main__":
    cast(Any, {"run": run, "sym": sym, "resid": resid, "writes": writes, "comps": comps})[
        sys.argv[1]
    ](*sys.argv[2:])
