"""Are the L31 "temperature" components temperature components, on the target pool and on fineweb?

Everything runs on the ORIGINAL model (dense Llama-3.1-8B, fp32 compute). A component is ablated by
editing its rank-one term inside the dense weight, `y = x W^T + (f(h) - h) U_c`, `h = x . V_c`, with
f = 0 (zero ablation) or f = the component's mean inner activation over the data (mean ablation, the
entropy-neuron paper's choice). No CI is involved, so the result does not depend on the CI fn.

Per position, with z_b the base logits and x_b / x_a the final residual without / with the ablation:
    total   KL(p_b || softmax(W (x_a / rms(x_a))))    the ablation
    direct  KL(p_b || softmax(W (x_a / rms(x_b))))    final-norm scale frozen (Stolfo et al.'s DE)
    norm    KL(p_b || softmax(W (x_b / rms(x_a))))    only the final-norm scale changes
    resid   min_beta KL(softmax(beta z_b) || p_a)     what a pure temperature change cannot explain
plus entropies, argmax flips, next-token CE, and the write dx = x_b - x_a projected on the
unembedding's bottom singular directions (null space) and on the uniform-logit direction.

    python -m param_decomp.arith_repr.rmsnorm.temperature scan <shard> <n_shards>   # every L30-31 comp, fineweb
    python -m param_decomp.arith_repr.rmsnorm.temperature deep                      # candidates, fineweb
    python -m param_decomp.arith_repr.rmsnorm.temperature addsub                    # candidates, addsub pool at =
    python -m param_decomp.arith_repr.rmsnorm.temperature neurons                   # Stolfo-style L31 neuron scan"""

import json
import sys
import time
from collections.abc import Callable, Set

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import (
    Weights,
    llama3_inv_freq,
    number_token_ids,
)
from param_decomp.arith_repr.isa.components_model import KINDS, _attention, _rms
from param_decomp.arith_repr.rmsnorm.run import OUT, savez
from param_decomp.arith_repr.vectors.common import DATASET, RESID

jax.config.update("jax_default_matmul_precision", "highest")

FW = OUT / "fineweb"
TEMP = OUT / "temperature"
CS = {"q": 144, "k": 144, "v": 256, "o": 512, "gate": 912, "up": 912, "down": 1024}
OFFSET = dict(zip(CS, np.cumsum([0] + list(CS.values()))[:-1].tolist(), strict=False))
BETAS = np.exp(np.linspace(-1.0, 1.0, 41)).astype(np.float32)
NULL_K = (10, 40, 100)
CANDIDATES = [
    "L31.gate.c14",
    "L31.gate.c238",
    "L31.gate.c288",
    "L31.up.c534",
    "L31.up.c36",
    "L31.up.c50",
]
CONTROLS = [
    "L31.down.c18",
    "L31.up.c731",
    "L31.down.c23",
    "L30.up.c367",
    "L30.gate.c45",
    "L31.v.c28",
]
EPS = 1e-5
NO_POINTS: frozenset[int] = frozenset()
Stats = dict[str, jax.Array]
Mods = dict[tuple[int, str], tuple[jax.Array, jax.Array, Callable[[jax.Array], jax.Array]]]

_ = ml_dtypes.bfloat16  # importing ml_dtypes registers bfloat16 for the safetensors reads


def log(*a: object) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@jax.jit
def _dense(x: jax.Array, W: jax.Array) -> jax.Array:
    return x @ W.astype(jnp.float32).T


@jax.jit
def _normed(x: jax.Array, g: jax.Array) -> jax.Array:
    return x / _rms(x, EPS)[..., None] * g


class TextModel:
    def __init__(self, T: int) -> None:
        w = Weights()
        cfg = w.config
        self.L, self.H, self.KV = (
            cfg["num_hidden_layers"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
        )
        self.hd = cfg["hidden_size"] // self.H
        self.W = [{k: jnp.asarray(w.get(f"model.layers.{li}.{s}.weight"), jnp.bfloat16) for k, s in KINDS.items()}
                  for li in range(self.L)]  # fmt: skip
        norms = np.load(RESID / "norms.npz")
        self.ln = [jnp.asarray(norms["ln1"]), jnp.asarray(norms["ln2"])]
        self.g = jnp.asarray(norms["final"])
        self.embed = jnp.asarray(w.get("model.embed_tokens.weight"), jnp.bfloat16)
        self.unembed = jnp.asarray(w.get("lm_head.weight"))
        self.Wd31 = self.W[31]["down"]
        inv = llama3_inv_freq(cfg)
        fr = np.arange(T, dtype=np.float32)[:, None] * inv[None]
        e = np.concatenate([fr, fr], -1)
        self.cos, self.sin = jnp.asarray(np.cos(e)), jnp.asarray(np.sin(e))
        # W_eff = W_U diag(g): bottom singular directions and the uniform-logit direction
        Weff = self.Weff = self.unembed * self.g[None]
        ev, evec = jnp.linalg.eigh(Weff.T @ Weff)
        self.null = {k: evec[:, :k] for k in NULL_K}
        u = Weff.T @ jnp.ones(Weff.shape[0])
        self.uniform = u / jnp.linalg.norm(u)
        self.eig = np.asarray(ev)

    def site(self, li: int, kind: str, xin: jax.Array, mods: Mods) -> jax.Array:
        y = _dense(xin, self.W[li][kind])
        if (li, kind) in mods:
            V, U, f = mods[li, kind]  # V (d_in,), U (d_out,), f: h -> replacement
            h = xin @ V
            y = y + (f(h) - h)[..., None] * U
        return y

    def block(
        self, li: int, off: int, x: jax.Array, mods: Mods, want_h: bool = False
    ) -> tuple[jax.Array, jax.Array | None]:
        B, T, _ = x.shape
        xin = _normed(x, self.ln[off][li])
        if off == 0:
            q = self.site(li, "q", xin, mods).reshape(B, T, self.H, self.hd)
            k = self.site(li, "k", xin, mods).reshape(B, T, self.KV, self.hd)
            v = self.site(li, "v", xin, mods).reshape(B, T, self.KV, self.hd)
            return x + self.site(
                li, "o", _attention(q, k, v, self.cos[:T], self.sin[:T]), mods
            ), None
        h = jax.nn.silu(self.site(li, "gate", xin, mods)) * self.site(li, "up", xin, mods)
        return x + self.site(li, "down", h, mods), (h if want_h else None)

    def run(
        self, x: jax.Array, t0: int, mods: Mods, capture: Set[int] = NO_POINTS, want_h: bool = False
    ) -> tuple[jax.Array, dict[int, jax.Array], jax.Array | None]:
        """From norm point t0 (stream x entering it) to the final residual; streams entering `capture`."""
        caps, h31 = {}, None
        for t in range(t0, 2 * self.L):
            if t in capture:
                caps[t] = x
            x, h = self.block(t // 2, t % 2, x, mods, want_h and t == 63)
            if h is not None:
                h31 = h
        return x, caps, h31

    def embed_tokens(self, tokens: np.ndarray) -> jax.Array:
        return self.embed[jnp.asarray(tokens)].astype(jnp.float32)

    def inner(self, x: jax.Array, li: int, kind: str, V: jax.Array) -> jax.Array:
        """Inner activations (…, n) of components V (d_in, n) at site (li, kind) given the stream entering it."""
        off = 0 if kind in ("q", "k", "v", "o") else 1
        xin = _normed(x, self.ln[off][li])
        if kind in ("q", "k", "v", "gate", "up"):
            return xin @ V
        B, T, _ = x.shape
        if kind == "o":
            q = _dense(xin, self.W[li]["q"]).reshape(B, T, self.H, self.hd)
            k = _dense(xin, self.W[li]["k"]).reshape(B, T, self.KV, self.hd)
            v = _dense(xin, self.W[li]["v"]).reshape(B, T, self.KV, self.hd)
            return _attention(q, k, v, self.cos[:T], self.sin[:T]) @ V
        h = jax.nn.silu(_dense(xin, self.W[li]["gate"])) * _dense(xin, self.W[li]["up"])
        return h @ V


@jax.jit
def metrics(xb: jax.Array, xa: jax.Array, g: jax.Array, WU: jax.Array, target: jax.Array) -> Stats:
    """Per-position effect of one ablation, xb / xa (B, T, d) the base / ablated final residual."""
    rb, ra = _rms(xb, EPS)[..., None], _rms(xa, EPS)[..., None]

    def lsm(x: jax.Array) -> jax.Array:
        return jax.nn.log_softmax((x * g) @ WU.T, -1)

    zb = (xb / rb * g) @ WU.T
    lb = jax.nn.log_softmax(zb, -1)
    pb = jnp.exp(lb)
    la, lf, ln = lsm(xa / ra), lsm(xa / rb), lsm(xb / ra)
    pa = jnp.exp(la)

    def kl(p: jax.Array, lp: jax.Array, lq: jax.Array) -> jax.Array:
        return (p * (lp - lq)).sum(-1)

    def tempered(beta: jax.Array) -> jax.Array:
        lt = jax.nn.log_softmax(beta * zb, -1)
        return kl(jnp.exp(lt), lt, la)

    res = jax.lax.map(tempered, jnp.asarray(BETAS))  # (nbeta, B, T)
    best = jnp.argmin(res, 0)
    tgt = jnp.maximum(target, 0)[..., None]
    ce = lambda lp: jnp.where(target >= 0, -jnp.take_along_axis(lp, tgt, -1)[..., 0], 0.0)  # noqa: E731
    return {
        "kl_total": kl(pb, lb, la), "kl_direct": kl(pb, lb, lf), "kl_norm": kl(pb, lb, ln),
        "kl_resid": res.min(0), "log_beta": jnp.log(jnp.asarray(BETAS))[best],
        "H_b": -(pb * lb).sum(-1), "H_a": -(pa * la).sum(-1),
        "flip": (jnp.argmax(la, -1) != jnp.argmax(lb, -1)).astype(jnp.float32),
        "ce_b": ce(lb), "ce_a": ce(la), "dlog_rms": jnp.log(ra / rb)[..., 0],
    }  # fmt: skip


def write_stats(M: TextModel, dx: jax.Array) -> Stats:
    """Second moments of the write dx (B, T, d): total, in the bottom-k null space, along the uniform-logit
    direction; and its logit change split into uniform shift and the rest."""
    dz = dx @ M.Weff.T
    out = {"e_total": (dx * dx).sum(-1), "e_uniform": (dx @ M.uniform) ** 2,
           "dz_mean2": dz.mean(-1) ** 2, "dz_var": dz.var(-1)}  # fmt: skip
    for k, B in M.null.items():
        out[f"e_null{k}"] = ((dx @ B) ** 2).sum(-1)
    return out


def comp_vectors(uv: dict[str, np.ndarray], kind: str, c: int) -> tuple[jax.Array, jax.Array]:
    return jnp.asarray(uv[f"{kind}.V"][:, c]), jnp.asarray(uv[f"{kind}.U"][c])


def parse(name: str) -> tuple[int, str, int]:
    li, kind, c = name.split(".")
    return int(li[1:]), kind, int(c[1:])


def start_point(li: int, kind: str) -> int:
    return 2 * li + (0 if kind in ("q", "k", "v", "o") else 1)


def fineweb(n_rows: int, _batch: int) -> tuple[np.ndarray, np.ndarray]:
    toks = np.load(FW / "tokens.npy")[:n_rows]
    target = np.concatenate([toks[:, 1:], np.full((len(toks), 1), -1)], 1)
    return toks, target


def base_pass(
    M: TextModel, toks: np.ndarray, batch: int, points: set[int]
) -> dict[int, list[jax.Array]]:
    caps = {t: [] for t in points | {64}}
    for s in range(0, len(toks), batch):
        xf, cp, _ = M.run(M.embed_tokens(toks[s : s + batch]), 0, {}, capture=points)
        for t in points:
            caps[t].append(cp[t])
        caps[64].append(xf)
    return caps


def scan(shard: int, n_shards: int, n_rows: int = 64, batch: int = 32) -> None:
    """Zero-ablate every L30-31 component (alive on addsub or not) on n_rows fineweb rows; aggregate
    stats per component, with the -05 CI (>0.01) as the screen's split."""
    toks, target = fineweb(n_rows, batch)
    M = TextModel(toks.shape[1])
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    ci = {li: np.load(FW / f"ci_L{li}.npy", mmap_mode="r") for li in (30, 31)}
    names = [f"L{li}.{k}.c{c}" for li in (30, 31) for k in CS for c in range(CS[k])][
        shard::n_shards
    ]
    caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    log("base done", len(names), "components")
    keys = ["kl_total", "kl_direct", "kl_norm", "kl_resid", "log_beta", "H_b", "H_a", "flip", "ce_b", "ce_a", "dlog_rms",
            "e_total", "e_uniform", "dz_mean2", "dz_var"] + [f"e_null{k}" for k in NULL_K]  # fmt: skip
    # [:, 0] CI-active positions, [:, 1] inactive, [:, 2] position 0 (no BOS: the rows' attention sink)
    agg = {k: np.zeros((len(names), 3)) for k in keys}
    n = np.zeros((len(names), 3))
    first = np.zeros((batch, toks.shape[1]), bool)
    first[:, 0] = True
    ci_sum = np.zeros(len(names))
    t0 = time.time()
    for i, nm in enumerate(names):
        li, kind, c = parse(nm)
        V, U = comp_vectors(uv[li], kind, c)
        mods: Mods = {(li, kind): (V, U, jnp.zeros_like)}
        tp = start_point(li, kind)
        for bi, s in enumerate(range(0, n_rows, batch)):
            xa, _, _ = M.run(caps[tp][bi], tp, mods)
            xb = caps[64][bi]
            m = metrics(xb, xa, M.g, M.unembed, jnp.asarray(target[s : s + batch]))
            m |= write_stats(M, xb - xa)
            act = np.asarray(ci[li][s : s + batch, :, OFFSET[kind] + c], np.float32)
            ci_sum[i] += act[:, 1:].sum()
            for j, sel in enumerate(((act > 0.01) & ~first, (act <= 0.01) & ~first, first)):
                n[i, j] += sel.sum()
                for k in keys:
                    agg[k][i, j] += float(np.asarray(m[k])[sel].sum())
        if i % 50 == 0:
            log(f"scan {shard}: {i}/{len(names)} ({time.time() - t0:.0f}s)")
    TEMP.mkdir(parents=True, exist_ok=True)
    savez(TEMP / f"scan_{shard}of{n_shards}.npz", names=np.array(names), n=n, ci_sum=ci_sum, **agg)


def deep(n_rows: int = 512, batch: int = 32) -> None:
    """Candidates and controls, per position: fineweb (zero and mean ablation, the inner activation and
    the -05 CI at each position, the write's per-neuron split for L31 MLP components) and the addsub pool
    at `=` (zero and mean ablation)."""
    toks, target = fineweb(n_rows, batch)
    M = TextModel(toks.shape[1])
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    ci = {li: np.load(FW / f"ci_L{li}.npy", mmap_mode="r") for li in (30, 31)}
    names = CANDIDATES + CONTROLS
    caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    out: dict[str, np.ndarray] = {}
    Wd = M.Wd31.astype(jnp.float32)
    for nm in names:
        li, kind, c = parse(nm)
        V, U = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        h_all = [M.inner(caps[tp][bi], li, kind, V[:, None])[..., 0] for bi in range(len(caps[64]))]
        mean_h = float(jnp.concatenate(h_all).mean())
        out[f"{nm}__inner"] = np.asarray(jnp.concatenate(h_all))
        out[f"{nm}__ci"] = np.asarray(ci[li][:n_rows, :, OFFSET[kind] + c], np.float32)
        for abl, f in (("zero", jnp.zeros_like), ("mean", lambda h, m=mean_h: jnp.full_like(h, m))):
            rec: dict[str, list[np.ndarray]] = {}
            share = jnp.zeros(Wd.shape[1])
            for bi, s in enumerate(range(0, n_rows, batch)):
                want = abl == "zero" and li == 31 and kind in ("gate", "up")
                xa, _, ha = M.run(caps[tp][bi], tp, {(li, kind): (V, U, f)}, want_h=want)
                xb = caps[64][bi]
                m = metrics(xb, xa, M.g, M.unembed, jnp.asarray(target[s : s + batch]))
                m |= write_stats(M, xb - xa)
                for k, v in m.items():
                    rec.setdefault(k, []).append(np.asarray(v))
                if want:
                    _, _, hb = M.run(caps[63][bi], 63, {}, want_h=True)
                    dx = xb - xa
                    # share_n = sum_pos dh_n (W_down[:, n] . dx) / sum_pos |dx|^2
                    dx = dx[:, 1:]  # position 0 is the rows' attention sink (no BOS)
                    assert hb is not None and ha is not None
                    share = (
                        share
                        + jnp.einsum("btn,btn->n", (hb - ha)[:, 1:], dx @ Wd) / (dx * dx).sum()
                    )
            for k, v in rec.items():
                out[f"{nm}__{abl}__{k}"] = np.concatenate(v)
            if abl == "zero" and li == 31 and kind in ("gate", "up"):
                out[f"{nm}__neuron_share"] = np.asarray(share) / (n_rows // batch)
            log(
                nm,
                abl,
                " ".join(
                    f"{k} {np.mean(out[f'{nm}__{abl}__{k}']):.4f}"
                    for k in ("kl_total", "kl_direct", "kl_norm", "kl_resid", "H_b", "H_a")
                ),
            )
    TEMP.mkdir(parents=True, exist_ok=True)
    savez(TEMP / "deep_fineweb.npz", tokens=toks, target=target, eig=M.eig, **out)


def deep_addsub(names: list[str], batch: int = 1000) -> None:
    ix = np.load(DATASET / "index.npz")
    toks = ix["tokens"].astype(np.int64)
    num_ids, minus = number_token_ids(200)
    a, b, op = ix["a"].astype(int), ix["b"].astype(int), ix["op"].astype(int)
    res = np.where(op == 0, a + b, a - b)
    ans = np.where(res >= 0, num_ids[np.clip(res, 0, 200)], minus)
    target = np.full(toks.shape, -1)
    target[:, -1] = ans
    M = TextModel(toks.shape[1])
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    out: dict[str, np.ndarray] = {}
    for nm in names:
        li, kind, c = parse(nm)
        V, U = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        h = jnp.concatenate(
            [M.inner(caps[tp][bi], li, kind, V[:, None])[..., 0] for bi in range(len(caps[64]))]
        )
        out[f"{nm}__inner"] = np.asarray(h)
        mean_eq = float(h[:, -1].mean())
        for abl, f in (
            ("zero", jnp.zeros_like),
            ("mean", lambda x, m=mean_eq: jnp.full_like(x, m)),
        ):
            rec: dict[str, list[np.ndarray]] = {}
            for bi, s in enumerate(range(0, len(toks), batch)):
                xa, _, _ = M.run(caps[tp][bi], tp, {(li, kind): (V, U, f)})
                xb = caps[64][bi]
                m = metrics(
                    xb[:, -1:], xa[:, -1:], M.g, M.unembed, jnp.asarray(target[s : s + batch, -1:])
                )
                m |= write_stats(M, (xb - xa)[:, -1:])
                for k, v in m.items():
                    rec.setdefault(k, []).append(np.asarray(v)[:, 0])
            for k, v in rec.items():
                out[f"{nm}__{abl}__{k}"] = np.concatenate(v)
            log(
                "addsub",
                nm,
                abl,
                " ".join(
                    f"{k} {np.mean(out[f'{nm}__{abl}__{k}']):.4f}"
                    for k in ("kl_total", "kl_direct", "kl_norm", "kl_resid", "H_b", "H_a")
                ),
            )
    savez(TEMP / "deep_addsub.npz", answer=ans, **out)


def const(m: float) -> Callable[[jax.Array], jax.Array]:
    return lambda x: jnp.full_like(x, m)


def neurons(n_rows: int = 256, batch: int = 32) -> None:
    """Stolfo et al.'s neuron-level view of Llama's L31 MLP: each neuron's output-weight norm, its share
    in the unembedding's bottom singular directions and along the uniform-logit direction; then the
    neurons that carry the candidates' writes (from `deep`) and the top neurons by norm x null share are
    mean-ablated on fineweb (total vs direct effect, temperature fit)."""
    toks, target = fineweb(n_rows, batch)
    M = TextModel(toks.shape[1])
    Wd = M.Wd31.astype(jnp.float32)  # (4096, 14336)
    Weff = M.Weff
    norm = jnp.linalg.norm(Wd, axis=0)
    stats = {"norm": np.asarray(norm), "uniform": np.asarray((M.uniform @ Wd) ** 2 / norm**2)}
    for k, B in M.null.items():
        stats[f"null{k}"] = np.asarray(((B.T @ Wd) ** 2).sum(0) / norm**2)
    # logit change per unit activation, dz = Weff @ Wd (V, 14336), via its moments over the vocabulary
    nv = Weff.shape[0]
    dz_mean = Weff.mean(0) @ Wd
    dz_sq = ((Weff.T @ Weff) @ Wd * Wd).sum(0) / nv
    stats["dz_uniform_share"] = np.asarray(dz_mean**2 / dz_sq)
    stats["logit_var"] = np.asarray(dz_sq - dz_mean**2) / np.asarray(norm) ** 2
    deepf = np.load(TEMP / "deep_fineweb.npz")
    picked: dict[str, list[int]] = {}
    for nm in CANDIDATES:
        if f"{nm}__neuron_share" in deepf.files:
            picked[nm] = np.argsort(-deepf[f"{nm}__neuron_share"])[:5].tolist()
    score = stats["norm"] * stats["null40"]
    picked["stolfo_top"] = np.argsort(-score)[:10].tolist()
    picked["uniform_top"] = np.argsort(-(stats["norm"] ** 2 * stats["dz_uniform_share"]))[
        :10
    ].tolist()
    caps = base_pass(M, toks, batch, {63})
    # mean activation of every neuron
    hs = []
    for bi in range(len(caps[64])):
        _, _, h = M.run(caps[63][bi], 63, {}, want_h=True)
        hs.append(h)
    h = jnp.concatenate(hs)
    hmean, hstd = h[:, 1:].mean((0, 1)), h[:, 1:].std((0, 1))
    stats["h_mean"], stats["h_std"] = np.asarray(hmean), np.asarray(hstd)
    del h, hs
    # the addsub pool at `=`: 2000 prompts
    ix = np.load(DATASET / "index.npz")
    rows = np.sort(np.random.default_rng(0).choice(len(ix["tokens"]), 2000, replace=False))
    atoks = ix["tokens"][rows].astype(np.int64)
    acaps = base_pass(M, atoks, 1000, {63})

    def eq_h(bi: int) -> jax.Array:
        h_eq = M.run(acaps[63][bi], 63, {}, want_h=True)[2]
        assert h_eq is not None
        return h_eq[:, -1]

    ah = jnp.concatenate([eq_h(bi) for bi in range(len(acaps[64]))])
    ahmean = ah.mean(0)
    stats["h_mean_eq"], stats["h_std_eq"] = np.asarray(ahmean), np.asarray(ah.std(0))
    del ah
    atgt = jnp.full((1000, 1), -1)
    results = {}
    for n_ in sorted({n for v in picked.values() for n in v}):
        e = jnp.zeros(Wd.shape[1]).at[n_].set(1.0)
        # mean-ablate neuron n_: its activation is replaced by the mean -> write change (mean - h) W_down[:, n]
        # fineweb positions 1..63 (0 is the rows' attention sink), mean over fineweb
        mods = {(31, "down"): (e, Wd[:, n_], const(float(hmean[n_])))}
        rec: dict[str, list[float]] = {}
        for bi, s in enumerate(range(0, n_rows, batch)):
            xa, _, _ = M.run(caps[63][bi], 63, mods)
            m = metrics(
                caps[64][bi][:, 1:],
                xa[:, 1:],
                M.g,
                M.unembed,
                jnp.asarray(target[s : s + batch, 1:]),
            )
            for k, v in m.items():
                rec.setdefault(k, []).append(float(np.asarray(v).mean()))
        # addsub `=`: zero and mean (over `=`) ablation
        for abl, val in (("eq_zero", 0.0), ("eq_mean", float(ahmean[n_]))):
            md = {(31, "down"): (e, Wd[:, n_], const(val))}
            for bi in range(len(acaps[64])):
                xa, _, _ = M.run(acaps[63][bi], 63, md)
                m = metrics(acaps[64][bi][:, -1:], xa[:, -1:], M.g, M.unembed, atgt)
                for k in ("kl_total", "kl_direct", "kl_resid", "H_a", "H_b"):
                    rec.setdefault(f"{abl}_{k}", []).append(float(np.asarray(m[k]).mean()))
        results[int(n_)] = {k: float(np.mean(v)) for k, v in rec.items()} | {
            "norm": float(stats["norm"][n_]), "null40": float(stats["null40"][n_]),
            "uniform": float(stats["uniform"][n_]), "dz_uniform_share": float(stats["dz_uniform_share"][n_]),
            "h_mean": float(hmean[n_]), "h_std": float(hstd[n_]),
            "h_mean_eq": float(ahmean[n_]), "h_std_eq": float(stats["h_std_eq"][n_])}  # fmt: skip
        log("neuron", n_, {k: round(v, 4) for k, v in results[int(n_)].items()})
    savez(TEMP / "neurons.npz", eig=M.eig, **stats)
    (TEMP / "neurons.json").write_text(json.dumps({"picked": picked, "results": results}, indent=1))


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0)


def dose(batch: int = 1000) -> None:
    """Addsub at `=`: each candidate / control's inner activation scaled by alpha (1 = the model). A
    temperature component traces a one-parameter family: log beta* moves monotonically with alpha and the
    residual after the best temperature stays small. Also the mean write dx = x(alpha=1) - x(alpha=0) at
    `=` and, on fineweb, at the positions the -05 CI flags, plus fineweb unigram counts (token-frequency
    neurons)."""
    ix = np.load(DATASET / "index.npz")
    toks = ix["tokens"].astype(np.int64)
    M = Mf = TextModel(64)  # rope tables sliced to each input's length
    uv = {li: dict(np.load(FW / f"uv_L{li}.npz")) for li in (30, 31)}
    names = CANDIDATES + CONTROLS
    caps = base_pass(M, toks, batch, {60, 61, 62, 63})
    out: dict[str, np.ndarray] = {}
    tgt = jnp.full((batch, 1), -1)
    for nm in names:
        li, kind, c = parse(nm)
        V, U = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        for al in ALPHAS:
            rec: dict[str, list[np.ndarray]] = {}
            dxs = jnp.zeros(4096)
            for bi in range(len(caps[64])):
                xa, _, _ = M.run(caps[tp][bi], tp, {(li, kind): (V, U, lambda h, a=al: a * h)})
                xb = caps[64][bi]
                m = metrics(xb[:, -1:], xa[:, -1:], M.g, M.unembed, tgt)
                for k in ("kl_total", "kl_resid", "log_beta", "H_a", "H_b", "dlog_rms"):
                    rec.setdefault(k, []).append(np.asarray(m[k])[:, 0])
                if al == 0.0:
                    dxs = dxs + (xb - xa)[:, -1].sum(0)
            for k, v in rec.items():
                out[f"{nm}__a{al}__{k}"] = np.concatenate(v)
            if al == 0.0:
                out[f"{nm}__dx_addsub"] = np.asarray(dxs / len(toks))
            log(
                "dose",
                nm,
                al,
                " ".join(f"{k} {np.mean(out[f'{nm}__a{al}__{k}']):+.4f}" for k in rec),
            )
    del caps
    # fineweb: mean write at CI-flagged positions
    ftoks, _ = fineweb(512, 32)
    ci = {li: np.load(FW / f"ci_L{li}.npy", mmap_mode="r") for li in (30, 31)}
    fcaps = base_pass(Mf, ftoks, 32, {60, 61, 62, 63})
    for nm in names:
        li, kind, c = parse(nm)
        vv, uu = comp_vectors(uv[li], kind, c)
        tp = start_point(li, kind)
        acc, cnt = jnp.zeros(4096), 0
        for bi, s in enumerate(range(0, 512, 32)):
            xa, _, _ = Mf.run(fcaps[tp][bi], tp, {(li, kind): (vv, uu, jnp.zeros_like)})
            act = jnp.asarray(
                np.asarray(ci[li][s : s + 32, :, OFFSET[kind] + c], np.float32) > 0.01
            )
            acc = acc + ((fcaps[64][bi] - xa) * act[..., None]).sum((0, 1))
            cnt += int(act.sum())
        out[f"{nm}__dx_fineweb_active"] = np.asarray(acc / max(cnt, 1))
    import pyarrow.parquet as pq

    f = pq.ParquetFile(
        sorted((FW.parents[4] / "datasets/fineweb_llama_tok_64").glob("shard_*.parquet"))[0]
    )
    counts = np.zeros(Mf.unembed.shape[0], np.int64)
    for rg in range(min(f.num_row_groups, 40)):
        t = f.read_row_group(rg, columns=["input_ids"])
        counts += np.bincount(
            np.concatenate([np.asarray(x) for x in t["input_ids"].to_pylist()]),
            minlength=len(counts),
        )
    out["unigram_counts"] = counts
    out["Weff_dx"] = np.stack(
        [np.asarray(Mf.Weff @ jnp.asarray(out[f"{nm}__dx_addsub"])) for nm in names]
    )
    out["Weff_dx_fw"] = np.stack(
        [np.asarray(Mf.Weff @ jnp.asarray(out[f"{nm}__dx_fineweb_active"])) for nm in names]
    )
    savez(TEMP / "dose.npz", names=np.array(names), alphas=np.array(ALPHAS), **out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        scan(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {
            "deep": deep,
            "addsub": lambda: deep_addsub(CANDIDATES + CONTROLS),
            "neurons": neurons,
            "dose": dose,
        }[cmd]()
