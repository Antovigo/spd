"""Synthetic stream and readers on the (a, b) grid, with known quantities, to test the atlas's stage A.

    python -m param_decomp.arith_repr.isa.synth [seed] [noise] [overlap] [leak]
        # runs `atlas.analyse_point` and prints, for each true quantity, the best-matching
        # quantity found: the overlap of its stream subspace with the true one, and its dimension

Quantities (value -> embedding): circles of a mod 50, b mod 50, a mod 10, b mod 10 and (a + b) mod
10; a binary square wave of b mod 20; a simplex of a mod 5 (five corners); the magnitude lin(a); a
product a-square x b-square of period 20 (a checkerboard); and two messy lookups of a and of b
(random 100-value tables in 12 dims each). Each is written into its own random subspace of a
D-dim stream, some pairs made to overlap, plus isotropic noise.

Readers: most read one quantity at a random phase (circles) or at one corner (simplex); a few mix
a mod 50 with b mod 50 (like L18 up.c61); some messy readers read random directions."""

import sys
from dataclasses import dataclass

import numpy as np

D = 512


@dataclass
class Synth:
    H: np.ndarray  # (N, n) reader inner activations
    X: np.ndarray  # (N, D) stream
    a: np.ndarray
    b: np.ndarray
    names: list[str]  # quantity names
    U: list[np.ndarray]  # (D, d_q) orthonormal embedding bases
    reader_q: list[tuple[int, ...]]  # quantities each reader reads (empty: messy)


def square(v: np.ndarray, period: int, onset: int) -> np.ndarray:
    return np.where(((v - onset) % period) < period // 2, 1.0, -1.0)


def features(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> list[tuple[str, np.ndarray, float]]:
    circ = lambda v, T: np.stack([np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)], 1)  # noqa: E731
    onehot = np.eye(5)[a % 5] - 0.2
    simplex = onehot @ np.linalg.qr(np.eye(5) - 0.2)[0][:, :4]  # 4-dim coordinates of 5 corners
    look_a = rng.standard_normal((101, 12))[a]
    look_b = rng.standard_normal((101, 12))[b]
    return [
        ("a mod 50", circ(a, 50), 3.0),
        ("b mod 50", circ(b, 50), 3.0),
        ("a mod 10", circ(a, 10), 2.0),
        ("b mod 10", circ(b, 10), 2.0),
        ("a+b mod 10", circ(a + b, 10), 1.5),
        ("b mod 20 square", square(b, 20, 5)[:, None], 1.5),
        ("a mod 5 simplex", simplex, 1.5),
        ("lin(a)", ((a - 50.5) / 28.87)[:, None], 1.0),
        ("a x b checkerboard", (square(a, 20, 5) * square(b, 20, 5))[:, None], 1.0),
        ("lookup(a)", look_a / np.sqrt(12), 0.8),
        ("lookup(b)", look_b / np.sqrt(12), 0.8),
    ]


def make(seed: int = 0, noise: float = 0.3, overlap: float = 0.5, leak: float = 0.05) -> Synth:
    rng = np.random.default_rng(seed)
    a = np.repeat(np.arange(1, 101), 100)
    b = np.tile(np.arange(1, 101), 100)
    feats = features(a, b, rng)
    names = [f[0] for f in feats]
    U = [np.linalg.qr(rng.standard_normal((D, f[1].shape[1])))[0] for f in feats]
    # make b mod 10 partly share a mod 10's directions, and the checkerboard lean on b's square wave
    for i, j in ((3, 2), (8, 5)):
        M = U[i] + overlap * U[j][:, : U[i].shape[1]]
        U[i] = np.linalg.qr(M)[0]
    signal = sum(amp * f @ Uq.T for (_, f, amp), Uq in zip(feats, U, strict=True))
    X = np.asarray(signal) + noise * rng.standard_normal((len(a), D))

    readers: list[np.ndarray] = []
    reader_q: list[tuple[int, ...]] = []
    # readers are filters: the dual basis of all embeddings, so a reader of q sees q only
    Uall = np.concatenate(U, 1)
    dual = Uall @ np.linalg.inv(Uall.T @ Uall)
    starts = np.cumsum([0] + [u.shape[1] for u in U])

    def read(q: int) -> np.ndarray:
        d = U[q].shape[1]
        if names[q].endswith("simplex"):
            corner = np.eye(5)[rng.integers(5)] - 0.2
            w = corner @ np.linalg.qr(np.eye(5) - 0.2)[0][:, :4]
        else:
            w = rng.standard_normal(d)
        f = dual[:, starts[q] : starts[q] + d] @ (w / np.linalg.norm(w))
        return f / np.linalg.norm(f)

    counts = {0: 14, 1: 14, 2: 3, 3: 3, 4: 8, 5: 6, 6: 10, 7: 5, 8: 6}
    for q, k in counts.items():
        for _ in range(k):
            readers.append(read(q))
            reader_q.append((q,))
    for _ in range(8):  # a mod 50 + b mod 50, like L18 up.c61
        w = rng.uniform(0.3, 0.7)
        readers.append(np.sqrt(w) * read(0) + np.sqrt(1 - w) * read(1))
        reader_q.append((0, 1))
    for _ in range(14):  # plaids: a mod 10 + b mod 10, the same period in both operands
        w = rng.uniform(0.3, 0.7)
        readers.append(np.sqrt(w) * read(2) + np.sqrt(1 - w) * read(3))
        reader_q.append((2, 3))
    for _ in range(40):  # messy: weak random reads of the two lookups plus noise
        q = 9 + int(rng.integers(2))
        v = (
            U[q] @ rng.standard_normal(U[q].shape[1])
            + 0.5 * rng.standard_normal(D) / np.sqrt(D) * 4
        )
        readers.append(0.15 * v / np.linalg.norm(v))
        reader_q.append(())
    V = np.stack(readers, 1) + leak * rng.standard_normal((D, len(readers))) / np.sqrt(D)
    return Synth(X @ V, X, a, b, names, U, reader_q)


def recovery(s: Synth, Zs: list[np.ndarray]) -> list[tuple[str, int, float, int]]:
    """For each true quantity (lookups excluded): (name, true dim, best overlap, found dim). The
    overlap is the share of the true subspace inside the stream patterns E[(x - mean) z] of the
    best-matching quantity found (1 = identical)."""
    Xc = s.X - s.X.mean(0)
    bases = [np.linalg.qr(Xc.T @ z / len(z))[0] for z in Zs]
    out = []
    for q, name in enumerate(s.names):
        if name.startswith("lookup"):
            continue
        best = (0.0, 0)
        for z, Q in zip(Zs, bases, strict=True):
            c = np.linalg.svd(s.U[q].T @ Q, compute_uv=False)
            best = max(best, (float((c**2).sum() / s.U[q].shape[1]), z.shape[1]))
        out.append((name, int(s.U[q].shape[1]), round(best[0], 3), best[1]))
    return out


if __name__ == "__main__":
    from param_decomp.arith_repr.isa.atlas import analyse_point

    defaults = [0.0, 0.3, 0.5, 0.05]
    args = [float(v) for v in sys.argv[1:]] + defaults[len(sys.argv) - 1 :]
    seed, noise, overlap, leak = int(args[0]), args[1], args[2], args[3]
    s = make(seed, noise=noise, overlap=overlap, leak=leak)
    # the synthetic stream has no norm: a constant x makes the RMS factor 1
    res = analyse_point(
        s.H, np.ones((len(s.H), 4), np.float32), [f"r{i}" for i in range(s.H.shape[1])]
    )
    for name, d, ov, k in recovery(s, [q["z"] for q in res["quantities"]]):
        print(f"{name:20s} ({d} dims): overlap {ov:.2f}, found with k = {k}")
