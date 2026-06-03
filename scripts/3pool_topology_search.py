"""Search 3-pool topologies for the throughput-optimal rank allocation.

Grounding model (per-sample compute, calibrated from a baseline trace):

  compute_ci   ≈ k_ci   * (B / n_ci)              # CI fn fwd/bwd, DP over batch
  compute_ppgd ≈ k_ppgd * (B / n_ppgd)            # full V/U inner loop, DP over batch
  compute_lw   ≈ k_lw_total * B / L               # all sites split over L = n_blocks*n_per_block
  step         ≈ max(compute_ci, compute_lw, compute_ppgd) + overhead

`throughput = B / step` (samples / ms) is the objective — comparable across B.

CAVEATS (why this is a *screen*, not a verdict):
  * Assumes per-rank compute scales LINEARLY with local batch. EMPIRICALLY FALSE
    for LW: the per-(site·sample) cost fell ~12× from bl_lw=4 to bl_lw=64 (large
    fixed per-site overhead — the serial per-site recon loop). So the LW-linear term
    badly over-credits adding LW ranks (predicts a linear speedup that won't
    materialize — cf. the thin-block result: +8.5%, not the modelled multiple).
    Calibrated here AT the big512 operating point (bl_lw=64); trust it only near
    that regime. Always validate the winner with a real run.
  * Assumes `overhead` (non-overlapped comm/idle/sync) is additive & constant.
    The baseline shows ~170ms of it; it may not hold as the topology changes.
  * Memory is flagged crudely (CI activations dominate; batch_local_ci>baseline
    risks OOM without grad checkpointing — notes task #4).
"""

from dataclasses import dataclass
from itertools import product

N_SITES = 96  # 48 layers × {q_proj, k_proj}
BUDGET = 224  # total ranks to search over (big512 production scale: 28 nodes × 8)

# ── Calibration (2026-06-03, current code: vendored ComponentGPT2, LW+CI ──
# torch.compile, activation checkpointing).
# Per-pool COMPUTE from the rebalance-6site trace (job 38431, 112 ranks
# LW64/CI16/PPGD32, B=256), analyzed by scripts/analyze_3pool_trace.py. Its per-rank
# batch_local (lw 64 / ci 16 / ppgd 8) and per-rank work are IDENTICAL to big512
# production (p-b6505e9c) — big512 doubled both B and every pool's DDP — so the
# per-rank compute carries over exactly. STEP WALL is from big512 production itself
# (~2358 ms at 224 ranks) so OVERHEAD reflects the 224-rank cross-pool cost (the
# rebalance trace's own 112-rank wall is ~2138; overhead grows with rank count).
# Recompute COMPUTE with: scripts/analyze_3pool_trace.py <trace_dir>  → plug means.
CALIB_B = 256  # batch of the compute-calibration trace (wall is from big512, B=512)
CALIB = {
    "ci": (579.1, 16),  # (compute_ms, batch_local)
    "ppgd": (1140.2, 8),
    "lw": (1243.5, 64),  # at sites_per_block=6
}
CALIB_LW_SITES_PER_BLOCK = 6
CALIB_STEP_WALL = 2358.0  # big512 production step_ms (p-b6505e9c, 224 ranks)

K_CI = CALIB["ci"][0] / CALIB["ci"][1]  # ms / sample
K_PPGD = CALIB["ppgd"][0] / CALIB["ppgd"][1]
K_LW_TOTAL = (CALIB["lw"][0] / (CALIB_LW_SITES_PER_BLOCK * CALIB["lw"][1])) * N_SITES
OVERHEAD = CALIB_STEP_WALL - max(CALIB["ci"][0], CALIB["ppgd"][0], CALIB["lw"][0])

CALIB_BATCH_LOCAL_CI = CALIB["ci"][1]  # memory baseline


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _div_ok(a: int, b: int, *, relaxed: bool) -> bool:
    """current: a | b.  relaxed: a | b OR b | a."""
    return (b % a == 0) or (relaxed and a % b == 0)


@dataclass(frozen=True)
class Topo:
    n_ci: int
    n_ppgd: int
    n_blocks: int
    n_per_block: int
    batch: int

    @property
    def n_lw(self) -> int:
        return self.n_blocks * self.n_per_block

    @property
    def sites_per_block(self) -> int:
        assert N_SITES % self.n_blocks == 0
        return N_SITES // self.n_blocks

    @property
    def compute_ci(self) -> float:
        return K_CI * self.batch / self.n_ci

    @property
    def compute_ppgd(self) -> float:
        return K_PPGD * self.batch / self.n_ppgd

    @property
    def compute_lw(self) -> float:
        return K_LW_TOTAL * self.batch / self.n_lw

    @property
    def step_ms(self) -> float:
        return max(self.compute_ci, self.compute_ppgd, self.compute_lw) + OVERHEAD

    @property
    def throughput(self) -> float:  # samples / ms
        return self.batch / self.step_ms

    @property
    def bottleneck(self) -> str:
        return max(
            (("ci", self.compute_ci), ("ppgd", self.compute_ppgd), ("lw", self.compute_lw)),
            key=lambda kv: kv[1],
        )[0]

    @property
    def batch_local_ci(self) -> int:
        return self.batch // self.n_ci

    @property
    def ci_oom_risk(self) -> bool:
        return self.batch_local_ci > CALIB_BATCH_LOCAL_CI

    def needs_relaxation(self) -> bool:
        cur = _div_ok(self.n_ci, self.n_per_block, relaxed=False) and _div_ok(
            self.n_ci, self.n_ppgd, relaxed=False
        )
        return not cur


def enumerate_topos(batches: list[int], *, relaxed: bool) -> list[Topo]:
    out = []
    for B in batches:
        bdivs = _divisors(B)  # n_ci, n_ppgd, n_per_block must each divide B
        for n_blocks in _divisors(N_SITES):
            for n_ci, n_ppgd, n_per_block in product(bdivs, bdivs, bdivs):
                n_lw = n_blocks * n_per_block
                if n_ci + n_ppgd + n_lw != BUDGET:
                    continue
                if not _div_ok(n_ci, n_per_block, relaxed=relaxed):
                    continue
                if not _div_ok(n_ci, n_ppgd, relaxed=relaxed):
                    continue
                out.append(Topo(n_ci, n_ppgd, n_blocks, n_per_block, B))
    return out


def fmt(t: Topo) -> str:
    flags = []
    if t.needs_relaxation():
        flags.append("RELAX")
    if t.ci_oom_risk:
        flags.append(f"CI-OOM?(bl={t.batch_local_ci})")
    return (
        f"B={t.batch:<3} ci={t.n_ci:<2} ppgd={t.n_ppgd:<2} "
        f"lw={t.n_blocks}x{t.n_per_block}={t.n_lw:<3} | "
        f"step~{t.step_ms:6.1f}ms thru={t.throughput:6.4f} "
        f"(ci {t.compute_ci:5.0f} | lw {t.compute_lw:5.0f} | ppgd {t.compute_ppgd:5.0f}, "
        f"bottleneck={t.bottleneck:<4}) {' '.join(flags)}"
    )


def main() -> None:
    baseline = Topo(n_ci=32, n_ppgd=64, n_blocks=16, n_per_block=8, batch=512)
    print(
        f"calibration: K_CI={K_CI:.1f} K_PPGD={K_PPGD:.1f} K_LW_TOTAL={K_LW_TOTAL:.0f} "
        f"OVERHEAD={OVERHEAD:.0f}ms (all ms, per-sample where applicable)"
    )
    print(f"\nBASELINE (current production): {fmt(baseline)}")
    print(f"  baseline throughput = {baseline.throughput:.4f} samples/ms\n")

    for batches, label in [
        ([512], "B=512 (current production batch)"),
        ([256, 512, 768, 1024], "B free (perf-only; changes opt dynamics)"),
    ]:
        print("=" * 100)
        print(f"SEARCH: {label}")
        print("=" * 100)
        cur = sorted(enumerate_topos(batches, relaxed=False), key=lambda t: -t.throughput)
        rel = sorted(enumerate_topos(batches, relaxed=True), key=lambda t: -t.throughput)
        # relaxed-only = configs valid only under relaxation
        rel_only = [t for t in rel if t.needs_relaxation()]

        print("\n-- best under CURRENT constraints --")
        for t in cur[:6]:
            print("  " + fmt(t))
        print("\n-- best that REQUIRE the relaxation (relaxed-only) --")
        if rel_only:
            for t in sorted(rel_only, key=lambda t: -t.throughput)[:6]:
                print("  " + fmt(t))
        else:
            print("  (none)")

        best_cur = cur[0].throughput if cur else 0.0
        best_rel = rel[0].throughput if rel else 0.0
        gain_cur = 100 * (best_cur / baseline.throughput - 1)
        gain_rel = 100 * (best_rel / baseline.throughput - 1)
        verdict = (
            "RELAXATION HELPS"
            if best_rel > best_cur * 1.005
            else "relaxation does NOT beat current-constraint best"
        )
        print(f"\n  best CURRENT: {best_cur:.4f} ({gain_cur:+.1f}% vs baseline)")
        print(f"  best RELAXED: {best_rel:.4f} ({gain_rel:+.1f}% vs baseline)  →  {verdict}\n")


if __name__ == "__main__":
    main()
