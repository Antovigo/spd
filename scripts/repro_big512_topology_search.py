"""Reproduce the 3-pool topology screen for the big512 production regime.

This is a config-specific RECORD, not a generic tool — see `scripts/topology_search.py`
for the model and `docs/3pool_topology_calibration_2026-06-03.md` for the findings.

Calibration provenance (2026-06-03, current code: vendored ComponentGPT2, LW+CI
torch.compile, activation checkpointing):
  * per-pool COMPUTE from the rebalance-6site torch.profiler trace (job 38431,
    112 ranks LW64/CI16/PPGD32, B=256) via `scripts/analyze_3pool_trace.py`. Its
    per-rank batch_local (lw 64 / ci 16 / ppgd 8) is IDENTICAL to big512 production,
    so the per-rank compute carries over exactly.
  * step WALL from big512 production itself (p-b6505e9c, 224 ranks, ~2358 ms) so
    OVERHEAD reflects the 224-rank cross-pool cost (the 112-rank trace's own wall is
    ~2138 ms; overhead grows with rank count).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.topology_search import Calibration, Topo, report  # noqa: E402

CALIBRATION = Calibration.from_measurements(
    n_sites=96,
    ci=(579.1, 16),  # (compute_ms, batch_local)
    ppgd=(1140.2, 8),
    lw=(1243.5, 64),
    lw_sites_per_block=6,
    step_wall_ms=2358.0,  # big512 production step_ms @ 224 ranks
)
BIG512 = Topo(n_ci=32, n_ppgd=64, n_blocks=16, n_per_block=8, batch=512)


if __name__ == "__main__":
    report(
        CALIBRATION,
        budget=224,
        baseline=BIG512,
        batch_groups=[
            ([512], "B=512 (current production batch)"),
            ([256, 512, 768, 1024], "B free (perf-only; changes opt dynamics)"),
        ],
    )
