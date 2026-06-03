"""Extract per-pool Gantt data from 3-pool torch.profiler traces → dashboard JSON.

Produces, for one representative ProfilerStep per pool: per-kernel-CATEGORY totals
(matmul / attention / reduction / elementwise / memory / other / nccl), a binned
step-relative GPU-occupancy timeline (one fraction per category per bin), and the CPU-side
NCCL time by op kind (recv = blocking wait on another pool). Categories come from the GPU
kernel names in the trace; nccl is mostly cross-pool wait.

Usage: python scripts/extract_gantt_json.py <trace_dir> <out_json>
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_3pool_trace import (  # noqa: E402
    _is_nccl,
    _nccl_op_kind,
    clip,
    merged_length,
)

NBINS = 240
# Stack order (bottom→top): productive GEMM first, comm/wait last.
CATEGORIES = ["matmul", "attention", "reduction", "elementwise", "memory", "other", "nccl"]


def _category(name: str, cat: str) -> str:
    """Bucket a GPU kernel by its name (cuBLAS/cutlass GEMM, cuDNN flash-attn, Triton
    fused norm-reductions vs pointwise, copies/memset, NCCL)."""
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memory"
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "sdpa" in n or "flash" in n or "attention" in n:
        return "attention"
    if any(g in n for g in ("gemm", "nvjet", "cutlass", "cublas", "wgrad", "dgrad")):
        return "matmul"
    if "triton_red" in n or "reduce" in n or "softmax" in n or "norm" in n:
        return "reduction"
    if "triton_poi" in n or "elementwise" in n or "pointwise" in n:
        return "elementwise"
    if "copy" in n or "memcpy" in n or "memset" in n:
        return "memory"
    return "other"


def extract_pool(path: Path) -> dict:  # raw trace JSON → bare dict (matches analyze_3pool_trace)
    d = json.loads(path.read_text())
    ev = d["traceEvents"]
    rank = d["distributedInfo"]["rank"]
    pool = path.stem.replace("trace_", "").split("_rank")[0]

    steps = [e for e in ev if e.get("ph") == "X" and "ProfilerStep" in str(e.get("name", ""))]
    by_name: dict[str, dict] = {}
    for e in steps:
        if e["name"] not in by_name or e["dur"] > by_name[e["name"]]["dur"]:
            by_name[e["name"]] = e
    steps_meta = sorted(by_name.values(), key=lambda e: e["ts"])
    assert steps_meta, f"no ProfilerStep in {path}"
    # representative = median-wall step (steady state)
    sm = sorted(steps_meta, key=lambda e: e["dur"])[len(steps_meta) // 2]
    w0, w1 = sm["ts"], sm["ts"] + sm["dur"]
    wall = sm["dur"]

    ivs: dict[str, list[tuple[float, float]]] = {c: [] for c in CATEGORIES}
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        c = _category(str(e.get("name", "")), str(e.get("cat")))
        ivs[c].append((e["ts"], e["ts"] + e.get("dur", 0.0)))
    clipped = {c: clip(ivs[c], (w0, w1)) for c in CATEGORIES}

    binw = wall / NBINS
    bins = []
    for i in range(NBINS):
        b0, b1 = w0 + i * binw, w0 + (i + 1) * binw
        row = [
            round(min(merged_length(clip(clipped[c], (b0, b1))) / binw if binw else 0.0, 1.0), 3)
            for c in CATEGORIES
        ]
        bins.append(row)

    op_time: dict[str, float] = {}
    for e in ev:
        if (
            e.get("cat") == "user_annotation"
            and _is_nccl(str(e.get("name", "")))
            and w0 <= e["ts"] < w1
        ):
            k = _nccl_op_kind(str(e["name"]))
            op_time[k] = op_time.get(k, 0.0) + e.get("dur", 0.0)

    by_category_ms = {c: round(merged_length(clipped[c]) / 1000, 1) for c in CATEGORIES}
    busy_ms = merged_length([iv for c in CATEGORIES for iv in clipped[c]]) / 1000
    wall_ms = wall / 1000
    return {
        "pool": pool,
        "rank": rank,
        "wall_ms": round(wall_ms, 1),
        "by_category_ms": by_category_ms,
        "idle_ms": round(wall_ms - busy_ms, 1),
        "idle_pct": round(100 * (wall_ms - busy_ms) / wall_ms, 1),
        "nccl_by_op_ms": {
            k: round(v / 1000, 1) for k, v in sorted(op_time.items(), key=lambda x: -x[1])
        },
        "bins": bins,
    }


def main() -> None:
    trace_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    paths = sorted(trace_dir.glob("trace_*.json"))
    assert paths, f"no traces in {trace_dir}"
    order = {"ci": 0, "layerwise": 1, "ppgd": 2}
    pools = []
    for p in paths:
        print(f"extracting {p.name} ({p.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        pools.append(extract_pool(p))
    pools.sort(key=lambda x: order.get(x["pool"], 9))
    payload = {
        "source": f"{trace_dir.name} · representative steady step",
        "nbins": NBINS,
        "categories": CATEGORIES,
        "step_ms": round(max(p["wall_ms"] for p in pools), 1),
        "pools": pools,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out}")
    for p in pools:
        cats = "  ".join(f"{c} {p['by_category_ms'][c]:.0f}" for c in CATEGORIES)
        print(
            f"  {p['pool']:>9} rank{p['rank']}: wall {p['wall_ms']:.0f}ms  idle {p['idle_pct']}%  | {cats}"
        )


if __name__ == "__main__":
    main()
