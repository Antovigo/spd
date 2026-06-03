"""Extract per-pool Gantt data from 3-pool torch.profiler traces → dashboard JSON.

For ONE step shared across all pools, each lane anchored at its own step-start (the per-step
barrier syncs the pools): per-kernel-CATEGORY occupancy bins (matmul / attention / reduction /
elementwise / memory / other / nccl), per-category ms totals, and **cross-pool comm edges** —
paired send/recv between the profiled ranks, each placed at its time-since-that-pool's-barrier.

Alignment is **barrier-relative, not absolute**: the per-rank trace clocks are skewed (they sit
on different nodes), so raw timestamps can't share an axis — `skew_ms` reports the median
send→recv delta per pool pair as evidence (negative = receiver clock behind sender). Treat
edge slope as indicative, not literal latency.

Note: these traces have no `with_stack` data (no call stacks); only kernel categories.

Usage: python scripts/extract_gantt_json.py <trace_dir> <out_json>
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_3pool_trace import clip, merged_length  # noqa: E402

NBINS = 240
CATEGORIES = ["matmul", "attention", "reduction", "elementwise", "memory", "other", "nccl"]


def _category(name: str, cat: str) -> str:
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


def _parse_nccl(name: str) -> tuple[str, int, int] | None:
    """`nccl:send 64->0` → (send, 64, 0); `nccl:recv 0<-64` → (recv, sender=64, receiver=0)."""
    if "nccl:" not in name:
        return None
    parts = name.split("nccl:", 1)[1].split()
    if parts[0] == "send" and len(parts) > 1 and "->" in parts[1]:
        s, r = parts[1].split("->")
        return ("send", int(s), int(r))
    if parts[0] == "recv" and len(parts) > 1 and "<-" in parts[1]:
        r, s = parts[1].split("<-")
        return ("recv", int(s), int(r))
    return None


def _step_windows(ev: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for e in ev:
        if e.get("ph") == "X" and "ProfilerStep" in str(e.get("name", "")):
            n, ts, dur = e["name"], e["ts"], e.get("dur", 0.0)
            if n not in out or dur > (out[n][1] - out[n][0]):
                out[n] = (ts, ts + dur)
    return out


def _category_intervals(ev: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    ivs: dict[str, list[tuple[float, float]]] = {c: [] for c in CATEGORIES}
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        c = _category(str(e.get("name", "")), str(e.get("cat")))
        ivs[c].append((e["ts"], e["ts"] + e.get("dur", 0.0)))
    return ivs


def _bins(ivs: dict[str, list[tuple[float, float]]], w0: float, span: float) -> list[list[float]]:
    binw = span / NBINS
    clipped = {c: clip(ivs[c], (w0, w0 + span)) for c in CATEGORIES}
    bins = []
    for i in range(NBINS):
        b0, b1 = w0 + i * binw, w0 + (i + 1) * binw
        bins.append(
            [
                round(
                    min(merged_length(clip(clipped[c], (b0, b1))) / binw if binw else 0.0, 1.0), 3
                )
                for c in CATEGORIES
            ]
        )
    return bins


def main() -> None:
    trace_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    paths = sorted(trace_dir.glob("trace_*.json"))
    assert paths, f"no traces in {trace_dir}"

    traces = []
    for p in paths:
        print(f"loading {p.name} ({p.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        d = json.loads(p.read_text())
        pool = p.stem.replace("trace_", "").split("_rank")[0]
        traces.append({"pool": pool, "rank": d["distributedInfo"]["rank"], "ev": d["traceEvents"]})
    rank_to_pool = {t["rank"]: t["pool"] for t in traces}

    windows = [_step_windows(t["ev"]) for t in traces]
    common = set.intersection(*[set(w) for w in windows])
    assert common, "no ProfilerStep common to all pools"
    step = max(common, key=lambda n: int(n.split("#")[1]))
    starts = {t["pool"]: w[step][0] for t, w in zip(traces, windows, strict=True)}
    span = max(w[step][1] - w[step][0] for w in windows)  # common relative duration

    order = {"ci": 0, "layerwise": 1, "ppgd": 2}
    pools = []
    for t in traces:
        s = starts[t["pool"]]
        ivs = _category_intervals(t["ev"])
        own = {c: clip(ivs[c], (s, s + span)) for c in CATEGORIES}
        busy = merged_length([iv for c in CATEGORIES for iv in own[c]]) / 1000
        pools.append(
            {
                "pool": t["pool"],
                "rank": t["rank"],
                "wall_ms": round(span / 1000, 1),
                "by_category_ms": {c: round(merged_length(own[c]) / 1000, 1) for c in CATEGORIES},
                "idle_ms": round(span / 1000 - busy, 1),
                "idle_pct": round(100 * (span / 1000 - busy) / (span / 1000), 1),
                "bins": _bins(ivs, s, span),
            }
        )
    pools.sort(key=lambda x: order.get(x["pool"], 9))

    # comm edges: pair sends (sender trace) with recvs (receiver trace), each relative to its
    # own pool's step-start. Collected within each pool's own [start, start+span] window.
    sends: dict[tuple[int, int], list[float]] = {}
    recvs: dict[tuple[int, int], list[float]] = {}
    for t in traces:
        s = starts[t["pool"]]
        for ev in t["ev"]:
            if ev.get("cat") != "user_annotation" or "nccl:" not in str(ev.get("name", "")):
                continue
            if not (s <= ev["ts"] <= s + span):
                continue
            parsed = _parse_nccl(str(ev["name"]))
            if parsed is None:
                continue
            kind, s_rank, r_rank = parsed
            rel = (ev["ts"] - s) / 1000
            (sends if kind == "send" else recvs).setdefault((s_rank, r_rank), []).append(rel)

    edges = []
    skew: dict[str, float] = {}
    for key in sorted(set(sends) & set(recvs)):
        s_rank, r_rank = key
        if s_rank not in rank_to_pool or r_rank not in rank_to_pool:
            continue
        sp, rp = rank_to_pool[s_rank], rank_to_pool[r_rank]
        if sp == rp:
            continue
        ss, rr = sorted(sends[key]), sorted(recvs[key])
        deltas = []
        for ms_s, ms_r in zip(ss, rr, strict=False):
            edges.append(
                {
                    "src_pool": sp,
                    "dst_pool": rp,
                    "send_ms": round(ms_s, 2),
                    "recv_ms": round(ms_r, 2),
                }
            )
            deltas.append(ms_r - ms_s)
        if deltas:
            skew[f"{sp}->{rp}"] = round(sorted(deltas)[len(deltas) // 2], 1)

    payload = {
        "source": f"{trace_dir.name} · {step} · barrier-relative (clocks skewed)",
        "nbins": NBINS,
        "categories": CATEGORIES,
        "step_ms": round(span / 1000, 1),
        "pools": pools,
        "edges": edges,
        "skew_ms": skew,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out}  ({step}, span {payload['step_ms']}ms, {len(edges)} comm edges)")
    print(f"  skew (median send→recv ms in barrier-relative frame): {skew}")
    for p in pools:
        cats = "  ".join(f"{c} {p['by_category_ms'][c]:.0f}" for c in CATEGORIES)
        print(f"  {p['pool']:>9} rank{p['rank']}: idle {p['idle_pct']}%  | {cats}")


if __name__ == "__main__":
    main()
