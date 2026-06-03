"""Attribute GPU kernels to their launching call stack, per pool × kernel category.

torch.profiler `with_stack` records the call stack as nested `python_function` (+ `cpu_op`)
frames on the launching CPU thread. A kernel links to its launch via `correlation`; the stack
is the chain of frames containing that launch ts on that thread. This aggregates kernel GPU
time by (category, trimmed stack) and emits the top stacks per category — i.e. *which code*
produces each pool's matmul / elementwise / reduction / … time.

Output JSON: {pool: {category: [{stack: [frames outer→inner], ms, count}, ...]}}.

Usage: python scripts/extract_stacks.py <trace_dir> <out_json>
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.extract_gantt_json import CATEGORIES, _category, _step_windows  # noqa: E402

TOP_PER_CAT = 6


def _trim(stack: list[str]) -> list[str]:
    """Drop boilerplate above the first app frame; keep from there to the innermost."""
    for i, f in enumerate(stack):
        if "param_decomp" in f:
            return [_short(f) for f in stack[i:]]
    return [_short(f) for f in stack[-6:]]  # no app frame → last few


def _short(frame: str) -> str:
    # "param_decomp_lab/three_pool/step_layerwise.py(302): _recon_one_forward" → keep as-is but
    # strip long site-packages prefixes for readability.
    if "site-packages/" in frame:
        frame = frame.split("site-packages/", 1)[1]
    return frame


def attribute_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    d = json.loads(path.read_text())
    ev = d["traceEvents"]
    wins = _step_windows(ev)
    step = max(wins, key=lambda n: int(n.split("#")[1]))
    w0, w1 = wins[step]

    # correlation -> (ts, tid) of the CPU-side launch. cuBLAS/cuDNN GEMM+attn launch via
    # cuda_driver (cuLaunchKernel); pointwise/reduction via cuda_runtime (cudaLaunchKernel).
    runtime_ts: dict[int, tuple[float, int]] = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                runtime_ts[corr] = (e["ts"], e["tid"])

    frames_by_tid: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for e in ev:
        if e.get("cat") in ("python_function", "cpu_op") and e.get("ph") == "X":
            frames_by_tid[e["tid"]].append((e["ts"], e["ts"] + e.get("dur", 0.0), str(e["name"])))

    # group kernel launch queries by tid
    queries: dict[int, list[tuple[float, str, float]]] = defaultdict(list)  # tid -> (ts, cat, dur)
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        if not (w0 <= e["ts"] <= w1):
            continue
        corr = e.get("args", {}).get("correlation")
        rt = runtime_ts.get(corr) if corr is not None else None
        if rt is None:
            continue
        ts, tid = rt
        cat = _category(str(e.get("name", "")), str(e.get("cat")))
        queries[tid].append((ts, cat, e.get("dur", 0.0)))

    # per tid: sweep frames (sorted by ts) maintaining the active stack; snapshot at each query
    agg: dict[tuple[str, tuple[str, ...]], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )  # ms,count
    for tid, qs in queries.items():
        frames = sorted(frames_by_tid.get(tid, []), key=lambda f: f[0])
        qs.sort()
        stack: list[tuple[float, str]] = []  # (end, name)
        fi = 0
        for q_ts, cat, dur in qs:
            while fi < len(frames) and frames[fi][0] <= q_ts:
                stack.append((frames[fi][1], frames[fi][2]))
                fi += 1
            stack = [(end, name) for end, name in stack if end >= q_ts]
            trimmed = tuple(_trim([name for _, name in stack]))
            key = (cat, trimmed)
            agg[key][0] += dur / 1000
            agg[key][1] += 1

    out: dict[str, list[dict[str, Any]]] = {}
    for cat in CATEGORIES:
        raw = sorted(
            ((ms, cnt, stk) for (c, stk), (ms, cnt) in agg.items() if c == cat),
            key=lambda r: -r[0],
        )
        out[cat] = [
            {"stack": list(stk), "ms": round(ms, 1), "count": int(cnt)}
            for ms, cnt, stk in raw[:TOP_PER_CAT]
        ]
    return out


def main() -> None:
    trace_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    paths = sorted(trace_dir.glob("trace_*.json"))
    assert paths, f"no traces in {trace_dir}"
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for p in paths:
        pool = p.stem.replace("trace_", "").split("_rank")[0]
        print(f"attributing {p.name} ({p.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        result[pool] = attribute_trace(p)
        for cat in CATEGORIES:
            if result[pool][cat]:
                top = result[pool][cat][0]
                print(
                    f"   {pool:>9} {cat:11s} top: {top['ms']:.0f}ms  {' › '.join(top['stack'][-3:])}"
                )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
