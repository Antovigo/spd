"""Slim debug scaffolding for hung-process diagnosis.

Three things, all gated behind env vars:
  1. SIGUSR1 handler that dumps per-thread Python stacks (kill -USR1 <pid>).
  2. faulthandler.dump_traceback_later() heartbeat — auto-dumps stacks if the
     process hangs (no events for PD_DEBUG_FAULT_TIMEOUT_S).
  3. Per-rank ``alive pid=...`` heartbeat line every PD_DEBUG_HEARTBEAT_S
     seconds, so you can see at a glance which ranks froze and when.

Used as a wrapper around the real entry point: see ``run_debug.py``.

Notes:
  * Slow-eval bf16 crash and the missing PhaseProfiler wiring are now both
    fixed upstream (see commits 2dd404f8 / db85f9e4 and the
    `_maybe_enable_torch_profiler` helper in `param_decomp_lab/experiments/
    lm/run.py`), so the old monkey-patches that lived here are gone.
"""

from __future__ import annotations

import datetime as _dt
import faulthandler
import os
import signal
import sys
import threading
import traceback
from pathlib import Path


def _debug_dir() -> Path:
    base = os.environ.get("PD_DEBUG_DIR", "/tmp/pd_3pool_debug")
    out = Path(base) / os.environ.get("SLURM_JOB_ID", "local")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _my_rank() -> int:
    for k in ("RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            return int(v)
    return 0


def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dump_all_thread_stacks(out: Path, reason: str) -> None:
    with open(out, "a") as f:
        f.write(f"\n===== STACK DUMP @ {_now()} (reason: {reason}) pid={os.getpid()}\n")
        for tid, frame in sys._current_frames().items():
            tname = next(
                (t.name for t in threading.enumerate() if t.ident == tid),
                f"unknown-{tid}",
            )
            f.write(f"\n--- thread {tname} (id={tid}) ---\n")
            traceback.print_stack(frame, file=f)
        f.flush()


def _install_signal_handler(stacks_path: Path) -> None:
    def handler(signum: int, frame) -> None:  # noqa: ARG001
        _dump_all_thread_stacks(stacks_path, reason=f"signal {signum}")
        print(f"[debug rank={_my_rank()}] dumped stacks to {stacks_path}", flush=True)

    signal.signal(signal.SIGUSR1, handler)


def _install_faulthandler(stacks_path: Path) -> None:
    # faulthandler holds this fd for the process lifetime.
    fh_file = open(stacks_path.with_suffix(".faulthandler.txt"), "a")  # noqa: SIM115
    faulthandler.enable(file=fh_file, all_threads=True)
    timeout = float(os.environ.get("PD_DEBUG_FAULT_TIMEOUT_S", "0") or 0)
    if timeout > 0:
        faulthandler.dump_traceback_later(timeout=timeout, repeat=True, file=fh_file, exit=False)


def _install_event_heartbeat(events_path: Path) -> None:
    import time

    interval = float(os.environ.get("PD_DEBUG_HEARTBEAT_S", "10"))
    if interval <= 0:
        return

    def loop() -> None:
        while True:
            time.sleep(interval)
            with open(events_path, "a") as f:
                f.write(f"[heartbeat {_now()}] alive pid={os.getpid()}\n")

    t = threading.Thread(target=loop, daemon=True, name="pd-heartbeat")
    t.start()


def install_debug_scaffolding() -> None:
    """Pre-torch-import setup: signal handler + faulthandler + heartbeat.

    Safe to call before torch is imported.
    """
    rank = _my_rank()
    out = _debug_dir()
    stacks = out / f"rank_{rank:03d}_stacks.txt"
    events = out / f"rank_{rank:03d}_events.txt"
    with open(events, "a") as f:
        f.write(
            f"[install {_now()}] rank={rank} world={_world_size()} pid={os.getpid()} "
            f"out_dir={out}\n"
        )
    _install_signal_handler(stacks)
    _install_faulthandler(stacks)
    _install_event_heartbeat(events)
    print(
        f"[debug rank={rank}] scaffolding installed — stacks={stacks} events={events} "
        f"(SIGUSR1 to dump stacks)",
        flush=True,
    )
