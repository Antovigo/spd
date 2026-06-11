"""CUDA-event per-phase timing for one training step. Profiling-only; a no-op unless
a timer is installed via `set_active`, so `phase(...)` call sites cost nothing in normal runs.

Events are async markers on the current CUDA stream (near-zero launch cost, no sync until
`report`), so the measured step is the real step — measuring does not perturb it. A phase's
time is wall-clock-on-stream between its markers, which includes GPU idle while blocked on a
recv. Async work on a *separate* stream (overlapped NCCL sends) is not captured.
"""

from contextlib import contextmanager

import torch


class PhaseTimer:
    """Accumulates per-phase CUDA time over a window of steady-state steps.

    Records nothing until step `skip_first` (past compile/warmup), then captures `n_measure`
    steps. `report` syncs once and returns mean ms/step per phase.
    """

    def __init__(self, device: torch.device, skip_first: int, n_measure: int):
        assert n_measure > 0
        self.device = device
        self.skip_first = skip_first
        self.n_measure = n_measure
        self._step = 0
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.reported = False

    @property
    def _measuring(self) -> bool:
        return self.skip_first <= self._step < self.skip_first + self.n_measure

    @property
    def finished(self) -> bool:
        return self._step >= self.skip_first + self.n_measure

    @contextmanager
    def phase(self, name: str):
        if not self._measuring:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.setdefault(name, []).append((start, end))

    def step(self) -> None:
        self._step += 1

    def report(self) -> dict[str, float]:
        torch.cuda.synchronize(self.device)
        return {
            name: sum(s.elapsed_time(e) for s, e in pairs) / self.n_measure
            for name, pairs in self._events.items()
        }


_active_timer: PhaseTimer | None = None


def set_active(timer: PhaseTimer | None) -> None:
    global _active_timer
    _active_timer = timer


@contextmanager
def phase(name: str):
    """Wrap a step phase for timing. No-op unless a `PhaseTimer` is installed via `set_active`."""
    if _active_timer is None:
        yield
    else:
        with _active_timer.phase(name):
            yield


def format_phase_table(per_phase_ms: dict[str, float], *, label: str, step_ms: float) -> str:
    rows = sorted(per_phase_ms.items(), key=lambda kv: kv[1], reverse=True)
    phase_total = sum(per_phase_ms.values())
    width = max((len(n) for n in per_phase_ms), default=5)
    lines = [
        f"=== phase breakdown: {label} (mean over measured steps; step_ms={step_ms:.1f}) ===",
        f"  {'phase':<{width}}  {'ms/step':>9}  {'% step':>7}",
    ]
    for name, ms in rows:
        lines.append(f"  {name:<{width}}  {ms:>9.2f}  {100 * ms / step_ms:>6.1f}%")
    lines.append(
        f"  {'phases summed':<{width}}  {phase_total:>9.2f}  {100 * phase_total / step_ms:>6.1f}%"
    )
    return "\n".join(lines)
