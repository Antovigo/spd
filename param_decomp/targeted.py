"""Delta-mask override for targeted decomposition.

Stdlib-only leaf — `masks.py` imports this, so importing anything heavier here would
cycle via `metrics.base -> component_model -> masks`.
"""

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

# None  -> normal random/adversarial delta mask (default everywhere)
# float -> delta mask pinned to this constant within the `with` scope
_DELTA_OVERRIDE: ContextVar[float | None] = ContextVar("delta_override", default=None)


def get_delta_override() -> float | None:
    return _DELTA_OVERRIDE.get()


@contextlib.contextmanager
def delta_override(value: float) -> Iterator[None]:
    """Pin the delta-component mask to `value` for all mask construction in this scope."""
    token = _DELTA_OVERRIDE.set(value)
    try:
        yield
    finally:
        _DELTA_OVERRIDE.reset(token)
