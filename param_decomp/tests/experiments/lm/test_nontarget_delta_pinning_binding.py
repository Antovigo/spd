"""Which eval operations get the delta pinned ON, at the BINDING level (SPEC T4).

The kernel-level guards live elsewhere (`tests/core/test_fresh_pgd_delta_pinning.py`,
`tests/experiments/lm/test_eval.py`): they pin what a `delta_pinned` step computes. This
pins the wiring instead — that a targeted run builds EVERY non-target-stream recon
operation pinned and every target-stream one unpinned, and that a plain run pins nothing.

This has been wrong before: an unpinned non-target probe measures the delta->0 attack no
component-side training can defend, which reads as a decomposition failure that is
really a measurement bug. The truth table is one function, so pin the function and pin
that all three recon families actually call it.
"""

from collections.abc import Callable
from typing import Any, cast

import pytest

from param_decomp.core.configs import PGDReconLossConfig
from param_decomp.core.eval_schedule import Every
from param_decomp.experiments.lm import scalar_eval_operations as ops
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import Stream, nontarget_delta_pinned

RECON_FAMILIES = ("ce_kl", "masked_kl", "fresh_pgd")


def test_only_a_targeted_runs_nontarget_stream_pins_the_delta():
    assert nontarget_delta_pinned(targeted=True, stream="nontarget")
    assert not nontarget_delta_pinned(targeted=True, stream="target")
    # On a plain run "nontarget" IS the ordinary stream; plain delta semantics stand.
    assert not nontarget_delta_pinned(targeted=False, stream="nontarget")


def _record_pinning(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Bind each recon family's operation and capture the `delta_pinned` it asked for."""
    seen: dict[str, bool] = {}

    def scorer(name: str) -> Callable[..., Any]:
        def make(*_args: Any, delta_pinned: bool = False, **_kwargs: Any) -> Any:
            seen[name] = delta_pinned
            return lambda *_a, **_k: {}

        return make

    monkeypatch.setattr(ops, "make_ce_kl_scorer", scorer("ce_kl"))
    monkeypatch.setattr(ops, "make_masked_kl_scorer", scorer("masked_kl"))
    monkeypatch.setattr(ops, "make_fresh_pgd_scorer", scorer("fresh_pgd"))
    return seen


def _bind(stream: Stream, *, targeted: bool) -> None:
    """Bind one operation of each recon family; only the (stream, run kind) pair varies.
    Every other argument is inert here — the scorers are stubbed."""
    unused = cast(Any, None)
    ops.make_ce_kl_operation(
        CEandKLLossesConfig(rounding_threshold=0.0),
        Every(1),
        stream,
        unused,
        unused,
        0,
        1,
        unused,
        {},
        "output",
        targeted=targeted,
    )
    ops.make_masked_kl_operation(
        "ci_masked", Every(1), stream, unused, unused, 0, 1, unused, {}, "output",
        targeted=targeted,
    )  # fmt: skip
    ops.make_fresh_pgd_operation(
        PGDReconLossConfig(init="random", source_shape="c", n_steps=2, step_size=0.1),
        Every(1),
        stream,
        unused,
        unused,
        0,
        1,
        unused,
        {},
        "output",
        targeted=targeted,
    )


@pytest.mark.parametrize(
    ("targeted", "stream", "expected"),
    [(True, "nontarget", True), (True, "target", False), (False, "nontarget", False)],
)
def test_every_recon_family_threads_the_rule(
    monkeypatch: pytest.MonkeyPatch, targeted: bool, stream: Stream, expected: bool
) -> None:
    seen = _record_pinning(monkeypatch)
    _bind(stream, targeted=targeted)
    assert set(seen) == set(RECON_FAMILIES), sorted(seen)
    assert seen == dict.fromkeys(RECON_FAMILIES, expected), seen
