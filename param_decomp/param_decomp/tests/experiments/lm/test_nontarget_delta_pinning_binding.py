"""Which eval operations get the delta pinned ON, at the BINDING level (SPEC T4).

The kernel-level guards live elsewhere (`tests/core/test_fresh_pgd_delta_pinning.py`,
`tests/experiments/lm/test_eval.py`): they pin what a `delta_pinned` step computes. This
pins the wiring instead — that a targeted run builds EVERY non-target-stream recon
operation pinned and every target-stream one unpinned, and that a plain run pins nothing.

This has been wrong before: an unpinned non-target probe measures the delta->0 attack no
component-side training can defend, which reads as a decomposition failure that is
really a measurement bug. The truth table is one function, so pin the function and pin
that the evaluation plan threads it into all three recon families.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from param_decomp.core.configs import CIMaskedReconEvalConfig, EvalPGDReconLossConfig
from param_decomp.core.run import MetricsSink, SharedForwardOperationPlan
from param_decomp.experiments.eval_config import AnyEvalMetricConfig, EvalConfig
from param_decomp.experiments.lm import eval_operations
from param_decomp.experiments.lm.eval_config import CEandKLLossesConfig
from param_decomp.experiments.lm.eval_context import (
    Stream,
    nontarget_delta_pinned,
    stream_log_prefix,
)
from param_decomp.lm.batch import LMBatchWithDocuments

RECON_FAMILIES: dict[str, AnyEvalMetricConfig] = {
    "ce_kl": CEandKLLossesConfig(rounding_threshold=0.0),
    "masked_kl": CIMaskedReconEvalConfig(),
    "fresh_pgd": EvalPGDReconLossConfig(init="random", source_shape="c", n_steps=2, step_size=0.1),
}


def test_only_a_targeted_runs_nontarget_stream_pins_the_delta():
    assert nontarget_delta_pinned(targeted=True, stream="nontarget")
    assert not nontarget_delta_pinned(targeted=True, stream="target")
    # On a plain run "nontarget" IS the ordinary stream; plain delta semantics stand.
    assert not nontarget_delta_pinned(targeted=False, stream="nontarget")


def _record_pinning(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, bool]]:
    """Stub each recon family's operation builder; capture, per log prefix (one per
    stream), the `delta_pinned` the evaluation plan asked it for."""
    seen: dict[str, dict[str, bool]] = {}

    def operation(name: str) -> Callable[..., Any]:
        def make(
            *_args: Any, log_prefix: str, delta_pinned: bool = False, **_kwargs: Any
        ) -> SharedForwardOperationPlan[Any, Any]:
            seen.setdefault(name, {})[log_prefix] = delta_pinned
            return SharedForwardOperationPlan(lambda *_a: cast(Any, None))

        return make

    monkeypatch.setattr(eval_operations, "make_ce_kl_operation", operation("ce_kl"))
    monkeypatch.setattr(eval_operations, "make_masked_kl_operation", operation("masked_kl"))
    monkeypatch.setattr(eval_operations, "make_fresh_pgd_operation", operation("fresh_pgd"))
    return seen


def _bind(metric: AnyEvalMetricConfig, *, targeted: bool) -> None:
    """Build the LM evaluation plan for one recon metric; only the run kind varies. Every
    other argument is inert here — the operation builders are stubbed and nothing runs."""
    batch = LMBatchWithDocuments.from_unsegmented_sequences(jnp.arange(4)[None])
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=10, seed=0),
        data=SimpleNamespace(eval_dir=Path("unused")),
        ci_fn=SimpleNamespace(capture_keys=frozenset()),
        target=None,
    )
    eval_operations.make_lm_evaluation(
        cast(Any, built),
        EvalConfig(batch_size=1, n_steps=1, every=1, slow_every=1, metrics=[metric]),
        cast(Any, SimpleNamespace(model=SimpleNamespace(site_names=("site",), sites=()))),
        jax.random.PRNGKey(0),
        cast(Any, None),
        n_proc=1,
        sink=MetricsSink.silent(),
        compiler_options={},
        sample_batch=lambda _step: batch,
        sample_target_batch=(lambda _step: batch) if targeted else None,
    )


@pytest.mark.parametrize(
    ("targeted", "stream", "expected"),
    [(True, "nontarget", True), (True, "target", False), (False, "nontarget", False)],
)
def test_every_recon_family_threads_the_rule(
    monkeypatch: pytest.MonkeyPatch, targeted: bool, stream: Stream, expected: bool
) -> None:
    seen = _record_pinning(monkeypatch)
    for metric in RECON_FAMILIES.values():
        _bind(metric, targeted=targeted)
    assert set(seen) == set(RECON_FAMILIES), sorted(seen)
    prefix = stream_log_prefix(stream, targeted)
    assert {name: by_prefix[prefix] for name, by_prefix in seen.items()} == dict.fromkeys(
        RECON_FAMILIES, expected
    ), seen
