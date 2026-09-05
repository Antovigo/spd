"""SPEC T14: the `CIAnomaly` eval reads both heads on ONE stream and logs under the
stream's namespace with no role segment."""

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from param_decomp.core.ci_fn import CI, DualCI
from param_decomp.core.eval_schedule import Every
from param_decomp.experiments.lm.diagnostic_eval_operations import make_ci_anomaly_operation


def _dual(output: list[list[float]], hidden: list[list[float]]) -> DualCI:
    return DualCI(
        output=CI.from_preactivations({"s": jnp.array(output, jnp.float32)}),
        hidden=CI.from_preactivations({"s": jnp.array(hidden, jnp.float32)}),
    )


def _context(stream: str, ci: DualCI, batch_index: int) -> Any:
    return SimpleNamespace(stream=stream, batch_index=batch_index, ci=ci)


def test_ci_anomaly_eval_logs_per_stream_without_a_role_segment() -> None:
    op = make_ci_anomaly_operation(Every(1), "target", compiler_options={})
    # Row = one (b,t) position, column = subcomponent. Gaps: [0, 0.5, 0, 0.5] per row.
    ci = _dual([[0.2, 0.9, 1.0, 0.5]], [[0.5, 0.4, 1.0, 0.0]])
    acc = op.init()
    acc = op.update(acc, _context("target", ci, 0))
    acc = op.update(acc, _context("nontarget", ci, 0))  # the other stream is ignored
    acc = op.update(acc, _context("target", ci, 1))
    record = op.finish(cast(Any, SimpleNamespace(targeted=True)), acc)

    assert set(record) == {
        "eval/ci_anomaly/linear",
        "eval/ci_anomaly/linear/s",
        "eval/ci_anomaly/squared",
        "eval/ci_anomaly/violating_fraction",
    }
    values = {k: cast(float, v) for k, v in record.items()}
    np.testing.assert_allclose(values["eval/ci_anomaly/linear"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(values["eval/ci_anomaly/linear/s"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(values["eval/ci_anomaly/squared"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(values["eval/ci_anomaly/violating_fraction"], 0.5)


def test_ci_anomaly_eval_is_zero_when_the_ordering_holds_and_labels_the_broad_stream() -> None:
    op = make_ci_anomaly_operation(Every(1), "nontarget", compiler_options={})
    ci = _dual([[0.1, 0.3]], [[0.1, 0.9]])
    acc = op.update(op.init(), _context("nontarget", ci, 0))
    record = op.finish(cast(Any, SimpleNamespace(targeted=True)), acc)
    assert record["eval/nontarget_data/ci_anomaly/linear"] == 0.0
    assert record["eval/nontarget_data/ci_anomaly/violating_fraction"] == 0.0
