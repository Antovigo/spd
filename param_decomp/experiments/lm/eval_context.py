"""The LM eval pass and its shared per-batch context (torch-oracle `MetricContext`).

`LMBatchContext` is built ONCE per eval batch by one jitted step — the clean forward
(capturing the CI taps plus every due operation's declared demands) and the CI envelope —
and every batched operation reads from it. Operations that need masked forwards or
ascents run their own steps ON TOP of these values; none recomputes the clean side.

A tPD run has TWO streams — its target prompt pool and the broad corpus — so the pass
carries both batch tuples and `batch_contexts` yields one tagged context per batch of
each; every batched operation folds only the stream it measures (`Stream`). A plain run
has the one stream `data.eval` supplies, spelled `"nontarget"` here so a metric's log key
is decided by the RUN KIND, not by the operation (`stream_log_prefix`).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import jax
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.ci_fn import CI, AnyCI, CIRole, PlacedCIFn, ci_for_role, evaluate_ci
from param_decomp.core.components import ComponentStacks
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.model import (
    CaptureKeys,
    PlacedModel,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.recon import ForwardObservations
from param_decomp.core.run import EvalInvocation
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.experiments.lm.eval import PreparedLMBatch

type Stream = Literal["nontarget", "target"]
"""Which STREAM an eval operation measures. ONE value, not a (batch source, log prefix)
pair, so target-stream batches cannot be spelled under the nontarget stream's log keys."""


@dataclass(frozen=True)
class LMEvalPass(EvalInvocation):
    """Pass-scoped inputs: the raw token batches feed `batch_contexts`; pass-level
    operations (arithmetic's own probe, well-temperedness) read them directly."""

    pass_index: int
    batches: tuple[jax.Array, ...]
    """The `data.eval` draws — a plain run's only stream, a tPD run's broad stream."""
    target_batches: tuple[jax.Array, ...] | None = None
    """A tPD run's target-stream draws; `None` on a plain run, which has no second stream.
    That `None` is also what tells every log key which run kind it is in
    (`stream_log_prefix`)."""

    @property
    def targeted(self) -> bool:
        return self.target_batches is not None


def stream_batches(stream: Stream, eval_pass: LMEvalPass) -> tuple[Array, ...]:
    match stream:
        case "nontarget":
            return eval_pass.batches
        case "target":
            assert eval_pass.target_batches is not None, (
                "target-stream metrics need a tPD run's prompt pool; a plain run has none"
            )
            return eval_pass.target_batches


def nontarget_delta_pinned(*, targeted: bool, stream: Stream) -> bool:
    """A targeted run's NON-TARGET stream recon evals compose delta-pinned (SPEC T4,
    amended 2026-08-20 for the fresh-PGD probe, 2026-08-28 for the CE/KL family): every
    non-target forward keeps the delta escape valve fully on, so what is measured is the
    component-only quantity training actually defends. On a plain run, "nontarget" IS
    the ordinary stream and the plain delta semantics stand."""
    return targeted and stream == "nontarget"


def role_log_segment(role: CIRole) -> str:
    """The CI-role namespace segment. EMPTY for `output`, so a single-role run's keys — and a
    dual run's output-role keys — are exactly what every previous run logged; the hidden role
    lands under `hidden_ci/`, the same segment the training metrics use."""
    match role:
        case "output":
            return ""
        case "hidden":
            return "hidden_ci/"


def stream_log_prefix(stream: Stream, targeted: bool, role: CIRole = "output") -> str:
    """The data a run optimizes for is unlabelled: a plain run keeps `eval/`; a tPD run's
    target pool takes the bare namespace and its broad corpus moves under
    `eval/nontarget_data/`."""
    role_segment = role_log_segment(role)
    match stream:
        case "nontarget":
            return f"eval/nontarget_data/{role_segment}" if targeted else f"eval/{role_segment}"
        case "target":
            assert targeted, "the target stream exists only on a tPD run"
            return f"eval/{role_segment}"


@dataclass(frozen=True)
class LMBatchContext:
    """One eval batch's shared forward products, all sharded device values.

    `ci` is the compute-precision envelope from `evaluate_ci` (what the masked-forward
    consumers read) — a bare `CI` on a single-role run, a `DualCI` on a dual one; consumers
    say which head they mean through `ci_for(role)`. Reduction consumers that historically
    squashed fp32 preactivations recompute their fp32 views from the preactivations,
    preserving each metric's exact numerics. `captures` holds only the operations' declared
    demands — the CI taps are consumed inside the context step and never leave it."""

    stream: Stream
    pass_index: int
    batch_index: int
    tokens: Array
    clean_output: Array
    captures: dict[str, Array]
    ci: AnyCI
    prepared_weights: Any

    def ci_for(self, role: CIRole) -> CI:
        return ci_for_role(self.ci, role)


type LMBatchContextStep = Callable[
    [PlacedModel, ComponentStacks, PlacedCIFn, Array],
    tuple[Array, Array, dict[str, Array], AnyCI, Any],
]
"""`(model, components, placed_ci_fn, token_ids) -> (tokens, clean_output, captures, ci,
prepared_weights)`. `model` (frozen-weight-bearing) is the jit ARG."""


def make_lm_batch_context_step(
    model_static: PlacedModel,
    ci_capture_keys: CaptureKeys,
    operation_capture_keys: CaptureKeys,
    mesh: Mesh | None,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> LMBatchContextStep:
    """One clean forward capturing the union of CI taps and operation demands, plus the
    CI envelope and the pass's bf16 compute weights — the whole shared side of a batch."""
    del model_static
    capture_keys = ci_capture_keys | operation_capture_keys

    def context_step(
        model: PlacedModel, components: ComponentStacks, placed_ci_fn: PlacedCIFn, token_ids: Array
    ) -> tuple[Array, Array, dict[str, Array], AnyCI, Any]:
        tokens = batch_shard_leading(token_ids, mesh)
        result = model.clean_forward(tokens, capture_keys)
        # No batch-sharding constraint on `ci.lower`: under the Explicit mesh the CI
        # envelope is already typed batch-sharded (and C ÷tp) by the placed CI fn — the
        # pre-Explicit replication OOM this once pinned is structurally impossible.
        ci = evaluate_ci(
            placed_ci_fn, select_captures(result.captures, ci_capture_keys), remat=False
        )
        clean_output = batch_shard_leading(result.output, mesh)
        captures = select_captures(result.captures, operation_capture_keys)
        return tokens, clean_output, captures, ci, prepare_compute_weights(model, components)

    return filter_jit(context_step, compiler_options=compiler_options)


def make_lm_batch_contexts(
    context_step: LMBatchContextStep, model: PlacedModel
) -> Callable[[LMEvalPass], Iterator[LMBatchContext]]:
    """The broad stream's batches first, then (on a tPD run) the target pool's — each
    context tagged with its stream so an operation folds only what it measures."""

    def batch_contexts(eval_pass: LMEvalPass) -> Iterator[LMBatchContext]:
        decomposition = eval_pass.state.decomposition
        streams: tuple[Stream, ...] = (
            ("nontarget", "target") if eval_pass.targeted else ("nontarget",)
        )
        for stream in streams:
            for batch_index, token_ids in enumerate(stream_batches(stream, eval_pass)):
                tokens, clean_output, captures, ci, prepared_weights = context_step(
                    model, decomposition.components, eval_pass.placed_ci_fn, token_ids
                )
                yield LMBatchContext(
                    stream=stream,
                    pass_index=eval_pass.pass_index,
                    batch_index=batch_index,
                    tokens=tokens,
                    clean_output=clean_output,
                    captures=captures,
                    ci=ci,
                    prepared_weights=prepared_weights,
                )

    return batch_contexts


def prepared_batch_from_context(
    context: LMBatchContext, role: CIRole, hidden_acts_capture_keys: CaptureKeys
) -> PreparedLMBatch[Any]:
    """The scalar kernels' batch view over the shared context — a reshaping, no compute.
    `role` picks which CI head builds the mask; a single-role run has only `"output"`."""
    return PreparedLMBatch(
        tokens=context.tokens,
        clean=ForwardObservations(
            context.clean_output,
            select_captures(context.captures, hidden_acts_capture_keys),
        ),
        prepared_weights=context.prepared_weights,
        ci_lower=context.ci_for(role).lower,
        valid_row_mask=None,
    )
