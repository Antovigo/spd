"""The LM eval pass and its shared per-batch context (torch-oracle `MetricContext`).

`LMBatchContext` is built ONCE per eval batch by one jitted step — the clean forward
(capturing the CI taps plus every due operation's declared demands) and the CI envelope —
and every shared-forward operation reads from it. Operations that need masked forwards or
ascents run their own steps ON TOP of these values; none recomputes the clean side.

A tPD run has TWO streams — its target prompt pool and the broad corpus — so the pass
carries both batch tuples and `batch_contexts` yields one tagged context per batch of
each; every operation folds only the stream it measures (`Stream`). A plain run has the
one stream `data.eval` supplies, spelled `"nontarget"` here so a metric's log key is
decided by the RUN KIND, not by the operation (`stream_log_prefix`).
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import jax
from jax.sharding import Mesh
from jaxtyping import Array

from param_decomp.core.adversary import SourceStacks
from param_decomp.core.ci_fn.interface import CI, AnyCI, CIFn, CIRole, ci_for_role
from param_decomp.core.ci_fn.runtime import evaluate_ci_from_captures
from param_decomp.core.components import ComponentStacks
from param_decomp.core.model import CaptureKeys, ComponentActivations, PlacedModel, select_captures
from param_decomp.core.recon import ForwardObservations
from param_decomp.core.run import EvalInvocation, SharedForwardOperation
from param_decomp.core.sharding import batch_shard_leading
from param_decomp.experiments.lm.eval import PreparedLMBatch
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.targets.lm_output import LMOutput

type Stream = Literal["nontarget", "target"]
"""Which STREAM an eval operation measures. ONE value, not a (batch source, log prefix)
pair, so target-stream batches cannot be spelled under the nontarget stream's log keys."""


@dataclass(frozen=True)
class LMEvalPass[TargetIn, Conditioning](EvalInvocation[Conditioning]):
    """Inputs for one evaluation occasion. Batches feed the shared forward or can be
    consumed directly by standalone operations such as well-temperedness."""

    pass_index: int
    batches: tuple[TargetIn, ...]
    """The `data.eval` draws — a plain run's only stream, a tPD run's broad stream."""
    target_batches: tuple[TargetIn, ...] | None = None
    """A tPD run's target-stream draws; `None` on a plain run, which has no second stream."""

    @property
    def targeted(self) -> bool:
        return self.target_batches is not None


def stream_batches[TargetIn, Conditioning](
    stream: Stream, eval_pass: LMEvalPass[TargetIn, Conditioning]
) -> tuple[TargetIn, ...]:
    match stream:
        case "nontarget":
            return eval_pass.batches
        case "target":
            assert eval_pass.target_batches is not None, (
                "target-stream metrics need a tPD run's prompt pool; a plain run has none"
            )
            return eval_pass.target_batches


def nontarget_delta_pinned(*, targeted: bool, stream: Stream) -> bool:
    """A targeted run's NON-TARGET stream recon evals compose delta-pinned: every
    non-target forward keeps the delta escape valve fully on, so what is measured is the
    component-only quantity training actually defends. On a plain run, "nontarget" IS the
    ordinary stream and the plain delta semantics stand."""
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMBatchForwardProducts[TargetIn, PreparedT: ComponentActivations, Conditioning]:
    """Device values produced together by one clean forward and CI evaluation.

    `tokens` retains the target's input type, including document layout. `conditioning`
    carries the complete input for masked execution; `captures` contains only the
    consuming operations' requested activations. The CI envelope stays in compute
    precision; consumers that require fp32 squashings derive them from preactivations.
    """

    tokens: TargetIn
    clean_output: LMOutput
    captures: dict[str, Array]
    ci: AnyCI
    """A bare `CI` on a single-role run, a `DualCI` on a dual one; consumers say which head
    they mean through `ci_for(role)`."""
    prepared_weights: PreparedT
    conditioning: Conditioning

    def ci_for(self, role: CIRole) -> CI:
        return ci_for_role(self.ci, role)


@dataclass(frozen=True)
class LMBatchContext[TargetIn, PreparedT: ComponentActivations, Conditioning]:
    """One batch's forward products, position in the eval pass, and persistent sources.

    Sources retain their stored representation; consumers read `source_values_to_float`.
    """

    pass_index: int
    batch_index: int
    forward: LMBatchForwardProducts[TargetIn, PreparedT, Conditioning]
    persistent_sources: dict[str, SourceStacks]
    stream: Stream = "nontarget"


type LMBatchContextStep[
    TargetIn,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
] = Callable[
    [
        PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        ComponentStacks,
        CIFn[Conditioning],
        TargetIn,
    ],
    LMBatchForwardProducts[TargetIn, PreparedT, Conditioning],
]
"""`(model, components, ci_fn, token_ids) -> LMBatchForwardProducts`.
The frozen model remains a jit argument.
"""


def make_lm_batch_context_step[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model_static: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
    ci_capture_keys: CaptureKeys,
    operation_capture_keys: CaptureKeys,
    mesh: Mesh | None,
) -> LMBatchContextStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]:
    """One clean forward capturing the union of CI taps and operation demands, plus the
    CI envelope and the pass's bf16 compute weights — the whole shared side of a batch."""
    del model_static
    capture_keys = ci_capture_keys | operation_capture_keys

    def context_step(
        model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
        components: ComponentStacks,
        ci_fn: CIFn[Conditioning],
        token_ids: TargetIn,
    ) -> LMBatchForwardProducts[TargetIn, PreparedT, Conditioning]:
        tokens = jax.tree.map(lambda a: batch_shard_leading(a, mesh), token_ids)
        result = model.clean_forward(tokens, capture_keys)
        # The CI function's output placement already carries batch and component ownership.
        prepared_weights = model.prepare_compute_weights(components)
        ci = evaluate_ci_from_captures(
            ci_fn.prepare(),
            select_captures(result.captures, ci_capture_keys),
            result.conditioning,
            prepared_weights,
            sequence=result.sequence,
            remat=False,
        )
        clean_output = model.pin_output_batch(result.output, mesh)
        captures = select_captures(result.captures, operation_capture_keys)
        return LMBatchForwardProducts(
            tokens=tokens,
            clean_output=clean_output,
            captures=captures,
            ci=ci,
            prepared_weights=prepared_weights,
            conditioning=result.conditioning,
        )

    return context_step


def make_lm_batch_contexts[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    context_steps: Mapping[
        Stream, LMBatchContextStep[TargetIn, PreparedT, Conditioning, PreparedMaskingT]
    ],
    model: PlacedModel[TargetIn, LMOutput, PreparedT, Conditioning, PreparedMaskingT],
) -> Callable[
    [LMEvalPass[TargetIn, Conditioning]],
    Iterator[LMBatchContext[TargetIn, PreparedT, Conditioning]],
]:
    """The broad stream's batches first, then (on a tPD run) the target pool's — each
    context tagged with its stream so an operation folds only what it measures. One
    context step per stream: the two streams' batches have different shapes."""

    def batch_contexts(
        eval_pass: LMEvalPass[TargetIn, Conditioning],
    ) -> Iterator[LMBatchContext[TargetIn, PreparedT, Conditioning]]:
        decomposition = eval_pass.decomposition
        for stream, context_step in context_steps.items():
            for batch_index, token_ids in enumerate(stream_batches(stream, eval_pass)):
                forward = context_step(
                    model, decomposition.components, decomposition.ci_fn, token_ids
                )
                yield LMBatchContext(
                    pass_index=eval_pass.pass_index,
                    batch_index=batch_index,
                    forward=forward,
                    persistent_sources=eval_pass.persistent_sources,
                    stream=stream,
                )

    return batch_contexts


def prepared_batch_from_context[
    TargetIn: LMBatch | LMBatchWithDocuments,
    PreparedT: ComponentActivations,
    Conditioning,
](
    context: LMBatchContext[TargetIn, PreparedT, Conditioning],
    reconstruction_capture_keys: CaptureKeys,
    role: CIRole = "output",
) -> PreparedLMBatch[TargetIn, PreparedT, Conditioning]:
    """The scalar kernels' batch view over the shared context — a reshaping, no compute.
    `role` picks which CI head builds the mask; a single-role run has only `"output"`."""
    return PreparedLMBatch(
        tokens=context.forward.tokens,
        clean=ForwardObservations(
            context.forward.clean_output,
            select_captures(context.forward.captures, reconstruction_capture_keys),
        ),
        prepared_weights=context.forward.prepared_weights,
        conditioning=context.forward.conditioning,
        ci_lower=context.forward.ci_for(role).lower,
        valid_row_mask=None,
    )


@dataclass(frozen=True)
class MultiStreamOperationPlan[PassT, ContextT]:
    """A shared-forward operation that folds SEVERAL streams' contexts (e.g. both streams'
    mean CI in one figure). Its kernels compile against each stream's example context —
    the streams' batches differ in shape — so it receives them all."""

    streams: tuple[Stream, ...]
    prepare: Callable[[PassT, Mapping[Stream, ContextT]], SharedForwardOperation[PassT, ContextT]]
