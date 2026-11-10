"""Evaluation through the CI lifecycle interface."""

from jaxtyping import Array

from param_decomp.core.ci_fn.interface import CI, AnyCI, CIFn, CIRole, ci_for_role
from param_decomp.core.model import CaptureRequest, Captures, ComponentActivations
from param_decomp.sequence import SequenceLayout


def evaluate_ci_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: Captures,
    conditioning: Conditioning,
    components: ComponentActivations,
    *,
    sequence: SequenceLayout | None,
    remat: bool,
) -> AnyCI:
    """Evaluate from the physical target captures CI requested, read as activation arrays;
    `captures` may hold others, which CI never sees. A dual CI fn returns both heads
    (`DualCI`); consumers scoring one role read it with `evaluate_ci_role_from_captures` or
    `ci_for_role`."""
    return ci_fn(
        captures.read(CaptureRequest(ci_fn.capture_keys, Array)),
        conditioning,
        components,
        sequence=sequence,
        remat=remat,
    )


def evaluate_ci_role_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: Captures,
    conditioning: Conditioning,
    components: ComponentActivations,
    *,
    sequence: SequenceLayout | None,
    remat: bool,
    role: CIRole,
) -> CI:
    """One role's bundle — the seam for consumers (evals, probes) that score a single role.
    `role` is REQUIRED: this is the CI read evals build on, so a default here is how a dual
    run ends up reporting one head's numbers under both names."""
    return ci_for_role(
        evaluate_ci_from_captures(
            ci_fn, captures, conditioning, components, sequence=sequence, remat=remat
        ),
        role,
    )
