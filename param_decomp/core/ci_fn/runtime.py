"""Evaluation through the CI lifecycle interface."""

from jaxtyping import Array

from param_decomp.core.ci_fn.interface import CI, AnyCI, CIFn, CIRole, ci_for_role
from param_decomp.core.model import ComponentActivations
from param_decomp.sequence import SequenceLayout


def evaluate_ci_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: dict[str, Array],
    conditioning: Conditioning,
    components: ComponentActivations,
    *,
    sequence: SequenceLayout | None,
    remat: bool,
) -> AnyCI:
    """Evaluate from exactly the physical target captures requested by CI. A dual CI fn
    returns both heads (`DualCI`); consumers scoring one role read it with
    `evaluate_ci_role_from_captures` or `ci_for_role`."""
    _validate_captures(ci_fn, captures)
    return ci_fn(captures, conditioning, components, sequence=sequence, remat=remat)


def evaluate_ci_role_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: dict[str, Array],
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


def _validate_captures[Conditioning](ci_fn: CIFn[Conditioning], captures: dict[str, Array]) -> None:
    expected = ci_fn.capture_keys
    if captures.keys() != expected:
        raise ValueError(
            f"CI captures must match its request: missing={sorted(expected - captures.keys())}, "
            f"unexpected={sorted(captures.keys() - expected)}"
        )
