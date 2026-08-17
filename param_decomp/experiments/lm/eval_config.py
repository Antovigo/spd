"""Authored metric schemas whose semantics require an LM target."""

from typing import ClassVar, Literal

from pydantic import PositiveInt

from param_decomp.core.base_config import BaseConfig, Probability


class CEandKLLossesConfig(BaseConfig):
    """Token-level CE/KL metrics for categorical LM outputs."""

    slow: ClassVar[bool] = False
    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIMaskedAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"
    n_mask_samples: PositiveInt = 1


class ABGridDatasetConfig(BaseConfig):
    """Per-component CI + inner-activation snapshots over an arithmetic operand grid,
    written under the run dir with the applet that browses them.

    Never runs on the first eval pass, whatever `eval.slow_on_first_step` says — an
    untrained decomposition saves every component and shows nothing
    (`experiments.eval_config.schedule_for`)."""

    slow: ClassVar[bool] = True
    type: Literal["ABGridDataset"] = "ABGridDataset"
    mean_ci_floor: Probability
    """Full `a x b` grids are stored only for components whose prompt-mean CI reaches this
    at SOME recorded position; the mean-CI vector is stored for every component, so the cut
    stays visible in the applet. It is the ONLY bound on snapshot size — ~40 KB per saved
    component per recorded position at a 100x100 grid (u8 CI + f16 inner, base64) — so read
    `eval/ab_grids/saved_components/total` before trusting a floor at production C."""
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    positions: list[int] | None = None
    """Prompt positions to record (negatives allowed). `null` records the answer (last)
    position alone. Each extra position multiplies both the gathered columns and the
    snapshot on disk."""
    chunk_prompts: PositiveInt | None = None
    """Rows per forward through the grid. `null` sends all `|a_range| x |b_range|` prompts
    in ONE forward — 10000 prompts needs ~20GiB and OOMs a 45GB L40. Chunking bounds the
    forward without shrinking the grid; the CI / inner columns are concatenated across
    chunks and the mean CI sums over them, so WHICH components are saved is
    chunk-count-invariant and their values match the unchunked pass up to float
    reassociation (SPEC D4). Prefer a multiple of the device count so no chunk carries
    sharding pad."""


class TwoStreamCIMeanPerComponentConfig(BaseConfig):
    """Both streams' mean CI per component on ONE axis per site, ordered by descending
    TARGET mean and coloured by stream.

    Computes `CIMeanPerComponent`'s reduction on both streams, so authoring both pays for
    the nontarget pass twice. Refuses on a plain run, which has no target stream."""

    slow: ClassVar[bool] = True
    type: Literal["TwoStreamCIMeanPerComponent"] = "TwoStreamCIMeanPerComponent"
