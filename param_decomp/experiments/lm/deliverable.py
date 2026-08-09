"""The PRODUCT description of a finished run — the config-side twin of the checkpoint's
`decomposition` item (SPEC S22).

A run dir holds one pinned document, `launch_config.yaml`: the PROCESS record, whose
shape varies by algorithm (plain VPD vs tPD) and whose sections mostly describe how the
run trained. Consumers (harvest, autointerp, clustering, `run_metadata`) need none of
that — only what was decomposed and with what apparatus. `DecompositionDeliverable` is
exactly those sections, so every run shape and any future algorithm yields the same
deliverable, and the consumer path never learns run shapes exist.

Today the deliverable's document is a PROJECTION of the pin (read out of
`launch_config.yaml`, tolerating the process sections it doesn't consume — the pin was
validated strictly by the root that authored it; this is not that boundary). Materializing
it as its own run-dir artifact later changes only `load_deliverable`'s document source.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ConfigDict

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.experiments.lm.config import (
    LMCIFnArch,
    LMDataConfig,
    LMDecompositionConfig,
    LMTargetConfig,
    resolve_decomposition,
    resolve_lm_ci_arch,
)
from param_decomp.experiments.lm.resolved import AnyLMTargetConfig, ResolvedLMData
from param_decomp.infra.dataset_store import resolve_dataset_ref


class _ScheduleSeed(BaseConfig):
    """The one `pd` fact consumers use: the seed that makes the training batch schedule
    a pure function of step, so harvest can walk the stream the run saw."""

    model_config = ConfigDict(extra="ignore")

    seed: int


class DecompositionDeliverable(BaseConfig):
    """What a decomposition IS: the frozen target, the apparatus decomposing it, and the
    identity of the stream it trained on (names, portable — harvest walks the same
    shards deterministically via `pd.seed`).

    Deliberately absent: everything the objective decided (losses, targeted streams,
    warmup — the algorithm), and everything about where the run executed (`runtime`:
    dp, placement, remat, XLA flags). Every field here is shared by ALL run shapes. The
    checkpoint pins shapes/dtypes/tree structure, never placement — a consumer restores
    to host (or lays out for its OWN mesh) with no knowledge of the launch topology."""

    target: LMTargetConfig
    decomposition: LMDecompositionConfig
    data: LMDataConfig
    pd: _ScheduleSeed


class _DeliverableProjection(DecompositionDeliverable):
    """The pin-reading arm: the same fields, tolerating the process sections around them."""

    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class ResolvedDeliverable:
    """The deliverable's sections resolved into consumable objects: the concrete target
    config (weights source + flat sites), the built CI-fn architecture, and the resolved
    shard dirs + schedule seed of the training stream."""

    target: AnyLMTargetConfig
    ci_fn: LMCIFnArch
    data: ResolvedLMData
    seed: int


def load_deliverable(run_dir: Path, data_root: Path) -> ResolvedDeliverable:
    """Read a finished run's product description. No placement input, no run-shape
    knowledge: the restore reference is placement- and key-invariant (leaves restore as
    host numpy), so the deliverable needs nothing from the process record."""
    raw = yaml.safe_load((run_dir / LAUNCH_CONFIG_FILENAME).read_text())
    doc = _DeliverableProjection.model_validate(raw)
    resolved = resolve_decomposition(doc.target, doc.decomposition, data_root)
    ci_fn = resolve_lm_ci_arch(resolved.tree, doc.decomposition.ci, resolved.grammar)
    data = ResolvedLMData(
        dir=resolve_dataset_ref(doc.data.train, data_root),
        eval_dir=resolve_dataset_ref(doc.data.eval, data_root),
    )
    return ResolvedDeliverable(target=resolved.target, ci_fn=ci_fn, data=data, seed=doc.pd.seed)
