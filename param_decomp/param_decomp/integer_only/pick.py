"""Score every learned mask on the check set `Q` and choose one.

Each finished mask (`1[ℓ_j > 0]`, `learn.py`) is scored per prompt of `Q` on `QUANTITIES`
(`objective.py`) against the target, as is the all-candidates model; the target itself is
scored on the quantities that do not compare against it. The RULE: among the sweep's sparsity
weights `λ`, the HIGHEST whose mask's mean `KL_int` to the target over all of `Q` is strictly
below the config's `reference_kl_int`; none qualifying is a failure, not a fallback."""

import json
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.integer_only.config import IntegerOnlyConfig
from param_decomp.integer_only.model import Arch, DenseWeights, Factors, Shared, Slots
from param_decomp.integer_only.objective import QUANTITIES, factored_logp, scores, target_logp
from param_decomp.integer_only.pool import PromptPool

KL_INT = QUANTITIES.index("kl_int")
TARGET = "target"
CANDIDATES = "candidates"


def score_masks(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    dense: DenseWeights,
    slots: Slots,
    masks: Mapping[str, np.ndarray],
    pool: PromptPool,
    rows: np.ndarray,
    int_ids: np.ndarray,
    config: IntegerOnlyConfig,
) -> dict[str, np.ndarray]:
    """`{name: (len(QUANTITIES), len(rows))}` per-prompt scores of each mask (`{name: (n,) bool}`
    over the slots) and of the target (`TARGET`, its KLs 0)."""
    jmasks = {name: slots.mask(on, arch.n_layer) for name, on in masks.items()}
    ids = jnp.asarray(int_ids)
    out: dict[str, list[np.ndarray]] = {name: [] for name in (TARGET, *masks)}
    batch = config.score_batch
    for s in range(0, rows.size, batch):
        chunk = rows[s : s + batch]
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], batch - chunk.size)])
        tokens = jnp.asarray(pool.tokens[pad])
        r = jnp.asarray(pool.r[pad])
        y = ids[r]
        ref = target_logp(arch, shared, dense, tokens, config.target_precision)
        out[TARGET].append(np.asarray(scores(ref, ref, r, y, ids))[:, : chunk.size])
        for name, m in jmasks.items():
            logp = factored_logp(arch, shared, factors, m, tokens, config.precision)
            out[name].append(np.asarray(scores(logp, ref, r, y, ids))[:, : chunk.size])
    return {name: np.concatenate(parts, axis=1) for name, parts in out.items()}


def summary(per_prompt: np.ndarray, op: np.ndarray, ops: tuple[str, ...]) -> dict[str, object]:
    """Mean of every quantity over all prompts and per operation, with the prompt counts."""
    out: dict[str, object] = {}
    groups = {"all": np.ones(op.size, bool)} | {name: op == k for k, name in enumerate(ops)}
    for name, sel in groups.items():
        means = {q: float(per_prompt[i, sel].mean()) for i, q in enumerate(QUANTITIES)}
        out[name] = means | {"n": int(sel.sum())}
    return out


def choose(kl_int: Mapping[float, float], bar: float) -> float | None:
    """The highest `λ` whose mean `KL_int` is strictly below `bar`, or None."""
    qualifying = [lam for lam, kl in kl_int.items() if kl < bar]
    return max(qualifying) if qualifying else None


SUMMARY_COLUMNS = ("model", "size", *(f"{q}_{g}" for g in ("all", "add", "sub") for q in (
    "kl_int", "kl_full", "ce_int", "correct")))  # fmt: skip


def write_summary(
    path: Path, rows: Mapping[str, tuple[int, Mapping[str, object]]], chosen: str | None
) -> None:
    """One TSV row per model: its size (kept components) and the means of `summary`."""
    lines = ["\t".join((*SUMMARY_COLUMNS, "chosen"))]
    for model, (size, summ) in rows.items():
        values: list[str] = [model, str(size)]
        for column in SUMMARY_COLUMNS[2:]:
            quantity, group = column.rsplit("_", 1)
            block = summ.get(group)
            values.append("" if not isinstance(block, dict) else f"{block[quantity]:.6g}")
        values.append("yes" if model == chosen else "")
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n")


def mask_json(
    slots: Slots,
    on: np.ndarray,
    all_sites: Mapping[str, int],
    rule: str,
    source: str,
    threshold: float,
) -> dict[str, object]:
    """A component list in `filtered/integer-only.json`'s schema: `components` `{site: [ids]}`
    over every site (`all_sites`: `{site: C}`), `n_alive`, `n_components`, `rule`, `source`,
    `threshold`."""
    kept: dict[str, list[int]] = {site: [] for site in sorted(all_sites)}
    for j in np.flatnonzero(on):
        kept[slots.sites[j]].append(int(slots.index[j]))
    for ids in kept.values():
        ids.sort()
    return {
        "rule": rule,
        "source": source,
        "threshold": threshold,
        "n_alive": int(on.sum()),
        "n_components": int(sum(all_sites.values())),
        "components": kept,
    }


def read_mask_json(path: Path) -> dict[str, list[int]]:
    doc = json.loads(path.read_text())
    assert {"rule", "source", "threshold", "n_alive", "n_components", "components"} <= set(doc)
    comps = {site: [int(c) for c in ids] for site, ids in doc["components"].items()}
    assert sum(len(v) for v in comps.values()) == doc["n_alive"], path
    return comps
