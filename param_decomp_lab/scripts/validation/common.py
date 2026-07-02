"""Shared helpers for the validation scripts."""

import base64
import csv
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from param_decomp.component_model import ComponentModel
from param_decomp.log import logger
from param_decomp_lab.experiments.lm.run import LMExperimentConfig, SavedLMRun
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME, REPO_ROOT
from param_decomp_lab.infra.slurm import SlurmConfig, generate_script, submit_slurm_job

# The arithmetic-analysis scripts (roadmap_addition_analysis) operate one operation at a
# time. Each operation maps to its infix symbol and its `1..100 × 1..100` prompt file.
OP_SYMBOL = {"add": "+", "sub": "-", "mult": "×"}
_OP_PROMPTS_FILE = {
    "add": "addition_1-100.txt",
    "sub": "subtraction_1-100.txt",
    "mult": "multiplication_1-100.txt",
}
_PROMPTS_DIR = REPO_ROOT / "param_decomp_lab" / "experiments" / "lm" / "prompts"
# The three L18 MLP matrices these scripts analyse (decomposed attn is ignored).
MLP_MATRICES = ("gate_proj", "up_proj", "down_proj")

# Residual-stream sites captured around a layer's MLP by the fourier-probes pipeline, in
# computation order: pre-MLP resid → RMSNorm output (MLP input) → MLP output (Δ) → block output.
RESID_SITES = ("pre", "norm", "mlp_out", "post")


def b64_f16(arr: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float16).tobytes()).decode("ascii")


def b64_i16(arr: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int16).tobytes()).decode("ascii")


def b64_i8(arr: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int8).tobytes()).decode("ascii")


def op_symbol(op: str) -> str:
    assert op in OP_SYMBOL, f"unknown operation {op!r}; expected one of {list(OP_SYMBOL)}"
    return OP_SYMBOL[op]


def op_prompts_file(op: str) -> Path:
    """The repo's `1..100 × 1..100` prompt file for one operation."""
    path = _PROMPTS_DIR / _OP_PROMPTS_FILE[op]
    assert path.exists(), f"missing prompt file for {op!r}: {path}"
    return path


_AB_PATTERNS = {op: re.compile(rf"^(\d+){re.escape(sym)}(\d+)=$") for op, sym in OP_SYMBOL.items()}


def parse_operands(prompt: str, op: str) -> tuple[int, int]:
    """Parse `a` and `b` from an `a<op>b=` prompt for the given operation."""
    match = _AB_PATTERNS[op].match(prompt)
    assert match is not None, f"prompt is not 'a{op_symbol(op)}b=': {prompt!r}"
    return int(match.group(1)), int(match.group(2))


@dataclass(frozen=True)
class AliveComponent:
    module: str  # full path, e.g. model.layers.18.mlp.gate_proj
    matrix: str  # the alive-TSV `matrix` cell, e.g. mlp.gate_proj
    proj: str  # the bare projection name, e.g. gate_proj
    component: int
    layer: int


def read_alive_components(
    tsv_path: Path, keep_projs: tuple[str, ...] | None = None
) -> list[AliveComponent]:
    """Read an alive-components TSV (schema `layer, matrix, component, ...`).

    `keep_projs`, when given, restricts to those bare projection names (e.g.
    `("gate_proj", "up_proj", "down_proj")` to drop decomposed attention).
    """
    assert tsv_path.exists(), f"missing alive-components TSV: {tsv_path}"
    out: list[AliveComponent] = []
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            matrix = row["matrix"]
            proj = matrix.split(".")[-1]
            if keep_projs is not None and proj not in keep_projs:
                continue
            layer = int(row["layer"])
            out.append(
                AliveComponent(
                    module=f"model.layers.{layer}.{matrix}",
                    matrix=matrix,
                    proj=proj,
                    component=int(row["component"]),
                    layer=layer,
                )
            )
    assert out, f"no alive components (kept projs {keep_projs}) in {tsv_path}"
    return out


def read_mean_ci(tsv_path: Path) -> dict[tuple[str, int], float]:
    """`(proj, component) -> mean CI` from an `alive_filtered_<op>.tsv`."""
    assert tsv_path.exists(), f"missing filtered-alive TSV: {tsv_path}"
    out: dict[tuple[str, int], float] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["matrix"].split(".")[-1], int(row["component"]))] = float(row["mean_ci"])
    return out


def read_subcomp_periods(tsv_path: Path) -> dict[tuple[str, int], int]:
    """`(proj, component) -> representative additive period` from a `subcomp_periods` TSV."""
    assert tsv_path.exists(), f"missing subcomp-periods TSV: {tsv_path}"
    out: dict[tuple[str, int], int] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["matrix"].split(".")[-1], int(row["component"]))] = int(row["period"])
    return out


@dataclass(frozen=True)
class PeriodGroup:
    kind: str  # "additive" | "log" | "none"
    value: float  # additive integer period, or log multiplicative ratio, or 0
    label: str  # e.g. "period 10", "×1.27", "no period"
    sort_key: tuple[int, float]  # additive < log < none, then by value


def read_subcomp_period_groups(tsv_path: Path) -> dict[tuple[str, int], PeriodGroup]:
    """`(proj, component) -> PeriodGroup`, using `period_type` (additive/log/none).

    Back-compatible with older period TSVs that lack the log columns (treated as additive).
    """
    assert tsv_path.exists(), f"missing subcomp-periods TSV: {tsv_path}"
    out: dict[tuple[str, int], PeriodGroup] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["matrix"].split(".")[-1], int(row["component"]))
            kind = row.get("period_type") or "additive"
            if kind == "log":
                ratio = float(row["log_period"])
                out[key] = PeriodGroup("log", ratio, f"×{ratio:g}", (1, ratio))
            elif kind == "additive":
                p = int(row["period"])
                out[key] = PeriodGroup("additive", float(p), f"period {p}", (0, float(p)))
            else:
                out[key] = PeriodGroup("none", 0.0, "no period", (2, 0.0))
    return out


def square_grid_size(ab: list[tuple[int, int]]) -> int:
    """Side `n` of the `1..n × 1..n` operand grid, asserting full unique coverage.

    The arithmetic scripts index dense `[n, n]` grids by `(a-1, b-1)`; a hole or duplicate
    would leave a silent 0 (poisoning means / periods), so we require every cell exactly once.
    """
    n = max(max(a, b) for a, b in ab)
    assert min(min(a, b) for a, b in ab) == 1, "operands must start at 1"
    assert len(set(ab)) == len(ab) == n * n, (
        f"expected a full {n}x{n} grid, got {len(ab)} prompts ({len(set(ab))} unique)"
    )
    return n


def load_component_uv(
    checkpoint: Path, layer: int, projs: tuple[str, ...]
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """`proj -> (V [d_in, C], U [C, d_out])` for each MLP matrix, read via mmap (no forward)."""
    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    prefix = f"_components.model-layers-{layer}-mlp"
    return {
        proj: (sd[f"{prefix}-{proj}.V"].float().numpy(), sd[f"{prefix}-{proj}.U"].float().numpy())
        for proj in projs
    }


def load_target_mlp_weights(
    checkpoint: Path, layer: int, projs: tuple[str, ...]
) -> dict[str, NDArray[np.float32]]:
    """`proj -> W [d_out, d_in]`, the frozen target-model MLP weight, read via mmap (no forward).

    A neuron's read direction is a row of the gate/up weight (`[d_int, d_model]`); its write
    direction is a column of the down weight (`[d_model, d_int]`).
    """
    sd = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    prefix = f"target_model.model.layers.{layer}.mlp"
    return {proj: sd[f"{prefix}.{proj}.weight"].float().numpy() for proj in projs}


@dataclass(frozen=True)
class LoadedRun:
    model: ComponentModel
    cfg: LMExperimentConfig
    run_dir: Path
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def load_lm_run(model_path: ModelPath) -> LoadedRun:
    """Resolve a run path into an eval-moded `ComponentModel` on the compute device,
    alongside its config, run directory, and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = SavedLMRun.from_path(model_path)
    model = saved.load_model().to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(saved.cfg.data.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return LoadedRun(
        model=model,
        cfg=saved.cfg,
        run_dir=saved.checkpoint_path.parent,
        tokenizer=tokenizer,
        device=device,
    )


# Analysis artifacts live under `<run>/analysis/` so `<run>/figures/` stays reserved for the
# figures the training loop emits. Shared datasets (alive lists, activation grids, periods, …)
# go one level deeper, in `<run>/analysis/datasets/`; figures and applets sit directly under
# `<run>/analysis/`.


def analysis_dir(run_dir: Path) -> Path:
    return run_dir / "analysis"


def analysis_datasets_dir(run_dir: Path) -> Path:
    return run_dir / "analysis" / "datasets"


def run_dir_of_dataset(dataset_path: Path) -> Path:
    """Recover the run folder owning a dataset stored under `<run>/analysis/datasets/`."""
    parent = dataset_path.parent
    assert parent.name == "datasets" and parent.parent.name == "analysis", (
        f"expected a dataset under <run>/analysis/datasets/, got {dataset_path}"
    )
    return parent.parent.parent


def escape_tsv_value(s: str) -> str:
    """Backslash-escape characters that break naive tab-splitting, reversibly."""
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


_LAYER_IN_NAME = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def parse_module_name(module_name: str) -> tuple[int, str]:
    """Split a decomposed module path into `(layer, matrix)`, e.g.
    `model.layers.18.mlp.gate_proj` -> `(18, "mlp.gate_proj")`. Falls back to
    `(-1, module_name)` when there is no layer number."""
    match = _LAYER_IN_NAME.search(module_name)
    if match is None:
        return -1, module_name
    return int(match.group(1)), module_name[match.end() :]


@dataclass(frozen=True)
class SlurmOptions:
    """The `--slurm` flag plus its SLURM knobs, shared by every GPU validation script.

    When `slurm` is set, the script re-submits itself as a single job instead of running
    locally — see `submit_self_to_slurm`.
    """

    slurm: bool = False
    partition: str | None = DEFAULT_PARTITION_NAME
    gpus: int = 1
    slurm_time: str = "1:00:00"
    slurm_mem: str | None = None


def submit_self_to_slurm(module: str, argv: list[str], opts: SlurmOptions, job_name: str) -> None:
    """Submit `python -m <module> <argv>` as a SLURM job (run from `REPO_ROOT`, no snapshot).

    `argv` must already be the non-SLURM CLI args (paths expanded to absolute, since the
    job's working directory is the repo root, not the caller's).
    """
    command = shlex.join(["python", "-m", module, *argv])
    config = SlurmConfig(
        job_name=job_name,
        partition=opts.partition,
        n_gpus=opts.gpus,
        time=opts.slurm_time,
        mem=opts.slurm_mem,
    )
    result = submit_slurm_job(generate_script(config, command), job_name)
    logger.section(f"Submitted {job_name} to SLURM")
    logger.values(
        {
            "Job ID": result.job_id,
            "Command": command,
            "Log file": result.log_pattern,
            "Script": str(result.script_path),
        }
    )
