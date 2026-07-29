"""Slow-eval (a,b)-grid snapshots of CI and inner activations on an `a<op>b=` prompt pool."""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, override

import numpy as np
import torch
from pydantic import PositiveInt
from torch import Tensor
from torch.distributed import ReduceOp
from transformers import AutoTokenizer

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.ci_fns import CIRole
from param_decomp.distributed import all_reduce, get_distributed_state, is_main_process
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.run_artifacts import RunDirArtifact

_PROMPT_RE = re.compile(r"^(\d+)([+\-*/])(\d+)=$")
_APPLET_FILENAME = "ab_grids_app.html"

# The output net keeps the original unsuffixed payload keys so the applet and any existing
# snapshot reader stay compatible; the hidden net's arrays are additive.
_MEAN_CI_KEY: dict[CIRole, str] = {"output": "mean_ci", "hidden": "mean_ci_hidden"}
_CI_KEY: dict[CIRole, str] = {"output": "ci", "hidden": "ci_hidden"}


class ABGridDatasetConfig(BaseConfig):
    """Every prompt in `prompts_file` must match `a<op>b=`; the pool is forwarded whole
    at each slow eval, so keep it grid-sized (e.g. 100x100 per op), not corpus-sized."""

    type: Literal["ABGridDataset"] = "ABGridDataset"
    prompts_file: str
    tokenizer_name: str
    forward_batch_size: PositiveInt
    mean_ci_floor: Probability
    positions: list[int] | None = None
    """Token positions to record (negatives allowed). None: the answer (last) position."""


@dataclass(frozen=True)
class PromptGrid:
    """Mapping from prompt-pool row order onto dense `[n_ops, n_a, n_b]` grids."""

    ops: list[str]
    op_idx: np.ndarray
    a_idx: np.ndarray
    b_idx: np.ndarray
    a_min: int
    b_min: int
    n_a: int
    n_b: int

    @classmethod
    def from_prompts(cls, prompts: list[str]) -> "PromptGrid":
        parsed = []
        for prompt in prompts:
            match = _PROMPT_RE.match(prompt)
            assert match is not None, f"prompt {prompt!r} does not match 'a<op>b='"
            parsed.append((match.group(2), int(match.group(1)), int(match.group(3))))
        ops = list(dict.fromkeys(op for op, _, _ in parsed))
        a_vals = [a for _, a, _ in parsed]
        b_vals = [b for _, _, b in parsed]
        a_min, b_min = min(a_vals), min(b_vals)
        grid = cls(
            ops=ops,
            op_idx=np.array([ops.index(op) for op, _, _ in parsed]),
            a_idx=np.array(a_vals) - a_min,
            b_idx=np.array(b_vals) - b_min,
            a_min=a_min,
            b_min=b_min,
            n_a=max(a_vals) - a_min + 1,
            n_b=max(b_vals) - b_min + 1,
        )
        flat = (grid.op_idx * grid.n_a + grid.a_idx) * grid.n_b + grid.b_idx
        assert len(np.unique(flat)) == len(prompts), "duplicate (op, a, b) prompts in pool"
        return grid

    def scatter(self, per_prompt: np.ndarray, fill: float) -> np.ndarray:
        """`[n_prompts, *rest]` -> `[n_ops, n_a, n_b, *rest]`, `fill` where no prompt exists."""
        out_shape = (len(self.ops), self.n_a, self.n_b) + per_prompt.shape[1:]
        out = np.full(out_shape, fill, dtype=per_prompt.dtype)
        out[self.op_idx, self.a_idx, self.b_idx] = per_prompt
        return out


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def encode_ci_u8(ci: np.ndarray) -> np.ndarray:
    return np.round(np.clip(ci, 0.0, 1.0) * 255.0).astype(np.uint8)


class ABGridDataset(Metric[ABGridDatasetConfig]):
    """Dataset + applet snapshots of per-prompt CI and inner activations over (a, b).

    At each slow eval, forwards the whole prompt pool through the model (input cache
    only — no masked passes), records per-subcomponent CI and normalized inner
    activation `(x . V_c) / ||V_c||` at the configured positions, and writes
    `<run>/ab_grids/step_<n>.js` plus the exploration applet (`index.html`). Full a x b
    grids are stored only for subcomponents whose per-position mean CI reaches
    `mean_ci_floor` at some position; the per-position mean-CI vector is stored for
    every subcomponent so the applet's threshold slider stays meaningful down to the
    floor. CI grids are quantized to u8 (1/255 steps), inner activations to f16.
    """

    log_namespace = "datasets"
    slow = True
    short_name = "ABGrids"

    def __init__(self, cfg: ABGridDatasetConfig) -> None:
        super().__init__(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.prompt_ids = load_prompts_dataset(cfg.prompts_file, tokenizer)
        with open(cfg.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        self.grid = PromptGrid.from_prompts(prompts)
        seq_len = self.prompt_ids.shape[1]
        raw_positions = cfg.positions if cfg.positions is not None else [-1]
        assert all(-seq_len <= p < seq_len for p in raw_positions), (
            f"positions {raw_positions} out of range for seq_len {seq_len}"
        )
        self.positions = [p % seq_len for p in raw_positions]
        assert len(set(self.positions)) == len(self.positions), (
            f"positions {raw_positions} collapse to duplicates over seq_len {seq_len}"
        )

    @override
    def reset(self) -> None:
        self.step: int | None = None

    @override
    def update(self, ctx: MetricContext) -> None:
        self.step = ctx.step
        return None

    @property
    def ci_roles(self) -> tuple[CIRole, ...]:
        """The CI nets this model has: `("output",)`, or both on a dual-CI run."""
        return ("output", "hidden") if self.model.ci_fn_hidden is not None else ("output",)

    def _saved_indices(self, mean_ci_per_role: dict[CIRole, Tensor]) -> Tensor:
        """Subcomponents reaching `mean_ci_floor` at some position under *either* CI net.

        Max over nets rather than over the output net alone: a subcomponent that matters
        only for the hidden activations is exactly what this experiment is looking for, so
        cutting it from the saved grids would hide the finding.
        """
        best_per_component = torch.stack(
            [mean_ci.amax(dim=0) for mean_ci in mean_ci_per_role.values()]
        ).amax(dim=0)
        return (best_per_component >= self.cfg.mean_ci_floor).nonzero().squeeze(-1)

    def _mean_ci_per_role(
        self, ci_acc: dict["CIRole", dict[str, Tensor]], module: str
    ) -> dict["CIRole", Tensor]:
        """`[n_pos, C]` prompt-mean CI per role for one module."""
        return {role: ci_acc[role][module].mean(dim=0) for role in ci_acc}

    def _accumulate_pool(self) -> tuple[dict["CIRole", dict[str, Tensor]], dict[str, Tensor]]:
        """Per-role, per-module `[n_prompts, n_pos, C]` CI plus inner activations (cpu f32)."""
        assert self.step is not None, "update() never ran before compute()"
        dist_state = get_distributed_state()
        world_size = dist_state.world_size if dist_state is not None else 1
        rank = dist_state.rank if dist_state is not None else 0

        modules = list(self.model.components)
        v_unit = {
            module: (comps.V / comps.V.norm(dim=0).clamp_min(1e-12)).detach().float()
            for module, comps in self.model.components.items()
        }
        n_prompts, n_pos = self.prompt_ids.shape[0], len(self.positions)
        ci_acc: dict[CIRole, dict[str, Tensor]] = {
            role: {m: torch.zeros(n_prompts, n_pos, self.model.module_to_c[m]) for m in modules}
            for role in self.ci_roles
        }
        inner_acc = {m: torch.zeros(n_prompts, n_pos, self.model.module_to_c[m]) for m in modules}

        batch_size = self.cfg.forward_batch_size
        for i, start in enumerate(range(0, n_prompts, batch_size)):
            if i % world_size != rank:
                continue
            chunk = self.prompt_ids[start : start + batch_size].to(self.device)
            out = self.model(chunk, cache_type="input")
            assert not isinstance(out, Tensor)
            row = slice(start, start + chunk.shape[0])
            # Both nets read the same cached acts: one forward, one CI-fn call per role.
            for role in self.ci_roles:
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=out.cache,
                    detach_inputs=False,
                    sampling="continuous",
                    role=role,
                )
                for module in modules:
                    ci_acc[role][module][row] = (
                        ci.lower_leaky[module][:, self.positions].float().cpu()
                    )
            for module in modules:
                raw_x = out.cache[module]
                assert raw_x.is_floating_point(), (
                    f"{module}: inner activations need float input acts "
                    f"(Linear/Conv1D targets), got {raw_x.dtype}"
                )
                x = raw_x[:, self.positions].float()
                inner_acc[module][row] = torch.einsum("bpd,dc->bpc", x, v_unit[module]).cpu()

        if dist_state is not None:
            for acc in (*ci_acc.values(), inner_acc):
                for module in modules:
                    acc[module] = all_reduce(acc[module].to(self.device), op=ReduceOp.SUM).cpu()
        return ci_acc, inner_acc

    def _payload(
        self, ci_acc: dict["CIRole", dict[str, Tensor]], inner_acc: dict[str, Tensor]
    ) -> dict[str, Any]:
        to_comp_major = (4, 3, 0, 1, 2)  # [ops, a, b, pos, comp] -> [comp, pos, ops, a, b]
        modules_payload: list[dict[str, Any]] = []
        for module in ci_acc["output"]:
            mean_ci = self._mean_ci_per_role(ci_acc, module)
            saved = self._saved_indices(mean_ci)
            entry: dict[str, Any] = {
                "name": module,
                "C": ci_acc["output"][module].shape[-1],
                "saved": saved.tolist(),
            }
            for role in self.ci_roles:
                entry[_MEAN_CI_KEY[role]] = _b64(mean_ci[role].numpy().astype(np.float32))
            if len(saved) > 0:
                for role in self.ci_roles:
                    ci_grid = self.grid.scatter(ci_acc[role][module][:, :, saved].numpy(), fill=0.0)
                    entry[_CI_KEY[role]] = _b64(encode_ci_u8(np.transpose(ci_grid, to_comp_major)))
                inner_grid = self.grid.scatter(inner_acc[module][:, :, saved].numpy(), fill=np.nan)
                entry["inner"] = _b64(np.transpose(inner_grid, to_comp_major).astype(np.float16))
            modules_payload.append(entry)
        return {
            "step": self.step,
            "positions": self.positions,
            "seq_len": self.prompt_ids.shape[1],
            "ops": self.grid.ops,
            "a_min": self.grid.a_min,
            "n_a": self.grid.n_a,
            "b_min": self.grid.b_min,
            "n_b": self.grid.n_b,
            "mean_ci_floor": self.cfg.mean_ci_floor,
            "ci_roles": list(self.ci_roles),
            "modules": modules_payload,
        }

    @override
    def compute(self) -> MetricResult:
        with torch.no_grad():
            ci_acc, inner_acc = self._accumulate_pool()
        n_saved = sum(
            len(self._saved_indices(self._mean_ci_per_role(ci_acc, module)))
            for module in ci_acc["output"]
        )
        if not is_main_process():
            return {"ab_grids_saved_components": n_saved}
        payload = self._payload(ci_acc, inner_acc)
        files = {
            f"step_{self.step}.js": f"window.registerABGrids({json.dumps(payload)});".encode(),
            "index.html": (Path(__file__).parent / _APPLET_FILENAME).read_bytes(),
        }
        artifact = RunDirArtifact(dir="ab_grids", files=files, manifest_var="AB_GRIDS_MANIFEST")
        return {"ab_grids": artifact, "ab_grids_saved_components": n_saved}
