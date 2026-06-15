"""Per-component ablation effect on every `a+b=` prompt — a cleaner metric than CI.

Causal importance is the learned gate; it can be imprecise or biased. This measures the
real thing: ablate one subcomponent (remove `U_c V_c^T` from its weight) and read how the
last-position next-token distribution moves vs the un-ablated model.

The reference ("original model") is the full reconstruction: every component on AND the
delta component on, which sums to exactly the target weight `W` — so KL is purely the
effect of removing component `c`, with no reconstruction-error floor.

Per (alive component, prompt) at the `=` position it records:
- `kl`        — KL(P_original || P_ablated) of the next-token distribution,
- `abl_token` / `abl_prob` — argmax token id under ablation and its probability,
- `inner_act` — normalized component activation `(x · V_c) / ||V_c||` (x = the component
                module's input: MLP input for gate/up, neuron acts for down),
- `ci`        — lower-leaky causal importance.
Per prompt: `orig_token` / `orig_prob`. Per component: `||U_c||`, `||V_c||`, their product.

All quantities are read at the `=` (last) position. KL is the weight-level ablation effect
(component removed at every position), so a component that fires only at operand positions
still shows up via KL at `=`, but its recorded `inner_act` / `ci` (sampled at `=`) may be
near-zero there.

Compute ordering (one GPU): a single reference forward per prompt-batch returns the logits
AND the input cache, so CI / inner-acts / norms for every component come for free; only the
KL + ablated-token need the per-component ablated forward, and only the last-position logits
are read. With `--skip-eps > 0`, a component whose max output perturbation `||U_c|| *
max_pos|x·V_c|` over the batch is below the threshold is recorded as zero-effect without a
forward (mechanically near-zero, not a CI-based guess).

An 8B forward needs a GPU; pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.collect_ablation_kl <model_path> \
        [--batch-size=512] [--ci-thr=0.1] [--skip-eps=0.0] [--max-prompts=N] \
        [--max-components=N] [--output-dir=PATH] \
        [--slurm [--partition=... --gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Output dir (default `<run_dir>/ablation_kl/`):
- `data.npz`       — grids [n_alive, N, N] (kl, abl_token, abl_prob, inner_act, ci) +
                     per-prompt [N, N] (orig_token, orig_prob) + component order arrays.
- `components.tsv` — one row per alive component: layer, matrix, component, u_norm, v_norm,
                     norm, mean_kl, max_kl.
- `meta.json`      — run id, grid size, KL direction, token-id -> string decode map, etc.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import fire
import numpy as np
import torch
from torch import Tensor

from param_decomp.log import logger
from param_decomp.masks import ComponentsMaskInfo
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    parse_module_name,
    submit_self_to_slurm,
)

_MODULE = "param_decomp_lab.scripts.validation.collect_ablation_kl"


@dataclass(frozen=True)
class _AliveComponent:
    module: str  # full path, e.g. model.layers.18.mlp.gate_proj
    short: str  # e.g. mlp.gate_proj
    component: int


def _read_alive(run_dir: Path) -> list[_AliveComponent]:
    tsv = run_dir / "alive_components.tsv"
    assert tsv.exists(), f"missing {tsv}; run find_alive_components first"
    out: list[_AliveComponent] = []
    with tsv.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            layer, short = row["layer"], row["matrix"]
            out.append(
                _AliveComponent(
                    module=f"model.layers.{layer}.{short}",
                    short=short,
                    component=int(row["component"]),
                )
            )
    assert out, "no alive components found"
    return out


def _parse_ab(prompt: str, op: str) -> tuple[int, int]:
    m = re.match(rf"^(\d+){re.escape(op)}(\d+)=$", prompt)
    assert m is not None, f"prompt is not 'a{op}b=': {prompt!r}"
    return int(m.group(1)), int(m.group(2))


@dataclass
class _Grids:
    """[n_alive, N, N] per-(component, prompt) outputs + per-prompt references."""

    kl: np.ndarray
    abl_token: np.ndarray
    abl_prob: np.ndarray
    inner_act: np.ndarray
    ci: np.ndarray
    orig_token: np.ndarray  # [N, N]
    orig_prob: np.ndarray  # [N, N]


def _all_on_masks(
    modules: list[str],
    weight_deltas: dict[str, Tensor],
    n_comp: dict[str, int],
    batch: int,
    seq: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Tensor], dict[str, ComponentsMaskInfo]]:
    """Component masks all-ones + delta fully on (= exact reconstruction). Returns the
    mutable per-module mask tensors and the assembled mask infos sharing them. `n_comp` is
    per-module: C can differ across decomposed matrices."""
    masks = {m: torch.ones(batch, seq, n_comp[m], device=device, dtype=dtype) for m in modules}
    delta_mask = torch.ones(batch, seq, device=device, dtype=dtype)
    infos = {
        m: ComponentsMaskInfo(
            component_mask=masks[m],
            routing_mask="all",
            weight_delta_and_mask=(weight_deltas[m], delta_mask),
        )
        for m in modules
    }
    return masks, infos


def collect_ablation_kl(
    model_path: ModelPath,
    batch_size: int = 512,
    ci_thr: float = 0.1,
    skip_eps: float = 0.0,
    max_prompts: int | None = None,
    max_components: int | None = None,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--batch-size={batch_size}",
            f"--ci-thr={ci_thr}",
            f"--skip-eps={skip_eps}",
        ]
        if max_prompts is not None:
            argv.append(f"--max-prompts={max_prompts}")
        if max_components is not None:
            argv.append(f"--max-components={max_components}")
        if output_dir is not None:
            argv.append(f"--output-dir={Path(output_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-collect-ablation-kl")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    assert cfg.data.prompts_file is not None, "requires prompts-based target data"
    op = "+"

    prompt_texts = [
        ln.strip()
        for ln in Path(cfg.data.prompts_file).expanduser().read_text().splitlines()
        if ln.strip()
    ]
    pool = load_prompts_dataset(cfg.data.prompts_file, cast(Any, tokenizer))
    assert pool.shape[0] == len(prompt_texts)
    if max_prompts is not None:
        pool, prompt_texts = pool[:max_prompts], prompt_texts[:max_prompts]
    pool = pool.to(device)
    seq = pool.shape[1]
    ab = [_parse_ab(t, op) for t in prompt_texts]
    n = max(max(a, b) for a, b in ab)

    alive = _read_alive(run.run_dir)
    if max_components is not None:
        alive = alive[:max_components]
    modules = sorted({a.module for a in alive}, key=parse_module_name)
    n_comp = {m: model.components[m].V.shape[1] for m in modules}
    logger.info(
        f"{len(alive)} alive components over {len(modules)} modules, {pool.shape[0]} prompts, N={n}"
    )

    # Per-component static norms and the V columns (for inner activations).
    v_unit: dict[str, Tensor] = {}  # [d_in, C] columns scaled to unit norm
    v_norm: dict[str, Tensor] = {}
    u_norm: dict[str, Tensor] = {}
    for m in modules:
        V, U = model.components[m].V.detach(), model.components[m].U.detach()  # [d_in,C], [C,d_out]
        vn = V.norm(dim=0)  # [C]
        v_unit[m] = V / vn.clamp_min(1e-12)
        v_norm[m], u_norm[m] = vn.float().cpu(), U.norm(dim=1).float().cpu()

    weight_deltas = model.calc_weight_deltas()
    dtype = next(model.parameters()).dtype

    g = _Grids(
        kl=np.zeros((len(alive), n, n), np.float32),
        abl_token=np.zeros((len(alive), n, n), np.int32),
        abl_prob=np.zeros((len(alive), n, n), np.float32),
        inner_act=np.zeros((len(alive), n, n), np.float32),
        ci=np.zeros((len(alive), n, n), np.float32),
        orig_token=np.zeros((n, n), np.int32),
        orig_prob=np.zeros((n, n), np.float32),
    )
    comp_index = {m: [i for i, a in enumerate(alive) if a.module == m] for m in modules}
    norm_cpu = {m: (v_norm[m] * u_norm[m]) for m in modules}  # [C] output-perturbation scale unit
    n_skipped = 0

    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for start in range(0, pool.shape[0], batch_size):
            chunk = pool[start : start + batch_size]
            b = chunk.shape[0]
            rows = [a_ - 1 for a_, _ in ab[start : start + b]]
            cols = [b_ - 1 for _, b_ in ab[start : start + b]]

            masks, infos = _all_on_masks(modules, weight_deltas, n_comp, b, seq, device, dtype)
            ref = model(chunk, mask_infos=infos, cache_type="input")
            if start == 0:
                # All-on + delta must reproduce the raw target model; otherwise the mask /
                # delta wiring is wrong and every KL would be measured against a bad reference.
                # Compare distributions (robust to bf16 logit-shift), not raw logits.
                tgt = torch.log_softmax(model(chunk)[:, -1].float(), dim=-1)
                rec = torch.log_softmax(ref.output[:, -1].float(), dim=-1)
                recon_kl = (tgt.exp() * (tgt - rec)).sum(-1).mean().item()
                agree = (tgt.argmax(-1) == rec.argmax(-1)).float().mean().item()
                logger.info(
                    f"reconstruction check: mean KL(target||all_on)={recon_kl:.4g}, argmax agree={agree:.3f}"
                )
                assert recon_kl < 0.5, (
                    f"all-on+delta != target (KL {recon_kl}); mask/delta wiring bug"
                )
            logP = torch.log_softmax(ref.output[:, -1].float(), dim=-1)  # [b, vocab]
            P = logP.exp()
            orig_token = logP.argmax(dim=-1).cpu().numpy()
            orig_prob = logP.max(dim=-1).values.exp().cpu().numpy()
            g.orig_token[rows, cols] = orig_token
            g.orig_prob[rows, cols] = orig_prob

            ci = model.calc_causal_importances(ref.cache, sampling="continuous")
            # CI, normalized inner activations, and the per-component output-perturbation
            # scale (for the skip test) — all from the single reference pass.
            pert: dict[str, np.ndarray] = {}
            for m in modules:
                x = ref.cache[m].float()  # [b, seq, d_in]
                inner_all = torch.einsum("bsd,dc->bsc", x, v_unit[m].float())  # normalized
                g_inner = inner_all[:, -1].cpu().numpy()  # [b, C] at the = position
                ci_last = ci.lower_leaky[m][:, -1].float().cpu().numpy()  # [b, C]
                if skip_eps > 0:
                    pert[m] = inner_all.abs().amax(dim=(0, 1)).cpu().numpy() * norm_cpu[m].numpy()
                for i in comp_index[m]:
                    c = alive[i].component
                    g.inner_act[i, rows, cols] = g_inner[:, c]
                    g.ci[i, rows, cols] = ci_last[:, c]

            # Per-component ablated forward (last-position logits only).
            for m in modules:
                mask_m = masks[m]
                for i in comp_index[m]:
                    c = alive[i].component
                    if skip_eps > 0 and pert[m][c] < skip_eps:
                        g.abl_token[i, rows, cols] = orig_token
                        g.abl_prob[i, rows, cols] = orig_prob
                        n_skipped += 1
                        continue
                    mask_m[:, :, c] = 0.0
                    abl = model(chunk, mask_infos=infos)
                    mask_m[:, :, c] = 1.0
                    logQ = torch.log_softmax(abl[:, -1].float(), dim=-1)
                    g.kl[i, rows, cols] = (P * (logP - logQ)).sum(dim=-1).cpu().numpy()
                    g.abl_token[i, rows, cols] = logQ.argmax(dim=-1).cpu().numpy()
                    g.abl_prob[i, rows, cols] = logQ.max(dim=-1).values.exp().cpu().numpy()

            logger.info(f"batch {start // batch_size + 1}: prompts {start}..{start + b}")

    _write_outputs(run.run_dir, output_dir, op, n, alive, v_norm, u_norm, g, tokenizer)
    logger.info(f"skipped {n_skipped} near-zero ablations (skip_eps={skip_eps})")
    return None


def _write_outputs(
    run_dir: Path,
    output_dir: str | None,
    op: str,
    n: int,
    alive: list[_AliveComponent],
    v_norm: dict[str, Tensor],
    u_norm: dict[str, Tensor],
    g: _Grids,
    tokenizer: Any,
) -> Path:
    out_dir = Path(output_dir).expanduser() if output_dir else run_dir / "ablation_kl"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "data.npz",
        kl=g.kl,
        abl_token=g.abl_token,
        abl_prob=g.abl_prob,
        inner_act=g.inner_act,
        ci=g.ci,
        orig_token=g.orig_token,
        orig_prob=g.orig_prob,
        module=np.array([a.module for a in alive]),
        short=np.array([a.short for a in alive]),
        component=np.array([a.component for a in alive], np.int32),
    )

    rows: list[dict[str, Any]] = []
    for i, a in enumerate(alive):
        layer, _ = parse_module_name(a.module)
        un, vn = float(u_norm[a.module][a.component]), float(v_norm[a.module][a.component])
        rows.append(
            {
                "layer": layer,
                "matrix": a.short,
                "component": a.component,
                "u_norm": round(un, 5),
                "v_norm": round(vn, 5),
                "norm": round(un * vn, 5),
                "mean_kl": round(float(g.kl[i].mean()), 6),
                "max_kl": round(float(g.kl[i].max()), 6),
            }
        )
    with (out_dir / "components.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    token_ids = sorted(
        {int(t) for t in np.unique(g.orig_token)} | {int(t) for t in np.unique(g.abl_token)}
    )
    decode = {int(t): tokenizer.decode([int(t)]) for t in token_ids}
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "op": op,
                "n": n,
                "kl_direction": "KL(P_original || P_ablated)",
                "position": "last (= answer)",
                "reference": "all components on + delta on (exact reconstruction)",
                "n_alive": len(alive),
                "token_decode": decode,
            },
            indent=2,
        )
    )
    logger.info(f"wrote {len(alive)} components × {n}x{n} grids to {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(collect_ablation_kl)
