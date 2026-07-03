"""Per-neuron last-position ablation effect on the `a<op>b=` grid — KL, answer flips, Δlogprob.

For each (neuron, prompt) pair this zeroes the neuron's post-SwiGLU activation at the `=`
position (equivalently: removes `act_j · W_down[:, j]` from L18's MLP output there) and reads
how the next-token distribution moves versus the clean model:

- `kl`                     — KL(P_clean || P_ablated) over the full vocab,
- `abl_token` / `abl_prob` — the argmax next token under ablation and its probability (decoding
                             it to a number is what error-mode analysis runs on: 44 → 43 vs 54),
- `answer_flip`            — does the argmax next token change,
- `delta_correct_logprob`  — logprob shift of the true answer's first token,
- `offset_logprob`         — (full grid only, `--stride=1`) ablated logprob of the first token
                             of `str(answer + δ)` for δ in `OFFSETS` (±1, ±2, ±5, … ±100): does
                             ablating a period-p neuron push mass to `answer ± p`? The clean
                             counterpart ships per prompt as `clean_offset_logprob`.

The ablation is **last-position only**: positions before `=` keep their clean K/V, so the
patched forward only re-runs layers `layer+1..31` (+ final norm + lm_head) on the single `=`
token against the clean KV cache — batched over (prompt × neuron-chunk) rows. That makes the
dense all-14336-neurons screen tractable; a neuron acting purely at operand positions is
invisible here by construction (its effect flows through frozen K/V).

The patched pass is hand-rolled (RMSNorm → QKV → RoPE at the last position → attention over
4 clean cached keys + own → MLP) rather than driving HF's Cache machinery. Its numerical
fidelity is measured per prompt-batch by a **null patch** (delta = 0, must reproduce the clean
logits): the resulting `null_kl` grid ships in the output as the noise floor any real KL must
clear, and its max is asserted small.

Prompts are sampled on a `--stride` subgrid (`a, b in VALUES[::stride]`): stride 5 → the 41×41
screen; stride 1 → the full 201×201 grid. Neurons default to all 14336; `--neurons-tsv` (a
`candidates.tsv` with a `neuron` column) restricts to a candidate set, and `--shard-index` /
`--shard-count` split that set contiguously across jobs.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.collect_neuron_ablation_kl \
        <model_path> [--op=add] [--stride=5] [--neurons-tsv=PATH] \
        [--shard-index=0 --shard-count=1] [--layer=18] [--batch-size=64] [--chunk=256] \
        [--output=PATH] [--slurm [--partition=... --gpus=1 --slurm-time=8:00:00 ...]]

Output (default under `<PARAM_DECOMP_OUT_DIR>/runs/neurons/`):
`ablation_screen_<op>.npz` (stride > 1) or `ablation_full_<op>[_shard<i>of<k>].npz` (stride 1):
`kl` / `abl_prob` / `delta_correct_logprob` fp16, `abl_token` int32 and `answer_flip` bool, all
`[n_neurons, n_a, n_b]`; `offset_logprob` fp16 `[n_neurons, n_a, n_b, n_offsets]` (stride 1
only); per-prompt `null_kl`, `orig_token`, `clean_offset_logprob`; plus `neuron_ids`, `a`, `b`
(the sampled operand values), `offsets`, `layer`, `op`, `stride`.
"""

import csv
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers.models.llama.modeling_llama import rotate_half

from param_decomp.log import logger
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    SlurmOptions,
    load_lm_run,
    op_symbol,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.neurons.common import (
    D_INT,
    NEURONS_DIR,
    OFFSETS,
    VALUES,
    correct_first_token_grid,
    offset_first_token_grid,
)

_MODULE = "param_decomp_lab.scripts.validation.neurons.collect_neuron_ablation_kl"

_KL_SLICE_ROWS = 2048  # rows per fp32 log_softmax slice, bounds the [rows, vocab] fp32 buffers
_NULL_KL_MAX = 0.02  # patched pass with delta=0 must reproduce clean logits up to bf16 noise


def _read_neuron_ids(neurons_tsv: Path) -> list[int]:
    with open(neurons_tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows and "neuron" in rows[0], f"expected a 'neuron' column in {neurons_tsv}"
    return [int(r["neuron"]) for r in rows]


class _PatchedTail:
    """Layers `layer+1..end` + final norm + lm_head, re-run on one patched last-position token.

    Built once per model; `capture(...)` hooks stash, per prompt-batch, everything the clean
    forward exposes (clean K/V per tail layer, the patch site's hidden state, the neuron
    activations); `forward(...)` then runs any number of patched rows against that batch.
    """

    def __init__(self, hf: Any, layer: int, device: torch.device) -> None:
        self.tail_layers = list(hf.model.layers[layer + 1 :])
        self.final_norm = hf.model.norm
        self.lm_head = hf.lm_head
        cfg = hf.config
        self.n_heads: int = cfg.num_attention_heads
        self.n_kv_heads: int = cfg.num_key_value_heads
        self.head_dim: int = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.d_model: int = cfg.hidden_size

        self.capture = True  # off during patched passes, whose k/v_proj calls also fire the hooks
        self.captured_kv: dict[int, dict[str, torch.Tensor]] = {}
        self.h_patch_site: torch.Tensor | None = None
        self.neuron_acts: torch.Tensor | None = None
        self._handles: list[Any] = []

        block = hf.model.layers[layer]
        self._handles.append(block.register_forward_hook(self._hook_block_out))
        self._handles.append(
            block.mlp.get_submodule("down_proj").register_forward_pre_hook(self._hook_down_in)
        )
        for i, tail in enumerate(self.tail_layers):
            attn = tail.self_attn
            self._handles.append(
                attn.get_submodule("k_proj").register_forward_hook(self._make_kv_hook(i, "k"))
            )
            self._handles.append(
                attn.get_submodule("v_proj").register_forward_hook(self._make_kv_hook(i, "v"))
            )

        # RoPE cos/sin for the fixed 5-token prompt positions, shared by every batch — in the
        # model's compute dtype, matching what HF's apply_rotary_pos_emb sees.
        rotary = hf.model.rotary_emb
        pos = torch.arange(5, device=device)[None]
        dummy = torch.zeros(1, 5, self.d_model, device=device, dtype=hf.dtype)
        cos, sin = rotary(dummy, pos)  # [1, 5, head_dim]
        self.cos, self.sin = cos[0], sin[0]

    def _hook_block_out(self, _m: Any, _i: Any, out: Any) -> None:
        self.h_patch_site = (out[0] if isinstance(out, tuple) else out)[:, -1]

    def _hook_down_in(self, _m: Any, args: Any) -> None:
        self.neuron_acts = args[0][:, -1]  # silu(gate) * up at the `=` position

    def _make_kv_hook(self, tail_idx: int, kind: str) -> Any:
        def hook(_m: Any, _i: Any, out: torch.Tensor) -> None:
            if self.capture:
                self.captured_kv.setdefault(tail_idx, {})[kind] = out

        return hook

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()

    def clean_kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per tail layer: RoPE-rotated K and raw V for the clean prefix positions 0..3."""
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(len(self.tail_layers)):
            k_pre = self.captured_kv[i]["k"]  # [B, 5, n_kv_heads * head_dim], pre-RoPE
            v = self.captured_kv[i]["v"]
            bsz = k_pre.shape[0]
            k_pre = k_pre.view(bsz, 5, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, 5, self.n_kv_heads, self.head_dim).transpose(1, 2)
            cos, sin = self.cos[None, None], self.sin[None, None]  # [1, 1, 5, head_dim]
            k_rot = k_pre * cos + rotate_half(k_pre) * sin
            out.append((k_rot[:, :, :4].contiguous(), v[:, :, :4].contiguous()))
        self.captured_kv.clear()
        return out

    def forward(
        self,
        h_rows: Float[torch.Tensor, "rows d_model"],
        clean_kv: list[tuple[torch.Tensor, torch.Tensor]],
        expand: int,
    ) -> Float[torch.Tensor, "rows vocab"]:
        """Run the tail on patched last-position rows; `rows = B * expand`, row-major in B."""
        rows = h_rows.shape[0]
        cos4, sin4 = self.cos[-1], self.sin[-1]  # [head_dim]
        gqa = self.n_heads // self.n_kv_heads
        h = h_rows
        for tail, (k_clean, v_clean) in zip(self.tail_layers, clean_kv, strict=True):
            attn = tail.self_attn
            ln = tail.input_layernorm(h)
            q = attn.q_proj(ln).view(rows, self.n_heads, self.head_dim)
            k = attn.k_proj(ln).view(rows, self.n_kv_heads, self.head_dim)
            v = attn.v_proj(ln).view(rows, self.n_kv_heads, self.head_dim)
            q = q * cos4 + rotate_half(q) * sin4
            k = k * cos4 + rotate_half(k) * sin4

            bsz = k_clean.shape[0]
            assert bsz * expand == rows, (bsz, expand, rows)
            kc = (
                k_clean[:, None]
                .expand(bsz, expand, self.n_kv_heads, 4, self.head_dim)
                .reshape(rows, self.n_kv_heads, 4, self.head_dim)
            )
            vc = (
                v_clean[:, None]
                .expand(bsz, expand, self.n_kv_heads, 4, self.head_dim)
                .reshape(rows, self.n_kv_heads, 4, self.head_dim)
            )
            keys = torch.cat([kc, k[:, :, None].to(kc.dtype)], dim=2)  # [rows, kv, 5, hd]
            vals = torch.cat([vc, v[:, :, None].to(vc.dtype)], dim=2)
            keys = keys.repeat_interleave(gqa, dim=1)  # [rows, n_heads, 5, hd]
            vals = vals.repeat_interleave(gqa, dim=1)
            out = F.scaled_dot_product_attention(q[:, :, None].to(keys.dtype), keys, vals)
            h = h + attn.o_proj(out.reshape(rows, self.n_heads * self.head_dim))
            h = h + tail.mlp(tail.post_attention_layernorm(h))
        return self.lm_head(self.final_norm(h))


class _AblationStats:
    """Per-row results of one patched batch, all on-device: `kl`, `abl_argmax`, `abl_prob`,
    `delta_correct_logprob`, and (when offset tokens are given) `offset_logprob [rows, n_off]`."""

    def __init__(
        self,
        abl_logits: Float[torch.Tensor, "rows vocab"],
        prompt_idx: Int[torch.Tensor, " rows"],
        clean_logprobs: Float[torch.Tensor, "bsz vocab"],
        correct_tok: Int[torch.Tensor, " bsz"],
        offset_tok: Int[torch.Tensor, "bsz n_off"] | None,
    ) -> None:
        rows, device = abl_logits.shape[0], abl_logits.device
        self.kl = torch.empty(rows, dtype=torch.float32, device=device)
        self.abl_argmax = torch.empty(rows, dtype=torch.int64, device=device)
        self.abl_prob = torch.empty(rows, dtype=torch.float32, device=device)
        self.delta_correct_logprob = torch.empty(rows, dtype=torch.float32, device=device)
        self.offset_logprob = (
            None
            if offset_tok is None
            else torch.empty(rows, offset_tok.shape[1], dtype=torch.float32, device=device)
        )
        for s in range(0, rows, _KL_SLICE_ROWS):
            sl = slice(s, min(s + _KL_SLICE_ROWS, rows))
            idx = prompt_idx[sl]
            lp_abl = F.log_softmax(abl_logits[sl].float(), dim=-1)
            lp_clean = clean_logprobs[idx]
            self.kl[sl] = (lp_clean.exp() * (lp_clean - lp_abl)).sum(dim=-1)
            argmax = lp_abl.argmax(dim=-1)
            self.abl_argmax[sl] = argmax
            self.abl_prob[sl] = lp_abl.gather(1, argmax[:, None])[:, 0].exp()
            corr = correct_tok[idx][:, None]
            self.delta_correct_logprob[sl] = (lp_abl.gather(1, corr) - lp_clean.gather(1, corr))[
                :, 0
            ]
            if self.offset_logprob is not None:
                assert offset_tok is not None
                self.offset_logprob[sl] = lp_abl.gather(1, offset_tok[idx])


def _to_grid(t: torch.Tensor, bsz: int, c: int, dtype: type) -> np.ndarray:
    """`[bsz * c, ...]` row-major rows → `[c, bsz, ...]` (neuron-major) numpy."""
    return t.view(bsz, c, *t.shape[1:]).transpose(0, 1).cpu().numpy().astype(dtype)


def collect_neuron_ablation_kl(
    model_path: ModelPath,
    op: str = "add",
    stride: int = 5,
    neurons_tsv: str | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    layer: int = 18,
    batch_size: int = 64,
    chunk: int = 256,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "8:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            f"--op={op}",
            f"--stride={stride}",
            f"--shard-index={shard_index}",
            f"--shard-count={shard_count}",
            f"--layer={layer}",
            f"--batch-size={batch_size}",
            f"--chunk={chunk}",
        ]
        if neurons_tsv is not None:
            argv.append(f"--neurons-tsv={Path(neurons_tsv).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name=f"val-abl-{op}-{shard_index}")
        return None

    all_neurons = list(range(D_INT)) if neurons_tsv is None else _read_neuron_ids(Path(neurons_tsv))
    shard_bounds = np.linspace(0, len(all_neurons), shard_count + 1).astype(int)
    neuron_ids = all_neurons[shard_bounds[shard_index] : shard_bounds[shard_index + 1]]
    assert neuron_ids, f"empty shard {shard_index}/{shard_count}"

    values = VALUES[::stride]
    n_val = len(values)
    sym = op_symbol(op)
    prompts = [f"{a}{sym}{b}=" for a in values for b in values]

    run = load_lm_run(model_path)
    hf = run.model.target_model  # the bare, frozen Llama-3.1-8B
    tokenizer = run.tokenizer
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    assert isinstance(input_ids, torch.Tensor) and input_ids.shape == (len(prompts), 5)
    correct = correct_first_token_grid(tokenizer, op)[::stride, ::stride].reshape(-1)
    with_offsets = stride == 1
    offset_tok_all = (
        offset_first_token_grid(tokenizer, op)[::stride, ::stride].reshape(-1, len(OFFSETS))
        if with_offsets
        else None
    )

    w_down = hf.get_submodule(f"model.layers.{layer}.mlp.down_proj").weight  # [d_model, d_int]
    assert isinstance(w_down, torch.Tensor)
    tail = _PatchedTail(hf, layer, run.device)

    n_n, n_p = len(neuron_ids), len(prompts)
    logger.info(f"{op}: {n_n} neurons x {n_p} prompts (stride {stride}), layer {layer}")
    kl_out = np.zeros((n_n, n_p), dtype=np.float16)
    abl_token_out = np.zeros((n_n, n_p), dtype=np.int32)
    abl_prob_out = np.zeros((n_n, n_p), dtype=np.float16)
    dlp_out = np.zeros((n_n, n_p), dtype=np.float16)
    offset_out = np.zeros((n_n, n_p, len(OFFSETS)), dtype=np.float16) if with_offsets else None
    clean_offset_out = np.zeros((n_p, len(OFFSETS)), dtype=np.float32) if with_offsets else None
    null_kl = np.zeros(n_p, dtype=np.float16)
    orig_token = np.zeros(n_p, dtype=np.int32)

    neuron_ids_t = torch.tensor(neuron_ids, device=run.device)
    with torch.no_grad(), bf16_autocast(enabled=run.cfg.runtime.autocast_bf16):
        n_batches = -(-n_p // batch_size)
        for bi, start in enumerate(range(0, n_p, batch_size)):
            batch = input_ids[start : start + batch_size].to(run.device)
            bsz = batch.shape[0]
            tail.capture = True
            clean_logits = hf(input_ids=batch).logits[:, -1].float()
            clean_logprobs = F.log_softmax(clean_logits, dim=-1)
            clean_argmax = clean_logprobs.argmax(dim=-1)
            correct_tok = torch.from_numpy(correct[start : start + bsz].astype(np.int64)).to(
                run.device
            )
            orig_token[start : start + bsz] = clean_argmax.cpu().numpy().astype(np.int32)
            offset_tok = None
            if offset_tok_all is not None:
                offset_tok = torch.from_numpy(
                    offset_tok_all[start : start + bsz].astype(np.int64)
                ).to(run.device)
                assert clean_offset_out is not None
                clean_offset_out[start : start + bsz] = (
                    clean_logprobs.gather(1, offset_tok).cpu().numpy()
                )

            clean_kv = tail.clean_kv()
            tail.capture = False
            h_site = tail.h_patch_site
            acts = tail.neuron_acts
            assert h_site is not None and acts is not None

            null_logits = tail.forward(h_site, clean_kv, expand=1)
            null_stats = _AblationStats(
                null_logits,
                torch.arange(bsz, device=run.device),
                clean_logprobs,
                correct_tok,
                offset_tok=None,
            )
            null_kl[start : start + bsz] = null_stats.kl.cpu().numpy().astype(np.float16)
            batch_null_max = float(null_stats.kl.max())
            assert batch_null_max < _NULL_KL_MAX, (
                f"null-patch KL noise floor too high on batch {bi}: {batch_null_max}"
            )

            for cs in range(0, n_n, chunk):
                nc = neuron_ids_t[cs : cs + chunk]
                c = nc.shape[0]
                delta = acts[:, nc]  # [bsz, c]
                dirs = w_down[:, nc]  # [d_model, c]
                h_rows = (h_site[:, None, :] - delta[:, :, None] * dirs.T[None, :, :]).reshape(
                    bsz * c, -1
                )
                prompt_idx = torch.arange(bsz, device=run.device).repeat_interleave(c)
                abl_logits = tail.forward(h_rows, clean_kv, expand=c)
                stats = _AblationStats(
                    abl_logits, prompt_idx, clean_logprobs, correct_tok, offset_tok
                )
                psl = slice(start, start + bsz)
                nsl = slice(cs, cs + c)
                kl_out[nsl, psl] = _to_grid(stats.kl, bsz, c, np.float16)
                abl_token_out[nsl, psl] = _to_grid(stats.abl_argmax, bsz, c, np.int32)
                abl_prob_out[nsl, psl] = _to_grid(stats.abl_prob, bsz, c, np.float16)
                dlp_out[nsl, psl] = _to_grid(stats.delta_correct_logprob, bsz, c, np.float16)
                if offset_out is not None:
                    assert stats.offset_logprob is not None
                    offset_out[nsl, psl] = _to_grid(stats.offset_logprob, bsz, c, np.float16)
            if bi % 10 == 0:
                logger.info(f"{op}: prompt batch {bi + 1}/{n_batches}")

    tail.remove_hooks()
    worst_null = float(null_kl.max())
    assert worst_null < _NULL_KL_MAX, f"null-patch KL noise floor too high: {worst_null}"
    logger.info(f"null-patch KL floor: mean {null_kl.mean():.2e}, max {worst_null:.2e}")

    mode = "screen" if stride > 1 else "full"
    shard_tag = f"_shard{shard_index}of{shard_count}" if shard_count > 1 else ""
    out_path = (
        Path(output).expanduser()
        if output
        else NEURONS_DIR / f"ablation_{mode}_{op}{shard_tag}.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = (n_n, n_val, n_val)
    offset_arrays: dict[str, Any] = (
        {}
        if offset_out is None or clean_offset_out is None
        else {
            "offset_logprob": offset_out.reshape(*grid, len(OFFSETS)),
            "clean_offset_logprob": clean_offset_out.reshape(n_val, n_val, len(OFFSETS)),
        }
    )
    np.savez_compressed(
        out_path,
        neuron_ids=np.array(neuron_ids, dtype=np.int32),
        kl=kl_out.reshape(grid),
        abl_token=abl_token_out.reshape(grid),
        abl_prob=abl_prob_out.reshape(grid),
        answer_flip=(abl_token_out != orig_token[None, :]).reshape(grid),
        delta_correct_logprob=dlp_out.reshape(grid),
        null_kl=null_kl.reshape(n_val, n_val),
        orig_token=orig_token.reshape(n_val, n_val),
        a=values,
        b=values,
        offsets=np.array(OFFSETS, dtype=np.int32),
        layer=layer,
        op=op,
        stride=stride,
        **offset_arrays,
    )
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


if __name__ == "__main__":
    fire.Fire(collect_neuron_ablation_kl)
