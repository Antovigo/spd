# experiments/lm/pretrain — in-house target-LM pretraining (JAX)

Pretrains the FROZEN in-house target LMs that the decomposition trainer (`jsp-train`)
then decomposes — e.g. the pile-pretrained `llama_simple_mlp` (target `t-9d2b8f02`). Small,
simple ML: next-token CE, AdamW, cosine LR + warmup. JAX-native (equinox), torch-free.

The torch original (`torch-oracle:param_decomp_lab/experiments/lm/pretrain/`) was a
torchrun trainer; this is a capability reimplementation (NOT bit-exact), reusing the JAX
single-pool trainer's data/sharding/checkpoint substrate.

## Split (mirrors `pd-jax-lm`)

The trainer needs JAX, so it lives in the JAX distribution; the submit wrapper is lab-side
(torch venv, no JAX import):

- **`param_decomp_jax/jax_single_pool/pretrain/`** — the trainer:
  - `models.py` — trainable equinox defs for all three archs (`GPT2Simple`, `LlamaSimple`,
    `LlamaSimpleMLP`). Weights are stored in torch `nn.Linear` orientation `(d_out, d_in)`
    and `state_dict()` emits the EXACT keys the decomposition loader reads.
  - `config.py` — `PretrainConfig` (the self-contained run yaml schema).
  - `train.py` — `jsp-pretrain <config.yaml>`: the composition root + only I/O layer. fp32
    masters, AdamW (decay on 2D weights only), cosine+warmup, grad clip, orbax sharded
    checkpoints, SIGTERM→save→requeue→resume. Reuses `jax_single_pool.data` (offline
    pre-tokenized parquet, never streamed) + `jax_single_pool.sharding`.
  - `cache.py` — writes the decomposition trainer's `pretrain_cache/<project>-<run_id>/`
    layout (safetensors + `model_config.yaml`) at every save.
  - `configs/` — the run yamls (`pile_llama_simple_mlp-*`, `gpt2_simple-2L`,
    `pile_llama_simple-4L-768`, `*_SMOKE`).
- **`param_decomp_lab/experiments/lm/pretrain/`** (here):
  - `jax_launch.py` — `pd-pretrain`: snapshot + immutable shared-FS workspace + sbatch
    `jsp-pretrain` (or `--local` to run in the current shell). Slim mirror of `pd-jax-lm`.
  - `run_info.py` — `find_pretrain_cache(project, run_id)`: the torch-free read-side index
    into the cache (the torch `PretrainRunInfo`'s wandb-download path is gone — the cache
    is written directly to shared FS).

## Cache compatibility (load-bearing)

A freshly-pretrained target is decomposable with NO conversion. `jsp-pretrain` writes
`PARAM_DECOMP_OUT_DIR/pretrain_cache/<project>-<run_id>/model_step_<N>.safetensors` keyed
`h.{i}.attn.{q,k,v,o}_proj.weight`, `h.{i}.mlp.{c_fc,down_proj}.weight`,
`h.{i}.rms_{1,2}.weight`, `wte.weight`, `ln_f.weight` (NO `lm_head.weight` — tied), every
weight `(d_out, d_in)` — exactly what `jax_single_pool.llama_simple_mlp` reads. A
decomposition config points at it via `target.spec`:

```yaml
target:
  spec:
    kind: pretrained
    model_class: param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: <entity>/<project>/runs/<run_id>
```

(`run_path` resolves to `pretrain_cache/<project>-<run_id>`.) The pretrain model's forward
is bit-identical to the loader's `clean_suffix_logits` round-trip — pinned by
`jax_single_pool/tests/test_pretrain.py`.

## Data

Offline pre-tokenized parquet ONLY (the prestage tool's output;
`jax_single_pool.data.ShardServer`). Shards are `block_size + 1` wide; the trainer serves
the full row and splits `x = tokens[:, :block]`, `y = tokens[:, 1:]` inside the step. The
ported configs point at the staged `datasets/pile_neox_tok_512` (the torch configs'
SimpleStories/streaming data is not staged — runtime tokenization is deliberately
unsupported).

## Usage

```bash
# SLURM (default), 1 node = 8 GPUs
pd-pretrain param_decomp_jax/jax_single_pool/pretrain/configs/pile_llama_simple_mlp-4L-768.yaml --nodes 1

# local (CPU / single GPU; runs the jax venv directly)
pd-pretrain param_decomp_jax/jax_single_pool/pretrain/configs/pile_llama_simple_mlp-2L-128_SMOKE.yaml --local
```
