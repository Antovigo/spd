# Commands — combine_layers

All commands run from the worktree root with the venv active:

```bash
cd /mnt/nw/home/a.vigouroux/Code/param-decomp/combine_layers
source .venv/bin/activate
export PARAM_DECOMP_OUT_DIR=/mnt/nw/home/a.vigouroux/out
```

## Objective 1 — eval readily-combined decompositions

GPU job (1× L40, ~1 h for 4 singles + combined):

```bash
sbatch ~/pd_scratch/combine_layers/obj1_eval.sbatch
```

which runs:

```bash
python -m param_decomp_lab.combine.eval_combined \
  --runs=addsub-L16-04-init-proj,addsub-L17-04-init-proj,addsub-L18-05-coupled,addsub-L19-05 \
  --out=/mnt/nw/home/a.vigouroux/out/combine/obj1_readily_combined.json
```

Key flags: `--ci_thr` (rounding threshold, default 0.01 = the training logs' rounded
recon threshold), `--ci_alive_thr` (L0 threshold, default 0.1), `--n_steps` (eval
batches, default 10), `--include_singles` (default true), `--seed`,
`--nontarget_batch_size` (default 64 — 128 OOMs a 44 GiB L40 with a 4-block model).

Prefix/pair scaling evals (2-, 3-block chains + off-chain pair):

```bash
sbatch ~/pd_scratch/combine_layers/obj1_prefix_eval.sbatch
```

Figures (CPU-only, still via SLURM):

```bash
srun -p compute --cpus-per-task=2 --mem=8G --time=0:10:00 bash -c \
  'source .venv/bin/activate && python -m param_decomp_lab.combine.plot_obj1 \
    --results_json=$HOME/out/combine/obj1_readily_combined.json \
    --out_dir=notes/combine_layers/report_figures \
    --runs_dir=$HOME/out/runs \
    --prefix2_json=$HOME/out/combine/obj1_prefix2.json \
    --prefix3_json=$HOME/out/combine/obj1_prefix3.json \
    --pair_json=$HOME/out/combine/obj1_pair_18_19.json'
```

## Objective 2 — fine-tune assembled (separate CI fns)

Memory probe (1 GPU ≈ one dp=2 rank; steps=3, no wandb; delete the probe run dir after):

```bash
sbatch ~/pd_scratch/combine_layers/obj2_probe_mem.sbatch
```

Real runs (2 GPUs via torchrun; label doubles as run id):

```bash
# 2a: freeze CI fns, train subcomponents only
sbatch ~/pd_scratch/combine_layers/obj2_finetune.sbatch combine-obj2-frozen-ci --freeze_ci_fns

# 2b: train both
sbatch ~/pd_scratch/combine_layers/obj2_finetune.sbatch combine-obj2-both
```

Extra knobs: `--steps` (default 2000), `--components_lr` (1e-4), `--ci_fn_lr` (5e-5),
`--impmin_coeff` (default min over sources = 3e-5), `--batch_size` (default 128 global).

Final comparison eval (fine-tuned checkpoints through the same eval as obj-1):

```bash
sbatch ~/pd_scratch/combine_layers/obj2_final_eval.sbatch
```

Obj-2 figures:

```bash
srun -p compute --cpus-per-task=2 --mem=8G --time=0:10:00 bash -c \
  'source .venv/bin/activate && python -m param_decomp_lab.combine.plot_obj2 \
    --obj1_json=$HOME/out/combine/obj1_readily_combined.json \
    --obj2_json=$HOME/out/combine/obj2_finetuned_eval.json \
    --out_dir=notes/combine_layers/report_figures \
    --runs_dir=$HOME/out/runs'
```

## Objective 3 — single fresh CI fn

```bash
sbatch -J obj3-freshci ~/pd_scratch/combine_layers/obj2_finetune.sbatch \
  combine-L16-19-obj3-freshci-01 \
  --ci_fn_mode=global_fresh --ci_fn_lr=1.6e-4 --steps=2000 --save_every=1000 \
  --group=combine-obj3
```

(`--ci_d_model` / `--ci_n_blocks` override the CI-fn size; default = source arch,
d512 × 4 blocks, which is already lighter than four per-block CI fns.)

## Tests

```bash
python -m pytest param_decomp/tests/test_grouped_ci_fn.py -q
```
