# Period separation — commands

All commands run via SLURM (login node stays idle). `$RUN` is a run dir under
`PARAM_DECOMP_OUT_DIR/runs/`; ad-hoc sbatch files live in `~/pd_scratch/psep/`.

## Score an analysed run (CPU)

Needs the run's `alive_subcomponents_per_position*.json` (from `find_alive_subcomponents`).

```bash
cd ~/Code/param-decomp/8B_targeted && source .venv/bin/activate
python -m param_decomp_lab.scripts.validation.score_period_separation \
    "$RUN/analysis/datasets/alive_subcomponents_per_position.json"
# → $RUN/analysis/datasets/period_separation.tsv + period_separation_summary.tsv
```

Batch over every analysed addsub run: `sbatch ~/pd_scratch/score_psep_all.sbatch`.

## Compare runs (the numbers that matter)

`+` rows at the answer position, MLP matrices:

```bash
awk -F'\t' 'NR==1 || ($1=="+" && $2==4 && $4 ~ /mlp/)' \
    "$RUN/analysis/datasets/period_separation_summary.tsv" | column -t -s$'\t'
```

## Launch a probe run (2 GPUs, ~3.5h at 5k steps)

Probe yamls: `~/pd_scratch/subspace_restriction/cifn_pipeline/psep/addsub-L18-06-psep-<v>.yaml`
(copied from the coupled stage4 yaml; `steps: 5000` + the probed change + `label`).

```bash
sbatch --gpus=2 --mem=64G --time=6:00:00 \
    ~/pd_scratch/subspace_restriction/run_ddp.sbatch \
    ~/pd_scratch/subspace_restriction/cifn_pipeline/psep/addsub-L18-06-psep-<v>.yaml \
    addsub-L18-06-psep-<v>
```

(NEVER submit `run_ddp.sbatch` bare — its in-file defaults are `--gpus=4 --time=00:30:00`.)

## Analyse a finished probe (1 GPU: alive sweep → score + heatmaps)

```bash
sbatch ~/pd_scratch/psep/psep_analyze.sbatch "$RUN" 5000
```

Runs `find_alive_subcomponents` on `model_5000.pth` from the `subspace_restriction`
worktree (probe configs validate natively there), then `score_period_separation` and
`plot_ab_heatmaps` (both ops) from `8B_targeted`. Default output names — probes have a
single checkpoint, so no `_step` suffixes.
