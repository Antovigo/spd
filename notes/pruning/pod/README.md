# Pruning on one A100 80 GB (RunPod)

Runs `param_decomp.pruning.run` (plan: `../plan.md`) on addsub-08 (`p-ba8b0c08`, step 40000, ceiling
filter `addsub-08-last-pos`). Outputs land in
`$DATA_ROOT/runs/p-ba8b0c08/analysis/pruning/step_40000/<name>/` and are pulled back into
`~/out/pod-backup/p-ba8b0c08/analysis/pruning/`.

1. **Pod**: 1x A100 80 GB, volume ≥ 80 GB (code + venv ~8, Llama 15, checkpoint 10, CI fn 3.4,
   screen shards ~3, XLA cache).
2. **Push** (cluster): `POD_HOST=... POD_PORT=... KEY=~/.ssh/runpod ./push.sh`. The code is
   `git archive HEAD` of this worktree: commit first.
3. **Environment** (pod):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
   source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
   cd /workspace/spd && uv python install 3.12 && uv sync --frozen --no-dev --extra cuda
   ```
4. **Stages 1** (pod): `cd /workspace/spd/notes/pruning/pod && ./run.sh` runs select, validate,
   references, screen, combine and the estimator check. Read `validation.json` (precision noise
   floor: switch `site_precision`/`head_precision` to `tf32` only if its `frac_below_0.1_se` is ≥
   0.99) and the `se_ce` column of `components.tsv`, then set `epsilon` in
   `/workspace/data/pruning/config.yaml`.
5. **Stage 2** (pod): `./run.sh --stages prune,report`. `prune` resumes from the newest
   `sets/S_<j>.npz`.
6. **Pull** (cluster): `./pull.sh`.
