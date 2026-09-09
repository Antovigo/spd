# addsub-all-layers-01 on a Runpod 8x H100 SXM pod — step by step

The full-network (32-block, 224-site) targeted dual-objective decomposition of Llama-3.1-8B
from `addsub-all-layers-sota.yaml`, run on one Runpod pod with 8x H100 SXM 80 GB (NVLink).
Companion scripts: `addsub-all-layers/` (read its `README.md`). Every command below is meant
to be pasted in the order given. `<...>` are placeholders you fill in.

Pinned code: `https://github.com/Antovigo/spd.git`, **tag `addsub-all-layers-01`** on branch
`feature/dual_obj_jax`. The config parses and builds at that tag (224 sites, 32 CI chunks,
`input_dim` 26624, checked on CPU).

**Why a tag and not the branch.** This run resumes across days, possibly on a fresh pod, and
its `launch_config.yaml` is byte-pinned while the CODE is not. If `feature/dual_obj_jax`
advanced between leg 1 and leg 2, the second leg would restore a checkpoint into whatever
the library had become: at best a schema change that refuses the pinned config, at worst a
numerics change halfway through one trajectory. This is why every run on this line launches
from a frozen checkout. A tag buys that immutability while still cloning in one readable
command, which a detached commit hash does not.

Order of operations: pod (1) → code (2) → data + weights (3) → secrets (4) → env sanity (5)
→ probe ladder (6a) → **set the mesh winner in the config** → real run (6b) → retrieval (7)
→ stop / resume / teardown (8).

---

## 1. Pod

**GPU:** 8x H100 SXM 80 GB, one node (NVLink). Not PCIe H100s: the whole mesh argument in the
config header assumes NVLink-priced gathers.

**Driver / CUDA:** the venv brings its own CUDA 12.8.1 libraries as wheels (`uv.lock`:
jax 0.10.1, cudnn 9.8.0.87, NCCL >= 2.28); only the HOST DRIVER matters. It must be
**>= r570 (CUDA 12.8)**: `attention_implementation: auto` uses the cuDNN graph API, which is
exactly what failed on the L40 box's older driver. In the Runpod deploy dialog filter by
CUDA version 12.8 or newer. Check on the pod:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # 8 rows, driver >= 570
```

If the pod's driver is r580 or newer, `--extra cuda` still loads (CUDA 12 wheels are
driver-backward-compatible); `--extra cuda13` is the alternative arm (README) and is
untested on this branch — stay on `cuda`.

**Template / image:** any Runpod Ubuntu 22.04 CUDA image works (e.g. the current
`runpod/pytorch:*-cuda12.8.1-*-ubuntu22.04` template); nothing from the image's python is
used — `uv` installs Python 3.12 and the locked venv. Convenient to have: `git`, `curl`, `rsync`
(`apt-get install -y git curl rsync tmux` if missing); none of them are actually required,
see "If the image is missing tools" in step 2 for the fallbacks. Enable **SSH over exposed TCP port** in the
template (needed for `rsync`/`scp`; the web terminal and the proxied `ssh.runpod.io` login
cannot carry rsync). Put your public key in Runpod → Settings → SSH Public Keys.

**Container disk:** 30 GB is enough (ephemeral: OS + `/root` + uv's build temp).

**Network volume:** create it in the SAME datacenter as the pod (a pod can only mount a
volume from its own DC; pick the DC by H100 SXM availability first). It mounts at
`/workspace`. Everything that must survive a pod restart lives there. Budget:

| item on the volume | GB |
|---|---|
| Llama-3.1-8B safetensors + tokenizer (`original/*` excluded — it is a 16 GB duplicate) | 16 |
| uv venv (`/workspace/spd/.venv`) | 7 |
| uv wheel cache (`/workspace/uv-cache`; deletable after install) | 5 |
| XLA compilation cache (`/workspace/xla-cache`; ~1 GB per distinct compiled step) | 3 |
| datasets `fineweb_llama_tok_64` + `_eval` (1.7 GB each) | 3.4 |
| neuron ranks `addsub1-100_llama31-8b_unit-energy` | 0.01 |
| real run checkpoints: ~20 GB each (4-block: 2.5 GB for 233M params → 1.86G params here), `keep_last 2` + one transient during save/prune | 60 |
| real run misc: `hlo/` dump ~1, `ab_grids/` 10 x ~0.2, wandb, logs | 4 |
| ladder: AB-grid smoke + resume smoke checkpoints (2 kept x ~20 GB each; delete before the real run) | 80 |
| ladder misc: `hlo/` dumps (~1 GB per trial), logs | 5 |
| **total** | **~185** |

→ **250 GB** network volume (200 GB is workable if you delete the ladder's checkpoints
before launching the real run, step 6b).

---

## 2. Code

```bash
cd /workspace
git clone --branch addsub-all-layers-01 https://github.com/Antovigo/spd.git spd   # the pin, by name
cd spd
git describe --tags --exact-match                     # must print addsub-all-layers-01
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
source notes/dual_objective/addsub-all-layers/env.sh   # creates the DATA_ROOT tree, sets UV_CACHE_DIR
uv python install 3.12
uv sync --frozen --no-dev --extra cuda          # README "Install": driver r525–r579 arm; ~7 GB
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # 0.10.1, 8x CudaDevice
```

If `jax.devices()` shows CPU only, the driver is the problem (step 1), not the install.

### If the image is missing tools (no `uv`, no `apt-get`, no `git`)

None of them are hard requirements. Only a Python with `pip` and outbound HTTPS are, and
every Runpod PyTorch image has both. What each one is for, and what to do without it:

**`uv` is never pre-installed** and the block above installs it; that line only assumes
`curl`. Without root or without `curl`, use pip instead, which is why the image's own Python
matters even though nothing else uses it:

```bash
python3 -m pip install --user uv && export PATH="$HOME/.local/bin:$PATH"
# or, if curl is missing but wget is there:
wget -qO- https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
```

Do NOT try to skip uv and `pip install -e ".[cuda]"` with the image's own interpreter: the
project is `requires-python >=3.12,<3.14` and these images ship 3.11, so it refuses. Getting
a 3.12 without `apt` is exactly what `uv python install 3.12` is for.

**`git` is only needed to fetch the code.** Nothing at install or run time uses it: the lock
has no git-sourced dependencies, and the build backend is plain setuptools with a static
version, so `uv sync --frozen` never shells out to git. The repository is public, so the
pinned commit downloads as a tarball with no credentials:

```bash
cd /workspace
URL=https://codeload.github.com/Antovigo/spd/tar.gz/refs/tags/addsub-all-layers-01
curl -L "$URL" -o spd.tar.gz
# or: wget -O spd.tar.gz "$URL"
# or, with neither: python3 -c "import urllib.request as u,os; u.urlretrieve(os.environ['URL'],'spd.tar.gz')"
tar xzf spd.tar.gz && mv spd-addsub-all-layers-01 spd && rm spd.tar.gz
```

Then continue from `cd spd` in the block above, skipping the `git describe` line. The run
scripts log `commit=tarball` instead of a hash and are otherwise unaffected. The cost of
this route is that you cannot ask git which code is running, but the tarball is taken from
the same immutable tag, so `addsub-all-layers-01` still names it.

**`rsync` matters only on the transfer path** (step 3 and step 7): `rsync` spawns a remote
`rsync`, so it must exist on BOTH ends. If the pod has none, `scp -r` usually works
(Runpod runs an OpenSSH server), and if even that fails, tar over ssh needs nothing but a
shell and `tar` on the pod:

```bash
# push a dataset from the cluster to the pod, no rsync and no scp needed
tar cz -C ~/out/datasets fineweb_llama_tok_64 \
  | ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tar xz -C /workspace/data/datasets'
# pull results back the same way
ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tar cz -C /workspace/data/runs/p-a1132b01 ab_grids' | tar xz
```

**`tmux` is never required.** The launcher and ladder driver detach with `setsid nohup` and
write to files under `$DATA_ROOT`; tmux is only convenience for watching them.

If `apt-get` fails merely because the index is stale rather than being blocked, one
`apt-get update` usually fixes it. Do not spend long on it: nothing above needs it.

---

## 3. Data and weights

**3a. Datasets and the neuron-ranks artifact** — copy from the cluster (fastest; the
regeneration is the fallback). On the CLUSTER login node (`~/out` is the cluster's
`DATA_ROOT`):

```bash
POD_HOST=<pod public ip>; POD_PORT=<exposed ssh port>; KEY=~/.ssh/<your key>
RS="rsync -avP -e 'ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new'"
eval $RS ~/out/datasets/fineweb_llama_tok_64 ~/out/datasets/fineweb_llama_tok_64_eval root@$POD_HOST:/workspace/data/datasets/
eval $RS ~/out/neuron_ranks/addsub1-100_llama31-8b_unit-energy root@$POD_HOST:/workspace/data/neuron_ranks/
```

Expected on the pod: `/workspace/data/datasets/fineweb_llama_tok_64/{meta.json,shard_00000.parquet}`
(1.7 GB, `meta.json` = `{"seq_len":64,"tokenizer_name":"meta-llama/Llama-3.1-8B"}`), the
same for `_eval`, and `/workspace/data/neuron_ranks/addsub1-100_llama31-8b_unit-energy/{meta.json,neuron_ranks.npz}`
(6 MB; its `meta.json` lists layers 0..31 and statistic `unit_energy` — the config's init
refuses anything else at load). If the cluster cannot reach the pod, run the same rsync in
the other direction from the pod (`rsync ... <cluster-user>@<cluster-login>:out/datasets/... /workspace/data/datasets/`).

**Fallback — regenerate the datasets on the pod** (~1 h, needs the tokenizer from 3b first;
same source revision as `~/pd_scratch/tpd_jax_tests/prestage_fineweb64.sbatch`):

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh; cd $REPO
for split in train eval; do
  case $split in train) out=$DATA_ROOT/datasets/fineweb_llama_tok_64; skip=0;;
                 eval)  out=$DATA_ROOT/datasets/fineweb_llama_tok_64_eval; skip=1;; esac
  $VENV_PY -m param_decomp.experiments.lm.prestage_tokenized --out_dir $out \
    --num_files 1 --skip_files $skip --task_id 0 --num_tasks 1 \
    --dataset_repo HuggingFaceFW/fineweb --subdir sample/350BT \
    --revision 9bb295ddab0e05d785b879661af7260fed5140fc \
    --tokenizer_name meta-llama/Llama-3.1-8B --seq_len 64 --column_name text --num_proc 16
done
```

The neuron-ranks artifact is regenerable with
`python -m param_decomp.experiments.lm.harvest_neuron_ranks --config notes/dual_objective/addsub-all-layers-sota.yaml --data_root $DATA_ROOT --out_dir $DATA_ROOT/neuron_ranks/addsub1-100_llama31-8b_unit-energy --local_device_count 1 --layers all --batch_size 128`
(one GPU, ~1 h; needs the weights) — but copying the existing 6 MB is strictly better: it is
an immutable name and the copy IS the artifact the 1-block and 4-block runs used.

**Data sufficiency:** the loader cycles the shard in epochs (`BatchSchedule`); one
`sample/350BT` file is ~700M tokens = ~11M rows at seq 64; 40000 steps x 128 rows = 5.1M
rows ≈ 0.47 epoch. No extra data needed.

**3b. Llama-3.1-8B weights** — download from HF on the pod (needs the HF token, step 4, and
Llama-3.1 access on that account). `hf_snapshot_dir` reads the standard hub cache, which
`env.sh` points at `/workspace/hf/hub`:

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
$VENV_PY - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("meta-llama/Llama-3.1-8B",
                      allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                      ignore_patterns=["original/*"])
print(p)
PY
ls $HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B/snapshots/*/   # 4 safetensors shards + index + tokenizer
```

---

## 4. Secrets

Two tokens, kept in ONE file on the volume, outside the repo, never committed:

| token | where to get it | what for |
|---|---|---|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens (read). The account must have accepted the Llama-3.1 license on `meta-llama/Llama-3.1-8B`. | weights download (3b), tokenizer |
| `WANDB_API_KEY` | wandb.ai/authorize | online logging of the real run to project `param-decomp-llama` (default entity of the key — the cluster runs logged under `antvig-pibbss`) |

```bash
cat > /workspace/secrets.env <<'EOF2'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF2
chmod 600 /workspace/secrets.env
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh   # sources it (set -a)
$VENV_PY -c "import huggingface_hub as h; print(h.whoami()['name'])"
$VENV_PY -m wandb login --verify 2>&1 | tail -1
```

Alternative: Runpod template "Environment Variables" for the same two names (they land in
every shell; `env.sh` does not overwrite them). wandb online is fine; `WANDB_DIR` is on the
volume. The ladder's trials carry no `wandb:` block (metrics.jsonl only); the real run does.

---

## 5. Layout and environment

`env.sh` is the single source of truth; source it in EVERY shell. What it sets:

```
VOLUME=/workspace
REPO=$VOLUME/spd                              code @ tag addsub-all-layers-01, venv at $REPO/.venv
DATA_ROOT=$VOLUME/data                        the trainer's positional <data_root>
  runs/<run id>/            launch_config.yaml (pinned), metrics.jsonl, ckpts/<step>/, ab_grids/, hlo/, neuron_alignment.json
  runs/by-name/<run_name>   -> ../<run id>    (symlink, made by pd_run.sh)
  datasets/<name>/          neuron_ranks/<name>/       wandb/      tmp/ (TMPDIR)
  logs/                     ladder/ (trial logs, .rc files, summary.md)     pids/
PARAM_DECOMP_OUT_DIR=$DATA_ROOT    (parity with the cluster scripts; the library does not read it)
WANDB_DIR=$DATA_ROOT/wandb   HF_HOME=$VOLUME/hf   HF_HUB_CACHE=$HF_HOME/hub   TMPDIR=$DATA_ROOT/tmp
UV_CACHE_DIR=$VOLUME/uv-cache
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   (the config's runtime.launch_env re-exports it in-process)
~/.cache/param-decomp -> $VOLUME/xla-cache      (runtime.compilation_cache_dir is ~/.cache/param-decomp/xla)
```

Entry point (what `pd_run.sh` runs; `8` = this process's GPU count = replicate x fsdp x tp):

```bash
$VENV_PY -m param_decomp.experiments.lm.run_targeted <config.yaml> $DATA_ROOT 8 --run_id <p-xxxxxxxx>
```

Sanity before spending GPU time (all CPU-side, ~1 min): parse + build every config against
the pod's data root — resolves the target architecture from the HF snapshot, both datasets'
`meta.json`, and the neuron-ranks artifact's provenance:

```bash
cd $REPO/notes/dual_objective/addsub-all-layers
$VENV_PY make_trials.py --check --data-root $DATA_ROOT
```

---

## 6. Detached running

Nothing here needs SLURM or a live shell: the scripts `setsid nohup` themselves, write logs
under `$DATA_ROOT/logs` / `$DATA_ROOT/ladder`, and `pd_run.sh` carries the two protections
ported from the cluster sbatch scripts — SIGTERM forwarding to the trainer (so a stop is a
checkpoint save, not a kill) and a log-age hang watchdog (a dp>1 OOM does NOT raise: one
rank dies mid-allocation, the rest wait on the NCCL clique; `WATCHDOG_FUSE` seconds without
log growth → `kill -9`, exit 124). tmux is optional convenience (`apt-get install -y tmux;
tmux new -s pd` / `tmux attach -t pd`); the runs do not depend on it.

### 6a. Probe ladder (~1.5–2 h; unattended)

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd $REPO/notes/dual_objective/addsub-all-layers
./run_ladder.sh --detach
tail -f $DATA_ROOT/ladder/driver.log          # Ctrl-C detaches from the tail only
```

What runs, in order (each 30 steps with the step-0 slow eval ON, under `timeout -k 600`):

1. `mesh-c` — replicate 2 / fsdp 4 / zero1 at batch 256/128 (75 min timeout; cold compile)
2. `mesh-b` — replicate 1 / fsdp 8 / zero1, same batch
3. `smoke-abgrid-<best>` — the AB-grid smoke at the fastest fitting mesh: `eval.every 10 /
   slow_every 20`, `mean_ci_floor 0.0`, grid `[1,10]^2`, checkpointing ON. Pass =
   `ab_grids/step_20.js` written, `eval/ab_grids/saved_components/total` > 0, no traceback.
4. `smoke-resume-<best>` — `save_every 20`; the driver SIGTERMs the trainer after the
   step-20 log, expects `SIGTERM: checkpoint saved`, relaunches on the same run id, expects
   `resumed from checkpoint step N` and a clean finish at step 30.

Not in the default list (hard time cap): `fallback-128x96-<m>` (target 128 / nontarget 96 /
eval 96). If NEITHER mesh fits, the driver says so and stops; then run by hand
`./run_ladder.sh --only fallback-128x96-c` (or `-b`), and `--only smoke-abgrid-<m>` /
`--only smoke-resume-<m>` at the shape you pick.

Output: `$DATA_ROOT/ladder/summary.md` — one table (trial, fits?, peak GB/rank, s/step,
projected hours for 40k steps, wall minutes, notes). Paste it back. Per-trial logs are
`$DATA_ROOT/ladder/<trial>.log`; run dirs are `$DATA_ROOT/runs/<run id>` (one fixed id per
trial, in `trials/manifest.tsv`). The driver skips any trial with a `$DATA_ROOT/ladder/<trial>.rc`;
to redo one, delete that file AND its run dir (fixed run ids: a leftover dir would resume
or refuse on the pinned config).

Cold compiles are silent: the log shows `targeted run addsub-all-layers-trial-... | 8 GPU`
and then nothing for tens of minutes. That is normal; the watchdog fuse for trials is
45 min (`WATCHDOG_FUSE=2700`, env-overridable) and the per-trial `timeout` is in the
manifest. Peak memory is read from `train/mem/peak_gb_per_rank`; a trial "fits" only if it
reached step 30 with rc 0 and no traceback in the log.

### 6b. The real run

Before launching — these are the only edits the config ever gets, and they must happen NOW
(the pinned `launch_config.yaml` byte-compares on every resume):

1. `runtime.replicate` / `runtime.fsdp` in `addsub-all-layers-sota.yaml` ← the ladder's
   winner (authored as (b) `1 / 8`; (c) is `2 / 4`). `sharding: zero1` either way.
2. If the pod driver turned out to be < r570 (it should not, step 1): set BOTH
   `target.attention_implementation` and `decomposition.ci.attention.attention_implementation`
   to `xla`.
3. Re-check: `$VENV_PY make_trials.py --check --data-root $DATA_ROOT` (parses the edited
   sota file too). Commit the edit on the pod or note it — the run dir pins the bytes anyway.
4. Free the volume: `rm -rf $DATA_ROOT/runs/p-1adde403 $DATA_ROOT/runs/p-1adde404 $DATA_ROOT/runs/p-1adde413 $DATA_ROOT/runs/p-1adde414`
   (the smokes' ~80 GB of checkpoints; `summary.md` and the `.log`s keep the evidence).

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd $REPO/notes/dual_objective/addsub-all-layers
./launch_run.sh                                  # run id p-a1132b01, run_name addsub-all-layers-01
tail -f $DATA_ROOT/logs/addsub-all-layers-01.latest.log
```

First minutes of a healthy launch: `persistent XLA compilation cache: /root/.cache/param-decomp/xla`
(→ the volume via the symlink), `targeted run addsub-all-layers-01 | 8 GPU / 1 proc | target B=256
nontarget B=128 seq=64 sites=224 steps=40000`, `target prompt pool: 20000 prompts x 5 positions`,
then silence for the compile (cache hit if the ladder ran the same mesh — same step HLO),
then `[step 0] eval/...` (the slow eval), `checkpoint saved @ step 0`, and `[step 100] ...
train/perf/step_time_s=... train/mem/peak_gb_per_rank=...` every ~100 steps. AB grids appear
at step 4000 (`$DATA_ROOT/runs/p-a1132b01/ab_grids/step_4000.js`). wandb: the run is
`param-decomp-llama/addsub-all-layers-01` (id `p-a1132b01`). Close the shell and the laptop;
the runner survives (`setsid nohup`). The hang watchdog fuse is 60 min for the real run
(`WATCHDOG_FUSE=3600`): the trainer logs every 100 steps (~8–10 min), a slow eval + 20 GB
save can be 20–30 min, a cold compile 20–40 min.

---

## 7. Retrieval while running

From your laptop (or the cluster). The run dir is `/workspace/data/runs/p-a1132b01/`:

```bash
POD_HOST=<pod public ip>; POD_PORT=<exposed ssh port>; KEY=~/.ssh/<your key>
RS="rsync -avP -e 'ssh -p $POD_PORT -i $KEY'"
R=root@$POD_HOST:/workspace/data
eval $RS $R/runs/p-a1132b01/ab_grids/ ./addsub-all-layers-01/ab_grids/      # index.html + manifest.js + step_*.js
open ./addsub-all-layers-01/ab_grids/index.html        # file:// works; the applet reads manifest.js
eval $RS $R/runs/p-a1132b01/metrics.jsonl $R/runs/p-a1132b01/launch_config.yaml ./addsub-all-layers-01/
eval $RS $R/ladder/summary.md $R/ladder/'*.log' ./addsub-all-layers-01/ladder/
eval $RS $R/logs/ ./addsub-all-layers-01/logs/
# optional, ~20 GB per step dir (decomposition ~7 GB + training ~13 GB):
eval $RS $R/runs/p-a1132b01/ckpts/<step>/ ./addsub-all-layers-01/ckpts/<step>/
```

`scp -P $POD_PORT -i $KEY -r root@$POD_HOST:/workspace/data/runs/p-a1132b01/ab_grids .` is
the no-rsync equivalent. Pull the whole `ab_grids/` dir, not single files: `index.html`
discovers snapshots through `manifest.js`. Quick status without pulling anything:

```bash
ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tail -2 /workspace/data/logs/addsub-all-layers-01.latest.log | cut -c1-300; ls /workspace/data/runs/p-a1132b01/ckpts'
```

---

## 8. Stop, resume, teardown

**Clean stop** (SIGTERM → the trainer finishes the current step, saves a checkpoint, exits):

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
$REPO/notes/dual_objective/addsub-all-layers/stop_run.sh      # waits; prints the newest ckpt steps
```

Equivalent by hand: `kill -TERM $(cat $DATA_ROOT/pids/p-a1132b01.runner.pid)`. Never
`kill -9` a healthy run — that loses everything since the last save (up to 4000 steps).

**Resume on the same run id** (after a clean stop, a crash, a pod restart, or a watchdog
kill): attach the same network volume to a fresh 8x H100 SXM pod, redo `source env.sh`
(the venv, caches and data are all on the volume; if `/root/.cache/param-decomp` is missing
on the new container, `env.sh` re-creates the symlink), then:

```bash
cd $REPO && git describe --tags --exact-match     # must still print addsub-all-layers-01
cd notes/dual_objective/addsub-all-layers && ./launch_run.sh
```

`launch_run.sh` prints `RESUME: checkpoints present: <steps>`; the log then shows
`resumed from checkpoint step N`. Same optimizers, PPGD sources and step counter come back
(orbax `training` item); the schedules continue from N. Constraints: the config file must be
byte-identical to the pinned `$DATA_ROOT/runs/p-a1132b01/launch_config.yaml` (the trainer
refuses otherwise), and the pod must have 8 GPUs (the mesh is in the config). A resumed run
has no step 0, so the step-0 slow eval does not rerun. If a wedged process from a crash is
still holding HBM, `pd_run.sh` aborts with `GPUs are not idle` — `nvidia-smi
--query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`, wait a minute, relaunch.

**Stop the pod without losing anything:** Runpod → the pod → **Stop** (storage-only billing
for the container disk) or **Terminate** (drops the container disk). The NETWORK volume is a
separate object and survives both; only deleting the volume itself loses data. Everything
that matters is under `/workspace` (step 5).

**At the end:**

1. Pull `ab_grids/`, `metrics.jsonl`, `launch_config.yaml`, `logs/`, `ladder/summary.md`,
   and — if you want the decomposition offline — the final `ckpts/40000/` (~20 GB; the
   `decomposition` item alone, ~7 GB, is what every consumer restores).
2. Delete on the volume: `rm -rf $DATA_ROOT/runs/p-1adde4*` (ladder), `$DATA_ROOT/runs/p-a1132b01/hlo`,
   `$DATA_ROOT/tmp/*`, `$VOLUME/uv-cache`.
3. Terminate the pod. Keep the volume while anything on it is still wanted (it bills per
   GB-month); delete it when the checkpoint has been pulled — the HF cache, venv and XLA
   cache are all re-creatable from steps 2–3.
