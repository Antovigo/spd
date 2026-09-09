# addsub-all-layers-4xh100-01 on a Runpod 4x H100 SXM pod — step by step

The full-network (32-block, 224-site) targeted dual-objective decomposition of Llama-3.1-8B
from `addsub-all-layers-4xh100-sota.yaml`, run on one Runpod pod with **4x H100 SXM 80 GB**
(NVLink), at batch **128 target / 128 non-target**. Companion scripts: `addsub-all-layers/`
(read its `README.md`). Every command below is meant to be pasted in the order given.
`<...>` are placeholders you fill in.

This is the four-GPU sibling of `addsub-all-layers-8xh100-guide.md`. The two are complete
and independent: pick one and follow it end to end. Every script here takes
`--profile 4xh100`, which selects this config, its own trials directory, its own ladder
directory and its own run ids, so both routes can share one volume without colliding.

**Why this variant, and what it costs.** Runpod prices GPUs linearly, so four H100s cost
$13.96/h against $27.92/h for eight at the same price per unit of compute, and four are
easier to get. The catch is that halving the devices doubles the per-rank batch, and this
codebase has **no gradient accumulation**: `pd.batch_size` is GLOBAL and shards over the data
mesh, so the global batch is the only lever. Carrying the 8-GPU config's 256/128 onto four
cards would put 64 target prompts and 32 broad rows on each rank, which the memory estimate
in step 6a does not support.

So the target batch halves to 128. That is not purely a concession: every run on this line,
the one-block seat and both four-block scale-ups included, trained at `pd.batch_size: 128`.
The 8-GPU file's 256 is the scale-up brief's untested bet that a bigger batch helps at 32
blocks; this variant declines that bet rather than testing it. **Read the two runs as
different experiments** — at equal steps they see different amounts of target data, so do
not compare a curve from this run against one from the 8-GPU config step-for-step.

The non-target batch stays at 128 (32 rows per rank). That is where the memory risk sits,
because the broad stream carries seq 64 against the pool's 5 tokens and therefore dominates
the activation term. If it does not fit, the ladder escalates to 128/96 on its own.

Code: `https://github.com/Antovigo/spd.git`, branch `feature/dual_obj_jax`. The config
parses and builds at its tip (224 sites, 32 CI chunks, `input_dim` 26624, checked on CPU).

One caution rather than a ceremony: the run's `launch_config.yaml` is byte-pinned but the
CODE is not, so if you push to this branch while the run is in flight, its next resume
restores the checkpoint into the new library. Either avoid pushing to it mid-run, or run
`cd $REPO && git log -1` on the pod before resuming so you know what it will pick up.

Order of operations: pod (1) → code (2) → data + weights (3) → secrets (4) → env sanity (5)
→ probe ladder (6a) → **set the mesh winner in the config** → real run (6b) → **start the
off-pod backup (7)** → stop / resume / teardown (8). On volume disk the backup is part of
launching, not part of finishing.

---

## 1. Pod

**GPU:** 4x H100 SXM 80 GB, one node (NVLink). Not PCIe H100s: the whole mesh argument in the
config header assumes NVLink-priced gathers.

**Driver / CUDA:** the venv brings its own CUDA 12.8.1 libraries as wheels (`uv.lock`:
jax 0.10.1, cudnn 9.8.0.87, NCCL >= 2.28); only the HOST DRIVER matters. It must be
**>= r570 (CUDA 12.8)**: `attention_implementation: auto` uses the cuDNN graph API, which is
exactly what failed on the L40 box's older driver. In the Runpod deploy dialog filter by
CUDA version 12.8 or newer. Check on the pod:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # 4 rows, driver >= 570
```

If the pod's driver is r580 or newer, `--extra cuda` still loads (CUDA 12 wheels are
driver-backward-compatible); `--extra cuda13` is the alternative arm (README) and is
untested on this branch — stay on `cuda`.

**Template / image:** any Runpod Ubuntu 22.04 CUDA image works (e.g. the current
`runpod/pytorch:*-cuda12.8.1-*-ubuntu22.04` template); nothing from the image's python is
used — `uv` installs Python 3.12 and the locked venv. Convenient to have: `git`, `curl`, `rsync`
(`apt-get install -y git curl rsync tmux` if missing); none of them are actually required,
see "If the image is missing tools" in step 2 for the fallbacks. See "SSH access" below: you need the DIRECT
connection, not the proxied one, or none of the transfers in steps 3 and 7 will work.

**SSH access — get the direct connection, not the tunnel.** Runpod offers two, and they are
not interchangeable:

| | looks like | carries rsync/scp? |
|---|---|---|
| direct, "SSH over exposed TCP" | `ssh root@<public ip> -p <port> -i <key>` | yes |
| proxied / tunnel | `ssh <pod-id>-<hash>@ssh.runpod.io -i <key>` | **no** |

The proxied form is a restricted relay: it gives you a shell, but `rsync` and `scp` need to
launch a remote program over the connection and it will not do that. So if the command
Runpod shows you points at `ssh.runpod.io`, that is the wrong one for this guide, and
`POD_HOST` / `POD_PORT` come from the direct form instead. To get it, set **Expose TCP
Ports = 22** on the pod (template settings, or Edit Pod), after which the Connect dialog
shows a `root@<ip> -p <port>` command. A key file merely NAMED something like `tunnel` is
fine; only the destination host matters.

**Making a key, if you do not have one.** On the machine that will run the transfers, which
for step 7 is the machine running the backup loop for days:

```bash
ssh-keygen -t ed25519 -C runpod -f ~/.ssh/runpod -N ''     # -N '' = no passphrase
cat ~/.ssh/runpod.pub                                      # paste this into Runpod
```

Leave it passphrase-free, or the unattended backup loop blocks on the first prompt; if you
would rather use a passphrase, load the key into `ssh-agent` and keep the agent alive for
the whole run. Paste the **public** half (`~/.ssh/runpod.pub`, one line starting
`ssh-ed25519`) into Runpod → Settings → SSH Public Keys, and pass the **private** half as
`KEY=~/.ssh/runpod`.

Runpod injects those keys when a pod is CREATED. If your pod is already running and was
created before you added the key, do not recreate it — paste the public line into the pod's
`/root/.ssh/authorized_keys` from the web terminal:

```bash
mkdir -p /root/.ssh && chmod 700 /root/.ssh
echo 'ssh-ed25519 AAAA... runpod' >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
```

Then check from your side before relying on it in step 3:

```bash
ssh -p <port> -i ~/.ssh/runpod root@<ip> 'nvidia-smi -L && which rsync'
```

Both lines must come back. If `rsync` is missing on the pod, the 8-GPU guide's
"If the image is missing tools" section has the tar-over-ssh fallback, which needs only a
shell.

**Container disk:** 30 GB is enough (ephemeral: OS + `/root` + uv's build temp).

**Storage: volume disk, because network volumes are not offered for these GPUs.** Set the
pod's **volume disk** to the size below; it mounts at `/workspace` and everything that must
survive a pod restart lives there. Know exactly what that buys you:

| event | volume disk | network volume |
|---|---|---|
| pod Stop then Start (same pod) | survives | survives |
| pod terminated, preempted, or its host lost | **GONE** | survives |
| move the data to a different pod | **impossible** | just remount |

So the disk is a single point of failure for the whole run, and there is no way to migrate
it. Two consequences, both non-negotiable:

1. **Take the pod on Secure Cloud / on-demand, never Spot or interruptible.** A preemption
   with volume disk destroys the run, the 16 GB weights cache and the venv together.
2. **Back up off the pod continuously**, not at the end — see step 7. `pull_backup.sh` does
   it; it caps a total loss at one checkpoint interval instead of the entire trajectory.

If a network volume becomes available for your GPU type, prefer it: create it in the SAME
datacenter as the pod (a pod can only mount one from its own DC), and the periodic backup
becomes belt-and-braces rather than the only line of defence.

Budget, either way:

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

→ **250 GB** volume disk (200 GB is workable if you delete the ladder's checkpoints before
launching the real run, step 6b).

---

## 2. Code

```bash
cd /workspace
git clone -b feature/dual_obj_jax https://github.com/Antovigo/spd.git spd
cd spd
git log -1 --oneline                            # note this, it is the code you are running
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
source notes/dual_objective/addsub-all-layers/env.sh   # creates the DATA_ROOT tree, sets UV_CACHE_DIR
uv python install 3.12
uv sync --frozen --no-dev --extra cuda          # README "Install": driver r525–r579 arm; ~7 GB
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # 0.10.1, 4x CudaDevice
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
export URL=https://codeload.github.com/Antovigo/spd/tar.gz/refs/heads/feature/dual_obj_jax
curl -L "$URL" -o spd.tar.gz
# or: wget -O spd.tar.gz "$URL"
# or, with neither: python3 -c "import urllib.request as u,os; u.urlretrieve(os.environ['URL'],'spd.tar.gz')"
tar xzf spd.tar.gz && mv spd-feature-dual_obj_jax spd && rm spd.tar.gz
```

Then continue from `cd spd` in the block above, skipping the `git log` line. The run scripts
log `commit=tarball` instead of a hash and are otherwise unaffected. The only cost is that
you cannot ask git which code is running, so note the date you downloaded it.

**`rsync` matters only on the transfer path** (step 3 and step 7): `rsync` spawns a remote
`rsync`, so it must exist on BOTH ends. If the pod has none, `scp -r` usually works
(Runpod runs an OpenSSH server), and if even that fails, tar over ssh needs nothing but a
shell and `tar` on the pod:

```bash
# push a dataset from the cluster to the pod, no rsync and no scp needed
tar cz -C ~/out/datasets fineweb_llama_tok_64 \
  | ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tar xz -C /workspace/data/datasets'
# pull results back the same way
ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tar cz -C /workspace/data/runs/p-b4132c01 ab_grids' | tar xz
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
`python -m param_decomp.experiments.lm.harvest_neuron_ranks --config notes/dual_objective/addsub-all-layers-4xh100-sota.yaml --data_root $DATA_ROOT --out_dir $DATA_ROOT/neuron_ranks/addsub1-100_llama31-8b_unit-energy --local_device_count 1 --layers all --batch_size 128`
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
REPO=$VOLUME/spd                              code @ branch feature/dual_obj_jax, venv at $REPO/.venv
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
$VENV_PY -m param_decomp.experiments.lm.run_targeted <config.yaml> $DATA_ROOT 4 --run_id <p-xxxxxxxx>
```

Note the `4`, not `8`. You never type it: the scripts read it from the config's own mesh and
refuse if the pod has a different number of GPUs.

Sanity before spending GPU time (all CPU-side, ~1 min): parse + build every config against
the pod's data root — resolves the target architecture from the HF snapshot, both datasets'
`meta.json`, and the neuron-ranks artifact's provenance:

```bash
cd $REPO/notes/dual_objective/addsub-all-layers
$VENV_PY make_trials.py --profile 4xh100 --check --data-root $DATA_ROOT
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

### 6a. Probe ladder (several hours, dominated by cold compiles; unattended)

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd $REPO/notes/dual_objective/addsub-all-layers
./run_ladder.sh --profile 4xh100 --detach
tail -f $DATA_ROOT/ladder-4xh100/driver.log          # Ctrl-C detaches from the tail only
```

What runs, in order (each 30 steps with the step-0 slow eval ON, under `timeout -k 600`):

1. `mesh-f4-b128x128` — replicate 1 / fsdp 4 / zero1 at batch 128/128, the natural HSDP
   layout (75 min timeout; cold compile)
2. `mesh-r2f2-b128x128` — replicate 2 / fsdp 2 / zero1, same batch: half the frozen-target
   gathers, bought with more static memory
3. **only if neither fits**, the same two meshes again at 128/96, automatically — "does not
   fit" is a memory answer, and shrinking the broad stream is the lever
4. `smoke-abgrid-<winner>` — the AB-grid smoke at the winning (mesh, batch): `eval.every 10 /
   slow_every 20`, `mean_ci_floor 0.0`, grid `[1,10]^2`, checkpointing ON. Pass =
   `ab_grids/step_20.js` written, `eval/ab_grids/saved_components/total` > 0, no traceback.
5. `smoke-resume-<winner>` — `save_every 20`; the driver SIGTERMs the trainer after the
   step-20 log, expects `SIGTERM: checkpoint saved`, relaunches on the same run id, expects
   `resumed from checkpoint step N` and a clean finish at step 30.

Trial names carry their batch (`<kind>-<mesh>-b<target>x<nontarget>`) because the ladder may
have to fall back, and the smokes must then run at whatever shape actually won. If nothing
fits at either batch the driver says so and stops; the next levers are a smaller non-target
batch, or the eval-only knobs `eval.batch_size` and `PGDReconLoss.n_batches`, which cost
diagnostic precision rather than training semantics.

The static-memory arithmetic the ladder is testing, per rank at `mem_fraction 0.95`:

| mesh | trainable (÷4) | frozen target | grad accumulator | static | left for activations |
|---|---|---|---|---|---|
| `f4` = 1x4 zero1 | 5.6 GB | 4.0 GB (÷4) | 1.9 GB | 11.4 GB | ~64 GB |
| `r2f2` = 2x2 zero1 | 5.6 GB | 8.0 GB (÷2) | 1.9 GB | 15.4 GB | ~60 GB |
| `ddp` = 4x1 (extra rung) | 22.3 GB | 16.0 GB | 7.4 GB | 45.7 GB | ~30 GB |

Under `zero1` the optimizer state shards over the full 4-wide data mesh in every row, so the
meshes differ in how the frozen target is sharded and how often it is gathered, not in
optimizer footprint. `ddp` is generated but never in the default order; run it with
`./run_ladder.sh --profile 4xh100 --only mesh-ddp-b128x128` if you want the number. Never
`zero1` at `fsdp: 1`: the config builder accepts it, so that guard is lore rather than code,
and on a degenerate sharding axis it measured 26 times the all-reduce volume.

Extrapolating the four-block L40 run's roughly 10.5 GB of activations at 28 live sites and
32/24 per rank to 224 sites gives 30 to 60 GB here, depending on how much of that term is
site-proportional. It should fit, but not by much, which is exactly why rung 3 exists.

There is a second pod shape with its own config, guide and ladder: eight H100s at batch
256/128, the default profile `8xh100`. See `addsub-all-layers-8xh100-guide.md`.

Output: `$DATA_ROOT/ladder-4xh100/summary.md` — one table (trial, fits?, peak GB/rank, s/step,
projected hours for 40k steps, wall minutes, notes). Paste it back. Per-trial logs are
`$DATA_ROOT/ladder-4xh100/<trial>.log`; run dirs are `$DATA_ROOT/runs/<run id>` (one fixed id per
trial, DERIVED FROM ITS NAME so the set can change without remapping, in `trials-4xh100/manifest.tsv`). The driver skips any trial with a `$DATA_ROOT/ladder-4xh100/<trial>.rc`;
to redo one, delete that file AND its run dir (fixed run ids: a leftover dir would resume
or refuse on the pinned config).

Cold compiles are silent and LONG: the log shows `targeted run
addsub-all-layers-trial-... | 4 GPU`, then the placement audit, then nothing at all for well
over an hour while XLA compiles at a few hundred percent CPU. Measured 2026-09-09: a 45-min
watchdog fuse killed four healthy trials mid-compile, two different meshes dying at an
identical 48 min — a timer, not a memory limit. The fuse now defaults to 4 h
(`WATCHDOG_FUSE`, env-overridable) and the per-trial `timeout` is 6 h, in the manifest.
Confirm a silent trial is compiling rather than wedged with
`ps -o pid,%cpu,etime,cmd -C python`: a few hundred percent CPU is XLA, near zero is a real
hang.

A compile reaches the XLA cache only when it COMPLETES, so a killed trial caches nothing and
its successor starts from scratch, and every (mesh, batch) pair compiles separately. On a
fresh pod, run the FIRST rung alone — `./run_ladder.sh --profile 4xh100 --only mesh-<m>-<batch>`
— so one compile lands in the cache before committing hours to the rest. Peak memory is read from `train/mem/peak_gb_per_rank`; a trial "fits" only if it
reached step 30 with rc 0 and no traceback in the log.

### 6b. The real run

Before launching — these are the only edits the config ever gets, and they must happen NOW
(the pinned `launch_config.yaml` byte-compares on every resume):

1. `runtime.replicate` / `runtime.fsdp` in `addsub-all-layers-4xh100-sota.yaml` ← the ladder's
   winner (authored as `f4` = `1 / 4`; `r2f2` is `2 / 2`). `sharding: zero1` either way.
   If the ladder escalated, ALSO set `nontarget.batch_size` and `eval.batch_size` to 96, to
   match the shape that actually fit.
2. If the pod driver turned out to be < r570 (it should not, step 1): set BOTH
   `target.attention_implementation` and `decomposition.ci.attention.attention_implementation`
   to `xla`.
3. Re-check: `$VENV_PY make_trials.py --profile 4xh100 --check --data-root $DATA_ROOT` (parses the edited
   sota file too). Commit the edit on the pod or note it — the run dir pins the bytes anyway.
4. Free the volume, dropping the smokes' ~80 GB of checkpoints (`summary.md` and the
   `.log`s keep the evidence):
   `awk -F'\t' '$4 ~ /smoke/ {print $2}' trials-4xh100/manifest.tsv | while read -r id; do rm -rf "$DATA_ROOT/runs/$id"; done`

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd $REPO/notes/dual_objective/addsub-all-layers
./launch_run.sh                                  # run id p-b4132c01, run_name addsub-all-layers-4xh100-01
tail -f $DATA_ROOT/logs/addsub-all-layers-4xh100-01.latest.log
```

First minutes of a healthy launch: `persistent XLA compilation cache: /root/.cache/param-decomp/xla`
(→ the volume via the symlink), `targeted run addsub-all-layers-4xh100-01 | 4 GPU / 1 proc | target B=128
nontarget B=128 seq=64 sites=224 steps=40000`, `target prompt pool: 20000 prompts x 5 positions`,
then silence for the compile (cache hit if the ladder ran the same mesh — same step HLO),
then `[step 0] eval/...` (the slow eval), `checkpoint saved @ step 0`, and `[step 100] ...
train/perf/step_time_s=... train/mem/peak_gb_per_rank=...` every ~100 steps. AB grids appear
at step 4000 (`$DATA_ROOT/runs/p-b4132c01/ab_grids/step_4000.js`). wandb: the run is
`param-decomp-llama/addsub-all-layers-4xh100-01` (id `p-b4132c01`). Close the shell and the laptop;
the runner survives (`setsid nohup`). The hang watchdog fuse is 60 min for the real run
(`WATCHDOG_FUSE=3600`): the trainer logs every 100 steps (~8–10 min), a slow eval + 20 GB
save can be 20–30 min, a cold compile 20–40 min.

**Expected wall clock and cost.** Per-rank work here is the 8-GPU plan's target geometry
(32 prompts x 5 positions) with twice its broad rows, and the broad stream dominates the
step, so expect roughly **6 to 10 s/step**. That is **3 to 5 days** for 40k steps including
the measured 6% eval and checkpoint overhead, so **$950 to $1550** at $13.96/h. The ladder's
own projection, computed from the trial that won, is in `summary.md` and supersedes this.

---

## 7. Retrieval while running — and the continuous backup

**Start this first, before the long run, and leave it running.** On volume disk the pod is a
single point of failure (step 1), so the backup is not optional bookkeeping: it is what
turns "the pod died" from losing the run into losing one checkpoint interval. Run it on the
cluster or your laptop, wherever you have room for a few 20 GB checkpoints:

```bash
cd <your checkout>/notes/dual_objective/addsub-all-layers
POD_HOST=<pod public ip> POD_PORT=<exposed ssh port> KEY=~/.ssh/<your key> \
  nohup ./pull_backup.sh --profile 4xh100 --interval 3600 --dest ~/out/pod-backup \
  > ~/out/pod-backup.log 2>&1 &
tail -f ~/out/pod-backup.log
```

Each hourly pass pulls `metrics.jsonl`, `launch_config.yaml`, the whole `ab_grids/`
directory, the ladder summary and every log, all of which are small, plus the newest
COMPLETE checkpoint if it is newer than the one already backed up. Completeness is decided by
`_CHECKPOINT_METADATA` inside the step directory, which orbax writes at finalize, so a
checkpoint mid-write is skipped rather than half-copied. `--keep 2` prunes older local
copies. At `save_every 4000` a new checkpoint appears every 8 to 11 hours, so an hourly
interval mostly transfers a few megabytes and occasionally 20 GB.

**Restoring onto a fresh pod.** Re-stage code, datasets, weights and secrets per steps 2 to
4, then push the run directory back and launch:

```bash
rsync -avP -e "ssh -p $POD_PORT -i $KEY" \
  ~/out/pod-backup/p-b4132c01/launch_config.yaml \
  ~/out/pod-backup/p-b4132c01/ckpts \
  root@$POD_HOST:/workspace/data/runs/p-b4132c01/
# then on the pod: ./launch_run.sh --profile 4xh100
```

`launch_run.sh` prints `RESUME: checkpoints present:` and the trainer logs `resumed from
checkpoint step N`. The pinned `launch_config.yaml` you push back IS the byte-compare
reference, so it must be the one from the backup, not a fresh copy of the source config.

### Ad-hoc pulls

From your laptop (or the cluster). The run dir is `/workspace/data/runs/p-b4132c01/`:

```bash
POD_HOST=<pod public ip>; POD_PORT=<exposed ssh port>; KEY=~/.ssh/<your key>
RS="rsync -avP -e 'ssh -p $POD_PORT -i $KEY'"
R=root@$POD_HOST:/workspace/data
eval $RS $R/runs/p-b4132c01/ab_grids/ ./addsub-all-layers-4xh100-01/ab_grids/      # index.html + manifest.js + step_*.js
open ./addsub-all-layers-4xh100-01/ab_grids/index.html        # file:// works; the applet reads manifest.js
eval $RS $R/runs/p-b4132c01/metrics.jsonl $R/runs/p-b4132c01/launch_config.yaml ./addsub-all-layers-4xh100-01/
eval $RS $R/ladder-4xh100/summary.md $R/ladder-4xh100/'*.log' ./addsub-all-layers-4xh100-01/ladder/
eval $RS $R/logs/ ./addsub-all-layers-4xh100-01/logs/
# optional, ~20 GB per step dir (decomposition ~7 GB + training ~13 GB):
eval $RS $R/runs/p-b4132c01/ckpts/<step>/ ./addsub-all-layers-4xh100-01/ckpts/<step>/
```

`scp -P $POD_PORT -i $KEY -r root@$POD_HOST:/workspace/data/runs/p-b4132c01/ab_grids .` is
the no-rsync equivalent. Pull the whole `ab_grids/` dir, not single files: `index.html`
discovers snapshots through `manifest.js`. Quick status without pulling anything:

```bash
ssh -p $POD_PORT -i $KEY root@$POD_HOST 'tail -2 /workspace/data/logs/addsub-all-layers-4xh100-01.latest.log | cut -c1-300; ls /workspace/data/runs/p-b4132c01/ckpts'
```

---

## 8. Stop, resume, teardown

**Clean stop** (SIGTERM → the trainer finishes the current step, saves a checkpoint, exits):

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
$REPO/notes/dual_objective/addsub-all-layers/stop_run.sh --profile 4xh100   # waits; prints the newest ckpt steps
```

Equivalent by hand: `kill -TERM $(cat $DATA_ROOT/pids/p-b4132c01.runner.pid)`. Never
`kill -9` a healthy run — that loses everything since the last save (up to 4000 steps).

**Resume on the same run id** (after a clean stop, a crash, a pod restart, or a watchdog
kill) **on the same pod**, whose volume disk still holds the venv, caches, data and
checkpoints: Start it, `source env.sh` (which re-creates the `/root/.cache/param-decomp`
symlink if the container was rebuilt), then:

```bash
cd $REPO && git log -1 --oneline                  # the code this resume will run
cd notes/dual_objective/addsub-all-layers && ./launch_run.sh --profile 4xh100
```

`launch_run.sh` prints `RESUME: checkpoints present: <steps>`; the log then shows
`resumed from checkpoint step N`. Same optimizers, PPGD sources and step counter come back
(orbax `training` item); the schedules continue from N. Constraints: the config file must be
byte-identical to the pinned `$DATA_ROOT/runs/p-b4132c01/launch_config.yaml` (the trainer
refuses otherwise), and the pod must have 4 GPUs (the mesh is in the config). A resumed run
has no step 0, so the step-0 slow eval does not rerun. If a wedged process from a crash is
still holding HBM, `pd_run.sh` aborts with `GPUs are not idle` — `nvidia-smi
--query-compute-apps=pid --format=csv,noheader | xargs -r kill -9`, wait a minute, relaunch.

**Stopping the pod.** Runpod → the pod → **Stop** keeps the volume disk and bills it while
idle, so `/workspace` is intact when you Start it again and `launch_run.sh` resumes. **Do
not Terminate until everything is off the pod**: terminating destroys the volume disk with
the container, and unlike a network volume there is nothing left to remount. Check your
backup first (step 7): `ls $DEST/<run id>/ckpts` should show a recent step.

**At the end:**

1. Run one final `./pull_backup.sh --profile 4xh100 --once`, then confirm
   `ls ~/out/pod-backup/p-b4132c01/ckpts` shows step 40000. The hourly loop already holds
   `ab_grids/`, `metrics.jsonl`, `launch_config.yaml`, the logs and
   `ladder-4xh100/summary.md`; this last pass is what gets the FINAL checkpoint (~20 GB; the
   `decomposition` item alone, ~7 GB, is what every consumer restores). Stop the backup loop
   afterwards.
2. Delete on the volume: every ladder trial's run dir,
   `awk -F'\t' '!/^#/ {print $2}' trials-4xh100/manifest.tsv | while read -r id; do rm -rf "$DATA_ROOT/runs/$id"; done`, plus `$DATA_ROOT/runs/p-b4132c01/hlo`,
   `$DATA_ROOT/tmp/*`, `$VOLUME/uv-cache`.
3. Terminate the pod, which destroys the volume disk with it. Do this ONLY after step 1
   confirms the final checkpoint is off the pod: unlike a network volume there is nothing
   left to keep or remount. The HF cache, venv and XLA cache are all re-creatable from
   steps 2 and 3, so they are not worth pulling.
