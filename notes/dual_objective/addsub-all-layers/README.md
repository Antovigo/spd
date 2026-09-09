# addsub-all-layers: probe ladders + launcher, for two pod shapes

Companion to the two production configs and their guides. Everything here runs ON THE POD
after `source env.sh`. Read the guide for your pod shape first.

| profile | config | guide | GPUs | batch | run id | trials | ladder dir |
|---|---|---|---|---|---|---|---|
| `8xh100` (default) | `../addsub-all-layers-8xh100-sota.yaml` | `../addsub-all-layers-8xh100-guide.md` | 8 | 256/128 | `p-a1132b01` | `trials/` | `$DATA_ROOT/ladder` |
| `4xh100` | `../addsub-all-layers-4xh100-sota.yaml` | `../addsub-all-layers-4xh100-guide.md` | 4 | 128/128 | `p-b4132c01` | `trials-4xh100/` | `$DATA_ROOT/ladder-4xh100` |

`PROFILES` in `make_trials.py` is the single source of truth for that table; the shell
scripts read it rather than carrying their own copy, and run ids are disjoint so both
ladders can share one `$DATA_ROOT`.

| file | role |
|---|---|
| `env.sh` | the pod environment (`DATA_ROOT` tree, HF/wandb/XLA-cache/tmp on the volume, secrets). Profile-independent. |
| `make_trials.py` | generates `trials*/` + `manifest.tsv` from a production config; `--check` parses and builds everything |
| `run_ladder.sh` | the unattended driver: meshes at the authored batch, auto-escalation to 128/96 if none fit, then the AB-grid and SIGTERM/resume smokes at the winner |
| `summarize_ladder.py` | reads the trials' `metrics.jsonl` + logs into the one markdown table |
| `pd_run.sh` | runs one trainer process with SIGTERM forwarding + hang watchdog (shared by ladder and launcher) |
| `launch_run.sh` | detached launcher for the real run; re-running RESUMES on the profile's fixed run id |
| `stop_run.sh` | clean stop (SIGTERM, checkpoint save, exit) |
| `pull_backup.sh` | periodic off-pod backup, run from the CLUSTER not the pod; mandatory on volume disk, where losing the pod loses everything |

Quick path, 4x H100:

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd /workspace/spd/notes/dual_objective/addsub-all-layers
$VENV_PY make_trials.py --profile 4xh100 --check --data-root "$DATA_ROOT"
./run_ladder.sh --profile 4xh100 --detach && tail -f "$DATA_ROOT/ladder-4xh100/driver.log"
cat "$DATA_ROOT/ladder-4xh100/summary.md"          # paste back
# set runtime.replicate/fsdp (and nontarget batch, if it escalated) in the config, then:
./launch_run.sh --profile 4xh100
```

Drop `--profile 4xh100` throughout for the 8-GPU route; `8xh100` is the default.

`trials*/` are GENERATED. After editing either production config, regenerate rather than
hand-editing a trial: `$VENV_PY make_trials.py --profile <name> --check`.
