# addsub-all-layers: probe ladder + launcher

Companion to `../addsub-all-layers-sota.yaml` (the run config) and
`../addsub-all-layers-guide.md` (pod setup, launch, retrieval, resume, teardown — read that
first). Everything here runs ON THE POD after `source env.sh`.

| file | role |
|---|---|
| `env.sh` | the pod environment (`DATA_ROOT` tree, HF/wandb/XLA-cache/tmp on the volume, secrets) |
| `make_trials.py` | generates `trials/*.yaml` + `trials/manifest.tsv` from the sota config; `--check` parses everything |
| `trials/` | the generated probe configs (never hand-edited; regenerate after any sota change) |
| `run_ladder.sh` | the unattended driver: mesh-c, mesh-b, AB-grid smoke, SIGTERM/resume smoke → `$DATA_ROOT/ladder/summary.md` |
| `summarize_ladder.py` | reads the trials' `metrics.jsonl` + logs into the one markdown table |
| `pd_run.sh` | runs one trainer process with SIGTERM forwarding + hang watchdog (shared by ladder and launcher) |
| `launch_run.sh` | detached launcher for the real run, fixed run id `p-a1132b01`; re-running RESUMES |
| `stop_run.sh` | clean stop (SIGTERM → checkpoint save → exit) |

Quick path:

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd /workspace/spd/notes/dual_objective/addsub-all-layers
$VENV_PY make_trials.py --check --data-root "$DATA_ROOT"   # parses + builds every config
./run_ladder.sh --detach && tail -f "$DATA_ROOT/ladder/driver.log"
cat "$DATA_ROOT/ladder/summary.md"                          # paste back
# set runtime.{replicate,fsdp} in ../addsub-all-layers-sota.yaml to the winner, then:
./launch_run.sh && tail -f "$DATA_ROOT/logs/addsub-all-layers-01.latest.log"
```
