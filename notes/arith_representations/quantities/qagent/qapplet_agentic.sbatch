#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -J q-applet-agentic
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/%x-%j.out
# The explorer applet for the agentic run (agentic/run1): lists -> qloop_t<t>_agentic_app.json,
# directions (qdirs.py), applet data in applet_agentic_app/.
set -euo pipefail
export OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
Q=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities
PY=/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python
W=$(mktemp -d); mkdir -p $W/qagent $W/agentic
cp $Q/qagent/*.py $Q/qagent/qapplet.html $W/qagent/; cp $Q/agentic/*.py $W/agentic/
cd $W/agentic && $PY -u applet_prep.py
cd $W/qagent
for t in 1 2 3 4; do $PY -u qdirs.py $t --tag agentic_app; done
$PY -u qapplet.py --tag agentic_app
