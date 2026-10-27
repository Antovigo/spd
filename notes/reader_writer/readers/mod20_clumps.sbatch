#!/bin/bash
#SBATCH -J mod20-clumps
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/track_mod20/%x-%j.out
# sbatch mod20_components.sbatch — dimension of the four quarter clumps (mod20_clumps.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/*.py "$FRESH/"
cd "$FRESH"
OMP_NUM_THREADS=16 /mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u -c "import faulthandler, runpy, sys; faulthandler.dump_traceback_later(120, repeat=True); sys.argv=[\"mod20_clumps.py\"]; runpy.run_path(\"mod20_clumps.py\", run_name=\"__main__\")"
