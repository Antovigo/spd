#!/bin/bash
#SBATCH -J mod20-copy
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/track_mod20/%x-%j.out
# sbatch mod20_components.sbatch — copy of the mod-20 circle to "=" (mod20_copy.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/*.py "$FRESH/"
cd "$FRESH"
OMP_NUM_THREADS=16 /mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u -c "import faulthandler, runpy, sys; faulthandler.dump_traceback_later(120, repeat=True); sys.argv=[\"mod20_copy.py\", \"--out\", \"$W/notebook_mod20/fig\"]; runpy.run_path(\"mod20_copy.py\", run_name=\"__main__\")"
