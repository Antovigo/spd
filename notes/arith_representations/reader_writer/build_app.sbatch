#!/bin/bash
#SBATCH -J rw-app
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# LAYER=18 sbatch build_app.sbatch — writes the applet data from extract_layer.py + attribute.py outputs.
set -euo pipefail
LAYER="${LAYER:-18}"
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/reader_writer
cp "$D/build_app.py" "$D/app.html" "$FRESH/"
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u build_app.py --layer "$LAYER" --positions ${POSITIONS:-4 3}
