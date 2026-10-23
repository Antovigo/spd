#!/bin/bash
#SBATCH -J copy-figs
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 04:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/copy_mod10/%x-%j.out
# sbatch copy_figs.sbatch — attention patterns and copy geometry (copy_figs.py) into the notebook dir.
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/*.py "$FRESH/"
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u copy_figs.py --out "$W/notebook"
