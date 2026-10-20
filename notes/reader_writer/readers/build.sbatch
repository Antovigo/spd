#!/bin/bash
#SBATCH -J readers-app
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/readers_app/%x-%j.out
# sbatch [--export=ALL,LAYERS="18"] build.sbatch — data for the MLP readers applet (build.py), copies index.html.
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
APP=/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/readers_app
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W/build.py" "$FRESH/"
cp "$W/index.html" "$APP/index.html"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u "$FRESH/build.py" ${LAYERS:+--layers $LAYERS}
