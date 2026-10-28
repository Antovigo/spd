#!/bin/bash
#SBATCH --job-name=mod50-mlp3
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# sbatch mod50_pos3.sbatch — linearity and period interventions of the position-3 MLPs (mod50_mlp3.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d -p /ephemeral/$USER 2>/dev/null || mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/mod50_mlp3.py "$FRESH/"
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u mod50_mlp3.py
