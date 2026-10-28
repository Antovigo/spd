#!/bin/bash
#SBATCH --job-name=mod50-pos3
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# sbatch mod50_pos3.sbatch — attribution of the main block-15 v components at position 3 (mod50_pos3.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d -p /ephemeral/$USER 2>/dev/null || mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/mod50_pos3.py "$FRESH/"
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u mod50_pos3.py
