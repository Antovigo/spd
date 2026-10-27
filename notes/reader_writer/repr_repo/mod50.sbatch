#!/bin/bash
#SBATCH -J mod50-l15
#SBATCH -c 8
#SBATCH --mem=40000M
#SBATCH -t 01:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/%x-%j.out
# sbatch mod50.sbatch — upstream writers of L15 g19 / g29's b mod 50 pattern (mod50_l15_writers.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp -r "$W/readers" "$W/repr_repo" "$FRESH/"
cd "$FRESH/repr_repo"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u mod50_l15_writers.py
