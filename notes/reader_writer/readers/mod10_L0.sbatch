#!/bin/bash
#SBATCH -J readers-mod10
#SBATCH -c 4
#SBATCH --mem=6000M
#SBATCH -t 00:30:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/readers_app/%x-%j.out
# sbatch [--export=ALL,TAU=0.5] mod10_L0.sbatch — block-0 a mod 10 analysis (mod10_L0.py) into the notebook dir.
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W/mod10_L0.py" "$FRESH/"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u "$FRESH/mod10_L0.py" \
  --out /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/readers_app/notebook ${TAU:+--tau $TAU}
