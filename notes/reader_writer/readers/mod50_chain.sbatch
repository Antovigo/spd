#!/bin/bash
#SBATCH --job-name=mod50-chain
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# sbatch mod50_chain.sbatch — mod50_eq.py compute + plot (3 readers, chain.npz), then mod50_chain.py.
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d -p /ephemeral/$USER 2>/dev/null || mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/mod50_eq.py "$W"/mod50_chain.py "$FRESH/"
cd "$FRESH"
PY=/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python
$PY -u mod50_eq.py all
$PY -u mod50_chain.py
