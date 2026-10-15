#!/bin/bash
#SBATCH -J rwv-test
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance/logs/%x-%j.out
# sbatch test.sbatch — synthetic checks of variance.py on CPU (node-local copy of the scripts).
set -euo pipefail
D=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/variance
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$D/variance.py" "$D/test_variance.py" "$FRESH/"
cd "$FRESH"
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/.venv/bin/python -m pytest -q -p no:cacheprovider test_variance.py
