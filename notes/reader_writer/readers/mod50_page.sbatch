#!/bin/bash
#SBATCH -J mod50-page
#SBATCH -c 4
#SBATCH --mem=24000M
#SBATCH -t 00:30:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/%x-%j.out
# sbatch mod50_page.sbatch — figures and entries.js of the mod 50 notebook page (mod50_page.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer/readers
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
cp "$W"/*.py "$FRESH/"
cd "$FRESH"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u mod50_page.py --out "$W/notebook_mod50"
