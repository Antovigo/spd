#!/bin/bash
#SBATCH -J reprepo-seed
#SBATCH -c 2
#SBATCH --mem=4000M
#SBATCH -t 00:20:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/%x-%j.out
# sbatch seed.sbatch — create the representation repository and seed it (seed_l0_mod10.py).
set -euo pipefail
W=/mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/reader_writer
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
mkdir -p "$FRESH/readers" "$FRESH/repr_repo"
cp "$W/readers/mod10_L0.py" "$FRESH/readers/"; cp "$W/repr_repo/repo.py" "$W/repr_repo/seed_l0_mod10.py" "$FRESH/repr_repo/"
COMMIT=$(git -C "$W" rev-parse --short HEAD)
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u "$FRESH/repr_repo/seed_l0_mod10.py" --commit "$COMMIT" --replace ${THRESHOLD:+--threshold $THRESHOLD}
cp "$W/repr_repo/plot_planes_raw.py" "$FRESH/repr_repo/"
/mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python -u "$FRESH/repr_repo/plot_planes_raw.py" \
  --out /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/readers_app/notebook
