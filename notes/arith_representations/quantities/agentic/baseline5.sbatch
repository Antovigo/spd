#!/bin/bash
#SBATCH -J q5-base
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/agentic/run5/baseline/%x-%j.out
# V5 baseline (PLAN_v5.md): run 1's final lists at t = 4 L18.attn and L18.mlp, dense vs sparse,
# exhaustive vs stepwise, for B in 1000, 3000, 10000.
cd /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/quantities/agentic
RUN1=/mnt/nw/home/a.vigouroux/out/pod-backup/p-ba5a0c05/analysis/quantities/agentic/run1
for B in 3000 1000 10000; do for site in L18.attn L18.mlp; do
  QA_B_BITS=$B /mnt/nw/home/a.vigouroux/Code/param-decomp/main/.venv/bin/python qa5.py baseline 4 $site $RUN1/t4/$site/final.py
done; done
