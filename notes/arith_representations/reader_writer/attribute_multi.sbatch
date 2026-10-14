#!/bin/bash
#SBATCH -J rw-attr-multi
#SBATCH --gres=gpu:l40:2
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o /mnt/nw/home/a.vigouroux/out/pod-backup/p-ba8b0c08/analysis/reader_writer/logs/%x-%j.out
# LAYERS="3 14 19" sbatch attribute_multi.sbatch — attribute.py on several layers, spread over the
# allocated cards that are actually free (an orphan process of another user can hold one card),
# one layer at a time per card.
set -euo pipefail
FRESH=$(mktemp -d); trap 'rm -rf "$FRESH"' EXIT
rsync -a --exclude __pycache__ /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/param_decomp "$FRESH/"
cp /mnt/nw/home/a.vigouroux/Code/param-decomp/arith_representations/notes/arith_representations/reader_writer/attribute.py "$FRESH/"
export PYTHONPATH=$FRESH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR=$(mktemp -d)
FREE_IDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F', ' '$2 >= 20000 {print $1}')
echo "allocated $CUDA_VISIBLE_DEVICES, free cards: $(echo $FREE_IDS)"
[ -n "$FREE_IDS" ] || { echo "no free card"; exit 1; }
read -ra CARDS <<< "$(echo $FREE_IDS)"
read -ra ALL <<< "${LAYERS:?set LAYERS}"
cd "$FRESH"
for c in "${!CARDS[@]}"; do
  (
    for j in "${!ALL[@]}"; do
      [ $((j % ${#CARDS[@]})) -eq "$c" ] || continue
      echo "card ${CARDS[$c]}: layer ${ALL[$j]}"
      CUDA_VISIBLE_DEVICES=${CARDS[$c]} /mnt/nw/home/a.vigouroux/Code/param-decomp/dual_obj_jax/.venv/bin/python -u attribute.py \
        --layer "${ALL[$j]}" --positions ${POSITIONS:-4 3} > "$TMPDIR/L${ALL[$j]}.log" 2>&1 \
        && echo "layer ${ALL[$j]} done" || echo "layer ${ALL[$j]} FAILED"
      grep -E "^t=|Error|Traceback" "$TMPDIR/L${ALL[$j]}.log" | grep -v "pass 1b" | sed "s/^/  L${ALL[$j]} /"
    done
  ) &
done
wait
