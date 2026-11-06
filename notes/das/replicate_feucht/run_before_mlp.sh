#!/usr/bin/env bash
# Run on the pod. DAS before layer 18's MLP at the last token, at their dimensions (their IIA: first operand
# 61.7%, second operand 63.8%, result 87.6%). A separate working folder keeps these PCA features and outputs
# apart from the after-MLP ones, because train_das.py reads PCA features from a fixed relative path.
set -euo pipefail
ROOT=/workspace/arithmetic-wild
MODEL=/workspace/models/Llama-3.1-8B
PY=$ROOT/.venv/bin/python
WRAP=/workspace/with_before_mlp_hook.py
mkdir -p "$ROOT/before_mlp"; cd "$ROOT/before_mlp"
ln -sfn "$ROOT/datasets" datasets
[ -d outputs/pca/Llama-3.1-8B/addition/features ] || \
  "$PY" "$WRAP" "$ROOT/src/collect_pca.py" --task-folder addition --model "$MODEL" --token-positions last_token
for spec in input:16 offset:15 output:16; do
  "$PY" "$WRAP" "$ROOT/src/train_das.py" --task-folder addition --model "$MODEL" \
    --target-variable "${spec%:*}" --layers 18 --token-positions last_token \
    --subspace-dim "${spec#*:}" --init random-pca
done
echo "before-MLP replication finished"
