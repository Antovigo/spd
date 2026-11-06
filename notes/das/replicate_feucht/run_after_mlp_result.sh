#!/usr/bin/env bash
# Run on the pod. Feucht et al.'s scripts, unmodified: the addition dataset, PCA features at the last token,
# and DAS on the result after layer 18's MLP, at 16 dimensions (their IIA: 96.6%).
set -euo pipefail
cd /workspace/arithmetic-wild
MODEL=/workspace/models/Llama-3.1-8B
PY=.venv/bin/python
[ -f datasets/Llama-3.1-8B/addition/filtered_dataset.json ] || \
  (cd src && ../$PY generate_dataset.py --task addition --model "$MODEL" --output-dir ../datasets/)
$PY -c "import json; d=json.load(open('datasets/Llama-3.1-8B/addition/filtered_dataset.json')); print('dataset rows', len(d))"
[ -d outputs/pca/Llama-3.1-8B/addition/features ] || \
  (cd src && ../$PY collect_pca.py --task-folder addition --model "$MODEL" --token-positions last_token \
     --dataset-path ../datasets/ --output-dir ../outputs/pca/)
$PY src/train_das.py --task-folder addition --model "$MODEL" --target-variable output --layers 18 \
  --token-positions last_token --subspace-dim 16 --init random-pca
echo "replication step finished"
