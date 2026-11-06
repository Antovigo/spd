#!/usr/bin/env bash
# Run on the cluster. Copies a folder under the pod's home back to the same place here.
#     POD_HOST=<ip> POD_PORT=<port> notes/das/launch/copy_results_from_runpod_pod.sh pd_scratch/das_pilot/feucht_comparison
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${1:?folder under the home directory}"
rsync -rlt -e "ssh -p $POD_PORT -i $HOME/.ssh/runpod -o StrictHostKeyChecking=accept-new" \
  "root@$POD_HOST:/root/$1/" "$HOME/$1/"
