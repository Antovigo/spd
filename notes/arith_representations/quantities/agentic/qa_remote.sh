#!/bin/bash
# Run qa.py on the compute pod (connection in ~/.config/qa_pod.env): same arguments as qa.py.
# The spec file is copied to the pod, qa.py runs there, and the output directory is copied back.
set -euo pipefail
source ~/.config/qa_pod.env
SSH="ssh -i $QA_KEY -p $QA_PORT -o ConnectTimeout=30 -o ServerAliveInterval=30"
HERE=$(cd "$(dirname "$0")" && pwd)
RUN="cd $HERE && HOME=/mnt/nw/home/a.vigouroux /root/venv/bin/python qa.py"
case "$1" in
  sites) $SSH "$QA_HOST" "$RUN sites $2" ;;
  show)
    out=$(realpath -m "$4")
    $SSH "$QA_HOST" "mkdir -p $out && $RUN show $2 $3 $out"
    mkdir -p "$out" && rsync -a -e "$SSH" "$QA_HOST:$out/" "$out/" ;;
  fit)
    spec=$(realpath "$4"); out=$(realpath -m "$5")
    $SSH "$QA_HOST" "mkdir -p $(dirname "$spec") $out"
    rsync -a -e "$SSH" "$spec" "$QA_HOST:$spec"
    $SSH "$QA_HOST" "$RUN fit $2 $3 $spec $out"
    mkdir -p "$out" && rsync -a -e "$SSH" "$QA_HOST:$out/" "$out/" ;;
  *) echo "usage: qa_remote.sh sites <t> | show <t> <site> <dir> | fit <t> <site> <spec.py> <dir>"; exit 1 ;;
esac
