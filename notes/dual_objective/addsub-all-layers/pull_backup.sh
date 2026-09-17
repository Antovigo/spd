#!/usr/bin/env bash
# Periodic off-pod backup of a running decomposition. Runs on the CLUSTER or your laptop,
# NOT on the pod — it pulls, so it needs nothing installed pod-side beyond the SSH server
# Runpod already gives you.
#
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./pull_backup.sh --profile 4xh100
#     POD_HOST=... ./pull_backup.sh --profile 4xh100 --once        # one pass, then exit
#
# WHY THIS EXISTS. On a pod with only VOLUME DISK (no network volume), `/workspace` survives
# a Stop/Start of that same pod but is destroyed when the pod is terminated, preempted, or
# lost with its host. There is no way to move the disk to another pod. So an unbacked 3-5 day
# run has a single point of failure for its entire trajectory. This caps the loss at one
# checkpoint interval instead of the whole run.
#
# Each pass pulls, into $DEST/<run id>/:
#   * metrics.jsonl, launch_config.yaml, neuron_alignment.json, the ab_grids/ directory, the
#     ladder summary and every log — all small, every pass
#   * the newest COMPLETE checkpoint, if it is newer than the newest one already here.
#     "Complete" = the directory is named with the STEP NUMBER ALONE. Orbax writes into
#     `<step>.orbax-checkpoint-tmp/` and RENAMES at finalize, so the name is the only honest
#     signal. `_CHECKPOINT_METADATA` is NOT one: orbax writes it inside the tmp directory
#     before finalizing (observed on p-ba6a0c06 2026-09-17, which is what this guard used to
#     test and why a mid-write checkpoint got pulled).
#     53 GB per checkpoint at 32 blocks (MEASURED on p-ba5a0c05: decomposition 10 GB +
#     training 43 GB); --keep prunes older ones locally, so --keep 2 is ~106 GB.
#
# To resume from a backup on a FRESH pod: re-stage code, datasets, weights and secrets per
# the guide, then push `<run id>/{launch_config.yaml,ckpts/<step>/}` back to
# $DATA_ROOT/runs/<run id>/ on the new pod and run launch_run.sh. The config must be
# byte-identical (it is: launch_config.yaml is the pinned copy) and the mesh is in it, so the
# new pod needs the same GPU count.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE=4xh100; INTERVAL=3600; ONCE=0; KEEP=2
DEST="${DEST:-$HOME/out/pod-backup}"
REMOTE_DATA="${REMOTE_DATA:-/workspace/data}"
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift ;;
    --interval) INTERVAL="$2"; shift ;;
    --dest) DEST="$2"; shift ;;
    --keep) KEEP="$2"; shift ;;
    --once) ONCE=1 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
: "${POD_HOST:?set POD_HOST}" "${POD_PORT:?set POD_PORT}" "${KEY:?set KEY (path to your ssh key)}"

# Ask make_trials.py for this profile's run id, so the mapping has ONE definition.
PY="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-$("$PY" -c "
import sys; sys.path.insert(0, '$HERE')
from make_trials import PROFILES
print(PROFILES['$PROFILE']['run_id'])")}"
SSH=(ssh -p "$POD_PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
REMOTE="root@$POD_HOST"
RUN_REMOTE="$REMOTE_DATA/runs/$RUN_ID"
OUT="$DEST/$RUN_ID"
mkdir -p "$OUT"

# Mirror the pod's `runs/by-name/<run_name> -> ../<run id>` convention into $DEST, so the
# backups are browsable by the name you launched under and not just by opaque run id. The
# name comes from the pinned launch_config.yaml we just pulled, so it is the run's OWN name
# rather than whatever the generator currently emits.
link_by_name() {
  local name
  name=$(sed -n 's/^run_name: *//p' "$OUT/launch_config.yaml" 2>/dev/null | head -1) || return 0
  [ -n "$name" ] || return 0
  mkdir -p "$DEST/by-name"
  ln -sfn "../$RUN_ID" "$DEST/by-name/$name"
}

pass_once() {
  echo "=== $(date -Is) backup pass: $REMOTE:$RUN_REMOTE -> $OUT ==="
  # 1. the small artifacts. `|| true` throughout: a pass that fails because the pod is
  # briefly unreachable must not kill the loop.
  rsync -az --partial -e "${SSH[*]}" \
    --include='metrics.jsonl' --include='launch_config.yaml' --include='neuron_alignment.json' \
    --include='ab_grids/' --include='ab_grids/**' --exclude='*' \
    "$REMOTE:$RUN_REMOTE/" "$OUT/" || { echo "small-artifact pull failed"; return 0; }
  rsync -az --partial -e "${SSH[*]}" "$REMOTE:$REMOTE_DATA/logs/" "$OUT/logs/" || true
  link_by_name
  for L in ladder ladder-4xh100; do
    rsync -az --partial -e "${SSH[*]}" --include='summary.md' --include='*.log' --include='*.rc' \
      --exclude='*' "$REMOTE:$REMOTE_DATA/$L/" "$OUT/$L/" 2>/dev/null || true
  done

  # 2. the newest COMPLETE checkpoint, if we do not already have it.
  local newest have
  # Digits-only basename: skips `<step>.orbax-checkpoint-tmp`, the in-flight save.
  newest=$("${SSH[@]}" "$REMOTE" "for d in $RUN_REMOTE/ckpts/*/; do b=\$(basename \$d); case \$b in ''|*[!0-9]*) continue;; esac; [ -f \"\$d/_CHECKPOINT_METADATA\" ] && echo \$b; done | sort -n | tail -1" 2>/dev/null || true)
  if [ -z "$newest" ]; then echo "no complete checkpoint on the pod yet"; return 0; fi
  have=$(ls "$OUT/ckpts" 2>/dev/null | sort -n | tail -1 || true)
  if [ "$newest" = "${have:-}" ]; then echo "newest checkpoint $newest already backed up"; return 0; fi
  echo "pulling checkpoint step $newest (~53 GB; have ${have:-none})"
  mkdir -p "$OUT/ckpts"
  if rsync -a --partial --info=progress2 -e "${SSH[*]}" \
       "$REMOTE:$RUN_REMOTE/ckpts/$newest" "$OUT/ckpts/"; then
    echo "checkpoint $newest backed up"
    # prune older local copies, newest $KEEP kept
    ls "$OUT/ckpts" | sort -n | head -n -"$KEEP" | while read -r old; do
      echo "pruning local checkpoint $old"; rm -rf "${OUT:?}/ckpts/$old"
    done
  else
    echo "checkpoint pull failed; leaving the partial for the next pass to resume"
  fi
}

echo "backup: run $RUN_ID (profile $PROFILE) -> $OUT ; interval ${INTERVAL}s ; keep $KEEP checkpoints"
while true; do
  pass_once
  [ "$ONCE" -eq 1 ] && break
  echo "=== sleeping ${INTERVAL}s ==="; sleep "$INTERVAL"
done
echo "=== done $(date -Is) ; local state: ==="
du -sh "$OUT" 2>/dev/null; ls "$OUT/ckpts" 2>/dev/null | tr '\n' ' '; echo
