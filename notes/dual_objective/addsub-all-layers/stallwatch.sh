#!/bin/bash
# stallwatch.sh <pid> <logfile>  — sample CPU every 60 s; report a stall as soon as
# the process burns < 60 ticks (0.6 s of CPU) over a 5-minute window, and dump its stack.
pid=$1; log=$2
ticks() { awk '{print $14+$15}' /proc/$pid/stat 2>/dev/null; }
prev=$(ticks); [ -z "$prev" ] && { echo "GONE at start"; exit 1; }
win=0; used=0
while sleep 60; do
  cur=$(ticks)
  if [ -z "$cur" ]; then
    echo "=== EXITED after $((win)) min ==="
    grep -vE cuda_vmm_allocator "$log" | tail -25; exit 0
  fi
  used=$((used + cur - prev)); prev=$cur; win=$((win+1))
  if grep -qE "^step |\"step\"|Traceback|RESOURCE_EXHAUSTED" "$log" 2>/dev/null; then
    echo "=== PROGRESS: step/error line at $win min ==="
    grep -vE cuda_vmm_allocator "$log" | tail -25; exit 0
  fi
  if [ $((win % 5)) -eq 0 ]; then
    echo "[$(date -u +%H:%M)] min=$win cpu_ticks_last_5min=$used"
    if [ "$used" -lt 60 ]; then
      echo "=== STALL: $used ticks in 5 min — dumping stack ==="
      kill -USR1 $pid; sleep 5
      grep -vE cuda_vmm_allocator "$log" | tail -30; exit 2
    fi
    used=0
  fi
done
# Why CPU and not log age: a cold XLA compile is minutes of silent, single-threaded work,
# so a silent log proves nothing. CPU ticks separate "compiling" from "hung" — the 2026-09-10
# cuDNN-attention hang sat at ~1.7% CPU for hours while pd_run.sh's log-age watchdog and a
# human both read it as a long compile.
