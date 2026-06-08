"""Launch the Stage 4 multi-process probe: 2 processes, 1 CPU device each."""

import subprocess
import sys

N = 2
procs = []
for pid in range(N):
    p = subprocess.Popen(
        [sys.executable, "stage4_worker.py", "--process_id", str(pid), "--num_processes", str(N)],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )
    procs.append(p)

codes = [p.wait() for p in procs]
print("exit codes:", codes)
sys.exit(max(codes))
