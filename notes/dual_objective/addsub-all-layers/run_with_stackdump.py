#!/usr/bin/env python
"""Run the targeted trainer with an on-demand stack dump. Diagnostic wrapper, not a launcher.

    $VENV_PY run_with_stackdump.py <config.yaml> <data_root> <n_gpu> --run_id p-xxxxxxxx
    kill -USR1 <pid>      # every thread's Python stack is printed to the log, process continues

WHY. When a run stops making progress the only question that matters is which call it is
sleeping in, and on a container that forbids ptrace (Runpod does) `py-spy` cannot attach —
it fails with "Failed to copy Py_Version symbol: Permission denied". `faulthandler` needs no
ptrace because the process dumps its own stacks from a signal handler, so this works
anywhere. `faulthandler.register` does NOT kill the process, unlike sending SIGABRT.

It changes nothing about the run: same module, same argv, same env-before-jax ordering
(`run_targeted.main` still sets `runtime.launch_env` before anything imports jax, because
this wrapper imports nothing but the standard library first)."""

import faulthandler
import runpy
import signal
import sys

# all_threads=True is the point: the interesting stack is usually a worker or the XLA
# callback thread, not the main one. Output goes to stderr, i.e. into the run's log.
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
print(f"[stackdump] pid {__import__('os').getpid()} — kill -USR1 it for a stack", flush=True)

sys.argv = ["param_decomp.experiments.lm.run_targeted", *sys.argv[1:]]
runpy.run_module("param_decomp.experiments.lm.run_targeted", run_name="__main__")
