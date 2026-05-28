"""Drop-in replacement for ``python -m param_decomp_lab.experiments.lm.run`` that
installs the debug scaffolding before delegating to the real ``cli``.

Usage (under torchrun):

    torchrun --standalone --nproc_per_node=8 \\
        scripts/repro_3pool_eval_deadlock/run_debug.py path/to/config.yaml

Environment knobs (all optional):
    PD_DEBUG_DIR=/path/for/output     (default: /tmp/pd_3pool_debug/$SLURM_JOB_ID)
    PD_DEBUG_HEARTBEAT_S=10           (0 to disable heartbeat)
    PD_DEBUG_FAULT_TIMEOUT_S=300      (0 to disable auto stack dump on hang)

The actual torch.profiler / memory-profile wiring lives in
``param_decomp_lab/experiments/lm/run.py`` (`_maybe_build_torch_profiler` and
`_maybe_enable_memory_profile`) and is gated by PD_TORCH_PROFILE_RANKS /
PD_MEMORY_PROFILE_RANKS env vars — no monkey-patching needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from debug_scaffolding import install_debug_scaffolding  # noqa: E402

install_debug_scaffolding()

from param_decomp_lab.experiments.lm.run import cli  # noqa: E402

if __name__ == "__main__":
    cli()
