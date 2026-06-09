"""Put the sibling `jax_spike/` dir on sys.path so `vendored_jax` (the bit-parity JAX
Llama the llama8b target wraps) imports in tests without a separate install."""

import sys
from pathlib import Path

_jax_spike = Path(__file__).resolve().parents[3] / "jax_spike"
if _jax_spike.is_dir() and str(_jax_spike) not in sys.path:
    sys.path.insert(0, str(_jax_spike))
