"""Self-contained vendoring of Llama-3.1 as a checkpointable, componentizable decomposition
target. Distinct from `pretrain/models/llama_simple.py` (a different, small pretrain-only
architecture — do not conflate). No code shared with the gpt2 vendoring.

Import from the submodules where names are defined:
  - `config`     — `VendoredLlamaConfig`, `Llama3RopeScaling`
  - `model`      — `VendoredLlama` (base arch + `from_hf_pretrained`)
  - `components` — `componentize_llama`, `ComponentLlama`, `ComponentLinear`
"""
