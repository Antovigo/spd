# param-decomp-config

Torch-free pydantic config schema for Parameter Decomposition. Depends only on
pydantic, numpy, pyyaml, and annotated-types, so non-torch consumers (e.g. JAX
reimplementations) can validate the same YAML run configs without pulling torch,
transformers, or wandb.
