"""The LM composition's read side of the neuron-ranking artifact (SPEC T13): resolve the
run's `pd.neuron_ranks`, check its provenance against THIS run's target and prompt pool,
and turn the ranking into the engine's `NeuronAlignment`. The write side is
`neuron_ranks_harvest`."""

import json
from pathlib import Path

import numpy as np

from param_decomp.core.components import NeuronAlignment
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    LMTargetedRun,
    TargetConfig,
)
from param_decomp.targets.glu_transformer import GLUDecomposedModel
from param_decomp.targets.neuron_alignment import (
    NeuronAlignMode,
    alignment_coverage,
    assert_neuron_ranks_provenance,
    mlp_blocks_of,
    neuron_alignment_from_ranks,
    read_neuron_ranks,
    resolve_neuron_ranks_ref,
    wraps_per_component,
)

NEURON_ALIGNMENT_FILENAME = "neuron_alignment.json"


def target_identity(target: AnyLMTargetConfig) -> str:
    """The string an artifact's `meta.target` must equal: the HF model name, or the
    lab-pretrained target's run path."""
    match target:
        case TargetConfig(model_name=model_name):
            return model_name
        case LlamaSimpleMLPTargetConfig(pretrain_run_path=run_path):
            return run_path


def load_neuron_alignment(
    built: LMTargetedRun,
    model: PlacedModel,
    pool_tokens: np.ndarray,
    data_root: Path,
    *,
    mode: NeuronAlignMode,
    write_summary_to: Path | None,
) -> NeuronAlignment:
    """A neuron-aligned run's alignment from its artifact, provenance-checked. Cheap (a
    file read), so every entry — fresh, requeue, fine-tune — takes it; a restore simply
    overwrites the aligned reference. `write_summary_to` (rank 0's run dir) gets
    `neuron_alignment.json`: the artifact, the mode, each site's C, the write-energy
    fraction its top-C covers and (`wrap`) how many neurons its fullest slot sums."""
    assert built.pd.neuron_ranks is not None, "validated at parse"
    glu = model.model
    assert isinstance(glu, GLUDecomposedModel), (
        f"a neuron-aligned init needs a transformer target with MLP blocks, got {type(glu)}"
    )
    artifact_dir = resolve_neuron_ranks_ref(built.pd.neuron_ranks, data_root)
    ranks = read_neuron_ranks(artifact_dir)
    blocks = tuple(mlp_blocks_of(glu.sites, glu.anatomy))
    assert blocks, "neuron-aligned init: the decomposition has no MLP site to align"
    assert_neuron_ranks_provenance(ranks.meta, target_identity(built.target), pool_tokens, blocks)
    alignment = neuron_alignment_from_ranks(ranks, glu.sites, glu.anatomy, mode)
    coverage = alignment_coverage(ranks, glu.sites, glu.anatomy)
    cs = {spec.name: spec.C for spec in glu.sites if spec.name in alignment}
    wraps = {
        name: wraps_per_component(ranks.meta.n_neurons, cs[name]) if mode == "wrap" else 1
        for name in alignment
    }
    summary = {
        "artifact": str(artifact_dir),
        "mode": mode,
        "meta": ranks.meta.model_dump(),
        "sites": {
            name: {
                "C": cs[name],
                "top_c_covered_write_energy": coverage[name],
                "neurons_per_component_max": wraps[name],
            }
            for name in alignment
        },
    }
    print(
        f"neuron-aligned init ({mode}): "
        + ", ".join(
            f"{name} C={cs[name]} top-C covers {coverage[name]:.3f} wraps={wraps[name]}"
            for name in alignment
        ),
        flush=True,
    )
    if write_summary_to is not None:
        write_summary_to.mkdir(parents=True, exist_ok=True)
        (write_summary_to / NEURON_ALIGNMENT_FILENAME).write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return alignment
