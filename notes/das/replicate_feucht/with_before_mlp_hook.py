"""Run one of Feucht et al.'s scripts with `ResidualStream` hooked on the stream before the MLP's norm, the
input of `post_attention_layernorm`, instead of the block's output. Their README says this site needs
"some hacking", and their repo does not include the hack.

    python with_before_mlp_hook.py <script.py> [arguments...]
"""

import runpy
import sys
from pathlib import Path

import pyvene.models.constants as pyvene_constants
import pyvene.models.intervenable_modelcard as modelcard
from causalab.neural import LM_units

COMPONENT = "post_attn_resid"

for model_type, modules in modelcard.type_to_module_mapping.items():
    if "Llama" in model_type.__name__:
        prefix = modules["block_output"][0]
        modules[COMPONENT] = (
            f"{prefix}.post_attention_layernorm",
            pyvene_constants.CONST_INPUT_HOOK,
        )
        modelcard.type_to_dimension_mapping[model_type][COMPONENT] = ("hidden_size",)

original_init = LM_units.ResidualStream.__init__


def init_before_mlp(self, layer, token_indices, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    original_init(self, layer, token_indices, **kwargs)
    if kwargs.get("target_output", False):
        self.component_type = COMPONENT


LM_units.ResidualStream.__init__ = init_before_mlp

script = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
