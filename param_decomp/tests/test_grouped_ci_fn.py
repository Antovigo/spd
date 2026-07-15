import pytest
import torch
from torch import nn

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    GroupedGlobalCiConfig,
    assign_ci_fn_groups,
    make_ci_fn_wrapper,
)
from param_decomp.components import make_components


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleDict(dict(a=nn.Linear(8, 8), b=nn.Linear(8, 4))) for _ in range(2)]
        )


_MODULE_TO_C = {"layers.0.a": 5, "layers.0.b": 3, "layers.1.a": 5, "layers.1.b": 3}
_TRANSFORMER_CFG = GlobalSharedTransformerCiConfig(
    d_model=16, n_blocks=1, attn_config=AttnConfig(n_heads=2, max_len=8)
)


def test_assign_ci_fn_groups_requires_exactly_one_match():
    groups = {"g0": ["layers.0.*"], "g1": ["layers.1.*"]}
    assignment = assign_ci_fn_groups(list(_MODULE_TO_C), groups)
    assert assignment == {
        "layers.0.a": "g0",
        "layers.0.b": "g0",
        "layers.1.a": "g1",
        "layers.1.b": "g1",
    }
    with pytest.raises(AssertionError, match="matched groups"):
        assign_ci_fn_groups(list(_MODULE_TO_C), {"g0": ["layers.*"], "g1": ["layers.1.*"]})
    with pytest.raises(AssertionError, match="matched no decomposition target"):
        assign_ci_fn_groups(list(_MODULE_TO_C), {"g0": ["layers.*"], "empty": ["nothing.*"]})


def test_grouped_wrapper_matches_single_global_ci_fn():
    torch.manual_seed(0)
    toy = _Toy().requires_grad_(False)
    components = make_components(toy, _MODULE_TO_C)

    grouped_cfg = GroupedGlobalCiConfig(
        fn_type="global_shared_transformer",
        simple_transformer_ci_cfg=_TRANSFORMER_CFG,
        groups={"g0": ["layers.0.*"], "g1": ["layers.1.*"]},
    )
    wrapper = make_ci_fn_wrapper(toy, _MODULE_TO_C, components, grouped_cfg)
    acts = {name: torch.randn(2, 6, 8) for name in _MODULE_TO_C}
    out = wrapper(acts)
    assert set(out) == set(_MODULE_TO_C)
    for name, c in _MODULE_TO_C.items():
        assert out[name].shape == (2, 6, c)

    single_cfg = GlobalCiConfig(
        fn_type="global_shared_transformer", simple_transformer_ci_cfg=_TRANSFORMER_CFG
    )
    sub_module_to_c = {k: v for k, v in _MODULE_TO_C.items() if k.startswith("layers.0")}
    sub_components = {k: components[k] for k in sub_module_to_c}
    single = make_ci_fn_wrapper(toy, sub_module_to_c, sub_components, single_cfg)

    remapped = {
        "_group_ci_fns.g0." + k.removeprefix("_global_ci_fn."): v
        for k, v in single.state_dict().items()
    }
    missing, unexpected = wrapper.load_state_dict(remapped, strict=False)
    assert not unexpected
    assert all(k.startswith("_group_ci_fns.g1.") for k in missing)

    out_single = single({k: acts[k] for k in sub_module_to_c})
    out_grouped = wrapper(acts)
    for name in sub_module_to_c:
        torch.testing.assert_close(out_grouped[name], out_single[name])
