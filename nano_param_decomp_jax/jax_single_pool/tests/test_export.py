"""CPU tests for the torch-layout exporter (`export.py`).

These pin the pure mapping (key names, V/U destacking, the site-order permutation,
frozen-key rename). The cross-framework numeric proof is the tools pair
(`tools/gen_export_fixture.py` + `tools/verify_export_torch.py`, the latter run in the
torch venv against the real `LMComponentModel` / `GlobalSharedTransformerCiFn`).
"""

import numpy as np
from jax import random
from vendored_jax.llama import LlamaConfig

from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.export import (
    ci_fn_state,
    components_state,
    frozen_target_keys,
    torch_site_order,
)
from jax_single_pool.llama8b import (
    KINDS,
    LayerRange,
    init_decomp_vu,
    llama_site_specs,
    site_name,
)

_C = 6
_ARCH = CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=24)


def _tiny_cfg(n_layer: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=48, n_layer=n_layer, n_head=2, n_kv_head=1, n_embd=16, n_intermediate=32,
        rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=512,
    )  # fmt: skip


def test_torch_site_order_is_lexicographic():
    names = tuple(site_name(layer, kind) for layer in (2, 10) for kind in KINDS)
    # torch sorts module paths as STRINGS: "layers.10..." < "layers.2...", and within a
    # layer (down, gate, up) — not the JAX (gate, up, down).
    assert torch_site_order(names) == (
        "layers.10.mlp.down_proj",
        "layers.10.mlp.gate_proj",
        "layers.10.mlp.up_proj",
        "layers.2.mlp.down_proj",
        "layers.2.mlp.gate_proj",
        "layers.2.mlp.up_proj",
    )


def test_components_state_destacks_per_site():
    layer_range = LayerRange(20, 21)
    vu = init_decomp_vu(_tiny_cfg(22), _C, layer_range.n_layers, random.PRNGKey(0))
    state = components_state(vu, layer_range)

    assert set(state) == {
        f"model.{site_name(layer, kind)}.components.{p}"
        for layer in (20, 21)
        for kind in KINDS
        for p in ("V", "U")
    }
    for layer_idx, layer in enumerate((20, 21)):
        for kind in KINDS:
            V, U = vu.site(layer_idx, kind)
            np.testing.assert_array_equal(
                state[f"model.{site_name(layer, kind)}.components.V"], np.asarray(V)
            )
            np.testing.assert_array_equal(
                state[f"model.{site_name(layer, kind)}.components.U"], np.asarray(U)
            )
    # torch LinearComponents stores V (d_in, C), U (C, d_out).
    assert state["model.layers.20.mlp.gate_proj.components.V"].shape == (16, _C)
    assert state["model.layers.20.mlp.gate_proj.components.U"].shape == (_C, 32)
    assert state["model.layers.20.mlp.down_proj.components.V"].shape == (32, _C)
    assert state["model.layers.20.mlp.down_proj.components.U"].shape == (_C, 16)


def test_ci_fn_state_keys_and_permutation():
    layer_range = LayerRange(20, 21)
    sites = llama_site_specs(_tiny_cfg(22), layer_range, _C)
    ci_fn = init_ci_fn(_ARCH, sites, random.PRNGKey(1))
    state = ci_fn_state(ci_fn, sites)

    prefix = "ci_fn._global_ci_fn"
    expected_keys = {f"{prefix}._input_projector.{p}" for p in ("W", "b")}
    expected_keys |= {f"{prefix}._output_head.{p}" for p in ("W", "b")}
    for i in range(_ARCH.n_blocks):
        expected_keys |= {
            f"{prefix}._blocks.{i}.attn.{name}_proj.weight" for name in ("q", "k", "v", "out")
        }
        expected_keys.add(f"{prefix}._blocks.{i}.attn.rope.inv_freq")
        expected_keys |= {f"{prefix}._blocks.{i}.mlp.{j}.{p}" for j in (0, 2) for p in ("W", "b")}
    assert set(state) == expected_keys

    jax_order = tuple(s.name for s in sites)
    sorted_order = torch_site_order(jax_order)
    assert jax_order != sorted_order

    def block_bounds(order: tuple[str, ...], sizes: dict[str, int], site: str) -> slice:
        offset = sum(sizes[s] for s in order[: order.index(site)])
        return slice(offset, offset + sizes[site])

    d_in = {s.name: s.d_in for s in sites}
    c = {s.name: s.C for s in sites}
    in_proj = np.asarray(ci_fn.in_proj_w)
    out_w = np.asarray(ci_fn.out_w)
    out_b = np.asarray(ci_fn.out_b)
    for site in jax_order:
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._input_projector.W"][block_bounds(sorted_order, d_in, site)],
            in_proj[block_bounds(jax_order, d_in, site)],
        )
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._output_head.W"][:, block_bounds(sorted_order, c, site)],
            out_w[:, block_bounds(jax_order, c, site)],
        )
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._output_head.b"][block_bounds(sorted_order, c, site)],
            out_b[block_bounds(jax_order, c, site)],
        )


def test_ci_fn_state_block_weights_unpermuted_fp32():
    layer_range = LayerRange(18, 18)
    sites = llama_site_specs(_tiny_cfg(19), layer_range, _C)
    ci_fn = init_ci_fn(_ARCH, sites, random.PRNGKey(2))
    state = ci_fn_state(ci_fn, sites)
    for i, block in enumerate(ci_fn.blocks):
        prefix = f"ci_fn._global_ci_fn._blocks.{i}"
        np.testing.assert_array_equal(state[f"{prefix}.attn.q_proj.weight"], np.asarray(block.wq))
        np.testing.assert_array_equal(state[f"{prefix}.attn.out_proj.weight"], np.asarray(block.wo))
        np.testing.assert_array_equal(state[f"{prefix}.mlp.0.W"], np.asarray(block.w1))
        np.testing.assert_array_equal(state[f"{prefix}.mlp.2.b"], np.asarray(block.b2))
        np.testing.assert_array_equal(
            state[f"{prefix}.attn.rope.inv_freq"], np.asarray(ci_fn.inv_freq)
        )
    assert all(v.dtype == np.float32 for v in state.values())


def test_frozen_target_keys_rename():
    keys = frozen_target_keys(n_layer=20, layer_range=LayerRange(18, 18))
    # Decomposed sites: frozen weight renames to the ComponentLinear buffer.
    assert (
        keys["model.layers.18.mlp.gate_proj.target_weight"]
        == "model.layers.18.mlp.gate_proj.weight"
    )
    assert "model.layers.18.mlp.gate_proj.weight" not in keys
    # Non-decomposed layers keep `.weight`; lm_head gains the `model.` prefix.
    assert keys["model.layers.19.mlp.gate_proj.weight"] == "model.layers.19.mlp.gate_proj.weight"
    assert keys["model.lm_head.weight"] == "lm_head.weight"
    assert keys["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    # 1 embed + 20 layers x (2 norms + 4 attn + 3 mlp) + final norm + lm_head.
    assert len(keys) == 1 + 20 * 9 + 2
