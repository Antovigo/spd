"""The layerwise per-site MLP CI fn — the sibling of `ci_fn.py`'s transformer for
positionless (`leading_axes=()`) targets like TMS.

One independent MLP per decomposed site maps that site's clean input `[*leading, d_in]`
to `[*leading, C]` pre-squash logits; the SAME logits feed the shared
`lower_leaky_hard` (recon/PPGD masks) and `upper_leaky_hard` (importance-minimality)
squashings (SPEC S5/S6), exactly as the transformer CI fn. Params are fp32 masters
(SPEC N1); the trainer casts for bf16 compute.

`expects_axes = ()` (no position axes): the MLP is applied independently over every
leading cell, so it places no structural constraint on the leading prefix — it works for
any `leading_axes` that the paired model declares empty. (The transformer CI fn, by
contrast, applies RoPE over a `sequence` axis and so declares `expects_axes=("sequence",)`.)

This is the vector-input per-site MLP: each site's MLP consumes the full `[*leading, d_in]`
site input and emits `C` logits (torch `VectorMLPCiFn` shape; see the JAX TMS CLAUDE
note on why this is the chosen `fn_type=mlp` realization rather than torch's scalar
`get_component_acts(x)=x@V` coupling, which would change the generic `ci_fn(site_inputs)`
contract).

This module also holds the **global** CI arch (`GlobalMLPCIFn`, torch
`GlobalSharedMLPCiFn`): ONE shared MLP over ALL sites jointly — per-site inputs are
concatenated in canonical site order, one MLP maps `Σ d_in -> Σ C`, and the output is
split back per site. (The layerwise `shared_mlp` fn_type is functionally the current
`LayerwiseMLPCIFn`; see `MLPCIArch` for why.)

Wiring `GlobalMLPCIArch` into `run_state.init_train_state`'s CI-fn dispatch and the
`config.py` arch builder is a LAB-side follow-up (both files are core, owned elsewhere):
this module only provides the building blocks (`GlobalMLPCIArch`, `GlobalMLPCIFn`,
`init_global_mlp_ci_fn`), which satisfy the same `__call__(site_inputs) -> CIValues`
interface as `LayerwiseMLPCIFn` / `CIFn` and are tested in isolation."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.ci_fn import CIValues, lower_leaky_hard_sigmoid, upper_leaky_hard_sigmoid
from jax_single_pool.lm import SiteSpec


@dataclass(frozen=True)
class MLPCIArch:
    """Hidden widths shared by every per-site MLP (torch `LayerwiseCiConfig.hidden_dims`).

    Covers BOTH `fn_type=mlp` and `fn_type=shared_mlp`: for the vector-input per-site port
    they are the SAME architecture. Torch's `VectorSharedMLPCiFn` ("shared") shares the
    hidden layers across a site's `C` output components and uses a per-component-count head
    `Linear(hidden, C)` — which is exactly `SiteMLP` here (one shared hidden stack, one
    `C`-wide head per site). Torch's per-component scalar `mlp` differs only by feeding
    `x@V` per component; the JAX port already collapses that to the vector-input form (see
    the module docstring), so `mlp` and `shared_mlp` build the identical `LayerwiseMLPCIFn`.
    The distinction that DOES change the architecture is `global` (`GlobalMLPCIFn`)."""

    hidden_dims: tuple[int, ...]


class SiteMLP(eqx.Module):
    """One site's MLP: `hidden_dims` ReLU-init Linear+GELU layers then a linear head to C.

    Matches torch `VectorMLPCiFn` layer structure: each hidden layer is Kaiming-`relu`
    (`gain √2`) initialized with zero bias, the final head is linear-gain (`1`)."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        n_hidden = len(self.weights) - 1
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            if layer_idx < n_hidden:
                x = jax.nn.gelu(x, approximate=False)
        return x


class LayerwiseMLPCIFn(eqx.Module):
    """A per-site MLP bundle behind the same dict-in / `CIValues`-out interface as the
    transformer `CIFn` (`__call__(site_inputs) -> CIValues`)."""

    site_mlps: dict[str, SiteMLP]
    site_names: tuple[str, ...] = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def site_logits(self, site_inputs: dict[str, Array]) -> dict[str, Array]:
        assert set(site_inputs) == set(self.site_names), (
            f"site_inputs keys {sorted(site_inputs)} != CI fn sites {sorted(self.site_names)}"
        )
        return {name: self.site_mlps[name](site_inputs[name]) for name in self.site_names}

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues:
        logits = self.site_logits(site_inputs)
        return CIValues(
            lower={name: lower_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
            upper={name: upper_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
        )


def _init_mlp_stack(dims: tuple[int, ...], key: PRNGKeyArray) -> SiteMLP:
    """One `Linear+GELU` stack mapping `dims[0] -> ... -> dims[-1]`: Kaiming `relu`-gain
    (`√2`) on the hidden layers (matching torch `init_param_` fan-in init), linear gain
    (`1`) on the final head, zero biases."""
    relu_gain = 2.0**0.5
    layer_keys = jax.random.split(key, len(dims) - 1)
    weights: list[Array] = []
    biases: list[Array] = []
    for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
        weights.append(jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5))
        biases.append(jnp.zeros((d_out,)))
    return SiteMLP(weights=weights, biases=biases)


def init_layerwise_mlp_ci_fn(
    arch: MLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    site_mlps = {
        spec.name: _init_mlp_stack(
            (spec.d_in, *arch.hidden_dims, spec.C), jax.random.fold_in(key, site_idx)
        )
        for site_idx, spec in enumerate(sites)
    }
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps, site_names=tuple(s.name for s in sites), expects_axes=()
    )


@dataclass(frozen=True)
class GlobalMLPCIArch:
    """Hidden widths of the single global MLP shared across ALL sites (torch
    `GlobalSharedMlpCiConfig.hidden_dims`)."""

    hidden_dims: tuple[int, ...]


class GlobalMLPCIFn(eqx.Module):
    """ONE shared MLP over all sites (torch `GlobalSharedMLPCiFn`), behind the same
    dict-in / `CIValues`-out interface as `LayerwiseMLPCIFn` / `CIFn`.

    Per-site inputs are concatenated along the last axis in canonical (`site_names`) order
    into `[*leading, Σ d_in]`, the MLP maps that to `[*leading, Σ C]`, and the output is
    split back per site by `c_sizes` in the SAME order — so concat and split are inverse
    permutations and a site's logits depend on every site's input."""

    mlp: SiteMLP
    site_names: tuple[str, ...] = eqx.field(static=True)
    in_sizes: tuple[int, ...] = eqx.field(static=True)
    c_sizes: tuple[int, ...] = eqx.field(static=True)
    expects_axes: tuple[str, ...] = eqx.field(static=True)

    def site_logits(self, site_inputs: dict[str, Array]) -> dict[str, Array]:
        assert set(site_inputs) == set(self.site_names), (
            f"site_inputs keys {sorted(site_inputs)} != CI fn sites {sorted(self.site_names)}"
        )
        for name, in_size in zip(self.site_names, self.in_sizes, strict=True):
            assert site_inputs[name].shape[-1] == in_size, (
                f"site {name} input d_in {site_inputs[name].shape[-1]} != expected {in_size}"
            )
        concatenated = jnp.concatenate([site_inputs[n] for n in self.site_names], axis=-1)
        logits = self.mlp(concatenated)
        offsets = [0]
        for c in self.c_sizes:
            offsets.append(offsets[-1] + c)
        return {
            name: logits[..., offsets[i] : offsets[i + 1]] for i, name in enumerate(self.site_names)
        }

    def __call__(self, site_inputs: dict[str, Array]) -> CIValues:
        logits = self.site_logits(site_inputs)
        return CIValues(
            lower={name: lower_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
            upper={name: upper_leaky_hard_sigmoid(logits[name]) for name in self.site_names},
        )


def init_global_mlp_ci_fn(
    arch: GlobalMLPCIArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray
) -> GlobalMLPCIFn:
    """Global MLP init: one stack `Σ d_in -> hidden_dims... -> Σ C`, same Kaiming
    `relu`-gain hidden / linear-gain head / zero-bias scheme as the per-site MLP."""
    assert arch.hidden_dims, "global MLP CI fn needs at least one hidden layer"
    in_sizes = tuple(s.d_in for s in sites)
    c_sizes = tuple(s.C for s in sites)
    dims = (sum(in_sizes), *arch.hidden_dims, sum(c_sizes))
    return GlobalMLPCIFn(
        mlp=_init_mlp_stack(dims, key),
        site_names=tuple(s.name for s in sites),
        in_sizes=in_sizes,
        c_sizes=c_sizes,
        expects_axes=(),
    )
