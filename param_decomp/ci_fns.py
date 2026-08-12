"""Causal-importance function configs, CI-fn modules, and wrappers."""

from dataclasses import dataclass
from typing import Literal, Self, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.ci_nn_blocks import Linear, ParallelLinear, TransformerBlock
from param_decomp.components import Components, EmbeddingComponents, get_module_input_dim

LayerwiseCiFnType = Literal["mlp", "vector_mlp", "shared_mlp"]
GlobalCiFnType = Literal["global_shared_mlp", "global_shared_transformer"]

CIRole = Literal["output", "hidden"]
"""Which CI network a value came from, or which one a metric reads.

`"output"` scores how much each subcomponent matters for reconstructing the target
model's final output; `"hidden"` scores how much it matters for reconstructing the
decomposed sites' activations. A run only has the latter under `pd.dual_hidden_ci`; the
two nets share one pool of subcomponents and differ only in the reconstruction loss that
trains them.
"""


class HiddenCIFloorConfig(BaseConfig):
    """Parameterize the hidden CI net so its CI can never fall below the output net's.

    A subcomponent reaches the model's output only through the output of the matrix it
    lives in, which is exactly what the hidden objective measures — so output-important
    implies hidden-important, while the converse rightly fails for a component whose
    contribution is cancelled downstream. This makes that implication structural instead
    of something the two nets have to rediscover.

    Applied in logit space by `floor_hidden_ci_logits`: both sigmoid branches are monotone,
    so one ordering there is inherited by the mask and the minimality penalty at once. The
    mask branch additionally requires the two roles to share their binomial jitter draw —
    see `ComponentModel.calc_causal_importances_both_roles`.

    Lives here rather than in `configs.py` because `component_model` consumes it and is
    itself reachable from `configs`; see the config placement rule in `metrics/CLAUDE.md`.
    """

    sharpness: PositiveFloat = Field(
        default=10.0,
        description="Softness of the floor, in logit units: the smooth max deviates from a "
        "hard max by at most `ln(2)/sharpness` (0.069 at the default, against a sigmoid "
        "window of width 1). Lower values blur the floor and let the hidden CI sit "
        "measurably below the output CI; higher values approach a hard max, whose gradient "
        "to the hidden head vanishes wherever the floor binds.",
    )


def floor_hidden_ci_logits(
    hidden_logits: dict[str, Float[Tensor, "... C"]],
    output_logits: dict[str, Float[Tensor, "... C"]],
    cfg: HiddenCIFloorConfig,
) -> dict[str, Float[Tensor, "... C"]]:
    """Smooth `max(hidden_logits, output_logits)`, per module.

    `softplus(x, beta)` is `log(1 + exp(beta x)) / beta`, so adding it to the floor gives a
    max that is above both arguments by at most `ln(2)/beta`, and whose gradient w.r.t. the
    hidden logit is `sigmoid(beta * gap)` rather than the hard max's zero — a hidden logit
    that has fallen under the floor can still climb back out. Torch's own linear-regime
    threshold handles the large-gap tail, so no overflow guard is needed here.

    That escape is not unconditional. The gradient decays as `exp(beta * gap)` and reaches
    fp32 underflow at `gap ≈ -87/beta` (-8.7 at the default); well before that, around
    `gap ≈ -2`, it drops under Adam's `eps` and the step size collapses even though the
    gradient is nominally nonzero. Logits sit near the `[0, 1]` sigmoid window in practice,
    so those gaps are far outside the operating range — but a subcomponent driven that far
    below the floor is pinned to the output net's CI for good. Lower `sharpness` widens the
    escapable band at the cost of a larger init offset.

    `output_logits` are detached by the caller: this constrains the hidden net, and must not
    become a path by which the hidden objective's sparsity pressure drags the output CI down.
    That detach also drops the floor's true dependence on its own inputs — the exact Jacobian
    carries a `(1 - sigmoid(beta * gap)) * d z_out / d acts` term — so where the floor binds
    hard, the hidden CI is treated as independent of the activations that produced it. Nothing
    consumes that gradient today (the trainer's `pre_weight_acts` come from a frozen model),
    but a caller differentiating CI w.r.t. activations under `role="hidden"` would be misled.

    Under `autocast_bf16` the forward is a *hard* max well before the gradient dies: bf16's
    ULP near 0.5 is ~0.002, which `softplus(gap, beta=10)` drops below at `gap ≈ -0.4`. The
    escape gradient survives regardless, autograd differentiating softplus analytically rather
    than through the rounded forward.
    """
    assert hidden_logits.keys() == output_logits.keys(), (
        f"floor needs one output logit per hidden logit: {sorted(hidden_logits)} vs "
        f"{sorted(output_logits)}"
    )
    return {
        name: output_logits[name] + F.softplus(logits - output_logits[name], beta=cfg.sharpness)
        for name, logits in hidden_logits.items()
    }


class OutputCICapConfig(BaseConfig):
    """Cap the output CI net at the hidden net's — the same ordering as `HiddenCIFloorConfig`,
    routed the other way.

    Both enforce `CI_hidden >= CI_output`. They differ in which net absorbs the correction,
    and that difference is the whole point:

    - **Floor** modifies the *hidden* net (`CI_hidden = max(raw, CI_output)`). Where it binds,
      a violation is resolved by declaring the subcomponent hidden-important. The ordering
      holds, but the anomaly it was meant to surface is papered over.
    - **Cap** modifies the *output* net (`CI_output = min(raw, CI_hidden)`). Where it binds,
      the output objective's gradient is redirected into the hidden logit: to rely on a
      subcomponent, the output reconstruction must first convince the hidden net that the
      subcomponent does real work at its own matrix's output — and the hidden
      importance-minimality penalty makes it pay for that. A component that moves the logits
      through something other than its own site's activations cannot be bought.

    So the cap is the direction that makes the hidden reconstruction *guide* the output
    reconstruction toward mechanistically faithful components, rather than merely agree with
    it. Mutually exclusive with `hidden_ci_floor`; the two are the same inequality.
    """

    sharpness: PositiveFloat = Field(
        default=10.0,
        description="Softness of the cap, in logit units; the smooth min deviates from a hard "
        "min by at most `ln(2)/sharpness`. Same role as `HiddenCIFloorConfig.sharpness`.",
    )


def cap_output_ci_logits(
    output_logits: dict[str, Float[Tensor, "... C"]],
    hidden_logits: dict[str, Float[Tensor, "... C"]],
    cfg: OutputCICapConfig,
) -> dict[str, Float[Tensor, "... C"]]:
    """Smooth `min(output_logits, hidden_logits)`, per module.

    The mirror of `floor_hidden_ci_logits`: `min(a, b) = b - softplus(b - a, beta)`. Gradient
    w.r.t. the output logit is `sigmoid(beta * (hidden - output))` — ~1 while the cap is slack,
    fading to 0 once it binds — and w.r.t. the hidden logit it is exactly the complement. So
    the output objective's pressure transfers to the hidden net precisely where the cap binds.

    `hidden_logits` are **not** detached, and must not be: that gradient path is the mechanism,
    not a leak. Contrast `floor_hidden_ci_logits`, where detaching the output logits keeps the
    hidden objective's sparsity pressure off the output net.
    """
    assert output_logits.keys() == hidden_logits.keys(), (
        f"cap needs one hidden logit per output logit: {sorted(output_logits)} vs "
        f"{sorted(hidden_logits)}"
    )
    return {
        name: hidden_logits[name] - F.softplus(hidden_logits[name] - logits, beta=cfg.sharpness)
        for name, logits in output_logits.items()
    }


class LayerwiseCiConfig(BaseConfig):
    """Layerwise CI fns — one independent CI fn per decomposition target."""

    mode: Literal["layerwise"] = "layerwise"
    fn_type: LayerwiseCiFnType = Field(
        ..., description="Type of layerwise CI function: mlp, vector_mlp, or shared_mlp"
    )
    hidden_dims: list[PositiveInt] = Field(
        ..., description="Hidden dimensions for the CI function MLP"
    )

    @model_validator(mode="after")
    def validate_hidden_dims(self) -> Self:
        if self.fn_type in ("mlp", "vector_mlp") and not self.hidden_dims:
            raise ValueError(f"hidden_dims must be non-empty for fn_type={self.fn_type!r}")
        return self


class AttnConfig(BaseConfig):
    """Self-attention config for the transformer CI fn. Uses RoPE for length generalization."""

    n_heads: PositiveInt = Field(
        ...,
        description="Number of attention heads. Must divide the input dimension.",
    )
    max_len: PositiveInt = Field(
        default=2048,
        description="Maximum sequence length for RoPE embeddings.",
    )
    rope_base: float = Field(
        default=10000.0,
        description="Base for RoPE frequency computation.",
    )


class GlobalSharedTransformerCiConfig(BaseConfig):
    """Config for the global transformer CI fn.

    `d_model` must be divisible by `attn_config.n_heads` and the resulting per-head dim
    must be even (RoPE). `mlp_hidden_dim` defaults to `[4 * d_model]`.
    """

    d_model: PositiveInt
    n_blocks: PositiveInt
    mlp_hidden_dim: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimension for transformer MLP blocks. "
        "If None, defaults to [4 * d_model].",
    )
    attn_config: AttnConfig
    final_rms_norm: bool = Field(
        default=False,
        description="RMS-norm the residual stream before the output head, decoupling "
        "logit scale from residual-stream growth.",
    )
    zero_init_readout: bool = Field(
        default=True,
        description="Zero-init the output head with bias 0.5 (all logits start "
        "mid-window). False restores fan-in random init with zero bias.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        assert self.d_model % self.attn_config.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"attn_config.n_heads ({self.attn_config.n_heads})"
        )
        d_head = self.d_model // self.attn_config.n_heads
        assert d_head % 2 == 0, (
            f"d_head ({d_head}) must be even for RoPE. "
            f"d_model={self.d_model}, "
            f"n_heads={self.attn_config.n_heads}"
        )
        return self


class GlobalCiConfig(BaseConfig):
    """A single global CI fn that maps all layers jointly."""

    mode: Literal["global"] = "global"
    fn_type: GlobalCiFnType = Field(
        ...,
        description="Type of global CI function: global_shared_mlp or global_shared_transformer",
    )
    hidden_dims: list[PositiveInt] | None = Field(
        default=None,
        description="Hidden dimensions for global_shared_mlp CI function.",
    )
    simple_transformer_ci_cfg: GlobalSharedTransformerCiConfig | None = None

    @model_validator(mode="after")
    def validate_ci_config(self) -> Self:
        if self.fn_type == "global_shared_mlp":
            assert self.hidden_dims is not None, (
                "hidden_dims must be specified when fn_type='global_shared_mlp'"
            )
        elif self.fn_type == "global_shared_transformer":
            assert self.simple_transformer_ci_cfg is not None, (
                "simple_transformer_ci_cfg must be specified when fn_type='global_shared_transformer'"
            )
            assert self.hidden_dims is None, (
                "hidden_dims is only used for fn_type='global_shared_mlp'"
            )
        return self


# Discriminated union (by `mode`) of every CI-fn config the trainer accepts. Pydantic
# picks the right branch from the YAML `pd.ci_config.mode` literal.
CiConfig = LayerwiseCiConfig | GlobalCiConfig


class MLPCiFn(nn.Module):
    """Per-component scalar-input MLP CI fn.

    Each of `C` components gets its own MLP mapping a scalar component activation to a
    scalar CI value; built from `ParallelLinear` layers operating on a singleton last dim.
    """

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = 1 if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        x = einops.rearrange(x, "... C -> ... C 1")
        x = self.layers(x)
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorMLPCiFn(nn.Module):
    """Per-component vector-input MLP CI fn.

    Each of `C` components gets its own MLP consuming the full `[..., d_in]` layer input;
    built from `ParallelLinear` so all `C` networks run in one batched einsum.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())

        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        x = self.layers(einops.rearrange(x, "... d_in -> ... 1 d_in"))
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorSharedMLPCiFn(nn.Module):
    """Shared MLP `[..., d_in] -> [..., C]`.

    All components share every hidden layer; only the final projection splits
    per-component.
    """

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else input_dim
        self.layers.append(Linear(final_dim, C, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return self.layers(x)


class GlobalSharedMLPCiFn(nn.Module):
    """Global MLP over all layers.

    Concatenates all decomposition-target inputs along the feature dim, runs one shared
    MLP, then splits the output back into per-layer `[..., C]` slices. Layer order is
    fixed by sorted layer name so concatenation is deterministic.
    """

    def __init__(
        self,
        layer_configs: dict[str, tuple[int, int]],  # layer_name -> (input_dim, C)
        hidden_dims: list[int],
    ):
        super().__init__()

        self.layer_order = sorted(layer_configs.keys())
        self.layer_configs = layer_configs
        self.split_sizes = [layer_configs[name][1] for name in self.layer_order]

        total_input_dim = sum(input_dim for input_dim, _ in layer_configs.values())
        total_C = sum(C for _, C in layer_configs.values())

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            in_dim = total_input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(Linear(in_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else total_input_dim
        self.layers.append(Linear(final_dim, total_C, nonlinearity="linear"))

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [input_acts[name] for name in self.layer_order]
        concatenated = torch.cat(inputs_list, dim=-1)
        output = self.layers(concatenated)
        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}


@dataclass
class TargetLayerConfig:
    """Per-target metadata consumed by `GlobalSharedTransformerCiFn`."""

    input_dim: int
    C: int


def _param_shapes(module: nn.Module) -> list[torch.Size]:
    return [p.shape for p in module.parameters()]


class GlobalSharedTransformerCiFn(nn.Module):
    """Global transformer attending over sequence to produce per-component CI.

    Per-layer inputs are RMS-normed, concatenated along the feature dim, projected to
    `d_model`, and run through `n_layers` `TransformerBlock`s with bidirectional
    self-attention. A final linear projection (zero-init with bias 0.5, optionally
    preceded by RMS norm) produces logits which are split back into per-layer
    `[..., C]` slices in sorted-name order. For 2D inputs (e.g. TMS, resid_mlp — no
    sequence axis) a singleton sequence dim is added before the transformer and
    squeezed out after.

    Everything up to the readout head is the *trunk*; `adopt_trunk` hands one net's trunk
    to another so both read one representation.
    """

    def __init__(
        self,
        target_model_layer_configs: dict[str, TargetLayerConfig],
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_len: int,
        mlp_hidden_dims: list[int] | None = None,
        rope_base: float = 10000.0,
        final_rms_norm: bool = False,
        zero_init_readout: bool = True,
    ):
        super().__init__()

        self.layer_order = sorted(target_model_layer_configs.keys())
        self.target_model_layer_configs = target_model_layer_configs
        self.split_sizes = [target_model_layer_configs[name].C for name in self.layer_order]
        self.d_model = d_model
        self.n_transformer_layers = n_layers
        self.n_heads = n_heads
        self.final_rms_norm = final_rms_norm

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [4 * d_model]

        total_input_dim = sum(config.input_dim for config in target_model_layer_configs.values())
        total_c = sum(config.C for config in target_model_layer_configs.values())

        self._output_head = Linear(d_model, total_c, nonlinearity="linear")
        if zero_init_readout:
            # Logits start at exactly 0.5: mid-window in the hard sigmoid's linear
            # region, within gradient reach of both losses.
            nn.init.zeros_(self._output_head.W)
            nn.init.constant_(self._output_head.b, 0.5)

        self._input_projector = Linear(total_input_dim, d_model, nonlinearity="relu")
        self._blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_hidden_dims=mlp_hidden_dims,
                    max_len=max_len,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ]
        )

    def adopt_trunk(self, source: "GlobalSharedTransformerCiFn") -> None:
        """Read through `source`'s input projector and blocks — the same parameters, not copies.

        Leaves this net's readout head private, so the two nets score the same inputs off
        one representation. This net's own trunk is discarded. Submodule names don't
        change, so a shared-trunk state dict stays key-identical to an independent pair's
        (both names address the one tensor).
        """
        assert self.layer_order == source.layer_order, (
            "a shared trunk must read the same sites in the same order: "
            f"{source.layer_order} vs {self.layer_order}"
        )
        assert self.n_heads == source.n_heads, (
            f"shared trunk has {source.n_heads} heads, this net was built for {self.n_heads}"
        )
        assert _param_shapes(self._input_projector) == _param_shapes(source._input_projector), (
            "shared trunk's input projector has a different shape — the two nets disagree "
            "on d_model or on the concatenated site width"
        )
        assert _param_shapes(self._blocks) == _param_shapes(source._blocks), (
            "shared trunk's blocks have different shapes — the two nets disagree on "
            "n_blocks, d_model or mlp_hidden_dim"
        )
        self._input_projector = source._input_projector
        self._blocks = source._blocks

    @override
    def forward(
        self,
        input_acts: dict[str, Float[Tensor, "... d_in"]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        inputs_list = [
            F.rms_norm(input_acts[name], (input_acts[name].shape[-1],)) for name in self.layer_order
        ]
        concatenated = torch.cat(inputs_list, dim=-1)
        projected: Tensor = self._input_projector(concatenated)

        # The transformer blocks expect a sequence dimension, so we add an extra dimension to our
        # activations if we only have 2D acts (e.g. in TMS and resid_mlp).
        added_seq_dim = False
        if projected.ndim < 3:
            projected = projected.unsqueeze(-2)
            added_seq_dim = True

        x = projected
        for block in self._blocks:
            x = block(x)

        if self.final_rms_norm:
            x = F.rms_norm(x, (self.d_model,))

        output = self._output_head(x)

        if added_seq_dim:
            output = output.squeeze(-2)

        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        outputs = {name: split_outputs[i] for i, name in enumerate(self.layer_order)}

        return outputs


class LayerwiseCiFnWrapper(nn.Module):
    """Bundle a dict of per-layer CI fns behind a single dict-in/dict-out interface.

    Runs each layer's CI fn on its own input. For `ci_fn_type == "mlp"` the per-component
    scalar activations are obtained via `Components.get_component_acts` first; the other
    variants receive the raw layer input. Layer names are stored under `ModuleDict` with
    `.` replaced by `-` so state-dict keys are well-formed.
    """

    def __init__(
        self,
        ci_fns: dict[str, nn.Module],
        components: dict[str, Components],
        ci_fn_type: LayerwiseCiFnType,
    ):
        super().__init__()
        self.layer_names = sorted(ci_fns.keys())
        self.components = components
        self.ci_fn_type = ci_fn_type

        # Store as ModuleDict with "." replaced by "-" for state dict compatibility
        self._ci_fns = nn.ModuleDict(
            {name.replace(".", "-"): ci_fns[name] for name in self.layer_names}
        )

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        outputs: dict[str, Float[Tensor, "... C"]] = {}

        for layer_name in self.layer_names:
            ci_fn = self._ci_fns[layer_name.replace(".", "-")]
            input_acts = layer_acts[layer_name]

            # MLPCiFn expects component activations, others take raw input
            if self.ci_fn_type == "mlp":
                ci_fn_input = self.components[layer_name].get_component_acts(input_acts)
            else:
                ci_fn_input = input_acts

            outputs[layer_name] = ci_fn(ci_fn_input)

        return outputs


class GlobalCiFnWrapper(nn.Module):
    """Gives the global CI fns the same dict-in/dict-out interface as the layerwise wrapper.

    For `EmbeddingComponents` the raw input is a tensor of token ids; this wrapper
    converts them to component activations via `EmbeddingComponents.get_component_acts`
    so the global CI fn always sees floating-point activations.
    """

    def __init__(
        self,
        global_ci_fn: GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn,
        components: dict[str, Components],
    ):
        super().__init__()
        self._global_ci_fn = global_ci_fn
        self.components = components

    @property
    def global_ci_fn(self) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
        return self._global_ci_fn

    @override
    def forward(
        self,
        layer_acts: dict[str, Float[Tensor, "..."]],
    ) -> dict[str, Float[Tensor, "... C"]]:
        transformed: dict[str, Float[Tensor, ...]] = {}

        for layer_name, acts in layer_acts.items():
            component = self.components[layer_name]
            if isinstance(component, EmbeddingComponents):
                # Embeddings pass token IDs; convert to component activations
                transformed[layer_name] = component.get_component_acts(acts)
            else:
                transformed[layer_name] = acts

        return self._global_ci_fn(transformed)


def _make_layerwise_ci_fn(
    target_module: nn.Module,
    C: int,
    ci_fn_type: LayerwiseCiFnType,
    ci_fn_hidden_dims: list[int],
) -> nn.Module:
    if isinstance(target_module, nn.Embedding):
        assert ci_fn_type == "mlp", "Embedding modules only supported for ci_fn_type='mlp'"

    if ci_fn_type == "mlp":
        return MLPCiFn(C=C, hidden_dims=ci_fn_hidden_dims)

    input_dim = get_module_input_dim(target_module)
    match ci_fn_type:
        case "vector_mlp":
            return VectorMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)
        case "shared_mlp":
            return VectorSharedMLPCiFn(C=C, input_dim=input_dim, hidden_dims=ci_fn_hidden_dims)


def _make_global_ci_fn(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: GlobalCiConfig,
) -> GlobalSharedMLPCiFn | GlobalSharedTransformerCiFn:
    ci_fn_type = ci_config.fn_type
    ci_fn_hidden_dims = ci_config.hidden_dims

    layer_configs: dict[str, tuple[int, int]] = {}
    for path, module_c in module_to_c.items():
        target_module = target_model.get_submodule(path)
        component = components[path]
        if isinstance(target_module, nn.Embedding):
            assert isinstance(component, EmbeddingComponents)
            input_dim = component.C
        else:
            input_dim = get_module_input_dim(target_module)
        layer_configs[path] = (input_dim, module_c)

    match ci_fn_type:
        case "global_shared_mlp":
            assert ci_fn_hidden_dims is not None
            return GlobalSharedMLPCiFn(layer_configs=layer_configs, hidden_dims=ci_fn_hidden_dims)
        case "global_shared_transformer":
            transformer_cfg = ci_config.simple_transformer_ci_cfg
            assert transformer_cfg is not None
            return GlobalSharedTransformerCiFn(
                target_model_layer_configs={
                    path: TargetLayerConfig(input_dim=input_dim, C=C)
                    for path, (input_dim, C) in layer_configs.items()
                },
                d_model=transformer_cfg.d_model,
                n_layers=transformer_cfg.n_blocks,
                n_heads=transformer_cfg.attn_config.n_heads,
                mlp_hidden_dims=transformer_cfg.mlp_hidden_dim,
                max_len=transformer_cfg.attn_config.max_len,
                rope_base=transformer_cfg.attn_config.rope_base,
                final_rms_norm=transformer_cfg.final_rms_norm,
                zero_init_readout=transformer_cfg.zero_init_readout,
            )


def make_ci_fn_wrapper(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    components: dict[str, Components],
    ci_config: CiConfig,
) -> LayerwiseCiFnWrapper | GlobalCiFnWrapper:
    """Build the CI-fn wrapper selected by `ci_config`.

    `LayerwiseCiConfig` → one inner CI fn per `module_to_c` entry inside a
    `LayerwiseCiFnWrapper`; `GlobalCiConfig` → a single global CI fn inside a
    `GlobalCiFnWrapper`.

    Args:
        target_model: Frozen target model; used to look up each decomposition target's
            input dimensionality.
        module_to_c: Map from decomposition-target submodule path to component count.
        components: Map from decomposition-target submodule path to its `Components`
            instance (used by `MLPCiFn` and embedding-target dispatch).
        ci_config: Discriminated CI-fn config; runtime type selects the wrapper.
    """
    match ci_config:
        case LayerwiseCiConfig():
            raw_ci_fns = {
                path: _make_layerwise_ci_fn(
                    target_module=target_model.get_submodule(path),
                    C=C,
                    ci_fn_type=ci_config.fn_type,
                    ci_fn_hidden_dims=ci_config.hidden_dims,
                )
                for path, C in module_to_c.items()
            }
            return LayerwiseCiFnWrapper(
                ci_fns=raw_ci_fns,
                components=components,
                ci_fn_type=ci_config.fn_type,
            )
        case GlobalCiConfig():
            raw_global = _make_global_ci_fn(
                target_model=target_model,
                module_to_c=module_to_c,
                components=components,
                ci_config=ci_config,
            )
            return GlobalCiFnWrapper(global_ci_fn=raw_global, components=components)


def share_transformer_trunk(
    source: LayerwiseCiFnWrapper | GlobalCiFnWrapper,
    target: LayerwiseCiFnWrapper | GlobalCiFnWrapper,
) -> None:
    """Point `target`'s CI fn at `source`'s trunk, leaving both readout heads private.

    The two nets then score the same inputs off one representation, and every objective
    training either of them shapes it. Only `global_shared_transformer` has a trunk to
    share.
    """
    assert isinstance(source, GlobalCiFnWrapper) and isinstance(target, GlobalCiFnWrapper), (
        "a shared trunk needs global CI fns, not layerwise ones"
    )
    source_ci_fn, target_ci_fn = source.global_ci_fn, target.global_ci_fn
    assert isinstance(source_ci_fn, GlobalSharedTransformerCiFn) and isinstance(
        target_ci_fn, GlobalSharedTransformerCiFn
    ), f"a shared trunk needs global_shared_transformer, not {type(source_ci_fn).__name__}"
    target_ci_fn.adopt_trunk(source_ci_fn)
