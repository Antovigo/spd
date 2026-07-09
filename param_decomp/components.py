"""`Components` ABC + dense (`LinearComponents` / `EmbeddingComponents`) and
`SVDLinearComponents` subclasses.

Also exposes `init_param_`, `get_module_input_dim`, and the `make_components` factory.
"""

import math
from abc import ABC, abstractmethod
from typing import Literal, override

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn.init import calculate_gain
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.decomposition_targets import Identity
from param_decomp.masks import WeightDeltaAndMask

# This is equivalent to `torch.nn.init._NonlinearityType`, but for some reason this is not always
# importable. see https://github.com/goodfire-ai/param-decomp/actions/runs/16927877557/job/47967138342
_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def init_param_(
    param: Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: _NonlinearityType = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Fill `param` in place from a Kaiming normal: `N(mean, gain(nonlinearity) / sqrt(fan_val))`.

    Args:
        param: Parameter tensor to fill in place.
        fan_val: Value used as `fan` in Kaiming normal; appears under the square root in
            the denominator of std.
        mean: Mean of the sampled normal distribution.
        nonlinearity: Nonlinearity name passed to `torch.nn.init.calculate_gain`.
        generator: Optional RNG for reproducibility.
    """
    gain: float = calculate_gain(nonlinearity)
    std: float = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)


class Components(ABC, nn.Module):
    """Per-layer components decomposing a target weight as a sum of `C` rank-1 outer products.

    `weight ≈ sum_c V[:, c] ⊗ U[c, :]`. `V` maps input activations to per-component
    scalars; `U` maps them back to the output space. Subclasses own the
    parameterization: dense classes store `V`/`U` as parameters directly,
    `SVDLinearComponents` derives them from coordinates in the target weight's SVD
    basis.
    """

    V: Tensor
    U: Tensor

    def __init__(self, C: int):
        super().__init__()
        self.C = C

    @property
    @abstractmethod
    def weight(self) -> Float[Tensor, "rows cols"]:
        raise NotImplementedError()

    @override
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
    ) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_component_acts(self, x: Tensor) -> Tensor:
        """Per-component scalar activations `V^T x`."""
        raise NotImplementedError()

    @abstractmethod
    def scale_subcomponents_(self, keep: Float[Tensor, " C"]) -> None:
        """Scale subcomponent `c` (column `c` of `V`, row `c` of `U`) by `keep[c]`, in place.

        Caller is responsible for wrapping in `torch.no_grad()`.
        """
        raise NotImplementedError()


class DenseComponents(Components, ABC):
    """Components storing `V [v_dim, C]` and `U [C, u_dim]` as free parameters."""

    def __init__(self, C: int, v_dim: int, u_dim: int):
        super().__init__(C)
        self.V = nn.Parameter(torch.empty(v_dim, C))
        self.U = nn.Parameter(torch.empty(C, u_dim))
        init_param_(self.V, fan_val=v_dim, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

    @override
    def scale_subcomponents_(self, keep: Float[Tensor, " C"]) -> None:
        self.V.mul_(keep[None, :])
        self.U.mul_(keep[:, None])


class LinearComponents(DenseComponents):
    """Components replacing an `nn.Linear`-shaped weight.

    Effective weight is `(V @ U).T` to match PyTorch's `[d_out, d_in]` storage; a frozen
    bias from the target module is re-added in the forward (biases are not trained in PD).
    """

    bias: Float[Tensor, "... d_out"] | None

    def __init__(
        self,
        C: int,
        d_in: int,
        d_out: int,
        bias: Tensor | None = None,
    ):
        super().__init__(C, v_dim=d_in, u_dim=d_out)  # NOTE: linear weights are (d_out, d_in)
        self.d_in = d_in
        self.d_out = d_out

        # We don't train biases in PD.
        self.register_buffer("bias", bias)

    @property
    @override
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    @override
    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        return einops.einsum(x.to(self.V.dtype), self.V, "... d_in, d_in C -> ... C")

    @override
    def forward(
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Apply `mask * (V^T x)` then project back by `U`, plus optional `weight_delta @ x`.

        When `component_acts_cache` is given, the pre- and post-detach component activations
        are stored under the keys `"pre_detach"` and `"post_detach"` for downstream gradient
        surgery (e.g. PPGD).
        """
        component_acts = self.get_component_acts(x)
        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... d_out -> ... d_out"
            )

        if self.bias is not None:
            out += self.bias

        return out


class SVDLinearComponents(Components):
    """Linear components learned in the SVD coordinates of the frozen target weight.

    From the economy SVD `W = Q_out diag(S) Q_in^T` (computed once at init, fp32),
    singular directions with `S > rank_threshold * S.max()` are kept (`r` of them) and
    stored as frozen persistent buffers. The learned parameters are coordinates
    `A [r, C]` and `B [C, r]`; the effective read/write vectors `V = Q_in A` /
    `U = B Q_out^T` therefore lie in `row(W)` / `col(W)` by construction. The
    rank-truncated tail of `W` is not representable by the component sum and flows
    into the weight delta. Init matches `DenseComponents`' effective distribution
    (Gaussians are rotation-invariant): `A ~ N(0, 1/d_in)`, `B ~ N(0, 1/C)`.
    """

    Q_in: Float[Tensor, "d_in r"]
    Q_out: Float[Tensor, "d_out r"]
    singular_values: Float[Tensor, " r"]
    bias: Float[Tensor, "... d_out"] | None

    def __init__(
        self,
        C: int,
        target_weight: Float[Tensor, "d_out d_in"],
        rank_threshold: float,
        bias: Tensor | None = None,
    ):
        super().__init__(C)
        d_out, d_in = target_weight.shape
        self.d_in = d_in
        self.d_out = d_out

        q_out, s, vh = torch.linalg.svd(target_weight.detach().float(), full_matrices=False)
        r = int((s > rank_threshold * s[0]).sum().item())
        assert r >= 1, f"rank_threshold {rank_threshold} keeps no singular directions"
        self.r = r
        self.register_buffer("Q_in", vh[:r].T.contiguous())
        self.register_buffer("Q_out", q_out[:, :r].contiguous())
        self.register_buffer("singular_values", s[:r].contiguous())

        self.A = nn.Parameter(torch.empty(r, C))
        self.B = nn.Parameter(torch.empty(C, r))
        init_param_(self.A, fan_val=d_in, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        # We don't train biases in PD.
        self.register_buffer("bias", bias)

    @property
    @override
    def V(self) -> Float[Tensor, "d_in C"]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return einops.einsum(self.Q_in, self.A, "d_in r, r C -> d_in C")

    @property
    @override
    def U(self) -> Float[Tensor, "C d_out"]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return einops.einsum(self.B, self.Q_out, "C r, d_out r -> C d_out")

    @property
    @override
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        core = einops.einsum(self.A, self.B, "r_in C, C r_out -> r_out r_in")
        return einops.einsum(
            self.Q_out, core, self.Q_in, "d_out r_out, r_out r_in, d_in r_in -> d_out d_in"
        )

    @override
    def get_component_acts(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        x_r = einops.einsum(x.to(self.A.dtype), self.Q_in, "... d_in, d_in r -> ... r")
        return einops.einsum(x_r, self.A, "... r, r C -> ... C")

    @override
    def scale_subcomponents_(self, keep: Float[Tensor, " C"]) -> None:
        self.A.mul_(keep[None, :])
        self.B.mul_(keep[:, None])

    @override
    def forward(
        self,
        x: Float[Tensor, "... d_in"],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... d_out"]:
        """Same contract as `LinearComponents.forward`; both projections factor through
        the SVD coordinates (`(x @ Q_in) @ A`, `(acts @ B) @ Q_out^T`)."""
        component_acts = self.get_component_acts(x)
        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out_r = einops.einsum(component_acts, self.B, "... C, C r -> ... r")
        out = einops.einsum(out_r, self.Q_out, "... r, d_out r -> ... d_out")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = einops.einsum(x, weight_delta, "... d_in, d_out d_in -> ... d_out")
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... d_out -> ... d_out"
            )

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponents(DenseComponents):
    """Components replacing an `nn.Embedding` weight.

    Avoids materialising one-hot vectors by indexing `V` directly with the input
    token ids.
    """

    def __init__(
        self,
        C: int,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__(C, v_dim=vocab_size, u_dim=embedding_dim)
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim

    @property
    @override
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        return einops.einsum(
            self.V, self.U, "vocab_size C, C embedding_dim -> vocab_size embedding_dim"
        )

    @override
    def get_component_acts(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... C"]:
        return self.V[x]

    @override
    def forward(
        self,
        x: Int[Tensor, "..."],
        mask: Float[Tensor, "... C"] | None = None,
        weight_delta_and_mask: WeightDeltaAndMask | None = None,
        component_acts_cache: dict[str, Float[Tensor, "... C"]] | None = None,
    ) -> Float[Tensor, "... embedding_dim"]:
        """Embedding forward: index `V[x]`, mask, project by `U`.

        Equivalent to `LinearComponents.forward` but uses `V[x]` instead of a one-hot
        matmul. See `LinearComponents.forward` for `component_acts_cache` semantics.
        """
        assert x.dtype == torch.long, "x must be an integer tensor"

        component_acts: Float[Tensor, "... C"] = self.get_component_acts(x)

        if component_acts_cache is not None:
            component_acts_cache["pre_detach"] = component_acts
            component_acts = component_acts.detach().requires_grad_(True)
            component_acts_cache["post_detach"] = component_acts

        if mask is not None:
            component_acts = component_acts * mask

        out = einops.einsum(component_acts, self.U, "... C, C embedding_dim -> ... embedding_dim")

        if weight_delta_and_mask is not None:
            weight_delta, weight_delta_mask = weight_delta_and_mask
            unmasked_delta_out = weight_delta[x]
            assert unmasked_delta_out.shape[:-1] == weight_delta_mask.shape
            out += einops.einsum(
                weight_delta_mask, unmasked_delta_out, "..., ... embedding_dim -> ... embedding_dim"
            )

        return out


def get_module_input_dim(target_module: nn.Module) -> int:
    """Input dimension `d_in` of a Linear-like target module.

    Supports `nn.Linear`, Radford `Conv1D`, and `Identity`. Embeddings have no scalar
    input dim and must be handled separately by the caller; this function raises
    `ValueError` for them.
    """
    match target_module:
        case nn.Linear():
            return target_module.weight.shape[1]
        case RadfordConv1D():
            return target_module.weight.shape[0]
        case Identity():
            return target_module.d
        case _:
            raise ValueError(
                f"Module {type(target_module)} not supported. "
                "Embedding modules should be handled separately."
            )


def make_components(
    target_model: nn.Module,
    module_to_c: dict[str, int],
    svd_rank_threshold: float | None = None,
) -> dict[str, Components]:
    """Build one `Components` instance per target module path.

    Dispatches by target-module type:

    - `nn.Linear` → `LinearComponents` (frozen bias carried over), or
      `SVDLinearComponents` when `svd_rank_threshold` is set.
    - Radford `Conv1D` → `LinearComponents` with shapes swapped for the transposed weight layout.
    - `Identity` → `LinearComponents` with `d_in == d_out` and no bias.
    - `nn.Embedding` → `EmbeddingComponents`.

    Args:
        target_model: Frozen model containing the submodules to decompose.
        module_to_c: Map from submodule path (as returned by `model.get_submodule`) to
            the number of components `C` to allocate for that module.
        svd_rank_threshold: When set, every `nn.Linear` target is parameterized in the
            SVD coordinates of its frozen weight (`SVDLinearComponents`), keeping
            singular directions with `sigma > threshold * sigma_max`; `0.0` keeps all
            nonzero directions. Only `nn.Linear` targets are supported in this mode.

    Returns:
        Dict keyed by the same submodule paths, mapping to a `Components` instance whose
        weights have been initialised but not yet trained.
    """
    out: dict[str, Components] = {}
    for path, C in module_to_c.items():
        target_module = target_model.get_submodule(path)
        if svd_rank_threshold is not None:
            assert isinstance(target_module, nn.Linear), (
                f"svd_rank_threshold only supports nn.Linear targets, got "
                f"{type(target_module)} for {path}"
            )
        match target_module:
            case nn.Linear() if svd_rank_threshold is not None:
                comp: Components = SVDLinearComponents(
                    C=C,
                    target_weight=target_module.weight,
                    rank_threshold=svd_rank_threshold,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case nn.Linear():
                d_out, d_in = target_module.weight.shape
                comp = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case RadfordConv1D():
                d_in, d_out = target_module.weight.shape
                comp = LinearComponents(
                    C=C,
                    d_in=d_in,
                    d_out=d_out,
                    bias=target_module.bias.data if target_module.bias is not None else None,  # pyright: ignore[reportUnnecessaryComparison]
                )
            case Identity():
                comp = LinearComponents(
                    C=C,
                    d_in=target_module.d,
                    d_out=target_module.d,
                    bias=None,
                )
            case nn.Embedding():
                comp = EmbeddingComponents(
                    C=C,
                    vocab_size=target_module.num_embeddings,
                    embedding_dim=target_module.embedding_dim,
                )
            case _:
                raise ValueError(f"Module {target_module} not supported")
        out[path] = comp
    return out
