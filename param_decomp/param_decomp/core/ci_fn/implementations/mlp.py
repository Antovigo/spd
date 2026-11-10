"""Pointwise MLP parameters, initialization, and analytical cost."""

import math

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.flops.types import ForwardBackwardFlops, MatrixParameters, ParameterCensus
from param_decomp.core.linear_plan import value_mesh
from param_decomp.core.placement import batch_axes


def mlp_parameter_shardings(
    mesh: Mesh, output_widths: tuple[int, ...]
) -> tuple[NamedSharding, NamedSharding]:
    """Shard MLP matrix storage over batch owners and replicate biases."""
    owners = batch_axes(mesh)
    n_owners = math.prod(mesh.shape[axis] for axis in owners)
    for layer, width in enumerate(output_widths):
        assert width % n_owners == 0, f"SiteMLP.weights[{layer}].d_out {width} not ÷ N={n_owners}"
    return NamedSharding(mesh, P(None, owners)), NamedSharding(mesh, P())


def mlp_linear(x: Array, weight: Array) -> Array:
    if value_mesh(x).empty:
        return einops.einsum(x, weight, "... i, i o -> ... o")
    # Batch owners must gather the matrix before contracting their local inputs.
    weight = jax.sharding.reshard(weight, P(None, None))
    leading = jax.typeof(x).sharding.spec[:-1]
    return jnp.einsum("...i,io->...o", x, weight, out_sharding=P(*leading, None))


def mlp_flops(dimensions: tuple[int, ...], n_tokens: int) -> ForwardBackwardFlops:
    forward = sum(
        2 * n_tokens * input_width * output_width
        for input_width, output_width in zip(dimensions[:-1], dimensions[1:], strict=True)
    )
    input_projection = 2 * n_tokens * dimensions[0] * dimensions[1]
    return ForwardBackwardFlops(forward, 2 * forward - input_projection)


def mlp_parameters(dimensions: tuple[int, ...]) -> ParameterCensus:
    return ParameterCensus(
        tuple(
            MatrixParameters(d_in, d_out, 1)
            for d_in, d_out in zip(dimensions[:-1], dimensions[1:], strict=True)
        ),
        sum(dimensions[1:]),
    )


class SiteMLP(eqx.Module):
    """`hidden_dims` Linear+GELU layers then a linear head: Kaiming-`relu` (`gain √2`)
    hidden layers with zero bias, linear-gain (`1`) final head."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]
    hidden_head: tuple[Float[Array, "d_in C"], Float[Array, " C"]] | None = None
    """The SECOND readout head of a dual CI fn — a replacement for the FINAL layer only,
    `None` in a single-role fn. The GELU stack up to it is the TRUNK and is shared by
    construction: it appears once in the pytree, so both roles read one representation and
    one pullback sums both heads' trunk gradients."""

    def shardings(self, mesh: Mesh) -> "SiteMLP":
        shard_out, repl = mlp_parameter_shardings(mesh, tuple(w.shape[1] for w in self.weights))
        sharded = eqx.tree_at(
            lambda m: (m.weights, m.biases),
            self,
            ([shard_out] * len(self.weights), [repl] * len(self.biases)),
        )
        if self.hidden_head is None:
            return sharded
        return eqx.tree_at(
            lambda m: m.hidden_head, sharded, (shard_out, repl), is_leaf=lambda x: x is None
        )

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        return self.role_preactivations(x)[0]

    def role_preactivations(
        self, x: Float[Array, "*leading d_in"]
    ) -> tuple[Float[Array, "*leading C"], ...]:
        """One preactivation per role: `(output,)`, or `(output, hidden)` when a hidden head
        exists. The trunk runs ONCE and both heads read its output."""
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            x = jax.nn.gelu(mlp_linear(x, w) + b, approximate=False)
        output = mlp_linear(x, self.weights[-1]) + self.biases[-1]
        if self.hidden_head is None:
            return (output,)
        hidden_w, hidden_b = self.hidden_head
        return (output, mlp_linear(x, hidden_w) + hidden_b)


def init_mlp_stack(dims: tuple[int, ...], key: PRNGKeyArray, *, dual: bool = False) -> SiteMLP:
    """One `Linear+GELU` stack `dims[0] -> ... -> dims[-1]`: Kaiming `relu`-gain (`√2`) on
    the hidden layers, linear gain (`1`) on the final head, zero biases.

    A dual stack adds a SECOND final head drawn off a fold of the final head's own key, so the
    trunk and the output head stay bit-identical to a single-role stack at the same seed (the
    split count is unchanged)."""
    relu_gain = 2.0**0.5
    layer_keys = jax.random.split(key, len(dims) - 1)
    weights: list[Array] = []
    biases: list[Array] = []
    for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
        weights.append(jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5))
        biases.append(jnp.zeros((d_out,)))
    hidden_head = None
    if dual:
        d_in, d_out = dims[-2], dims[-1]
        hidden_w = (
            jax.random.normal(jax.random.fold_in(layer_keys[-1], 1), (d_in, d_out)) / d_in**0.5
        )
        hidden_head = (hidden_w, jnp.zeros((d_out,)))
    return SiteMLP(weights=weights, biases=biases, hidden_head=hidden_head)
