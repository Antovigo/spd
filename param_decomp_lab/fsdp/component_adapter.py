"""`FsdpComponentAdapter` — presents the core `ComponentModel` surface over a vendored
`LMComponentModel` so the shared step helpers + loss/eval metrics consume it unchanged.

`fully_shard` is applied to the inner blocks (`lm.model._layers` / CI-fn blocks), so the
adapter holds the wrapped `LMComponentModel` as its only submodule and owns no parameters
of its own. It exists purely to map the core `forward(batch, mask_infos, cache_type)`
overloads + `forward_with_output_acts` onto the vendored model's named methods, and to
re-expose the pure queries the metrics read directly.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, overload, override

from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from param_decomp.ci_fns import GlobalCiFnWrapper, LayerwiseCiFnWrapper
from param_decomp.component_model import CIOutputs, OutputWithCache
from param_decomp.components import Components
from param_decomp.masks import ComponentsMaskInfo, SamplingType
from param_decomp_lab.experiments.lm.vendored.component_model import (
    ComponentTarget,
    LMComponentModel,
)


def _to_full(t: Tensor) -> Tensor:
    """Gather a sharded `DTensor` to a full local tensor; pass a plain tensor through."""
    return t.full_tensor() if isinstance(t, DTensor) else t


class FsdpComponentAdapter(nn.Module):
    def __init__(self, lm: LMComponentModel):
        super().__init__()
        self.lm = lm

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        cache_type: Literal["output"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: object, **kwargs: object) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["input", "output", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        match cache_type:
            case "input":
                logits, pre_weight_acts = self.lm.forward_with_pre_weight_acts(batch, mask_infos)
                return OutputWithCache(output=logits, cache=pre_weight_acts)
            case "output":
                logits, output_acts = self.lm.forward_with_output_acts(batch, mask_infos)
                return OutputWithCache(output=logits, cache=output_acts)
            case "none":
                return self.lm.forward(batch, mask_infos)

    def forward_with_output_acts(
        self,
        batch: Int[Tensor, "batch pos"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.lm.forward_with_output_acts(batch, mask_infos)

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
    ) -> CIOutputs:
        return self.lm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, sampling=sampling, detach_inputs=detach_inputs
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Per-site `target_weight - components.weight`, gathered to full tensors.

        FSDP2 shards the components' V/U (they live inside the sharded transformer blocks),
        so accessed outside a forward they are DTensors; the frozen `target_weight` is either
        a replicated buffer (a plain tensor) or — under `shard_frozen_target` — a sharded
        param (a DTensor). The vendored `calc_weight_deltas` subtracts the two directly, which
        raises `aten.sub got mixed torch.Tensor and DTensor` whenever the operands' DTensor-ness
        differs. Gathering each operand to a full tensor first makes the subtraction a plain
        op. The full per-site weight matrix is materialised regardless (the faithfulness loss
        needs the whole delta), so the gather adds no asymptotic memory over the bare call.
        """
        deltas: dict[str, Tensor] = {}
        for path in self.lm.target_module_paths:
            target = _to_full(self.lm.target_weight(path))
            components = _to_full(self.lm.components[path].weight)
            deltas[path] = target - components
        return deltas

    @contextmanager
    def use_cached_residual(self, batch: Int[Tensor, "batch pos"]) -> Iterator[None]:
        with self.lm.use_cached_residual(batch):
            yield

    @property
    def module_to_c(self) -> dict[str, int]:
        return self.lm.module_to_c

    @property
    def target_module_paths(self) -> list[str]:
        return self.lm.target_module_paths

    @property
    def components(self) -> dict[str, Components]:
        return self.lm.components

    @property
    def ci_fn(self) -> GlobalCiFnWrapper | LayerwiseCiFnWrapper | None:
        return self.lm.ci_fn

    @property
    def model(self) -> ComponentTarget:
        return self.lm.model
