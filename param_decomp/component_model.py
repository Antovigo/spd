"""`ComponentModel` — wraps the target model with `Components` modules + a CI fn.

Emits gradient-aware cached forward passes consumed by the loss metrics.
"""

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, overload, override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from transformers.pytorch_utils import Conv1D as RadfordConv1D

from param_decomp.base_config import runtime_cast
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import (
    CiConfig,
    CIRole,
    HiddenCIFloorConfig,
    OutputCICapConfig,
    cap_output_ci_logits,
    floor_hidden_ci_logits,
    make_ci_fn_wrapper,
    share_transformer_trunk,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES, SigmoidType
from param_decomp.components import Components, make_components
from param_decomp.decomposition_targets import DecompositionTarget, Identity
from param_decomp.masks import ComponentsMaskInfo, SamplingType


class _SiteCacheComplete(Exception):
    """Aborts a forward pass once every hooked site has been cached (see `site_outputs`).

    A forward hook cannot ask PyTorch to stop the pass, so unwinding the stack is the only
    way to skip the model's remaining layers.
    """


class OutputWithCache(NamedTuple):
    """Forward output paired with per-module cached activations.

    Cache keys are target-module paths (or `f"{path}_{kind}"` for component-acts entries);
    contents depend on the `cache_type` requested.
    """

    output: Tensor
    cache: dict[str, Tensor]


@dataclass
class CIOutputs:
    """Triple of CI tensors keyed by target module path.

    `lower_leaky` is multiplied into component contributions (bounded above by 1);
    `upper_leaky` is used by importance-minimality losses (bounded below by 0);
    `pre_sigmoid` is the raw CI-fn output.
    """

    lower_leaky: dict[str, Float[Tensor, "... C"]]
    upper_leaky: dict[str, Float[Tensor, "... C"]]
    pre_sigmoid: dict[str, Tensor]


class ComponentModel(nn.Module):
    """Wrapper around a frozen target model that exposes parameter components.

    The underlying base model can be any `nn.Module` (e.g. `LlamaForCausalLM`,
    `AutoModelForCausalLM`) as long as the sub-module paths to decompose are
    provided in `decomposition_targets`. The wrapper registers components and the
    causal-importance function (`ci_fn`) as submodules so they participate in DDP
    parameter sync and `.to(device)` semantics.

    The target model's parameters must not require grad — the constructor asserts this.
    Forward pass supports four cache modes and optional component replacement; see
    `forward` for the matrix of behaviors.
    """

    def __init__(
        self,
        target_model: nn.Module,
        run_batch: RunBatch,
        decomposition_targets: list[DecompositionTarget],
        ci_config: CiConfig,
        sigmoid_type: SigmoidType,
        dual_hidden_ci: bool = False,
        dual_hidden_ci_shared_trunk: bool = False,
        hidden_ci_floor: HiddenCIFloorConfig | None = None,
        output_ci_cap: OutputCICapConfig | None = None,
        hidden_readout_sites: Mapping[str, str] | None = None,
    ):
        """Wrap `target_model` with parameter-component machinery.

        Args:
            target_model: Frozen model whose weights are being decomposed. Constructor
                asserts every parameter has `requires_grad=False`.
            run_batch: Callable that runs the target model on one batch and returns its
                output tensor; invoked through the wrapper for caching / DDP.
            decomposition_targets: Resolved target list — one `(module_path, C)` per
                module to decompose. Produced by `resolve_decomposition_targets`.
            ci_config: Discriminated CI-fn config selecting layerwise vs global.
            sigmoid_type: Sigmoid used to squash raw CI-fn outputs. `"leaky_hard"`
                splits into lower- and upper-leaky variants; everything else uses one
                function for both branches.
            dual_hidden_ci: Build a second CI fn (`ci_fn_hidden`) of the same
                architecture, scoring importance for reconstructing the decomposed sites'
                activations rather than the model output. Both nets score the same pool of
                subcomponents; select between them with `CIRole`.
            dual_hidden_ci_shared_trunk: Build the second CI fn on the first one's input
                projector and transformer blocks — the same parameters, not copies — so
                only the readout head is private per role and both objectives shape one
                representation. Requires `dual_hidden_ci` and a
                `global_shared_transformer` CI fn.
            hidden_ci_floor: When set, the hidden net's CI is floored at the output net's,
                so a subcomponent can never be scored less important for the sites'
                activations than for the model's output. Costs one extra output-net
                forward on every `role="hidden"` CI call. Requires `dual_hidden_ci`.
            output_ci_cap: The same ordering routed the other way — the output net's CI is
                capped at the hidden net's, so the output reconstruction can only rely on a
                subcomponent the hidden reconstruction also supports. Mutually exclusive with
                `hidden_ci_floor`. Requires `dual_hidden_ci`.
            hidden_readout_sites: Extra measurement points for the hidden-activation
                losses, as `{measurement_name: module_path}`. The *input* of each listed
                module is captured — both clean (`forward(cache_type="input")`) and masked
                (`site_outputs`) — and joins the decomposed sites in `measurement_sites`,
                so a metric's `site_patterns` can select it. Lets the hidden objective
                target activations that are not themselves a decomposed matrix's output,
                the residual stream being the motivating case.
        """
        super().__init__()
        self._run_batch: RunBatch = run_batch

        for name, param in target_model.named_parameters():
            assert not param.requires_grad, (
                f"Target model should not have any trainable parameters. "
                f"Found {param.requires_grad} for {name}"
            )

        self.target_model = target_model
        self.module_to_c = {target.module_path: target.C for target in decomposition_targets}
        self.target_module_paths = list(self.module_to_c.keys())

        self.hidden_readout_sites = dict(hidden_readout_sites) if hidden_readout_sites else {}
        collisions = set(self.hidden_readout_sites) & set(self.target_module_paths)
        assert not collisions, f"readout names collide with decomposed sites: {sorted(collisions)}"
        readout_paths = list(self.hidden_readout_sites.values())
        assert len(set(readout_paths)) == len(readout_paths), (
            f"two readout sites hook the same module: {readout_paths}"
        )
        assert not set(readout_paths) & set(self.target_module_paths), (
            "a readout site may not hook a decomposed module — its input is already cached "
            f"as a pre-weight activation: {sorted(set(readout_paths) & set(self.target_module_paths))}"
        )
        for module_path in readout_paths:
            target_model.get_submodule(module_path)

        self.components = make_components(target_model, self.module_to_c)
        self._components = nn.ModuleDict(
            {k.replace(".", "-"): self.components[k] for k in sorted(self.components)}
        )

        self.ci_fn = make_ci_fn_wrapper(
            target_model=target_model,
            module_to_c=self.module_to_c,
            components=self.components,
            ci_config=ci_config,
        )
        self.ci_fn_hidden = (
            make_ci_fn_wrapper(
                target_model=target_model,
                module_to_c=self.module_to_c,
                components=self.components,
                ci_config=ci_config,
            )
            if dual_hidden_ci
            else None
        )
        if dual_hidden_ci_shared_trunk:
            assert self.ci_fn_hidden is not None, (
                "dual_hidden_ci_shared_trunk needs a second CI net; set dual_hidden_ci"
            )
            share_transformer_trunk(source=self.ci_fn, target=self.ci_fn_hidden)

        self.hidden_ci_floor = hidden_ci_floor
        self.output_ci_cap = output_ci_cap
        assert not (hidden_ci_floor is not None and output_ci_cap is not None), (
            "hidden_ci_floor and output_ci_cap are the same inequality routed in opposite "
            "directions; enabling both is circular"
        )
        assert (
            hidden_ci_floor is None and output_ci_cap is None
        ) or self.ci_fn_hidden is not None, (
            "a CI ordering constraint needs a hidden CI net; set dual_hidden_ci"
        )

        if sigmoid_type == "leaky_hard":
            self.lower_leaky_fn = SIGMOID_TYPES["lower_leaky_hard"]
            self.upper_leaky_fn = SIGMOID_TYPES["upper_leaky_hard"]
        else:
            # For other sigmoid types, use the same function for both
            self.lower_leaky_fn = SIGMOID_TYPES[sigmoid_type]
            self.upper_leaky_fn = SIGMOID_TYPES[sigmoid_type]

    @property
    def measurement_sites(self) -> list[str]:
        """Everything the hidden-activation losses can measure at.

        Decomposed sites are measured at their (component-replaced) output; readout sites
        at the captured input of their module. Both are addressed by the same
        `site_patterns` fnmatch in the metric configs.
        """
        return self.target_module_paths + list(self.hidden_readout_sites)

    def target_weight(self, module_name: str) -> Float[Tensor, "rows cols"]:
        """Weight matrix of a target module in PD's `[d_out, d_in]` row-major convention.

        Radford `Conv1D` is transposed back from its stored `[d_in, d_out]` layout so all
        targets share the same shape. For an `Identity` shim the returned tensor is the
        identity matrix of size `target_module.d` on the model's device/dtype.
        """
        target_module = self.target_model.get_submodule(module_name)

        match target_module:
            case RadfordConv1D():
                return target_module.weight.T
            case nn.Linear() | nn.Embedding():
                return target_module.weight
            case Identity():
                p = next(self.parameters())
                return torch.eye(target_module.d, device=p.device, dtype=p.dtype)
            case _:
                raise ValueError(f"Module {target_module} not supported")

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["component_acts"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["input"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        cache_type: Literal["output"],
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
    ) -> OutputWithCache: ...

    @overload
    def __call__(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["none"] = "none",
    ) -> Tensor: ...

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | OutputWithCache:
        return super().__call__(*args, **kwargs)

    @override
    def forward(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo] | None = None,
        cache_type: Literal["component_acts", "input", "output", "none"] = "none",
    ) -> Tensor | OutputWithCache:
        """Run the target model with optional component replacement and/or caching.

        With no extra args, this is just a forward pass through the frozen target model.
        If `mask_infos` is given, those modules' outputs are replaced by their
        components' forward pass under the supplied masks. Returns an `OutputWithCache`
        when `cache_type != "none"`, else the bare output.

        Args:
            batch: Passed unchanged to the wrapped `run_batch` callable.
            mask_infos: Per-module mask payload. If set, listed modules are replaced via
                forward hooks running the corresponding `Components` instance.
            cache_type: What each hooked module records. `"input"` caches pre-weight
                activations; `"output"` caches post-weight (post-replacement) outputs;
                `"component_acts"` caches per-component activations under the keys
                `f"{module_path}_pre_detach"` / `f"{module_path}_post_detach"`;
                `"none"` disables caching.
        """
        if mask_infos is None and cache_type == "none":
            return self._run_batch(self.target_model, batch)

        cache: dict[str, Tensor] = {}
        hooks: dict[str, Callable[..., Any]] = {}

        hook_module_names = list(mask_infos.keys()) if mask_infos else self.target_module_paths

        for module_name in hook_module_names:
            mask_info = mask_infos[module_name] if mask_infos else None
            components = self.components[module_name] if mask_info else None

            hooks[module_name] = partial(
                self._components_and_cache_hook,
                module_name=module_name,
                components=components,
                mask_info=mask_info,
                cache_type=cache_type,
                cache=cache,
                stop_when_cached=None,
            )

        # Readout sites ride the clean input-caching pass: that is where the hidden-acts
        # losses get their frozen targets from, and it costs nothing beyond the capture.
        if cache_type == "input":
            hooks |= self._readout_hooks(cache, stop_when_cached=None)

        with self._attach_forward_hooks(hooks):
            out: Tensor = self._run_batch(self.target_model, batch)

        match cache_type:
            case "input" | "output" | "component_acts":
                return OutputWithCache(output=out, cache=cache)
            case "none":
                return out

    def site_outputs(
        self,
        batch: Any,
        mask_infos: dict[str, ComponentsMaskInfo],
    ) -> dict[str, Float[Tensor, "..."]]:
        """Masked outputs of the decomposed sites, cutting the forward short once they're all in.

        Site-level losses (hidden-activation reconstruction) read nothing downstream of the
        last decomposition target, so running the rest of the model wastes compute and —
        more importantly under memory pressure — makes autograd retain every downstream
        activation. The pass therefore aborts as soon as the cache holds one entry per
        module in `mask_infos`, which needs no knowledge of execution order. The aborting
        module's output is already cached with its graph intact; only the value PyTorch
        would have propagated onward is discarded.
        """
        cache: dict[str, Tensor] = {}
        n_expected = len(mask_infos) + len(self.hidden_readout_sites)
        hooks: dict[str, Callable[..., Any]] = {
            module_name: partial(
                self._components_and_cache_hook,
                module_name=module_name,
                components=self.components[module_name],
                mask_info=mask_info,
                cache_type="output",
                cache=cache,
                stop_when_cached=n_expected,
            )
            for module_name, mask_info in mask_infos.items()
        }
        hooks |= self._readout_hooks(cache, stop_when_cached=n_expected)
        try:
            with self._attach_forward_hooks(hooks):
                self._run_batch(self.target_model, batch)
        except _SiteCacheComplete:
            pass
        expected = set(mask_infos) | set(self.hidden_readout_sites)
        assert cache.keys() == expected, (
            f"site_outputs cached {sorted(cache)}, expected {sorted(expected)}"
        )
        return cache

    def _readout_hooks(
        self, cache: dict[str, Tensor], stop_when_cached: int | None
    ) -> dict[str, Callable[..., Any]]:
        """Input-capturing hooks for every readout site, keyed by the module to hook."""
        return {
            module_path: partial(
                self._readout_cache_hook,
                readout_name=readout_name,
                cache=cache,
                stop_when_cached=stop_when_cached,
            )
            for readout_name, module_path in self.hidden_readout_sites.items()
        }

    def _components_and_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        output: Any,
        module_name: str,
        components: Components | None,
        mask_info: ComponentsMaskInfo | None,
        cache_type: Literal["component_acts", "input", "output", "none"],
        cache: dict[str, Tensor],
        stop_when_cached: int | None,
    ) -> Any | None:
        """Forward hook handling both component replacement and caching.

        Returns the replaced output when components are applied, else `None` (telling
        PyTorch to keep the original output). When `stop_when_cached` is set, raises
        `_SiteCacheComplete` once the cache holds that many entries — see `site_outputs`.
        """
        assert len(args) == 1, "Expected 1 argument"
        assert len(kwargs) == 0, "Expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), "Expected input tensor"

        if cache_type == "input":
            cache[module_name] = x

        if components is not None and mask_info is not None:
            assert isinstance(output, Tensor), (
                f"Only supports single-tensor outputs, got {type(output)}"
            )

            component_acts_cache = {} if cache_type == "component_acts" else None
            components_out = components(
                x,
                mask=mask_info.component_mask,
                weight_delta_and_mask=mask_info.weight_delta_and_mask,
                component_acts_cache=component_acts_cache,
            )
            if component_acts_cache is not None:
                for k, v in component_acts_cache.items():
                    cache[f"{module_name}_{k}"] = v

            final_out = (
                components_out
                if mask_info.routing_mask == "all"
                else torch.where(mask_info.routing_mask[..., None], components_out, output)
            )

            if stop_when_cached is not None:
                assert module_name not in cache, (
                    f"{module_name} ran twice; site_outputs counts cached sites to decide when "
                    "to stop, which assumes one call per site per forward"
                )
            if cache_type == "output":
                cache[module_name] = final_out
            if stop_when_cached is not None and len(cache) == stop_when_cached:
                raise _SiteCacheComplete
            return final_out

        # No component replacement - keep original output
        if cache_type == "output":
            assert isinstance(output, Tensor)
            cache[module_name] = output
        assert stop_when_cached is None, "site_outputs always replaces components at every site"
        return None

    def _readout_cache_hook(
        self,
        _module: nn.Module,
        args: list[Any],
        kwargs: dict[Any, Any],
        _output: Any,
        readout_name: str,
        cache: dict[str, Tensor],
        stop_when_cached: int | None,
    ) -> None:
        """Cache a readout site's input, leaving the module's own output untouched.

        Returns `None` so PyTorch keeps the original output — a readout site is observed,
        never replaced.
        """
        assert len(args) == 1, f"{readout_name}: expected 1 positional argument, got {len(args)}"
        assert len(kwargs) == 0, f"{readout_name}: expected no keyword arguments"
        x = args[0]
        assert isinstance(x, Tensor), f"{readout_name}: expected input tensor, got {type(x)}"
        assert readout_name not in cache, (
            f"{readout_name} ran twice in one forward; readout sites must execute exactly once"
        )
        cache[readout_name] = x
        if stop_when_cached is not None and len(cache) == stop_when_cached:
            raise _SiteCacheComplete

    @contextmanager
    def _attach_forward_hooks(self, hooks: dict[str, Callable[..., Any]]) -> Generator[None]:
        """Attach forward hooks to the listed target modules for the block's lifetime."""
        handles: list[RemovableHandle] = []
        for module_name, hook in hooks.items():
            target_module = self.target_model.get_submodule(module_name)
            handle = target_module.register_forward_hook(hook, with_kwargs=True)
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool = False,
        role: CIRole = "output",
    ) -> CIOutputs:
        """CI values for every decomposition target.

        Runs the selected CI fn on `pre_weight_acts` and squashes through both lower- and
        upper-leaky sigmoids. Under `sampling="binomial"`, the lower-leaky branch has a
        small amount of uniform noise mixed in before squashing.

        Args:
            pre_weight_acts: Per-module input activations (or token-id tensors for
                embedding targets), typically the cache from a `cache_type="input"`
                forward pass. Only the decomposition targets' entries are read, so a
                cache also carrying `hidden_readout_sites` is accepted as-is.
            sampling: Selects the stochastic mask regime; gates the noise injection on
                the lower-leaky branch.
            detach_inputs: When true, gradients do not flow from CI back into
                `pre_weight_acts`. Used by metrics that want to optimise CI without
                perturbing the upstream graph.
            role: Which CI net to run. `"hidden"` requires the model to have been built
                with `dual_hidden_ci`. Under `hidden_ci_floor` this role additionally runs
                the output net, so that every caller — trainer, eval metrics, and the
                analysis tools alike — sees the floored value the model actually masks with.

        A caller that wants *both* roles should use `calc_causal_importances_both_roles`,
        which shares one binomial jitter draw between them and so preserves the
        `hidden_ci_floor` ordering on the lower-leaky branch as well.
        """
        ci_inputs = self._ci_inputs(pre_weight_acts, detach_inputs)
        ci_fn_outputs = self.ci_fn_for(role)(ci_inputs)
        if role == "hidden" and self.hidden_ci_floor is not None:
            # `no_grad`, not `.detach()`: the floor must not become a path from the hidden
            # objective back into the output net, and there is no reason to build the graph
            # only to throw it away.
            with torch.no_grad():
                output_logits = self.ci_fn(ci_inputs)
            ci_fn_outputs = floor_hidden_ci_logits(
                ci_fn_outputs, output_logits, self.hidden_ci_floor
            )
        if role == "output" and self.output_ci_cap is not None:
            # Deliberately *not* under `no_grad`: for the cap the gradient into the hidden net
            # is the mechanism. This path is analysis-only on a dual run (the trainer uses
            # `calc_causal_importances_both_roles`), so the graph usually costs nothing —
            # callers evaluating under `torch.no_grad()` build none.
            assert self.ci_fn_hidden is not None
            ci_fn_outputs = cap_output_ci_logits(
                ci_fn_outputs, self.ci_fn_hidden(ci_inputs), self.output_ci_cap
            )
        return self._apply_sigmoid_to_ci_outputs(
            ci_fn_outputs, sampling, jitter=self._binomial_jitter(ci_fn_outputs, sampling)
        )

    def calc_causal_importances_both_roles(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        sampling: SamplingType,
        detach_inputs: bool,
    ) -> tuple[CIOutputs, CIOutputs]:
        """`(output, hidden)` CI from one pass, sharing the binomial jitter draw.

        Sharing that draw is what makes `hidden_ci_floor` hold on `lower_leaky` — the branch
        that actually masks components. The floor orders the *logits*, and both squashes are
        monotone, but under `sampling="binomial"` the lower-leaky branch first mixes in
        `-0.05 * rand_like`. Drawn independently per role, that noise inverts the order
        whenever the logit gap is under 0.0476 — which is exactly where the floor binds, so
        the mechanism would fail precisely on the entries it exists for. One shared draw is
        a monotone transform of both, and the ordering survives it exactly.

        Also removes a redundant output-net forward: the floor needs the output logits, and
        this way they are the same tensor the output role returns rather than a second
        evaluation of the same net.
        """
        assert self.ci_fn_hidden is not None, (
            "calc_causal_importances_both_roles needs a model built with dual_hidden_ci"
        )
        ci_inputs = self._ci_inputs(pre_weight_acts, detach_inputs)
        output_logits = self.ci_fn(ci_inputs)
        hidden_logits = self.ci_fn_hidden(ci_inputs)
        if self.hidden_ci_floor is not None:
            hidden_logits = floor_hidden_ci_logits(
                hidden_logits,
                {name: logits.detach() for name, logits in output_logits.items()},
                self.hidden_ci_floor,
            )
        if self.output_ci_cap is not None:
            # Attached, unlike the floor: where the cap binds, the output objective's gradient
            # flows into the hidden logit, which is exactly the guidance being tested.
            output_logits = cap_output_ci_logits(output_logits, hidden_logits, self.output_ci_cap)
        jitter = self._binomial_jitter(output_logits, sampling)
        return (
            self._apply_sigmoid_to_ci_outputs(output_logits, sampling, jitter=jitter),
            self._apply_sigmoid_to_ci_outputs(hidden_logits, sampling, jitter=jitter),
        )

    def _ci_inputs(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        detach_inputs: bool,
    ) -> dict[str, Tensor]:
        ci_inputs = {path: pre_weight_acts[path] for path in self.target_module_paths}
        return {k: v.detach() for k, v in ci_inputs.items()} if detach_inputs else ci_inputs

    @staticmethod
    def _binomial_jitter(
        ci_fn_outputs: dict[str, Float[Tensor, "... C"]], sampling: SamplingType
    ) -> dict[str, Float[Tensor, "... C"]] | None:
        """One uniform draw per module, or `None` outside the binomial regime."""
        if sampling != "binomial":
            return None
        return {name: torch.rand_like(logits) for name, logits in ci_fn_outputs.items()}

    def ci_fn_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Every CI-fn parameter exactly once, named `ci_fn.<...>` / `ci_fn_hidden.<...>`.

        Under a shared trunk the two nets reach the trunk parameters under both names;
        `nn.Module.named_parameters` dedupes each to the `ci_fn.` one it meets first, so
        optimizer param groups and grad-norm sums never count a shared parameter twice.
        """
        return [
            (name, param)
            for name, param in self.named_parameters()
            if name.startswith(("ci_fn.", "ci_fn_hidden."))
        ]

    def ci_fn_for(self, role: CIRole) -> nn.Module:
        """The CI fn scoring importance for `role`."""
        match role:
            case "output":
                return self.ci_fn
            case "hidden":
                assert self.ci_fn_hidden is not None, (
                    "role='hidden' needs a model built with dual_hidden_ci=True"
                )
                return self.ci_fn_hidden

    def _apply_sigmoid_to_ci_outputs(
        self,
        ci_fn_outputs: dict[str, Float[Tensor, "... C"]],
        sampling: SamplingType,
        jitter: dict[str, Float[Tensor, "... C"]] | None,
    ) -> CIOutputs:
        """Squash raw CI-fn outputs through the lower- and upper-leaky sigmoids.

        `jitter` supplies the binomial regime's uniform draw. Passing the *same* jitter for
        both CI roles keeps a logit-space ordering intact through the lower-leaky branch;
        see `calc_causal_importances_both_roles`.
        """
        causal_importances_lower_leaky = {}
        causal_importances_upper_leaky = {}
        pre_sigmoid = {}
        assert (jitter is None) == (sampling != "binomial"), (
            f"jitter is {'absent' if jitter is None else 'present'} under sampling={sampling!r}"
        )

        for target_module_name, ci_fn_output in ci_fn_outputs.items():
            if jitter is not None:
                ci_fn_output_for_lower_leaky = (
                    1.05 * ci_fn_output - 0.05 * jitter[target_module_name]
                )
            else:
                ci_fn_output_for_lower_leaky = ci_fn_output

            lower_leaky_output = self.lower_leaky_fn(ci_fn_output_for_lower_leaky)
            assert (lower_leaky_output <= 1.0).all()
            causal_importances_lower_leaky[target_module_name] = lower_leaky_output

            upper_leaky_output = self.upper_leaky_fn(ci_fn_output)
            assert (upper_leaky_output >= 0).all()
            causal_importances_upper_leaky[target_module_name] = upper_leaky_output

            pre_sigmoid[target_module_name] = ci_fn_output

        return CIOutputs(
            lower_leaky=causal_importances_lower_leaky,
            upper_leaky=causal_importances_upper_leaky,
            pre_sigmoid=pre_sigmoid,
        )

    def calc_weight_deltas(self) -> dict[str, Float[Tensor, "d_out d_in"]]:
        """Per-target `target_weight - sum_components` residuals.

        Used by the delta-component pathway and by faithfulness diagnostics.
        """
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] = {}
        for comp_name, components in self.components.items():
            weight_deltas[comp_name] = self.target_weight(comp_name) - components.weight
        return weight_deltas


def _grad_norm_key(qualified_ci_fn_param: str) -> str:
    """`ci_fn.W` -> `ci_fns/W`; `ci_fn_hidden.W` -> `ci_fns/hidden.W`.

    The output net keeps the unprefixed names single-CI runs have always logged, so these
    keys stay comparable across runs whether or not a hidden net exists.
    """
    if qualified_ci_fn_param.startswith("ci_fn_hidden."):
        return f"ci_fns/hidden.{qualified_ci_fn_param.removeprefix('ci_fn_hidden.')}"
    return f"ci_fns/{qualified_ci_fn_param.removeprefix('ci_fn.')}"


def component_grad_norms(
    component_model: ComponentModel, device: torch.device | str
) -> dict[str, float]:
    """Per-parameter and summary gradient norms for components and the CI fn.

    Returns a flat dict with three key families:

    - `components/<module_path>.<param>` — L2 norm of each component parameter's
      gradient. `NaN` if its grad was never populated.
    - `ci_fns/<param>` — L2 norm of each CI-fn parameter's gradient. `NaN` if its grad
      was never populated. The hidden CI net (when present) reports under
      `ci_fns/hidden.<param>`, so per-net norms stay separable. Trunk parameters shared
      between the two nets report once, under the unprefixed output-net name.
    - `summary/components`, `summary/ci_fns`, `summary/total` — aggregate L2 norms over
      each pool and over both pools. `NaN` if any contributing grad was missing.
    """
    out: dict[str, float] = {}

    comp_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    missing_component_grad = False
    for target_module_path, component in component_model.components.items():
        for local_param_name, local_param in component.named_parameters():
            if local_param.grad is None:
                missing_component_grad = True
                out[f"components/{target_module_path}.{local_param_name}"] = float("nan")
                continue
            param_grad = runtime_cast(Tensor, local_param.grad)
            param_grad_sum_sq = param_grad.pow(2).sum()
            key = f"components/{target_module_path}.{local_param_name}"
            out[key] = param_grad_sum_sq.sqrt().item()
            comp_grad_norm_sq_sum += param_grad_sum_sq

    ci_fn_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    missing_ci_fn_grad = False
    for qualified_name, local_param in component_model.ci_fn_named_parameters():
        key = _grad_norm_key(qualified_name)
        assert key not in out, f"Key {key} already exists in grad norms log"
        if local_param.grad is None:
            missing_ci_fn_grad = True
            out[key] = float("nan")
            continue
        ci_fn_grad = runtime_cast(Tensor, local_param.grad)
        ci_fn_grad_sum_sq = ci_fn_grad.pow(2).sum()
        out[key] = ci_fn_grad_sum_sq.sqrt().item()
        ci_fn_grad_norm_sq_sum += ci_fn_grad_sum_sq

    out["summary/components"] = (
        float("nan") if missing_component_grad else comp_grad_norm_sq_sum.sqrt().item()
    )
    out["summary/ci_fns"] = (
        float("nan") if missing_ci_fn_grad else ci_fn_grad_norm_sq_sum.sqrt().item()
    )
    out["summary/total"] = (
        float("nan")
        if missing_component_grad or missing_ci_fn_grad
        else (comp_grad_norm_sq_sum + ci_fn_grad_norm_sq_sum).sqrt().item()
    )
    return out
