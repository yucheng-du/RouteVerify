"""Gemma 4 adapter for the native Transformers multimodal MoE layout."""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any

from routeverify.config import ModelConfig
from routeverify.instrumentation.patching import router_manipulation_metrics
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    MultiPositionPatchSpec,
    PatchBundle,
    PatchRequest,
    PatchSiteSpec,
    SiteKind,
    patch_index,
    patch_positions,
    patch_specs,
    site_key,
)
from routeverify.models.base import ActivationBatch
from routeverify.models.moe import MoEConfigPaths, RouterOutput, RouterSemantics
from routeverify.models.olmoe import OLMoEAdapter, _require_torch, _torch_dtype

GEMMA4_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
GEMMA4_TRANSFORMERS_VERSION = "5.12.1"


def _gemma4_route_from_logits(router_module: Any, router_logits: Any) -> RouterOutput:
    """Reproduce Gemma 4's native softmax, top-k, and expert scaling."""

    torch = _require_torch()
    config = getattr(router_module, "config", None)
    top_k = getattr(config, "top_k_experts", None)
    if top_k is None:
        top_k = getattr(router_module, "top_k", None)
    if top_k is None:
        raise RuntimeError("Gemma 4 router does not expose top_k_experts")
    per_expert_scale = getattr(router_module, "per_expert_scale", None)
    if per_expert_scale is None:
        raise RuntimeError("Gemma 4 router does not expose per_expert_scale")

    probabilities = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, k=int(top_k), dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * per_expert_scale[topk_ids]
    return RouterOutput(
        logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


class Gemma4MoEAdapter(OLMoEAdapter):
    """Adapter for Gemma 4's parallel dense and routed FFN branches.

    The multimodal ``Gemma4ForConditionalGeneration`` wrapper exposes its text
    decoder at ``model.language_model.layers``. Within each decoder layer, the
    dense ``mlp`` and routed ``experts`` branches run in parallel from the
    pre-FFN residual. The routed contribution boundary is the output of
    ``post_feedforward_layernorm_2``, immediately before it is added to the
    dense branch.

    ``Gemma4TextRouter`` returns probabilities as its first output, despite
    Transformers recording that value under the name ``router_logits``. Raw
    router scores are therefore captured and patched at ``router.proj``. This
    leaves the native softmax, top-k selection, selected-weight normalization,
    per-expert scaling, and expert dispatch unchanged.
    """

    alias = "gemma4_moe"
    model_id = "google/gemma-4-26B-A4B-it"
    revision = GEMMA4_REVISION
    moe_config_paths = MoEConfigPaths(
        total_blocks=("text_config.num_hidden_layers",),
        experts=("text_config.num_experts",),
        top_k=("text_config.top_k_experts",),
    )
    router_semantics = RouterSemantics(
        identifier="gemma4_softmax_topk_per_expert_scale_v1",
        route_from_logits=_gemma4_route_from_logits,
    )

    def load(self, config: ModelConfig) -> None:
        """Load only the pinned Gemma 4 model and tokenizer revision."""

        if config.model_id != self.model_id:
            raise ValueError(
                f"Gemma 4 adapter requires model_id={self.model_id}; got {config.model_id}"
            )
        revisions = {"model": config.revision, "tokenizer": config.tokenizer_revision}
        for label, revision in revisions.items():
            if revision != self.revision or re.fullmatch(r"[0-9a-f]{40}", revision or "") is None:
                raise ValueError(
                    f"Gemma 4 {label} revision must equal pinned commit {self.revision}"
                )
        if config.expected_transformers_version != GEMMA4_TRANSFORMERS_VERSION:
            raise ValueError(
                "Gemma 4 expected_transformers_version must equal the adapter contract "
                f"{GEMMA4_TRANSFORMERS_VERSION}"
            )
        try:
            installed_transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Gemma 4 runtime requires transformers={GEMMA4_TRANSFORMERS_VERSION}"
            ) from exc
        if installed_transformers_version != GEMMA4_TRANSFORMERS_VERSION:
            raise RuntimeError(
                "Gemma 4 Transformers version mismatch: "
                f"actual={installed_transformers_version} "
                f"expected={GEMMA4_TRANSFORMERS_VERSION}"
            )
        super().load(config)
        # Fail immediately if a future upstream wrapper or decoder layout no
        # longer matches the semantics implemented below.
        self._validate_moe_layout()
        if self.discovery_report is not None:
            self.discovery_report.metadata["runtime_inspect"] = self.runtime_inspect()

    def discover_layers(self) -> list[LayerSite]:
        self._require_model()
        sites: list[LayerSite] = []
        for layer_index, layer in enumerate(self._layers()):
            if not self._is_moe_layer(layer):
                continue
            self._validate_layer_layout(layer_index, layer)
            for kind in (
                "pre_moe_residual",
                "router_input",
                "router_logits",
                "router_topk_ids",
                "router_topk_weights",
                "moe_output",
                "post_moe_residual",
            ):
                sites.append(
                    LayerSite(
                        layer_index=layer_index,
                        kind=kind,
                        module_name=self._module_name(layer_index, kind),
                    )
                )
        if not sites:
            raise RuntimeError("loaded Gemma 4 text model does not expose any MoE decoder layers")
        return sites

    def _layers(self) -> Any:
        text_model, _prefix = self._text_model_and_prefix()
        return text_model.layers

    def _text_model_and_prefix(self) -> tuple[Any, str]:
        """Resolve both the multimodal wrapper and direct text-only variants."""

        model = self._require_model()
        candidates = (
            (("model", "language_model"), "model.language_model"),
            (("model",), "model"),
            (("language_model",), "language_model"),
            ((), ""),
        )
        for attributes, prefix in candidates:
            current = model
            for attribute in attributes:
                current = getattr(current, attribute, None)
                if current is None:
                    break
            if current is not None and hasattr(current, "layers"):
                return current, prefix
        raise RuntimeError(
            "loaded Gemma 4 model does not expose text layers through "
            "model.language_model.layers or a supported text-only path"
        )

    def _layer_prefix(self) -> str:
        _text_model, prefix = self._text_model_and_prefix()
        return f"{prefix}.layers" if prefix else "layers"

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        prefix = f"{self._layer_prefix()}.{layer_index}"
        if kind == "pre_moe_residual":
            return f"{prefix}.pre_feedforward_layernorm"
        if kind in {"router_input", "router_logits"}:
            # router_input is the normalized/scaled activation consumed by the
            # raw-score projection; router_logits is that projection's output.
            return f"{prefix}.router.proj"
        if kind in {"router_topk_ids", "router_topk_weights"}:
            # These are observed from the arguments actually dispatched here.
            return f"{prefix}.experts"
        if kind == "moe_output":
            # Native Gemma normalizes the reshaped routed expert sum before
            # adding it to the parallel always-on dense branch.
            return f"{prefix}.post_feedforward_layernorm_2"
        if kind == "post_moe_residual":
            # The semantic value is pre-FF residual + this module's output.
            # Capture/patch compose that addition explicitly before Gemma's
            # subsequent PLE residual.
            return f"{prefix}.post_feedforward_layernorm"
        raise ValueError(f"unsupported site kind: {kind}")

    def _site_module(self, site: LayerSite) -> Any:
        layer = self._layer(site.layer_index)
        self._validate_layer_layout(site.layer_index, layer)
        if site.kind == "pre_moe_residual":
            return layer.pre_feedforward_layernorm
        if site.kind in {"router_input", "router_logits"}:
            return layer.router.proj
        if site.kind in {"router_topk_ids", "router_topk_weights"}:
            return layer.experts
        if site.kind == "moe_output":
            return layer.post_feedforward_layernorm_2
        if site.kind == "post_moe_residual":
            return layer.post_feedforward_layernorm
        raise ValueError(f"unsupported site kind: {site.kind}")

    def _capture_inputs(
        self, inputs: dict[str, Any], spec: CaptureSpec
    ) -> tuple[ActivationBatch, Any]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        self._validate_sequence_positions(inputs, spec.token_positions, label="capture")
        arrays: dict[str, Any] = {}
        fire_counts = {site_key(site): 0 for site in spec.sites}
        handles: list[Any] = []
        post_residual_inputs: dict[int, Any] = {}

        def store(site: LayerSite, tensor: Any) -> None:
            sequence_tensor = self._as_sequence_tensor(tensor, batch_size, sequence_length)
            selected = sequence_tensor[:, list(spec.token_positions), ...].detach()
            if selected.is_floating_point():
                selected = selected.to(_torch_dtype(spec.dtype))
            arrays[site_key(site)] = selected.cpu()
            fire_counts[site_key(site)] += 1

        dispatch_sites: dict[int, list[LayerSite]] = {}
        post_residual_sites: dict[int, list[LayerSite]] = {}
        try:
            for site in spec.sites:
                module = self._site_module(site)
                if site.kind == "pre_moe_residual":

                    def pre_moe_hook(
                        _module: Any,
                        args: tuple[Any, ...],
                        *,
                        site: LayerSite = site,
                    ) -> None:
                        store(site, args[0])

                    handles.append(module.register_forward_pre_hook(pre_moe_hook))
                elif site.kind == "router_input":

                    def router_input_hook(
                        _module: Any,
                        args: tuple[Any, ...],
                        *,
                        site: LayerSite = site,
                    ) -> None:
                        store(site, args[0])

                    handles.append(module.register_forward_pre_hook(router_input_hook))
                elif site.kind == "router_logits":

                    def router_logits_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        site: LayerSite = site,
                    ) -> None:
                        store(site, output)

                    handles.append(module.register_forward_hook(router_logits_hook))
                elif site.kind in {"router_topk_ids", "router_topk_weights"}:
                    dispatch_sites.setdefault(site.layer_index, []).append(site)
                elif site.kind == "moe_output":

                    def moe_output_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        site: LayerSite = site,
                    ) -> None:
                        store(site, output)

                    handles.append(module.register_forward_hook(moe_output_hook))
                elif site.kind == "post_moe_residual":
                    post_residual_sites.setdefault(site.layer_index, []).append(site)
                else:  # pragma: no cover - exhaustive for current SiteKind
                    raise ValueError(f"unsupported Gemma 4 capture site: {site.kind}")

            for layer_index, sites in dispatch_sites.items():
                experts = self._layer(layer_index).experts

                def dispatch_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    sites: list[LayerSite] = sites,
                ) -> None:
                    top_k_ids, top_k_weights = self._dispatch_artifacts(args, kwargs)
                    for site in sites:
                        store(
                            site,
                            top_k_ids if site.kind == "router_topk_ids" else top_k_weights,
                        )

                handles.append(experts.register_forward_pre_hook(dispatch_hook, with_kwargs=True))

            for layer_index, sites in post_residual_sites.items():
                layer = self._layer(layer_index)

                def save_residual(
                    _module: Any,
                    args: tuple[Any, ...],
                    *,
                    layer_index: int = layer_index,
                ) -> None:
                    post_residual_inputs[layer_index] = args[0]

                def post_residual_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    layer_index: int = layer_index,
                    sites: list[LayerSite] = sites,
                ) -> None:
                    residual = post_residual_inputs.get(layer_index)
                    if residual is None:
                        raise RuntimeError(
                            f"Gemma 4 layer {layer_index} post-FF residual input was not observed"
                        )
                    for site in sites:
                        store(site, residual + output)

                handles.append(
                    layer.pre_feedforward_layernorm.register_forward_pre_hook(save_residual)
                )
                handles.append(
                    layer.post_feedforward_layernorm.register_forward_hook(post_residual_hook)
                )

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, output_router_logits=False)
            logits = outputs.logits.detach().cpu()
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(key for key, count in fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered Gemma 4 capture hooks did not fire: {missing}")
        metadata = {
            "logits_shape": tuple(outputs.logits.shape),
            "hook_fire_counts": fire_counts,
            "all_registered_hooks_fired": not missing,
            "router_logits_semantics": "raw_pre_softmax_scores",
            "router_topk_semantics": "actual_expert_dispatch_arguments",
            "moe_output_semantics": "normalized_routed_branch_only",
            "post_moe_residual_semantics": "post_ffn_residual_before_native_ple",
        }
        return ActivationBatch(arrays=arrays, metadata=metadata), logits

    def _run_with_patch_inputs_diagnostics(
        self,
        inputs: dict[str, Any],
        request: PatchRequest,
    ) -> tuple[Any, dict[str, Any]]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        patches = patch_specs(request)
        hook_fire_counts = {site_key(patch.site): 0 for patch in patches}
        manipulation_by_site: dict[str, dict[str, float]] = {}
        handles: list[Any] = []

        router_output_fire_counts: dict[str, int] = {}
        dispatch_fire_counts: dict[str, int] = {}
        router_after_patch: dict[str, bool] = {}
        dispatch_after_router: dict[str, bool] = {}
        native_topk_matches: dict[str, bool | None] = {}
        dispatch_topk_matches: dict[str, bool | None] = {}
        dispatch_ids_match_router: dict[str, bool | None] = {}
        dispatch_weights_match_router: dict[str, bool | None] = {}
        expected_topk: dict[str, Any] = {}
        native_router_artifacts: dict[str, tuple[Any, Any]] = {}
        post_residual_inputs: dict[int, Any] = {}

        try:
            for patch in patches:
                self._validate_sequence_positions(
                    inputs, patch_positions(patch), label=site_key(patch.site)
                )
                key = site_key(patch.site)
                layer = self._layer(patch.site.layer_index)
                self._validate_layer_layout(patch.site.layer_index, layer)

                if patch.site.kind == "router_logits":
                    router = layer.router
                    router_output_fire_counts[key] = 0
                    dispatch_fire_counts[key] = 0
                    router_after_patch[key] = False
                    dispatch_after_router[key] = False
                    native_topk_matches[key] = None
                    dispatch_topk_matches[key] = None
                    dispatch_ids_match_router[key] = None
                    dispatch_weights_match_router[key] = None

                    def raw_router_patch_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        router_logits: Any,
                        *,
                        patch: PatchSiteSpec = patch,
                        key: str = key,
                        router: Any = router,
                    ) -> Any:
                        hook_fire_counts[key] += 1
                        receiver_sequence = self._as_sequence_tensor(
                            router_logits, batch_size, sequence_length
                        )
                        receiver_target = receiver_sequence[:, patch_index(patch), ...]
                        receiver_ids = self.router_semantics.route(
                            router,
                            receiver_target,
                        ).topk_ids
                        if bool(patch.metadata.get("forced_reroute")):
                            if not receiver_target.dtype.is_floating_point:
                                raise TypeError(
                                    f"{key} forced reroute requires floating router logits"
                                )
                            limits = torch.finfo(receiver_target.dtype)
                            selected = torch.zeros_like(receiver_target, dtype=torch.bool)
                            selected.scatter_(-1, receiver_ids, True)
                            replacement = torch.where(
                                selected,
                                torch.full_like(receiver_target, limits.min),
                                torch.full_like(receiver_target, limits.max),
                            )
                            patched_sequence = receiver_sequence.clone()
                            patched_sequence[:, patch_index(patch), ...] = replacement
                        else:
                            patched_sequence = self._replace_sequence_slice(
                                router_logits, patch, batch_size, sequence_length
                            )
                        patched_logits = patched_sequence.reshape_as(router_logits)
                        patched_target = patched_sequence[:, patch_index(patch), ...]
                        patched_ids = self.router_semantics.route(
                            router,
                            patched_target,
                        ).topk_ids
                        expected_topk[key] = patched_ids.detach().clone()
                        manipulation_by_site[key] = router_manipulation_metrics(
                            receiver_target.detach().float().cpu().numpy(),
                            patched_target.detach().float().cpu().numpy(),
                            receiver_topk_ids=receiver_ids.detach().cpu().numpy(),
                            patched_topk_ids=patched_ids.detach().cpu().numpy(),
                        )
                        return patched_logits

                    def native_router_output_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        key: str = key,
                        patch: PatchSiteSpec = patch,
                    ) -> None:
                        if not isinstance(output, tuple) or len(output) != 3:
                            raise RuntimeError(
                                "Gemma 4 router output must be "
                                "(probabilities, top_k_weights, top_k_ids)"
                            )
                        _probabilities, top_k_weights, top_k_ids = output
                        router_output_fire_counts[key] += 1
                        router_after_patch[key] = hook_fire_counts[key] > 0
                        expected = expected_topk.get(key)
                        actual = self._as_sequence_tensor(top_k_ids, batch_size, sequence_length)[
                            :, patch_index(patch), ...
                        ]
                        native_topk_matches[key] = bool(
                            expected is not None
                            and torch.equal(actual.to(expected.device), expected)
                        )
                        native_router_artifacts[key] = (top_k_ids, top_k_weights)

                    def expert_dispatch_hook(
                        _module: Any,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                        *,
                        key: str = key,
                        patch: PatchSiteSpec = patch,
                    ) -> None:
                        dispatch_fire_counts[key] += 1
                        dispatch_after_router[key] = (
                            hook_fire_counts[key] > 0
                            and router_output_fire_counts[key] > 0
                            and router_after_patch[key]
                        )
                        top_k_ids, top_k_weights = self._dispatch_artifacts(args, kwargs)
                        expected = expected_topk.get(key)
                        actual = self._as_sequence_tensor(top_k_ids, batch_size, sequence_length)[
                            :, patch_index(patch), ...
                        ]
                        dispatch_topk_matches[key] = bool(
                            expected is not None
                            and torch.equal(actual.to(expected.device), expected)
                        )
                        native = native_router_artifacts.get(key)
                        if native is not None:
                            native_ids, native_weights = native
                            dispatch_ids_match_router[key] = bool(
                                top_k_ids is native_ids or torch.equal(top_k_ids, native_ids)
                            )
                            dispatch_weights_match_router[key] = bool(
                                top_k_weights is native_weights
                                or torch.equal(top_k_weights, native_weights)
                            )

                    handles.append(router.proj.register_forward_hook(raw_router_patch_hook))
                    handles.append(router.register_forward_hook(native_router_output_hook))
                    handles.append(
                        layer.experts.register_forward_pre_hook(
                            expert_dispatch_hook, with_kwargs=True
                        )
                    )
                elif patch.site.kind == "moe_output":

                    def moe_output_patch_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        patch: PatchSiteSpec = patch,
                        key: str = key,
                    ) -> Any:
                        hook_fire_counts[key] += 1
                        return self._replace_sequence_slice(
                            output, patch, batch_size, sequence_length
                        ).reshape_as(output)

                    handles.append(
                        layer.post_feedforward_layernorm_2.register_forward_hook(
                            moe_output_patch_hook
                        )
                    )
                elif patch.site.kind == "post_moe_residual":

                    def save_post_residual_input(
                        _module: Any,
                        args: tuple[Any, ...],
                        *,
                        layer_index: int = patch.site.layer_index,
                    ) -> None:
                        post_residual_inputs[layer_index] = args[0]

                    def post_residual_patch_hook(
                        _module: Any,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        patch: PatchSiteSpec = patch,
                        key: str = key,
                    ) -> Any:
                        residual = post_residual_inputs.get(patch.site.layer_index)
                        if residual is None:
                            raise RuntimeError(
                                f"Gemma 4 layer {patch.site.layer_index} post-FF residual "
                                "input was not observed"
                            )
                        hook_fire_counts[key] += 1
                        return self._replace_post_residual_output(
                            output,
                            residual,
                            patch,
                            batch_size,
                            sequence_length,
                        )

                    handles.append(
                        layer.pre_feedforward_layernorm.register_forward_pre_hook(
                            save_post_residual_input
                        )
                    )
                    handles.append(
                        layer.post_feedforward_layernorm.register_forward_hook(
                            post_residual_patch_hook
                        )
                    )
                else:
                    raise ValueError(
                        f"patching site {patch.site.kind!r} is not implemented for Gemma 4"
                    )

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, output_router_logits=False)
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(key for key, count in hook_fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered Gemma 4 patch hooks did not fire: {missing}")

        dispatch_by_site: dict[str, dict[str, Any]] = {}
        for patch in patches:
            if patch.site.kind != "router_logits":
                continue
            key = site_key(patch.site)
            downstream_verified = bool(
                dispatch_fire_counts[key] > 0
                and dispatch_after_router[key]
                and native_topk_matches[key] is True
                and dispatch_topk_matches[key] is True
                and dispatch_ids_match_router[key] is True
                and dispatch_weights_match_router[key] is True
            )
            dispatch_by_site[key] = {
                "router_execution_mode": "native_softmax_topk",
                "router_output_hook_fire_count": router_output_fire_counts[key],
                "expert_dispatch_hook_fire_count": dispatch_fire_counts[key],
                "router_output_after_raw_logit_patch": router_after_patch[key],
                "dispatch_after_router_intercept": dispatch_after_router[key],
                "native_topk_matches_patched_topk": native_topk_matches[key],
                "actual_topk_matches_patched_topk": dispatch_topk_matches[key],
                "dispatch_ids_match_native_router": dispatch_ids_match_router[key],
                "dispatch_weights_match_native_router": dispatch_weights_match_router[key],
                "downstream_dispatch_verified": downstream_verified,
            }

        diagnostics: dict[str, Any] = {
            "simultaneous": isinstance(request, PatchBundle)
            or any(len(patch_positions(patch)) > 1 for patch in patches),
            "token_positions_by_site": {
                site_key(patch.site): list(patch_positions(patch)) for patch in patches
            },
            "registered_site_count": len(patches),
            "registered_sites": [site_key(patch.site) for patch in patches],
            "hook_fire_counts": hook_fire_counts,
            "all_registered_hooks_fired": not missing,
            "manipulation_by_site": manipulation_by_site,
            "dispatch_by_site": dispatch_by_site,
            "all_router_dispatch_verified": all(
                bool(value["downstream_dispatch_verified"]) for value in dispatch_by_site.values()
            ),
        }
        return outputs, diagnostics

    def _replace_post_residual_output(
        self,
        branch_output: Any,
        residual: Any,
        patch: PatchSiteSpec,
        batch_size: int,
        sequence_length: int,
    ) -> Any:
        """Patch the pre-PLE ``residual + branch_output`` boundary exactly."""

        torch = _require_torch()
        branch_sequence = self._as_sequence_tensor(branch_output, batch_size, sequence_length)
        residual_sequence = self._as_sequence_tensor(residual, batch_size, sequence_length)
        post_residual = residual_sequence + branch_sequence
        target = post_residual[:, patch_index(patch), ...]
        replacement = self._replacement_for_target(
            patch.replacement,
            target,
            strict_shape=patch.strict_shape,
            preserve_token_axis=isinstance(patch, MultiPositionPatchSpec),
        )

        # Preserve exact identity when a capture from this same input is patched
        # back. Otherwise subtraction/re-addition can introduce low-precision
        # roundoff despite a semantic no-op.
        if torch.equal(replacement, target):
            return branch_output

        patched_branch = branch_sequence.clone()
        # A mapped window can mix identity and changed positions. Keep each
        # identity vector exact even when another target in this window changes.
        same_vector = torch.eq(replacement, target).all(dim=-1, keepdim=True)
        patched_branch[:, patch_index(patch), ...] = torch.where(
            same_vector,
            branch_sequence[:, patch_index(patch), ...],
            replacement - residual_sequence[:, patch_index(patch), ...],
        )
        return patched_branch.reshape_as(branch_output)

    def _validate_moe_layout(self) -> None:
        runtime = self.runtime_inspect()
        actual = runtime["actual_contract"]
        if not runtime["layout_consistent"]:
            raise RuntimeError("loaded Gemma 4 text layers contradict the native MoE layout")

        config = self.config
        expected = {
            "text_layers": (
                None if config is None else config.expected_total_block_count
            ),
            "routed_layers": (
                None if config is None else config.expected_router_block_count
            ),
            "experts": None if config is None else config.expected_experts,
            "top_k": None if config is None else config.expected_top_k,
        }
        mismatches = {
            key: (actual[key], value)
            for key, value in expected.items()
            if value is not None and int(actual[key]) != int(value)
        }
        if mismatches:
            raise RuntimeError(
                f"loaded Gemma 4 runtime contract does not match pinned expectations: {mismatches}"
            )

        if config is not None:
            if (
                config.expected_model_type is not None
                and runtime["model_type"] != config.expected_model_type
            ):
                raise RuntimeError(
                    "loaded Gemma 4 runtime model type mismatch: "
                    f"actual={runtime['model_type']} expected={config.expected_model_type}"
                )
            if config.expected_architectures is not None and set(runtime["architectures"]) != set(
                config.expected_architectures
            ):
                raise RuntimeError(
                    "loaded Gemma 4 runtime architectures mismatch: "
                    f"actual={runtime['architectures']} "
                    f"expected={config.expected_architectures}"
                )

    def runtime_inspect(self) -> dict[str, Any]:
        """Describe and verify the actual loaded routed-text layout."""

        model = self._require_model()
        text_model, prefix = self._text_model_and_prefix()
        layers = list(text_model.layers)
        layer_rows: list[dict[str, Any]] = []
        expert_counts: set[int] = set()
        top_ks: set[int] = set()
        routed_layers = 0
        layout_consistent = True

        for layer_index, layer in enumerate(layers):
            routed = self._is_moe_layer(layer)
            row: dict[str, Any] = {
                "layer_index": layer_index,
                "routed": routed,
                "layer_class": self._class_path(layer),
            }
            if routed:
                routed_layers += 1
                try:
                    self._validate_layer_layout(layer_index, layer)
                    router_experts = self._router_expert_count(layer.router)
                    expert_module_count = int(getattr(layer.experts, "num_experts", router_experts))
                    top_k = self._router_top_k(layer.router)
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    layout_consistent = False
                    row["layout_error"] = str(exc)
                else:
                    expert_counts.add(router_experts)
                    top_ks.add(top_k)
                    if expert_module_count != router_experts:
                        layout_consistent = False
                    row.update(
                        {
                            "router_class": self._class_path(layer.router),
                            "experts_class": self._class_path(layer.experts),
                            "router_num_experts": router_experts,
                            "expert_module_count": expert_module_count,
                            "router_top_k": top_k,
                            "parallel_dense_routed": layer.mlp is not layer.experts,
                        }
                    )
            else:
                layout_consistent = False
            layer_rows.append(row)

        layout_consistent = bool(
            layout_consistent
            and layers
            and routed_layers == len(layers)
            and len(expert_counts) == 1
            and len(top_ks) == 1
        )
        outer_config = getattr(model, "config", None)
        architectures = list(getattr(outer_config, "architectures", None) or [])
        return {
            "schema_version": 1,
            "adapter": self.alias,
            "model_id": str(getattr(outer_config, "_name_or_path", self.model_id)),
            "model_type": str(getattr(outer_config, "model_type", "unknown")),
            "architectures": architectures,
            "text_model_prefix": prefix,
            "actual_contract": {
                "text_layers": len(layers),
                "routed_layers": routed_layers,
                "experts": next(iter(expert_counts), -1) if len(expert_counts) == 1 else -1,
                "top_k": next(iter(top_ks), -1) if len(top_ks) == 1 else -1,
            },
            "layout_consistent": layout_consistent,
            "all_text_layers_routed": routed_layers == len(layers),
            "layers": layer_rows,
        }

    @staticmethod
    def _router_expert_count(router: Any) -> int:
        proj = router.proj
        out_features = getattr(proj, "out_features", None)
        if out_features is not None:
            return int(out_features)
        weight = getattr(proj, "weight", None)
        if weight is None or len(weight.shape) < 1:
            raise RuntimeError("Gemma 4 router projection does not expose expert count")
        return int(weight.shape[0])

    @staticmethod
    def _class_path(value: Any) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"

    @staticmethod
    def _is_moe_layer(layer: Any) -> bool:
        enabled = getattr(layer, "enable_moe_block", None)
        if enabled is False:
            return False
        return enabled is True or hasattr(layer, "router") or hasattr(layer, "experts")

    @staticmethod
    def _validate_layer_layout(layer_index: int, layer: Any) -> None:
        required = (
            "mlp",
            "pre_feedforward_layernorm",
            "router",
            "experts",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
            "post_feedforward_layernorm",
        )
        missing = [name for name in required if not hasattr(layer, name)]
        if missing:
            raise RuntimeError(
                f"Gemma 4 text layer {layer_index} does not match the native parallel "
                f"dense+routed layout; missing={missing}"
            )
        if not hasattr(layer.router, "proj"):
            raise RuntimeError(
                f"Gemma 4 text layer {layer_index} router does not expose raw-score proj"
            )
        if layer.mlp is layer.experts:
            raise RuntimeError(
                f"Gemma 4 text layer {layer_index} dense and routed branches are not distinct"
            )

    @staticmethod
    def _router_top_k(router: Any) -> int:
        config = getattr(router, "config", None)
        value = getattr(config, "top_k_experts", None)
        if value is None:
            value = getattr(router, "top_k", None)
        if value is None:
            raise RuntimeError("Gemma 4 router does not expose top_k_experts")
        return int(value)

    @staticmethod
    def _dispatch_artifacts(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
        top_k_ids = args[1] if len(args) > 1 else kwargs.get("top_k_index")
        top_k_weights = args[2] if len(args) > 2 else kwargs.get("top_k_weights")
        if top_k_ids is None or top_k_weights is None:
            raise RuntimeError("Gemma 4 expert dispatch did not expose top_k_index/top_k_weights")
        return top_k_ids, top_k_weights
