"""Pinned Qwen3.6-35B-A3B text-runtime adapter."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from collections.abc import Mapping
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.instrumentation.sites import LayerSite, SiteKind
from routeverify.models.base import ActivationBatch
from routeverify.models.moe import (
    MoEBlockLayout,
    MoEConfigPaths,
    RouterOutput,
    RouterSemantics,
    build_moe_sites,
    preflight_moe_layout,
    resolve_moe_config,
)
from routeverify.models.olmoe import _require_torch
from routeverify.models.qwen3_moe import Qwen3MoEAdapter

QWEN36_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
QWEN36_TRANSFORMERS_VERSION = "5.12.1"
QWEN36_INDEX_KEY_COUNTS = {
    "total": 1045,
    "language_model": 692,
    "lm_head": 1,
    "visual": 333,
    "mtp": 19,
}


def _qwen36_route_from_logits(router_module: Any, router_logits: Any) -> RouterOutput:
    """Reproduce Qwen3.6's unconditional selected-top-k renormalization."""

    torch = _require_torch()
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_weights, router_ids = torch.topk(router_probs, int(router_module.top_k), dim=-1)
    router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
    return RouterOutput(
        logits=router_logits,
        topk_weights=router_weights.to(router_logits.dtype),
        topk_ids=router_ids,
    )


class Qwen36MoEAdapter(Qwen3MoEAdapter):
    """Adapter for the exact Qwen3.6 composite checkpoint's text-only runtime.

    Transformers 5.12.1 intentionally unwraps the composite
    ``Qwen3_5MoeForConditionalGeneration`` checkpoint when it is loaded through
    ``AutoModelForCausalLM``. Runtime decoder paths are therefore
    ``model.layers``. Every text block is routed; the alternating
    Gated-DeltaNet/full-attention layout affects only the token mixer.

    The common endpoint is the weighted output of ``mlp.experts`` before the
    sigmoid-gated shared expert is added. This matches the accepted routed-only
    common-core estimand and must not be described as the complete MLP output.
    """

    alias = "qwen36_moe"
    model_id = "Qwen/Qwen3.6-35B-A3B"
    pinned_revision = QWEN36_REVISION
    pinned_transformers_version = QWEN36_TRANSFORMERS_VERSION
    capture_loading_info = True
    moe_config_paths = MoEConfigPaths(
        total_blocks=("text_config.num_hidden_layers",),
        experts=("text_config.num_experts",),
        top_k=("text_config.num_experts_per_tok",),
    )
    router_semantics = RouterSemantics(
        identifier="qwen3_5_moe_softmax_topk_renorm_v1",
        route_from_logits=_qwen36_route_from_logits,
    )

    @classmethod
    def static_config_preflight(
        cls,
        model_config: ModelConfig,
        hf_config: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        """Validate the composite config and text site layout without weights."""

        cls._validate_requested_config(model_config)
        resolved = resolve_moe_config(hf_config, cls.moe_config_paths)
        if resolved.total_blocks is None:
            raise RuntimeError("Qwen3.6 config does not expose text_config.num_hidden_layers")
        text_config = _config_child(hf_config, "text_config")
        layer_types = tuple(str(value) for value in _config_value(text_config, "layer_types", ()))
        layout = MoEBlockLayout(
            total_block_count=resolved.total_blocks,
            router_block_indices=tuple(range(resolved.total_blocks)),
        )
        adapter = cls()
        report = preflight_moe_layout(
            model_config=model_config,
            hf_config=hf_config,
            layout=layout,
            sites=build_moe_sites(layout, adapter._module_name),
            router_semantics_id=adapter.router_semantics.identifier,
            config_paths=adapter.moe_config_paths,
            adapter_class=f"{cls.__module__}.{cls.__qualname__}",
            adapter_alias=cls.alias,
            adapter_model_id=cls.model_id,
        )
        errors = list(report["errors"])
        expected_layer_types = tuple(
            str(value)
            for value in (getattr(model_config, "expected_layer_types", ()) or ())
        )
        if len(layer_types) != resolved.total_blocks:
            errors.append(
                "Qwen3.6 layer_types length mismatch: "
                f"actual={len(layer_types)} expected={resolved.total_blocks}"
            )
        invalid_layer_types = sorted(set(layer_types) - {"linear_attention", "full_attention"})
        if invalid_layer_types:
            errors.append(f"unsupported Qwen3.6 layer types: {invalid_layer_types}")
        if expected_layer_types and layer_types != expected_layer_types:
            errors.append("Qwen3.6 layer_types do not match the pinned hybrid layout")

        hook_facts = {
            "text_model_type": _config_value(text_config, "model_type"),
            "hidden_size": _optional_int(_config_value(text_config, "hidden_size")),
            "shared_expert_intermediate_size": _optional_int(
                _config_value(text_config, "shared_expert_intermediate_size")
            ),
            "moe_intermediate_size": _optional_int(
                _config_value(text_config, "moe_intermediate_size")
            ),
            "mtp_num_hidden_layers": _optional_int(
                _config_value(text_config, "mtp_num_hidden_layers")
            ),
            "shared_experts": int(
                int(_config_value(text_config, "shared_expert_intermediate_size", 0)) > 0
            ),
            "full_attention_indices": [
                index for index, value in enumerate(layer_types) if value == "full_attention"
            ],
        }
        expected = {
            "text_model_type": getattr(model_config, "expected_text_model_type", None),
            "hidden_size": getattr(model_config, "expected_hidden_size", None),
            "shared_expert_intermediate_size": getattr(
                model_config, "expected_shared_expert_intermediate_size", None
            ),
            "moe_intermediate_size": getattr(
                model_config, "expected_moe_intermediate_size", None
            ),
            "mtp_num_hidden_layers": getattr(
                model_config, "expected_mtp_num_hidden_layers", None
            ),
            "shared_experts": getattr(model_config, "expected_shared_experts", None),
        }
        for key, expected_value in expected.items():
            if expected_value is not None and hook_facts[key] != expected_value:
                errors.append(
                    f"{key} mismatch: actual={hook_facts[key]} expected={expected_value}"
                )

        report.update(
            {
                "hook_relevant_config": hook_facts,
                "hook_relevant_expected": expected,
                "layer_types": list(layer_types),
                "moe_output_semantics": (
                    "weighted_routed_aggregate_before_shared_expert_addition"
                ),
                "full_moe_output_boundary": "model.layers.<block>.mlp",
                "post_moe_residual_boundary": "model.layers.<block>",
                "composite_checkpoint_runtime": "AutoModelForCausalLM_text_unwrap",
                "errors": errors,
                "passed": not errors,
            }
        )
        return report

    def load(self, config: ModelConfig) -> None:
        """Load the pinned text runtime after version and identity checks."""

        self._validate_requested_config(config)
        try:
            installed = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Qwen3.6 runtime requires transformers={self.pinned_transformers_version}"
            ) from exc
        if installed != self.pinned_transformers_version:
            raise RuntimeError(
                "Qwen3.6 Transformers version mismatch: "
                f"actual={installed} expected={self.pinned_transformers_version}"
            )
        ensure_external_cache()
        try:
            from transformers import AutoConfig
        except ImportError as exc:  # pragma: no cover - exercised by model users
            raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc
        config_factory = cast(Any, AutoConfig)
        hf_config = config_factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        static_report = self.static_config_preflight(config, hf_config)
        if not static_report["passed"]:
            raise RuntimeError(f"Qwen3.6 static preflight failed: {static_report['errors']}")
        super().load(config)
        self._validate_chat_template_contract()
        self._validate_moe_layout()
        if self.discovery_report is not None:
            self.discovery_report.metadata["static_preflight"] = static_report
            self.discovery_report.metadata["runtime_inspect"] = self.runtime_inspect()

    def _chat_completion_prompt(self, prompt: str) -> str:
        tokenizer = self._require_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Qwen3.6 tokenizer does not expose apply_chat_template")
        rendered = str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if not rendered.endswith(suffix):
            raise RuntimeError(
                "Qwen3.6 non-thinking chat template did not emit the pinned assistant suffix"
            )
        return rendered

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        if kind == "moe_output":
            return f"model.layers.{layer_index}.mlp.experts"
        return super()._module_name(layer_index, kind)

    def _site_module(self, site: LayerSite) -> Any:
        if site.kind == "moe_output":
            return self._layer(site.layer_index).mlp.experts
        return super()._site_module(site)

    def _router_dispatch_module(self, layer_index: int) -> Any:
        return self._layer(layer_index).mlp.experts

    def _capture_inputs(
        self,
        inputs: dict[str, Any],
        spec: Any,
    ) -> tuple[ActivationBatch, Any]:
        capture, logits = super()._capture_inputs(inputs, spec)
        metadata = dict(capture.metadata)
        metadata.update(
            {
                "router_logits_semantics": "raw_pre_softmax_scores",
                "router_topk_semantics": (
                    "captured_from_native_gate_tuple; expert_dispatch_equality_is_"
                    "required_separately_in_intervention_diagnostics"
                ),
                "moe_output_semantics": (
                    "weighted_routed_aggregate_before_shared_expert_addition"
                ),
                "full_moe_output_boundary": "model.layers.<block>.mlp",
                "post_moe_residual_boundary": "model.layers.<block>",
            }
        )
        return ActivationBatch(arrays=capture.arrays, metadata=metadata), logits

    def _validate_chat_template_contract(self) -> None:
        config = self.config
        tokenizer = self._require_tokenizer()
        template = getattr(tokenizer, "chat_template", None)
        expected = None if config is None else config.expected_chat_template_sha256
        if not isinstance(template, str) or not template:
            raise RuntimeError("Qwen3.6 tokenizer has no chat_template text")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if expected is None or actual != expected:
            raise RuntimeError(
                "Qwen3.6 chat template hash mismatch: "
                f"actual={actual} expected={expected}"
            )

    def _validate_moe_layout(self) -> None:
        runtime = self.runtime_inspect()
        if not runtime["contract_matches"]:
            raise RuntimeError("loaded Qwen3.6 text runtime contradicts the pinned MoE layout")

    def runtime_inspect(self) -> dict[str, Any]:
        """Inspect the unwrapped text model and all native dispatch boundaries."""

        layers = list(self._layers())
        rows: list[dict[str, Any]] = []
        expert_counts: set[int] = set()
        top_ks: set[int] = set()
        hidden_sizes: set[int] = set()
        layer_types: list[str] = []
        for index, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"Qwen3.6 layer {index} has no MLP module")
            required = ("gate", "experts", "shared_expert", "shared_expert_gate")
            missing = [name for name in required if not hasattr(mlp, name)]
            if missing:
                raise RuntimeError(f"Qwen3.6 layer {index} is missing MoE modules: {missing}")
            gate = mlp.gate
            experts = mlp.experts
            expert_counts.update(
                (int(gate.weight.shape[0]), int(experts.num_experts))
            )
            top_ks.add(int(gate.top_k))
            hidden_sizes.add(int(gate.weight.shape[1]))
            layer_type = str(getattr(layer, "layer_type", "unknown"))
            layer_types.append(layer_type)
            rows.append(
                {
                    "block_index": index,
                    "router_ordinal": index,
                    "normalized_depth": index / max(len(layers) - 1, 1),
                    "layer_type": layer_type,
                    "layer_class": _class_path(layer),
                    "mlp_class": _class_path(mlp),
                    "router_class": _class_path(gate),
                    "experts_class": _class_path(experts),
                    "shared_expert_class": _class_path(mlp.shared_expert),
                    "gate_weight_shape": [int(value) for value in gate.weight.shape],
                }
            )
        actual = {
            "total_blocks": len(layers),
            "router_blocks": len(rows),
            "experts": next(iter(expert_counts), None) if len(expert_counts) == 1 else None,
            "top_k": next(iter(top_ks), None) if len(top_ks) == 1 else None,
            "hidden_size": next(iter(hidden_sizes), None) if len(hidden_sizes) == 1 else None,
        }
        config = self.config
        runtime_model = self._require_model()
        runtime_model_type = str(getattr(runtime_model.config, "model_type", "unknown"))
        runtime_architecture = type(runtime_model).__name__
        loading_summary = _qwen36_loading_info_summary(
            runtime_model,
            self.loading_info,
        )
        expected = {
            "total_blocks": None if config is None else config.expected_total_block_count,
            "router_blocks": None if config is None else config.expected_router_block_count,
            "experts": None if config is None else config.expected_experts,
            "top_k": None if config is None else config.expected_top_k,
            "hidden_size": None if config is None else getattr(config, "expected_hidden_size", None),
            "model_type": (
                None if config is None else getattr(config, "expected_text_model_type", None)
            ),
            "runtime_architecture": (
                None
                if config is None
                else getattr(config, "expected_runtime_architecture", None)
            ),
            "layer_types": (
                ()
                if config is None
                else tuple(getattr(config, "expected_layer_types", ()) or ())
            ),
        }
        contract_matches = all(
            expected[key] is None or actual[key] == int(cast(Any, expected[key]))
            for key in ("total_blocks", "router_blocks", "experts", "top_k", "hidden_size")
        )
        contract_matches = bool(
            contract_matches
            and set(layer_types) <= {"linear_attention", "full_attention"}
            and (expected["model_type"] is None or runtime_model_type == expected["model_type"])
            and (
                expected["runtime_architecture"] is None
                or runtime_architecture == expected["runtime_architecture"]
            )
            and (not expected["layer_types"] or tuple(layer_types) == expected["layer_types"])
            and loading_summary["only_expected_composite_auxiliary_ignored"]
        )
        return {
            "checkpoint_model_type": (
                None if config is None else config.expected_model_type
            ),
            "runtime_model_type": runtime_model_type,
            "runtime_model_class": _class_path(runtime_model),
            "runtime_architecture": runtime_architecture,
            "actual_contract": actual,
            "expected_contract": expected,
            "layout_consistent": contract_matches,
            "contract_matches": contract_matches,
            "all_text_layers_routed": len(rows) == len(layers),
            "loading_info": loading_summary,
            "loading_info_ok": loading_summary[
                "only_expected_composite_auxiliary_ignored"
            ],
            "layers": rows,
            "moe_output_semantics": (
                "weighted_routed_aggregate_before_shared_expert_addition"
            ),
            "full_moe_output_boundary": "model.layers.<block>.mlp",
        }

    @classmethod
    def _validate_requested_config(cls, config: ModelConfig) -> None:
        if config.alias != cls.alias:
            raise ValueError(f"Qwen3.6 adapter requires alias={cls.alias}")
        if config.model_id != cls.model_id:
            raise ValueError(f"Qwen3.6 adapter requires model_id={cls.model_id}")
        for label, revision in {
            "model": config.revision,
            "tokenizer": config.tokenizer_revision,
        }.items():
            if revision != cls.pinned_revision or re.fullmatch(r"[0-9a-f]{40}", revision or "") is None:
                raise ValueError(
                    f"Qwen3.6 {label} revision must equal {cls.pinned_revision}"
                )
        if config.trust_remote_code:
            raise ValueError("Qwen3.6 uses native Transformers code; trust_remote_code must be false")
        if config.expected_transformers_version != cls.pinned_transformers_version:
            raise ValueError(
                "Qwen3.6 config must pin Transformers "
                f"{cls.pinned_transformers_version}"
            )
        expected_template = config.expected_chat_template_sha256
        if expected_template is None or len(expected_template) != 64:
            raise ValueError("Qwen3.6 config must pin the chat template SHA-256")


def _config_child(config: Mapping[str, Any] | Any, name: str) -> Any:
    value = _config_value(config, name)
    if value is None:
        raise RuntimeError(f"Qwen3.6 config is missing {name}")
    return value


def _config_value(config: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _class_path(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _qwen36_loading_info_summary(
    model: Any,
    loading_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Summarize the official composite-to-text loading diagnostics."""

    ignore_patterns = [
        str(value)
        for value in (getattr(model, "_keys_to_ignore_on_load_unexpected", ()) or ())
    ]
    expected_ignore_patterns = {r"^mtp.*", r"^model.visual.*"}
    exact_ignore_patterns = set(ignore_patterns) == expected_ignore_patterns
    mtp_ignore_pattern_present = r"^mtp.*" in ignore_patterns
    visual_ignore_pattern_present = r"^model.visual.*" in ignore_patterns
    expected_loading_keys = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
    }
    allowed_collection_types = (list, tuple, set, frozenset)
    schema_valid = bool(
        isinstance(loading_info, Mapping)
        and set(loading_info) == expected_loading_keys
        and all(
            isinstance(loading_info[key], allowed_collection_types)
            for key in expected_loading_keys
        )
    )
    if not isinstance(loading_info, Mapping):
        return {
            "available": False,
            "schema_valid": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_messages": [],
            "ignore_patterns": ignore_patterns,
            "mtp_ignore_pattern_present": mtp_ignore_pattern_present,
            "visual_ignore_pattern_present": visual_ignore_pattern_present,
            "exact_ignore_patterns": exact_ignore_patterns,
            "expected_index_key_counts": dict(QWEN36_INDEX_KEY_COUNTS),
            "only_expected_composite_auxiliary_ignored": False,
        }

    def values(key: str) -> list[str]:
        raw = loading_info.get(key)
        if not isinstance(raw, allowed_collection_types):
            return []
        return [str(value) for value in raw]

    missing_keys = values("missing_keys")
    unexpected_keys = values("unexpected_keys")
    mismatched_keys = values("mismatched_keys")
    error_messages = values("error_msgs")
    clean = bool(
        schema_valid
        and not missing_keys
        and not unexpected_keys
        and not mismatched_keys
        and not error_messages
        and exact_ignore_patterns
    )
    return {
        "available": True,
        "schema_valid": schema_valid,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "error_messages": error_messages,
        "ignore_patterns": ignore_patterns,
        "mtp_ignore_pattern_present": mtp_ignore_pattern_present,
        "visual_ignore_pattern_present": visual_ignore_pattern_present,
        "exact_ignore_patterns": exact_ignore_patterns,
        "expected_index_key_counts": dict(QWEN36_INDEX_KEY_COUNTS),
        "only_expected_composite_auxiliary_ignored": clean,
    }
