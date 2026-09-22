"""NVIDIA Nemotron-3 Nano adapter for the native Transformers Nemotron-H layout."""

from __future__ import annotations

import importlib.metadata
import re
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.data.prompts import PromptRender, shift_render_into_prompt
from routeverify.instrumentation.patching import router_manipulation_metrics
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchRequest,
    PatchSiteSpec,
    SiteKind,
    patch_index,
    patch_positions,
    patch_specs,
    site_key,
)
from routeverify.models.base import ActivationBatch, PreparedBatch
from routeverify.models.discovery import (
    discover_from_named_modules,
    validate_expected_metadata,
)
from routeverify.models.moe import MoEConfigPaths, RouterOutput, RouterSemantics
from routeverify.models.olmoe import OLMoEAdapter, _require_torch, _torch_dtype

_NEMOTRON_HUB_KERNELS = {
    "causal-conv1d": {
        "repo_id": "kernels-community/causal-conv1d",
        "revision": "6e9a58272220a363dc50e59cba737abcedd7b59f",
    },
    "mamba-ssm": {
        "repo_id": "kernels-community/mamba-ssm",
        "revision": "768416abf0ad7262a23d83c6efef10b9a4786489",
    },
}
_HUB_KERNEL_MAPPING_LOCK = threading.RLock()
_MISSING = object()


@contextmanager
def _temporary_pinned_nemotron_hub_kernels(
    *,
    scope: str = "nemotron3_nano_model_load",
) -> Iterator[dict[str, Any]]:
    """Pin Nemotron's two Transformers hub kernels without leaking global state."""

    from transformers.integrations import hub_kernels

    hub_mapping = hub_kernels._HUB_KERNEL_MAPPING
    module_cache = hub_kernels._KERNEL_MODULE_MAPPING
    bound_defaults = getattr(hub_kernels.lazy_load_kernel, "__defaults__", ()) or ()
    if not bound_defaults or bound_defaults[-1] is not module_cache:
        raise RuntimeError("Transformers lazy_load_kernel cache binding changed")

    state: dict[str, Any] = {
        "schema_version": 1,
        "scope": scope,
        "transformers_version": importlib.metadata.version("transformers"),
        "kernels_version": importlib.metadata.version("kernels"),
        "kernels": {
            name: {
                **pin,
                "loaded": False,
                "module_name": None,
                "module_file": None,
            }
            for name, pin in _NEMOTRON_HUB_KERNELS.items()
        },
        "mapping_mutated_in_place": False,
        "module_cache_mutated_in_place": False,
        "mapping_restored": False,
        "module_cache_restored": False,
    }

    with _HUB_KERNEL_MAPPING_LOCK:
        original_entries: dict[str, dict[str, Any]] = {}
        original_values: dict[str, dict[str, Any]] = {}
        original_modules: dict[str, Any] = {}
        for name, pin in _NEMOTRON_HUB_KERNELS.items():
            entry = hub_mapping.get(name)
            if not isinstance(entry, dict):
                raise RuntimeError(f"Transformers hub kernel mapping missing {name}")
            if entry.get("repo_id") != pin["repo_id"] or entry.get("version") != 1:
                raise RuntimeError(f"Transformers hub kernel mapping changed for {name}: {entry}")
            original_entries[name] = entry
            original_values[name] = dict(entry)
            original_modules[name] = module_cache.get(name, _MISSING)

        try:
            for name, pin in _NEMOTRON_HUB_KERNELS.items():
                entry = original_entries[name]
                entry.clear()
                entry.update(pin)
                module_cache.pop(name, None)
            state["mapping_mutated_in_place"] = all(
                hub_mapping[name] is original_entries[name]
                and hub_mapping[name] == _NEMOTRON_HUB_KERNELS[name]
                for name in _NEMOTRON_HUB_KERNELS
            )
            state["module_cache_mutated_in_place"] = all(
                name not in module_cache for name in _NEMOTRON_HUB_KERNELS
            )
            yield state
        finally:
            for name in _NEMOTRON_HUB_KERNELS:
                loaded_module = module_cache.get(name, _MISSING)
                kernel_state = state["kernels"][name]
                if loaded_module is not _MISSING and loaded_module is not None:
                    kernel_state["loaded"] = True
                    kernel_state["module_name"] = str(
                        getattr(loaded_module, "__name__", "unknown")
                    )
                    kernel_state["module_file"] = str(
                        getattr(loaded_module, "__file__", "unknown")
                    )

            for name in _NEMOTRON_HUB_KERNELS:
                entry = original_entries[name]
                if hub_mapping.get(name) is not entry:
                    raise RuntimeError(
                        f"Transformers replaced the {name} mapping during model load"
                    )
                entry.clear()
                entry.update(original_values[name])
                module_cache.pop(name, None)
                original_module = original_modules[name]
                if original_module is not _MISSING:
                    module_cache[name] = original_module

            state["mapping_restored"] = all(
                hub_mapping[name] is original_entries[name]
                and hub_mapping[name] == original_values[name]
                for name in _NEMOTRON_HUB_KERNELS
            )
            state["module_cache_restored"] = all(
                (
                    name not in module_cache
                    if original_modules[name] is _MISSING
                    else module_cache.get(name) is original_modules[name]
                )
                for name in _NEMOTRON_HUB_KERNELS
            )


@dataclass(frozen=True)
class NemotronMoEBlock:
    """Stable coordinates for one routed block in the hybrid 52-block stack."""

    absolute_block_index: int
    moe_ordinal: int
    normalized_depth: float


@dataclass(frozen=True)
class NemotronMoESite(LayerSite):
    """A standard LayerSite carrying the hybrid-stack coordinates needed for analysis."""

    absolute_block_index: int
    moe_ordinal: int
    normalized_depth: float

    def __post_init__(self) -> None:
        if self.layer_index != self.absolute_block_index:
            raise ValueError("Nemotron site layer_index must equal absolute_block_index")


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _nemotron_route_from_logits(mixer: Any, router_logits: Any) -> RouterOutput:
    """Delegate sigmoid/correction/grouped top-k routing to the native mixer."""

    topk_ids, topk_weights = mixer.route_tokens_to_experts(router_logits)
    return RouterOutput(
        logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


class Nemotron3NanoAdapter(OLMoEAdapter):
    """Adapter for the pinned native Transformers 5.12.1 Nemotron-H checkpoint.

    Nemotron-H is a hybrid stack, so only blocks whose ``block_type`` is exactly
    ``"moe"`` are exposed as RouteVerify sites. Router interventions replace the
    raw output of ``mixer.gate``; the model's own sigmoid, correction bias, grouped
    selection, top-k, renormalization, and routed scaling remain downstream.
    """

    alias = "nemotron3_nano"
    model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    pinned_revision = "2d59de1cbd51c0adf384eb906b766d1aee0e0517"
    pinned_transformers_version = "5.12.1"
    capture_loading_info = False
    moe_config_paths = MoEConfigPaths(
        experts=("n_routed_experts",),
        top_k=("num_experts_per_tok",),
    )
    router_semantics = RouterSemantics(
        identifier="nemotron_h_sigmoid_correction_group_topk_v1",
        route_from_logits=_nemotron_route_from_logits,
    )

    def __init__(self) -> None:
        super().__init__()
        self._hub_kernel_load: dict[str, Any] = {
            "schema_version": 1,
            "scope": f"{self.alias}_model_load",
            "kernels": {
                name: {**pin, "loaded": False} for name, pin in _NEMOTRON_HUB_KERNELS.items()
            },
            "mapping_restored": True,
            "module_cache_restored": True,
        }

    @classmethod
    def static_architecture_preflight(
        cls,
        hf_config: Any,
        *,
        expected: ModelConfig | None = None,
    ) -> dict[str, Any]:
        """Validate the checkpoint config without constructing or loading weights."""

        if expected is not None:
            cls._validate_requested_config(expected)
        raw_block_types = _config_value(hf_config, "layers_block_type")
        if not isinstance(raw_block_types, (list, tuple)) or not raw_block_types:
            raise RuntimeError("Nemotron-H config must expose a non-empty layers_block_type")
        block_types = tuple(str(value) for value in raw_block_types)
        valid_types = {"mamba", "attention", "moe", "mlp"}
        invalid = sorted(set(block_types) - valid_types)
        if invalid:
            raise RuntimeError(f"Nemotron-H config has unsupported block types: {invalid}")

        moe_indices = tuple(
            index for index, block_type in enumerate(block_types) if block_type == "moe"
        )
        total_blocks = len(block_types)
        moe_blocks = tuple(
            {
                "absolute_block_index": absolute_index,
                "moe_ordinal": ordinal,
                "normalized_depth": cls._normalized_depth(absolute_index, total_blocks),
            }
            for ordinal, absolute_index in enumerate(moe_indices)
        )
        architectures = tuple(_config_value(hf_config, "architectures", ()) or ())
        actual = {
            "total_blocks": total_blocks,
            "router_blocks": len(moe_indices),
            "experts": cls._optional_int(_config_value(hf_config, "n_routed_experts")),
            "shared_experts": cls._optional_int(
                _config_value(hf_config, "n_shared_experts")
            ),
            "top_k": cls._optional_int(_config_value(hf_config, "num_experts_per_tok")),
            "hidden_size": cls._optional_int(_config_value(hf_config, "hidden_size")),
            "norm_topk_prob": _config_value(hf_config, "norm_topk_prob"),
            "routed_scaling_factor": _config_value(hf_config, "routed_scaling_factor"),
        }
        hybrid_pattern = "".join(
            {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-"}[
                block_type
            ]
            for block_type in block_types
        )
        expected_contract: dict[str, Any] = {
            "total_blocks": (
                None
                if expected is None
                else expected.expected_total_blocks or expected.expected_layers
            ),
            "router_blocks": None if expected is None else expected.expected_router_blocks,
            "experts": None if expected is None else expected.expected_experts,
            "shared_experts": (
                None if expected is None else _config_value(expected, "expected_shared_experts")
            ),
            "top_k": None if expected is None else expected.expected_top_k,
            "hidden_size": (
                None if expected is None else _config_value(expected, "expected_hidden_size")
            ),
            "norm_topk_prob": (
                None if expected is None else _config_value(expected, "expected_norm_topk_prob")
            ),
            "routed_scaling_factor": (
                None
                if expected is None
                else _config_value(expected, "expected_routed_scaling_factor")
            ),
            "hybrid_override_pattern": (
                None
                if expected is None
                else _config_value(expected, "expected_hybrid_override_pattern")
            ),
            "router_block_indices": (
                None if expected is None else expected.expected_router_block_indices
            ),
        }

        mismatches: list[str] = []
        if (
            expected is not None
            and expected.expected_total_blocks is not None
            and expected.expected_layers is not None
            and expected.expected_total_blocks != expected.expected_layers
        ):
            mismatches.append(
                "expected_total_blocks="
                f"{expected.expected_total_blocks} conflicts with "
                f"legacy expected_layers={expected.expected_layers}"
            )
        for key in (
            "total_blocks",
            "router_blocks",
            "experts",
            "shared_experts",
            "top_k",
            "hidden_size",
        ):
            expected_value = expected_contract[key]
            if expected_value is None:
                continue
            actual_value = actual[key]
            if actual_value is None or int(actual_value) != int(expected_value):
                mismatches.append(f"{key}={actual_value} expected={expected_value}")
        for key in ("norm_topk_prob", "routed_scaling_factor"):
            expected_value = expected_contract[key]
            if expected_value is not None and actual[key] != expected_value:
                mismatches.append(f"{key}={actual[key]} expected={expected_value}")
        expected_pattern = expected_contract["hybrid_override_pattern"]
        if expected_pattern is not None and hybrid_pattern != expected_pattern:
            mismatches.append(
                f"hybrid_override_pattern={hybrid_pattern} expected={expected_pattern}"
            )
        expected_indices = expected_contract["router_block_indices"]
        if expected_indices is not None and tuple(expected_indices) != moe_indices:
            mismatches.append(
                f"router_block_indices={list(moe_indices)} expected={expected_indices}"
            )
        model_type = str(_config_value(hf_config, "model_type", "unknown"))
        if expected is not None and expected.expected_model_type is not None:
            if model_type != expected.expected_model_type:
                mismatches.append(
                    f"model_type={model_type} expected={expected.expected_model_type}"
                )
        if expected is not None and expected.expected_architectures is not None:
            if set(architectures) != set(expected.expected_architectures):
                mismatches.append(
                    "architectures="
                    f"{list(architectures)} expected={expected.expected_architectures}"
                )
        if mismatches:
            raise RuntimeError(
                "Nemotron-3 Nano static config contract mismatch: " + "; ".join(mismatches)
            )

        return {
            "schema_version": 1,
            "adapter": cls.alias,
            "model_id": cls.model_id,
            "model_type": model_type,
            "architectures": list(architectures),
            "actual_contract": actual,
            "expected_contract": expected_contract,
            "contract_matches": True,
            "block_types": list(block_types),
            "block_type_counts": {
                block_type: block_types.count(block_type) for block_type in sorted(valid_types)
            },
            "hybrid_override_pattern": hybrid_pattern,
            "router_block_indices": list(moe_indices),
            "moe_layers": list(moe_blocks),
            "moe_latent_size": cls._optional_int(_config_value(hf_config, "moe_latent_size")),
        }

    def load(self, config: ModelConfig) -> None:
        """Load the pinned checkpoint after a weight-free static config preflight."""

        self._validate_requested_config(config)
        ensure_external_cache()
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by model users
            raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

        if transformers.__version__ != self.pinned_transformers_version:
            raise RuntimeError(
                "Nemotron-3 Nano Transformers version mismatch: "
                f"actual={transformers.__version__} expected={self.pinned_transformers_version}"
            )

        config_factory = cast(Any, AutoConfig)
        hf_config = config_factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        static_report = self.static_architecture_preflight(hf_config, expected=config)

        tokenizer_factory = cast(Any, AutoTokenizer)
        tokenizer = tokenizer_factory.from_pretrained(
            pretrained_model_name_or_path=config.model_id,
            revision=config.tokenizer_revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "revision": config.revision,
            "cache_dir": config.cache_dir,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device_map,
            "dtype": _torch_dtype(config.dtype),
        }
        if config.attn_implementation is not None:
            kwargs["attn_implementation"] = config.attn_implementation
        model_factory = cast(Any, AutoModelForCausalLM)
        if self.capture_loading_info:
            kwargs["output_loading_info"] = True
        with _temporary_pinned_nemotron_hub_kernels(
            scope=f"{self.alias}_model_load"
        ) as hub_kernel_load:
            loaded = model_factory.from_pretrained(config.model_id, **kwargs)
        self._hub_kernel_load = hub_kernel_load
        if not all(bool(kernel["loaded"]) for kernel in hub_kernel_load["kernels"].values()):
            raise RuntimeError(f"{self.alias} did not load both pinned hub kernels")
        if not (
            hub_kernel_load["mapping_restored"]
            and hub_kernel_load["module_cache_restored"]
        ):
            raise RuntimeError(f"{self.alias} hub kernel state restoration failed")
        if self.capture_loading_info:
            if not isinstance(loaded, tuple) or len(loaded) != 2:
                raise RuntimeError("model loader did not return requested loading information")
            model, loading_info = loaded
            if not isinstance(loading_info, dict):
                raise RuntimeError("model loading information must be a mapping")
            self.loading_info: dict[str, Any] | None = loading_info
        else:
            model = loaded
            self.loading_info = None
        model.eval()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        runtime = self.runtime_inspect()
        if not runtime["contract_matches"]:
            raise RuntimeError("Nemotron-3 Nano runtime contradicts the static contract")

        report = discover_from_named_modules(
            model.named_modules(),
            alias=config.alias,
            model_id=config.model_id,
        )
        actual = runtime["actual_contract"]
        report.metadata.update(
            {
                "architectures": runtime["architectures"],
                "model_type": runtime["model_type"],
                "num_hidden_layers": actual["total_blocks"],
                "num_moe_layers": actual["router_blocks"],
                "num_experts": actual["experts"],
                "num_local_experts": actual["experts"],
                "num_experts_per_tok": actual["top_k"],
                "load_weights": True,
                "transformers_version": transformers.__version__,
                "static_preflight": static_report,
                "runtime_inspect": runtime,
            }
        )
        validate_expected_metadata(report, config)
        self.discovery_report = report

    def discover_moe_blocks(self) -> list[NemotronMoEBlock]:
        """Return only routed blocks, preserving absolute and MoE-relative coordinates."""

        return list(self._moe_blocks())

    def discover_layers(self) -> list[LayerSite]:
        sites: list[LayerSite] = []
        for block in self._moe_blocks():
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
                    NemotronMoESite(
                        layer_index=block.absolute_block_index,
                        kind=kind,
                        module_name=self._module_name(block.absolute_block_index, kind),
                        absolute_block_index=block.absolute_block_index,
                        moe_ordinal=block.moe_ordinal,
                        normalized_depth=block.normalized_depth,
                    )
                )
        return sites

    def runtime_inspect(self) -> dict[str, Any]:
        """Inspect loaded block types and MoE modules as the final runtime truth."""

        model = self._require_model()
        static_report = self.static_architecture_preflight(model.config, expected=self.config)
        rows: list[dict[str, Any]] = []
        expert_counts: set[int] = set()
        top_ks: set[int] = set()
        hidden_sizes: set[int] = set()
        norm_topk_values: set[bool] = set()
        scaling_factors: set[float] = set()
        for block in self._moe_blocks():
            layer = self._layer(block.absolute_block_index)
            mixer = layer.mixer
            gate_experts = int(mixer.gate.weight.shape[0])
            module_experts = int(getattr(mixer.experts, "num_experts", gate_experts))
            top_k = int(mixer.top_k)
            expert_counts.update((gate_experts, module_experts))
            top_ks.add(top_k)
            hidden_sizes.add(int(mixer.gate.weight.shape[1]))
            norm_topk_values.add(bool(mixer.norm_topk_prob))
            scaling_factors.add(float(mixer.routed_scaling_factor))
            rows.append(
                {
                    "absolute_block_index": block.absolute_block_index,
                    "moe_ordinal": block.moe_ordinal,
                    "normalized_depth": block.normalized_depth,
                    "block_class": self._class_path(layer),
                    "mixer_class": self._class_path(mixer),
                    "gate_class": self._class_path(mixer.gate),
                    "experts_class": self._class_path(mixer.experts),
                    "fc2_latent_proj_class": self._class_path(mixer.fc2_latent_proj),
                    "shared_experts_class": self._class_path(mixer.shared_experts),
                    "gate_weight_shape": [int(value) for value in mixer.gate.weight.shape],
                    "gate_num_experts": gate_experts,
                    "expert_module_num_experts": module_experts,
                    "top_k": top_k,
                }
            )

        static_actual = static_report["actual_contract"]
        actual_module_experts = (
            next(iter(expert_counts), None) if len(expert_counts) == 1 else None
        )
        module_top_k = next(iter(top_ks), None) if len(top_ks) == 1 else None
        module_hidden_size = next(iter(hidden_sizes), None) if len(hidden_sizes) == 1 else None
        module_norm_topk = (
            next(iter(norm_topk_values), None) if len(norm_topk_values) == 1 else None
        )
        module_scaling = (
            next(iter(scaling_factors), None) if len(scaling_factors) == 1 else None
        )
        layout_consistent = (
            len(rows) == int(static_actual["router_blocks"])
            and actual_module_experts == static_actual["experts"]
            and module_top_k == static_actual["top_k"]
            and module_hidden_size == static_actual["hidden_size"]
            and module_norm_topk == static_actual["norm_topk_prob"]
            and module_scaling == static_actual["routed_scaling_factor"]
        )
        return {
            **static_report,
            "layout_consistent": layout_consistent,
            "contract_matches": bool(static_report["contract_matches"] and layout_consistent),
            "moe_layers": rows,
            "module_contract": {
                "experts": actual_module_experts,
                "top_k": module_top_k,
                "hidden_size": module_hidden_size,
                "norm_topk_prob": module_norm_topk,
                "routed_scaling_factor": module_scaling,
            },
            "hub_kernel_load": self._hub_kernel_load,
        }

    def forward_logits(self, batch: PreparedBatch) -> Any:
        """Run Nemotron without forwarding OLMoE-only router-output kwargs."""

        torch = _require_torch()
        model = self._require_model()
        with torch.inference_mode():
            outputs = model(**self._encode_batch(batch), use_cache=False)
        return outputs.logits.detach().cpu()

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        """Apply the non-thinking chat template while preserving aligned step spans."""

        prompt = self._chat_completion_prompt(render.prompt)
        offset = prompt.find(render.prompt)
        if offset < 0:
            raise RuntimeError("Nemotron chat template did not preserve the raw prompt text")
        return shift_render_into_prompt(render, prompt=prompt, offset=offset)

    def _chat_completion_prompt(self, prompt: str) -> str:
        tokenizer = self._require_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Nemotron tokenizer does not expose apply_chat_template")
        messages = [{"role": "user", "content": prompt}]
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        prefix = f"model.layers.{layer_index}"
        if kind == "pre_moe_residual":
            return f"{prefix}.norm"
        if kind == "router_input":
            return f"{prefix}.mixer"
        if kind == "router_logits":
            return f"{prefix}.mixer.gate"
        if kind in {"router_topk_ids", "router_topk_weights"}:
            return f"{prefix}.mixer.experts"
        if kind == "moe_output":
            return f"{prefix}.mixer.fc2_latent_proj"
        if kind == "post_moe_residual":
            return prefix
        raise ValueError(f"unsupported site kind: {kind}")

    def _site_module(self, site: LayerSite) -> Any:
        layer = self._moe_layer(site.layer_index)
        if site.kind == "pre_moe_residual":
            return layer.norm
        if site.kind == "router_input":
            return layer.mixer
        if site.kind == "router_logits":
            return layer.mixer.gate
        if site.kind in {"router_topk_ids", "router_topk_weights"}:
            return layer.mixer.experts
        if site.kind == "moe_output":
            return layer.mixer.fc2_latent_proj
        if site.kind == "post_moe_residual":
            return layer
        raise ValueError(f"unsupported site kind: {site.kind}")

    def _capture_inputs(
        self,
        inputs: dict[str, Any],
        spec: CaptureSpec,
    ) -> tuple[ActivationBatch, Any]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        self._validate_sequence_positions(inputs, spec.token_positions, label="capture")
        arrays: dict[str, Any] = {}
        fire_counts = {site_key(site): 0 for site in spec.sites}
        handles: list[Any] = []
        supported_kinds = {
            "pre_moe_residual",
            "router_input",
            "router_logits",
            "router_topk_ids",
            "router_topk_weights",
            "moe_output",
            "post_moe_residual",
        }
        for site in spec.sites:
            self._moe_layer(site.layer_index)
            if site.kind not in supported_kinds:  # pragma: no cover - guarded by SiteKind
                raise ValueError(f"unsupported Nemotron capture site: {site.kind}")

        def register(method: Any, *args: Any, **kwargs: Any) -> None:
            try:
                handle = method(*args, **kwargs)
            except BaseException:
                for installed in handles:
                    installed.remove()
                handles.clear()
                raise
            handles.append(handle)

        def store(site: LayerSite, tensor: Any) -> None:
            sequence_tensor = self._as_sequence_tensor(tensor, batch_size, sequence_length)
            selected = sequence_tensor[:, list(spec.token_positions), ...].detach()
            if selected.is_floating_point():
                selected = selected.to(_torch_dtype(spec.dtype))
            arrays[site_key(site)] = selected.cpu()
            fire_counts[site_key(site)] += 1

        gate_sites: dict[int, list[LayerSite]] = {}
        dispatch_sites: dict[int, list[LayerSite]] = {}
        for site in spec.sites:
            if site.kind == "router_logits":
                gate_sites.setdefault(site.layer_index, []).append(site)
            elif site.kind in {"router_topk_ids", "router_topk_weights"}:
                dispatch_sites.setdefault(site.layer_index, []).append(site)
            elif site.kind in {"pre_moe_residual", "router_input"}:

                def input_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    *,
                    site: LayerSite = site,
                ) -> None:
                    if not args:
                        raise RuntimeError(f"{site_key(site)} received no positional input")
                    store(site, args[0])

                register(self._site_module(site).register_forward_pre_hook, input_hook)
            elif site.kind in {"moe_output", "post_moe_residual"}:

                def output_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    site: LayerSite = site,
                ) -> None:
                    store(site, self._primary_output(output))

                register(self._site_module(site).register_forward_hook, output_hook)
            else:  # pragma: no cover - exhaustive for current SiteKind
                raise ValueError(f"unsupported Nemotron capture site: {site.kind}")

        for layer_index, sites in gate_sites.items():
            gate = self._moe_layer(layer_index).mixer.gate

            def gate_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                sites: list[LayerSite] = sites,
            ) -> Any:
                if not torch.is_tensor(output):
                    raise RuntimeError("Nemotron mixer.gate must return raw router logits")
                for site in sites:
                    store(site, output)
                return output

            register(gate.register_forward_hook, gate_hook)

        for layer_index, sites in dispatch_sites.items():
            experts = self._moe_layer(layer_index).mixer.experts

            def experts_pre_hook(
                _module: Any,
                args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                *,
                sites: list[LayerSite] = sites,
            ) -> None:
                if len(args) < 3:
                    raise RuntimeError(
                        "Nemotron experts forward-pre args must be "
                        "(hidden_states, top_k_indices, top_k_weights)"
                    )
                for site in sites:
                    store(site, args[1] if site.kind == "router_topk_ids" else args[2])

            register(
                experts.register_forward_pre_hook,
                experts_pre_hook,
                with_kwargs=True,
            )

        try:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
            logits = outputs.logits.detach().cpu()
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(key for key, count in fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered Nemotron capture hooks did not fire: {missing}")
        metadata = {
            "logits_shape": tuple(outputs.logits.shape),
            "hook_fire_counts": fire_counts,
            "all_registered_hooks_fired": not missing,
            "hooks_restored": True,
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
        dispatch_fire_counts: dict[str, int] = {}
        dispatch_after_gate: dict[str, bool] = {}
        dispatch_ids_match: dict[str, bool | None] = {}
        dispatch_weights_match: dict[str, bool | None] = {}
        expected_dispatch: dict[str, tuple[Any, Any]] = {}
        handles: list[Any] = []

        def register(method: Any, *args: Any, **kwargs: Any) -> None:
            try:
                handle = method(*args, **kwargs)
            except BaseException:
                for installed in handles:
                    installed.remove()
                handles.clear()
                raise
            handles.append(handle)

        for patch in patches:
            self._validate_sequence_positions(
                inputs, patch_positions(patch), label=site_key(patch.site)
            )
            self._moe_layer(patch.site.layer_index)
            if patch.site.kind not in {"moe_output", "post_moe_residual", "router_logits"}:
                raise ValueError(
                    f"patching site {patch.site.kind!r} is not implemented for Nemotron"
                )

        for patch in patches:
            layer = self._moe_layer(patch.site.layer_index)
            key = site_key(patch.site)
            if patch.site.kind in {"moe_output", "post_moe_residual"}:

                def output_patch_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    patch: PatchSiteSpec = patch,
                    key: str = key,
                ) -> Any:
                    hook_fire_counts[key] += 1
                    primary = self._primary_output(output)
                    patched = self._replace_sequence_slice(
                        primary, patch, batch_size, sequence_length
                    ).reshape_as(primary)
                    return self._restore_primary_output(output, patched)

                register(
                    self._site_module(patch.site).register_forward_hook,
                    output_patch_hook,
                )
            elif patch.site.kind == "router_logits":
                mixer = layer.mixer
                dispatch_fire_counts[key] = 0
                dispatch_after_gate[key] = False
                dispatch_ids_match[key] = None
                dispatch_weights_match[key] = None

                def router_patch_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    router_logits: Any,
                    *,
                    patch: PatchSiteSpec = patch,
                    key: str = key,
                    mixer: Any = mixer,
                ) -> Any:
                    if not torch.is_tensor(router_logits):
                        raise RuntimeError(f"{key} mixer.gate must return raw router logits")
                    hook_fire_counts[key] += 1
                    receiver_sequence = self._as_sequence_tensor(
                        router_logits, batch_size, sequence_length
                    )
                    receiver_ids, _receiver_weights = mixer.route_tokens_to_experts(
                        router_logits
                    )
                    receiver_ids_sequence = self._as_sequence_tensor(
                        receiver_ids, batch_size, sequence_length
                    )
                    if bool(patch.metadata.get("forced_reroute")):
                        target = receiver_sequence[:, patch_index(patch), ...]
                        if not target.dtype.is_floating_point:
                            raise TypeError(f"{key} forced reroute requires floating router logits")
                        original_ids = receiver_ids_sequence[:, patch_index(patch), ...]
                        selected = torch.zeros_like(target, dtype=torch.bool)
                        selected.scatter_(-1, original_ids, True)
                        limits = torch.finfo(target.dtype)
                        replacement = torch.where(
                            selected,
                            torch.full_like(target, limits.min),
                            torch.full_like(target, limits.max),
                        )
                        patched_sequence = receiver_sequence.clone()
                        patched_sequence[:, patch_index(patch), ...] = replacement
                    else:
                        patched_sequence = self._replace_sequence_slice(
                            router_logits, patch, batch_size, sequence_length
                        )
                    patched_logits = patched_sequence.reshape_as(router_logits)
                    expected_ids, expected_weights = mixer.route_tokens_to_experts(patched_logits)
                    expected_dispatch[key] = (
                        expected_ids.detach().clone(),
                        expected_weights.detach().clone(),
                    )
                    expected_ids_sequence = self._as_sequence_tensor(
                        expected_ids, batch_size, sequence_length
                    )
                    manipulation_by_site[key] = router_manipulation_metrics(
                        receiver_sequence[:, patch_index(patch), ...]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        patched_sequence[:, patch_index(patch), ...]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        receiver_topk_ids=receiver_ids_sequence[
                            :, patch_index(patch), ...
                        ]
                        .detach()
                        .cpu()
                        .numpy(),
                        patched_topk_ids=expected_ids_sequence[
                            :, patch_index(patch), ...
                        ]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    return patched_logits

                register(mixer.gate.register_forward_hook, router_patch_hook)

                def experts_pre_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    _kwargs: dict[str, Any],
                    *,
                    key: str = key,
                ) -> None:
                    dispatch_fire_counts[key] += 1
                    dispatch_after_gate[key] = hook_fire_counts[key] > 0
                    if len(args) < 3:
                        raise RuntimeError(
                            f"{key} experts did not receive positional top-k IDs and weights"
                        )
                    expected = expected_dispatch.get(key)
                    if expected is None:
                        return
                    actual_ids, actual_weights = args[1], args[2]
                    dispatch_ids_match[key] = bool(
                        torch.equal(actual_ids, expected[0].to(actual_ids.device))
                    )
                    dispatch_weights_match[key] = bool(
                        torch.equal(actual_weights, expected[1].to(actual_weights.device))
                    )

                register(
                    mixer.experts.register_forward_pre_hook,
                    experts_pre_hook,
                    with_kwargs=True,
                )
            else:
                raise ValueError(
                    f"patching site {patch.site.kind!r} is not implemented for Nemotron"
                )

        try:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(key for key, count in hook_fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered Nemotron patch hooks did not fire: {missing}")

        dispatch_by_site: dict[str, dict[str, Any]] = {}
        for patch in patches:
            if patch.site.kind != "router_logits":
                continue
            key = site_key(patch.site)
            verified = (
                dispatch_fire_counts[key] > 0
                and dispatch_after_gate[key]
                and dispatch_ids_match[key] is True
                and dispatch_weights_match[key] is True
            )
            dispatch_by_site[key] = {
                "expert_dispatch_hook_fire_count": dispatch_fire_counts[key],
                "dispatch_after_raw_gate_patch": dispatch_after_gate[key],
                "actual_topk_ids_match_native_patched_routing": dispatch_ids_match[key],
                "actual_topk_weights_match_native_patched_routing": dispatch_weights_match[key],
                "downstream_dispatch_verified": verified,
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
                bool(value["downstream_dispatch_verified"])
                for value in dispatch_by_site.values()
            ),
            "hooks_restored": True,
        }
        return outputs, diagnostics

    def _moe_blocks(self) -> tuple[NemotronMoEBlock, ...]:
        model = self._require_model()
        layers = self._layers()
        block_types = _config_value(model.config, "layers_block_type")
        if not isinstance(block_types, (list, tuple)):
            raise RuntimeError("loaded Nemotron model config has no layers_block_type list")
        if len(block_types) != len(layers):
            raise RuntimeError(
                "Nemotron layer/module count mismatch: "
                f"config={len(block_types)} modules={len(layers)}"
            )
        blocks: list[NemotronMoEBlock] = []
        for absolute_index, (configured_type, layer) in enumerate(
            zip(block_types, layers, strict=True)
        ):
            runtime_type = getattr(layer, "block_type", None)
            if runtime_type != configured_type:
                raise RuntimeError(
                    "Nemotron block type mismatch at absolute index "
                    f"{absolute_index}: config={configured_type} runtime={runtime_type}"
                )
            if runtime_type != "moe":
                continue
            mixer = getattr(layer, "mixer", None)
            required = ("gate", "experts", "fc2_latent_proj", "shared_experts")
            missing = [name for name in required if not hasattr(mixer, name)]
            if missing:
                raise RuntimeError(
                    f"Nemotron MoE block {absolute_index} is missing modules: {missing}"
                )
            blocks.append(
                NemotronMoEBlock(
                    absolute_block_index=absolute_index,
                    moe_ordinal=len(blocks),
                    normalized_depth=self._normalized_depth(absolute_index, len(layers)),
                )
            )
        expected_router_blocks = (
            None if self.config is None else self.config.expected_router_blocks
        )
        if expected_router_blocks is not None and len(blocks) != expected_router_blocks:
            raise RuntimeError(
                "Nemotron MoE block count mismatch: "
                f"actual={len(blocks)} expected={expected_router_blocks}"
            )
        expected_indices = (
            None if self.config is None else self.config.expected_router_block_indices
        )
        actual_indices = [block.absolute_block_index for block in blocks]
        if expected_indices is not None and actual_indices != expected_indices:
            raise RuntimeError(
                "Nemotron router block indices mismatch: "
                f"actual={actual_indices} expected={expected_indices}"
            )
        return tuple(blocks)

    def _router_block_indices(self) -> tuple[int, ...]:
        """Expose the hybrid stack's absolute routed-block coordinates."""

        return tuple(block.absolute_block_index for block in self._moe_blocks())

    def _moe_layer(self, absolute_block_index: int) -> Any:
        layer = self._layer(absolute_block_index)
        if getattr(layer, "block_type", None) != "moe":
            raise ValueError(
                f"absolute block {absolute_block_index} is not a Nemotron MoE block"
            )
        return layer

    @classmethod
    def _validate_requested_config(cls, config: ModelConfig) -> None:
        if config.alias != cls.alias:
            raise ValueError(f"Nemotron adapter requires alias={cls.alias}")
        if config.model_id != cls.model_id:
            raise ValueError(f"Nemotron adapter requires model_id={cls.model_id}")
        revisions = {
            "model": config.revision,
            "tokenizer": config.tokenizer_revision,
        }
        for label, revision in revisions.items():
            if revision != cls.pinned_revision or re.fullmatch(r"[0-9a-f]{40}", revision or "") is None:
                raise ValueError(
                    f"Nemotron-3 Nano {label} revision must equal {cls.pinned_revision}"
                )
        if config.expected_transformers_version != cls.pinned_transformers_version:
            raise ValueError(
                "Nemotron-3 Nano config must pin "
                f"Transformers {cls.pinned_transformers_version}"
            )

    @staticmethod
    def _normalized_depth(absolute_block_index: int, total_blocks: int) -> float:
        if total_blocks <= 1:
            return 0.0
        return absolute_block_index / (total_blocks - 1)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _primary_output(output: Any) -> Any:
        if isinstance(output, tuple):
            if not output:
                raise RuntimeError("Nemotron module returned an empty tuple")
            return output[0]
        return output

    @staticmethod
    def _restore_primary_output(original: Any, primary: Any) -> Any:
        if isinstance(original, tuple):
            return (primary, *original[1:])
        return primary

    @staticmethod
    def _class_path(value: Any) -> str:
        return f"{type(value).__module__}.{type(value).__qualname__}"
