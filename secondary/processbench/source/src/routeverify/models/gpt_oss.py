"""GPT-OSS adapter with Harmony rendering and MXFP4-safe router interventions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.data.prompts import PromptRender, shift_render_into_prompt
from routeverify.data.schema import stable_hash
from routeverify.eval.candidate_scoring import (
    CandidateScore,
    encode_prompt_candidate_hf,
    score_candidates_hf,
)
from routeverify.instrumentation.patching import router_manipulation_metrics
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchRequest,
    PatchSpec,
    SiteKind,
    patch_specs,
    site_key,
)
from routeverify.models.base import ActivationBatch
from routeverify.models.discovery import discover_from_named_modules, validate_expected_metadata
from routeverify.models.moe import RouterOutput, RouterSemantics
from routeverify.models.olmoe import OLMoEAdapter, _require_torch, _torch_dtype

_LINEAR_INTERCEPT_LOCK = threading.RLock()
_HARMONY_MARKER_PREFIX = "ROUTEVERIFY_CANDIDATE_BOUNDARY_"
_HARMONY_DATE_PATTERN = re.compile(r"Current date: (\d{4}-\d{2}-\d{2})")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _gpt_oss_route_from_logits(router_module: Any, router_logits: Any) -> RouterOutput:
    """Reproduce GPT-OSS eager routing: raw-logit top-k, then selected softmax."""

    torch = _require_torch()
    top_values, top_ids = torch.topk(router_logits, int(router_module.top_k), dim=-1)
    top_weights = torch.softmax(top_values, dim=-1, dtype=top_values.dtype)
    return RouterOutput(
        logits=router_logits,
        topk_weights=top_weights,
        topk_ids=top_ids,
    )


@dataclass(frozen=True)
class HarmonyCandidateRender:
    """One cached Harmony render shared by baseline, capture, and patch paths."""

    raw_prompt: str
    candidate_prefix: str
    candidate_suffix: str
    template_sha256: str
    rendered_date: str | None
    reasoning_effort: str

    def render_candidate(self, candidate: str) -> str:
        if not candidate:
            raise ValueError("candidate must be non-empty")
        return f"{self.candidate_prefix}{candidate}{self.candidate_suffix}"

    def provenance(self, candidates: tuple[str, ...] = ()) -> dict[str, Any]:
        prefix_hash = stable_hash({"prompt": self.candidate_prefix}, length=64)
        return {
            "render_family": "openai_harmony",
            "rendered_prompt_hash": prefix_hash,
            "rendered_prompt_sha256": _sha256_text(self.candidate_prefix),
            "chat_template_sha256": self.template_sha256,
            "rendered_date": self.rendered_date,
            "reasoning_effort": self.reasoning_effort,
            "candidate_prefix_length": len(self.candidate_prefix),
            "candidate_suffix": self.candidate_suffix,
            "candidate_contracts": {
                candidate: {
                    "target_char_span": [
                        len(self.candidate_prefix),
                        len(self.candidate_prefix) + len(candidate),
                    ],
                    "full_render_sha256": _sha256_text(self.render_candidate(candidate)),
                }
                for candidate in candidates
            },
        }


@contextmanager
def _temporary_functional_linear_interceptors(
    callbacks_by_weight: Mapping[int, Callable[[Any], Any]],
) -> Iterator[None]:
    """Intercept only selected router ``F.linear`` outputs and restore globally.

    Transformers 5.12.1 replaces GPT-OSS MXFP4 MLP forwards with an implementation
    that calls ``torch.nn.functional.linear`` directly on ``mlp.router.weight``.
    Consequently a forward hook on ``mlp.router`` does not fire.  Matching the exact
    weight object keeps this scoped to the requested routers while covering both the
    eager router module and the native MXFP4 path.
    """

    if not callbacks_by_weight:
        yield
        return
    torch = _require_torch()
    functional = torch.nn.functional
    with _LINEAR_INTERCEPT_LOCK:
        original_linear = functional.linear

        def intercepted_linear(input_tensor: Any, weight: Any, bias: Any = None) -> Any:
            output = original_linear(input_tensor, weight, bias)
            callback = callbacks_by_weight.get(id(weight))
            return output if callback is None else callback(output)

        functional.linear = intercepted_linear
        try:
            yield
        finally:
            functional.linear = original_linear


@contextmanager
def _temporary_mxfp4_routing_interceptor(
    callback: Callable[[Any, tuple[Any, Any, Any]], None],
) -> Iterator[dict[str, bool]]:
    """Observe native MXFP4 routing inputs/results and restore the callable."""

    state = {"installed": False, "restored": True}
    try:
        from transformers.integrations import mxfp4

        hub = getattr(mxfp4, "triton_kernels_hub", None)
        routing_namespace = None if hub is None else getattr(hub, "routing", None)
        original_routing = (
            None if routing_namespace is None else getattr(routing_namespace, "routing", None)
        )
    except ImportError:
        original_routing = None
        routing_namespace = None
    if routing_namespace is None or not callable(original_routing):
        yield state
        return

    with _LINEAR_INTERCEPT_LOCK:

        def intercepted_routing(
            router_logits: Any,
            top_k: Any,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Any, Any, Any]:
            result = original_routing(router_logits, top_k, *args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 3:
                raise RuntimeError("native MXFP4 routing must return three dispatch artifacts")
            callback(router_logits, result)
            return result

        routing_namespace.routing = intercepted_routing
        state["installed"] = True
        state["restored"] = False
        try:
            yield state
        finally:
            routing_namespace.routing = original_routing
            state["restored"] = routing_namespace.routing is original_routing


@contextmanager
def _temporary_pinned_mxfp4_kernel_loader(
    *,
    repository: str,
    revision: str,
) -> Iterator[dict[str, Any]]:
    """Force Transformers' dynamic MXFP4 kernel load to one immutable revision."""

    if not repository or not revision:
        raise ValueError("MXFP4 kernel repository and revision must be non-empty")
    from transformers.integrations import hub_kernels

    hub_kernels_module = cast(Any, hub_kernels)
    original_hub_loader = hub_kernels_module.get_kernel
    state: dict[str, Any] = {
        "repository": repository,
        "revision": revision,
        "matched_call_count": 0,
        "module_names": [],
        "module_files": [],
        "restored": False,
    }

    def pinned_loader(
        kernel_name: str,
        revision: str | None = None,
        version: int | str | None = None,
        allow_all_kernels: bool = False,
    ) -> Any:
        resolved_revision = revision
        if kernel_name == repository:
            if revision is not None and revision != state["revision"]:
                raise RuntimeError(
                    f"MXFP4 kernel revision conflict: {revision} != {state['revision']}"
                )
            if version is not None:
                raise RuntimeError(
                    "MXFP4 kernel version conflicts with the required immutable revision"
                )
            resolved_revision = str(state["revision"])
        module = original_hub_loader(
            kernel_name,
            revision=resolved_revision,
            version=version,
            allow_all_kernels=allow_all_kernels,
        )
        if kernel_name == repository:
            state["matched_call_count"] += 1
            module_name = getattr(module, "__name__", None)
            module_file = getattr(module, "__file__", None)
            state["module_names"].append("unknown" if not module_name else str(module_name))
            state["module_files"].append("unknown" if not module_file else str(module_file))
        return module

    with _LINEAR_INTERCEPT_LOCK:
        hub_kernels_module.get_kernel = pinned_loader
        try:
            yield state
        finally:
            hub_kernels_module.get_kernel = original_hub_loader
            state["restored"] = hub_kernels_module.get_kernel is original_hub_loader


class GptOssAdapter(OLMoEAdapter):
    """Adapter for the native Transformers GPT-OSS eager and MXFP4 layouts."""

    alias = "gpt_oss_20b"
    model_id = "openai/gpt-oss-20b"
    persist_rendered_text = True
    router_semantics = RouterSemantics(
        identifier="gpt_oss_raw_topk_selected_softmax_v1",
        route_from_logits=_gpt_oss_route_from_logits,
    )

    def __init__(self) -> None:
        super().__init__()
        self._harmony_cache: dict[str, HarmonyCandidateRender] = {}
        self._mxfp4_kernel_load: dict[str, Any] = {
            "matched_call_count": 0,
            "restored": True,
        }

    def load(self, config: ModelConfig) -> None:
        """Load the pinned checkpoint and inspect the actual post-quantization layout."""

        if not config.revision or not config.tokenizer_revision:
            raise ValueError("GPT-OSS model and tokenizer revisions must both be pinned")
        if not config.expected_mxfp4_kernel_repo or not config.expected_mxfp4_kernel_revision:
            raise ValueError("GPT-OSS MXFP4 kernel repository and revision must both be pinned")
        immutable_revisions = {
            "model": config.revision,
            "tokenizer": config.tokenizer_revision,
            "MXFP4 kernel": config.expected_mxfp4_kernel_revision,
        }
        for label, revision in immutable_revisions.items():
            if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError(f"GPT-OSS {label} revision must be a 40-character commit SHA")
        ensure_external_cache()
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by model users
            raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

        package_contracts = {
            "transformers": config.expected_transformers_version,
            "kernels": config.expected_kernels_version,
            "triton": config.expected_triton_version,
        }
        for package, expected_version in package_contracts.items():
            if expected_version is None:
                continue
            try:
                actual_version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"GPT-OSS runtime requires {package}={expected_version}"
                ) from exc
            if actual_version != expected_version:
                raise RuntimeError(
                    f"GPT-OSS {package} version mismatch: "
                    f"actual={actual_version} expected={expected_version}"
                )

        model_factory = cast(Any, AutoModelForCausalLM)
        tokenizer_factory = cast(Any, AutoTokenizer)
        tokenizer = tokenizer_factory.from_pretrained(
            config.model_id,
            revision=config.tokenizer_revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.config = config
        self.tokenizer = tokenizer
        self._harmony_cache.clear()
        self._validate_chat_template_contract()

        kwargs: dict[str, Any] = {
            "revision": config.revision,
            "cache_dir": config.cache_dir,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device_map,
            "dtype": _torch_dtype(config.dtype),
        }
        if config.attn_implementation is not None:
            kwargs["attn_implementation"] = config.attn_implementation
        with _temporary_pinned_mxfp4_kernel_loader(
            repository=config.expected_mxfp4_kernel_repo,
            revision=config.expected_mxfp4_kernel_revision,
        ) as kernel_load:
            model = model_factory.from_pretrained(config.model_id, **kwargs)
        model.eval()

        self._mxfp4_kernel_load = dict(kernel_load)
        self.model = model

        report = discover_from_named_modules(
            model.named_modules(),
            alias=config.alias,
            model_id=config.model_id,
        )
        runtime = self.runtime_inspect()
        if (
            config.expected_model_type is not None
            and runtime["model_type"] != config.expected_model_type
        ):
            raise RuntimeError(
                "GPT-OSS runtime model type mismatch: "
                f"actual={runtime['model_type']} expected={config.expected_model_type}"
            )
        if config.expected_architectures is not None and set(runtime["architectures"]) != set(
            config.expected_architectures
        ):
            raise RuntimeError(
                "GPT-OSS runtime architectures mismatch: "
                f"actual={runtime['architectures']} expected={config.expected_architectures}"
            )
        if (
            config.expected_transformers_version is not None
            and transformers.__version__ != config.expected_transformers_version
        ):
            raise RuntimeError(
                "GPT-OSS Transformers version mismatch: "
                f"actual={transformers.__version__} "
                f"expected={config.expected_transformers_version}"
            )
        if config.require_native_mxfp4 and not runtime["native_mxfp4"]:
            raise RuntimeError("GPT-OSS runtime did not preserve the required native MXFP4 path")
        if config.require_native_mxfp4 and (
            int(runtime["mxfp4_kernel_load"].get("matched_call_count", 0)) < 1
            or runtime["mxfp4_kernel_load"].get("restored") is not True
        ):
            raise RuntimeError("GPT-OSS pinned MXFP4 kernel load/restoration contract failed")
        report.metadata.update(
            {
                "architectures": runtime["architectures"],
                "model_type": runtime["model_type"],
                "num_hidden_layers": runtime["actual_contract"]["layers"],
                "num_experts": runtime["actual_contract"]["experts"],
                "num_local_experts": runtime["actual_contract"]["experts"],
                "num_experts_per_tok": runtime["actual_contract"]["top_k"],
                "load_weights": True,
                "transformers_version": transformers.__version__,
                "runtime_inspect": runtime,
            }
        )
        validate_expected_metadata(report, config)
        if not runtime["contract_matches"]:
            raise RuntimeError("GPT-OSS runtime contract contradicts the pinned static contract")
        self.discovery_report = report

    def discover_layers(self) -> list[LayerSite]:
        return super().discover_layers()

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        if kind == "pre_moe_residual":
            return f"model.layers.{layer_index}.post_attention_layernorm"
        if kind == "router_input":
            return f"model.layers.{layer_index}.mlp"
        if kind in {"router_logits", "router_topk_ids", "router_topk_weights"}:
            return f"model.layers.{layer_index}.mlp.router"
        if kind == "moe_output":
            return f"model.layers.{layer_index}.mlp"
        if kind == "post_moe_residual":
            return f"model.layers.{layer_index}"
        raise ValueError(f"unsupported site kind: {kind}")

    def _site_module(self, site: LayerSite) -> Any:
        layer = self._layer(site.layer_index)
        if site.kind == "pre_moe_residual":
            return layer.post_attention_layernorm
        if site.kind == "router_input":
            return layer.mlp
        if site.kind in {"router_logits", "router_topk_ids", "router_topk_weights"}:
            return layer.mlp.router
        if site.kind == "moe_output":
            return layer.mlp
        if site.kind == "post_moe_residual":
            return layer
        raise ValueError(f"unsupported site kind: {site.kind}")

    def harmony_candidate_render(self, prompt: str, candidate: str) -> str:
        """Return the exact Harmony training-style final-channel candidate render."""

        return self._harmony_render(prompt).render_candidate(candidate)

    def score_candidates(
        self,
        prompt: str,
        candidates: list[str],
    ) -> dict[str, CandidateScore]:
        """Score candidates using the same complete Harmony render as patch paths."""

        render = self._harmony_render(prompt)
        return score_candidates_hf(
            model=self._require_model(),
            tokenizer=self._require_tokenizer(),
            prompt=render.candidate_prefix,
            candidates=candidates,
            suffix=render.candidate_suffix,
            device=str(self._input_device()),
        )

    def harmony_render_provenance(
        self,
        prompt: str,
        *,
        candidates: tuple[str, ...] = ("CORRECT", "INCORRECT"),
        include_text: bool = False,
    ) -> dict[str, Any]:
        render = self._harmony_render(prompt)
        payload = render.provenance(candidates)
        if include_text:
            payload["rendered_prompt_text"] = render.candidate_prefix
        return payload

    def _chat_completion_prompt(self, prompt: str) -> str:
        """Return the cached prefix ending at Harmony's final-message boundary."""

        return self._harmony_render(prompt).candidate_prefix

    def _encode_prompt_candidate(
        self,
        prompt: str,
        candidate: str,
    ) -> tuple[dict[str, Any], int, int, list[int]]:
        """Encode the exact Harmony training render while scoring only the candidate."""

        render = self._harmony_render(prompt)
        encoded = encode_prompt_candidate_hf(
            tokenizer=self._require_tokenizer(),
            prompt=render.candidate_prefix,
            candidate=candidate,
            suffix=render.candidate_suffix,
            device=self._input_device(),
        )
        return (
            encoded.inputs,
            encoded.target_token_start,
            encoded.target_token_end,
            encoded.target_ids,
        )

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        """Embed the trace in a cached Harmony final-channel prefix with exact spans."""

        prompt = self._harmony_render(render.prompt).candidate_prefix
        offset = prompt.find(render.prompt)
        if offset < 0:
            raise RuntimeError("GPT-OSS Harmony template did not preserve the raw prompt text")
        return shift_render_into_prompt(render, prompt=prompt, offset=offset)

    def _harmony_render(self, prompt: str) -> HarmonyCandidateRender:
        cached = self._harmony_cache.get(prompt)
        if cached is not None:
            return cached
        tokenizer = self._require_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("GPT-OSS tokenizer does not expose apply_chat_template")
        template = self._chat_template()
        marker = f"{_HARMONY_MARKER_PREFIX}{_sha256_text(prompt)[:24]}"
        if marker in prompt:
            raise ValueError("Harmony candidate boundary marker collides with the raw prompt")
        reasoning_effort = str(
            getattr(self.config, "harmony_reasoning_effort", "medium")
            if self.config is not None
            else "medium"
        )
        rendered = str(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": marker},
                ],
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort=reasoning_effort,
            )
        )
        if rendered.count(marker) != 1:
            raise RuntimeError("GPT-OSS Harmony template did not preserve one candidate marker")
        prefix, suffix = rendered.split(marker, 1)
        if prompt not in prefix:
            raise RuntimeError("GPT-OSS Harmony template did not preserve the raw prompt text")
        if not suffix:
            raise RuntimeError("GPT-OSS Harmony candidate render is missing its terminal suffix")
        date_match = _HARMONY_DATE_PATTERN.search(prefix)
        result = HarmonyCandidateRender(
            raw_prompt=prompt,
            candidate_prefix=prefix,
            candidate_suffix=suffix,
            template_sha256=_sha256_text(template),
            rendered_date=None if date_match is None else date_match.group(1),
            reasoning_effort=reasoning_effort,
        )
        self._harmony_cache[prompt] = result
        return result

    def _chat_template(self) -> str:
        tokenizer = self._require_tokenizer()
        if hasattr(tokenizer, "get_chat_template"):
            template = tokenizer.get_chat_template()
        else:
            template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            raise RuntimeError("GPT-OSS tokenizer has no default Harmony chat template")
        return template

    def _validate_chat_template_contract(self) -> None:
        expected = (
            getattr(self.config, "expected_chat_template_sha256", None)
            if self.config is not None
            else None
        )
        actual = _sha256_text(self._chat_template())
        if expected is not None and actual != str(expected):
            raise RuntimeError(
                f"GPT-OSS Harmony template hash mismatch: actual={actual} expected={expected}"
            )

    def runtime_inspect(self) -> dict[str, Any]:
        """Inspect the loaded, post-quantization modules as the final runtime truth."""

        model = self._require_model()
        layers = self._layers()
        model_config = model.config
        layer_rows: list[dict[str, Any]] = []
        router_expert_counts: set[int] = set()
        router_top_ks: set[int] = set()
        expert_module_counts: set[int] = set()
        for index, layer in enumerate(layers):
            router = layer.mlp.router
            router_experts = int(router.weight.shape[0])
            router_top_k = int(router.top_k)
            expert_module_count = int(getattr(layer.mlp.experts, "num_experts", router_experts))
            router_expert_counts.add(router_experts)
            router_top_ks.add(router_top_k)
            expert_module_counts.add(expert_module_count)
            layer_rows.append(
                {
                    "layer": index,
                    "mlp_class": self._class_path(layer.mlp),
                    "router_class": self._class_path(router),
                    "experts_class": self._class_path(layer.mlp.experts),
                    "router_execution_mode": self._router_execution_mode(index),
                    "router_weight_shape": [int(value) for value in router.weight.shape],
                    "router_num_experts": router_experts,
                    "expert_module_num_experts": expert_module_count,
                    "router_top_k": router_top_k,
                }
            )
        layout_consistent = (
            len(router_expert_counts) == 1
            and len(router_top_ks) == 1
            and len(expert_module_counts) == 1
            and router_expert_counts == expert_module_counts
        )
        actual = {
            "layers": len(layers),
            "experts": next(iter(router_expert_counts), -1)
            if len(router_expert_counts) == 1
            else -1,
            "top_k": next(iter(router_top_ks), -1) if len(router_top_ks) == 1 else -1,
        }
        expected = {
            "layers": (
                None if self.config is None else self.config.expected_total_block_count
            ),
            "experts": None if self.config is None else self.config.expected_experts,
            "top_k": None if self.config is None else self.config.expected_top_k,
        }
        config_contract = {
            "layers": int(model_config.num_hidden_layers),
            "experts": int(model_config.num_local_experts),
            "top_k": int(model_config.num_experts_per_tok),
        }
        contract_matches = (
            layout_consistent
            and config_contract == actual
            and all(value is None or int(value) == actual[key] for key, value in expected.items())
        )
        modes = sorted({str(row["router_execution_mode"]) for row in layer_rows})
        architectures = list(getattr(model_config, "architectures", None) or [])
        quantization_config = self._jsonable_quantization_config(
            getattr(model_config, "quantization_config", None)
        )
        native_mxfp4 = (
            modes == ["mxfp4_functional_linear"]
            and isinstance(quantization_config, dict)
            and quantization_config.get("quant_method") == "mxfp4"
            and quantization_config.get("dequantize") is not True
        )
        package_versions: dict[str, str | None] = {}
        for package in ("transformers", "kernels", "triton"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        return {
            "schema_version": 1,
            "adapter": self.alias,
            "model_id": str(getattr(model_config, "_name_or_path", self.model_id)),
            "model_type": str(getattr(model_config, "model_type", "unknown")),
            "architectures": architectures,
            "actual_contract": actual,
            "config_contract": config_contract,
            "expected_contract": expected,
            "layout_consistent": layout_consistent,
            "contract_matches": contract_matches,
            "router_execution_modes": modes,
            "native_mxfp4": native_mxfp4,
            "functional_linear_intercept_required": any(
                mode == "mxfp4_functional_linear" for mode in modes
            ),
            "layers": layer_rows,
            "chat_template_sha256": _sha256_text(self._chat_template()),
            "quantization_config": quantization_config,
            "runtime_package_versions": package_versions,
            "mxfp4_kernel_load": self._mxfp4_kernel_load,
        }

    def _capture_inputs(
        self, inputs: dict[str, Any], spec: CaptureSpec
    ) -> tuple[ActivationBatch, Any]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        arrays: dict[str, Any] = {}
        handles: list[Any] = []
        fire_counts = {site_key(site): 0 for site in spec.sites}

        def store(site: LayerSite, tensor: Any) -> None:
            sequence_tensor = self._as_sequence_tensor(tensor, batch_size, sequence_length)
            selected = sequence_tensor[:, list(spec.token_positions), ...].detach()
            if selected.is_floating_point():
                selected = selected.to(_torch_dtype(spec.dtype))
            arrays[site_key(site)] = selected.cpu()
            fire_counts[site_key(site)] += 1

        router_sites: dict[int, list[LayerSite]] = {}
        for site in spec.sites:
            if site.kind in {"router_logits", "router_topk_ids", "router_topk_weights"}:
                router_sites.setdefault(site.layer_index, []).append(site)
            elif site.kind == "pre_moe_residual":

                def pre_moe_hook(
                    _module: Any, args: tuple[Any, ...], *, site: LayerSite = site
                ) -> None:
                    store(site, args[0])

                handles.append(self._site_module(site).register_forward_pre_hook(pre_moe_hook))
            elif site.kind == "router_input":

                def router_input_hook(
                    _module: Any, args: tuple[Any, ...], *, site: LayerSite = site
                ) -> None:
                    store(site, args[0])

                handles.append(self._site_module(site).register_forward_pre_hook(router_input_hook))
            elif site.kind in {"moe_output", "post_moe_residual"}:

                def output_hook(
                    _module: Any, _args: tuple[Any, ...], output: Any, *, site: LayerSite = site
                ) -> None:
                    store(site, self._primary_output(output))

                handles.append(self._site_module(site).register_forward_hook(output_hook))
            else:  # pragma: no cover - exhaustive for current SiteKind
                raise ValueError(f"unsupported GPT-OSS capture site: {site.kind}")

        callbacks: dict[int, Callable[[Any], Any]] = {}
        for layer_index, sites in router_sites.items():
            router = self._layer(layer_index).mlp.router

            def router_capture(
                router_logits: Any,
                *,
                router: Any = router,
                sites: list[LayerSite] = sites,
            ) -> Any:
                top_values, top_ids = torch.topk(router_logits, int(router.top_k), dim=-1)
                top_weights = torch.softmax(top_values, dim=-1, dtype=top_values.dtype)
                for site in sites:
                    if site.kind == "router_logits":
                        store(site, router_logits)
                    elif site.kind == "router_topk_ids":
                        store(site, top_ids)
                    else:
                        store(site, top_weights)
                return router_logits

            callbacks[id(router.weight)] = router_capture

        try:
            with _temporary_functional_linear_interceptors(callbacks):
                with torch.inference_mode():
                    outputs = model(**inputs, use_cache=False, output_router_logits=False)
            logits = outputs.logits.detach().cpu()
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(key for key, count in fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered GPT-OSS capture hooks did not fire: {missing}")
        metadata = {
            "logits_shape": tuple(outputs.logits.shape),
            "hook_fire_counts": fire_counts,
            "all_registered_hooks_fired": not missing,
            "router_execution_modes": {
                str(index): self._router_execution_mode(index) for index in router_sites
            },
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
        dispatch_fire_counts: dict[str, int] = {}
        dispatch_after_router: dict[str, bool] = {}
        actual_topk_matches: dict[str, bool | None] = {}
        routing_input_matches: dict[str, bool | None] = {}
        routing_artifacts_consumed: dict[str, bool | None] = {}
        patched_router_object_ids: dict[int, str] = {}
        routing_artifact_ids: dict[str, tuple[int, int, int]] = {}
        expected_topk: dict[str, Any] = {}
        manipulation_by_site: dict[str, dict[str, float]] = {}
        handles: list[Any] = []
        callbacks: dict[int, Callable[[Any], Any]] = {}

        for patch in patches:
            if patch.token_position >= sequence_length:
                raise IndexError(
                    f"{site_key(patch.site)} token_position {patch.token_position} "
                    f"out of range for sequence length {sequence_length}"
                )
            key = site_key(patch.site)
            if patch.site.kind in {"moe_output", "post_moe_residual"}:

                def output_patch_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    patch: PatchSpec = patch,
                    key: str = key,
                ) -> Any:
                    hook_fire_counts[key] += 1
                    primary = self._replace_sequence_slice(
                        self._primary_output(output), patch, batch_size, sequence_length
                    )
                    return self._restore_primary_output(output, primary)

                handles.append(
                    self._site_module(patch.site).register_forward_hook(output_patch_hook)
                )
            elif patch.site.kind == "router_logits":
                router = self._site_module(patch.site)
                dispatch_fire_counts[key] = 0
                dispatch_after_router[key] = False
                actual_topk_matches[key] = None
                routing_input_matches[key] = None
                routing_artifacts_consumed[key] = None

                def router_patch(
                    router_logits: Any,
                    *,
                    patch: PatchSpec = patch,
                    key: str = key,
                    router: Any = router,
                ) -> Any:
                    hook_fire_counts[key] += 1
                    receiver_sequence = self._as_sequence_tensor(
                        router_logits, batch_size, sequence_length
                    )
                    receiver_target = receiver_sequence[:, patch.token_position, ...]
                    receiver_ids = torch.topk(receiver_target, int(router.top_k), dim=-1).indices
                    if bool(patch.metadata.get("forced_reroute")):
                        if not receiver_target.dtype.is_floating_point:
                            raise TypeError(f"{key} forced reroute requires floating router logits")
                        limits = torch.finfo(receiver_target.dtype)
                        selected = torch.zeros_like(receiver_target, dtype=torch.bool)
                        selected.scatter_(-1, receiver_ids, True)
                        replacement = torch.where(
                            selected,
                            torch.full_like(receiver_target, limits.min),
                            torch.full_like(receiver_target, limits.max),
                        )
                        patched_sequence = receiver_sequence.clone()
                        patched_sequence[:, patch.token_position, ...] = replacement
                    else:
                        patched_sequence = self._replace_sequence_slice(
                            router_logits, patch, batch_size, sequence_length
                        )
                    patched_logits = patched_sequence.reshape_as(router_logits)
                    patched_target = patched_sequence[:, patch.token_position, ...]
                    patched_ids = torch.topk(patched_target, int(router.top_k), dim=-1).indices
                    expected_topk[key] = patched_ids.detach().clone()
                    patched_router_object_ids[id(patched_logits)] = key
                    manipulation_by_site[key] = router_manipulation_metrics(
                        receiver_target.detach().float().cpu().numpy(),
                        patched_target.detach().float().cpu().numpy(),
                        receiver_topk_ids=receiver_ids.detach().cpu().numpy(),
                        patched_topk_ids=patched_ids.detach().cpu().numpy(),
                    )
                    return patched_logits

                callbacks[id(router.weight)] = router_patch
                experts = self._layer(patch.site.layer_index).mlp.experts

                def experts_pre_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    key: str = key,
                    patch: PatchSpec = patch,
                ) -> None:
                    dispatch_fire_counts[key] += 1
                    dispatch_after_router[key] = hook_fire_counts[key] > 0
                    if (
                        len(args) > 1
                        and torch.is_tensor(args[1])
                        and args[1].dtype
                        in {
                            torch.int32,
                            torch.int64,
                        }
                    ):
                        actual = self._as_sequence_tensor(args[1], batch_size, sequence_length)[
                            :, patch.token_position, ...
                        ]
                        expected = expected_topk.get(key)
                        actual_topk_matches[key] = bool(
                            expected is not None
                            and torch.equal(actual.to(expected.device), expected)
                        )
                    else:
                        expected_artifacts = routing_artifact_ids.get(key)
                        scatter = kwargs.get("scatter_idx")
                        if scatter is None and len(args) > 3:
                            scatter = args[3]
                        if expected_artifacts is not None and len(args) > 2 and scatter is not None:
                            observed_artifacts = (id(args[1]), id(args[2]), id(scatter))
                            routing_artifacts_consumed[key] = (
                                observed_artifacts == expected_artifacts
                            )

                handles.append(
                    experts.register_forward_pre_hook(experts_pre_hook, with_kwargs=True)
                )
            else:
                raise ValueError(
                    f"patching site {patch.site.kind!r} is not implemented for GPT-OSS"
                )

        def observe_mxfp4_routing(
            router_logits: Any,
            artifacts: tuple[Any, Any, Any],
        ) -> None:
            key = patched_router_object_ids.pop(id(router_logits), None)
            if key is None:
                return
            patch = next(item for item in patches if site_key(item.site) == key)
            actual_ids = torch.topk(
                self._as_sequence_tensor(router_logits, batch_size, sequence_length)[
                    :, patch.token_position, ...
                ],
                int(self._site_module(patch.site).top_k),
                dim=-1,
            ).indices
            expected = expected_topk.get(key)
            routing_input_matches[key] = bool(
                expected is not None and torch.equal(actual_ids.to(expected.device), expected)
            )
            routing_artifact_ids[key] = (
                id(artifacts[0]),
                id(artifacts[1]),
                id(artifacts[2]),
            )

        routing_state: dict[str, bool]
        try:
            with _temporary_mxfp4_routing_interceptor(observe_mxfp4_routing) as routing_state:
                with _temporary_functional_linear_interceptors(callbacks):
                    with torch.inference_mode():
                        outputs = model(**inputs, use_cache=False, output_router_logits=False)
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(key for key, count in hook_fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered GPT-OSS patch hooks did not fire: {missing}")
        dispatch_by_site: dict[str, dict[str, Any]] = {}
        for patch in patches:
            if patch.site.kind != "router_logits":
                continue
            key = site_key(patch.site)
            actual_match = actual_topk_matches[key]
            consumed = dispatch_fire_counts[key] > 0 and dispatch_after_router[key]
            native_routing_verified = (
                routing_input_matches[key] is True and routing_artifacts_consumed[key] is True
            )
            dispatch_by_site[key] = {
                "router_execution_mode": self._router_execution_mode(patch.site.layer_index),
                "expert_dispatch_hook_fire_count": dispatch_fire_counts[key],
                "dispatch_after_router_intercept": dispatch_after_router[key],
                "actual_topk_matches_patched_topk": actual_match,
                "routing_input_matches_patched_topk": routing_input_matches[key],
                "routing_artifacts_consumed": routing_artifacts_consumed[key],
                "downstream_dispatch_verified": consumed
                and (actual_match is True or native_routing_verified),
            }
        diagnostics: dict[str, Any] = {
            "simultaneous": isinstance(request, PatchBundle),
            "registered_site_count": len(patches),
            "registered_sites": [site_key(patch.site) for patch in patches],
            "hook_fire_counts": hook_fire_counts,
            "all_registered_hooks_fired": not missing,
            "manipulation_by_site": manipulation_by_site,
            "dispatch_by_site": dispatch_by_site,
            "mxfp4_routing_interceptor": routing_state,
            "all_router_dispatch_verified": all(
                bool(value["downstream_dispatch_verified"]) for value in dispatch_by_site.values()
            ),
        }
        return outputs, diagnostics

    def _router_execution_mode(self, layer_index: int) -> str:
        forward = self._layer(layer_index).mlp.forward
        function = getattr(forward, "__func__", forward)
        module = str(getattr(function, "__module__", ""))
        name = str(getattr(function, "__name__", ""))
        if module.endswith("transformers.integrations.mxfp4") or name == "mlp_forward":
            return "mxfp4_functional_linear"
        return "router_module_forward"

    @staticmethod
    def _primary_output(output: Any) -> Any:
        if isinstance(output, tuple):
            if not output:
                raise RuntimeError("GPT-OSS module returned an empty tuple")
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

    @staticmethod
    def _jsonable_quantization_config(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            return value
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return str(value)
