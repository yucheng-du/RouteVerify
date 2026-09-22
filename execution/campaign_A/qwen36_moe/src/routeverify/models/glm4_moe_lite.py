"""GLM-4.7-Flash native BF16 adapter for Transformers 5.12.1.

Attention/MLA and expert computation remain upstream. The intervention endpoint
is the routed aggregate before the independent shared branch is added.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Mapping
from typing import Any

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
)
from routeverify.models.olmoe import _require_torch
from routeverify.models.qwen3_moe import Qwen3MoEAdapter

GLM47_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
GLM47_TEMPLATE_SHA256 = "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"
GLM47_TRANSFORMERS_VERSION = "5.12.1"
GLM47_MOE_OUTPUT = "weighted_routed_aggregate_before_shared_expert_addition"


def _value(config: Any, key: str, default: Any = None) -> Any:
    return (
        config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)
    )


def _glm4_route_from_logits(gate: Any, logits: Any) -> RouterOutput:
    """Match native noaux_tc selection, including unbiased mixture weights.

    Correction bias affects membership only. Group scores sum the top two
    corrected sigmoid scores; selected uncorrected scores are optionally
    normalized, then multiplied by the routed scaling factor. Return FP32
    weights as the upstream GLM expert call does, including with BF16 states.
    """

    torch = _require_torch()
    scores = logits.sigmoid()
    choice = scores + gate.e_score_correction_bias
    groups = choice.view(-1, gate.n_group, gate.n_routed_experts // gate.n_group)
    group_scores = groups.topk(2, dim=-1).values.sum(dim=-1)
    group_ids = torch.topk(group_scores, k=gate.topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_ids, 1)
    mask = group_mask.unsqueeze(-1).expand_as(groups).reshape(-1, gate.n_routed_experts)
    choice = choice.masked_fill(~mask.bool(), float("-inf"))
    ids = torch.topk(choice, k=gate.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if gate.norm_topk_prob:
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    weights = weights * gate.routed_scaling_factor
    return RouterOutput(logits=logits, topk_weights=weights, topk_ids=ids)


class Glm4MoELiteAdapter(Qwen3MoEAdapter):
    """Pinned native GLM runtime, inheriting the common capture/patch engine.

    ``gate`` returns raw FP32 logits, while actual IDs/weights are consumed by
    ``mlp.experts``. Shared output is never patched at the common MoE endpoint.
    Dense block 0 is omitted from routed sites. Inheriting the common engine
    also preserves its single/multi-position patch ABI as it evolves.
    """

    alias = "glm4_moe_lite"
    model_id = "zai-org/GLM-4.7-Flash"
    capture_loading_info = True
    moe_config_paths = MoEConfigPaths(experts=("n_routed_experts",))
    router_semantics = RouterSemantics(
        identifier="glm4_moe_lite_sigmoid_bias_group_topk_norm_scale_v1",
        route_from_logits=_glm4_route_from_logits,
    )

    @classmethod
    def _validate_requested_config(cls, config: ModelConfig) -> None:
        if config.alias != cls.alias or config.model_id != cls.model_id:
            raise ValueError("GLM adapter requires the GLM-4.7-Flash alias and checkpoint")
        if config.revision != GLM47_REVISION or config.tokenizer_revision != GLM47_REVISION:
            raise ValueError(f"GLM model and tokenizer revisions must equal {GLM47_REVISION}")
        if config.dtype not in {"bfloat16", "bf16"}:
            raise ValueError("GLM-4.7-Flash protocol requires native BF16")
        if config.trust_remote_code:
            raise ValueError("GLM uses native Transformers code; trust_remote_code must be false")
        if config.expected_transformers_version != GLM47_TRANSFORMERS_VERSION:
            raise ValueError(f"GLM requires Transformers {GLM47_TRANSFORMERS_VERSION}")
        if config.expected_chat_template_sha256 != GLM47_TEMPLATE_SHA256:
            raise ValueError("GLM config must pin the official chat template SHA-256")

    @classmethod
    def static_config_preflight(cls, model_config: ModelConfig, hf_config: Any) -> dict[str, Any]:
        """Validate metadata only; never call a loader or fetch weights."""

        cls._validate_requested_config(model_config)
        count = int(_value(hf_config, "num_hidden_layers", 0))
        # TF 5.12.1 defaults to exactly this pattern when mlp_layer_types is absent.
        types = _value(hf_config, "mlp_layer_types") or ["dense"] + ["sparse"] * (count - 1)
        layout = MoEBlockLayout(count, tuple(i for i, kind in enumerate(types) if kind == "sparse"))
        adapter = cls()
        report = preflight_moe_layout(
            model_config=model_config,
            hf_config=hf_config,
            layout=layout,
            sites=build_moe_sites(layout, adapter._module_name),
            router_semantics_id=cls.router_semantics.identifier,
            config_paths=cls.moe_config_paths,
            adapter_alias=cls.alias,
            adapter_model_id=cls.model_id,
            adapter_class=f"{cls.__module__}.{cls.__qualname__}",
        )
        facts = {
            key: _value(hf_config, key, default)
            for key, default in {
                "hidden_size": None,
                "n_shared_experts": None,
                "moe_intermediate_size": None,
                "n_group": None,
                "topk_group": None,
                "norm_topk_prob": None,
                "routed_scaling_factor": None,
                "q_lora_rank": None,
                "kv_lora_rank": None,
                "qk_nope_head_dim": None,
                "qk_rope_head_dim": None,
                "v_head_dim": None,
                "rope_interleave": True,
                "first_k_dense_replace": 1,
                "topk_method": "noaux_tc",
            }.items()
        }
        errors = list(report["errors"])
        if len(types) != count or set(types) - {"dense", "sparse"}:
            errors.append("GLM mlp_layer_types does not match the decoder layout")
        for key, actual in facts.items():
            expected = getattr(model_config, f"expected_{key}", None)
            if expected is not None and actual != expected:
                errors.append(f"{key} mismatch: actual={actual} expected={expected}")
        dtype = str(_value(hf_config, "dtype", _value(hf_config, "torch_dtype", ""))).replace(
            "torch.", ""
        )
        if dtype != "bfloat16":
            errors.append(f"checkpoint dtype must be bfloat16, got {dtype!r}")
        if _value(hf_config, "quantization_config") is not None:
            errors.append("GLM BF16 protocol disallows a quantization_config")
        report.update(
            hook_relevant_config=facts,
            mlp_layer_types=list(types),
            moe_output_semantics=GLM47_MOE_OUTPUT,
            full_moe_output_boundary="model.layers.<block>.mlp",
            post_moe_residual_boundary="model.layers.<block>",
            attention_implementation="upstream_MLA_unchanged",
            checkpoint_gpu_smoke=False,
            errors=errors,
            passed=not errors,
        )
        return report

    def load(self, config: ModelConfig) -> None:
        self._validate_requested_config(config)
        if importlib.metadata.version("transformers") != GLM47_TRANSFORMERS_VERSION:
            raise RuntimeError(f"GLM requires Transformers {GLM47_TRANSFORMERS_VERSION}")
        ensure_external_cache()
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=False,
        )
        report = self.static_config_preflight(config, hf_config)
        if not report["passed"]:
            raise RuntimeError(f"GLM static preflight failed: {report['errors']}")
        super().load(config)
        self._validate_chat_template_contract()
        runtime = self.runtime_inspect()
        if not runtime["contract_matches"]:
            raise RuntimeError(f"GLM loaded runtime preflight failed: {runtime['errors']}")
        if self.discovery_report is not None:
            self.discovery_report.metadata.update(static_preflight=report, runtime_inspect=runtime)

    def _chat_completion_prompt(self, prompt: str) -> str:
        rendered = str(
            self._require_tokenizer().apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if not rendered.endswith("<|assistant|></think>"):
            raise RuntimeError("GLM non-thinking template did not emit the pinned assistant suffix")
        return rendered

    def _validate_chat_template_contract(self) -> None:
        template = getattr(self._require_tokenizer(), "chat_template", None)
        actual = (
            hashlib.sha256(template.encode()).hexdigest() if isinstance(template, str) else None
        )
        if actual != GLM47_TEMPLATE_SHA256:
            raise RuntimeError(f"GLM chat template hash mismatch: {actual}")

    def _router_block_indices(self) -> tuple[int, ...]:
        return tuple(i for i, layer in enumerate(self._layers()) if hasattr(layer.mlp, "gate"))

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        if kind == "moe_output":
            return f"model.layers.{layer_index}.mlp.experts"
        if kind == "router_input":
            return f"model.layers.{layer_index}.mlp.gate"
        return super()._module_name(layer_index, kind)

    def _site_module(self, site: LayerSite) -> Any:
        if site.layer_index not in self._router_block_indices():
            raise ValueError(f"GLM block {site.layer_index} is not a routed block")
        if site.kind == "moe_output":
            return self._layer(site.layer_index).mlp.experts
        if site.kind == "router_input":
            return self._layer(site.layer_index).mlp.gate
        return super()._site_module(site)

    def _unpack_router_output(
        self,
        router_module: Any,
        output: Any,
        *,
        site_key_value: str = "router",
    ) -> RouterOutput:
        if not _require_torch().is_tensor(output) or output.ndim != 2:
            raise RuntimeError(f"{site_key_value} GLM gate must return [tokens,experts] raw logits")
        return self._route_from_logits(router_module, output)

    def _restore_router_output(
        self, router_module: Any, original: Any, routed: RouterOutput
    ) -> Any:
        # Native mlp recomputes actual dispatch from these patched raw logits.
        return routed.logits

    def _router_dispatch_module(self, layer_index: int) -> Any:
        return self._layer(layer_index).mlp.experts

    def _capture_inputs(self, inputs: dict[str, Any], spec: Any) -> tuple[ActivationBatch, Any]:
        capture, logits = super()._capture_inputs(inputs, spec)
        metadata = dict(capture.metadata)
        metadata.update(
            router_logits_semantics="raw_fp32_pre_sigmoid_pre_bias_scores",
            router_topk_semantics="reconstructed_native_selection; actual_dispatch_verified_in_patch_diagnostics",
            moe_output_semantics=GLM47_MOE_OUTPUT,
            shared_branch="receiver_shared_experts_preserved",
            attention_implementation="upstream_MLA_unchanged",
        )
        return ActivationBatch(arrays=capture.arrays, metadata=metadata), logits

    def runtime_inspect(self) -> dict[str, Any]:
        """Inspect actual module, dtype and loading boundaries; no inference claims."""

        model = self._require_model()
        layers = list(self._layers())
        indices = self._router_block_indices()
        errors: list[str] = []
        rows: list[dict[str, Any]] = []
        for i in indices:
            mlp = layers[i].mlp
            required = ("gate", "experts", "shared_experts", "route_tokens_to_experts")
            if any(not hasattr(mlp, key) for key in required):
                errors.append(f"GLM block {i} lacks a native routed/shared/selection boundary")
                continue
            rows.append(
                {
                    "block_index": i,
                    "experts": int(mlp.gate.weight.shape[0]),
                    "top_k": int(mlp.gate.top_k),
                    "hidden_size": int(mlp.gate.weight.shape[1]),
                    "gate_dtype": str(mlp.gate.weight.dtype),
                    "correction_bias_dtype": str(mlp.gate.e_score_correction_bias.dtype),
                    "experts_dtype": str(mlp.experts.gate_up_proj.dtype),
                    "shared_dtype": str(mlp.shared_experts.gate_proj.weight.dtype),
                    "expert_class": type(mlp.experts).__name__,
                    "attention_class": type(layers[i].self_attn).__name__,
                }
            )
        if self.config is not None:
            static = self.static_config_preflight(self.config, model.config)
            errors.extend(static["errors"])
            if indices != tuple(self.config.expected_router_block_indices or ()):
                errors.append("actual routed block indices disagree with the checkpoint contract")
            for row in rows:
                if (
                    row["experts"] != self.config.expected_experts
                    or row["top_k"] != self.config.expected_top_k
                    or row["hidden_size"] != getattr(self.config, "expected_hidden_size", None)
                ):
                    errors.append(f"block {row['block_index']} expert/top-k contract mismatch")
                if any(
                    row[key] != "torch.bfloat16"
                    for key in ("gate_dtype", "experts_dtype", "shared_dtype")
                ):
                    errors.append(
                        f"block {row['block_index']} did not preserve native BF16 weights"
                    )
                if row["correction_bias_dtype"] != "torch.float32":
                    errors.append(f"block {row['block_index']} correction bias must remain FP32")
        info = self.loading_info
        loading = {
            key: list((info or {}).get(key, []))
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        }
        allowed_prefix = f"model.layers.{len(layers)}."
        unexpected = [
            key for key in loading["unexpected_keys"] if not key.startswith(allowed_prefix)
        ]
        loading_ok = (
            info is not None
            and not any(
                loading[key]
                for key in (
                    "missing_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            )
            and not unexpected
        )
        if self.config is not None and not loading_ok:
            errors.append(
                "loading diagnostics missing or contain non-MTP missing/unexpected/mismatched weights"
            )
        return {
            "runtime_architecture": type(model).__name__,
            "total_blocks": len(layers),
            "router_block_indices": list(indices),
            "layers": rows,
            "moe_output_semantics": GLM47_MOE_OUTPUT,
            "loading_info": loading,
            "allowed_auxiliary_prefix": allowed_prefix,
            "loading_info_ok": loading_ok,
            "contract_matches": not errors,
            "layout_consistent": not errors,
            "errors": errors,
            "checkpoint_gpu_smoke": False,
        }
