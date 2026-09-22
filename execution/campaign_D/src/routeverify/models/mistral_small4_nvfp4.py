"""Mistral Small 4: native tokenizer and worker-scoped NVFP4 interventions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from routeverify.config import ModelConfig
from routeverify.data.prompts import PromptRender
from routeverify.eval.candidate_scoring import CandidateScore
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchRequest,
    SiteKind,
    patch_positions,
    patch_specs,
    site_key,
)
from routeverify.models.base import ActivationBatch, GenerationConfig, ModelOutcome, PreparedBatch
from routeverify.models.discovery import DiscoveryReport, ModuleCandidate
from routeverify.models.moe import MoEBlockLayout, MoEConfigPaths, RouterSemantics, build_moe_sites
from routeverify.models.olmoe import OLMoEAdapter

MISTRAL_MODEL_ID = "mistralai/Mistral-Small-4-119B-2603-NVFP4"
MISTRAL_REVISION = "b1a9048590131d38491bd23a7c9f6ed0962f0358"
MISTRAL_VLLM_REVISION = "537af2c3a4ba7462ddc9bc94ec7a4ea496da6d2e"
MISTRAL_TEKKEN_SHA256 = "b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7"


def _route_in_worker(_module: Any, _logits: Any) -> Any:
    raise RuntimeError("Mistral router selection is delegated to the native worker; use recorded actual dispatch")


class MistralSmall4NVFP4Adapter(OLMoEAdapter):
    """Implement common candidate/capture/patch APIs over raw prompt logprobs.

    The driver holds a backend client, not a duplicate torch model. Dense
    vocabulary logits are intentionally not transferred over worker RPC; the
    scoring contract uses the actual teacher-forced candidate token logprobs.
    """
    alias = "mistral_small4_nvfp4"
    model_id = MISTRAL_MODEL_ID
    moe_config_paths = MoEConfigPaths(total_blocks=("n_layers",), experts=("moe.num_experts",), top_k=("moe.num_experts_per_tok",))
    router_semantics = RouterSemantics("mistral_small4_native_softmax_top4_renorm_v1", _route_in_worker)

    def __init__(self) -> None:
        super().__init__()
        self._request_count = 0
        self._runtime_layout: MoEBlockLayout | None = None

    @classmethod
    def validate_requested_config(cls, config: ModelConfig) -> None:
        required = {"alias": cls.alias, "model_id": MISTRAL_MODEL_ID,
                    "revision": MISTRAL_REVISION, "tokenizer_revision": MISTRAL_REVISION,
                    "expected_total_blocks": 36, "expected_router_blocks": 36,
                    "expected_experts": 128, "expected_top_k": 4,
                    "expected_vllm_revision": MISTRAL_VLLM_REVISION,
                    "expected_tekken_sha256": MISTRAL_TEKKEN_SHA256,
                    "require_native_nvfp4": True, "native_reasoning_effort": "none"}
        for key, expected in required.items():
            if getattr(config, key, None) != expected:
                raise ValueError(f"Mistral pinned contract requires {key}={expected!r}")
        if config.dtype not in {"bfloat16", "bf16"} or config.trust_remote_code:
            raise ValueError("Mistral native NVFP4 requires BF16 unquantized modules and trust_remote_code=false")
        if config.expected_router_block_indices != list(range(36)):
            raise ValueError("Mistral requires all 36 absolute routed-block indices")

    def load_tokenizer(self, config: ModelConfig, tokenizer_path: str | Path | None = None) -> None:
        from routeverify.models.mistral_tokenizer import NativeMistralTokenizer
        self.validate_requested_config(config)
        root = Path(tokenizer_path) if tokenizer_path else Path(config.cache_dir) / ("models--" + config.model_id.replace("/", "--")) / "snapshots" / config.tokenizer_revision
        path = root / "tekken.json" if root.is_dir() else root
        self.tokenizer = NativeMistralTokenizer.from_file(
            path, expected_sha256=MISTRAL_TEKKEN_SHA256,
            reasoning_effort=config.native_reasoning_effort,
        )
        self.config = config

    def tokenizer_provenance(self, tokenizer_path: str | None = None) -> dict[str, Any]:
        tokenizer = self._require_tokenizer()
        return {"tokenizer_class": type(tokenizer).__name__,
                "asset_sha256": {"tekken.json": tokenizer.source_sha256},
                "mistral_common_version": tokenizer.mistral_common_version,
                "reasoning_effort": tokenizer.reasoning_effort,
                "offset_alignment": "exact_native_token_bytes_projected_to_unicode_character_coverage",
                "requested_revision": MISTRAL_REVISION,
                "tokenizer_path_override": tokenizer_path,
                "chat_template_sha256": None, "template_contract": "mistral_common_native_request_serialization"}

    def load(self, config: ModelConfig) -> None:
        from routeverify.models.mistral_vllm_backend import VllmMistralBackend
        self.validate_requested_config(config)
        self.load_tokenizer(config, getattr(config, "tokenizer_path", None))
        backend = VllmMistralBackend.load(config)
        self.model = backend
        self.loading_info = backend.runtime_report
        self._runtime_layout = MoEBlockLayout(36, tuple(range(36)))
        self.discovery_report = DiscoveryReport(
            alias=self.alias, model_id=self.model_id,
            candidates=[ModuleCandidate(self._module_name(i, "router_logits"), "GateLinear", "router") for i in range(36)],
            metadata={"revision": config.revision, "load_weights": True,
                      "execution_backend": "vllm_worker", "runtime_inspect": self.runtime_inspect()},
        )

    def moe_block_layout(self) -> MoEBlockLayout:
        if self._runtime_layout is not None:
            return self._runtime_layout
        return MoEBlockLayout(36, tuple(range(36)))

    def discover_layers(self) -> list[LayerSite]:
        return build_moe_sites(self.moe_block_layout(), self._module_name)

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        root = f"language_model.model.layers.{layer_index}"
        return root + {
            "pre_moe_residual": ".post_attention_layernorm.input_sum",
            "router_input": ".mlp.gate.input", "router_logits": ".mlp.gate",
            "router_topk_ids": ".mlp.experts.runner.routed_experts.forward_modular.ids",
            "router_topk_weights": ".mlp.experts.runner.routed_experts.forward_modular.weights",
            "moe_output": ".mlp.experts.runner.apply_routed_output_transform",
            "post_moe_residual": ".output_fp32_sum",
        }[kind]

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        return self._require_tokenizer().chat_completion_render(render)

    def _chat_completion_prompt(self, prompt: str) -> str:
        return self.chat_completion_render(PromptRender(prompt, ())).prompt

    def _input_device(self) -> str:
        return "cpu"

    def _encode_prompt_candidate(self, prompt: str, candidate: str) -> tuple[dict, int, int, list[int]]:
        import torch
        encoded = self._require_tokenizer().encode_candidate(prompt, candidate)
        ids = torch.tensor([encoded.input_ids], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, encoded.target_start, encoded.target_end, list(encoded.target_ids)

    def _encode_batch(self, batch: PreparedBatch) -> dict[str, Any]:
        import torch
        if len(batch.prompts) != 1:
            raise ValueError("native Mistral research requests require a single prompt")
        ids = torch.tensor([self._require_tokenizer().encode_chat_prompt(batch.prompts[0])], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def _request(self, inputs: dict, *, spec: CaptureSpec | None = None, patch: PatchRequest | None = None) -> dict:
        backend = self._require_model()
        ids = inputs["input_ids"]
        if tuple(ids.shape[:1]) != (1,) or ids.ndim != 2:
            raise ValueError("Mistral worker requires [1,sequence] inputs")
        if "attention_mask" in inputs and not bool(inputs["attention_mask"].all()):
            raise ValueError("Mistral worker forbids padded/truncated requests")
        self._request_count += 1
        request = {"schema_version": 1, "request_id": f"routeverify-{self._request_count}",
                   "input_ids": ids[0].tolist(), "capture_sites": [], "patches": [], "capture_dtype": "float32"}
        if spec is not None:
            if spec.dtype not in {"float32", "fp32"}:
                raise ValueError("Mistral capture requires float32 to preserve the delayed-residual sum")
            request["capture_sites"] = [{"layer_index": s.layer_index, "kind": s.kind,
                                          "token_positions": list(spec.token_positions), "key": site_key(s)} for s in spec.sites]
        if patch is not None:
            for item in patch_specs(patch):
                replacement = item.replacement
                if hasattr(replacement, "detach"):
                    replacement = replacement.detach().float().cpu().numpy()
                request["patches"].append({"layer_index": item.site.layer_index, "kind": item.site.kind,
                                            "token_positions": list(patch_positions(item)), "key": site_key(item.site),
                                            "replacement": np.asarray(replacement, dtype=np.float32).tolist(),
                                            "metadata": dict(item.metadata), "strict_shape": item.strict_shape})
        return backend.execute(request)

    @staticmethod
    def _candidate_score(candidate: str, output: dict, start: int, end: int) -> CandidateScore:
        values = output["prompt_token_logprobs"][start:end]
        if len(values) != end - start or not values or any(value is None or not math.isfinite(value) for value in values):
            raise RuntimeError("native candidate logprob span is missing or nonfinite")
        total = math.fsum(values)
        return CandidateScore(candidate, total, total / len(values), len(values))

    def score_candidates(self, prompt: str, candidates: list[str]) -> dict[str, CandidateScore]:
        if not candidates:
            raise ValueError("candidates must be nonempty")
        result = {}
        for candidate in candidates:
            inputs, start, end, _ = self._encode_prompt_candidate(prompt, candidate)
            result[candidate] = self._candidate_score(candidate, self._request(inputs), start, end)
        return result

    def _score_candidate_with_patch_diagnostics(self, prompt: str, candidate: str, patch: PatchRequest) -> tuple[CandidateScore, dict]:
        inputs, start, end, _ = self._encode_prompt_candidate(prompt, candidate)
        output = self._request(inputs, patch=patch)
        diagnostics = {**output["diagnostics"], "router_semantics_id": self.router_semantics.identifier,
                       "score_backend": "vllm_raw_prompt_token_logprobs"}
        return self._candidate_score(candidate, output, start, end), diagnostics

    def _capture_inputs(self, inputs: dict, spec: CaptureSpec) -> tuple[ActivationBatch, None]:
        output = self._request(inputs, spec=spec)
        arrays = {key: np.asarray(value, dtype=np.int64 if key.endswith("router_topk_ids") else np.float32)
                  for key, value in output["captures"].items()}
        metadata = {"capture_token_axis": 1, "token_positions": list(spec.token_positions),
                    "moe_layout": self.moe_block_layout().to_metadata(),
                    "router_semantics_id": self.router_semantics.identifier,
                    "moe_output_semantics": "native_weighted_routed_aggregate_before_shared_add",
                    "post_moe_residual_semantics": "fp32_sum_of_native_delayed_residual_components",
                    "native_capture_metadata": output.get("capture_metadata", {}),
                    "worker_diagnostics": output["diagnostics"],
                    "dense_logits_returned": False}
        return ActivationBatch(arrays, metadata), None

    def run_with_patch(self, batch: PreparedBatch, patch: PatchRequest) -> ModelOutcome:
        output = self._request(self._encode_batch(batch), patch=patch)
        return ModelOutcome(logits=None, metadata={"patch_diagnostics": output["diagnostics"],
                           "prompt_token_logprobs": output["prompt_token_logprobs"],
                           "output_contract": "actual_prompt_token_logprobs; dense_vocabulary_logits_not_returned"})

    def run_with_patch_bundle(self, batch: PreparedBatch, bundle: PatchBundle) -> ModelOutcome:
        return self.run_with_patch(batch, bundle)

    def forward_logits(self, batch: PreparedBatch) -> Any:
        raise NotImplementedError("use score_candidates: this worker transports actual token logprobs, not dense logits")

    def generate(self, prompt: str, generation: GenerationConfig) -> str:
        raise NotImplementedError("native Mistral research bridge exposes candidate scoring; free generation is not implemented")

    def runtime_inspect(self) -> dict[str, Any]:
        backend = self._require_model()
        worker = backend.runtime_report.get("worker", {})
        return {"layout_consistent": worker.get("passed") is True, "contract_matches": worker.get("passed") is True,
                "moe_layout": self.moe_block_layout().to_metadata(), "worker": worker,
                "execution_backend": "native_vllm_worker", "native_nvfp4_gpu_verified": False,
                "gpu_verification_note": "loading/module inspection alone does not certify research smoke"}

    def resource_metrics(self) -> dict[str, Any]:
        return self._require_model().resource_metrics()
