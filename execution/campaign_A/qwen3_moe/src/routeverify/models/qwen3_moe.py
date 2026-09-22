"""Qwen3 MoE adapter using the HF eager MoE layout."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from typing import Any

from routeverify.config import ModelConfig
from routeverify.data.prompts import PromptRender, shift_render_into_prompt
from routeverify.eval.candidate_scoring import CandidateScore, score_candidates_hf
from routeverify.models.olmoe import OLMoEAdapter


class Qwen3MoEAdapter(OLMoEAdapter):
    """Adapter for HF Qwen3-MoE eager modules.

    Transformers' Qwen3-MoE implementation exposes the same hook-relevant
    decoder layout used by the OLMoE adapter: decoder layers live under
    ``model.layers``, the sparse branch is ``layer.mlp``, and the router is
    ``layer.mlp.gate``.
    """

    alias = "qwen3_moe"
    model_id = "Qwen/Qwen3-30B-A3B"

    capture_loading_info = True

    def load(self, config: ModelConfig) -> None:
        """Record and check Qwen3's actual loaded parameters and diagnostics."""

        super().load(config)
        # Qwen3.6 and GLM inherit this adapter and retain their own runtime checks.
        if self.alias != "qwen3_moe":
            return
        runtime = self.runtime_inspect()
        if self.discovery_report is None:
            raise RuntimeError("Qwen3 load did not produce a discovery report")
        self.discovery_report.metadata["runtime_inspect"] = runtime
        if not runtime["passed"]:
            raise RuntimeError(f"Qwen3 loaded runtime failed technical checks: {runtime}")

    def runtime_inspect(self) -> dict[str, Any]:
        """Inspect model parameters only; static discovery and buffers stay separate."""

        model = self._require_model()
        by_dtype: Counter[str] = Counter()
        by_device: Counter[str] = Counter()
        numel_by_dtype: Counter[str] = Counter()
        numel_by_device: Counter[str] = Counter()
        all_cuda = True
        for parameter in model.parameters():
            dtype, device = str(parameter.dtype), str(parameter.device)
            by_dtype[dtype] += 1
            by_device[device] += 1
            numel_by_dtype[dtype] += int(parameter.numel())
            numel_by_device[device] += int(parameter.numel())
            all_cuda = all_cuda and parameter.device.type == "cuda"
        tensor_count = sum(by_dtype.values())
        numel = sum(numel_by_dtype.values())
        all_bf16 = tensor_count > 0 and numel > 0 and set(by_dtype) == {"torch.bfloat16"}
        all_cuda = tensor_count > 0 and numel > 0 and all_cuda
        expected_keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        schema_valid = bool(
            isinstance(self.loading_info, Mapping)
            and set(self.loading_info) == expected_keys
            and all(isinstance(value, (list, tuple, set, frozenset))
                    for value in self.loading_info.values())
        )
        clean = schema_valid and not any(self.loading_info.values())
        # Preserve every returned diagnostic, including additional/malformed keys.
        # HF collection values may be sets; normalize them for saved JSON artifacts.
        loading_info = json.loads(json.dumps(
            self.loading_info,
            default=lambda value: sorted(value, key=repr)
            if isinstance(value, (set, frozenset)) else repr(value),
        ))
        architecture = type(model).__name__
        model_type = getattr(model.config, "model_type", None)
        errors = []
        if architecture != "Qwen3MoeForCausalLM" or model_type != "qwen3_moe":
            errors.append("unexpected actual Qwen3 model class/config type")
        if not clean:
            errors.append("checkpoint loading diagnostics are nonempty or malformed")
        if not all_bf16:
            errors.append("actual parameters are not nonempty BF16 weights")
        if not all_cuda:
            errors.append("actual parameters are not entirely on CUDA")
        return {
            "schema_version": 1,
            "inspection_subject": "loaded_adapter_model",
            "runtime_model_class": f"{type(model).__module__}.{architecture}",
            "runtime_architecture": architecture,
            "runtime_model_type": model_type,
            "parameter_tensor_count": tensor_count,
            "parameter_numel": numel,
            "parameter_tensor_count_by_dtype": dict(by_dtype),
            "parameter_tensor_count_by_device": dict(by_device),
            "parameter_numel_by_dtype": dict(numel_by_dtype),
            "parameter_numel_by_device": dict(numel_by_device),
            "all_parameters_bfloat16": all_bf16,
            "all_parameters_cuda": all_cuda,
            "loading_info": loading_info,
            "loading_info_schema_valid": schema_valid,
            "loading_info_ok": clean,
            "errors": errors,
            "passed": not errors,
        }

    def score_candidates(self, prompt: str, candidates: list[str]) -> dict[str, CandidateScore]:
        """Score assistant completions under Qwen3's chat template in non-thinking mode."""

        return score_candidates_hf(
            model=self._require_model(),
            tokenizer=self._require_tokenizer(),
            prompt=self._chat_completion_prompt(prompt),
            candidates=candidates,
            device=str(self._input_device()),
        )

    def _encode_prompt_candidate(
        self, prompt: str, candidate: str
    ) -> tuple[dict[str, Any], int, int, list[int]]:
        """Encode candidate-scoring inputs under the same chat template as score_candidates.

        The OLMoE base encodes the raw prompt, but ``score_candidates`` wraps the prompt
        with Qwen3's chat template. The capture/patch paths
        (``capture_candidate_activations`` and ``score_candidates_with_candidate_patches``)
        go through this method, so without this override they would score the *raw* prompt
        while the unpatched baseline scores the *chat-templated* prompt -- a constant
        per-pair offset that makes the self-patch identity control fail.
        """

        return self._encode_rendered_prompt_candidate(
            self._chat_completion_prompt(prompt), candidate
        )

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        """Wrap a rendered prompt with Qwen3's chat template while preserving step spans."""

        prompt = self._chat_completion_prompt(render.prompt)
        offset = prompt.find(render.prompt)
        if offset < 0:
            raise RuntimeError("Qwen3 chat template did not preserve the raw prompt text")
        return shift_render_into_prompt(render, prompt=prompt, offset=offset)

    def _chat_completion_prompt(self, prompt: str) -> str:
        tokenizer = self._require_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("Qwen3 tokenizer does not expose apply_chat_template")
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
        except TypeError:
            kwargs.pop("enable_thinking")
            return str(tokenizer.apply_chat_template(messages, **kwargs))
