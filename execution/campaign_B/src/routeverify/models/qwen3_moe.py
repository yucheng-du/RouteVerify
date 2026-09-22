"""Qwen3 MoE adapter using the HF eager MoE layout."""

from __future__ import annotations

from typing import Any

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
