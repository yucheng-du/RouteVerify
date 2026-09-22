"""Model adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from routeverify.config import ModelConfig
from routeverify.data.schema import TokenizedTrace, TraceExample
from routeverify.eval.candidate_scoring import CandidateScore


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class PreparedBatch:
    prompts: tuple[str, ...]
    tokenized: tuple[TokenizedTrace, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivationBatch:
    arrays: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutcome:
    logits: Any
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    alias: str
    model_id: str

    def load(self, config: ModelConfig) -> None: ...

    def format_prompt(self, example: TraceExample, mode: str, **kwargs: Any) -> str: ...

    def tokenize_with_spans(
        self,
        prompt: str,
        step_spans: list[tuple[int, int]],
    ) -> TokenizedTrace: ...

    def discover_layers(self) -> list[Any]: ...

    def score_candidates(self, prompt: str, candidates: list[str]) -> dict[str, CandidateScore]: ...

    def generate(self, prompt: str, generation: GenerationConfig) -> str: ...

    def capture(self, batch: PreparedBatch, spec: Any) -> ActivationBatch: ...

    def run_with_patch(self, batch: PreparedBatch, patch: Any) -> ModelOutcome: ...
