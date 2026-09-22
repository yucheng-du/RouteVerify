"""Validated data objects shared across datasets and prompts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast


def _stable_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any, *, length: int = 16) -> str:
    """Return a deterministic short hash for JSON-like content."""

    return hashlib.sha256(_stable_payload(value).encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True)
class TraceExample:
    sample_id: str
    dataset: str
    subset: str
    problem: str
    steps: tuple[str, ...]
    first_error: int | None
    error_steps: tuple[int, ...]
    is_all_correct: bool
    generator: str | None = None
    pair_id: str | None = None
    variant: Literal["natural", "clean", "corrupt"] = "natural"
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not self.dataset:
            raise ValueError("dataset is required")
        if not self.subset:
            raise ValueError("subset is required")
        if not self.problem.strip():
            raise ValueError("problem must be non-empty")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        if any(not step.strip() for step in self.steps):
            raise ValueError("steps must not contain empty strings")
        if self.first_error is None and self.error_steps:
            raise ValueError("first_error cannot be None when error_steps are present")
        if self.first_error is not None:
            if self.first_error < 0 or self.first_error >= len(self.steps):
                raise ValueError("first_error is out of bounds")
            if self.first_error not in self.error_steps:
                raise ValueError("first_error must be present in error_steps")
        if any(step < 0 or step >= len(self.steps) for step in self.error_steps):
            raise ValueError("error_steps contains an out-of-bounds index")
        if self.is_all_correct != (self.first_error is None and not self.error_steps):
            raise ValueError("is_all_correct must match first_error/error_steps")

    @property
    def group_id(self) -> str:
        if self.pair_id is not None:
            return f"{self.dataset}:{self.subset}:pair:{self.pair_id}"
        payload = {
            "dataset": self.dataset,
            "subset": self.subset,
            "problem": normalize_text(self.problem),
        }
        return f"{self.dataset}:{self.subset}:problem:{stable_hash(payload)}"

    @property
    def content_hash(self) -> str:
        payload = {
            "dataset": self.dataset,
            "subset": self.subset,
            "problem": self.problem,
            "steps": self.steps,
            "first_error": self.first_error,
            "error_steps": self.error_steps,
            "variant": self.variant,
        }
        return stable_hash(payload)

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "subset": self.subset,
            "problem": self.problem,
            "steps": list(self.steps),
            "first_error": self.first_error,
            "error_steps": list(self.error_steps),
            "is_all_correct": self.is_all_correct,
            "generator": self.generator,
            "pair_id": self.pair_id,
            "variant": self.variant,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
            "group_id": self.group_id,
        }


def trace_example_from_json(row: dict[str, Any]) -> TraceExample:
    """Rebuild a TraceExample from the repository's normalized JSONL schema."""

    return TraceExample(
        sample_id=str(row["sample_id"]),
        dataset=str(row["dataset"]),
        subset=str(row["subset"]),
        problem=str(row["problem"]),
        steps=tuple(str(step) for step in row["steps"]),
        first_error=None if row["first_error"] is None else int(row["first_error"]),
        error_steps=tuple(int(step) for step in row.get("error_steps", ())),
        is_all_correct=bool(row["is_all_correct"]),
        generator=None if row.get("generator") is None else str(row.get("generator")),
        pair_id=None if row.get("pair_id") is None else str(row.get("pair_id")),
        variant=cast(Literal["natural", "clean", "corrupt"], row.get("variant", "natural")),
        error_type=None if row.get("error_type") is None else str(row.get("error_type")),
        metadata=dict(row.get("metadata") or {}),
    )


@dataclass(frozen=True)
class TokenizedTrace:
    sample_id: str
    input_ids: Any
    attention_mask: Any
    step_char_spans: tuple[tuple[int, int], ...]
    step_token_spans: tuple[tuple[int, int], ...]
    step_end_tokens: tuple[int, ...]
    query_token_span: tuple[int, int] | None

    def __post_init__(self) -> None:
        if len(self.step_char_spans) != len(self.step_token_spans):
            raise ValueError("step char/token span counts differ")
        if len(self.step_end_tokens) != len(self.step_char_spans):
            raise ValueError("step_end_tokens count differs from steps")
        for start, end in self.step_char_spans:
            if start < 0 or end <= start:
                raise ValueError("invalid step char span")
        for start, end in self.step_token_spans:
            if start < 0 or end <= start:
                raise ValueError("invalid step token span")


def normalize_text(text: str) -> str:
    """Conservative whitespace normalization for grouping and pair checks."""

    return " ".join(text.split())


def normalize_error_label(label: int | None, n_steps: int) -> tuple[int | None, tuple[int, ...], bool]:
    """Normalize ProcessBench-style earliest-error labels."""

    if label is None or int(label) == -1:
        return None, (), True
    first_error = int(label)
    if first_error < 0 or first_error >= n_steps:
        raise ValueError(f"first error index {first_error} is out of bounds for {n_steps} steps")
    return first_error, (first_error,), False
