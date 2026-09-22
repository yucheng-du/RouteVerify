"""Candidate sequence log-likelihood scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class CandidateScore:
    candidate: str
    total_logprob: float
    mean_logprob: float
    token_count: int


@dataclass(frozen=True)
class EncodedCandidate:
    inputs: dict[str, Any]
    target_token_start: int
    target_token_end: int
    target_ids: list[int]


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_z = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return cast(np.ndarray, shifted - log_z)


def score_token_sequence(
    *,
    logits: np.ndarray,
    target_ids: list[int] | tuple[int, ...],
    candidate: str,
) -> CandidateScore:
    """Score target tokens from per-position next-token logits."""

    targets = np.asarray(target_ids, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocab]")
    if logits.shape[0] != len(targets):
        raise ValueError("logits rows must match target token count")
    if len(targets) == 0:
        raise ValueError("target_ids must be non-empty")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("target_ids contain out-of-vocabulary ids")
    log_probs = log_softmax(logits)
    token_log_probs = log_probs[np.arange(len(targets)), targets]
    total = float(np.sum(token_log_probs))
    return CandidateScore(
        candidate=candidate,
        total_logprob=total,
        mean_logprob=float(total / len(targets)),
        token_count=len(targets),
    )


def margin(
    scores: dict[str, CandidateScore],
    *,
    positive: str,
    negative: str,
    normalized: bool = False,
) -> float:
    """Return positive minus negative log-likelihood margin."""

    field = "mean_logprob" if normalized else "total_logprob"
    return float(getattr(scores[positive], field) - getattr(scores[negative], field))


def score_candidates_hf(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: list[str],
    suffix: str = "",
    device: str | None = None,
) -> dict[str, CandidateScore]:
    """Score candidates using a Hugging Face causal LM."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by model users
        raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

    if not candidates:
        raise ValueError("candidates must be non-empty")

    scores: dict[str, CandidateScore] = {}
    if device is None:
        device = next(model.parameters()).device
    for candidate in candidates:
        encoded = encode_prompt_candidate_hf(
            tokenizer=tokenizer,
            prompt=prompt,
            candidate=candidate,
            suffix=suffix,
            device=device,
        )
        with torch.inference_mode():
            logits = model(**encoded.inputs, use_cache=False).logits
        start = encoded.target_token_start - 1
        end = encoded.target_token_end - 1
        selected_logits = logits[0, start:end, :].detach().float().cpu().numpy()
        scores[candidate] = score_token_sequence(
            logits=selected_logits,
            target_ids=encoded.target_ids,
            candidate=candidate,
        )
    return scores


def encode_prompt_candidate_hf(
    *,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    suffix: str = "",
    device: str | Any | None = None,
) -> EncodedCandidate:
    """Tokenize the full rendered string and identify only the candidate tokens.

    Candidate scoring must follow the model's tokenization of the complete input.
    Tokenizing prompt and candidate separately can change BPE boundaries at the
    prompt/candidate join and silently score a different sequence.  ``suffix`` is
    useful for chat templates whose exact training render contains a terminal token;
    the suffix is included in the model input but excluded from the scored span.
    """

    if candidate == "":
        raise ValueError("candidate must be non-empty")
    full_text = prompt + candidate + suffix
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except (TypeError, NotImplementedError):
        encoded = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping", None)
    inputs = dict(encoded)
    token_start: int | None = None
    token_end: int | None = None
    if offsets is not None:
        candidate_start = len(prompt)
        candidate_end = candidate_start + len(candidate)
        overlapping = [
            index
            for index, (start, end) in enumerate(_offset_rows(offsets))
            if end > start and end > candidate_start and start < candidate_end
        ]
        if overlapping:
            token_start = overlapping[0]
            token_end = overlapping[-1] + 1

    if (token_start is None or token_end is None) and suffix:
        raise ValueError(
            "tokenizer offset mappings are required when a rendered candidate has a suffix"
        )
    if token_start is None or token_end is None:
        token_start, token_end = _common_prefix_suffix_span(
            tokenizer=tokenizer,
            prompt=prompt,
            full_text=full_text,
        )
    if token_start <= 0:
        raise ValueError("candidate starts at the first token and cannot be scored by a causal LM")
    if token_end <= token_start:
        raise ValueError("candidate suffix produced no target tokens")

    if device is not None:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    input_ids = inputs["input_ids"]
    target_ids = _token_ids_slice(input_ids, token_start, token_end)
    return EncodedCandidate(
        inputs=inputs,
        target_token_start=token_start,
        target_token_end=token_end,
        target_ids=target_ids,
    )


def _offset_rows(offsets: Any) -> list[tuple[int, int]]:
    if hasattr(offsets, "detach"):
        offsets = offsets.detach().cpu().tolist()
    elif hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _token_ids_slice(input_ids: Any, start: int, end: int) -> list[int]:
    row = input_ids[0] if hasattr(input_ids, "__getitem__") else input_ids
    sliced = row[start:end]
    if hasattr(sliced, "detach"):
        return [int(value) for value in sliced.detach().cpu().tolist()]
    return [int(value) for value in sliced]


def _common_prefix_suffix_span(*, tokenizer: Any, prompt: str, full_text: str) -> tuple[int, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=True, return_tensors="pt")["input_ids"]
    prompt_row = _token_ids_slice(prompt_ids, 0, _sequence_length(prompt_ids))
    full_row = _token_ids_slice(full_ids, 0, _sequence_length(full_ids))
    common = 0
    for left, right in zip(prompt_row, full_row, strict=False):
        if left != right:
            break
        common += 1
    if common >= len(full_row):
        raise ValueError("candidate suffix could not be found in full tokenization")
    return common, len(full_row)


def _sequence_length(input_ids: Any) -> int:
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[1])
    return len(input_ids[0])
