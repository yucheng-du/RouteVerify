"""ProcessBench behavior metrics and parsing."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from routeverify.data.schema import TraceExample

_INT_RE = re.compile(r"-?\d+")
InvalidFirstError = Literal["INVALID"]
FirstErrorPrediction = int | None | InvalidFirstError
INVALID_FIRST_ERROR: InvalidFirstError = "INVALID"


def parse_first_error_answer(
    text: str,
    *,
    n_steps: int | None = None,
) -> FirstErrorPrediction:
    """Parse one unambiguous human step number into a zero-based prediction.

    ``0`` and values below ``-1`` are never legal ProcessBench answers.  When
    ``n_steps`` is supplied, out-of-range human step numbers also fail closed.
    Text containing multiple integers is ambiguous rather than silently using
    the first number.
    """

    if n_steps is not None and n_steps <= 0:
        raise ValueError("n_steps must be positive")
    matches = list(_INT_RE.finditer(text))
    if len(matches) != 1:
        return INVALID_FIRST_ERROR
    value = int(matches[0].group(0))
    if value == -1:
        return None
    if value <= 0:
        return INVALID_FIRST_ERROR
    if n_steps is not None and value > n_steps:
        return INVALID_FIRST_ERROR
    # The RouteVerify prompt asks for human-readable, one-based step numbers.
    return value - 1


def first_error_candidate_labels(example: TraceExample) -> list[str]:
    """Return the only legal first-error answers for candidate scoring."""

    return ["-1", *(str(index + 1) for index in range(len(example.steps)))]


def first_error_prediction_from_candidate(candidate: str) -> FirstErrorPrediction:
    """Map a legal first-error candidate label to the normalized prediction."""

    if candidate == "-1":
        return None
    if not candidate.isdecimal():
        raise ValueError(f"invalid first-error candidate: {candidate}")
    value = int(candidate)
    if value <= 0:
        raise ValueError(f"invalid first-error candidate: {candidate}")
    return value - 1


def first_error_accuracy(
    examples: Sequence[TraceExample],
    predictions: Sequence[FirstErrorPrediction],
) -> float:
    if len(examples) != len(predictions):
        raise ValueError("examples/predictions length mismatch")
    if not examples:
        raise ValueError("examples must be non-empty")
    correct = sum(
        example.first_error == prediction
        for example, prediction in zip(examples, predictions, strict=True)
    )
    return correct / len(examples)


def first_error_metrics(
    examples: Sequence[TraceExample],
    predictions: Sequence[FirstErrorPrediction],
    *,
    include_breakdowns: bool = True,
) -> dict[str, Any]:
    """Compute strict first-error localization metrics plus useful baselines."""

    if len(examples) != len(predictions):
        raise ValueError("examples/predictions length mismatch")
    if not examples:
        raise ValueError("examples must be non-empty")

    exact_count = 0
    presence_tp = 0
    presence_fp = 0
    presence_fn = 0
    localization_tp = 0
    localization_fp = 0
    localization_fn = 0
    all_correct_count = 0
    all_correct_false_positives = 0
    invalid_count = 0
    distances: list[int] = []

    for example, prediction in zip(examples, predictions, strict=True):
        invalid = prediction == INVALID_FIRST_ERROR
        invalid_count += int(invalid)
        gold_error = example.first_error is not None
        predicted_error = prediction is not None and not invalid
        exact = prediction == example.first_error
        exact_count += int(exact)

        if gold_error and predicted_error:
            presence_tp += 1
        elif not gold_error and predicted_error:
            presence_fp += 1
        elif gold_error and not predicted_error:
            presence_fn += 1

        if gold_error and exact:
            localization_tp += 1
        elif predicted_error and not exact:
            localization_fp += 1
        if gold_error and not exact:
            localization_fn += 1

        if example.first_error is None:
            all_correct_count += 1
            all_correct_false_positives += int(predicted_error)
        elif isinstance(prediction, int):
            distances.append(abs(prediction - example.first_error))

    payload: dict[str, Any] = {
        "n": len(examples),
        "accuracy": exact_count / len(examples),
        "exact_count": exact_count,
        "error_presence": _prf(presence_tp, presence_fp, presence_fn),
        "first_error_localization": _prf(
            localization_tp,
            localization_fp,
            localization_fn,
        ),
        "mean_abs_error_index_distance": None if not distances else sum(distances) / len(distances),
        "all_correct_count": all_correct_count,
        "all_correct_false_positive_rate": (
            None
            if all_correct_count == 0
            else all_correct_false_positives / all_correct_count
        ),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(examples),
    }
    if include_breakdowns:
        payload["breakdowns"] = first_error_breakdowns(examples, predictions)
    return payload


def first_error_breakdowns(
    examples: Sequence[TraceExample],
    predictions: Sequence[FirstErrorPrediction],
) -> dict[str, dict[str, Any]]:
    """Group first-error metrics by stable, report-friendly dimensions."""

    return {
        "subset": _grouped_metrics(examples, predictions, lambda example: example.subset),
        "generator": _grouped_metrics(
            examples,
            predictions,
            lambda example: example.generator or "unknown",
        ),
        "trace_length": _grouped_metrics(
            examples,
            predictions,
            lambda example: _trace_length_bucket(len(example.steps)),
        ),
        "error_position": _grouped_metrics(examples, predictions, _error_position_bucket),
    }


def _grouped_metrics(
    examples: Sequence[TraceExample],
    predictions: Sequence[FirstErrorPrediction],
    key_fn: Callable[[TraceExample], str],
) -> dict[str, Any]:
    groups: dict[str, tuple[list[TraceExample], list[FirstErrorPrediction]]] = {}
    for example, prediction in zip(examples, predictions, strict=True):
        group_examples, group_predictions = groups.setdefault(key_fn(example), ([], []))
        group_examples.append(example)
        group_predictions.append(prediction)
    return {
        key: first_error_metrics(group_examples, group_predictions, include_breakdowns=False)
        for key, (group_examples, group_predictions) in sorted(groups.items())
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int | None]:
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _trace_length_bucket(length: int) -> str:
    if length <= 4:
        return "01-04"
    if length <= 8:
        return "05-08"
    if length <= 16:
        return "09-16"
    return "17+"


def _error_position_bucket(example: TraceExample) -> str:
    if example.first_error is None:
        return "all_correct"
    if example.first_error == 0:
        return "first_step"
    if len(example.steps) <= 1:
        return "only_step"
    relative_position = example.first_error / (len(example.steps) - 1)
    if relative_position < 1 / 3:
        return "early"
    if relative_position < 2 / 3:
        return "middle"
    return "late"
