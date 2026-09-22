"""Correct-versus-corrupt continuation scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from routeverify.eval.candidate_scoring import CandidateScore


def continuation_margin(
    scores: dict[str, CandidateScore],
    *,
    correct_candidate: str,
    corrupt_candidate: str,
    normalized: bool = True,
) -> float:
    field = "mean_logprob" if normalized else "total_logprob"
    return float(getattr(scores[correct_candidate], field) - getattr(scores[corrupt_candidate], field))


def continuation_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_breakdowns: bool = True,
) -> dict[str, Any]:
    """Summarize correct-next-step preference rows."""

    normalized_margins = [float(row["margin_correct_minus_corrupt_mean"]) for row in rows]
    total_margins = [float(row["margin_correct_minus_corrupt_total"]) for row in rows]
    payload: dict[str, Any] = {
        "n": len(rows),
        "mean_margin_correct_minus_corrupt_mean": _mean_or_none(normalized_margins),
        "mean_margin_correct_minus_corrupt_total": _mean_or_none(total_margins),
        "preference_rate_correct_over_corrupt_mean": _preference_rate(normalized_margins),
        "preference_rate_correct_over_corrupt_total": _preference_rate(total_margins),
    }
    if include_breakdowns:
        payload["breakdowns"] = {
            "error_type": _grouped_continuation_metrics(rows, "error_type"),
            "boundary_step": _grouped_continuation_metrics(rows, "boundary_step"),
        }
    return payload


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _preference_rate(values: Sequence[float]) -> float | None:
    return None if not values else sum(value > 0 for value in values) / len(values)


def _grouped_continuation_metrics(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        group = "unknown" if value is None else str(value)
        groups.setdefault(group, []).append(row)
    return {
        group: continuation_metrics(group_rows, include_breakdowns=False)
        for group, group_rows in sorted(groups.items())
    }
