"""Binary step-verification helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from routeverify.analysis.probes import binary_auroc
from routeverify.eval.candidate_scoring import CandidateScore


def verification_margin(scores: dict[str, CandidateScore], *, normalized: bool = False) -> float:
    field = "mean_logprob" if normalized else "total_logprob"
    return float(getattr(scores["INCORRECT"], field) - getattr(scores["CORRECT"], field))


def binary_verification_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_breakdowns: bool = True,
) -> dict[str, Any]:
    """Summarize strict CORRECT/INCORRECT verifier rows."""

    if not rows:
        return {
            "n": 0,
            "accuracy": None,
            "auroc": None,
            "positives": 0,
            "negatives": 0,
            "mean_margin_incorrect_minus_correct": None,
        }

    labels = np.asarray([row["gold"] == "INCORRECT" for row in rows], dtype=bool)
    predictions = np.asarray([row["prediction"] == "INCORRECT" for row in rows], dtype=bool)
    margins = np.asarray([float(row["margin_incorrect_minus_correct"]) for row in rows], dtype=float)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    payload: dict[str, Any] = {
        "n": len(rows),
        "accuracy": float(np.mean(labels == predictions)),
        "auroc": _safe_binary_auroc(labels, margins),
        "positives": positives,
        "negatives": negatives,
        "mean_margin_incorrect_minus_correct": float(margins.mean()),
    }
    if include_breakdowns:
        payload["breakdowns"] = {
            "dataset": _grouped_binary_metrics(rows, "dataset"),
            "subset": _grouped_binary_metrics(rows, "subset"),
            "generator": _grouped_binary_metrics(rows, "generator"),
            "variant": _grouped_binary_metrics(rows, "variant"),
            "error_type": _grouped_binary_metrics(rows, "error_type"),
        }
    return payload


def _safe_binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return binary_auroc(labels, scores)
    except ValueError:
        return None


def _grouped_binary_metrics(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        group = "unknown" if value is None else str(value)
        groups.setdefault(group, []).append(row)
    return {
        group: binary_verification_metrics(group_rows, include_breakdowns=False)
        for group, group_rows in sorted(groups.items())
    }
