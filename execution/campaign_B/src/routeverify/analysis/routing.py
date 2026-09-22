"""Routing descriptive statistics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def topk_jaccard(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    left_set = set(np.asarray(left).tolist())
    right_set = set(np.asarray(right).tolist())
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def fraction_changed(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError("route id arrays must have the same shape")
    return float(np.mean(left_array != right_array))


def expert_set_turnover(left: list[int] | np.ndarray, right: list[int] | np.ndarray) -> float:
    """Return 1-Jaccard for selected expert sets."""

    return 1.0 - topk_jaccard(left, right)


def js_divergence(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    left_array = left_array / left_array.sum()
    right_array = right_array / right_array.sum()
    midpoint = 0.5 * (left_array + right_array)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(left_array, midpoint) + 0.5 * kl(right_array, midpoint)


def aligned_gate_weight_js_divergence(
    left_ids: list[int] | np.ndarray,
    left_weights: list[float] | np.ndarray,
    right_ids: list[int] | np.ndarray,
    right_weights: list[float] | np.ndarray,
) -> float:
    """Compute JS divergence after aligning sparse gate weights by expert id."""

    left_vector, right_vector = aligned_gate_weight_vectors(
        left_ids,
        left_weights,
        right_ids,
        right_weights,
    )
    return js_divergence(left_vector, right_vector)


def aligned_gate_weight_vectors(
    left_ids: list[int] | np.ndarray,
    left_weights: list[float] | np.ndarray,
    right_ids: list[int] | np.ndarray,
    right_weights: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_id_array = np.asarray(left_ids, dtype=int).reshape(-1)
    right_id_array = np.asarray(right_ids, dtype=int).reshape(-1)
    left_weight_array = np.asarray(left_weights, dtype=float).reshape(-1)
    right_weight_array = np.asarray(right_weights, dtype=float).reshape(-1)
    if left_id_array.shape != left_weight_array.shape:
        raise ValueError("left ids and weights must have the same shape")
    if right_id_array.shape != right_weight_array.shape:
        raise ValueError("right ids and weights must have the same shape")
    expert_ids = sorted(set(left_id_array.tolist()) | set(right_id_array.tolist()))
    left_by_id = _weights_by_expert(left_id_array, left_weight_array)
    right_by_id = _weights_by_expert(right_id_array, right_weight_array)
    return (
        np.asarray([left_by_id.get(expert_id, 0.0) for expert_id in expert_ids], dtype=float),
        np.asarray([right_by_id.get(expert_id, 0.0) for expert_id in expert_ids], dtype=float),
    )


def _weights_by_expert(ids: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    output: dict[int, float] = {}
    for expert_id, weight in zip(ids, weights, strict=True):
        output[int(expert_id)] = output.get(int(expert_id), 0.0) + float(weight)
    return output


def routing_metrics_markdown(metrics: Mapping[str, Any], *, max_rows: int = 40) -> str:
    """Render a compact Markdown table of routing summaries."""

    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping) or not sites:
        return "No routing metrics found.\n"
    rows = sorted(
        sites.items(),
        key=lambda item: (
            int(item[1].get("n_pairs", 0)) if isinstance(item[1], Mapping) else 0,
            abs(float(item[1].get("mean_fraction_changed", 0.0) or 0.0))
            if isinstance(item[1], Mapping)
            else 0.0,
        ),
        reverse=True,
    )
    lines = [
        (
            "| site | routes | pairs | error boundaries | mean Jaccard | "
            "mean rank changed | mean set turnover | mean aligned JS | mean margin delta |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site, value in rows[:max_rows]:
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| {site} | {routes} | {pairs} | {errors} | {jaccard} | {changed} | {turnover} | {js} | {delta} |".format(
                site=site,
                routes=int(value.get("n_routes", 0)),
                pairs=int(value.get("n_pairs", 0)),
                errors=int(value.get("n_error_boundaries", 0)),
                jaccard=_format_optional_float(value.get("mean_topk_jaccard")),
                changed=_format_optional_float(value.get("mean_fraction_changed")),
                turnover=_format_optional_float(value.get("mean_expert_set_turnover")),
                js=_format_optional_float(value.get("mean_gate_weight_js_divergence")),
                delta=_format_optional_float(value.get("mean_routing_margin_delta_corrupt_minus_clean")),
            )
        )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"
