"""Finite-difference router visibility analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from routeverify.analysis.probes import mean_difference_direction


def router_visibility_metrics(
    gate_module: Any,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float = 0.1,
    random_directions: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure local router-logit sensitivity to an empirical error direction."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    if features.ndim != 2:
        raise ValueError("features must be [examples, dimensions]")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels/features length mismatch")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    payload: dict[str, Any] = {
        "status": "ok",
        "n": int(features.shape[0]),
        "positives": positives,
        "negatives": negatives,
        "feature_dim": int(features.shape[1]),
        "epsilon": epsilon,
        "random_directions": random_directions,
    }
    if positives == 0 or negatives == 0:
        payload["status"] = "insufficient_classes"
        return payload

    direction = mean_difference_direction(features, labels).astype(np.float32)
    main = _finite_difference_summary(gate_module, features, direction, epsilon=epsilon)
    payload.update({f"error_direction_{key}": value for key, value in main.items()})

    random_summaries = []
    rng = np.random.default_rng(seed)
    for _ in range(random_directions):
        random_direction = rng.standard_normal(size=(features.shape[1],)).astype(np.float32)
        norm = float(np.linalg.norm(random_direction))
        if norm == 0.0:
            continue
        random_summaries.append(
            _finite_difference_summary(
                gate_module,
                features,
                random_direction / norm,
                epsilon=epsilon,
            )
        )
    payload.update(_random_baseline_summary(random_summaries))
    random_jvp = payload.get("random_mean_normalized_jvp")
    error_jvp = payload.get("error_direction_mean_normalized_jvp")
    payload["visibility_ratio_vs_random"] = (
        None
        if not isinstance(random_jvp, float) or random_jvp == 0.0
        else float(cast(float, error_jvp) / random_jvp)
    )
    return payload


def router_visibility_markdown(metrics: Mapping[str, Any], *, max_rows: int = 40) -> str:
    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping) or not sites:
        return "No router visibility metrics found.\n"
    rows = sorted(
        sites.items(),
        key=lambda item: (
            float(item[1].get("visibility_ratio_vs_random", 0.0))
            if isinstance(item[1], Mapping)
            and item[1].get("visibility_ratio_vs_random") is not None
            else 0.0
        ),
        reverse=True,
    )
    lines = [
        "| site | status | n | positives | negatives | JVP norm | random JVP | ratio | top-k change |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site, value in rows[:max_rows]:
        if not isinstance(value, Mapping):
            continue
        lines.append(
            "| {site} | {status} | {n} | {pos} | {neg} | {jvp} | {random} | {ratio} | {topk} |".format(
                site=site,
                status=value.get("status", "unknown"),
                n=int(value.get("n", 0)),
                pos=int(value.get("positives", 0)),
                neg=int(value.get("negatives", 0)),
                jvp=_format_optional_float(value.get("error_direction_mean_normalized_jvp")),
                random=_format_optional_float(value.get("random_mean_normalized_jvp")),
                ratio=_format_optional_float(value.get("visibility_ratio_vs_random")),
                topk=_format_optional_float(value.get("error_direction_topk_change_fraction")),
            )
        )
    return "\n".join(lines) + "\n"


def _finite_difference_summary(
    gate_module: Any,
    features: np.ndarray,
    direction: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, float]:
    import torch

    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        raise ValueError("direction has zero norm")
    normalized_direction = direction.astype(np.float32) / direction_norm
    parameter = next(gate_module.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    gate_module.eval()
    with torch.inference_mode():
        hidden = torch.as_tensor(features, device=device, dtype=dtype)
        direction_tensor = torch.as_tensor(
            normalized_direction, device=device, dtype=dtype
        ).reshape(1, -1)
        baseline_logits, baseline_ids = _gate_logits_and_ids(gate_module(hidden), gate_module)
        plus_logits, plus_ids = _gate_logits_and_ids(
            gate_module(hidden + epsilon * direction_tensor),
            gate_module,
        )
        minus_logits, minus_ids = _gate_logits_and_ids(
            gate_module(hidden - epsilon * direction_tensor),
            gate_module,
        )
        jvp = (plus_logits.float() - minus_logits.float()) / (2 * epsilon)
        per_example_norm = torch.linalg.vector_norm(jvp.reshape(jvp.shape[0], -1), dim=-1)
        centered_delta = plus_logits.float() - baseline_logits.float()
        per_example_plus_delta = torch.linalg.vector_norm(
            centered_delta.reshape(centered_delta.shape[0], -1),
            dim=-1,
        )
    return {
        "mean_normalized_jvp": float(per_example_norm.mean().detach().cpu().item()),
        "max_normalized_jvp": float(per_example_norm.max().detach().cpu().item()),
        "mean_plus_logit_delta_norm": float(per_example_plus_delta.mean().detach().cpu().item()),
        "topk_change_fraction": _topk_change_fraction(baseline_ids, plus_ids, minus_ids),
    }


def _gate_logits_and_ids(output: Any, gate_module: Any) -> tuple[Any, Any]:
    import torch

    if isinstance(output, tuple):
        logits = output[0]
        ids = output[2] if len(output) > 2 else None
    else:
        logits = output
        ids = None
    if ids is None:
        top_k = int(getattr(gate_module, "top_k", min(2, int(logits.shape[-1]))))
        _, ids = torch.topk(torch.softmax(logits.float(), dim=-1), top_k, dim=-1)
    return logits, ids


def _topk_change_fraction(baseline_ids: Any, plus_ids: Any, minus_ids: Any) -> float:
    baseline = _ids_to_numpy(baseline_ids)
    plus = _ids_to_numpy(plus_ids)
    minus = _ids_to_numpy(minus_ids)
    changed = 0
    for base_row, plus_row, minus_row in zip(baseline, plus, minus, strict=True):
        base_set = set(base_row.tolist())
        changed += int(set(plus_row.tolist()) != base_set or set(minus_row.tolist()) != base_set)
    return float(changed / baseline.shape[0])


def _ids_to_numpy(ids: Any) -> np.ndarray:
    return cast(np.ndarray, ids.detach().cpu().numpy()).reshape(ids.shape[0], -1)


def _random_baseline_summary(summaries: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
    if not summaries:
        return {
            "random_mean_normalized_jvp": None,
            "random_max_normalized_jvp": None,
            "random_topk_change_fraction": None,
        }
    return {
        "random_mean_normalized_jvp": float(
            np.mean([summary["mean_normalized_jvp"] for summary in summaries])
        ),
        "random_max_normalized_jvp": float(
            np.mean([summary["max_normalized_jvp"] for summary in summaries])
        ),
        "random_topk_change_fraction": float(
            np.mean([summary["topk_change_fraction"] for summary in summaries])
        ),
    }


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"
