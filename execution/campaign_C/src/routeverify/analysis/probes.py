"""Low-capacity probe helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from routeverify.analysis.protocols import (
    build_repeated_grouped_split_manifest,
    group_bootstrap_metric_ci,
    permute_labels_by_group,
    validate_repeated_grouped_split_manifest,
)
from routeverify.data.schema import stable_hash

SAFE_METADATA_NUMERIC_FIELDS = (
    "boundary_step",
    "token_position",
    "token_index",
    "num_positions_in_file",
    "relative_capture_position",
    "is_last_captured_boundary",
)
SAFE_METADATA_CATEGORICAL_FIELDS = (
    "dataset",
    "subset",
    "generator",
    "prompt_mode",
    "input_format",
    "boundary_strategy",
    "error_type",
)
DIRECT_LABEL_LEAKAGE_FIELDS = frozenset(
    {
        "variant",
        "label",
        "gold",
        "target",
        "prediction",
        "is_error_boundary",
        "is_all_correct",
        "first_error",
        "error_steps",
        "boundary_match_role",
        "boundary_match_distance",
        "matched_target_sample_id",
        "sample_id",
        "pair_id",
        "group_id",
        "content_hash",
        "prompt_hash",
        "input_ids_hash",
        "activation_file",
    }
)


def mean_difference_direction(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be [examples, dimensions]")
    labels = np.asarray(labels, dtype=bool)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels/features length mismatch")
    if labels.all() or (~labels).all():
        raise ValueError("both classes are required")
    direction = features[labels].mean(axis=0) - features[~labels].mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("mean-difference direction has zero norm")
    return cast(np.ndarray, direction / norm)


def cosine_scores(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    feature_norms = np.linalg.norm(features, axis=1)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("direction has zero norm")
    return cast(
        np.ndarray, (features @ direction) / np.maximum(feature_norms * direction_norm, 1e-12)
    )


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC for boolean labels using average ranks for ties."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels/scores length mismatch")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("both classes are required")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=float)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def mean_difference_probe_summary(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit and evaluate a mean-difference cosine probe on the same rows."""

    labels = np.asarray(labels, dtype=bool)
    summary: dict[str, Any] = {
        "n": int(labels.shape[0]),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
        "probe": "mean_difference_cosine",
    }
    if features.ndim != 2:
        summary["status"] = "invalid_feature_shape"
        return summary
    if summary["positives"] == 0 or summary["negatives"] == 0:
        summary["status"] = "insufficient_classes"
        return summary
    direction = mean_difference_direction(features, labels)
    scores = cosine_scores(features, direction)
    summary.update(
        {
            "status": "ok",
            "auroc": binary_auroc(labels, scores),
            "positive_score_mean": float(scores[labels].mean()),
            "negative_score_mean": float(scores[~labels].mean()),
            "score_gap": float(scores[labels].mean() - scores[~labels].mean()),
        }
    )
    return summary


def grouped_holdout_probe_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    seed: int = 0,
    holdout_fraction: float = 0.2,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Fit on grouped train rows and evaluate on held-out groups."""

    labels = np.asarray(labels, dtype=bool)
    if features.ndim != 2:
        return {"status": "invalid_feature_shape"}
    if features.shape[0] != labels.shape[0] or len(groups) != labels.shape[0]:
        raise ValueError("features, labels, and groups must have the same row count")
    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"status": "insufficient_groups", "num_groups": len(unique_groups)}
    split_manifest = build_repeated_grouped_split_manifest(
        groups,
        seed=seed,
        repeats=1,
        eval_fraction=holdout_fraction,
    )
    fold = cast(dict[str, Any], split_manifest["folds"][0])
    summary = fixed_group_holdout_probe_summary(
        features,
        labels,
        groups,
        train_group_ids=cast(list[str], fold["train_group_ids"]),
        eval_group_ids=cast(list[str], fold["eval_group_ids"]),
        split_seed=seed,
        split_hash=str(fold["fold_hash"]),
        bootstrap_resamples=bootstrap_resamples,
    )
    summary["holdout_fraction"] = holdout_fraction
    summary["split_manifest_hash"] = split_manifest["manifest_hash"]
    return summary


def fixed_group_holdout_probe_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    train_group_ids: Sequence[str],
    eval_group_ids: Sequence[str],
    split_seed: int | None = None,
    split_hash: str | None = None,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Fit/evaluate a probe on an exact precomputed group partition."""

    labels = np.asarray(labels, dtype=bool)
    if features.ndim != 2:
        return {"status": "invalid_feature_shape"}
    if features.shape[0] != labels.shape[0] or len(groups) != labels.shape[0]:
        raise ValueError("features, labels, and groups must have the same row count")
    train_groups = set(train_group_ids)
    eval_groups = set(eval_group_ids)
    overlap = train_groups & eval_groups
    if overlap:
        raise ValueError(f"group leakage across train/eval: {sorted(overlap)[:5]}")
    observed_groups = set(groups)
    if train_groups | eval_groups != observed_groups:
        raise ValueError("precomputed split does not cover the observed feature groups")

    eval_mask = np.asarray([group in eval_groups for group in groups], dtype=bool)
    train_mask = ~eval_mask
    train_labels = labels[train_mask]
    eval_labels = labels[eval_mask]
    summary: dict[str, Any] = {
        "status": "ok",
        "split": "grouped_manifest",
        "split_seed": split_seed,
        "split_hash": split_hash,
        "num_groups": len(observed_groups),
        "num_train_groups": len(train_groups),
        "num_holdout_groups": len(eval_groups),
        "train_group_ids": sorted(train_groups),
        "eval_group_ids": sorted(eval_groups),
        "n_train": int(train_labels.shape[0]),
        "n_eval": int(eval_labels.shape[0]),
        "train_positives": int(train_labels.sum()),
        "train_negatives": int((~train_labels).sum()),
        "eval_positives": int(eval_labels.sum()),
        "eval_negatives": int((~eval_labels).sum()),
    }
    if summary["train_positives"] == 0 or summary["train_negatives"] == 0:
        summary["status"] = "insufficient_train_classes"
        return summary
    if summary["eval_positives"] == 0 or summary["eval_negatives"] == 0:
        summary["status"] = "insufficient_eval_classes"
        return summary

    try:
        direction = mean_difference_direction(features[train_mask], train_labels)
    except ValueError as exc:
        summary["status"] = "probe_fit_failed"
        summary["reason"] = str(exc)
        return summary
    scores = cosine_scores(features[eval_mask], direction)
    summary.update(
        {
            "auroc": binary_auroc(eval_labels, scores),
            "positive_score_mean": float(scores[eval_labels].mean()),
            "negative_score_mean": float(scores[~eval_labels].mean()),
            "score_gap": float(scores[eval_labels].mean() - scores[~eval_labels].mean()),
        }
    )
    if bootstrap_resamples > 0:
        eval_groups_by_row = [
            group for group, is_eval in zip(groups, eval_mask, strict=True) if is_eval
        ]
        summary["group_bootstrap_auroc"] = group_bootstrap_auroc_ci(
            eval_labels,
            scores,
            eval_groups_by_row,
            seed=0 if split_seed is None else split_seed,
            n_resamples=bootstrap_resamples,
        )
    return summary


def repeated_grouped_holdout_probe_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    seed: int = 0,
    repeats: int = 5,
    holdout_fraction: float = 0.2,
    strata: Sequence[str | None] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Run several grouped hash holdouts and summarize successful folds."""

    if repeats <= 0:
        return {"status": "disabled", "repeats": repeats}
    if split_manifest is None:
        resolved_manifest = build_repeated_grouped_split_manifest(
            groups,
            strata=strata,
            seed=seed,
            repeats=repeats,
            eval_fraction=holdout_fraction,
        )
    else:
        resolved_manifest = dict(split_manifest)
        validate_repeated_grouped_split_manifest(
            resolved_manifest,
            expected_groups=sorted(set(groups)),
        )
    folds = resolved_manifest.get("folds")
    if not isinstance(folds, list):
        raise ValueError("split manifest folds must be a list")
    fold_summaries: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ValueError("split manifest fold must be a mapping")
        fold_summaries.append(
            fixed_group_holdout_probe_summary(
                features,
                labels,
                groups,
                train_group_ids=cast(list[str], fold["train_group_ids"]),
                eval_group_ids=cast(list[str], fold["eval_group_ids"]),
                split_seed=int(fold["seed"]),
                split_hash=str(fold["fold_hash"]),
                bootstrap_resamples=bootstrap_resamples,
            )
        )
    ok_aurocs = [
        float(summary["auroc"])
        for summary in fold_summaries
        if summary.get("status") == "ok" and summary.get("auroc") is not None
    ]
    payload: dict[str, Any] = {
        "status": "ok" if ok_aurocs else "no_ok_folds",
        "repeats": len(fold_summaries),
        "num_ok_folds": len(ok_aurocs),
        "split_manifest_hash": resolved_manifest.get("manifest_hash"),
        "split_manifest": resolved_manifest,
        "folds": fold_summaries,
    }
    if ok_aurocs:
        payload.update(
            {
                "mean_auroc": float(np.mean(ok_aurocs)),
                "min_auroc": float(np.min(ok_aurocs)),
                "max_auroc": float(np.max(ok_aurocs)),
            }
        )
    return payload


def shared_split_probe_summaries(
    features_by_name: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: list[str],
    *,
    split_manifest: Mapping[str, Any],
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    """Evaluate every feature family on the exact same saved group splits."""

    if not features_by_name:
        raise ValueError("at least one feature family is required")
    validate_repeated_grouped_split_manifest(
        split_manifest,
        expected_groups=sorted(set(groups)),
    )
    summaries: dict[str, Any] = {}
    for name, features in features_by_name.items():
        summary = repeated_grouped_holdout_probe_summary(
            features,
            labels,
            groups,
            split_manifest=split_manifest,
            repeats=len(cast(list[Any], split_manifest["folds"])),
            bootstrap_resamples=bootstrap_resamples,
        )
        summary.pop("split_manifest", None)
        summaries[name] = summary
    return {
        "split_manifest_hash": split_manifest.get("manifest_hash"),
        "split_manifest": dict(split_manifest),
        "features": summaries,
    }


def group_bootstrap_auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute an AUROC confidence interval by resampling whole groups."""

    return group_bootstrap_metric_ci(
        labels,
        scores,
        groups,
        metric=binary_auroc,
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
    )


def metadata_probe_summary(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Probe labels from capture metadata only, as a leakage/artifact baseline."""

    features, feature_names = manifest_metadata_features(rows)
    summary = mean_difference_probe_summary(features, labels)
    summary["probe"] = "metadata_mean_difference"
    summary["feature_names"] = feature_names
    return summary


def manifest_metadata_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    numeric_fields: Sequence[str] = SAFE_METADATA_NUMERIC_FIELDS,
    categorical_fields: Sequence[str] = SAFE_METADATA_CATEGORICAL_FIELDS,
) -> tuple[np.ndarray, list[str]]:
    """Build an allow-listed metadata baseline with an explicit leakage guard."""

    if not rows:
        raise ValueError("metadata features require at least one row")
    assert_no_direct_label_leakage_fields([*numeric_fields, *categorical_fields])
    numeric_names = list(numeric_fields)
    category_names = list(categorical_fields)
    categories: dict[str, list[str]] = {
        name: sorted({_category_value(row.get(name)) for row in rows}) for name in category_names
    }
    feature_names = [*numeric_names]
    for name in category_names:
        feature_names.extend(f"{name}={value}" for value in categories[name])

    matrix: list[list[float]] = []
    for row in rows:
        values = [_metadata_numeric_value(name, row) for name in numeric_names]
        for name in category_names:
            row_value = _category_value(row.get(name))
            values.extend(float(row_value == category) for category in categories[name])
        matrix.append(values)
    features = np.asarray(matrix, dtype=np.float32)
    return _standardize_columns(features), feature_names


def assert_no_direct_label_leakage_fields(fields: Sequence[str]) -> None:
    """Reject fields whose values or presence directly reveal clean/corrupt labels."""

    leaked: list[str] = []
    for field in fields:
        base_field = str(field).split("=", 1)[0]
        if base_field in DIRECT_LABEL_LEAKAGE_FIELDS or base_field.startswith("matched_"):
            leaked.append(str(field))
    if leaked:
        raise ValueError(f"direct label leakage fields are forbidden: {sorted(set(leaked))}")


def _metadata_numeric_value(name: str, row: Mapping[str, Any]) -> float:
    num_positions = _float_or_zero(row.get("num_positions_in_file"))
    token_index = _float_or_zero(row.get("token_index"))
    if name == "relative_capture_position":
        return 0.0 if num_positions <= 1 else token_index / max(num_positions - 1.0, 1.0)
    if name == "is_last_captured_boundary":
        return float(num_positions > 0 and token_index == num_positions - 1.0)
    return _float_or_zero(row.get(name))


def activation_norm_probe_summary(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Baseline using only activation vector norm."""

    labels = np.asarray(labels, dtype=bool)
    payload: dict[str, Any] = {
        "probe": "activation_norm",
        "n": int(labels.shape[0]),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
    }
    if labels.sum() == 0 or (~labels).sum() == 0:
        payload["status"] = "insufficient_classes"
        return payload
    norms = np.linalg.norm(features, axis=1)
    auroc = binary_auroc(labels, norms)
    payload.update(
        {
            "status": "ok",
            "auroc": auroc,
            "best_orientation_auroc": max(auroc, 1.0 - auroc),
            "positive_norm_mean": float(norms[labels].mean()),
            "negative_norm_mean": float(norms[~labels].mean()),
        }
    )
    return payload


def top_pc_probe_summary(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Baseline using only the first principal component of activations."""

    labels = np.asarray(labels, dtype=bool)
    payload: dict[str, Any] = {
        "probe": "top_pc",
        "n": int(labels.shape[0]),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
    }
    if features.ndim != 2:
        payload["status"] = "invalid_feature_shape"
        return payload
    if features.shape[0] < 2:
        payload["status"] = "insufficient_rows"
        return payload
    if labels.sum() == 0 or (~labels).sum() == 0:
        payload["status"] = "insufficient_classes"
        return payload
    centered = features - features.mean(axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        payload["status"] = "svd_failed"
        return payload
    scores = centered @ vh[0]
    auroc = binary_auroc(labels, scores)
    payload.update(
        {
            "status": "ok",
            "auroc": auroc,
            "best_orientation_auroc": max(auroc, 1.0 - auroc),
        }
    )
    return payload


def label_permutation_probe_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    seed: int = 0,
    permutations: int = 20,
) -> dict[str, Any]:
    """Compare in-sample AUROC against pair/group-preserving label permutations."""

    labels = np.asarray(labels, dtype=bool)
    if permutations <= 0:
        return {"status": "disabled", "permutations": permutations}
    observed = mean_difference_probe_summary(features, labels)
    if observed.get("status") != "ok" or observed.get("auroc") is None:
        return {
            "status": "observed_not_ok",
            "permutations": permutations,
            "observed": observed,
        }
    rng = np.random.default_rng(seed)
    permuted_aurocs: list[float] = []
    for _ in range(permutations):
        permuted = permute_labels_by_group(labels, groups, rng=rng)
        if permuted.sum() == 0 or (~permuted).sum() == 0:
            continue
        try:
            summary = mean_difference_probe_summary(features, permuted)
        except ValueError:
            continue
        if summary.get("status") == "ok" and summary.get("auroc") is not None:
            permuted_aurocs.append(float(summary["auroc"]))
    observed_auc = float(observed["auroc"])
    payload: dict[str, Any] = {
        "status": "ok" if permuted_aurocs else "no_ok_permutations",
        "permutations": permutations,
        "num_ok_permutations": len(permuted_aurocs),
        "observed_auroc": observed_auc,
    }
    if permuted_aurocs:
        null = np.asarray(permuted_aurocs, dtype=float)
        payload.update(
            {
                "null_mean_auroc": float(null.mean()),
                "null_max_auroc": float(null.max()),
                "empirical_p_ge_observed": float(
                    (np.sum(null >= observed_auc) + 1) / (len(null) + 1)
                ),
            }
        )
    return payload


def grouped_split_label_permutation_probe_summary(
    features: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    split_manifest: Mapping[str, Any],
    seed: int = 0,
    permutations: int = 100,
) -> dict[str, Any]:
    """Permutation test for repeated grouped-holdout AUROC on fixed splits."""

    if permutations <= 0:
        return {"status": "disabled", "permutations": permutations}
    observed = repeated_grouped_holdout_probe_summary(
        features,
        labels,
        groups,
        split_manifest=split_manifest,
        repeats=len(cast(list[Any], split_manifest["folds"])),
    )
    observed_auc = observed.get("mean_auroc")
    if observed.get("status") != "ok" or observed_auc is None:
        return {
            "status": "observed_not_ok",
            "permutations": permutations,
            "observed": observed,
        }

    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels, dtype=bool)
    null_aurocs: list[float] = []
    for _ in range(permutations):
        permuted = permute_labels_by_group(labels_array, groups, rng=rng)
        summary = repeated_grouped_holdout_probe_summary(
            features,
            permuted,
            groups,
            split_manifest=split_manifest,
            repeats=len(cast(list[Any], split_manifest["folds"])),
        )
        if summary.get("status") == "ok" and summary.get("mean_auroc") is not None:
            null_aurocs.append(float(summary["mean_auroc"]))

    observed_value = float(observed_auc)
    payload: dict[str, Any] = {
        "status": "ok" if null_aurocs else "no_ok_permutations",
        "permutations": permutations,
        "num_ok_permutations": len(null_aurocs),
        "observed_mean_auroc": observed_value,
        "split_manifest_hash": split_manifest.get("manifest_hash"),
        "permutation_unit": "paired_within_group_or_homogeneous_group_block",
    }
    if null_aurocs:
        null = np.asarray(null_aurocs, dtype=float)
        payload.update(
            {
                "null_mean_auroc": float(null.mean()),
                "null_max_auroc": float(null.max()),
                "empirical_p_ge_observed": float(
                    (np.sum(null >= observed_value) + 1) / (len(null) + 1)
                ),
            }
        )
    return payload


def matched_negative_indices(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    *,
    max_negatives_per_positive: int = 1,
    seed: int = 0,
) -> list[int]:
    """Select positive rows plus metadata-nearest negative rows from other groups."""

    labels = np.asarray(labels, dtype=bool)
    if max_negatives_per_positive <= 0:
        return []
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    selected: set[int] = set(positives)
    if not positives or not negatives:
        return sorted(selected)
    for positive_index in positives:
        positive = rows[positive_index]
        candidates = [
            negative_index
            for negative_index in negatives
            if str(rows[negative_index].get("group_id") or rows[negative_index].get("sample_id"))
            != str(positive.get("group_id") or positive.get("sample_id"))
        ]
        if not candidates:
            candidates = negatives
        ranked = sorted(
            candidates,
            key=lambda index: (
                _match_distance(positive, rows[index]),
                stable_hash({"seed": seed, "positive": positive_index, "negative": index}),
            ),
        )
        selected.update(ranked[:max_negatives_per_positive])
    return sorted(selected)


def _match_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    distance = 0.0
    for key, scale in (
        ("token_position", 256.0),
        ("boundary_step", 4.0),
        ("token_index", 4.0),
        ("num_positions_in_file", 4.0),
    ):
        distance += abs(_float_or_zero(left.get(key)) - _float_or_zero(right.get(key))) / scale
    for key in ("subset", "generator", "prompt_mode"):
        distance += 10.0 * float(_category_value(left.get(key)) != _category_value(right.get(key)))
    return distance


def _standardize_columns(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return cast(np.ndarray, (features - mean) / np.maximum(std, 1e-6))


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _category_value(value: Any) -> str:
    return "unknown" if value is None else str(value)


def probe_metrics_markdown(metrics: Mapping[str, Any], *, max_rows: int = 40) -> str:
    """Render a compact Markdown table of probe site summaries."""

    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping) or not sites:
        return "No probe metrics found.\n"
    rows = sorted(
        sites.items(),
        key=lambda item: (
            float(item[1].get("auroc", -1.0)) if isinstance(item[1], Mapping) else -1.0,
            abs(float(item[1].get("score_gap", 0.0))) if isinstance(item[1], Mapping) else 0.0,
        ),
        reverse=True,
    )
    lines = [
        (
            "| site | status | n | positives | negatives | AUROC | metadata AUROC | "
            "norm best | PC best | repeated AUROC | holdout AUROC | matched AUROC | matched metadata |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for site, value in rows[:max_rows]:
        if not isinstance(value, Mapping):
            continue
        holdout = value.get("holdout")
        holdout_auroc = ""
        if isinstance(holdout, Mapping):
            holdout_auroc = _format_optional_float(holdout.get("auroc"))
        metadata = value.get("metadata_baseline")
        norm = value.get("activation_norm_baseline")
        pc = value.get("top_pc_baseline")
        repeated = value.get("repeated_holdout")
        matched = value.get("matched_negative_subset")
        matched_metadata = (
            matched.get("metadata_baseline") if isinstance(matched, Mapping) else None
        )
        lines.append(
            "| {site} | {status} | {n} | {pos} | {neg} | {auc} | {metadata_auc} | {norm_auc} | {pc_auc} | {repeated_auc} | {holdout_auc} | {matched_auc} | {matched_metadata_auc} |".format(
                site=site,
                status=value.get("status", "unknown"),
                n=int(value.get("n", 0)),
                pos=int(value.get("positives", 0)),
                neg=int(value.get("negatives", 0)),
                auc=_format_optional_float(value.get("auroc")),
                metadata_auc=_mapping_float(metadata, "auroc"),
                norm_auc=_mapping_float(norm, "best_orientation_auroc"),
                pc_auc=_mapping_float(pc, "best_orientation_auroc"),
                repeated_auc=_mapping_float(repeated, "mean_auroc"),
                holdout_auc=holdout_auroc,
                matched_auc=_mapping_float(matched, "auroc"),
                matched_metadata_auc=_mapping_float(matched_metadata, "auroc"),
            )
        )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _mapping_float(value: Any, key: str) -> str:
    if not isinstance(value, Mapping):
        return ""
    return _format_optional_float(value.get(key))
