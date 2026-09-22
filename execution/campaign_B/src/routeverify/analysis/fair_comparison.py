"""Matched-estimator comparisons on shared grouped evaluation folds."""

from __future__ import annotations

import importlib.metadata
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from routeverify.analysis.probes import binary_auroc
from routeverify.analysis.protocols import group_bootstrap_metric_ci
from routeverify.analysis.surface_baselines import validate_grouped_kfold_manifest


def evaluate_dense_logistic_grouped_cv(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_ids: Sequence[str],
    pair_group_ids: Sequence[str],
    analysis_group_ids: Sequence[str],
    split_manifest: dict[str, Any],
    analysis_group_definition: str,
    seed: int,
    n_splits: int,
    bootstrap_resamples: int,
    logistic_c: float = 1.0,
    class_weight: str | None = "balanced",
    solver: str = "liblinear",
    max_iter: int = 2_000,
    standardize: bool = True,
) -> dict[str, Any]:
    """Evaluate a dense L2-logistic probe with source-safe OOF predictions."""

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    row_count = target.shape[0]
    if matrix.ndim != 2:
        raise ValueError("features must be [rows, dimensions]")
    if matrix.shape[0] != row_count:
        raise ValueError("features and labels must have the same row count")
    if not (len(sample_ids) == len(pair_group_ids) == len(analysis_group_ids) == row_count):
        raise ValueError("row identifiers and labels must have the same row count")
    if len(set(sample_ids)) != row_count:
        raise ValueError("sample_ids must be unique")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    if logistic_c <= 0:
        raise ValueError("logistic_c must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not analysis_group_definition:
        raise ValueError("analysis_group_definition must be non-empty")

    analysis_groups = tuple(str(value) for value in analysis_group_ids)
    validate_grouped_kfold_manifest(
        split_manifest,
        expected_groups=set(analysis_groups),
        expected_seed=seed,
        expected_n_splits=n_splits,
    )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "Install analysis dependencies with `uv sync --extra analysis`."
        ) from exc

    oof_scores = np.full(row_count, np.nan, dtype=float)
    oof_folds = np.full(row_count, -1, dtype=int)
    folds: list[dict[str, Any]] = []
    for fold in split_manifest["folds"]:
        train_groups = set(str(value) for value in fold["train_group_ids"])
        eval_groups = set(str(value) for value in fold["eval_group_ids"])
        train_mask = np.asarray([group in train_groups for group in analysis_groups], dtype=bool)
        eval_mask = np.asarray([group in eval_groups for group in analysis_groups], dtype=bool)
        train_labels = target[train_mask]
        eval_labels = target[eval_mask]
        if np.unique(train_labels).size != 2 or np.unique(eval_labels).size != 2:
            raise ValueError("every dense-probe fold must contain both classes")

        train_features = matrix[train_mask]
        eval_features = matrix[eval_mask]
        if standardize:
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            eval_features = scaler.transform(eval_features)
        classifier = LogisticRegression(
            C=logistic_c,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=seed,
        )
        classifier.fit(train_features, train_labels.astype(int))
        scores = np.asarray(classifier.decision_function(eval_features), dtype=float)
        fold_index = int(fold["fold_index"])
        oof_scores[eval_mask] = scores
        oof_folds[eval_mask] = fold_index
        folds.append(
            {
                **fold,
                "n_train_rows": int(train_mask.sum()),
                "n_eval_rows": int(eval_mask.sum()),
                "n_train_pairs": len(set(np.asarray(pair_group_ids)[train_mask].tolist())),
                "n_eval_pairs": len(set(np.asarray(pair_group_ids)[eval_mask].tolist())),
                "n_train_analysis_groups": len(train_groups),
                "n_eval_analysis_groups": len(eval_groups),
                "eval_auroc": binary_auroc(eval_labels, scores),
                "eval_positive_count": int(eval_labels.sum()),
                "eval_negative_count": int((~eval_labels).sum()),
            }
        )
    if np.isnan(oof_scores).any() or (oof_folds < 0).any():
        raise AssertionError("dense grouped CV did not score every row exactly once")

    uncertainty = group_bootstrap_metric_ci(
        target,
        oof_scores,
        analysis_groups,
        metric=binary_auroc,
        seed=seed,
        n_resamples=bootstrap_resamples,
    )
    predictions = [
        {
            "sample_id": str(sample_id),
            "group_id": str(pair_group_id),
            "analysis_group_id": str(analysis_group_id),
            "label": "corrupt" if label else "clean",
            "score": float(score),
            "fold_index": int(fold_index),
        }
        for sample_id, pair_group_id, analysis_group_id, label, score, fold_index in zip(
            sample_ids,
            pair_group_ids,
            analysis_groups,
            target,
            oof_scores,
            oof_folds,
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "estimator": {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "C": logistic_c,
            "class_weight": class_weight,
            "solver": solver,
            "max_iter": max_iter,
            "random_state": seed,
            "score": "decision_function_log_odds",
        },
        "preprocessing": {
            "standardize": standardize,
            "implementation": ("sklearn.preprocessing.StandardScaler" if standardize else None),
            "fit_scope": "training_fold_only",
        },
        "runtime_versions": {
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "feature_dim": int(matrix.shape[1]),
        "n_rows": row_count,
        "n_pairs": len(set(pair_group_ids)),
        "analysis_group_definition": analysis_group_definition,
        "n_analysis_groups": len(set(analysis_groups)),
        "split_manifest_hash": split_manifest["manifest_hash"],
        "fold_aurocs": [float(fold["eval_auroc"]) for fold in folds],
        "mean_fold_auroc": float(np.mean([fold["eval_auroc"] for fold in folds])),
        "micro_oof_auroc": binary_auroc(target, oof_scores),
        "group_bootstrap_ci": uncertainty,
        "folds": folds,
        "oof_predictions": predictions,
    }


def paired_group_bootstrap_auroc_difference(
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    groups: Sequence[str],
    *,
    left_name: str,
    right_name: str,
    seed: int,
    n_resamples: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap a paired AUROC difference by resampling whole source groups."""

    target = np.asarray(labels, dtype=bool)
    left = np.asarray(left_scores, dtype=float)
    right = np.asarray(right_scores, dtype=float)
    if target.ndim != 1 or left.ndim != 1 or right.ndim != 1:
        raise ValueError("labels and score arrays must be one-dimensional")
    if not (target.shape == left.shape == right.shape) or len(groups) != target.shape[0]:
        raise ValueError("labels, scores, and groups must have the same row count")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("scores must be finite")
    if not left_name or not right_name:
        raise ValueError("comparison names must be non-empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(groups):
        by_group[str(group_id)].append(index)
    unique_groups = sorted(by_group)
    if not unique_groups:
        raise ValueError("at least one group is required")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate(
            [np.asarray(by_group[str(group_id)], dtype=np.int64) for group_id in sampled_groups]
        )
        try:
            values.append(
                binary_auroc(target[indices], left[indices])
                - binary_auroc(target[indices], right[indices])
            )
        except ValueError:
            continue

    left_auc = binary_auroc(target, left)
    right_auc = binary_auroc(target, right)
    payload: dict[str, Any] = {
        "status": "ok" if values else "no_valid_resamples",
        "left": {"name": left_name, "auroc": left_auc},
        "right": {"name": right_name, "auroc": right_auc},
        "difference_definition": f"{left_name}_minus_{right_name}",
        "difference": left_auc - right_auc,
        "seed": seed,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_valid_resamples": len(values),
        "num_groups": len(unique_groups),
        "resampling_unit": "analysis_group",
    }
    if values:
        low, high = np.quantile(
            np.asarray(values, dtype=float),
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        payload["ci_low"] = float(low)
        payload["ci_high"] = float(high)
    return payload
