"""Leakage-resistant TF-IDF baselines for paired PRMBench traces."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

from routeverify.analysis.probes import binary_auroc
from routeverify.analysis.protocols import (
    group_bootstrap_metric_ci,
    validate_discovery_validation_manifest,
)
from routeverify.data.prmbench import PRMBenchPair
from routeverify.data.schema import stable_hash

SurfaceTextField = Literal["boundary_step", "prefix_through_boundary", "full_trace"]
SurfaceAnalyzer = Literal["word", "char"]


@dataclass(frozen=True)
class TfidfBaselineSpec:
    """Fully specified TF-IDF and logistic-regression baseline."""

    name: str
    text_field: SurfaceTextField
    analyzer: SurfaceAnalyzer
    ngram_min: int
    ngram_max: int
    lowercase: bool = True
    min_df: int = 2
    max_features: int | None = 50_000
    sublinear_tf: bool = True
    logistic_c: float = 1.0
    class_weight: Literal["balanced"] | None = "balanced"
    solver: Literal["liblinear"] = "liblinear"
    max_iter: int = 2_000

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("baseline name is required")
        if self.ngram_min <= 0 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid n-gram range")
        if self.min_df <= 0:
            raise ValueError("min_df must be positive")
        if self.max_features is not None and self.max_features <= 0:
            raise ValueError("max_features must be positive")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ngram_range"] = [self.ngram_min, self.ngram_max]
        payload["tokenization"] = {
            "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
            "analyzer": self.analyzer,
            "lowercase": self.lowercase,
            "input": "content",
            "encoding": "utf-8",
            "decode_error": "strict",
            "strip_accents": None,
            "preprocessor": None,
            "tokenizer": None,
            "stop_words": None,
            "token_pattern": r"(?u)\b\w\w+\b",
            "binary": False,
            "max_df": 1.0,
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "dtype": "float64",
        }
        payload["classifier"] = {
            "implementation": "sklearn.linear_model.LogisticRegression",
            "positive_class": "corrupt",
            "score": "decision_function_log_odds",
            "penalty": "l2",
            "dual": False,
            "tol": 0.0001,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "n_jobs": None,
            "verbose": 0,
            "warm_start": False,
        }
        return payload


@dataclass(frozen=True)
class SurfaceDataset:
    texts: tuple[str, ...]
    labels: np.ndarray
    group_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    error_types: tuple[str | None, ...]
    boundary_steps: tuple[int, ...]
    boundary_definition: str


DEFAULT_TFIDF_SPECS = (
    TfidfBaselineSpec(
        name="boundary_word",
        text_field="boundary_step",
        analyzer="word",
        ngram_min=1,
        ngram_max=2,
    ),
    TfidfBaselineSpec(
        name="boundary_char",
        text_field="boundary_step",
        analyzer="char",
        ngram_min=3,
        ngram_max=5,
    ),
    TfidfBaselineSpec(
        name="prefix_char",
        text_field="prefix_through_boundary",
        analyzer="char",
        ngram_min=3,
        ngram_max=5,
    ),
    TfidfBaselineSpec(
        name="full_char",
        text_field="full_trace",
        analyzer="char",
        ngram_min=3,
        ngram_max=5,
    ),
)


def surface_text_for_pair_member(
    pair: PRMBenchPair,
    *,
    member: Literal["clean", "corrupt"],
    text_field: SurfaceTextField,
    boundary_step: int | None = None,
) -> str:
    """Render a pair member at a supplied or pair-corrupt boundary."""

    boundary = pair.corrupt.first_error if boundary_step is None else int(boundary_step)
    if boundary is None:
        raise ValueError("PRMBench corrupt member must have first_error")
    example = pair.clean if member == "clean" else pair.corrupt
    if boundary < 0 or boundary >= len(example.steps):
        raise ValueError("surface boundary is outside the selected trace")

    if text_field == "boundary_step":
        return example.steps[boundary].strip()
    numbered_steps = "\n".join(
        f"{index + 1}. {step.strip()}"
        for index, step in enumerate(
            example.steps if text_field == "full_trace" else example.steps[: boundary + 1]
        )
    )
    if text_field == "prefix_through_boundary":
        return numbered_steps
    if text_field == "full_trace":
        return f"Problem:\n{example.problem.strip()}\n\nSolution:\n{numbered_steps}"
    raise ValueError(f"unsupported surface text field: {text_field}")


def build_surface_dataset(
    pairs: list[PRMBenchPair],
    *,
    text_field: SurfaceTextField,
    boundary_step_by_sample_id: Mapping[str, int] | None = None,
) -> SurfaceDataset:
    """Build two rows per complete pair with an explicit corrupt-positive label."""

    expected_sample_ids = {
        example.sample_id for pair in pairs for example in (pair.clean, pair.corrupt)
    }
    if boundary_step_by_sample_id is not None:
        observed_sample_ids = set(boundary_step_by_sample_id)
        missing = sorted(expected_sample_ids - observed_sample_ids)
        extra = sorted(observed_sample_ids - expected_sample_ids)
        if missing or extra:
            raise ValueError(
                "explicit surface boundaries must exactly cover selected samples: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    texts: list[str] = []
    labels: list[bool] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    error_types: list[str | None] = []
    boundary_steps: list[int] = []
    seen_groups: set[str] = set()
    for pair in pairs:
        group_id = pair.clean.group_id
        if group_id != pair.corrupt.group_id:
            raise ValueError("clean/corrupt group IDs differ")
        if group_id in seen_groups:
            raise ValueError(f"duplicate PRMBench pair group: {group_id}")
        seen_groups.add(group_id)
        for member, label in (("clean", False), ("corrupt", True)):
            example = pair.clean if member == "clean" else pair.corrupt
            boundary = (
                pair.corrupt.first_error
                if boundary_step_by_sample_id is None
                else int(boundary_step_by_sample_id[example.sample_id])
            )
            if boundary is None:
                raise ValueError("PRMBench corrupt member must have first_error")
            texts.append(
                surface_text_for_pair_member(
                    pair,
                    member=cast(Literal["clean", "corrupt"], member),
                    text_field=text_field,
                    boundary_step=boundary,
                )
            )
            labels.append(label)
            groups.append(group_id)
            sample_ids.append(example.sample_id)
            error_types.append(pair.corrupt.error_type)
            boundary_steps.append(boundary)
    return SurfaceDataset(
        texts=tuple(texts),
        labels=np.asarray(labels, dtype=bool),
        group_ids=tuple(groups),
        sample_ids=tuple(sample_ids),
        error_types=tuple(error_types),
        boundary_steps=tuple(boundary_steps),
        boundary_definition=(
            "paired_corrupt_first_error"
            if boundary_step_by_sample_id is None
            else "explicit_sample_boundary_step"
        ),
    )


def surface_dataset_manifest(dataset: SurfaceDataset) -> dict[str, Any]:
    """Hash exact rendered inputs and labels without persisting full source text."""

    rows = [
        {
            "sample_id": sample_id,
            "group_id": group_id,
            "label": "corrupt" if label else "clean",
            "error_type": error_type,
            "boundary_step": boundary_step,
            "rendered_text_hash": stable_hash(text, length=64),
        }
        for text, label, group_id, sample_id, error_type, boundary_step in zip(
            dataset.texts,
            dataset.labels,
            dataset.group_ids,
            dataset.sample_ids,
            dataset.error_types,
            dataset.boundary_steps,
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "boundary_definition": dataset.boundary_definition,
        "n_rows": len(rows),
        "n_groups": len(set(dataset.group_ids)),
        "rows_hash": stable_hash(rows, length=64),
        "rows": rows,
    }


def surface_runtime_versions() -> dict[str, str]:
    """Record versions that can affect tokenization, folds, or optimization."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
    }


def build_grouped_kfold_manifest(
    group_ids: list[str] | tuple[str, ...],
    *,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    """Create deterministic balanced folds by shuffling unique groups once."""

    unique_groups = np.asarray(sorted(set(group_ids)), dtype=object)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if n_splits > unique_groups.shape[0]:
        raise ValueError("n_splits exceeds the number of groups")
    # RandomState deliberately matches sklearn GroupKFold(shuffle=True,
    # random_state=seed), including its historical MT19937 ordering.
    shuffled = np.random.RandomState(seed).permutation(unique_groups)
    fold_groups = np.array_split(shuffled, n_splits)
    all_groups = set(str(group_id) for group_id in unique_groups)
    folds: list[dict[str, Any]] = []
    for fold_index, values in enumerate(fold_groups):
        eval_groups = sorted(str(group_id) for group_id in values)
        train_groups = sorted(all_groups - set(eval_groups))
        if set(train_groups) & set(eval_groups):
            raise AssertionError("group leakage in generated TF-IDF split")
        fold: dict[str, Any] = {
            "fold_index": fold_index,
            "seed": seed,
            "train_group_ids": train_groups,
            "eval_group_ids": eval_groups,
            "train_groups_hash": stable_hash(train_groups, length=64),
            "eval_groups_hash": stable_hash(eval_groups, length=64),
        }
        fold["fold_hash"] = stable_hash(fold, length=64)
        folds.append(fold)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "paired_grouped_kfold",
        "implementation": "unique_group_shuffle_then_numpy_array_split",
        "random_number_generator": "numpy.random.RandomState(MT19937)",
        "seed": seed,
        "n_splits": n_splits,
        "num_groups": len(all_groups),
        "group_ids_hash": stable_hash(sorted(all_groups), length=64),
        "folds": folds,
    }
    manifest["manifest_hash"] = stable_hash(manifest, length=64)
    return manifest


def evaluate_tfidf_grouped_cv(
    pairs: list[PRMBenchPair],
    spec: TfidfBaselineSpec,
    *,
    seed: int,
    n_splits: int = 5,
    bootstrap_resamples: int = 1_000,
    split_manifest: dict[str, Any] | None = None,
    analysis_group_by_pair_group: Mapping[str, str] | None = None,
    analysis_group_definition: str = "clean_corrupt_pair",
    boundary_step_by_sample_id: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Fit TF-IDF within every training fold and return OOF scores and group CI.

    ``analysis_group_by_pair_group`` can lift the split and bootstrap unit above
    an individual clean/corrupt pair, for example to a source problem shared by
    several synthetic error variants.  Text rendering and pair identities stay
    unchanged; only the train/eval and uncertainty grouping changes.
    """

    dataset = build_surface_dataset(
        pairs,
        text_field=spec.text_field,
        boundary_step_by_sample_id=boundary_step_by_sample_id,
    )
    pair_groups = set(dataset.group_ids)
    if analysis_group_by_pair_group is None:
        analysis_groups = dataset.group_ids
    else:
        missing = sorted(pair_groups - set(analysis_group_by_pair_group))
        if missing:
            raise ValueError(
                f"analysis-group mapping does not cover every surface pair: {missing[:5]}"
            )
        analysis_groups = tuple(
            str(analysis_group_by_pair_group[group_id]) for group_id in dataset.group_ids
        )
    if not analysis_group_definition:
        raise ValueError("analysis_group_definition must be non-empty")
    manifest = split_manifest or build_grouped_kfold_manifest(
        analysis_groups,
        seed=seed,
        n_splits=n_splits,
    )
    validate_grouped_kfold_manifest(
        manifest,
        expected_groups=set(analysis_groups),
        expected_seed=seed,
        expected_n_splits=n_splits,
    )
    oof_scores = np.full(dataset.labels.shape[0], np.nan, dtype=float)
    oof_folds = np.full(dataset.labels.shape[0], -1, dtype=int)
    folds: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        train_groups = set(str(value) for value in fold["train_group_ids"])
        eval_groups = set(str(value) for value in fold["eval_group_ids"])
        train_mask = np.asarray([group in train_groups for group in analysis_groups], dtype=bool)
        eval_mask = np.asarray([group in eval_groups for group in analysis_groups], dtype=bool)
        scores, fit_report = _fit_tfidf_scores(
            [text for text, include in zip(dataset.texts, train_mask, strict=True) if include],
            dataset.labels[train_mask],
            [text for text, include in zip(dataset.texts, eval_mask, strict=True) if include],
            spec=spec,
            seed=seed,
        )
        fold_index = int(fold["fold_index"])
        oof_scores[eval_mask] = scores
        oof_folds[eval_mask] = fold_index
        fold_labels = dataset.labels[eval_mask]
        folds.append(
            {
                **fold,
                **fit_report,
                "n_train_rows": int(train_mask.sum()),
                "n_eval_rows": int(eval_mask.sum()),
                "n_train_pairs": int(train_mask.sum()) // 2,
                "n_eval_pairs": int(eval_mask.sum()) // 2,
                "n_train_analysis_groups": len(train_groups),
                "n_eval_analysis_groups": len(eval_groups),
                "eval_auroc": binary_auroc(fold_labels, scores),
                "eval_positive_count": int(fold_labels.sum()),
                "eval_negative_count": int((~fold_labels).sum()),
            }
        )
    if np.isnan(oof_scores).any() or (oof_folds < 0).any():
        raise AssertionError("TF-IDF grouped CV did not score every row exactly once")
    overall_auc = binary_auroc(dataset.labels, oof_scores)
    uncertainty = group_bootstrap_metric_ci(
        dataset.labels,
        oof_scores,
        analysis_groups,
        metric=binary_auroc,
        seed=seed,
        n_resamples=bootstrap_resamples,
    )
    predictions = [
        {
            "sample_id": sample_id,
            "group_id": group_id,
            "analysis_group_id": analysis_group_id,
            "label": "corrupt" if label else "clean",
            "score": float(score),
            "fold_index": int(fold_index),
            "error_type": error_type,
            "boundary_step": boundary_step,
        }
        for (
            sample_id,
            group_id,
            analysis_group_id,
            label,
            score,
            fold_index,
            error_type,
            boundary_step,
        ) in zip(
            dataset.sample_ids,
            dataset.group_ids,
            analysis_groups,
            dataset.labels,
            oof_scores,
            oof_folds,
            dataset.error_types,
            dataset.boundary_steps,
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "baseline": spec.to_json(),
        "seed": seed,
        "runtime_versions": surface_runtime_versions(),
        "surface_input_scope": (
            "pair-aligned textual sensitivity baseline using each corrupt pair member's "
            "first-error step"
            if boundary_step_by_sample_id is None
            else "exact sample-level boundary_step from the matched activation manifest"
        ),
        "dataset_manifest": surface_dataset_manifest(dataset),
        "split_manifest": manifest,
        "analysis_group_definition": analysis_group_definition,
        "n_analysis_groups": len(set(analysis_groups)),
        "n_rows": len(dataset.texts),
        "n_pairs": len(set(dataset.group_ids)),
        "fold_aurocs": [float(fold["eval_auroc"]) for fold in folds],
        "mean_fold_auroc": float(np.mean([fold["eval_auroc"] for fold in folds])),
        "micro_oof_auroc": overall_auc,
        "group_bootstrap_ci": uncertainty,
        "folds": folds,
        "oof_predictions": predictions,
    }


def evaluate_surface_discovery_validation(
    pairs: list[PRMBenchPair],
    specs: list[TfidfBaselineSpec] | tuple[TfidfBaselineSpec, ...],
    discovery_validation_manifest: dict[str, Any],
    *,
    seed: int,
    discovery_folds: int = 5,
    bootstrap_resamples: int = 1_000,
) -> dict[str, Any]:
    """Select a surface specification on discovery and score validation once."""

    if not specs:
        raise ValueError("at least one TF-IDF specification is required")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("TF-IDF candidate names must be unique")
    all_groups = [pair.clean.group_id for pair in pairs]
    validate_discovery_validation_manifest(
        discovery_validation_manifest,
        expected_groups=all_groups,
    )
    discovery_groups = set(discovery_validation_manifest["discovery_group_ids"])
    validation_groups = set(discovery_validation_manifest["validation_group_ids"])
    discovery_pairs = [pair for pair in pairs if pair.clean.group_id in discovery_groups]
    if len(discovery_pairs) < discovery_folds:
        raise ValueError("not enough discovery pairs for requested cross-validation folds")

    discovery_split = build_grouped_kfold_manifest(
        [pair.clean.group_id for pair in discovery_pairs],
        seed=seed,
        n_splits=discovery_folds,
    )
    discovery_results = [
        evaluate_tfidf_grouped_cv(
            discovery_pairs,
            spec,
            seed=seed,
            n_splits=discovery_folds,
            bootstrap_resamples=bootstrap_resamples,
            split_manifest=discovery_split,
        )
        for spec in specs
    ]
    # The name is a deterministic tie-breaker; validation scores are never consulted.
    selected_result = sorted(
        discovery_results,
        key=lambda result: (
            -float(result["mean_fold_auroc"]),
            str(result["baseline"]["name"]),
        ),
    )[0]
    selected_name = str(selected_result["baseline"]["name"])
    selected_spec = next(spec for spec in specs if spec.name == selected_name)
    selected_dataset = build_surface_dataset(pairs, text_field=selected_spec.text_field)
    train_mask = np.asarray(
        [group_id in discovery_groups for group_id in selected_dataset.group_ids],
        dtype=bool,
    )
    eval_mask = np.asarray(
        [group_id in validation_groups for group_id in selected_dataset.group_ids],
        dtype=bool,
    )
    validation_scores, fit_report = _fit_tfidf_scores(
        [text for text, include in zip(selected_dataset.texts, train_mask, strict=True) if include],
        selected_dataset.labels[train_mask],
        [text for text, include in zip(selected_dataset.texts, eval_mask, strict=True) if include],
        spec=selected_spec,
        seed=seed,
    )
    validation_labels = selected_dataset.labels[eval_mask]
    validation_group_rows = [
        group_id
        for group_id, include in zip(selected_dataset.group_ids, eval_mask, strict=True)
        if include
    ]
    validation_ci = group_bootstrap_metric_ci(
        validation_labels,
        validation_scores,
        validation_group_rows,
        metric=binary_auroc,
        seed=seed,
        n_resamples=bootstrap_resamples,
    )
    validation_predictions: list[dict[str, Any]] = []
    eval_indices = np.flatnonzero(eval_mask)
    for row_index, score in zip(eval_indices, validation_scores, strict=True):
        validation_predictions.append(
            {
                "sample_id": selected_dataset.sample_ids[int(row_index)],
                "group_id": selected_dataset.group_ids[int(row_index)],
                "label": "corrupt" if selected_dataset.labels[int(row_index)] else "clean",
                "score": float(score),
                "error_type": selected_dataset.error_types[int(row_index)],
            }
        )
    return {
        "schema_version": 1,
        "protocol": "discovery_select_then_single_frozen_validation",
        "seed": seed,
        "discovery_validation_manifest_hash": discovery_validation_manifest["manifest_hash"],
        "candidate_names": [spec.name for spec in specs],
        "selection_metric": "mean_discovery_grouped_cv_auroc",
        "selection_tie_breaker": "lexicographically_smaller_name",
        "selected_baseline": selected_name,
        "discovery_split_manifest": discovery_split,
        "discovery_results": discovery_results,
        "validation": {
            "status": "ok",
            "evaluation_count": 1,
            "baseline": selected_spec.to_json(),
            "train_partition": "discovery",
            "eval_partition": "validation",
            "n_train_pairs": len(discovery_groups),
            "n_eval_pairs": len(validation_groups),
            "n_train_rows": int(train_mask.sum()),
            "n_eval_rows": int(eval_mask.sum()),
            "auroc": binary_auroc(validation_labels, validation_scores),
            "group_bootstrap_ci": validation_ci,
            **fit_report,
            "predictions": validation_predictions,
        },
    }


def evaluate_heldout_error_types(
    pairs: list[PRMBenchPair],
    spec: TfidfBaselineSpec,
    *,
    seed: int,
    small_sample_pairs: int = 20,
    bootstrap_resamples: int = 1_000,
) -> dict[str, Any]:
    """Train on all other error types and pool one held-out score per row."""

    if small_sample_pairs <= 0:
        raise ValueError("small_sample_pairs must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    dataset = build_surface_dataset(pairs, text_field=spec.text_field)
    normalized_types = tuple(
        "unknown" if error_type is None or not error_type.strip() else error_type
        for error_type in dataset.error_types
    )
    unique_types = sorted(set(normalized_types))
    all_scores = np.full(dataset.labels.shape[0], np.nan, dtype=float)
    reports: list[dict[str, Any]] = []
    for error_type in unique_types:
        eval_mask = np.asarray([value == error_type for value in normalized_types], dtype=bool)
        train_mask = ~eval_mask
        n_eval_pairs = len(
            {
                group_id
                for group_id, include in zip(dataset.group_ids, eval_mask, strict=True)
                if include
            }
        )
        flags: list[str] = []
        if n_eval_pairs < small_sample_pairs:
            flags.append("small_sample")
        if error_type == "unknown":
            flags.append("unidentifiable_error_type")
        scores, fit_report = _fit_tfidf_scores(
            [text for text, include in zip(dataset.texts, train_mask, strict=True) if include],
            dataset.labels[train_mask],
            [text for text, include in zip(dataset.texts, eval_mask, strict=True) if include],
            spec=spec,
            seed=seed,
        )
        all_scores[eval_mask] = scores
        eval_labels = dataset.labels[eval_mask]
        eval_indices = np.flatnonzero(eval_mask)
        eval_group_rows = [dataset.group_ids[int(row_index)] for row_index in eval_indices]
        per_type_ci = group_bootstrap_metric_ci(
            eval_labels,
            scores,
            eval_group_rows,
            metric=binary_auroc,
            seed=seed,
            n_resamples=bootstrap_resamples,
        )
        reports.append(
            {
                "error_type": error_type,
                "status": "ok_with_flags" if flags else "ok",
                "flags": flags,
                "n_eval_pairs": n_eval_pairs,
                "n_eval_rows": int(eval_mask.sum()),
                "n_train_pairs": len(set(dataset.group_ids)) - n_eval_pairs,
                "n_train_rows": int(train_mask.sum()),
                "positive_count": int(eval_labels.sum()),
                "negative_count": int((~eval_labels).sum()),
                "auroc": binary_auroc(eval_labels, scores),
                "group_bootstrap_ci": per_type_ci,
                **fit_report,
            }
        )
    if np.isnan(all_scores).any():
        raise AssertionError("held-out error-type evaluation did not score every row")
    macro = float(np.mean([report["auroc"] for report in reports]))
    micro = binary_auroc(dataset.labels, all_scores)
    normalized_score_sensitivity = _within_type_percentile_scores(
        all_scores,
        normalized_types,
    )
    macro_ci = _heldout_macro_group_bootstrap_ci(
        dataset.labels,
        all_scores,
        dataset.group_ids,
        normalized_types,
        seed=seed,
        n_resamples=bootstrap_resamples,
    )
    micro_ci = group_bootstrap_metric_ci(
        dataset.labels,
        all_scores,
        dataset.group_ids,
        metric=binary_auroc,
        seed=seed,
        n_resamples=bootstrap_resamples,
    )
    predictions = [
        {
            "sample_id": sample_id,
            "group_id": group_id,
            "label": "corrupt" if label else "clean",
            "heldout_error_type": error_type,
            "decision_score": float(score),
            "within_type_percentile_score": float(percentile_score),
        }
        for sample_id, group_id, label, error_type, score, percentile_score in zip(
            dataset.sample_ids,
            dataset.group_ids,
            dataset.labels,
            normalized_types,
            all_scores,
            normalized_score_sensitivity,
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "protocol": "leave_one_error_type_out",
        "baseline": spec.to_json(),
        "seed": seed,
        "small_sample_pairs_threshold": small_sample_pairs,
        "macro_auroc": macro,
        "macro_definition": "unweighted mean of per-error-type AUROCs",
        "macro_group_bootstrap_ci": macro_ci,
        "micro_auroc": micro,
        "micro_definition": "AUROC after pooling each row's held-out-type decision score",
        "micro_interpretation_warning": (
            "Decision scores come from separately fitted leave-one-type-out models and "
            "may differ in scale; use macro AUROC as primary."
        ),
        "micro_group_bootstrap_ci": micro_ci,
        "micro_within_type_percentile_auroc": binary_auroc(
            dataset.labels,
            normalized_score_sensitivity,
        ),
        "micro_within_type_percentile_definition": (
            "sensitivity analysis pooling average-rank percentiles computed separately "
            "inside each held-out error type"
        ),
        "num_error_types": len(reports),
        "num_pairs": len(set(dataset.group_ids)),
        "num_rows": len(dataset.texts),
        "runtime_versions": surface_runtime_versions(),
        "dataset_manifest": surface_dataset_manifest(dataset),
        "error_types": reports,
        "heldout_predictions": predictions,
    }


def _within_type_percentile_scores(
    scores: np.ndarray,
    error_types: tuple[str, ...],
) -> np.ndarray:
    percentiles = np.empty(scores.shape[0], dtype=float)
    for error_type in sorted(set(error_types)):
        indices = np.asarray(
            [index for index, value in enumerate(error_types) if value == error_type],
            dtype=np.int64,
        )
        values = scores[indices]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=float)
        start = 0
        while start < values.shape[0]:
            end = start + 1
            while end < values.shape[0] and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        percentiles[indices] = ranks / max(values.shape[0], 1)
    return percentiles


def _heldout_macro_group_bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: tuple[str, ...],
    error_types: tuple[str, ...],
    *,
    seed: int,
    n_resamples: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    by_group: dict[str, list[int]] = {}
    for index, group_id in enumerate(groups):
        by_group.setdefault(group_id, []).append(index)
    unique_groups = sorted(by_group)
    required_types = sorted(set(error_types))
    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [np.asarray(by_group[str(group_id)], dtype=np.int64) for group_id in sampled_groups]
        )
        sampled_types = np.asarray(error_types, dtype=object)[sampled_indices]
        type_aurocs: list[float] = []
        for error_type in required_types:
            type_mask = sampled_types == error_type
            if not np.any(type_mask):
                type_aurocs = []
                break
            type_aurocs.append(
                binary_auroc(
                    labels[sampled_indices][type_mask],
                    scores[sampled_indices][type_mask],
                )
            )
        if type_aurocs:
            bootstrap_values.append(float(np.mean(type_aurocs)))
    payload: dict[str, Any] = {
        "status": "ok" if bootstrap_values else "no_valid_resamples",
        "seed": seed,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_valid_resamples": len(bootstrap_values),
        "num_groups": len(unique_groups),
        "resampling_unit": "clean_corrupt_pair",
        "requires_all_error_types_in_resample": True,
    }
    if bootstrap_values:
        low, high = np.quantile(
            np.asarray(bootstrap_values, dtype=float),
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        payload["ci_low"] = float(low)
        payload["ci_high"] = float(high)
    return payload


def _fit_tfidf_scores(
    train_texts: list[str],
    train_labels: np.ndarray,
    eval_texts: list[str],
    *,
    spec: TfidfBaselineSpec,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError(
            "Install analysis dependencies with `uv sync --extra analysis`."
        ) from exc

    if np.unique(train_labels).size != 2:
        raise ValueError("TF-IDF training split must contain both classes")
    vectorizer = TfidfVectorizer(
        analyzer=spec.analyzer,
        ngram_range=(spec.ngram_min, spec.ngram_max),
        lowercase=spec.lowercase,
        min_df=spec.min_df,
        max_features=spec.max_features,
        sublinear_tf=spec.sublinear_tf,
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    eval_matrix = vectorizer.transform(eval_texts)
    classifier = LogisticRegression(
        C=spec.logistic_c,
        class_weight=spec.class_weight,
        solver=spec.solver,
        max_iter=spec.max_iter,
        random_state=seed,
    )
    classifier.fit(train_matrix, train_labels.astype(int))
    scores = np.asarray(classifier.decision_function(eval_matrix), dtype=float)
    report = {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_matrix_shape": [int(value) for value in train_matrix.shape],
        "eval_matrix_shape": [int(value) for value in eval_matrix.shape],
    }
    return scores, report


def validate_grouped_kfold_manifest(
    manifest: dict[str, Any],
    *,
    expected_groups: set[str],
    expected_seed: int,
    expected_n_splits: int,
) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != "paired_grouped_kfold":
        raise ValueError("unexpected TF-IDF split manifest schema or protocol")
    if manifest.get("seed") != expected_seed:
        raise ValueError("TF-IDF split manifest seed differs from the requested seed")
    if manifest.get("n_splits") != expected_n_splits:
        raise ValueError("TF-IDF split manifest n_splits differs from the requested value")
    if manifest.get("num_groups") != len(expected_groups):
        raise ValueError("TF-IDF split manifest group count is inconsistent")
    if manifest.get("group_ids_hash") != stable_hash(sorted(expected_groups), length=64):
        raise ValueError("TF-IDF split manifest group hash is invalid")
    manifest_payload = dict(manifest)
    observed_manifest_hash = manifest_payload.pop("manifest_hash", None)
    if observed_manifest_hash != stable_hash(manifest_payload, length=64):
        raise ValueError("TF-IDF split manifest self-hash is invalid")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("TF-IDF split manifest must contain folds")
    if len(folds) != expected_n_splits:
        raise ValueError("TF-IDF split manifest fold count is inconsistent")
    seen_eval: set[str] = set()
    seen_fold_indices: set[int] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("TF-IDF split fold must be a mapping")
        fold_index = int(fold.get("fold_index", -1))
        if fold_index in seen_fold_indices:
            raise ValueError("TF-IDF split contains duplicate fold indices")
        seen_fold_indices.add(fold_index)
        if fold.get("seed") != expected_seed:
            raise ValueError("TF-IDF split fold seed is inconsistent")
        train = set(str(value) for value in fold["train_group_ids"])
        evaluate = set(str(value) for value in fold["eval_group_ids"])
        if fold.get("train_groups_hash") != stable_hash(sorted(train), length=64):
            raise ValueError("TF-IDF split train group hash is invalid")
        if fold.get("eval_groups_hash") != stable_hash(sorted(evaluate), length=64):
            raise ValueError("TF-IDF split eval group hash is invalid")
        fold_payload = dict(fold)
        observed_fold_hash = fold_payload.pop("fold_hash", None)
        if observed_fold_hash != stable_hash(fold_payload, length=64):
            raise ValueError("TF-IDF split fold self-hash is invalid")
        if train & evaluate:
            raise ValueError("TF-IDF split contains train/eval group overlap")
        if train | evaluate != expected_groups:
            raise ValueError("TF-IDF fold does not cover all expected groups")
        if seen_eval & evaluate:
            raise ValueError("TF-IDF group appears in multiple eval folds")
        seen_eval.update(evaluate)
    if seen_fold_indices != set(range(expected_n_splits)):
        raise ValueError("TF-IDF split fold indices must be contiguous from zero")
    if seen_eval != expected_groups:
        raise ValueError("TF-IDF eval folds do not exhaust all groups")
