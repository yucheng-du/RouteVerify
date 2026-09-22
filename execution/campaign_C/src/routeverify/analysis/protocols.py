"""Deterministic grouped analysis protocols and group-aware uncertainty helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from routeverify.data.schema import stable_hash

PROTOCOL_SCHEMA_VERSION = 1
UNKNOWN_STRATUM = "unknown"


def build_repeated_grouped_split_manifest(
    groups: Sequence[str],
    *,
    strata: Sequence[str | None] | None = None,
    seed: int = 0,
    repeats: int = 5,
    eval_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build exact, reusable grouped holdout splits.

    ``groups`` and optional ``strata`` are row-level sequences. Repeated rows from
    the same group are collapsed, and a group may belong to only one stratum.
    """

    if not groups:
        raise ValueError("at least one group is required")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")

    group_strata = _group_strata(groups, strata)
    all_groups = sorted(group_strata)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in group_strata.items():
        by_stratum[stratum].append(group_id)

    folds: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        split_seed = seed + repeat_index
        eval_groups: set[str] = set()
        for _stratum, stratum_groups in sorted(by_stratum.items()):
            ordered = sorted(
                stratum_groups,
                key=lambda group_id: stable_hash({"group": group_id, "seed": split_seed}),
            )
            eval_count = _bounded_split_count(len(ordered), eval_fraction)
            eval_groups.update(ordered[:eval_count])

        # If every stratum is a singleton, per-stratum clamping can leave no eval
        # groups. Preserve a usable grouped split while reporting the sparse strata.
        if not eval_groups and len(all_groups) > 1:
            eval_groups.add(
                min(
                    all_groups,
                    key=lambda group_id: stable_hash({"group": group_id, "seed": split_seed}),
                )
            )
        if len(eval_groups) == len(all_groups) and len(all_groups) > 1:
            eval_groups.remove(
                max(
                    eval_groups,
                    key=lambda group_id: stable_hash({"group": group_id, "seed": split_seed}),
                )
            )

        train_group_ids = sorted(set(all_groups) - eval_groups)
        eval_group_ids = sorted(eval_groups)
        fold: dict[str, Any] = {
            "fold_index": repeat_index,
            "seed": split_seed,
            "train_group_ids": train_group_ids,
            "eval_group_ids": eval_group_ids,
            "train_groups_hash": stable_hash(train_group_ids, length=64),
            "eval_groups_hash": stable_hash(eval_group_ids, length=64),
            "stratum_counts": _fold_stratum_counts(
                by_stratum,
                train_groups=set(train_group_ids),
                eval_groups=set(eval_group_ids),
            ),
        }
        _assert_group_partition(train_group_ids, eval_group_ids, expected_groups=all_groups)
        fold["fold_hash"] = stable_hash(fold, length=64)
        folds.append(fold)

    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": "repeated_grouped_holdout",
        "seed": seed,
        "repeats": repeats,
        "eval_fraction": eval_fraction,
        "stratified": strata is not None,
        "num_groups": len(all_groups),
        "group_ids_hash": stable_hash(all_groups, length=64),
        "folds": folds,
    }
    payload["manifest_hash"] = stable_hash(payload, length=64)
    validate_repeated_grouped_split_manifest(payload, expected_groups=all_groups)
    return payload


def validate_repeated_grouped_split_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_groups: Sequence[str] | None = None,
) -> None:
    """Validate exact group coverage and train/eval disjointness for every fold."""

    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("split manifest must contain at least one fold")
    expected = None if expected_groups is None else sorted(set(expected_groups))
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ValueError("split manifest fold must be a mapping")
        train_groups = _string_list(fold.get("train_group_ids"), field="train_group_ids")
        eval_groups = _string_list(fold.get("eval_group_ids"), field="eval_group_ids")
        _assert_group_partition(train_groups, eval_groups, expected_groups=expected)
        if fold.get("train_groups_hash") != stable_hash(train_groups, length=64):
            raise ValueError("train group hash does not match split IDs")
        if fold.get("eval_groups_hash") != stable_hash(eval_groups, length=64):
            raise ValueError("eval group hash does not match split IDs")
        expected_fold_hash = stable_hash(
            {key: value for key, value in fold.items() if key != "fold_hash"},
            length=64,
        )
        if fold.get("fold_hash") != expected_fold_hash:
            raise ValueError("fold hash does not match fold contents")
    expected_manifest_hash = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"},
        length=64,
    )
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("manifest hash does not match split contents")


def build_discovery_validation_manifest(
    group_ids: Sequence[str],
    error_types: Sequence[str | None],
    *,
    seed: int = 0,
    discovery_fraction: float = 0.5,
    rare_threshold: int = 20,
) -> dict[str, Any]:
    """Create a fixed error-type-stratified discovery/validation partition."""

    if not 0.0 < discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be between 0 and 1")
    if rare_threshold <= 0:
        raise ValueError("rare_threshold must be positive")
    group_error_types = _group_strata(group_ids, error_types)
    if not group_error_types:
        raise ValueError("at least one group is required")

    by_error_type: dict[str, list[str]] = defaultdict(list)
    for group_id, error_type in group_error_types.items():
        by_error_type[error_type].append(group_id)

    discovery: set[str] = set()
    validation: set[str] = set()
    stratum_reports: dict[str, dict[str, Any]] = {}
    rare_error_types: list[dict[str, Any]] = []
    for error_type, stratum_groups in sorted(by_error_type.items()):
        ordered = sorted(
            stratum_groups,
            key=lambda group_id: stable_hash(
                {
                    "protocol": "discovery_validation",
                    "group": group_id,
                    "seed": seed,
                }
            ),
        )
        if len(ordered) == 1:
            discovery_count = 1
        else:
            discovery_count = round(len(ordered) * discovery_fraction)
            discovery_count = max(1, min(discovery_count, len(ordered) - 1))
        discovery.update(ordered[:discovery_count])
        validation.update(ordered[discovery_count:])

        flags: list[str] = []
        if len(ordered) < rare_threshold:
            flags.append("small_sample")
        if error_type == UNKNOWN_STRATUM:
            flags.append("unidentifiable_error_type")
        if len(ordered) < 2:
            flags.append("cannot_represent_in_both_splits")
        report = {
            "num_groups": len(ordered),
            "num_discovery_groups": discovery_count,
            "num_validation_groups": len(ordered) - discovery_count,
            "flags": flags,
        }
        stratum_reports[error_type] = report
        if flags:
            rare_error_types.append({"error_type": error_type, **report})

    all_groups = sorted(group_error_types)
    discovery_group_ids = sorted(discovery)
    validation_group_ids = sorted(validation)
    _assert_group_partition(
        discovery_group_ids,
        validation_group_ids,
        expected_groups=all_groups,
    )
    payload: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": "error_type_stratified_discovery_validation",
        "seed": seed,
        "discovery_fraction": discovery_fraction,
        "rare_threshold": rare_threshold,
        "num_groups": len(all_groups),
        "group_ids_hash": stable_hash(all_groups, length=64),
        "discovery_group_ids": discovery_group_ids,
        "validation_group_ids": validation_group_ids,
        "discovery_groups_hash": stable_hash(discovery_group_ids, length=64),
        "validation_groups_hash": stable_hash(validation_group_ids, length=64),
        "error_type_counts": stratum_reports,
        "rare_or_unidentifiable_error_types": rare_error_types,
    }
    payload["manifest_hash"] = stable_hash(payload, length=64)
    validate_discovery_validation_manifest(payload, expected_groups=all_groups)
    return payload


def validate_discovery_validation_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_groups: Sequence[str] | None = None,
) -> None:
    """Assert discovery and validation group IDs are disjoint and exhaustive."""

    discovery = _string_list(
        manifest.get("discovery_group_ids"),
        field="discovery_group_ids",
    )
    validation = _string_list(
        manifest.get("validation_group_ids"),
        field="validation_group_ids",
    )
    expected = None if expected_groups is None else sorted(set(expected_groups))
    _assert_group_partition(discovery, validation, expected_groups=expected)
    if manifest.get("discovery_groups_hash") != stable_hash(discovery, length=64):
        raise ValueError("discovery group hash does not match split IDs")
    if manifest.get("validation_groups_hash") != stable_hash(validation, length=64):
        raise ValueError("validation group hash does not match split IDs")
    expected_manifest_hash = stable_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"},
        length=64,
    )
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("manifest hash does not match split contents")


def group_bootstrap_metric_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int = 0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap for a metric, resampling whole groups."""

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if labels_array.shape[0] != scores_array.shape[0] or len(groups) != labels_array.shape[0]:
        raise ValueError("labels, scores, and groups must have the same row count")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    grouped_indices = _indices_by_group(groups)
    unique_groups = sorted(grouped_indices)
    if not unique_groups:
        raise ValueError("at least one group is required")
    estimate = float(metric(labels_array, scores_array))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate(
            [
                np.asarray(grouped_indices[str(group_id)], dtype=np.int64)
                for group_id in sampled_groups
            ]
        )
        try:
            values.append(
                float(metric(labels_array[sampled_indices], scores_array[sampled_indices]))
            )
        except ValueError:
            continue

    payload: dict[str, Any] = {
        "status": "ok" if values else "no_valid_resamples",
        "estimate": estimate,
        "seed": seed,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "n_valid_resamples": len(values),
        "num_groups": len(unique_groups),
        "resampling_unit": "group",
    }
    if values:
        lower, upper = np.quantile(np.asarray(values, dtype=float), [alpha / 2, 1 - alpha / 2])
        payload["ci_low"] = float(lower)
        payload["ci_high"] = float(upper)
    return payload


def permute_labels_by_group(
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels without breaking paired/grouped row structure.

    Mixed-label groups (for example clean/corrupt pairs) are permuted within
    group. Homogeneous groups are permuted as whole blocks among groups with the
    same row count, which also handles singleton ProcessBench groups.
    """

    labels_array = np.asarray(labels, dtype=bool)
    if labels_array.ndim != 1 or len(groups) != labels_array.shape[0]:
        raise ValueError("labels and groups must have the same one-dimensional row count")
    permuted = labels_array.copy()
    grouped_indices = _indices_by_group(groups)
    homogeneous_by_size: dict[int, list[str]] = defaultdict(list)
    for group_id, indices in grouped_indices.items():
        group_values = labels_array[indices]
        if np.unique(group_values).size > 1:
            shuffled = group_values.copy()
            rng.shuffle(shuffled)
            permuted[indices] = shuffled
        else:
            homogeneous_by_size[len(indices)].append(group_id)

    for size_groups in homogeneous_by_size.values():
        group_values = np.asarray(
            [labels_array[grouped_indices[group_id][0]] for group_id in size_groups],
            dtype=bool,
        )
        rng.shuffle(group_values)
        for group_id, value in zip(size_groups, group_values, strict=True):
            permuted[grouped_indices[group_id]] = value
    return permuted


def _group_strata(
    groups: Sequence[str],
    strata: Sequence[str | None] | None,
) -> dict[str, str]:
    if strata is not None and len(strata) != len(groups):
        raise ValueError("groups and strata must have the same row count")
    group_strata: dict[str, str] = {}
    for index, raw_group in enumerate(groups):
        group_id = str(raw_group)
        raw_stratum = None if strata is None else strata[index]
        stratum = UNKNOWN_STRATUM if raw_stratum is None else str(raw_stratum)
        previous = group_strata.get(group_id)
        if previous is not None and previous != stratum:
            raise ValueError(f"group {group_id!r} belongs to multiple strata")
        group_strata[group_id] = stratum
    return group_strata


def _bounded_split_count(size: int, fraction: float) -> int:
    if size < 2:
        return 0
    count = round(size * fraction)
    return max(1, min(count, size - 1))


def _fold_stratum_counts(
    by_stratum: Mapping[str, Sequence[str]],
    *,
    train_groups: set[str],
    eval_groups: set[str],
) -> dict[str, dict[str, int]]:
    return {
        stratum: {
            "total": len(group_ids),
            "train": sum(group_id in train_groups for group_id in group_ids),
            "eval": sum(group_id in eval_groups for group_id in group_ids),
        }
        for stratum, group_ids in sorted(by_stratum.items())
    }


def _indices_by_group(groups: Sequence[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(groups):
        grouped[str(group_id)].append(index)
    return {group_id: np.asarray(indices, dtype=np.int64) for group_id, indices in grouped.items()}


def _assert_group_partition(
    first: Sequence[str],
    second: Sequence[str],
    *,
    expected_groups: Sequence[str] | None,
) -> None:
    first_set = set(first)
    second_set = set(second)
    overlap = first_set & second_set
    if overlap:
        raise ValueError(f"group leakage across splits: {sorted(overlap)[:5]}")
    if len(first_set) != len(first) or len(second_set) != len(second):
        raise ValueError("duplicate group IDs within a split")
    if expected_groups is not None and first_set | second_set != set(expected_groups):
        raise ValueError("split group IDs do not exhaust the expected groups")


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [str(item) for item in value]
