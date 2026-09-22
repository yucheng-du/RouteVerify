#!/usr/bin/env python3
"""Run versioned source-grouped and matched-estimator Qwen3 CPU corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from safetensors.numpy import load_file

from routeverify.analysis.fair_comparison import (
    evaluate_dense_logistic_grouped_cv,
    paired_group_bootstrap_auroc_difference,
)
from routeverify.analysis.routing_probes import routing_features_from_manifest_arrays
from routeverify.analysis.surface_baselines import (
    TfidfBaselineSpec,
    build_grouped_kfold_manifest,
    evaluate_tfidf_grouped_cv,
)
from routeverify.data.prmbench import (
    PRMBenchPair,
    load_prmbench_pairs_with_provenance,
    prmbench_source_problem_components,
    prmbench_source_problem_id,
)
from routeverify.data.schema import normalize_text, stable_hash

CLAIM_CRITICAL_CODE_PATHS = (
    "scripts/recompute_qwen_cpu_corrections.py",
    "src/routeverify/analysis/fair_comparison.py",
    "src/routeverify/analysis/probes.py",
    "src/routeverify/analysis/protocols.py",
    "src/routeverify/analysis/routing_probes.py",
    "src/routeverify/analysis/surface_baselines.py",
    "src/routeverify/data/prmbench.py",
    "src/routeverify/data/schema.py",
)


def main() -> None:
    args = _parse_args()
    repository_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    config_path = args.config.resolve()
    config = cast(dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")))
    _enforce_cpu_offline(config, hf_home=args.hf_home.resolve())

    output_config = _mapping(config, "output")
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else (repository_root / str(output_config["directory"])).resolve()
    )
    if bool(output_config["refuse_nonempty_directory"]) and output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = _mapping(config, "dataset")
    selection_config = _mapping(dataset_config, "selection")
    load_result = load_prmbench_pairs_with_provenance(
        hf_id=str(dataset_config["hf_id"]),
        split=str(dataset_config["split"]),
        cache_dir=args.hf_home.resolve() / "datasets",
        revision=str(dataset_config["requested_revision"]),
        expected_resolved_data_revision=str(dataset_config["expected_resolved_data_revision"]),
        expected_raw_sha256=str(dataset_config["expected_raw_sha256"]),
        limit=int(selection_config["limit_pairs"]),
        seed=int(selection_config["seed"]),
        include_ambiguous_index=bool(selection_config["include_ambiguous_index"]),
    )
    pairs = load_result.pairs
    if len(pairs) != int(selection_config["limit_pairs"]):
        raise ValueError(f"expected 580 selected PRMBench pairs, observed {len(pairs)}")

    historical_config = _mapping(config, "historical_inputs")
    selected_config = _mapping(historical_config, "selected_pairs")
    selected_path = _resolve_repository_path(repository_root, str(selected_config["path"]))
    _verify_sha(selected_path, str(selected_config["sha256"]), label="historical selected pairs")
    historical_selection = json.loads(selected_path.read_text(encoding="utf-8"))
    if historical_selection["manifest_hash"] != load_result.selection_manifest["manifest_hash"]:
        raise ValueError("reloaded selection manifest differs from the historical selection")

    split_config = _mapping(historical_config, "discovery_validation_split")
    historical_split_path = _resolve_repository_path(repository_root, str(split_config["path"]))
    _verify_sha(
        historical_split_path,
        str(split_config["sha256"]),
        label="historical discovery/validation split",
    )
    historical_split = json.loads(historical_split_path.read_text(encoding="utf-8"))

    grouping_config = _mapping(config, "source_problem_grouping")
    source_manifest, pair_to_source = _build_source_problem_manifest(
        pairs,
        historical_split,
        definition=str(grouping_config["definition"]),
    )
    historical_leakage = cast(dict[str, Any], source_manifest["historical_split_leakage"])
    expected_counts = {
        "source_problem_count": int(grouping_config["expected_source_problem_count"]),
        "cross_partition_source_count": int(
            grouping_config["expected_historical_cross_partition_source_count"]
        ),
        "affected_pair_count": int(grouping_config["expected_historical_affected_pair_count"]),
    }
    observed_counts = {
        "source_problem_count": int(source_manifest["num_source_problems"]),
        "cross_partition_source_count": int(historical_leakage["cross_partition_source_count"]),
        "affected_pair_count": int(historical_leakage["affected_pair_count"]),
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"source-problem audit differs from the registered counts: "
            f"observed={observed_counts}, expected={expected_counts}"
        )

    cv_config = _mapping(grouping_config, "grouped_cv")
    seed = int(cv_config["seed"])
    n_splits = int(cv_config["folds"])
    bootstrap_resamples = int(cv_config["bootstrap_resamples"])
    shared_split = build_grouped_kfold_manifest(
        [pair_to_source[pair.clean.group_id] for pair in pairs],
        seed=seed,
        n_splits=n_splits,
    )
    source_manifest["corrected_grouped_cv_split"] = shared_split
    source_manifest["manifest_hash"] = stable_hash(source_manifest, length=64)
    _write_json(output_dir / "source_problem_manifest.json", source_manifest)

    comparison_config = _mapping(config, "matched_estimator_comparison")
    routing_config = _mapping(comparison_config, "routing")
    routing_run_config = _mapping(historical_config, "routing_run")
    routing_run = Path(str(routing_run_config["path"]))
    routing_features, routing_rows, routing_inputs = _load_routing_feature(
        pairs,
        routing_run,
        expected_manifest_sha=str(routing_run_config["data_manifest_sha256"]),
        layer=int(routing_config["layer"]),
        feature_name=str(routing_config["feature"]),
        num_experts=int(routing_config["num_experts"]),
    )

    ordered_examples = [example for pair in pairs for example in (pair.clean, pair.corrupt)]
    sample_ids = [example.sample_id for example in ordered_examples]
    pair_group_ids = [example.group_id for example in ordered_examples]
    analysis_group_ids = [pair_to_source[group_id] for group_id in pair_group_ids]
    labels = np.asarray([example.variant == "corrupt" for example in ordered_examples], dtype=bool)
    feature_matrix = np.stack([routing_features[sample_id] for sample_id in sample_ids])
    boundary_step_by_sample_id: dict[str, int] = {}
    for example, row in zip(ordered_examples, routing_rows, strict=True):
        if example.sample_id != str(row["sample_id"]):
            raise AssertionError("routing rows are not aligned to the selected sample order")
        if (example.variant == "corrupt") != bool(row["is_error_boundary"]):
            raise ValueError("routing artifact labels differ from selected PRMBench variants")
        boundary_step_by_sample_id[example.sample_id] = int(row["boundary_step"])

    estimator_config = _mapping(comparison_config, "estimator")
    routing_result = evaluate_dense_logistic_grouped_cv(
        feature_matrix,
        labels,
        sample_ids=sample_ids,
        pair_group_ids=pair_group_ids,
        analysis_group_ids=analysis_group_ids,
        split_manifest=shared_split,
        analysis_group_definition=str(grouping_config["definition"]),
        seed=seed,
        n_splits=n_splits,
        bootstrap_resamples=bootstrap_resamples,
        logistic_c=float(estimator_config["C"]),
        class_weight=(
            None
            if estimator_config.get("class_weight") is None
            else str(estimator_config["class_weight"])
        ),
        solver=str(estimator_config["solver"]),
        max_iter=int(estimator_config["max_iter"]),
        standardize=True,
    )
    for prediction in routing_result["oof_predictions"]:
        sample_id = str(prediction["sample_id"])
        prediction["boundary_step"] = boundary_step_by_sample_id[sample_id]
    routing_result["feature"] = {
        "layer": int(routing_config["layer"]),
        "name": str(routing_config["feature"]),
        "num_experts": int(routing_config["num_experts"]),
    }

    surface_config = _mapping(comparison_config, "surface")
    if str(surface_config["boundary_source"]) != "routing_manifest_row_boundary_step":
        raise ValueError("surface comparison must use routing manifest row boundaries")
    ngram_range = [int(value) for value in surface_config["ngram_range"]]
    surface_spec = TfidfBaselineSpec(
        name=str(surface_config["baseline"]),
        text_field=cast(Any, str(surface_config["text_field"])),
        analyzer=cast(Any, str(surface_config["analyzer"])),
        ngram_min=ngram_range[0],
        ngram_max=ngram_range[1],
        lowercase=bool(surface_config["lowercase"]),
        min_df=int(surface_config["min_df"]),
        max_features=int(surface_config["max_features"]),
        sublinear_tf=bool(surface_config["sublinear_tf"]),
        logistic_c=float(estimator_config["C"]),
        class_weight=cast(Any, estimator_config.get("class_weight")),
        solver=cast(Any, str(estimator_config["solver"])),
        max_iter=int(estimator_config["max_iter"]),
    )
    surface_result = evaluate_tfidf_grouped_cv(
        pairs,
        surface_spec,
        seed=seed,
        n_splits=n_splits,
        bootstrap_resamples=bootstrap_resamples,
        split_manifest=shared_split,
        analysis_group_by_pair_group=pair_to_source,
        analysis_group_definition=str(grouping_config["definition"]),
        boundary_step_by_sample_id=boundary_step_by_sample_id,
    )

    routing_predictions = {str(row["sample_id"]): row for row in routing_result["oof_predictions"]}
    surface_predictions = {str(row["sample_id"]): row for row in surface_result["oof_predictions"]}
    if set(routing_predictions) != set(surface_predictions) or set(routing_predictions) != set(
        sample_ids
    ):
        raise ValueError("routing and surface predictions do not cover the same samples")
    routing_scores: list[float] = []
    surface_scores: list[float] = []
    combined_predictions: list[dict[str, Any]] = []
    for example in ordered_examples:
        routing_row = routing_predictions[example.sample_id]
        surface_row = surface_predictions[example.sample_id]
        for field in (
            "group_id",
            "analysis_group_id",
            "label",
            "fold_index",
            "boundary_step",
        ):
            if routing_row[field] != surface_row[field]:
                raise ValueError(f"routing/surface prediction mismatch for {field}")
        routing_score = float(routing_row["score"])
        surface_score = float(surface_row["score"])
        routing_scores.append(routing_score)
        surface_scores.append(surface_score)
        combined_predictions.append(
            {
                "sample_id": example.sample_id,
                "group_id": example.group_id,
                "source_problem_id": routing_row["analysis_group_id"],
                "error_type": example.error_type,
                "label": routing_row["label"],
                "fold_index": routing_row["fold_index"],
                "boundary_step": routing_row["boundary_step"],
                "routing_score": routing_score,
                "surface_score": surface_score,
            }
        )

    paired_config = _mapping(comparison_config, "comparison")
    paired_difference = paired_group_bootstrap_auroc_difference(
        labels,
        np.asarray(routing_scores, dtype=float),
        np.asarray(surface_scores, dtype=float),
        analysis_group_ids,
        left_name="routing",
        right_name="surface",
        seed=seed,
        n_resamples=int(paired_config["paired_bootstrap_resamples"]),
        alpha=float(paired_config["alpha"]),
    )
    comparison_result = {
        "schema_version": 1,
        "status": "ok",
        "analysis_status": "post_hoc_versioned_sensitivity_not_prospective_validation",
        "estimand": "clean_vs_corrupt_discrimination_on_shared_source_problem_grouped_folds",
        "shared_split_manifest_hash": shared_split["manifest_hash"],
        "source_problem_definition": grouping_config["definition"],
        "estimator_match": {
            "status": "matched_supervised_estimator",
            "capture_position_match": {
                "status": "exact",
                "definition": "routing_manifest_row_boundary_step",
            },
            "shared": {
                "implementation": estimator_config["implementation"],
                "penalty": estimator_config["penalty"],
                "C": estimator_config["C"],
                "class_weight": estimator_config["class_weight"],
                "solver": estimator_config["solver"],
                "max_iter": estimator_config["max_iter"],
                "positive_class": estimator_config["positive_class"],
                "score": estimator_config["score"],
            },
            "representation_specific_unsupervised_preprocessing": {
                "routing": "StandardScaler fit on each training fold",
                "surface": "TF-IDF vocabulary/IDF fit on each training fold with L2 row norm",
            },
        },
        "routing": routing_result,
        "surface": surface_result,
        "paired_auroc_difference": paired_difference,
        "predictions": combined_predictions,
    }
    _write_json(output_dir / "matched_estimator_comparison.json", comparison_result)

    provenance = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "registration": config["registration"],
        "constraints": config["constraints"],
        "command": {"argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd())},
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "analysis_code": {
            **_git_provenance(repository_root),
            "claim_critical_files": _file_hash_manifest(
                repository_root,
                CLAIM_CRITICAL_CODE_PATHS,
            ),
        },
        "dataset": {
            "source": load_result.source_provenance,
            "selection_manifest_hash": load_result.selection_manifest["manifest_hash"],
            "pairs_hash": load_result.selection_manifest["pairs_hash"],
        },
        "historical_inputs": {
            "selected_pairs": {"path": str(selected_path), "sha256": _sha256_file(selected_path)},
            "discovery_validation_split": {
                "path": str(historical_split_path),
                "sha256": _sha256_file(historical_split_path),
            },
        },
        "routing_inputs": routing_inputs,
        "outputs": {
            path.name: _sha256_file(path)
            for path in sorted(output_dir.glob("*.json"))
            if path.name != "analysis_provenance.json"
        },
    }
    _write_json(output_dir / "analysis_provenance.json", provenance)
    print(
        json.dumps(
            {
                "source_problems": source_manifest["num_source_problems"],
                "historical_cross_partition_sources": historical_leakage[
                    "cross_partition_source_count"
                ],
                "historical_affected_pairs": historical_leakage["affected_pair_count"],
                "routing_micro_oof_auroc": routing_result["micro_oof_auroc"],
                "surface_micro_oof_auroc": surface_result["micro_oof_auroc"],
                "routing_minus_surface": paired_difference["difference"],
                "difference_ci": [
                    paired_difference.get("ci_low"),
                    paired_difference.get("ci_high"),
                ],
                "output_directory": str(output_dir),
            },
            indent=2,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/qwen3_cpu_corrections_v1.yaml"),
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(".cache/huggingface"),
        help="Existing read-only-compatible Hugging Face cache root.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _enforce_cpu_offline(config: Mapping[str, Any], *, hf_home: Path) -> None:
    constraints = _mapping(config, "constraints")
    if str(constraints["device"]) != "cpu":
        raise ValueError("this correction script only permits device=cpu")
    if bool(constraints["allow_model_weight_loading"]):
        raise ValueError("model weight loading must remain disabled")
    if bool(constraints["allow_network_downloads"]):
        raise ValueError("network downloads must remain disabled")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _build_source_problem_manifest(
    pairs: list[PRMBenchPair],
    historical_split: Mapping[str, Any],
    *,
    definition: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    discovery_groups = set(str(value) for value in historical_split["discovery_group_ids"])
    validation_groups = set(str(value) for value in historical_split["validation_group_ids"])
    expected_groups = {pair.clean.group_id for pair in pairs}
    if discovery_groups & validation_groups:
        raise ValueError("historical pair groups overlap across discovery/validation")
    if discovery_groups | validation_groups != expected_groups:
        raise ValueError("historical split does not cover the selected pairs")

    pair_to_source = prmbench_source_problem_components(pairs)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_group_id = pair.clean.group_id
        source_problem_id = pair_to_source[pair_group_id]
        historical_partition = "discovery" if pair_group_id in discovery_groups else "validation"
        rows.append(
            {
                "pair_id": pair.clean.pair_id,
                "pair_group_id": pair_group_id,
                "source_idx": pair.clean.metadata["source_idx"],
                "source_suffix_id": prmbench_source_problem_id(pair),
                "normalized_problem_hash": stable_hash(
                    normalize_text(pair.clean.problem),
                    length=64,
                ),
                "source_problem_id": source_problem_id,
                "error_type": pair.corrupt.error_type,
                "historical_partition": historical_partition,
            }
        )

    discovery_sources = {
        row["source_problem_id"] for row in rows if row["historical_partition"] == "discovery"
    }
    validation_sources = {
        row["source_problem_id"] for row in rows if row["historical_partition"] == "validation"
    }
    overlap = sorted(discovery_sources & validation_sources)
    affected_pair_ids = sorted(
        str(row["pair_id"]) for row in rows if row["source_problem_id"] in set(overlap)
    )
    cluster_sizes = Counter(str(row["source_problem_id"]) for row in rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "definition": definition,
        "definition_contract": (
            "Source-problem groups are connected components over two pair relations: "
            "equal '<source_suffix>' after stripping the exact error-type prefix, or "
            "equal problem text after conservative whitespace normalization."
        ),
        "n_pairs": len(pairs),
        "num_source_problems": len(cluster_sizes),
        "max_pairs_per_source_problem": max(cluster_sizes.values()),
        "cluster_size_counts": {
            str(size): count for size, count in sorted(Counter(cluster_sizes.values()).items())
        },
        "historical_split_manifest_hash": historical_split["manifest_hash"],
        "historical_split_leakage": {
            "discovery_source_count": len(discovery_sources),
            "validation_source_count": len(validation_sources),
            "cross_partition_source_count": len(overlap),
            "cross_partition_source_ids": overlap,
            "affected_pair_count": len(affected_pair_ids),
            "affected_pair_fraction": len(affected_pair_ids) / len(pairs),
            "affected_pair_ids": affected_pair_ids,
        },
        "pairs": rows,
        "pairs_hash": stable_hash(rows, length=64),
    }
    return payload, pair_to_source


def _load_routing_feature(
    pairs: list[PRMBenchPair],
    run_dir: Path,
    *,
    expected_manifest_sha: str,
    layer: int,
    feature_name: str,
    num_experts: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "data_manifest.jsonl"
    _verify_sha(manifest_path, expected_manifest_sha, label="routing data manifest")
    selected_samples = {
        example.sample_id for pair in pairs for example in (pair.clean, pair.corrupt)
    }
    rows_by_sample: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(manifest_path):
        sample_id = str(row["sample_id"])
        if sample_id not in selected_samples:
            continue
        if sample_id in rows_by_sample:
            raise ValueError(f"duplicate routing manifest sample: {sample_id}")
        rows_by_sample[sample_id] = row
    if set(rows_by_sample) != selected_samples:
        raise ValueError("routing manifest does not exactly cover selected PRMBench samples")

    ordered_samples = [
        example.sample_id for pair in pairs for example in (pair.clean, pair.corrupt)
    ]
    features: dict[str, np.ndarray] = {}
    activation_hash_rows: list[dict[str, str]] = []
    seen_activation_files: set[Path] = set()
    for sample_id in ordered_samples:
        row = rows_by_sample[sample_id]
        activation_path = _resolve_activation(run_dir, str(row["activation_file"]))
        arrays = load_file(activation_path)
        vectors = routing_features_from_manifest_arrays(
            row,
            arrays,
            site_prefix=f"layer_{layer}",
            num_experts=num_experts,
        )
        if feature_name not in vectors:
            raise ValueError(f"unsupported routing feature: {feature_name}")
        features[sample_id] = vectors[feature_name]
        if activation_path not in seen_activation_files:
            activation_hash_rows.append(
                {
                    "file": str(activation_path.relative_to(run_dir)),
                    "sha256": _sha256_file(activation_path),
                }
            )
            seen_activation_files.add(activation_path)
    activation_hash_rows.sort(key=lambda row: row["file"])
    ordered_rows = [rows_by_sample[sample_id] for sample_id in ordered_samples]
    environment_path = run_dir / "environment.json"
    resolved_config_path = run_dir / "config.resolved.yaml"
    inputs = {
        "run_directory": str(run_dir),
        "data_manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "environment": {
            "path": str(environment_path),
            "sha256": _sha256_file(environment_path),
        },
        "resolved_config": {
            "path": str(resolved_config_path),
            "sha256": _sha256_file(resolved_config_path),
        },
        "activation_files": {
            "count": len(activation_hash_rows),
            "aggregate_sha256": stable_hash(activation_hash_rows, length=64),
            "files": activation_hash_rows,
        },
    }
    return features, ordered_rows, inputs


def _resolve_activation(run_dir: Path, artifact_value: str) -> Path:
    raw = Path(artifact_value)
    candidates = (
        [raw]
        if raw.is_absolute()
        else [raw, run_dir / raw, run_dir / raw.name, run_dir / "activations" / raw.name]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"activation artifact not found: {artifact_value}")


def _resolve_repository_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _verify_sha(path: Path, expected: str, *, label: str) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}")


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    resolved = value.get(key)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return dict(resolved)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(cast(dict[str, Any], json.loads(line)))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hash_manifest(
    repository_root: Path,
    relative_paths: tuple[str, ...],
) -> dict[str, Any]:
    files = [
        {
            "path": relative_path,
            "sha256": _sha256_file(repository_root / relative_path),
        }
        for relative_path in relative_paths
    ]
    return {
        "count": len(files),
        "aggregate_sha256": stable_hash(files, length=64),
        "files": files,
    }


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=repository_root, text=True
    ).splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


if __name__ == "__main__":
    main()
