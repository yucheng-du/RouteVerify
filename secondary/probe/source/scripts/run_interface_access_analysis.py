#!/usr/bin/env python3
"""Run the recapture-only multi-model interface-access analysis on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from routeverify.analysis.interface_access import (
    INTERFACE_FEATURE_FAMILIES,
    assert_row_identity_matches,
    canonicalize_common_rows,
    compact_probe_result,
    evaluate_feature_families,
    extract_layer_features,
    load_canonical_source_problem_groups,
    load_jsonl_rows,
    load_selection_source_ids,
    recapture_source_components,
)
from routeverify.analysis.surface_baselines import build_grouped_kfold_manifest
from routeverify.data.schema import stable_hash

CLAIM_CRITICAL_CODE_PATHS = (
    "scripts/run_interface_access_analysis.py",
    "src/routeverify/analysis/interface_access.py",
    "src/routeverify/analysis/fair_comparison.py",
    "src/routeverify/analysis/probes.py",
    "src/routeverify/analysis/protocols.py",
    "src/routeverify/analysis/routing_probes.py",
    "src/routeverify/analysis/surface_baselines.py",
)

SUMMARY_FIELDS = (
    "model_id",
    "display_name",
    "protocol_grade",
    "recapture_protocol",
    "anchor",
    "layer",
    "feature_family",
    "feature_dim",
    "n_rows",
    "n_pairs",
    "n_analysis_groups",
    "micro_oof_auroc",
    "ci_low",
    "ci_high",
    "mean_fold_auroc",
    "split_manifest_hash",
)

COMPARISON_FIELDS = (
    "model_id",
    "display_name",
    "protocol_grade",
    "recapture_protocol",
    "anchor",
    "layer",
    "left_feature",
    "right_feature",
    "left_auroc",
    "right_auroc",
    "difference",
    "ci_low",
    "ci_high",
    "n_analysis_groups",
)

COVERAGE_FIELDS = (
    "model_id",
    "display_name",
    "protocol_grade",
    "recapture_protocol",
    "available",
    "n_manifest_rows",
    "n_common_rows",
    "n_common_pairs",
    "n_analysis_groups",
    "num_experts",
    "top_k",
    "anchors",
    "feature_families",
    "caveat",
)


def main() -> None:
    args = _parse_args()
    repository_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    config_path = args.config.resolve()
    archive_root = args.archive_root.resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(f"archive root does not exist: {archive_root}")
    config = cast(dict[str, Any], yaml.safe_load(config_path.read_text(encoding="utf-8")))
    _validate_constraints(config)
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else (repository_root / str(_mapping(config, "output")["directory"])).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = _mapping(config, "protocol")
    seed = int(protocol["seed"])
    n_splits = int(protocol["folds"])
    bootstrap_resamples = int(protocol["bootstrap_resamples"])
    logistic_c = float(_mapping(protocol, "estimator")["C"])
    analysis_group_definition = str(protocol["analysis_group_definition"])

    reference_selection_path = archive_root / str(config["reference_selection_manifest"])
    pair_source_ids, reference_selection = load_selection_source_ids(reference_selection_path)
    pair_group_ids = _selection_pair_group_ids(reference_selection)
    canonical_manifest_value = config.get("canonical_source_problem_manifest")
    grouping_source: dict[str, Any]
    if canonical_manifest_value is not None:
        canonical_manifest_path = archive_root / str(canonical_manifest_value)
        expected_sha256 = str(config.get("canonical_source_problem_manifest_sha256", ""))
        if expected_sha256 and _sha256(canonical_manifest_path) != expected_sha256:
            raise ValueError("canonical source-problem manifest SHA-256 mismatch")
        pair_analysis_groups, canonical_manifest = load_canonical_source_problem_groups(
            canonical_manifest_path,
            allowed_pair_group_ids=pair_group_ids,
            expected_definition=str(protocol["analysis_group_definition"]),
        )
        canonical_provenance_value = config.get("canonical_grouping_provenance")
        canonical_provenance_path = (
            archive_root / str(canonical_provenance_value)
            if canonical_provenance_value is not None
            else None
        )
        grouping_source = {
            "kind": "pre_existing_canonical_source_problem_manifest",
            "path": str(canonical_manifest_path),
            "sha256": _sha256(canonical_manifest_path),
            "upstream_manifest_hash": canonical_manifest.get("manifest_hash"),
            "upstream_pairs_hash": canonical_manifest.get("pairs_hash"),
            "upstream_n_pairs": canonical_manifest.get("n_pairs"),
            "upstream_num_source_problems": canonical_manifest.get(
                "num_source_problems"
            ),
            "allowlist_joined_pairs": len(pair_analysis_groups),
        }
    else:
        canonical_manifest_path = None
        canonical_provenance_path = None
        pair_analysis_groups = recapture_source_components(reference_selection)
        grouping_source = {
            "kind": "recapture_native_source_suffix_or_exact_clean_content_hash",
            "path": str(reference_selection_path),
            "sha256": _sha256(reference_selection_path),
            "allowlist_joined_pairs": len(pair_analysis_groups),
        }
    observed_analysis_groups = len(set(pair_analysis_groups.values()))
    expected_analysis_groups = int(protocol["expected_analysis_group_count"])
    if observed_analysis_groups != expected_analysis_groups:
        raise ValueError(
            f"source-component count mismatch: expected {expected_analysis_groups}, "
            f"observed {observed_analysis_groups}"
        )

    reference_manifest_path = archive_root / str(config["reference_recapture_manifest"])
    reference_manifest_rows = load_jsonl_rows(reference_manifest_path)
    reference_rows, reference_groups = canonicalize_common_rows(
        reference_manifest_rows,
        pair_source_ids=pair_source_ids,
        pair_analysis_group_ids=pair_analysis_groups,
    )
    expected_rows = int(protocol["row_count"])
    expected_pairs = int(protocol["pair_count"])
    if len(reference_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} common rows, observed {len(reference_rows)}")
    if len({str(row["pair_id"]) for row in reference_rows}) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} common pairs")
    split_manifest = build_grouped_kfold_manifest(
        tuple(reference_groups),
        seed=seed,
        n_splits=n_splits,
    )

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    input_paths = {reference_selection_path, reference_manifest_path, config_path}
    if canonical_manifest_path is not None:
        input_paths.add(canonical_manifest_path)
    if canonical_provenance_path is not None:
        input_paths.add(canonical_provenance_path)

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("analysis config must contain models")
    for model_value in models:
        if not isinstance(model_value, Mapping):
            raise ValueError("model config must be an object")
        model = dict(model_value)
        model_id = str(model["model_id"])
        display_name = str(model["display_name"])
        available = bool(model.get("available", True))
        base_coverage = {
            "model_id": model_id,
            "display_name": display_name,
            "protocol_grade": str(model["protocol_grade"]),
            "recapture_protocol": str(model["recapture_protocol"]),
            "available": available,
            "n_manifest_rows": None,
            "n_common_rows": None,
            "n_common_pairs": None,
            "n_analysis_groups": None,
            "num_experts": model.get("num_experts"),
            "top_k": model.get("top_k"),
            "anchors": "",
            "feature_families": "",
            "caveat": str(model.get("caveat", "")),
        }
        if not available:
            coverage_rows.append(base_coverage)
            model_results.append(
                {
                    "model_id": model_id,
                    "display_name": display_name,
                    "protocol_grade": str(model["protocol_grade"]),
                    "recapture_protocol": str(model["recapture_protocol"]),
                    "status": "not_available",
                    "caveat": str(model.get("caveat", "")),
                }
            )
            print(f"[{model_id}] unavailable: {model.get('caveat', '')}", flush=True)
            continue

        recapture_root = archive_root / str(model["recapture_root"])
        manifest_path = recapture_root / "data_manifest.jsonl"
        config_resolved_path = recapture_root / "config.resolved.yaml"
        input_paths.update({manifest_path, config_resolved_path})
        manifest_rows = load_jsonl_rows(manifest_path)
        rows, groups = canonicalize_common_rows(
            manifest_rows,
            pair_source_ids=pair_source_ids,
            pair_analysis_group_ids=pair_analysis_groups,
        )
        if groups != reference_groups:
            raise ValueError(f"analysis groups differ from reference for {model_id}")
        if bool(model.get("strict_common_row_identity", False)):
            assert_row_identity_matches(reference_rows, rows)

        selection_manifest_value = model.get("selection_manifest")
        selection_hash = None
        if selection_manifest_value is not None:
            selection_path = archive_root / str(selection_manifest_value)
            input_paths.add(selection_path)
            model_pair_source_ids, selection_payload = load_selection_source_ids(selection_path)
            if model_pair_source_ids != pair_source_ids:
                raise ValueError(f"selection pair/source IDs differ for {model_id}")
            selection_hash = selection_payload.get("pairs_hash")
            if selection_hash != reference_selection.get("pairs_hash"):
                raise ValueError(f"selection pairs_hash differs for {model_id}")

        anchors = _mapping(model, "anchors")
        anchor_rows = [f"{name}:{int(layer)}" for name, layer in anchors.items()]
        coverage = {
            **base_coverage,
            "n_manifest_rows": len(manifest_rows),
            "n_common_rows": len(rows),
            "n_common_pairs": len({str(row["pair_id"]) for row in rows}),
            "n_analysis_groups": len(set(groups)),
            "anchors": ";".join(anchor_rows),
            "feature_families": ";".join(INTERFACE_FEATURE_FAMILIES),
        }
        coverage_rows.append(coverage)
        model_result: dict[str, Any] = {
            "model_id": model_id,
            "display_name": display_name,
            "protocol_grade": str(model["protocol_grade"]),
            "recapture_protocol": str(model["recapture_protocol"]),
            "status": "ok",
            "caveat": str(model.get("caveat", "")),
            "selection_pairs_hash": selection_hash,
            "n_manifest_rows": len(manifest_rows),
            "n_common_rows": len(rows),
            "n_common_pairs": expected_pairs,
            "n_analysis_groups": observed_analysis_groups,
            "num_experts": int(model["num_experts"]),
            "top_k": int(model["top_k"]),
            "anchors": {},
        }

        for anchor, layer_value in anchors.items():
            layer = int(layer_value)
            print(f"[{model_id}] extracting {anchor} layer {layer}", flush=True)
            features = extract_layer_features(
                rows,
                recapture_root=recapture_root,
                layer=layer,
                num_experts=int(model["num_experts"]),
            )
            print(f"[{model_id}] fitting {anchor} layer {layer}", flush=True)
            probes, comparisons = evaluate_feature_families(
                features,
                rows,
                analysis_group_ids=groups,
                split_manifest=split_manifest,
                analysis_group_definition=analysis_group_definition,
                seed=seed,
                n_splits=n_splits,
                bootstrap_resamples=bootstrap_resamples,
                logistic_c=logistic_c,
            )
            compact_probes: dict[str, Any] = {}
            for feature_name in INTERFACE_FEATURE_FAMILIES:
                probe = cast(dict[str, Any], probes[feature_name])
                uncertainty = _mapping(probe, "group_bootstrap_ci")
                summary_rows.append(
                    {
                        "model_id": model_id,
                        "display_name": display_name,
                        "protocol_grade": str(model["protocol_grade"]),
                        "recapture_protocol": str(model["recapture_protocol"]),
                        "anchor": str(anchor),
                        "layer": layer,
                        "feature_family": feature_name,
                        "feature_dim": int(probe["feature_dim"]),
                        "n_rows": int(probe["n_rows"]),
                        "n_pairs": int(probe["n_pairs"]),
                        "n_analysis_groups": int(probe["n_analysis_groups"]),
                        "micro_oof_auroc": float(probe["micro_oof_auroc"]),
                        "ci_low": float(uncertainty["ci_low"]),
                        "ci_high": float(uncertainty["ci_high"]),
                        "mean_fold_auroc": float(probe["mean_fold_auroc"]),
                        "split_manifest_hash": str(probe["split_manifest_hash"]),
                    }
                )
                for prediction in cast(list[Mapping[str, Any]], probe["oof_predictions"]):
                    oof_rows.append(
                        {
                            "model_id": model_id,
                            "protocol_grade": str(model["protocol_grade"]),
                            "recapture_protocol": str(model["recapture_protocol"]),
                            "anchor": str(anchor),
                            "layer": layer,
                            "feature_family": feature_name,
                            **prediction,
                        }
                    )
                compact_probes[feature_name] = compact_probe_result(probe)

            for comparison in comparisons:
                left = _mapping(comparison, "left")
                right = _mapping(comparison, "right")
                comparison_rows.append(
                    {
                        "model_id": model_id,
                        "display_name": display_name,
                        "protocol_grade": str(model["protocol_grade"]),
                        "recapture_protocol": str(model["recapture_protocol"]),
                        "anchor": str(anchor),
                        "layer": layer,
                        "left_feature": str(left["name"]),
                        "right_feature": str(right["name"]),
                        "left_auroc": float(left["auroc"]),
                        "right_auroc": float(right["auroc"]),
                        "difference": float(comparison["difference"]),
                        "ci_low": comparison.get("ci_low"),
                        "ci_high": comparison.get("ci_high"),
                        "n_analysis_groups": int(comparison["num_groups"]),
                    }
                )
            model_result["anchors"][str(anchor)] = {
                "layer": layer,
                "features": compact_probes,
                "adjacent_interface_differences": comparisons,
            }
        model_results.append(model_result)

    source_group_rows = [
        {
            "pair_id": pair_id,
            "source_id": pair_source_ids[pair_id],
            "analysis_group_id": pair_analysis_groups[pair_id],
        }
        for pair_id in sorted(pair_source_ids)
    ]
    source_group_manifest: dict[str, Any] = {
        "schema_version": 1,
        "definition": analysis_group_definition,
        "limitation": str(protocol["grouping_limitation"]),
        "n_pairs": len(source_group_rows),
        "n_analysis_groups": observed_analysis_groups,
        "grouping_source": grouping_source,
        "pairs_hash": stable_hash(source_group_rows, length=64),
        "pairs": source_group_rows,
        "split_manifest": split_manifest,
    }
    source_group_manifest["manifest_hash"] = stable_hash(source_group_manifest, length=64)

    results_payload = {
        "schema_version": 1,
        "analysis_id": str(config["analysis_id"]),
        "status": "ok",
        "estimand": str(protocol["outcome"]),
        "positive_class": str(protocol["positive_class"]),
        "protocol": protocol,
        "common_selection": {
            "n_pairs": expected_pairs,
            "n_rows": expected_rows,
            "pairs_hash": reference_selection.get("pairs_hash"),
            "selection_manifest_hash": reference_selection.get("manifest_hash"),
            "analysis_group_count": observed_analysis_groups,
            "analysis_group_manifest_hash": source_group_manifest["manifest_hash"],
            "split_manifest_hash": split_manifest["manifest_hash"],
        },
        "models": model_results,
    }

    output_paths = {
        "results": output_dir / "interface_access_results.json",
        "summary": output_dir / "interface_access_summary.csv",
        "comparisons": output_dir / "interface_access_adjacent_differences.csv",
        "coverage": output_dir / "interface_access_coverage.csv",
        "oof": output_dir / "interface_access_oof_predictions.jsonl",
        "groups": output_dir / "source_component_folds.json",
        "readme": output_dir / "README.md",
    }
    _write_json(output_paths["results"], results_payload)
    _write_csv(output_paths["summary"], SUMMARY_FIELDS, summary_rows)
    _write_csv(output_paths["comparisons"], COMPARISON_FIELDS, comparison_rows)
    _write_csv(output_paths["coverage"], COVERAGE_FIELDS, coverage_rows)
    _write_jsonl(output_paths["oof"], oof_rows)
    _write_json(output_paths["groups"], source_group_manifest)
    output_paths["readme"].write_text(
        _render_readme(summary_rows, coverage_rows, protocol=protocol),
        encoding="utf-8",
    )

    provenance = _build_provenance(
        repository_root=repository_root,
        config_path=config_path,
        archive_root=archive_root,
        input_paths=sorted(input_paths),
        output_paths=output_paths,
        argv=sys.argv,
        results_payload=results_payload,
    )
    provenance_path = output_dir / "analysis_provenance.json"
    _write_json(provenance_path, provenance)
    print(f"Wrote {output_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/interface_access_common_20260829.yaml"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive://"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_constraints(config: Mapping[str, Any]) -> None:
    constraints = _mapping(config, "constraints")
    if str(constraints.get("device")) != "cpu":
        raise ValueError("interface-access analysis must be CPU-only")
    for key in (
        "allow_model_weight_loading",
        "allow_validation_access",
        "allow_source_archive_writes",
    ):
        if bool(constraints.get(key)):
            raise ValueError(f"forbidden analysis capability enabled: {key}")


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    resolved = value.get(key)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"expected object field: {key}")
    return dict(resolved)


def _selection_pair_group_ids(selection_manifest: Mapping[str, Any]) -> dict[str, str]:
    values = selection_manifest.get("pairs")
    if not isinstance(values, list) or not values:
        raise ValueError("selection manifest lacks pairs")
    mapping: dict[str, str] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("selection-manifest pair must be an object")
        pair_id = str(value.get("pair_id", ""))
        pair_group_id = str(value.get("group_id", ""))
        if not pair_id or not pair_group_id:
            raise ValueError("selection-manifest pair lacks pair_id/group_id")
        if pair_id in mapping:
            raise ValueError(f"duplicate pair in selection manifest: {pair_id}")
        mapping[pair_id] = pair_group_id
    return mapping


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _render_readme(
    summary_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> str:
    middle = [row for row in summary_rows if row["anchor"] == "middle"]
    lines = [
        "# Interface-access common analysis",
        "",
        "CPU-only matched-estimator reanalysis of archived PRMBench recaptures.",
        "The positive class is `corrupt`; AUROC measures held-out clean/corrupt",
        "discrimination, not causal use or correctness-specific semantics.",
        "",
        "## Protocol coverage",
        "",
        "| model | grade | recapture protocol | available | caveat |",
        "|---|---|---|:---:|---|",
    ]
    for row in coverage_rows:
        caveat = str(row.get("caveat", "")).replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {row['display_name']} | {row['protocol_grade']} | "
            f"{row['recapture_protocol']} | {'yes' if row['available'] else 'no'} | {caveat} |"
        )
    lines.extend(
        [
            "",
            "## Middle-anchor matched L2-logistic OOF AUROC",
            "",
            "| model | feature | AUROC | source-group 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in middle:
        lines.append(
            f"| {row['display_name']} | {row['feature_family']} | "
            f"{float(row['micro_oof_auroc']):.3f} | "
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] |"
        )
    contrast = _mapping(protocol, "contrast_feature_control")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            f"- Grouping: `{protocol['analysis_group_definition']}`.",
            f"- Grouping limitation: {protocol['grouping_limitation']}",
            f"- Contrast-feature control: `{contrast['status']}` — {contrast['reason']}",
            "- Qwen3 is a historical retrospective sensitivity analysis and is not pooled",
            "  with the paired-first-error common subset.",
            "- OLMoE has no archived recapture tensors; its access entries are NA.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_provenance(
    *,
    repository_root: Path,
    config_path: Path,
    archive_root: Path,
    input_paths: Sequence[Path],
    output_paths: Mapping[str, Path],
    argv: Sequence[str],
    results_payload: Mapping[str, Any],
) -> dict[str, Any]:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository_root, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=repository_root, text=True
    ).splitlines()
    code_hashes = {
        path: _sha256(repository_root / path) for path in CLAIM_CRITICAL_CODE_PATHS
    }
    return {
        "schema_version": 1,
        "analysis_id": results_payload["analysis_id"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *argv],
        "repository": {
            "root": str(repository_root),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_status_short": git_status,
            "claim_critical_code_sha256": code_hashes,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "safetensors": importlib.metadata.version("safetensors"),
        },
        "scope": {
            "device": "cpu",
            "model_weights_loaded": False,
            "validation_accessed": False,
            "source_archives_modified": False,
            "contrast_feature_control": "not_run_missing_predeclared_labels",
        },
        "inputs": {
            "archive_root": str(archive_root),
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "files": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in input_paths
                if path.is_file()
            ],
            "activation_integrity_contract": (
                "Every accessed activation basename must exist under the declared recapture "
                "root; required keys, dimensions, expert count, row count, and finite values "
                "are checked. Large immutable activation files are not globally rehashed."
            ),
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in output_paths.items()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
