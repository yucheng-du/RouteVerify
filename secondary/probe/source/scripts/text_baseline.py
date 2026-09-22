#!/usr/bin/env python3
"""Boundary-step TF-IDF baseline on the saved canonical interface-access folds.

This CPU-only analysis reads the public-source Arrow cache through a frozen
290-pair discovery allowlist. It never loads a checkpoint or validation result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from routeverify.analysis.fair_comparison import paired_group_bootstrap_auroc_difference
from routeverify.analysis.interface_access import canonicalize_common_rows, load_jsonl_rows
from routeverify.analysis.probes import binary_auroc
from routeverify.analysis.protocols import group_bootstrap_metric_ci
from routeverify.analysis.surface_baselines import (
    TfidfBaselineSpec,
    build_surface_dataset,
    evaluate_tfidf_grouped_cv,
    surface_dataset_manifest,
    validate_grouped_kfold_manifest,
)
from routeverify.data.prmbench import PRMBenchPair, eligible_pair_from_prmbench_record

SOURCE_ARROW = Path(
    "runpod_7d34x2ijognj73_olmoe_residual_v3_20260807/huggingface/datasets/"
    "hitsmy___prm_bench_preview/default/0.0.0/"
    "5cc7683d0ae5797f84d7aeac0607966f277c39e1/prm_bench_preview-train.arrow"
)
SPEC = TfidfBaselineSpec(
    name="boundary_word", text_field="boundary_step", analyzer="word",
    ngram_min=1, ngram_max=2,
)
IDENTITY_FIELDS = ("sample_id", "group_id", "analysis_group_id", "label", "fold_index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_discovery_pairs(path: Path, selection: dict[str, Any]) -> list[PRMBenchPair]:
    """Filter Arrow batches by source ID before converting source text to Python."""
    import pyarrow as pa
    import pyarrow.compute as pc

    selected = {row["pair_id"]: row for row in selection["pairs"]}
    source_ids = sorted({row["source_id"] for row in selected.values()})
    if len(selected) != len(selection["pairs"]) or len(source_ids) != len(selected):
        raise ValueError("selection pair/source IDs must be unique")
    records: list[dict[str, Any]] = []
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        if "idx" not in reader.schema.names:
            raise ValueError("source Arrow lacks idx column")
        for batch in reader:
            indices = batch.column(batch.schema.get_field_index("idx"))
            keep = pc.is_in(indices, value_set=pa.array(source_ids, type=indices.type))
            records.extend(batch.filter(keep).to_pylist())
    pairs: dict[str, PRMBenchPair] = {}
    for record in records:
        pair = eligible_pair_from_prmbench_record(record)
        if pair is None:
            raise ValueError(f"selected source failed normalization: {record['idx']}")
        pair_id = str(pair.clean.pair_id)
        if pair_id not in selected or pair_id in pairs:
            raise ValueError(f"unexpected or duplicate normalized pair: {pair_id}")
        expected = selected[pair_id]
        for variant in ("clean", "corrupt"):
            example = getattr(pair, variant)
            if example.sample_id != expected[f"{variant}_sample_id"]:
                raise ValueError(f"source sample ID mismatch: {example.sample_id}")
            if example.content_hash != expected[f"{variant}_content_hash"]:
                raise ValueError(f"source content hash mismatch: {example.sample_id}")
            if example.group_id != expected["group_id"]:
                raise ValueError(f"source pair group mismatch: {example.sample_id}")
        pairs[pair_id] = pair
    if set(pairs) != set(selected):
        raise ValueError(f"source missing selected pairs: {sorted(set(selected) - set(pairs))[:5]}")
    return [pairs[pair_id] for pair_id in sorted(pairs)]


def assert_oof_identity(
    text_rows: list[dict[str, Any]], router_rows: list[dict[str, Any]]
) -> None:
    """Require row order as well as every row/label/component/fold identity."""
    if len(text_rows) != len(router_rows):
        raise ValueError("text/router row counts differ")
    for text_row, router_row in zip(text_rows, router_rows, strict=True):
        for field in IDENTITY_FIELDS:
            if text_row[field] != router_row[field]:
                raise ValueError(f"text/router {field} mismatch: {text_row['sample_id']}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(archive_root: Path, repo: Path, output: Path, source_arrow: Path | None = None) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    canonical = repo / "docs/results/interface_access_canonical_common_20260829"
    config_path = repo / "configs/analysis/interface_access_canonical_common_20260829.yaml"
    config = yaml.safe_load(config_path.read_text())
    protocol = config["protocol"]
    estimator = protocol["estimator"]
    for field, expected in {
        "penalty": "l2", "C": SPEC.logistic_c, "class_weight": SPEC.class_weight,
        "solver": SPEC.solver, "max_iter": SPEC.max_iter,
    }.items():
        if estimator[field] != expected:
            raise ValueError(f"text/router classifier setting differs: {field}")
    seed, n_splits = int(protocol["seed"]), int(protocol["folds"])
    bootstrap = int(protocol["bootstrap_resamples"])
    groups_path = canonical / "source_component_folds.json"
    grouping = json.loads(groups_path.read_text())
    split = grouping["split_manifest"]
    pair_sources = {row["pair_id"]: row["source_id"] for row in grouping["pairs"]}
    pair_groups = {row["pair_id"]: row["analysis_group_id"] for row in grouping["pairs"]}
    validate_grouped_kfold_manifest(
        split, expected_groups=set(pair_groups.values()), expected_seed=seed,
        expected_n_splits=n_splits,
    )
    selection_path = archive_root / config["reference_selection_manifest"]
    selection = json.loads(selection_path.read_text())
    if {row["pair_id"]: row["source_id"] for row in selection["pairs"]} != pair_sources:
        raise ValueError("canonical selection and fold pair/source allowlists differ")
    source_path = source_arrow or archive_root / SOURCE_ARROW
    pairs = load_discovery_pairs(source_path, selection)
    if len(pairs) != 290 or len(set(pair_groups.values())) != 235:
        raise ValueError("expected the frozen 290 discovery pairs / 235 components")
    examples = {ex.sample_id: ex for pair in pairs for ex in (pair.clean, pair.corrupt)}
    analysis_groups = {ex.group_id: pair_groups[str(ex.pair_id)] for ex in examples.values()}
    oof_path = canonical / "interface_access_oof_predictions.jsonl"
    router_oof = [
        row for row in map(json.loads, oof_path.read_text().splitlines())
        if row["feature_family"] == "router_logits"
    ]
    summary_path = canonical / "interface_access_summary.csv"
    with summary_path.open() as handle:
        old_summary = {
            (row["model_id"], row["anchor"]): row for row in csv.DictReader(handle)
            if row["feature_family"] == "router_logits"
        }
    source_provenance = json.loads((canonical / "analysis_provenance.json").read_text())
    for name, path in {"groups": groups_path, "oof": oof_path, "summary": summary_path}.items():
        if sha256(path) != source_provenance["outputs"][name]["sha256"]:
            raise ValueError(f"canonical output hash changed: {path}")
    if sha256(config_path) != source_provenance["inputs"]["config"]["sha256"]:
        raise ValueError("canonical interface-access config hash changed")
    expected_input_hashes = {
        item["path"]: item["sha256"] for item in source_provenance["inputs"]["files"]
    }
    inputs = {config_path, groups_path, selection_path, source_path, oof_path, summary_path,
              canonical / "analysis_provenance.json"}
    cache: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for model in config["models"]:
        if not model.get("available", True):
            continue
        model_id = model["model_id"]
        manifest_path = archive_root / model["recapture_root"] / "data_manifest.jsonl"
        inputs.add(manifest_path)
        # These are recapture manifests, never validation experiment outputs.
        original_key = "archive://" + str(
            manifest_path.relative_to(archive_root)
        )
        if sha256(manifest_path) != expected_input_hashes[original_key]:
            raise ValueError(f"canonical recapture manifest hash changed: {manifest_path}")
        rows, _ = canonicalize_common_rows(
            load_jsonl_rows(manifest_path), pair_source_ids=pair_sources,
            pair_analysis_group_ids=pair_groups,
        )
        boundaries = {}
        for row in rows:
            example = examples[row["sample_id"]]
            for field in ("content_hash", "group_id", "variant", "first_error", "error_type"):
                if getattr(example, field) != row[field]:
                    raise ValueError(f"recapture/source {field} differs: {example.sample_id}")
            boundaries[row["sample_id"]] = int(row["boundary_step"])
        dataset = build_surface_dataset(
            pairs, text_field="boundary_step", boundary_step_by_sample_id=boundaries,
        )
        dataset_manifest = surface_dataset_manifest(dataset)
        input_hash = dataset_manifest["rows_hash"]
        if input_hash not in cache:
            print(f"[{model_id}] fitting boundary-word TF-IDF on saved folds", flush=True)
            cache[input_hash] = evaluate_tfidf_grouped_cv(
                pairs, SPEC, seed=seed, n_splits=n_splits, bootstrap_resamples=bootstrap,
                split_manifest=split, analysis_group_by_pair_group=analysis_groups,
                analysis_group_definition=grouping["definition"],
                boundary_step_by_sample_id=boundaries,
            )
        result = cache[input_hash]
        text_rows = result["oof_predictions"]
        labels = np.asarray([row["label"] == "corrupt" for row in text_rows])
        text_scores = np.asarray([row["score"] for row in text_rows])
        groups = [row["analysis_group_id"] for row in text_rows]
        common = {
            "model_id": model_id, "display_name": model["display_name"],
            "protocol_grade": model["protocol_grade"],
            "recapture_protocol": model["recapture_protocol"],
        }
        ci = result["group_bootstrap_ci"]
        summaries.append({
            **common, "baseline": SPEC.name, "n_rows": result["n_rows"],
            "n_pairs": result["n_pairs"], "n_analysis_groups": result["n_analysis_groups"],
            "micro_oof_auroc": result["micro_oof_auroc"], "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"], "mean_fold_auroc": result["mean_fold_auroc"],
            "split_manifest_hash": split["manifest_hash"], "input_rows_hash": input_hash,
        })
        predictions.extend({**common, **row} for row in text_rows)
        identities.extend({
            **common, **row, "content_hash": examples[row["sample_id"]].content_hash,
            "analysis_group_id": groups[index], "fold_index": text_rows[index]["fold_index"],
        } for index, row in enumerate(dataset_manifest["rows"]))
        for anchor, layer in model["anchors"].items():
            router_rows = [row for row in router_oof
                           if row["model_id"] == model_id and row["anchor"] == anchor]
            assert_oof_identity(text_rows, router_rows)
            if {row["layer"] for row in router_rows} != {layer}:
                raise ValueError(f"router layer mismatch: {model_id}/{anchor}")
            router_scores = np.asarray([row["score"] for row in router_rows])
            auc = binary_auroc(labels, router_scores)
            canonical_row = old_summary[(model_id, anchor)]
            router_ci = group_bootstrap_metric_ci(
                labels, router_scores, groups, metric=binary_auroc, seed=seed,
                n_resamples=bootstrap,
            )
            errors = {
                "auroc_abs_error": abs(auc - float(canonical_row["micro_oof_auroc"])),
                "ci_low_abs_error": abs(router_ci["ci_low"] - float(canonical_row["ci_low"])),
                "ci_high_abs_error": abs(router_ci["ci_high"] - float(canonical_row["ci_high"])),
            }
            if max(errors.values()) > 1e-12:
                raise ValueError(f"canonical router metric failed: {model_id}/{anchor}: {errors}")
            checks.append({
                "model_id": model_id, "anchor": anchor, "layer": layer,
                "n_identical_rows_labels_groups_folds": len(text_rows),
                "canonical_auroc": float(canonical_row["micro_oof_auroc"]),
                "recomputed_auroc": auc, **errors,
            })
            comparison = paired_group_bootstrap_auroc_difference(
                labels, router_scores, text_scores, groups,
                left_name="router_logits", right_name="boundary_word_tfidf",
                seed=seed, n_resamples=bootstrap,
            )
            comparisons.append({
                **common, "anchor": anchor, "layer": layer, "n_rows": len(labels),
                "n_analysis_groups": len(set(groups)), "router_auroc": auc,
                "text_auroc": result["micro_oof_auroc"],
                "difference_definition": comparison["difference_definition"],
                "difference": comparison["difference"], "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
                "bootstrap_resamples": comparison["n_resamples"],
                "valid_bootstrap_resamples": comparison["n_valid_resamples"],
            })
        models.append({
            **common, "input_rows_hash": input_hash,
            "content_hashes_verified": len(rows),
            "boundary_step_by_sample_id": boundaries,
            "baseline": {key: value for key, value in result.items()
                         if key not in {"dataset_manifest", "split_manifest", "oof_predictions"}},
        })
    output.mkdir(parents=True)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "router_minus_text.csv", comparisons)
    write_csv(output / "canonical_reproduction.csv", checks)
    write_csv(output / "input_row_manifest.csv", identities)
    with (output / "oof_predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    payload = {
        "analysis_id": "revision_boundary_text_20260907", "schema_version": 1,
        "scope": {"device": "cpu", "checkpoint_loaded": False,
                  "validation_results_accessed": False, "source_archives_modified": False,
                  "source_records_normalized": len(pairs)},
        "fixed_baseline": SPEC.to_json(), "hyperparameter_search": False,
        "preprocessing": "Training-fold-only TF-IDF vocabulary, document frequency, and IDF; "
                         "L2-normalized TF-IDF; no additional StandardScaler.",
        "estimator_match": "Same L2 logistic C=1, balanced classes, liblinear solver, "
                           "max_iter=2000, and random_state=20260829 as router probes. "
                           "Text and activation feature maps differ; this is not capacity matched.",
        "source": {"path": str(source_path), "sha256": sha256(source_path),
                   "canonical_source_provenance": selection["source"],
                   "allowlist": str(groups_path), "pair_count": len(pairs)},
        "shared_folds": split, "grouping_limitation": grouping["limitation"],
        "unique_text_input_sets_fitted": len(cache), "models": models,
        "canonical_reproduction_checks": checks,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    **{package: importlib.metadata.version(package)
                       for package in ("numpy", "scipy", "scikit-learn", "pyarrow")}},
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in sorted(inputs)],
        "code": [{"path": str(path), "sha256": sha256(path)} for path in (
            Path(__file__).resolve(), repo / "src/routeverify/analysis/surface_baselines.py",
            repo / "src/routeverify/analysis/fair_comparison.py",
            repo / "src/routeverify/analysis/interface_access.py",
            repo / "src/routeverify/analysis/probes.py",
            repo / "src/routeverify/analysis/protocols.py",
            repo / "src/routeverify/data/prmbench.py", repo / "src/routeverify/data/schema.py",
        )],
    }
    write_json(output / "results.json", payload)
    lines = [
        "# Boundary-step text baseline on canonical interface-access folds", "",
        "A single fixed word unigram/bigram TF-IDF + L2-logistic baseline was trained "
        "with vocabulary and IDF fitted only on each training fold. No tuning was run. "
        "The classifier matches the router probe's C=1, balanced classes, liblinear, "
        "max_iter=2000, and random state; feature preprocessing is TF-IDF rather than "
        "activation StandardScaler, so the representation dimensions/capacity are not matched.", "",
        "Each comparison uses the identical 580 rows (290 clean, 290 corrupt), 290 discovery "
        "pairs, 235 source components, and all five saved folds. The positive class is corrupt. "
        "All 18 router-anchor OOF AUROCs and source-component 95% intervals were reproduced "
        "to <1e-12. Every source/recapture content hash was verified. The full public-source "
        "Arrow cache was filtered by frozen discovery source IDs before text normalization; "
        "the 20-pair processed pilot was not used.", "",
        "Intervals resample the same 235 source components, with 1,000 paired draws and "
        "seed 20260829. They condition on fitted OOF predictions and do not retrain models "
        "inside each bootstrap. Differences are router AUROC minus text AUROC.", "",
        "| model | text OOF AUROC [95% CI] | middle router AUROC | router minus text [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for summary in summaries:
        comp = next(row for row in comparisons
                    if row["model_id"] == summary["model_id"] and row["anchor"] == "middle")
        lines.append(
            f"| {summary['display_name']} | {summary['micro_oof_auroc']:.3f} "
            f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}] | {comp['router_auroc']:.3f} | "
            f"{comp['difference']:+.3f} [{comp['ci_low']:+.3f}, {comp['ci_high']:+.3f}] |"
        )
    lines.extend([
        "", "Qwen3 uses its archived historical sample-level boundaries and remains a "
        "retrospective sensitivity; it is not pooled with the five paired-first-error models. "
        "OLMoE has no archived recapture manifest and is not imputed. Identical text inputs "
        "and folds are fitted once and reused, not treated as independent text experiments.", "",
        "The baseline diagnoses how well this fixed surface model distinguishes clean/corrupt "
        "boundary steps. A router advantage over it does not establish correctness-specific "
        "semantics or eliminate all textual confounds: router states have contextual/model "
        "features unavailable to this single-step word baseline. This is a post-hoc discovery "
        "comparison, not prospective confirmation or a causal intervention.", "",
        "Reproduce from the repository root into a fresh directory:", "", "```bash",
        "PYTHONPATH=src .venv/bin/python "
        "docs/results/revision_cpu_20260907/text_baseline.py "
        "--output build/text-baseline",
        "```", "", "Dependencies: numpy, scipy, scikit-learn, pyarrow, pyyaml. "
        "All input paths, hashes, versions, fold memberships, boundary indices and definitions "
        "are in results.json; input_row_manifest.csv stores only hashes/IDs, not source prose.",
    ])
    (output / "evidence_notes.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "unique_text_inputs": len(cache),
                      "models": len(models), "canonical_checks": len(checks)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path("archive://"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source-arrow", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "text_baseline")
    args = parser.parse_args()
    run(args.archive_root.resolve(), args.repo_root.resolve(), args.output.resolve(),
        None if args.source_arrow is None else args.source_arrow.resolve())
