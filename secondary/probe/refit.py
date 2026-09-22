#!/usr/bin/env python3
"""Refit the eight-model interface probes from bundled boundary features on CPU.

The archived numerical implementation performs the fits. By default this command
also recomputes each cell's 1,000-draw interval. Use --fit-only to check fitted OOF
scores and AUROCs without repeating the published uncertainty calculation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_variable] = "2"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source" / "src"))

import numpy as np
from routeverify.analysis.interface_access import INTERFACE_FEATURE_FAMILIES
from routeverify.analysis.routing_probes import routing_feature_vectors

NATIVE_SITES = (
    "post_moe_residual", "router_input", "router_logits",
    "router_topk_ids", "router_topk_weights",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def matrix_identity(matrix: np.ndarray) -> dict:
    return {
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    }


def features_from_native(arrays, *, layer: int, num_experts: int) -> dict[str, np.ndarray]:
    """Recreate the original six float32 X matrices without rounding/rescaling."""
    native = {site: arrays[f"layer_{layer}.{site}"] for site in NATIVE_SITES}
    n_rows = native["router_logits"].shape[0]
    for site, matrix in native.items():
        if matrix.ndim != 2 or matrix.shape[0] != n_rows:
            raise ValueError(f"Invalid native feature shape: {site}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Non-finite native features: {site}")
    if native["router_logits"].shape[1] != num_experts:
        raise ValueError("Router logits do not match expert count")
    values = {name: [] for name in INTERFACE_FEATURE_FAMILIES[3:]}
    for ids, weights in zip(native["router_topk_ids"], native["router_topk_weights"], strict=True):
        routing = routing_feature_vectors(ids, weights, num_experts=num_experts)
        for name, vector in routing.items():
            values[name].append(vector)
    features = {
        name: native[name].astype(np.float32, copy=False)
        for name in INTERFACE_FEATURE_FAMILIES[:3]
    }
    features.update({name: np.stack(vectors) for name, vectors in values.items()})
    return features


def load_reference(path: Path, model_id: str) -> dict:
    references = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["model_id"] != model_id:
                continue
            key = (row["anchor"], row["feature_family"], row["sample_id"])
            if key in references:
                raise ValueError(f"Duplicate saved OOF identity: {key}")
            references[key] = row
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Optional model IDs from features/index.json")
    parser.add_argument("--fit-only", action="store_true", help="Skip interval recomputation; still fit all selected cells and check OOF scores")
    parser.add_argument("--strict-reference-match", action="store_true", help="Exit with an error if completed fits do not match saved OOF scores within the reported strict tolerances")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    from routeverify.analysis.fair_comparison import evaluate_dense_logistic_grouped_cv
    from routeverify.analysis.probes import binary_auroc
    from routeverify.analysis.surface_baselines import validate_grouped_kfold_manifest

    started = time.monotonic()
    index = json.loads((ROOT / "features/index.json").read_text())
    chosen = set(args.models or [m["model_id"] for m in index["models"]])
    known = {m["model_id"] for m in index["models"]}
    if chosen - known:
        parser.error(f"Unknown model IDs: {sorted(chosen - known)}")
    fold_path = ROOT / index["folds"]["path"]
    if sha256(fold_path) != index["folds"]["sha256"]:
        raise ValueError("Bundled fold file changed")
    frozen = json.loads(fold_path.read_text())
    split = frozen["split_manifest"]
    pair_groups = {p["pair_id"]: p["analysis_group_id"] for p in frozen["pairs"]}
    fold_by_group = {g: f["fold_index"] for f in split["folds"] for g in f["eval_group_ids"]}
    validate_grouped_kfold_manifest(split, expected_groups=set(pair_groups.values()), expected_seed=20260829, expected_n_splits=5)

    summaries, score_differences, auc_differences, problems = [], [], [], []
    total_predictions = 0
    runtime = {package: importlib.metadata.version(package) for package in (
        "numpy", "scikit-learn", "scipy", "threadpoolctl", "pydantic",
    )}
    with threadpool_limits(limits=2), gzip.open(args.output / "interface_access_oof_predictions.jsonl.gz", "wt") as output:
        for model in index["models"]:
            model_id = model["model_id"]
            if model_id not in chosen:
                continue
            for member in ("bundle", "rows", "reference_oof"):
                if sha256(ROOT / model[member]["path"]) != model[member]["sha256"]:
                    raise ValueError(f"Changed {model_id} {member}")
            rows = [json.loads(line) for line in (ROOT / model["rows"]["path"]).read_text().splitlines()]
            sample_ids = [r["sample_id"] for r in rows]
            if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != 580:
                raise ValueError("Expected 580 unique rows in recorded sorted sample order")
            groups = [r["analysis_group_id"] for r in rows]
            for row in rows:
                if row["analysis_group_id"] != pair_groups[row["pair_id"]] or row["fold_index"] != fold_by_group[row["analysis_group_id"]]:
                    raise ValueError("Row grouping or fold differs from canonical manifest")
                if row["label"] != row["variant"] or row["label"] not in {"clean", "corrupt"}:
                    raise ValueError("Invalid label")
            labels = np.asarray([row["variant"] == "corrupt" for row in rows])
            if int(labels.sum()) != 290 or len(set(groups)) != 235:
                raise ValueError("Unexpected class or component counts")
            references = load_reference(ROOT / model["reference_oof"]["path"], model_id)
            with np.load(ROOT / model["bundle"]["path"], allow_pickle=False) as arrays:
                if set(arrays.files) != set(model["native_arrays"]):
                    raise ValueError("Unexpected arrays in NPZ bundle")
                for name, expected in model["native_arrays"].items():
                    if matrix_identity(arrays[name]) != expected:
                        raise ValueError(f"Changed native array: {model_id}/{name}")
                for anchor, metadata in model["anchors"].items():
                    layer = metadata["layer"]
                    features = features_from_native(arrays, layer=layer, num_experts=model["num_experts"])
                    for family in INTERFACE_FEATURE_FAMILIES:
                        matrix = features[family]
                        if matrix_identity(matrix) != metadata["feature_digests"][family]:
                            raise ValueError(f"Feature reconstruction changed: {model_id}/{anchor}/{family}")
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            # The unmodified archived evaluator requires a positive
                            # bootstrap count. In fit-only mode its one-draw interval
                            # is discarded and is never reported as uncertainty.
                            fitted = evaluate_dense_logistic_grouped_cv(
                                matrix, labels, sample_ids=sample_ids,
                                pair_group_ids=[r["group_id"] for r in rows],
                                analysis_group_ids=groups, split_manifest=split,
                                analysis_group_definition=frozen["definition"],
                                seed=20260829, n_splits=5,
                                bootstrap_resamples=1 if args.fit_only else 1000,
                                logistic_c=1.0,
                            )
                        if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
                            raise RuntimeError(f"Probe failed to converge: {model_id}/{anchor}/{family}")
                        saved_scores, new_scores = [], []
                        identity = dict(model_id=model_id, anchor=anchor, layer=layer, feature_family=family)
                        for prediction in fitted["oof_predictions"]:
                            saved = references[(anchor, family, prediction["sample_id"])]
                            for field in ("sample_id", "group_id", "analysis_group_id", "label", "fold_index"):
                                if saved[field] != prediction[field]:
                                    raise ValueError(f"OOF row identity changed: {field}")
                            if saved["layer"] != layer:
                                raise ValueError("OOF anchor layer changed")
                            saved_scores.append(saved["score"])
                            new_scores.append(prediction["score"])
                            output.write(json.dumps(identity | prediction, sort_keys=True) + "\n")
                        total_predictions += len(new_scores)
                        score_error = float(np.max(np.abs(np.asarray(saved_scores) - new_scores)))
                        saved_auc = binary_auroc(labels, np.asarray(saved_scores))
                        auc_error = abs(saved_auc - fitted["micro_oof_auroc"])
                        score_differences.append(score_error)
                        auc_differences.append(auc_error)
                        # Float64 BLAS/solver differences across operating systems
                        # may change final score bits. AUC tolerance allows one of
                        # the 290*290 positive-negative pairs to exchange rank.
                        passed = bool(np.allclose(new_scores, saved_scores, rtol=1e-5, atol=1e-7) and auc_error <= 1 / (290 * 290) + 1e-15)
                        row = identity | dict(feature_dim=fitted["feature_dim"], n_rows=580,
                            n_pairs=290, n_analysis_groups=235, micro_oof_auroc=fitted["micro_oof_auroc"],
                            saved_micro_oof_auroc=saved_auc, mean_fold_auroc=fitted["mean_fold_auroc"],
                            max_abs_score_difference=score_error, abs_auroc_difference=auc_error,
                            comparison_passed=passed)
                        if not args.fit_only:
                            row.update({k: fitted["group_bootstrap_ci"][k] for k in ("ci_low", "ci_high")})
                        summaries.append(row)
                        if not passed:
                            problems.append(identity | dict(max_abs_score_difference=score_error, abs_auroc_difference=auc_error))
                    print(f"Refitted {model_id} {anchor} L{layer}: six families / 30 fits", flush=True)
    with (args.output / "interface_access_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    validation = dict(
        status="completed", reference_comparison="matched" if not problems else "different",
        models=sorted(chosen), cells=len(summaries),
        fitted_classifiers=len(summaries) * 5, oof_predictions=total_predictions,
        strict_reference_matched_cells=sum(r["comparison_passed"] for r in summaries),
        exactly_matching_score_cells=sum(r["max_abs_score_difference"] == 0 for r in summaries),
        max_abs_score_difference=max(score_differences), max_abs_auroc_difference=max(auc_differences),
        score_comparison=dict(rtol=1e-5, atol=1e-7), auroc_comparison_atol=1 / (290 * 290) + 1e-15,
        bootstrap_resamples=None if args.fit_only else 1000, bootstrap_seed=None if args.fit_only else 20260829,
        limitations="Only the eight-model, 580-row canonical interface panel. Excludes the 238-group sensitivity, historical 1160-row Qwen comparison and text baselines. No model forwards or GPU recapture.",
        runtime=runtime, python=platform.python_version(), platform=platform.platform(), threads=2,
        elapsed_seconds=time.monotonic() - started, reference_differences=problems,
    )
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2))
    if problems and args.strict_reference_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
