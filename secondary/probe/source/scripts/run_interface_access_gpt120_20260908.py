#!/usr/bin/env python3
"""GPT120 recapture-only extension of the saved canonical CPU probe protocol.

No model imports or forward passes; only the three specified safetensor layers
are read. Existing source components/folds and estimators are reused unchanged.
"""
# Thread limits and the repository import path must be set before numerical imports.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

# Set before NumPy/sklearn imports, including on machines with a high default.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from routeverify.analysis.fair_comparison import paired_group_bootstrap_auroc_difference
from routeverify.analysis.interface_access import (
    PRIMARY_INTERFACE_CHAIN,
    assert_row_identity_matches,
    canonicalize_common_rows,
    compact_probe_result,
    evaluate_feature_families,
    extract_layer_features,
    load_jsonl_rows,
    load_selection_source_ids,
)
from routeverify.analysis.surface_baselines import validate_grouped_kfold_manifest


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/analysis/interface_access_gpt120_20260908.yaml")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = args.output or ROOT / config["output"]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    constraints = config["constraints"]
    assert constraints == dict(device="cpu", threads=2, allow_model_weight_loading=False,
                               allow_validation_access=False, allow_source_archive_writes=False)
    protocol_path = ROOT / config["canonical_protocol"]
    protocol = yaml.safe_load(protocol_path.read_text())["protocol"]
    assert (protocol["seed"], protocol["folds"], protocol["bootstrap_resamples"]) == (20260829, 5, 1000)
    assert protocol["estimator"] == dict(implementation="sklearn.linear_model.LogisticRegression",
        penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000,
        preprocessing="StandardScaler_fit_on_training_fold_only")
    fold_path = ROOT / config["canonical_folds"]
    assert sha(fold_path) == config["canonical_folds_sha256"]
    frozen = json.loads(fold_path.read_text())
    split = frozen["split_manifest"]
    assert split["manifest_hash"] == config["expected_split_manifest_hash"]
    source_ids = {x["pair_id"]: x["source_id"] for x in frozen["pairs"]}
    group_ids = {x["pair_id"]: x["analysis_group_id"] for x in frozen["pairs"]}
    validate_grouped_kfold_manifest(split, expected_groups=set(group_ids.values()),
                                   expected_seed=protocol["seed"], expected_n_splits=5)
    assert len(group_ids) == 290 and len(set(group_ids.values())) == 235
    capture = Path(config["recapture_root"])
    reference = Path(config["reference_recapture_root"])
    capture_sources, selection = load_selection_source_ids(capture / "selection_manifest.json")
    reference_sources, reference_selection = load_selection_source_ids(reference / "selection_manifest.json")
    assert source_ids == capture_sources == reference_sources
    assert selection["pairs_hash"] == reference_selection["pairs_hash"]
    assert selection["pairs"] == reference_selection["pairs"]
    assert selection["frozen_split"]["partition"] == "discovery"
    raw_rows = load_jsonl_rows(capture / "data_manifest.jsonl")
    assert len(raw_rows) == 580
    rows, groups = canonicalize_common_rows(raw_rows, pair_source_ids=source_ids,
                                           pair_analysis_group_ids=group_ids)
    ref_rows, ref_groups = canonicalize_common_rows(load_jsonl_rows(reference / "data_manifest.jsonl"),
        pair_source_ids=source_ids, pair_analysis_group_ids=group_ids)
    assert groups == ref_groups
    assert_row_identity_matches(ref_rows, rows)
    fields = ("sample_id", "pair_id", "group_id", "variant", "boundary_step", "error_type",
              "content_hash", "prompt_mode", "input_format", "token_position", "prompt_hash", "input_ids_hash")
    identity_counts = {field: sum(a.get(field) == b.get(field) for a, b in zip(rows, ref_rows, strict=True))
                       for field in fields}
    for field in ("content_hash", "prompt_mode", "input_format"):
        assert identity_counts[field] == 580, field
    selection_by_pair = {x["pair_id"]: x for x in selection["pairs"]}
    for row in rows:
        pair = selection_by_pair[row["pair_id"]]
        assert row["content_hash"] == pair[f"{row['variant']}_content_hash"]
        assert row["sample_id"] == pair[f"{row['variant']}_sample_id"]
        assert row["boundary_strategy"] == "paired_first_error"
        assert row["prompt_mode"] == "step_binary" and row["input_format"] == "chat"
        assert row["num_positions_in_file"] == 1 and row["token_index"] == 0
        if row["variant"] == "corrupt":
            assert row["is_error_boundary"] is True and row["first_error"] == row["boundary_step"]
        else:
            assert row["is_all_correct"] is True and row["error_steps"] == []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        assert a["pair_id"] == b["pair_id"] and a["boundary_step"] == b["boundary_step"]
    resolved_path = capture / "config.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    assert resolved["split_partition"] == "discovery"
    assert resolved["layers"] == list(range(36))
    assert resolved["sites"] == ["post_moe_residual", "router_input", "router_logits", "router_topk_ids", "router_topk_weights"]
    assert config["anchors"] == dict(early=0, middle=18, late=35)
    discovery = json.loads((capture / "model_discovery.json").read_text())
    assert discovery["metadata"]["runtime_inspect"]["actual_contract"] == dict(layers=36, experts=128, top_k=4)
    assert discovery["summary"] == dict(experts=36, router=36)
    for row in rows:
        assert len(row["arrays"]) == 180
        for layer in config["anchors"].values():
            for feature, dim in (("post_moe_residual",2880), ("router_input",2880),
                                 ("router_logits",128), ("router_topk_ids",4), ("router_topk_weights",4)):
                assert row["arrays"][f"layer_{layer}.{feature}"] == [1, 1, dim]
    fold_by_group = {group: f["fold_index"] for f in split["folds"] for group in f["eval_group_ids"]}
    identities = [{k: row.get(k) for k in (*fields, "boundary_strategy", "boundary_match_role",
                  "first_error", "is_all_correct", "is_error_boundary", "num_positions_in_file", "token_index")}
                  | dict(analysis_group_id=group, fold_index=fold_by_group[group],
                         activation_file=Path(row["activation_file"]).name)
                  for row, group in zip(rows, groups, strict=True)]
    write_json(out / "row_identity.json", identities)
    write_json(out / "source_component_folds.json", frozen)
    validation = dict(status="passed", n_rows=580, n_pairs=290, n_source_components=235,
        classes={role: sum(r["variant"] == role for r in rows) for role in ("clean", "corrupt")},
        anchors=config["anchors"], reference_identity_equal_counts=identity_counts,
        selection_pairs_exactly_equal=True, boundary_role_content_checks=True,
        source_disjoint_folds=True, saved_fold_manifest_reused=True,
        split_manifest_hash=split["manifest_hash"], training_fold_only_scaling=True,
        labels="manifest variant only; corrupt=positive, clean=negative",
        feature_scope="Only captured numerical tensors; no labels, identity, boundary or text columns in X",
        prompt_scope="Prompt-only final position; no completion/answer continuation captured",
        rank_weights="legacy secondary; expert-ID membership and aligned weights preserved")
    write_json(out / "validation.json", validation)
    print("Preflight passed: 580 rows / 290 pairs / 235 groups; frozen folds verified", flush=True)

    inputs = [args.config, protocol_path, fold_path, capture / "data_manifest.jsonl",
              capture / "selection_manifest.json", resolved_path, capture / "model_discovery.json",
              capture / "environment.json", reference / "data_manifest.jsonl",
              reference / "selection_manifest.json", reference / "config.resolved.yaml",
              ROOT / config["reference_oof"], ROOT / config["reference_results"]]
    code = [Path(__file__), ROOT / "scripts/run_interface_access_analysis.py"] + [
        ROOT / "src/routeverify/analysis" / name for name in
        ("interface_access.py", "fair_comparison.py", "probes.py", "protocols.py", "routing_probes.py", "surface_baselines.py")]
    provenance = dict(analysis_id=config["analysis_id"], input_snapshot_sha256=config["input_snapshot_sha256"],
        worktree=str(ROOT), command=[sys.executable, *sys.argv], constraints=constraints,
        inputs={str(p.resolve()): sha(p) for p in inputs}, code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},
        runtime=dict(python=sys.version, platform=platform.platform(), packages={
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "safetensors", "scipy", "PyYAML", "threadpoolctl")}),
        capture_path_resolution="Existing extractor resolves archived activation paths by basename under recapture/activations",
        tensor_conversion="No format conversion; existing safe_open(framework=np), selected layer keys only, float32 feature matrices",
        capture_hash_scope="Existing archive acceptance reused; no repeat whole-payload hashes; manifests plus consumed feature-matrix digests recorded",
        feature_digests={}, routing_weight_checks={}, warnings=[])
    summaries, differences, oof, anchors = [], [], [], {}
    base = {k: config[k] for k in ("model_id", "display_name", "protocol_grade", "recapture_protocol")}
    with threadpool_limits(limits=2):
        provenance["runtime"]["threadpools"] = threadpool_info()
        for anchor, layer in config["anchors"].items():
            print(f"Extracting and fitting {anchor} L{layer}", flush=True)
            features = extract_layer_features(rows, recapture_root=capture, layer=layer, num_experts=128)
            membership = features["expert_id_membership_count"]
            weights = features["expert_id_aligned_sparse_weights"]
            assert np.all((membership == 0) | (membership == 1)) and np.all(membership.sum(axis=1) == 4)
            # Weights were computed in the archived BF16 runtime, then stored as
            # float32; preserve their original rounded values without renormalizing.
            sum_error = float(np.max(np.abs(weights.sum(axis=1) - 1)))
            assert np.all(weights >= 0) and sum_error <= 2**-8
            assert np.all(weights[membership == 0] == 0)
            provenance["routing_weight_checks"][anchor] = dict(max_abs_sum_error=sum_error,
                tolerance=2**-8, renormalized=False, reason="archived BF16-rounded weights")
            provenance["feature_digests"][anchor] = {name:dict(shape=list(x.shape), dtype=str(x.dtype),
                sha256=hashlib.sha256(x.tobytes(order="C")).hexdigest()) for name,x in features.items()}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                probes, adjacent = evaluate_feature_families(features, rows,
                    analysis_group_ids=groups, split_manifest=split, analysis_group_definition=frozen["definition"],
                    seed=20260829, n_splits=5, bootstrap_resamples=1000, logistic_c=1)
            provenance["warnings"].extend(dict(anchor=anchor, category=w.category.__name__, message=str(w.message)) for w in caught)
            if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
                raise RuntimeError("Logistic probe did not converge; retain partial outputs for diagnosis")
            for name, probe in probes.items():
                assert probe["split_manifest_hash"] == split["manifest_hash"]
                for prediction in probe["oof_predictions"]:
                    assert prediction["fold_index"] == fold_by_group[prediction["analysis_group_id"]]
                    oof.append(base | dict(anchor=anchor, layer=layer, feature_family=name) | prediction)
                ci = probe["group_bootstrap_ci"]
                summaries.append(base | dict(anchor=anchor, layer=layer, feature_family=name,
                    feature_dim=probe["feature_dim"], n_rows=580, n_pairs=290, n_analysis_groups=235,
                    micro_oof_auroc=probe["micro_oof_auroc"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                    mean_fold_auroc=probe["mean_fold_auroc"], split_manifest_hash=split["manifest_hash"]))
            direct = []
            for left, right in itertools.combinations(PRIMARY_INTERFACE_CHAIN, 2):
                known = next((c for c in adjacent if c["left"]["name"] == left and c["right"]["name"] == right), None)
                c = known or paired_group_bootstrap_auroc_difference(
                    np.array([r["variant"] == "corrupt" for r in rows]),
                    np.array([r["score"] for r in probes[left]["oof_predictions"]]),
                    np.array([r["score"] for r in probes[right]["oof_predictions"]]), groups,
                    left_name=left, right_name=right, seed=20260829, n_resamples=1000)
                direct.append(c)
                differences.append(base | dict(anchor=anchor, layer=layer, left_feature=left, right_feature=right,
                    left_auroc=c["left"]["auroc"], right_auroc=c["right"]["auroc"], difference=c["difference"],
                    ci_low=c["ci_low"], ci_high=c["ci_high"], n_analysis_groups=235,
                    contrast_scope="canonical_adjacent" if known else "supplemental_direct"))
            anchors[anchor] = dict(layer=layer, features={k:compact_probe_result(v) for k,v in probes.items()},
                                   adjacent_interface_differences=adjacent, all_primary_interface_differences=direct)
            write_csv(out / "interface_access_summary.csv", summaries)
            write_csv(out / "interface_access_direct_differences.csv", differences)
            with (out / "interface_access_oof_predictions.jsonl").open("w") as f:
                for row in oof:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
            write_json(out / "interface_access_results.json", dict(schema_version=1, analysis_id=config["analysis_id"],
                status="ok" if len(anchors) == 3 else "in_progress", protocol=protocol,
                ci_version="native_probe_1000_seed20260829_source_component_percentile_v1",
                common_selection=dict(n_pairs=290, n_rows=580, n_analysis_groups=235,
                    selection_pairs_hash=selection["pairs_hash"], split_manifest_hash=split["manifest_hash"]),
                models=[base | dict(status="ok", anchors=anchors)]))
            provenance["elapsed_seconds"] = time.time() - started
            write_json(out / "analysis_provenance.json", provenance)
            print(f"Completed {anchor} L{layer}: " + ", ".join(f"{n}={p['micro_oof_auroc']:.6f}" for n,p in probes.items()), flush=True)
    validation["oof_rows"] = len(oof)
    validation["same_fold_for_every_feature"] = True
    validation["expert_presence_binary_top4"] = True
    validation["aligned_weights_on_expert_ids_and_unit_sum_with_bf16_rounding"] = True
    validation["convergence_warnings"] = 0
    write_json(out / "validation.json", validation)
    print(f"Completed all 18 panels and 30 direct differences in {time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
