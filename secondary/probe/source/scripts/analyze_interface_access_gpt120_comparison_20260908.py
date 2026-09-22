#!/usr/bin/env python3
"""Audit archived GPT prompts and compare saved CPU OOF AUROCs without refitting."""

# Set thread limits and the repository import path before importing numerical libraries.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _thread_variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "2"

import numpy as np

from routeverify.analysis.fair_comparison import paired_group_bootstrap_auroc_difference
from routeverify.analysis.interface_access import INTERFACE_FEATURE_FAMILIES
from routeverify.analysis.surface_baselines import validate_grouped_kfold_manifest
from routeverify.data.schema import stable_hash

OUTPUT = ROOT / "docs/results/interface_access_gpt120_20260908"
OLD_RESULTS = ROOT / "docs/results/interface_access_canonical_common_20260829"
OLD_RECAPTURE = Path(
    "archive://runpod_gpt_oss_full_20260809T110315Z_complete/"
    "routeverify-runs/gpt_oss_full_20260809T110315Z/recapture"
)
NEW_RECAPTURE = Path(
    "archive://campaign_C/artifacts/"
    "campaign_20260907_C/gpt_oss_120b_common_core_full_v1/recapture"
)
SEED = 20260829
BOOTSTRAP_RESAMPLES = 1000
ANCHORS = {"early": (0, 0), "middle": (18, 12), "late": (35, 23)}
SEMANTIC_FIELDS = (
    "sample_id", "pair_id", "group_id", "variant", "content_hash", "boundary_step",
    "boundary_strategy", "boundary_match_role", "paired_corrupt_sample_id",
    "paired_group_id", "error_type", "error_steps", "first_error", "is_all_correct",
    "is_error_boundary", "input_format", "prompt_mode", "token_position", "token_index",
    "num_positions_in_file", "dataset", "subset",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _recapture(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapped = {row["sample_id"]: row for row in rows}
    _require(len(rows) == len(mapped) == 580, f"recapture must contain 580 unique rows: {path}")
    _require(Counter(row["variant"] for row in rows) == {"clean": 290, "corrupt": 290},
             "recapture class balance differs")
    return mapped


def _prompt_comparability(old_root: Path, new_root: Path, folds_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_folds = _json(folds_path)
    unhashed = dict(source_folds)
    observed_hash = unhashed.pop("manifest_hash")
    _require(stable_hash(unhashed, length=64) == observed_hash, "source-component manifest hash invalid")
    pair_groups = {row["pair_id"]: row["analysis_group_id"] for row in source_folds["pairs"]}
    _require(len(pair_groups) == 290 and len(set(pair_groups.values())) == 235,
             "expected 290 pairs and 235 canonical source groups")
    split = source_folds["split_manifest"]
    validate_grouped_kfold_manifest(split, expected_groups=set(pair_groups.values()),
                                   expected_seed=SEED, expected_n_splits=5)
    expected_ids = {f"{pair}/{role}" for pair in pair_groups for role in ("clean", "corrupt")}
    captures: dict[str, dict[str, Any]] = {}
    prompts: dict[str, dict[str, Any]] = {}
    selections: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    inputs = [folds_path]
    rendered_dates: dict[str, dict[str, int]] = {}
    for model, recapture_root in (("gpt20", old_root), ("gpt120", new_root)):
        recapture_path = recapture_root / "data_manifest.jsonl"
        prompt_path = recapture_root.parent / "router/data_manifest.json"
        selection_path = recapture_root / "selection_manifest.json"
        inputs.extend([recapture_path, prompt_path, selection_path])
        captures[model] = _recapture(recapture_path)
        _require(set(captures[model]) == expected_ids, f"{model} recapture differs from canonical pair IDs")
        selection = _json(selection_path)
        selections[model] = selection
        _require(selection["frozen_split"]["partition"] == "discovery", "non-discovery recapture")
        selection_pairs = {row["pair_id"]: row for row in selection["pairs"]}
        _require(len(selection_pairs) == 290 and set(selection_pairs) == set(pair_groups),
                 "selection pair allowlist differs")
        prompt_rows = _json(prompt_path)["prompt_inputs"]
        _require(len(prompt_rows) == 290, "expected 290 paired prompt records")
        prompts[model] = {f"{row['pair_id']}/{role}": row[role]
                          for row in prompt_rows for role in ("clean", "corrupt")}
        _require(set(prompts[model]) == expected_ids, "prompt records differ from capture sample IDs")
        for sid, prompt in prompts[model].items():
            captured = captures[model][sid]
            checks = {
                "rendered_text_hash_matches": stable_hash({"prompt": prompt["rendered_prompt_text"]}, length=64)
                == prompt["rendered_prompt_hash"],
                "recapture_prompt_prefix_matches": prompt["rendered_prompt_hash"][:32] == captured["prompt_hash"],
                "recapture_input_prefix_matches": prompt["input_ids_hash"][:32] == captured["input_ids_hash"],
                "captured_role_matches_id": sid == f"{captured['pair_id']}/{captured['variant']}",
                "prompt_only_single_position": captured["num_positions_in_file"] == 1 and captured["token_index"] == 0,
                "protocol_matches": captured["input_format"] == "chat" and captured["prompt_mode"] == "step_binary"
                and captured["boundary_strategy"] == "paired_first_error",
            }
            for check, passed in checks.items():
                _require(passed, f"{model} {sid}: {check} failed")
                counts[f"{model}_{check}"] += 1
        rendered_dates[model] = dict(Counter(row["adapter_render"]["rendered_date"]
                                              for row in prompts[model].values()))
    _require(selections["gpt20"]["pairs"] == selections["gpt120"]["pairs"], "selection pair provenance differs")
    _require(selections["gpt20"]["pairs_hash"] == selections["gpt120"]["pairs_hash"], "selection pairs_hash differs")
    semantic_differences = {field: 0 for field in SEMANTIC_FIELDS}
    for sid in sorted(expected_ids):
        a, b = prompts["gpt20"][sid], prompts["gpt120"][sid]
        old_capture, new_capture = captures["gpt20"][sid], captures["gpt120"][sid]
        for field in SEMANTIC_FIELDS:
            semantic_differences[field] += old_capture.get(field) != new_capture.get(field)
        old_text, new_text = a["rendered_prompt_text"], b["rendered_prompt_text"]
        checks = {
            "raw_prompt_hash_same": a["raw_prompt_hash"] == b["raw_prompt_hash"],
            "header_date_only_difference": old_text.count("Current date: 2026-08-09\n") == 1
            and new_text.count("Current date: 2026-09-08\n") == 1
            and old_text.replace("Current date: 2026-08-09\n", "Current date: 2026-09-08\n", 1) == new_text,
            "template_sha_same": a["adapter_render"]["chat_template_sha256"] == b["adapter_render"]["chat_template_sha256"],
            "reasoning_effort_same": a["adapter_render"]["reasoning_effort"] == b["adapter_render"]["reasoning_effort"] == "medium",
            "add_special_tokens_same": a["add_special_tokens"] == b["add_special_tokens"],
            "tokenizer_class_same": a["tokenizer_class"] == b["tokenizer_class"],
        }
        for check, passed in checks.items():
            _require(passed, f"prompt comparison {sid}: {check} failed")
            counts[check] += 1
        counts["recapture_prompt_hash_different"] += old_capture["prompt_hash"] != new_capture["prompt_hash"]
        counts["recapture_input_ids_hash_different"] += old_capture["input_ids_hash"] != new_capture["input_ids_hash"]
    _require(not any(semantic_differences.values()), "recapture semantic identity differs")
    payload = {
        "schema_version": 1, "analysis_id": "gpt120_gpt20_prompt_comparability_20260908",
        "status": "ok", "n_pairs": 290, "n_rows": 580, "n_analysis_groups": 235,
        "split_manifest_hash": split["manifest_hash"],
        "selection_pairs_hash": selections["gpt120"]["pairs_hash"],
        "checks_passed_counts": dict(counts), "semantic_field_difference_counts": semantic_differences,
        "rendered_dates": rendered_dates,
        "interpretation": "All 580 rendered prompts differ only in the Harmony system-header date: "
        "GPT-OSS-20B 2026-08-09 versus GPT-OSS-120B 2026-09-08. Raw prompts, paired content, "
        "boundaries, roles, and token positions match. These are not identical rendered/token inputs.",
        "inputs": [_file_record(path) for path in inputs],
        "full_prompt_text_copied_to_output": False,
    }
    return payload, {"split": split, "pair_groups": pair_groups, "captures": captures,
                     "sample_ids": sorted(expected_ids)}


def _load_oof(path: Path, model_id: str, context: dict[str, Any], model_key: str) -> dict[Any, Any]:
    grouped: dict[Any, dict[str, Any]] = defaultdict(dict)
    expected_ids = set(context["sample_ids"])
    group_fold = {group: fold["fold_index"] for fold in context["split"]["folds"]
                  for group in fold["eval_group_ids"]}
    selected_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model_id"] != model_id:
                continue
            selected_rows += 1
            sid, anchor, feature = row["sample_id"], row["anchor"], row["feature_family"]
            _require(sid in expected_ids and anchor in ANCHORS and feature in INTERFACE_FEATURE_FAMILIES,
                     "unexpected OOF row identity")
            expected_layer = ANCHORS[anchor][0 if model_key == "gpt120" else 1]
            _require(row["layer"] == expected_layer, "OOF layer differs from relative anchor")
            captured = context["captures"][model_key][sid]
            group = context["pair_groups"][captured["pair_id"]]
            _require(row["analysis_group_id"] == group and row["fold_index"] == group_fold[group],
                     "OOF source group or fold mismatch")
            _require(row["group_id"] == captured["group_id"] and row["label"] == captured["variant"],
                     "OOF pair or label mismatch")
            _require(np.isfinite(float(row["score"])), "nonfinite OOF score")
            _require(sid not in grouped[(anchor, feature)], "duplicate OOF sample")
            grouped[(anchor, feature)][sid] = row
    _require(selected_rows == 10440 and len(grouped) == 18, f"incomplete {model_id} OOF predictions")
    _require(all(set(rows) == expected_ids for rows in grouped.values()), "OOF feature coverage incomplete")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-recapture-root", type=Path, default=OLD_RECAPTURE)
    parser.add_argument("--new-recapture-root", type=Path, default=NEW_RECAPTURE)
    parser.add_argument("--source-folds", type=Path, default=OLD_RESULTS / "source_component_folds.json")
    parser.add_argument("--old-oof", type=Path, default=OLD_RESULTS / "interface_access_oof_predictions.jsonl")
    parser.add_argument("--new-oof", type=Path, default=OUTPUT / "interface_access_oof_predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()
    prompt_result, context = _prompt_comparability(args.old_recapture_root, args.new_recapture_root, args.source_folds)
    prompt_output = args.output_dir / "prompt_comparability.json"
    _write(prompt_output, prompt_result)
    print("Prompt comparability: 580/580 identical except Harmony header date", flush=True)
    if args.prompt_only:
        return
    old = _load_oof(args.old_oof, "gpt_oss", context, "gpt20")
    new = _load_oof(args.new_oof, "gpt_oss_120b", context, "gpt120")
    ids = context["sample_ids"]
    labels = np.asarray([context["captures"]["gpt120"][sid]["variant"] == "corrupt" for sid in ids])
    groups = [context["pair_groups"][sid.rsplit("/", 1)[0]] for sid in ids]
    comparisons = []
    for anchor, (new_layer, old_layer) in ANCHORS.items():
        for feature in INTERFACE_FEATURE_FAMILIES:
            key = anchor, feature
            left = np.asarray([new[key][sid]["score"] for sid in ids])
            right = np.asarray([old[key][sid]["score"] for sid in ids])
            comparison = paired_group_bootstrap_auroc_difference(
                labels, left, right, groups, left_name="gpt_oss_120b", right_name="gpt_oss_20b",
                seed=SEED, n_resamples=BOOTSTRAP_RESAMPLES,
            )
            _require(comparison["n_valid_resamples"] == BOOTSTRAP_RESAMPLES, "invalid bootstrap resamples")
            comparisons.append({"anchor": anchor, "gpt120_layer": new_layer, "gpt20_layer": old_layer,
                                "feature_family": feature, "n_rows": 580, "n_pairs": 290,
                                "split_manifest_hash": context["split"]["manifest_hash"], **comparison})
            print(f"{anchor} {feature}: 120B-20B={comparison['difference']:.6f}", flush=True)
    limitations = [
        "Secondary paired checkpoint/protocol contrast on common discovery rows and identical source folds; "
        "GPT20 captures are from 2026-08-09 and GPT120 captures from 2026-09-08.",
        "All 580 rendered prompts differ only by the verified Harmony system-header date; input token hashes differ.",
        "Relative anchors compare 0/18/35 with 0/12/23. Checkpoint, layer count, expert count (128 versus 32), "
        "training and implementation history are not independently controlled. Differences cannot identify a model-size effect.",
        "AUROC measures clean/corrupt discrimination, not correctness-specific information or causal utilization.",
        "Two-sided 95% percentile CIs resample 235 canonical source components with fixed OOF predictions; "
        "1000 draws, seed 20260829, no refitting, no multiplicity adjustment across 18 comparisons. "
        "This probe-CI version is separate from the campaign's intervention 50000-draw analyses.",
    ]
    payload = {
        "schema_version": 1, "analysis_id": "interface_access_gpt120_vs_gpt20_secondary_20260908_v1",
        "status": "ok", "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv], "difference_definition": "GPT-OSS-120B_minus_GPT-OSS-20B_micro_OOF_AUROC",
        "protocol": {"seed": SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "alpha": 0.05,
                     "split_manifest_hash": context["split"]["manifest_hash"], "fit_status": "reused_saved_OOF_no_refitting"},
        "validation": {"gpt120_oof_rows": 10440, "gpt20_oof_rows": 10440, "complete_anchor_feature_cells_per_model": 18,
                       "rows_per_cell": 580, "oof_fold_group_role_mismatches": 0,
                       "source_group_train_eval_overlap": 0, "paired_scores_joined_by": "sample_id"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "thread_limit": 2, "platform": platform.platform()},
        "inputs": [_file_record(path) for path in [args.old_oof, args.new_oof, prompt_output, args.source_folds]],
        "code": [_file_record(path) for path in [Path(__file__), ROOT / "src/routeverify/analysis/fair_comparison.py",
                                                  ROOT / "src/routeverify/analysis/probes.py"]],
        "limitations": limitations, "comparisons": comparisons,
    }
    _write(args.output_dir / "cross_checkpoint_comparisons.json", payload)
    csv_path = args.output_dir / "cross_checkpoint_comparisons.csv"
    fields = ["anchor", "gpt120_layer", "gpt20_layer", "feature_family", "n_rows", "n_pairs", "num_groups",
              "gpt120_auroc", "gpt20_auroc", "difference", "ci_low", "ci_high", "seed", "n_resamples", "split_manifest_hash"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({field: row["left"]["auroc"] if field == "gpt120_auroc" else
                             row["right"]["auroc"] if field == "gpt20_auroc" else row[field] for field in fields})
    lines = ["# GPT-OSS-120B versus archived GPT-OSS-20B: secondary interface-access comparison", "",
             "Matched 580 discovery rows, 235 canonical source groups, five frozen folds. "
             "Each interval is a direct paired 120B-20B AUROC contrast.", "",
             "| Relative anchor | Feature | 120B AUROC | 20B AUROC | Difference [95% CI] |",
             "|---|---|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['anchor']} ({row['gpt120_layer']}/{row['gpt20_layer']}) | {row['feature_family']} | "
                     f"{row['left']['auroc']:.4f} | {row['right']['auroc']:.4f} | "
                     f"{row['difference']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    lines.extend(["", *[f"- {text}" for text in limitations], ""])
    (args.output_dir / "cross_checkpoint_comparisons.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote 18 paired checkpoint comparisons to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
