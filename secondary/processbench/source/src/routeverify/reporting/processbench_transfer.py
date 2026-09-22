"""Artifact-only review for the minimal ProcessBench transfer protocol."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

TRANSFER_ROLES = ("matched_correct", "first_error")
TRANSFER_DIRECTIONS = (
    "matched_correct_to_first_error",
    "first_error_to_matched_correct",
)


def review_processbench_transfer_artifacts(
    *,
    repo_root: Path,
    profile_config: Path,
    model_profile: str,
    run_role: str,
    runtime_report: Path,
    behavior_run: Path,
    recapture_run: Path,
    transfer_run: Path,
) -> dict[str, Any]:
    """Review hashes, coverage, identity, and native router dispatch without a model."""

    gate = _yaml_mapping(profile_config)
    profiles = _mapping(gate.get("profiles"), label="profiles")
    run_roles = _mapping(gate.get("run_roles"), label="run_roles")
    profile = _mapping(profiles.get(model_profile), label=f"profiles.{model_profile}")
    role = _mapping(run_roles.get(run_role), label=f"run_roles.{run_role}")
    pair_count = int(role["pair_count"])
    anchor_layer = int(profile["anchor_layer"])
    sites = [f"layer_{anchor_layer}.{kind}" for kind in gate["sites"]]

    data_config_path = _resolve(repo_root, str(gate["data_config"]))
    data_config = _yaml_mapping(data_config_path)
    source_manifest_path = _resolve(repo_root, str(data_config["selection_manifest"]))
    source_manifest = _json_mapping(source_manifest_path)
    expected_pair_ids = list(source_manifest["run_roles"][run_role]["pair_ids"])

    runtime = _json_mapping(runtime_report)
    behavior_rows = _jsonl(behavior_run / "predictions.jsonl")
    behavior_environment = _json_mapping(behavior_run / "environment.json")
    behavior_selection = _json_mapping(behavior_run / "selection_manifest.json")
    recapture_rows = _jsonl(recapture_run / "data_manifest.jsonl")
    recapture_environment = _json_mapping(recapture_run / "environment.json")
    recapture_selection = _json_mapping(recapture_run / "selection_manifest.json")
    transfer_rows = _jsonl(transfer_run / "predictions.jsonl")
    transfer_environment = _json_mapping(transfer_run / "environment.json")
    transfer_selection = _json_mapping(transfer_run / "selection_manifest.json")
    transfer_manifest = _json_mapping(transfer_run / "data_manifest.json")
    transfer_metrics = _json_mapping(transfer_run / "metrics.json")
    runtime_environment = runtime.get("environment")
    runtime_cuda = (
        runtime_environment.get("cuda")
        if isinstance(runtime_environment, Mapping)
        else None
    )

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, **details: Any) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), **details})

    check(
        "runtime_contract",
        runtime.get("passed") is True
        and runtime.get("adapter_alias") == profile["adapter_alias"]
        and runtime.get("adapter_class") == profile["adapter_class"]
        and runtime.get("model_id") == profile["model_id"]
        and runtime.get("revision") == profile["revision"]
        and runtime.get("tokenizer_revision") == profile["tokenizer_revision"]
        and runtime.get("router_semantics_id") == profile["router_semantics_id"]
        and isinstance(runtime.get("runtime_inspect"), Mapping)
        and isinstance(runtime_environment, Mapping)
        and isinstance(runtime_cuda, Mapping)
        and runtime_cuda.get("cuda_available") is True
        and _positive_int(runtime_cuda.get("device_count"))
        and bool(runtime_environment.get("torch"))
        and bool(runtime_environment.get("transformers"))
        and runtime.get("validation_accessed") is False,
    )
    for stage, environment in (
        ("behavior", behavior_environment),
        ("recapture", recapture_environment),
        ("transfer", transfer_environment),
    ):
        model = environment.get("model")
        data = environment.get("data")
        provenance = environment.get("data_provenance")
        check(
            f"{stage}_environment_provenance",
            environment.get("validation_accessed") is False
            and isinstance(model, Mapping)
            and model.get("revision") == profile["revision"]
            and model.get("tokenizer_revision") == profile["tokenizer_revision"]
            and isinstance(data, Mapping)
            and data.get("revision") == data_config["revision"]
            and isinstance(provenance, Mapping)
            and provenance.get("validation_accessed") is False
            and isinstance(environment.get("git"), Mapping)
            and bool(environment["git"].get("commit")),
        )

    parent_manifest_hash = source_manifest["manifest_hash"]
    for stage, selection in (
        ("behavior", behavior_selection),
        ("recapture", recapture_selection),
        ("transfer", transfer_selection),
    ):
        selection_pairs = selection.get("pairs")
        selected_pair_ids = (
            [str(row.get("pair_id")) for row in selection_pairs]
            if isinstance(selection_pairs, Sequence)
            and all(isinstance(row, Mapping) for row in selection_pairs)
            else []
        )
        check(
            f"{stage}_frozen_selection",
            selection.get("parent_manifest_hash") == parent_manifest_hash
            and selection.get("validation_accessed") is False
            and selected_pair_ids == expected_pair_ids,
        )

    behavior_coverage = Counter(
        (str(row.get("pair_id")), str(row.get("transfer_role")))
        for row in behavior_rows
    )
    expected_role_coverage = Counter(
        (pair_id, transfer_role)
        for pair_id in expected_pair_ids
        for transfer_role in TRANSFER_ROLES
    )
    behavior_finite = all(
        _finite(row.get("margin_incorrect_minus_correct_mean"))
        and row.get("gold")
        == ("CORRECT" if row.get("transfer_role") == "matched_correct" else "INCORRECT")
        for row in behavior_rows
    )
    check(
        "behavior_exact_coverage",
        len(behavior_rows) == 2 * pair_count
        and behavior_coverage == expected_role_coverage
        and behavior_finite,
        observed_rows=len(behavior_rows),
        expected_rows=2 * pair_count,
    )

    recapture_coverage = Counter(
        (str(row.get("pair_id")), str(row.get("transfer_role")))
        for row in recapture_rows
    )
    recapture_arrays = all(
        isinstance(row.get("arrays"), Mapping)
        and set(row["arrays"]) == set(sites)
        and row.get("mapped_boundary_step") == row.get("boundary_step")
        and row.get("boundary_source") is not None
        and _resolve(repo_root, str(row.get("activation_file"))).is_file()
        for row in recapture_rows
    )
    check(
        "recapture_exact_coverage",
        len(recapture_rows) == 2 * pair_count
        and recapture_coverage == expected_role_coverage
        and recapture_arrays,
        observed_rows=len(recapture_rows),
        expected_rows=2 * pair_count,
        sites=sites,
    )

    expected_transfer_keys: Counter[tuple[str, str, str | None]] = Counter()
    for pair_id in expected_pair_ids:
        for transfer_role in TRANSFER_ROLES:
            expected_transfer_keys[(pair_id, f"unpatched:{transfer_role}", None)] += 1
            expected_transfer_keys[(pair_id, f"unpatched_rerun:{transfer_role}", None)] += 1
        for site in sites:
            for transfer_role in TRANSFER_ROLES:
                expected_transfer_keys[(pair_id, f"self_patch:{transfer_role}", site)] += 1
            for direction in TRANSFER_DIRECTIONS:
                expected_transfer_keys[(pair_id, direction, site)] += 1
    observed_transfer_keys: Counter[tuple[str, str, str | None]] = Counter()
    transfer_rows_valid = True
    for row in transfer_rows:
        treatment = str(row.get("treatment_semantic"))
        receiver = str(row.get("receiver_role"))
        site_value = row.get("site")
        row_site = None if site_value is None else str(site_value)
        if treatment in {"unpatched", "unpatched_rerun", "self_patch"}:
            semantic_arm = f"{treatment}:{receiver}"
        else:
            semantic_arm = treatment
        observed_transfer_keys[(str(row.get("pair_id")), semantic_arm, row_site)] += 1
        scores = row.get("scores")
        metadata = row.get("metadata")
        transfer_rows_valid = transfer_rows_valid and (
            isinstance(scores, Mapping)
            and set(scores) == {"CORRECT", "INCORRECT"}
            and _finite(row.get("margin_incorrect_minus_correct_mean"))
            and isinstance(row.get("input_provenance"), Mapping)
        )
        if row_site is not None:
            transfer_rows_valid = transfer_rows_valid and (
                isinstance(metadata, Mapping)
                and float(metadata.get("alpha", float("nan"))) == 1.0
            )
    expected_transfer_rows = 12 * pair_count
    check(
        "transfer_exact_coverage",
        len(transfer_rows) == expected_transfer_rows
        and observed_transfer_keys == expected_transfer_keys
        and transfer_rows_valid,
        observed_rows=len(transfer_rows),
        expected_rows=expected_transfer_rows,
    )
    check(
        "transfer_protocol_manifest",
        transfer_manifest.get("validation_accessed") is False
        and set(transfer_manifest.get("sites", [])) == set(sites)
        and transfer_manifest.get("controls") == gate["controls"]
        and transfer_manifest.get("selection_manifest_hash")
        == transfer_selection.get("manifest_hash")
        and transfer_manifest.get("protocol", {}).get("alphas") == [1.0]
        and transfer_manifest.get("protocol", {}).get("include_equivalence_analysis")
        is False,
    )
    check(
        "identity_controls",
        transfer_metrics.get("identity", {}).get("passed") is True
        and float(transfer_metrics.get("identity", {}).get("tolerance", math.inf))
        == float(gate["identity_tolerance"]),
        identity=transfer_metrics.get("identity"),
    )
    check(
        "native_router_dispatch",
        transfer_metrics.get("router_native_dispatch", {}).get("passed") is True
        and transfer_metrics.get("router_native_dispatch", {}).get(
            "topk_change_required"
        )
        is False,
        router_native_dispatch=transfer_metrics.get("router_native_dispatch"),
    )
    serialized_metrics = json.dumps(transfer_metrics, sort_keys=True).lower()
    check(
        "minimal_analysis_only",
        all(
            forbidden not in serialized_metrics
            for forbidden in ("tost", "sesoi", "max_t", "benjamini", "fdr")
        ),
    )

    passed = all(bool(item["passed"]) for item in checks)
    return {
        "schema_version": 1,
        "review_id": "processbench_natural_transfer_artifact_review_v1",
        "passed": passed,
        "model_profile": model_profile,
        "run_role": run_role,
        "expected_pair_count": pair_count,
        "expected_pair_ids": expected_pair_ids,
        "expected_sites": sites,
        "checks": checks,
        "validation_accessed": False,
        "interpretation": (
            "artifact integrity review only; no effect-sign or significance gate"
        ),
    }


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _positive_int(value: Any) -> bool:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, bool) and number >= 1
