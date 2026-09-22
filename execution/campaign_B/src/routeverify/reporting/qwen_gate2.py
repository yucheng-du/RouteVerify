"""Gate checks for bounded Qwen3 Gate 2 repeat artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from routeverify.data.schema import stable_hash

EXPECTED_BOUNDARY_STRATEGY = "first_error_or_matched_all_correct"
EXPECTED_PROMPT_RUNS = {
    "qwen3_diagnostic_pilot": ("plain_trace", "raw"),
    "qwen3_diagnostic_verifier_pilot": ("step_binary", "chat"),
}
FORBIDDEN_PATH_MARKERS = (
    "router_visibility",
    "patch",
    "rescue",
    "sweep",
)


@dataclass(frozen=True)
class Gate2Check:
    name: str
    passed: bool
    details: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "data": self.data,
        }


def qwen_gate2_review(
    *,
    run_root: Path,
    expected_examples: int = 16,
    expected_seed: int = 0,
) -> dict[str, Any]:
    """Review a bounded Qwen3 Gate 2 artifact tree without loading model weights."""

    checks: list[Gate2Check] = []
    checks.extend(_required_files(run_root))
    checks.extend(_smoke_review_checks(run_root / "qwen3_smoke_review.pre_gate2.json"))
    checks.extend(_run_manifest_checks(run_root / "run_manifest.json", expected_seed=expected_seed))
    checks.extend(_behavior_checks(run_root, expected_examples=expected_examples))
    checks.extend(_diagnostic_checks(run_root, expected_examples=expected_examples))
    checks.extend(_forbidden_artifact_checks(run_root))
    checks.extend(_sha256_checks(run_root / "SHA256SUMS", run_root=run_root))

    failed = [check for check in checks if not check.passed]
    return {
        "gate": "qwen3_gate2_repeat",
        "status": "PASS" if not failed else "NO-GO",
        "passed": not failed,
        "run_root": str(run_root),
        "summary": {
            "num_checks": len(checks),
            "num_failed": len(failed),
            "expected_examples": expected_examples,
            "expected_seed": expected_seed,
        },
        "checks": [check.to_json() for check in checks],
    }


def _required_files(run_root: Path) -> list[Gate2Check]:
    paths = [
        "run_manifest.json",
        "SHA256SUMS",
        "qwen3_gate2_run.log",
        "qwen3_smoke_review.pre_gate2.json",
        "qwen3_moe/model_discovery.json",
        "qwen3_moe/capture_check.json",
        "qwen3_moe/behavior/processbench_step-binary/metrics.json",
        "qwen3_moe/behavior/processbench_first-error/metrics.json",
        "qwen3_diagnostic_pilot/metrics.json",
        "qwen3_diagnostic_pilot/environment.json",
        "qwen3_diagnostic_pilot/data_manifest.jsonl",
        "qwen3_diagnostic_pilot/report/summary.json",
        "qwen3_diagnostic_verifier_pilot/metrics.json",
        "qwen3_diagnostic_verifier_pilot/environment.json",
        "qwen3_diagnostic_verifier_pilot/data_manifest.jsonl",
        "qwen3_diagnostic_verifier_pilot/report/summary.json",
        "gate1_snapshot/qwen3_diagnostic_pilot/metrics.json",
        "gate1_snapshot/qwen3_diagnostic_pilot/environment.json",
        "gate1_snapshot/qwen3_moe/model_discovery.json",
        "gate1_snapshot/qwen3_moe/capture_check.json",
    ]
    return [
        Gate2Check(
            name=f"required artifact: {path}",
            passed=(run_root / path).exists(),
            details=str(run_root / path),
        )
        for path in paths
    ]


def _smoke_review_checks(path: Path) -> list[Gate2Check]:
    payload = _read_json(path)
    if payload is None:
        return []
    has_passed = "passed" in payload
    has_status = "status" in payload
    passed_ok = payload.get("passed") is True if has_passed else True
    status_ok = payload.get("status") == "PASS" if has_status else True
    return [
        Gate2Check(
            name="pre-Gate2 smoke review: passed",
            passed=(has_passed or has_status) and passed_ok and status_ok,
            details="qwen3_smoke_review.pre_gate2.json must report passed=true and status=PASS when present",
            data={
                "passed": payload.get("passed"),
                "status": payload.get("status"),
            },
        )
    ]


def _run_manifest_checks(path: Path, *, expected_seed: int) -> list[Gate2Check]:
    payload = _read_json(path)
    if payload is None:
        return []
    env = payload.get("environment", {})
    checks = [
        Gate2Check(
            name="run manifest: success",
            passed=payload.get("success") is True and payload.get("exit_code") == 0,
            details="run_manifest.json must record success=true and exit_code=0",
            data={"success": payload.get("success"), "exit_code": payload.get("exit_code")},
        ),
        Gate2Check(
            name="run manifest: diagnostics enabled",
            passed=env.get("RUN_QWEN_DIAGNOSTICS") == "1",
            details="RUN_QWEN_DIAGNOSTICS must be 1 for Gate 2",
            data={"actual": env.get("RUN_QWEN_DIAGNOSTICS")},
        ),
        Gate2Check(
            name="run manifest: router visibility disabled",
            passed=env.get("RUN_ROUTER_VISIBILITY") == "0",
            details="RUN_ROUTER_VISIBILITY must be 0 for bounded Gate 2",
            data={"actual": env.get("RUN_ROUTER_VISIBILITY")},
        ),
        Gate2Check(
            name="run manifest: seed",
            passed=_int_or_none(env.get("SEED")) == expected_seed,
            details=f"expected SEED={expected_seed}",
            data={"actual": env.get("SEED")},
        ),
    ]
    for key in (
        "QWEN_SMOKE_LIMIT",
        "BEHAVIOR_LIMIT",
        "FIRST_ERROR_LIMIT",
        "DIAGNOSTIC_LIMIT",
        "VERIFIER_DIAGNOSTIC_LIMIT",
    ):
        checks.append(
            Gate2Check(
                name=f"run manifest: {key}",
                passed=_int_or_none(env.get(key)) == 16,
                details=f"expected {key}=16",
                data={"actual": env.get(key)},
            )
        )
    return checks


def _behavior_checks(run_root: Path, *, expected_examples: int) -> list[Gate2Check]:
    step_metrics = _read_json(
        run_root / "qwen3_moe/behavior/processbench_step-binary/metrics.json"
    )
    first_metrics = _read_json(
        run_root / "qwen3_moe/behavior/processbench_first-error/metrics.json"
    )
    checks: list[Gate2Check] = []
    if step_metrics is not None:
        checks.extend(
            [
                Gate2Check(
                    name="step-binary: n",
                    passed=_int_or_none(step_metrics.get("n")) == expected_examples,
                    details=f"expected n={expected_examples}",
                    data={"actual": step_metrics.get("n")},
                ),
                Gate2Check(
                    name="step-binary: balanced labels",
                    passed=step_metrics.get("positives") == step_metrics.get("negatives") == 8,
                    details="expected 8 INCORRECT and 8 CORRECT examples",
                    data={
                        "positives": step_metrics.get("positives"),
                        "negatives": step_metrics.get("negatives"),
                    },
                ),
            ]
        )
    if first_metrics is not None:
        checks.extend(
            [
                Gate2Check(
                    name="first-error: n",
                    passed=_int_or_none(first_metrics.get("n")) == expected_examples,
                    details=f"expected n={expected_examples}",
                    data={"actual": first_metrics.get("n")},
                ),
                Gate2Check(
                    name="first-error: no invalid predictions",
                    passed=_int_or_none(first_metrics.get("invalid_count")) == 0,
                    details="candidate-scored first-error endpoint should have invalid_count=0",
                    data={"actual": first_metrics.get("invalid_count")},
                ),
            ]
        )
    return checks


def _diagnostic_checks(run_root: Path, *, expected_examples: int) -> list[Gate2Check]:
    checks: list[Gate2Check] = []
    for run_name, (prompt_mode, input_format) in EXPECTED_PROMPT_RUNS.items():
        run_dir = run_root / run_name
        metrics = _read_json(run_dir / "metrics.json")
        environment = _read_json(run_dir / "environment.json")
        rows = _read_jsonl(run_dir / "data_manifest.jsonl")
        if environment is not None:
            checks.extend(_data_provenance_checks(run_name, environment))
        if metrics is not None:
            checks.extend(
                [
                    Gate2Check(
                        name=f"{run_name}: n",
                        passed=_int_or_none(metrics.get("n")) == expected_examples,
                        details=f"expected n={expected_examples}",
                        data={"actual": metrics.get("n")},
                    ),
                    Gate2Check(
                        name=f"{run_name}: one boundary per example",
                        passed=_int_or_none(metrics.get("num_boundaries")) == expected_examples,
                        details="matched Gate 2 diagnostics should capture one boundary per example",
                        data={"actual": metrics.get("num_boundaries")},
                    ),
                    Gate2Check(
                        name=f"{run_name}: prompt mode",
                        passed=metrics.get("prompt_mode") == prompt_mode,
                        details=f"expected {prompt_mode}",
                        data={"actual": metrics.get("prompt_mode")},
                    ),
                    Gate2Check(
                        name=f"{run_name}: input format",
                        passed=metrics.get("input_format") == input_format,
                        details=f"expected {input_format}",
                        data={"actual": metrics.get("input_format")},
                    ),
                    Gate2Check(
                        name=f"{run_name}: boundary strategy",
                        passed=metrics.get("boundary_strategy") == EXPECTED_BOUNDARY_STRATEGY,
                        details=f"expected {EXPECTED_BOUNDARY_STRATEGY}",
                        data={"actual": metrics.get("boundary_strategy")},
                    ),
                ]
            )
        if rows is not None:
            checks.extend(_manifest_checks(run_name, rows, run_dir=run_dir, run_root=run_root))
    return checks


def _manifest_checks(
    run_name: str,
    rows: list[dict[str, Any]],
    *,
    run_dir: Path,
    run_root: Path,
) -> list[Gate2Check]:
    labels = [bool(row.get("is_error_boundary")) for row in rows]
    activation_missing = _missing_activation_files(rows, run_dir=run_dir, run_root=run_root)
    rows_with_match_role = [
        row for row in rows if isinstance(row.get("boundary_match_role"), str)
    ]
    negative_rows = [row for row in rows if not bool(row.get("is_error_boundary"))]
    negative_rows_with_match = [
        row
        for row in negative_rows
        if row.get("boundary_match_role") == "negative_matched_all_correct"
        and isinstance(row.get("matched_target_sample_id"), str)
        and row.get("matched_target_step") is not None
        and row.get("matched_target_token_position") is not None
        and row.get("matched_step_delta") is not None
        and row.get("matched_token_delta") is not None
    ]
    return [
        Gate2Check(
            name=f"{run_name}: manifest rows",
            passed=len(rows) == 16,
            details="expected 16 manifest rows",
            data={"actual": len(rows)},
        ),
        Gate2Check(
            name=f"{run_name}: manifest unique samples",
            passed=len({str(row.get("sample_id")) for row in rows}) == len(rows),
            details="expected one row per sample",
            data={"unique_samples": len({str(row.get("sample_id")) for row in rows})},
        ),
        Gate2Check(
            name=f"{run_name}: manifest balanced labels",
            passed=sum(labels) == 8 and len(labels) - sum(labels) == 8,
            details="expected 8 positive and 8 matched negative rows",
            data={"positives": sum(labels), "negatives": len(labels) - sum(labels)},
        ),
        Gate2Check(
            name=f"{run_name}: one position per file",
            passed={row.get("num_positions_in_file") for row in rows} == {1},
            details="matched captures should store one position per activation file",
            data={"actual": sorted({str(row.get("num_positions_in_file")) for row in rows})},
        ),
        Gate2Check(
            name=f"{run_name}: prompt/input hashes",
            passed=all(row.get("prompt_hash") and row.get("input_ids_hash") for row in rows),
            details="every row should include prompt_hash and input_ids_hash",
        ),
        Gate2Check(
            name=f"{run_name}: activation files exist",
            passed=not activation_missing,
            details="all manifest activation_file references should resolve",
            data={"missing": activation_missing[:20], "num_missing": len(activation_missing)},
        ),
        Gate2Check(
            name=f"{run_name}: matched-boundary metadata present",
            passed=len(rows_with_match_role) == len(rows),
            details="every Gate 2 manifest row should carry boundary_match_role",
            data={"actual": len(rows_with_match_role), "expected": len(rows)},
        ),
        Gate2Check(
            name=f"{run_name}: negative match metadata present",
            passed=len(negative_rows_with_match) == len(negative_rows),
            details="matched all-correct negatives should identify their positive target boundary",
            data={"actual": len(negative_rows_with_match), "expected": len(negative_rows)},
        ),
    ]


def _data_provenance_checks(run_name: str, environment: dict[str, Any]) -> list[Gate2Check]:
    provenance = environment.get("data_provenance")
    selection = provenance.get("selection") if isinstance(provenance, dict) else None
    rows = selection.get("rows") if isinstance(selection, dict) else None
    source_type = provenance.get("source_type") if isinstance(provenance, dict) else None
    rows_hash = selection.get("rows_hash") if isinstance(selection, dict) else None
    computed_rows_hash = stable_hash(rows, length=64) if isinstance(rows, list) else None
    if source_type == "processed_jsonl" and isinstance(provenance, dict):
        source_details_ok = _processed_source_details_ok(provenance)
    elif source_type == "hf_loader" and isinstance(provenance, dict):
        source_details_ok = bool(provenance.get("hf_id") and provenance.get("revision"))
    else:
        source_details_ok = False
    return [
        Gate2Check(
            name=f"{run_name}: data provenance present",
            passed=isinstance(provenance, dict) and source_type in {"processed_jsonl", "hf_loader"},
            details="environment.json should record whether data came from processed JSONL or HF loader",
            data={"source_type": source_type},
        ),
        Gate2Check(
            name=f"{run_name}: selected row provenance present",
            passed=isinstance(rows, list) and len(rows) == 16 and isinstance(rows_hash, str),
            details="data provenance should include the selected rows and rows_hash for n=16 Gate 2",
            data={
                "row_count": len(rows) if isinstance(rows, list) else None,
                "rows_hash_present": isinstance(rows_hash, str),
            },
        ),
        Gate2Check(
            name=f"{run_name}: selected rows hash verifies",
            passed=isinstance(rows_hash, str) and rows_hash == computed_rows_hash,
            details="selection.rows_hash should match the selected row provenance payload",
            data={"expected": computed_rows_hash, "actual": rows_hash},
        ),
        Gate2Check(
            name=f"{run_name}: data source details present",
            passed=source_details_ok,
            details="HF provenance should include hf_id/revision; processed provenance should include file hash and row count",
            data={"source_type": source_type},
        ),
    ]


def _processed_source_details_ok(provenance: dict[str, Any]) -> bool:
    processed_file = provenance.get("processed_file")
    return (
        isinstance(processed_file, dict)
        and isinstance(processed_file.get("path"), str)
        and isinstance(processed_file.get("sha256"), str)
        and _int_or_none(processed_file.get("row_count")) is not None
    )


def _forbidden_artifact_checks(run_root: Path) -> list[Gate2Check]:
    found = [
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if any(marker in path.name for marker in FORBIDDEN_PATH_MARKERS)
    ]
    return [
        Gate2Check(
            name="forbidden artifacts absent",
            passed=not found,
            details="router visibility, patch/rescue, sweep artifacts must not be present",
            data={"found": found[:30], "num_found": len(found)},
        )
    ]


def _sha256_checks(path: Path, *, run_root: Path) -> list[Gate2Check]:
    if not path.exists():
        return []
    failures: list[str] = []
    entries = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries += 1
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            failures.append(line)
            continue
        expected, relative = parts
        relative = relative.removeprefix("*").removeprefix("./")
        candidate = run_root / relative
        if not candidate.exists() or _sha256(candidate) != expected:
            failures.append(relative)
    return [
        Gate2Check(
            name="sha256sums: entries verify",
            passed=not failures and entries > 0,
            details="all SHA256SUMS entries should match current files",
            data={"entries": entries, "failures": failures[:20], "num_failures": len(failures)},
        )
    ]


def _missing_activation_files(
    rows: list[dict[str, Any]], *, run_dir: Path, run_root: Path
) -> list[str]:
    missing: list[str] = []
    for row in rows:
        value = row.get("activation_file")
        if not isinstance(value, str):
            missing.append("<missing activation_file>")
            continue
        path = Path(value)
        candidates = [path] if path.is_absolute() else [path, run_root / value, run_dir / path.name]
        if not any(candidate.exists() for candidate in candidates):
            missing.append(value)
    return sorted(set(missing))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_or_none(value: object) -> int | None:
    if value is None or not isinstance(value, str | int | float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
