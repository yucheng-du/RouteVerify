"""Gate checks for Qwen3 engineering-smoke artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EXPECTED_ALIAS = "qwen3_moe"
EXPECTED_MODEL_ID = "Qwen/Qwen3-30B-A3B"
EXPECTED_LAYERS = 48
EXPECTED_EXPERTS = 128
EXPECTED_TOP_K = 8
EXPECTED_MODULE_COUNTS = {"moe_block": 48, "router": 48, "experts": 48}
EXPECTED_PROMPT_MODE = "plain_trace"
EXPECTED_BOUNDARY_STRATEGY = "first_error_or_matched_all_correct"
EXPECTED_SITES = (
    "pre_moe_residual",
    "router_input",
    "router_logits",
    "router_topk_ids",
    "router_topk_weights",
    "moe_output",
    "post_moe_residual",
)
EXPECTED_SMOKE_LAYERS = (0, 12, 24, 36, 47)


@dataclass(frozen=True)
class SmokeCheck:
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


def qwen_smoke_review(
    *,
    artifact_root: Path = Path("artifacts"),
    run_dir: Path = Path("artifacts/qwen3_diagnostic_pilot"),
    min_examples: int = 8,
    max_examples: int | None = 16,
    require_activation_files: bool = False,
) -> dict[str, Any]:
    """Review RunPod Qwen3 smoke artifacts without loading model weights."""

    checks: list[SmokeCheck] = []
    qwen_model_dir = artifact_root / EXPECTED_ALIAS
    expected_paths = {
        "model_discovery": qwen_model_dir / "model_discovery.json",
        "capture_check": qwen_model_dir / "capture_check.json",
        "resolved_config": run_dir / "config.resolved.yaml",
        "environment": run_dir / "environment.json",
        "run_model_discovery": run_dir / "model_discovery.json",
        "metrics": run_dir / "metrics.json",
        "data_manifest": run_dir / "data_manifest.jsonl",
    }
    for name, path in expected_paths.items():
        checks.append(
            SmokeCheck(
                name=f"required artifact: {name}",
                passed=path.exists(),
                details=str(path),
            )
        )

    config = _read_yaml_if_present(expected_paths["resolved_config"])
    metrics = _read_json_if_present(expected_paths["metrics"])
    manifest_rows = _read_jsonl_if_present(expected_paths["data_manifest"])
    capture_check = _read_json_if_present(expected_paths["capture_check"])
    model_discovery = _read_json_if_present(expected_paths["model_discovery"])
    run_model_discovery = _read_json_if_present(expected_paths["run_model_discovery"])

    checks.extend(_model_discovery_checks(model_discovery, label="inspect discovery"))
    checks.extend(_model_discovery_checks(run_model_discovery, label="run discovery"))
    checks.extend(_capture_checks(capture_check))
    checks.extend(_config_checks(config, min_examples=min_examples, max_examples=max_examples))
    checks.extend(_metrics_checks(metrics, config, min_examples=min_examples, max_examples=max_examples))
    checks.extend(
        _manifest_checks(
            manifest_rows,
            metrics,
            config,
            run_dir=run_dir,
            require_activation_files=require_activation_files,
        )
    )
    checks.append(_diagnostics_absent_check(artifact_root=artifact_root, run_dir=run_dir))

    failed = [check for check in checks if not check.passed]
    status = "PASS" if not failed else "NO-GO"
    return {
        "gate": "qwen3_engineering_smoke",
        "status": status,
        "passed": not failed,
        "artifact_root": str(artifact_root),
        "run_dir": str(run_dir),
        "summary": {
            "num_checks": len(checks),
            "num_failed": len(failed),
            "missing_artifacts": [
                check.details
                for check in checks
                if check.name.startswith("required artifact") and not check.passed
            ],
            "manifest_rows": len(manifest_rows) if manifest_rows is not None else None,
            "unique_manifest_samples": (
                len({str(row.get("sample_id")) for row in manifest_rows})
                if manifest_rows is not None
                else None
            ),
            "metrics_n": _int_or_none(metrics.get("n")) if metrics is not None else None,
            "metrics_num_boundaries": (
                _int_or_none(metrics.get("num_boundaries")) if metrics is not None else None
            ),
        },
        "checks": [check.to_json() for check in checks],
        "next_action": _next_action(status),
    }


def _model_discovery_checks(payload: dict[str, Any] | None, *, label: str) -> list[SmokeCheck]:
    if payload is None:
        return []
    metadata = payload.get("metadata", {})
    summary = payload.get("summary", {})
    checks = [
        SmokeCheck(
            name=f"{label}: alias",
            passed=payload.get("alias") == EXPECTED_ALIAS,
            details=f"expected {EXPECTED_ALIAS}",
            data={"actual": payload.get("alias")},
        ),
        SmokeCheck(
            name=f"{label}: model id",
            passed=payload.get("model_id") == EXPECTED_MODEL_ID,
            details=f"expected {EXPECTED_MODEL_ID}",
            data={"actual": payload.get("model_id")},
        ),
        SmokeCheck(
            name=f"{label}: hidden layers",
            passed=_int_or_none(metadata.get("num_hidden_layers")) == EXPECTED_LAYERS,
            details=f"expected {EXPECTED_LAYERS}",
            data={"actual": metadata.get("num_hidden_layers")},
        ),
        SmokeCheck(
            name=f"{label}: experts",
            passed=_int_or_none(metadata.get("num_experts")) == EXPECTED_EXPERTS,
            details=f"expected {EXPECTED_EXPERTS}",
            data={"actual": metadata.get("num_experts")},
        ),
        SmokeCheck(
            name=f"{label}: top k",
            passed=_int_or_none(metadata.get("num_experts_per_tok")) == EXPECTED_TOP_K,
            details=f"expected {EXPECTED_TOP_K}",
            data={"actual": metadata.get("num_experts_per_tok")},
        ),
    ]
    if isinstance(summary, dict):
        for role, expected_count in EXPECTED_MODULE_COUNTS.items():
            checks.append(
                SmokeCheck(
                    name=f"{label}: {role} count",
                    passed=_int_or_none(summary.get(role)) == expected_count,
                    details=f"expected {expected_count}",
                    data={"actual": summary.get(role)},
                )
            )
    else:
        checks.append(
            SmokeCheck(
                name=f"{label}: module summary",
                passed=False,
                details="missing discovery summary",
            )
        )
    return checks


def _capture_checks(payload: dict[str, Any] | None) -> list[SmokeCheck]:
    if payload is None:
        return []
    tolerance = _float_or_none(payload.get("tolerance")) or 1e-6
    capture_delta = _float_or_none(payload.get("max_abs_delta_baseline_vs_capture"))
    rerun_delta = _float_or_none(payload.get("max_abs_delta_baseline_vs_rerun"))
    model = payload.get("model", {})
    return [
        SmokeCheck(
            name="capture identity: passed flag",
            passed=payload.get("passed") is True,
            details="capture_check.json must report passed=true",
            data={"actual": payload.get("passed")},
        ),
        SmokeCheck(
            name="capture identity: capture delta",
            passed=capture_delta is not None and capture_delta <= tolerance,
            details=f"max_abs_delta_baseline_vs_capture <= {tolerance}",
            data={"actual": capture_delta},
        ),
        SmokeCheck(
            name="capture identity: rerun delta",
            passed=rerun_delta is not None and rerun_delta <= tolerance,
            details=f"max_abs_delta_baseline_vs_rerun <= {tolerance}",
            data={"actual": rerun_delta},
        ),
        SmokeCheck(
            name="capture identity: model alias",
            passed=isinstance(model, dict) and model.get("alias") == EXPECTED_ALIAS,
            details=f"expected {EXPECTED_ALIAS}",
            data={"actual": model.get("alias") if isinstance(model, dict) else None},
        ),
        SmokeCheck(
            name="capture identity: full weights loaded",
            passed=isinstance(model, dict) and model.get("load_weights") is True,
            details="capture identity should run against loaded weights",
            data={"actual": model.get("load_weights") if isinstance(model, dict) else None},
        ),
    ]


def _config_checks(
    config: dict[str, Any] | None,
    *,
    min_examples: int,
    max_examples: int | None,
) -> list[SmokeCheck]:
    if config is None:
        return []
    layers = _int_tuple(config.get("layers"))
    sites = _str_tuple(config.get("sites"))
    limit = _int_or_none(config.get("limit"))
    checks = [
        SmokeCheck(
            name="resolved config: prompt mode",
            passed=config.get("prompt_mode") == EXPECTED_PROMPT_MODE,
            details=f"expected {EXPECTED_PROMPT_MODE}",
            data={"actual": config.get("prompt_mode")},
        ),
        SmokeCheck(
            name="resolved config: boundary strategy",
            passed=config.get("boundary_strategy") == EXPECTED_BOUNDARY_STRATEGY,
            details=f"expected {EXPECTED_BOUNDARY_STRATEGY}",
            data={"actual": config.get("boundary_strategy")},
        ),
        SmokeCheck(
            name="resolved config: layers",
            passed=layers == EXPECTED_SMOKE_LAYERS,
            details=f"expected {list(EXPECTED_SMOKE_LAYERS)}",
            data={"actual": list(layers)},
        ),
        SmokeCheck(
            name="resolved config: sites",
            passed=sites == EXPECTED_SITES,
            details=f"expected {list(EXPECTED_SITES)}",
            data={"actual": list(sites)},
        ),
        SmokeCheck(
            name="resolved config: limit lower bound",
            passed=limit is not None and limit >= min_examples,
            details=f"expected limit >= {min_examples}",
            data={"actual": limit},
        ),
    ]
    if max_examples is not None:
        checks.append(
            SmokeCheck(
                name="resolved config: limit upper bound",
                passed=limit is not None and limit <= max_examples,
                details=f"expected limit <= {max_examples}",
                data={"actual": limit},
            )
        )
    return checks


def _metrics_checks(
    metrics: dict[str, Any] | None,
    config: dict[str, Any] | None,
    *,
    min_examples: int,
    max_examples: int | None,
) -> list[SmokeCheck]:
    if metrics is None:
        return []
    n = _int_or_none(metrics.get("n"))
    num_boundaries = _int_or_none(metrics.get("num_boundaries"))
    num_sites = _int_or_none(metrics.get("num_sites"))
    expected_sites = _expected_site_count(config)
    checks = [
        SmokeCheck(
            name="metrics: example count lower bound",
            passed=n is not None and n >= min_examples,
            details=f"expected n >= {min_examples}",
            data={"actual": n},
        ),
        SmokeCheck(
            name="metrics: boundaries cover examples",
            passed=n is not None and num_boundaries is not None and num_boundaries == n,
            details="matched smoke should capture exactly one boundary per example",
            data={"n": n, "num_boundaries": num_boundaries},
        ),
        SmokeCheck(
            name="metrics: prompt mode",
            passed=metrics.get("prompt_mode") == EXPECTED_PROMPT_MODE,
            details=f"expected {EXPECTED_PROMPT_MODE}",
            data={"actual": metrics.get("prompt_mode")},
        ),
        SmokeCheck(
            name="metrics: boundary strategy",
            passed=metrics.get("boundary_strategy") == EXPECTED_BOUNDARY_STRATEGY,
            details=f"expected {EXPECTED_BOUNDARY_STRATEGY}",
            data={"actual": metrics.get("boundary_strategy")},
        ),
    ]
    if max_examples is not None:
        checks.append(
            SmokeCheck(
                name="metrics: example count upper bound",
                passed=n is not None and n <= max_examples,
                details=f"expected n <= {max_examples}",
                data={"actual": n},
            )
        )
    if expected_sites is not None:
        checks.append(
            SmokeCheck(
                name="metrics: selected site count",
                passed=num_sites == expected_sites,
                details=f"expected {expected_sites}",
                data={"actual": num_sites},
            )
        )
    return checks


def _manifest_checks(
    rows: list[dict[str, Any]] | None,
    metrics: dict[str, Any] | None,
    config: dict[str, Any] | None,
    *,
    run_dir: Path,
    require_activation_files: bool,
) -> list[SmokeCheck]:
    if rows is None:
        return []
    row_count = len(rows)
    n = _int_or_none(metrics.get("n")) if metrics is not None else None
    num_boundaries = _int_or_none(metrics.get("num_boundaries")) if metrics is not None else None
    expected_sites = _expected_site_count(config)
    unique_samples = {str(row.get("sample_id")) for row in rows}
    prompt_modes = {row.get("prompt_mode") for row in rows}
    boundary_strategies = {row.get("boundary_strategy") for row in rows}
    rows_with_arrays = [row for row in rows if isinstance(row.get("arrays"), dict)]
    labels = [bool(row.get("is_error_boundary")) for row in rows]
    checks = [
        SmokeCheck(
            name="manifest: nonempty",
            passed=row_count > 0,
            details="data_manifest.jsonl should contain boundary rows",
            data={"rows": row_count},
        ),
        SmokeCheck(
            name="manifest: row count matches metrics",
            passed=num_boundaries is not None and row_count == num_boundaries,
            details="manifest rows should match metrics.num_boundaries",
            data={"rows": row_count, "num_boundaries": num_boundaries},
        ),
        SmokeCheck(
            name="manifest: unique samples match metrics",
            passed=n is not None and len(unique_samples) == n,
            details="unique sample IDs should match metrics.n",
            data={"unique_samples": len(unique_samples), "n": n},
        ),
        SmokeCheck(
            name="manifest: prompt mode",
            passed=prompt_modes == {EXPECTED_PROMPT_MODE},
            details=f"expected only {EXPECTED_PROMPT_MODE}",
            data={"actual": sorted(str(value) for value in prompt_modes)},
        ),
        SmokeCheck(
            name="manifest: boundary strategy",
            passed=boundary_strategies == {EXPECTED_BOUNDARY_STRATEGY},
            details=f"expected only {EXPECTED_BOUNDARY_STRATEGY}",
            data={"actual": sorted(str(value) for value in boundary_strategies)},
        ),
        SmokeCheck(
            name="manifest: arrays present",
            passed=len(rows_with_arrays) == row_count,
            details="each boundary row should include captured array shapes",
            data={"rows_with_arrays": len(rows_with_arrays), "rows": row_count},
        ),
        SmokeCheck(
            name="manifest: matched labels",
            passed=sum(labels) > 0 and row_count - sum(labels) > 0,
            details="matched smoke should include positive and negative boundaries",
            data={"positives": sum(labels), "negatives": row_count - sum(labels)},
        ),
        SmokeCheck(
            name="manifest: one position per file",
            passed={row.get("num_positions_in_file") for row in rows} == {1},
            details="matched smoke should capture one boundary per activation file",
            data={"actual": sorted(str(row.get("num_positions_in_file")) for row in rows)},
        ),
        SmokeCheck(
            name="manifest: prompt/input hashes",
            passed=all(row.get("prompt_hash") and row.get("input_ids_hash") for row in rows),
            details="each row should include prompt_hash and input_ids_hash",
        ),
    ]
    if rows_with_arrays and expected_sites is not None:
        array_counts = {len(row["arrays"]) for row in rows_with_arrays}
        checks.append(
            SmokeCheck(
                name="manifest: array site count",
                passed=array_counts == {expected_sites},
                details=f"expected {expected_sites} arrays per row",
                data={"actual": sorted(array_counts)},
            )
        )
    if rows_with_arrays:
        checks.extend(_array_shape_checks(rows_with_arrays[0]))
    if require_activation_files:
        missing_files = _missing_activation_files(rows, run_dir=run_dir)
        checks.append(
            SmokeCheck(
                name="manifest: activation files exist",
                passed=not missing_files,
                details="all referenced activation files should be present",
                data={"missing": missing_files[:20], "num_missing": len(missing_files)},
            )
        )
    return checks


def _array_shape_checks(row: dict[str, Any]) -> list[SmokeCheck]:
    arrays = row.get("arrays", {})
    if not isinstance(arrays, dict):
        return []
    layer_ids = sorted(
        {
            int(key.split(".", 1)[0].removeprefix("layer_"))
            for key in arrays
            if key.startswith("layer_") and "." in key
        }
    )
    topk_shapes = [
        value
        for key, value in arrays.items()
        if key.endswith(".router_topk_ids") or key.endswith(".router_topk_weights")
    ]
    logits_shapes = [value for key, value in arrays.items() if key.endswith(".router_logits")]
    return [
        SmokeCheck(
            name="manifest arrays: selected layers",
            passed=tuple(layer_ids) == EXPECTED_SMOKE_LAYERS,
            details=f"expected {list(EXPECTED_SMOKE_LAYERS)}",
            data={"actual": layer_ids},
        ),
        SmokeCheck(
            name="manifest arrays: top-k shape",
            passed=bool(topk_shapes) and all(_last_dim(shape) == EXPECTED_TOP_K for shape in topk_shapes),
            details=f"router_topk_* last dim should be {EXPECTED_TOP_K}",
            data={"num_shapes": len(topk_shapes)},
        ),
        SmokeCheck(
            name="manifest arrays: router-logit shape",
            passed=bool(logits_shapes)
            and all(_last_dim(shape) == EXPECTED_EXPERTS for shape in logits_shapes),
            details=f"router_logits last dim should be {EXPECTED_EXPERTS}",
            data={"num_shapes": len(logits_shapes)},
        ),
    ]


def _diagnostics_absent_check(*, artifact_root: Path, run_dir: Path) -> SmokeCheck:
    diagnostics_paths = [
        run_dir / "analysis" / "probe_metrics.json",
        run_dir / "analysis" / "routing_metrics.json",
        run_dir / "analysis" / "router_visibility_metrics.json",
        artifact_root / EXPECTED_ALIAS / "behavior" / "processbench_step-binary" / "metrics.json",
        artifact_root / EXPECTED_ALIAS / "behavior" / "processbench_first-error" / "metrics.json",
        artifact_root / "qwen3_diagnostic_verifier_pilot" / "metrics.json",
    ]
    present = [str(path) for path in diagnostics_paths if path.exists()]
    return SmokeCheck(
        name="diagnostics remain disabled",
        passed=not present,
        details="RUN_QWEN_DIAGNOSTICS should stay off for Gate 1 smoke",
        data={"present": present},
    )


def _expected_site_count(config: dict[str, Any] | None) -> int | None:
    if config is None:
        return None
    layers = _int_tuple(config.get("layers"))
    sites = _str_tuple(config.get("sites"))
    if not layers or not sites:
        return None
    return len(layers) * len(sites)


def _missing_activation_files(rows: list[dict[str, Any]], *, run_dir: Path) -> list[str]:
    missing: list[str] = []
    for row in rows:
        value = row.get("activation_file")
        if not isinstance(value, str):
            missing.append("<missing activation_file field>")
            continue
        path = Path(value)
        candidates = [path] if path.is_absolute() else [path, run_dir / path.name, run_dir / value]
        if not any(candidate.exists() for candidate in candidates):
            missing.append(value)
    return sorted(set(missing))


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _read_yaml_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    return rows


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    output: list[int] = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is None:
            return ()
        output.append(parsed)
    return tuple(output)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _last_dim(shape: Any) -> int | None:
    if not isinstance(shape, list) or not shape:
        return None
    return _int_or_none(shape[-1])


def _next_action(status: str) -> str:
    if status == "PASS":
        return (
            "Gate 1 smoke artifacts look internally consistent. Keep diagnostics disabled "
            "until an external Gate 1 -> Gate 2 audit reviews these artifacts."
        )
    return (
        "Do not enable RUN_QWEN_DIAGNOSTICS. Re-run the RunPod smoke-only command or copy "
        "back the missing artifacts, then re-run this check."
    )
