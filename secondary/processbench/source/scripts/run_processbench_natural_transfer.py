#!/usr/bin/env python3
"""Plan or run the pinned three-model ProcessBench natural-error transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from routeverify.data.processbench_transfer import (
    load_processbench_transfer_bundle,
)
from routeverify.data.schema import stable_hash

PROFILE_NAMES = ("gpt_oss", "gemma4", "nemotron3_nano")
RUN_ROLES = ("smoke", "full")
EXPECTED_SITES = ("post_moe_residual", "router_logits")
EXPECTED_CONTROLS = ("unpatched", "unpatched_rerun", "self_patch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-profile",
        choices=(*PROFILE_NAMES, "all"),
        default="all",
        help="Model to execute; dry-run defaults to all three profiles.",
    )
    parser.add_argument("--run-role", choices=RUN_ROLES, default="full")
    parser.add_argument(
        "--profile-config",
        type=Path,
        default=Path("configs/analysis/processbench_natural_transfer.yaml"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="New artifact root; only valid when one model profile is selected.",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="Optional JSON plan output; unlike the run root this may be written in dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local pins and print the exact plan without probing CUDA or loading weights.",
    )
    return parser


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _routeverify_command(*args: str) -> list[str]:
    return ["uv", "run", "--frozen", "--all-extras", "routeverify", *args]


def _require_equal(*, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def _authorization_env(profile: dict[str, Any], run_role: str) -> str:
    authorization = profile.get("authorization_env")
    if isinstance(authorization, dict):
        value = authorization.get(run_role)
    else:
        value = authorization
    if not isinstance(value, str) or not value:
        raise ValueError(f"profile lacks a {run_role} authorization environment variable")
    return value


def _row_contract(pair_count: int, *, num_sites: int) -> dict[str, int]:
    return {
        "behavior": 2 * pair_count,
        "recapture": 2 * pair_count,
        # 4N shared baseline/rerun rows plus 4N per site (self + bidirectional transfer).
        "transfer": (4 + 4 * num_sites) * pair_count,
    }


def _build_profile_plan(
    *,
    repo_root: Path,
    gate_path: Path,
    gate: dict[str, Any],
    profile_name: str,
    run_role: str,
    run_root_override: Path | None,
) -> dict[str, Any]:
    _require_equal(label="schema_version", actual=gate.get("schema_version"), expected=1)
    if gate.get("validation_accessed") is not False:
        raise ValueError("analysis config must record validation_accessed=false")
    _require_equal(
        label="sites",
        actual=tuple(gate.get("sites", ())),
        expected=EXPECTED_SITES,
    )
    _require_equal(
        label="controls",
        actual=tuple(gate.get("controls", ())),
        expected=EXPECTED_CONTROLS,
    )
    _require_equal(
        label="alphas",
        actual=[float(value) for value in gate.get("alphas", ())],
        expected=[1.0],
    )

    run_roles = gate.get("run_roles")
    profiles = gate.get("profiles")
    if not isinstance(run_roles, dict) or not isinstance(profiles, dict):
        raise ValueError("analysis config must define run_roles and profiles mappings")
    role = run_roles.get(run_role)
    profile = profiles.get(profile_name)
    if not isinstance(role, dict) or not isinstance(profile, dict):
        raise ValueError(f"missing {run_role}/{profile_name} protocol entry")
    pair_count = int(role["pair_count"])
    stage_rows = _row_contract(pair_count, num_sites=len(EXPECTED_SITES))
    expected_rows = {**stage_rows, "total": sum(stage_rows.values())}
    configured_rows = role.get("expected_rows")
    if configured_rows is not None:
        _require_equal(
            label=f"{run_role}.expected_rows",
            actual={key: int(value) for key, value in configured_rows.items()},
            expected=expected_rows,
        )

    data_path = _resolve(repo_root, str(gate["data_config"]))
    data_config = _load_mapping(data_path)
    _require_equal(
        label="data dataset",
        actual=data_config.get("dataset"),
        expected="processbench_transfer",
    )
    if data_config.get("validation_accessed") is not False:
        raise ValueError("data config must record validation_accessed=false")
    processed_path = _resolve(repo_root, str(data_config["output"]))
    selection_path = _resolve(repo_root, str(data_config["selection_manifest"]))
    bundle = load_processbench_transfer_bundle(
        data_path=processed_path,
        manifest_path=selection_path,
        expected_data_sha256=str(data_config["expected_processed_sha256"]),
        expected_manifest_sha256=str(
            data_config["expected_selection_manifest_sha256"]
        ),
    )
    _require_equal(
        label="analysis/data protocol_id",
        actual=gate.get("protocol_id"),
        expected=bundle.manifest.get("protocol_id"),
    )
    _require_equal(
        label="analysis/data pairing_rule_id",
        actual=gate.get("pairing_rule_id"),
        expected=bundle.manifest["selection"].get("pairing_rule_id"),
    )
    _require_equal(
        label="data revision",
        actual=data_config.get("revision"),
        expected=bundle.manifest["source"].get("revision"),
    )
    raw_hashes = {
        str(row["sha256"])
        for row in bundle.manifest["source"]["files"]
        if isinstance(row, dict) and "sha256" in row
    }
    if str(data_config.get("expected_raw_sha256")) not in raw_hashes:
        raise ValueError("data config raw SHA256 is absent from the frozen manifest")
    manifest_role = bundle.manifest["run_roles"][run_role]
    _require_equal(
        label=f"manifest {run_role} pair_count",
        actual=int(manifest_role["pair_count"]),
        expected=pair_count,
    )
    selected_pair_ids = list(manifest_role["pair_ids"])
    if len(selected_pair_ids) != pair_count or len(set(selected_pair_ids)) != pair_count:
        raise ValueError(f"manifest {run_role} pair IDs are not exact and unique")

    model_path = _resolve(repo_root, str(profile["model_config"]))
    model_config = _load_mapping(model_path)
    for field in ("adapter_alias", "model_id", "revision"):
        model_field = "alias" if field == "adapter_alias" else field
        _require_equal(
            label=f"{profile_name}.{field}",
            actual=model_config.get(model_field),
            expected=profile.get(field),
        )
    _require_equal(
        label=f"{profile_name}.tokenizer_revision",
        actual=model_config.get("tokenizer_revision"),
        expected=profile.get("tokenizer_revision"),
    )
    _require_equal(
        label=f"{profile_name}.sites",
        actual=tuple(profile.get("sites", ())),
        expected=EXPECTED_SITES,
    )
    if model_config.get("load_weights") is not False:
        raise ValueError(f"committed model config must keep load_weights=false: {model_path}")
    anchor_layer = int(profile["anchor_layer"])
    if anchor_layer < 0:
        raise ValueError("anchor_layer must be non-negative")

    if run_root_override is None:
        run_root = (
            repo_root
            / "artifacts"
            / "processbench_natural_transfer"
            / run_role
            / profile_name
        )
    else:
        run_root = _resolve(repo_root, run_root_override)
    stage_outputs = {
        "runtime": run_root / "runtime",
        "behavior": run_root / "behavior",
        "recapture": run_root / "recapture",
        "transfer": run_root / "transfer",
    }
    resolved_dir = run_root / "resolved_configs"
    recapture_config_path = resolved_dir / "recapture.yaml"
    transfer_config_path = resolved_dir / "transfer.yaml"
    runtime_report = stage_outputs["runtime"] / "runtime_preflight.json"
    review_path = run_root / f"processbench_transfer_{run_role}_review.json"

    shared_experiment = {
        "model": str(model_path),
        "data": str(data_path),
        "seed": int(gate["seed"]),
        "limit": pair_count,
        "prompt_mode": "step_binary",
        "input_format": "chat",
        "boundary_strategy": "paired_first_error",
        "sites": list(EXPECTED_SITES),
        "layers": [anchor_layer],
    }
    recapture_config = {
        **shared_experiment,
        "output_dir": str(stage_outputs["recapture"]),
    }
    transfer_config = {
        **shared_experiment,
        "output_dir": str(stage_outputs["transfer"]),
        "patch_mode": "single_site_sweep",
        "alphas": [1.0],
        "controls": list(EXPECTED_CONTROLS),
        "include_equivalence_analysis": False,
        "identity_tolerance": float(gate["identity_tolerance"]),
    }

    commands = [
        _routeverify_command(
            "model",
            "runtime-preflight",
            "--model",
            str(model_path),
            "--output",
            str(runtime_report),
        ),
        _routeverify_command(
            "eval",
            "behavior",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--mode",
            "step-binary",
            "--limit",
            str(pair_count),
            "--seed",
            str(gate["seed"]),
            "--output-dir",
            str(stage_outputs["behavior"]),
        ),
        _routeverify_command(
            "extract",
            "--experiment",
            str(recapture_config_path),
            "--limit",
            str(pair_count),
            "--seed",
            str(gate["seed"]),
            "--output-dir",
            str(stage_outputs["recapture"]),
        ),
        _routeverify_command(
            "intervene",
            "patch",
            "--experiment",
            str(transfer_config_path),
            "--limit",
            str(pair_count),
            "--seed",
            str(gate["seed"]),
        ),
        [
            "uv",
            "run",
            "--frozen",
            "--all-extras",
            "python",
            str(repo_root / "scripts" / "review_processbench_natural_transfer.py"),
            "--profile-config",
            str(gate_path),
            "--model-profile",
            profile_name,
            "--run-role",
            run_role,
            "--runtime-report",
            str(runtime_report),
            "--behavior-run",
            str(stage_outputs["behavior"]),
            "--recapture-run",
            str(stage_outputs["recapture"]),
            "--transfer-run",
            str(stage_outputs["transfer"]),
            "--output",
            str(review_path),
        ],
    ]
    return {
        "profile": profile_name,
        "run_role": run_role,
        "model": {
            "adapter_alias": profile["adapter_alias"],
            "model_id": profile["model_id"],
            "revision": profile["revision"],
            "tokenizer_revision": profile["tokenizer_revision"],
            "model_config": str(model_path),
            "model_config_sha256": _sha256_file(model_path),
            "anchor_layer": anchor_layer,
            "anchor_coordinate": profile["anchor_coordinate"],
            "anchor_router_ordinal": profile.get("anchor_router_ordinal"),
            "anchor_site_ids": [
                f"layer_{anchor_layer}.{site}" for site in EXPECTED_SITES
            ],
            "normalized_depth": float(profile["normalized_depth"]),
            "router_semantics_id": profile["router_semantics_id"],
        },
        "authorization_env": _authorization_env(profile, run_role),
        "run_root": str(run_root),
        "pair_count": pair_count,
        "selected_pair_ids": selected_pair_ids,
        "expected_rows": expected_rows,
        "expected_total_rows": expected_rows["total"],
        "review_output": str(review_path),
        "resolved_configs": {
            "recapture_path": str(recapture_config_path),
            "recapture": recapture_config,
            "transfer_path": str(transfer_config_path),
            "transfer": transfer_config,
        },
        "commands": commands,
        "profile_config": str(gate_path),
        "profile_config_sha256": _sha256_file(gate_path),
    }


def _build_plan(
    *,
    repo_root: Path,
    gate_path: Path,
    gate: dict[str, Any],
    profile_names: tuple[str, ...],
    run_role: str,
    run_root: Path | None,
) -> dict[str, Any]:
    if run_root is not None and len(profile_names) != 1:
        raise ValueError("--run-root requires one --model-profile")
    profiles = [
        _build_profile_plan(
            repo_root=repo_root,
            gate_path=gate_path,
            gate=gate,
            profile_name=profile_name,
            run_role=run_role,
            run_root_override=run_root,
        )
        for profile_name in profile_names
    ]
    shared_contract = {
        "protocol_id": gate["protocol_id"],
        "seed": int(gate["seed"]),
        "data_config": str(_resolve(repo_root, str(gate["data_config"]))),
        "sites": list(EXPECTED_SITES),
        "alphas": [1.0],
        "controls": list(EXPECTED_CONTROLS),
        "prompt_mode": "step_binary",
        "boundary_strategy": "paired_first_error",
        "pairing_rule_id": gate["pairing_rule_id"],
        "validation_accessed": False,
    }
    data_config = _load_mapping(Path(shared_contract["data_config"]))
    data_path = _resolve(repo_root, str(data_config["output"]))
    selection_path = _resolve(repo_root, str(data_config["selection_manifest"]))
    source_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "cpu_preflight_only",
        "run_role": run_role,
        "shared_protocol": shared_contract,
        "shared_protocol_hash": stable_hash(shared_contract, length=64),
        "data": {
            "hf_id": source_manifest["source"]["hf_id"],
            "revision": source_manifest["source"]["revision"],
            "hf_config": source_manifest["source"]["hf_config"],
            "source_files": source_manifest["source"]["files"],
            "source_trace_count": source_manifest["counts"]["source_rows"],
            "source_counts_by_subset": source_manifest["counts"]["by_subset"],
            "selected_pair_count": source_manifest["counts"]["selected_pairs"],
            "selected_trace_count": source_manifest["counts"]["selected_traces"],
            "natural_error_boundary_count": source_manifest["counts"]
            ["natural_error_boundaries"],
            "mapped_correct_boundary_count": source_manifest["counts"]
            ["mapped_correct_boundaries"],
            "processed_path": str(data_path),
            "processed_sha256": _sha256_file(data_path),
            "selection_manifest_path": str(selection_path),
            "selection_manifest_sha256": _sha256_file(selection_path),
            "selection_manifest_hash": source_manifest["manifest_hash"],
            "run_roles": source_manifest["run_roles"],
        },
        "profiles": profiles,
        "symmetry": {
            "passed": True,
            "only_allowed_profile_differences": [
                "adapter_alias",
                "model_id",
                "revision",
                "model_config",
                "anchor_layer",
                "normalized_depth",
                "router_semantics_id",
                "authorization_env",
                "output_path",
            ],
        },
        "minimum_provenance": gate.get("minimum_provenance", []),
        "execution_policy": gate.get("execution_policy", {}),
        "validation_accessed": False,
        "cuda_accessed": False,
        "weights_loaded": False,
        "network_accessed": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _execute_profile(*, repo_root: Path, profile_plan: dict[str, Any]) -> None:
    authorization_env = str(profile_plan["authorization_env"])
    run_role = str(profile_plan["run_role"])
    if os.environ.get("ALLOW_GPU_RUN") != "1" or os.environ.get(
        authorization_env
    ) != "1":
        raise SystemExit(
            f"refusing GPU {run_role} without ALLOW_GPU_RUN=1 and "
            f"{authorization_env}=1"
        )

    run_root = Path(profile_plan["run_root"])
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"refusing non-empty run root: {run_root}")
    resolved = profile_plan["resolved_configs"]
    recapture_path = Path(resolved["recapture_path"])
    transfer_path = Path(resolved["transfer_path"])
    recapture_path.parent.mkdir(parents=True, exist_ok=True)
    recapture_path.write_text(
        yaml.safe_dump(resolved["recapture"], sort_keys=False),
        encoding="utf-8",
    )
    transfer_path.write_text(
        yaml.safe_dump(resolved["transfer"], sort_keys=False),
        encoding="utf-8",
    )
    run_plan = {
        "schema_version": 1,
        "profile": profile_plan["profile"],
        "run_role": run_role,
        "expected_rows": profile_plan["expected_rows"],
        "commands": profile_plan["commands"],
        "validation_accessed": False,
    }
    _write_json(run_root / "run_plan.json", run_plan)
    for command in profile_plan["commands"]:
        subprocess.run(command, cwd=repo_root, check=True)


def main() -> None:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    gate_path = _resolve(repo_root, args.profile_config)
    gate = _load_mapping(gate_path)
    profile_names = (
        PROFILE_NAMES if args.model_profile == "all" else (args.model_profile,)
    )
    plan = _build_plan(
        repo_root=repo_root,
        gate_path=gate_path,
        gate=gate,
        profile_names=profile_names,
        run_role=args.run_role,
        run_root=args.run_root,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    for profile in plan["profiles"]:
        for command in profile["commands"]:
            print(shlex.join(command))
    if args.plan_output is not None:
        _write_json(_resolve(repo_root, args.plan_output), plan)
    if args.dry_run:
        return
    if len(plan["profiles"]) != 1:
        raise SystemExit("actual execution requires one --model-profile")
    _execute_profile(repo_root=repo_root, profile_plan=plan["profiles"][0])


if __name__ == "__main__":
    main()
