"""Artifact-only gates for the fourth/fifth-model common-core smoke runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from routeverify.config import ExperimentConfig, ModelConfig
from routeverify.data.schema import stable_hash
from routeverify.eval.verification import binary_verification_metrics
from routeverify.instrumentation.router_weights import load_router_weight_export

RECAPTURE_SITE_KINDS = (
    "post_moe_residual",
    "router_input",
    "router_logits",
    "router_topk_ids",
    "router_topk_weights",
)
ENDPOINT_SITE_KINDS = ("post_moe_residual", "moe_output", "router_logits")
ROUTER_CONTROLS = (
    "unpatched",
    "unpatched_rerun",
    "self_patch",
    "alpha_zero_identity",
    "forced_reroute_if_needed",
)
ROUTER_FULL_CONTROLS = ROUTER_CONTROLS[:-1]
ENDPOINT_CONTROLS = (
    "unpatched",
    "unpatched_rerun",
    "self_patch",
    "norm_random_delta",
    "pre_error_when_available",
    "random_matched_donor",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return cast(dict[str, Any], payload)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passed_smoke_prerequisite(
    *,
    path: Path,
    model_profile: str,
    expected_sha256: str,
    expected_commit: str,
    execution_source_sha256: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"passed smoke prerequisite is unavailable: {path}")
    observed_sha256 = _sha256_file(path)
    payload = _read_json(path)
    checks = payload.get("checks")
    environment_check = next(
        (
            item
            for item in checks or []
            if isinstance(item, Mapping)
            and item.get("name") == "same clean model environment and no validation"
        ),
        None,
    )
    git_states = (
        environment_check.get("details", {}).get("git", [])
        if isinstance(environment_check, Mapping)
        else []
    )
    if execution_source_sha256 is not None:
        required_checks = {
            "runtime preflight", "model-specific runtime fidelity",
            "same valid discovery selection", "same clean model environment and no validation",
            "behavior exact coverage and recomputed capability",
            "recapture exact site and activation contract",
            "router rows, identity, hooks, and native dispatch",
            "router weight schema-v2 topology and hash",
            "endpoint rows, identity, hooks, controls, and router dispatch",
        }
        runtime_check = next((item for item in checks or [] if isinstance(item, Mapping)
                              and item.get("name") == "runtime preflight"), {})
        runtime = runtime_check.get("details", {})
        fidelity = payload.get("identity_dispatch_provenance", {})
        if not (
            required_checks <= {item.get("name") for item in checks or [] if isinstance(item, Mapping)}
            and
            payload.get("campaign_policy") == "glm_common_core_20260907_v1"
            and payload.get("run_role") == "smoke"
            and runtime.get("adapter_alias") == "glm4_moe_lite"
            and runtime.get("model_id") == "zai-org/GLM-4.7-Flash"
            and runtime.get("revision") == "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
            and runtime.get("environment", {}).get("cuda", {}).get("cuda_available") is True
            and runtime.get("environment", {}).get("execution_source_sha256")
            == execution_source_sha256
            and _model_specific_runtime_ok("glm4_moe_lite", runtime.get("runtime_inspect", {}))
            and fidelity.get("router_identity_dispatch_passed") is True
            and fidelity.get("endpoint_identity_dispatch_passed") is True
        ):
            raise ValueError("GLM full requires a real pinned CUDA checkpoint technical smoke")
    valid = (
        payload.get("schema_version") == 1
        and payload.get("passed") is True
        and payload.get("model_profile") == model_profile
        and payload.get("expected_pairs") == 8
        and payload.get("validation_accessed") is False
        and isinstance(checks, list)
        and bool(checks)
        and all(
            isinstance(item, Mapping) and item.get("passed") is True for item in checks
        )
        and observed_sha256 == expected_sha256
        and (
            execution_source_sha256 is None
            or payload.get("execution_source_sha256") == execution_source_sha256
        )
        and isinstance(git_states, list)
        and len(git_states) == 4
        and (execution_source_sha256 is None or all(state == git_states[0] for state in git_states))
        and all(
            isinstance(state, Mapping)
            and (execution_source_sha256 is not None or state.get("commit") == expected_commit)
            and (execution_source_sha256 is not None or state.get("dirty") is False)
            for state in git_states
        )
    )
    if not valid:
        raise ValueError(
            "full discovery requires the exact passed n=8 smoke review at the "
            "pinned clean commit"
        )
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "commit": git_states[0].get("commit") if execution_source_sha256 is not None else expected_commit,
        "execution_source_sha256": execution_source_sha256,
        "model_profile": model_profile,
        "expected_pairs": 8,
        "passed": True,
        "validation_accessed": False,
    }


def resolve_common_core_stage_payload(
    *,
    source: Path,
    output_dir: Path,
    seed: int,
    limit: int,
    split_manifest: Path,
    split_partition: str,
) -> dict[str, Any]:
    """Bind runner-wide selection inputs identically across experiment stages."""

    payload = _read_yaml(source)
    payload["output_dir"] = str(output_dir)
    payload["seed"] = seed
    payload["limit"] = limit
    payload["split_manifest"] = str(split_manifest)
    payload["split_partition"] = split_partition
    return payload


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _model_specific_runtime_ok(
    adapter_alias: str,
    runtime_inspect: Mapping[str, Any],
) -> bool:
    """Apply only architecture-necessary runtime gates on top of common core."""

    if adapter_alias == "nemotron35_lightning":
        loading_info = runtime_inspect.get("loading_info")
        mamba_fast_path = runtime_inspect.get("mamba_fast_path")
        return bool(
            runtime_inspect.get("lightning_latent_projections_are_identity") is True
            and runtime_inspect.get("mtp_in_main_decoder_sites") is False
            and not runtime_inspect.get("mtp_runtime_modules")
            and runtime_inspect.get("mtp_ignore_pattern_present") is True
            and runtime_inspect.get("loading_info_ok") is True
            and isinstance(loading_info, Mapping)
            and loading_info.get("only_expected_mtp_ignored") is True
            and isinstance(mamba_fast_path, Mapping)
            and mamba_fast_path.get("ready") is True
            and mamba_fast_path.get("single_device") is True
            and len(mamba_fast_path.get("layers", [])) == 23
        )
    if adapter_alias == "qwen36_moe":
        loading_info = runtime_inspect.get("loading_info")
        return bool(
            runtime_inspect.get("runtime_model_type") == "qwen3_5_moe_text"
            and runtime_inspect.get("runtime_architecture")
            == "Qwen3_5MoeForCausalLM"
            and runtime_inspect.get("all_text_layers_routed") is True
            and len(runtime_inspect.get("layers", [])) == 40
            and runtime_inspect.get("loading_info_ok") is True
            and isinstance(loading_info, Mapping)
            and loading_info.get("only_expected_composite_auxiliary_ignored")
            is True
            and loading_info.get("mtp_ignore_pattern_present") is True
            and loading_info.get("visual_ignore_pattern_present") is True
            and loading_info.get("schema_valid") is True
            and loading_info.get("exact_ignore_patterns") is True
        )
    if adapter_alias == "glm4_moe_lite":
        layers = runtime_inspect.get("layers", [])
        return bool(
            runtime_inspect.get("runtime_architecture") == "Glm4MoeLiteForCausalLM"
            and runtime_inspect.get("loading_info_ok") is True
            and runtime_inspect.get("allowed_auxiliary_prefix") == "model.layers.47."
            and runtime_inspect.get("router_block_indices") == list(range(1, 47))
            and len(layers) == 46
            and all(
                row.get("experts") == 64 and row.get("top_k") == 4
                and row.get("correction_bias_dtype") == "torch.float32"
                and all(row.get(key) == "torch.bfloat16" for key in (
                    "gate_dtype", "experts_dtype", "shared_dtype"
                ))
                for row in layers
            )
        )
    return True


def build_common_core_smoke_contract(
    *,
    gate_config: Mapping[str, Any],
    model_profile: str,
    root: Path = Path("."),
    prerequisite_smoke_review: Path | None = None,
    planning_only: bool = False,
) -> dict[str, Any]:
    """Resolve a data-driven model-4/5 common-core profile without model I/O."""

    if gate_config.get("schema_version") != 1:
        raise ValueError("common-core smoke gate must use schema_version=1")
    run_role = str(gate_config.get("run_role", "smoke"))
    if run_role not in {"smoke", "full_discovery"}:
        raise ValueError("common-core run_role must be smoke or full_discovery")
    profiles = _as_mapping(gate_config.get("profiles"), label="profiles")
    if model_profile not in profiles:
        raise ValueError(f"unknown common-core smoke profile: {model_profile}")
    profile = _as_mapping(profiles[model_profile], label=f"profiles.{model_profile}")
    campaign_policy = gate_config.get("campaign_policy")
    if campaign_policy is not None and (
        campaign_policy != "glm_common_core_20260907_v1" or model_profile != "glm4_moe_lite"
    ):
        raise ValueError("unknown or mismatched common-core campaign policy")
    campaign = campaign_policy is not None
    execution_source_sha256 = gate_config.get("execution_source_sha256") if campaign else None
    if execution_source_sha256 is not None and (
        len(str(execution_source_sha256)) != 64
        or any(c not in "0123456789abcdef" for c in str(execution_source_sha256))
    ):
        raise ValueError("invalid execution source SHA256")
    stages = _as_mapping(profile.get("stages"), label=f"profiles.{model_profile}.stages")
    if set(stages) != {"recapture", "router", "endpoint"}:
        raise ValueError("common-core smoke stages must be recapture, router, and endpoint")

    model_path = Path(str(profile["model_config"]))
    data_path = Path(str(gate_config["data_config"]))
    split_path = Path(str(gate_config["split_manifest"]))
    resolved_split_path = (root / split_path).resolve()
    model_config = ModelConfig.model_validate(_read_yaml(root / model_path))
    expected_pairs = int(gate_config["expected_pairs"])
    expected_behavior_pairs = int(gate_config["expected_behavior_pairs"])
    min_behavior_auroc = float(gate_config["behavior_min_auroc"])
    seed = int(profile["seed"])
    router_layers = [int(value) for value in profile["router_block_indices"]]
    endpoint_layers = [int(value) for value in profile["endpoint_layers"]]
    artifact_contract = _as_mapping(
        profile.get("artifact_contract"),
        label=f"profiles.{model_profile}.artifact_contract",
    )
    smoke_prerequisite: dict[str, Any] | None = None
    if run_role == "full_discovery":
        if prerequisite_smoke_review is None and not planning_only:
            raise ValueError("full discovery requires --prerequisite-smoke-review")
    if run_role == "full_discovery" and prerequisite_smoke_review is not None:
        prerequisite_path = prerequisite_smoke_review
        if not prerequisite_path.is_absolute():
            prerequisite_path = root / prerequisite_path
        smoke_prerequisite = _passed_smoke_prerequisite(
            path=prerequisite_path.resolve(),
            model_profile=model_profile,
            expected_sha256=str(
                profile.get(
                    "prerequisite_smoke_review_sha256",
                    gate_config.get("prerequisite_smoke_review_sha256", ""),
                )
            ),
            execution_source_sha256=execution_source_sha256,
            expected_commit=str(
                profile.get(
                    "prerequisite_smoke_commit",
                    gate_config.get("prerequisite_smoke_commit", ""),
                )
            ),
        )

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    split_payload = _read_json(resolved_split_path)
    split_unhashed = dict(split_payload)
    recorded_split_hash = split_unhashed.pop("manifest_hash", None)
    pinned_split_hash = str(gate_config["split_manifest_hash"])
    discovery_group_ids = [
        str(value) for value in split_payload.get("discovery_group_ids", [])
    ]
    validation_group_ids = {
        str(value) for value in split_payload.get("validation_group_ids", [])
    }
    expected_discovery_group_ids = discovery_group_ids[:expected_pairs]
    expected_pair_ids = [
        group_id.split(":pair:", maxsplit=1)[1]
        if ":pair:" in group_id
        else group_id
        for group_id in expected_discovery_group_ids
    ]

    require(model_config.alias == profile.get("adapter_alias"), "adapter alias mismatch")
    require(model_config.model_id == profile.get("model_id"), "model id mismatch")
    require(model_config.revision == profile.get("revision"), "model revision mismatch")
    require(
        model_config.tokenizer_revision == profile.get("revision"),
        "tokenizer revision mismatch",
    )
    require(
        model_config.expected_router_block_count == len(router_layers),
        "router block count mismatch",
    )
    require(
        model_config.expected_total_block_count is not None,
        "total block count must be pinned",
    )
    if model_config.expected_router_block_indices is not None:
        require(
            model_config.expected_router_block_indices == router_layers,
            "router block indices mismatch",
        )
    require(
        gate_config.get("split_partition") == "discovery",
        "common-core run must use discovery",
    )
    require(
        recorded_split_hash == pinned_split_hash,
        "split manifest does not match pinned hash",
    )
    require(
        recorded_split_hash == stable_hash(split_unhashed, length=64),
        "split manifest self-hash is invalid",
    )
    require(
        len(discovery_group_ids) >= expected_pairs
        and len(set(discovery_group_ids)) == len(discovery_group_ids)
        and not (set(discovery_group_ids) & validation_group_ids),
        "split manifest discovery IDs are insufficient, duplicated, or overlap validation",
    )
    require(
        len(expected_pair_ids) == expected_pairs
        and len(set(expected_pair_ids)) == expected_pairs,
        "split manifest first-N discovery pair IDs are invalid",
    )
    if run_role == "smoke":
        require(expected_pairs == 8, "common-core smoke must use exactly 8 pairs")
        require(expected_behavior_pairs == 8, "smoke behavior must use exactly 8 pairs")
    else:
        require(
            model_profile
            in {
                "gemma4",
                "nemotron3_nano",
                "nemotron35_lightning",
                "qwen36_moe",
                "glm4_moe_lite",
            },
            "full discovery model profile is not registered",
        )
        require(expected_pairs == 290, "full discovery must use exactly 290 pairs")
        require(
            expected_behavior_pairs == 64,
            "full discovery behavior must use exactly 64 pairs",
        )
    require(
        math.isfinite(min_behavior_auroc) and 0.5 <= min_behavior_auroc <= 1.0,
        "behavior AUROC minimum must be finite and in [0.5, 1.0]",
    )
    expected_authorization_env = (
        {
            "gemma4": "ALLOW_GEMMA4_COMMON_CORE_SMOKE",
            "nemotron3_nano": "ALLOW_NEMOTRON3_NANO_COMMON_CORE_SMOKE",
            "nemotron35_lightning": "ALLOW_NEMOTRON35_LIGHTNING_COMMON_CORE_SMOKE",
            "qwen36_moe": "ALLOW_QWEN36_COMMON_CORE_SMOKE",
            "glm4_moe_lite": "ALLOW_GLM4_COMMON_CORE_SMOKE",
        }.get(model_profile)
        if run_role == "smoke"
        else {
            "gemma4": "ALLOW_GEMMA4_COMMON_CORE_FULL",
            "nemotron3_nano": "ALLOW_NEMOTRON3_NANO_COMMON_CORE_FULL",
            "nemotron35_lightning": "ALLOW_NEMOTRON35_LIGHTNING_COMMON_CORE_FULL",
            "qwen36_moe": "ALLOW_QWEN36_COMMON_CORE_FULL",
            "glm4_moe_lite": "ALLOW_GLM4_COMMON_CORE_FULL",
        }.get(model_profile)
    )
    require(
        profile.get("authorization_env") == expected_authorization_env,
        "model-specific authorization environment variable mismatch",
    )
    require(
        artifact_contract.get("patch_diagnostics_schema") == 1,
        "patch diagnostics schema must be 1",
    )
    require(
        artifact_contract.get("require_candidate_patch_diagnostics") is True,
        "candidate patch diagnostics must be required",
    )
    require(
        artifact_contract.get("require_router_dispatch_for_site_kinds")
        == ["router_logits"],
        "router dispatch must be required for router_logits",
    )
    require(
        artifact_contract.get("require_capture_site_contract") is True,
        "capture site contract must be required",
    )
    require(
        artifact_contract.get("router_weight_schema_version") == 2,
        "router weight schema must be 2",
    )

    stage_configs: dict[str, ExperimentConfig] = {}
    stage_paths: dict[str, Path] = {}
    allowed_experiment_keys = set(ExperimentConfig.model_fields)
    for stage_name, raw_stage_path in stages.items():
        stage_path = Path(str(raw_stage_path))
        raw_config = _read_yaml(root / stage_path)
        unknown_keys = sorted(set(raw_config) - allowed_experiment_keys)
        require(not unknown_keys, f"{stage_name} has unknown keys: {unknown_keys}")
        config = ExperimentConfig.model_validate(raw_config)
        stage_paths[str(stage_name)] = stage_path
        stage_configs[str(stage_name)] = config
        require(Path(config.model) == model_path, f"{stage_name} model config mismatch")
        require(Path(config.data) == data_path, f"{stage_name} data config mismatch")
        require(config.seed == seed, f"{stage_name} seed mismatch")
        require(config.limit == expected_pairs, f"{stage_name} pair limit mismatch")
        require(config.prompt_mode == "step_binary", f"{stage_name} prompt mode mismatch")
        require(config.input_format == "chat", f"{stage_name} input format mismatch")
        require(
            config.boundary_strategy == "paired_first_error",
            f"{stage_name} boundary mismatch",
        )
        require(
            config.split_manifest == split_path,
            f"{stage_name} split manifest mismatch",
        )
        require(config.split_partition == "discovery", f"{stage_name} must use discovery")

    recapture = stage_configs["recapture"]
    router = stage_configs["router"]
    endpoint = stage_configs["endpoint"]
    require(recapture.sites == list(RECAPTURE_SITE_KINDS), "recapture site kinds mismatch")
    require(recapture.layers == router_layers, "recapture router layers mismatch")
    require(router.patch_mode == "simultaneous_bundle", "router patch mode mismatch")
    require(router.sites == ["router_logits"], "router site kinds mismatch")
    require(router.layers == router_layers, "router layers mismatch")
    require(router.expected_router_hooks == len(router_layers), "router hook count mismatch")
    require(router.alphas == [0.0, 0.5, 1.0, 2.0], "router alphas mismatch")
    expected_forced_reroute = run_role == "smoke" and not campaign
    expected_router_controls = (
        list(ROUTER_CONTROLS)
        if expected_forced_reroute
        else list(ROUTER_FULL_CONTROLS)
    )
    require(
        router.forced_reroute_control is expected_forced_reroute,
        "router forced control role mismatch",
    )
    require(router.controls == expected_router_controls, "router controls mismatch")
    require(router.export_router_weights == Path("router_weights.npz"), "router export mismatch")
    require(router.router_weight_schema_version == 2, "router export schema mismatch")
    require(endpoint.patch_mode == "single_site_sweep", "endpoint patch mode mismatch")
    require(endpoint.sites == list(ENDPOINT_SITE_KINDS), "endpoint site kinds mismatch")
    require(endpoint.layers == endpoint_layers, "endpoint layers mismatch")
    expected_endpoint_controls = [
        control for control in ENDPOINT_CONTROLS
        if not campaign or control != "pre_error_when_available"
    ]
    require(endpoint.controls == expected_endpoint_controls, "endpoint controls mismatch")
    if campaign:
        require(router_layers == list(range(1, 47)), "GLM routed indices mismatch")
        # Same upper-middle routed ordinal as the new window protocol; the last
        # routed endpoint is omitted under the accepted structural-zero decision.
        require(endpoint_layers == [router_layers[0], router_layers[len(router_layers) // 2]],
                "GLM first/upper-middle routed endpoint rule mismatch")
        require(model_config.model_id == "zai-org/GLM-4.7-Flash", "GLM identity mismatch")
        require(model_config.revision == "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
                "GLM revision mismatch")

    output_dirs = {config.output_dir for config in stage_configs.values()}
    require(len(output_dirs) == 3, "stage output directories must be distinct")

    if errors:
        raise ValueError(f"invalid common-core {run_role} profile: " + "; ".join(errors))
    stage_payloads = {
        key: value.model_dump(mode="json") for key, value in stage_configs.items()
    }
    return {
        "schema_version": 1,
        "run_role": run_role,
        "model_profile": model_profile,
        "adapter_alias": model_config.alias,
        "adapter_class": str(profile["adapter_class"]),
        "model_id": model_config.model_id,
        "revision": model_config.revision,
        "tokenizer_revision": model_config.tokenizer_revision,
        "model_config": str(model_path),
        "data_config": str(data_path),
        "split_manifest": str(split_path),
        "resolved_split_manifest": str(resolved_split_path),
        "split_manifest_hash": pinned_split_hash,
        "split_partition": "discovery",
        "expected_discovery_group_ids": expected_discovery_group_ids,
        "expected_pair_ids": expected_pair_ids,
        "seed": seed,
        "expected_pairs": expected_pairs,
        "expected_behavior_pairs": expected_behavior_pairs,
        "behavior_min_auroc": min_behavior_auroc,
        "behavior_is_gate": not campaign,
        "campaign_policy": campaign_policy,
        "execution_source_sha256": execution_source_sha256,
        "pre_error_control": not campaign,
        "forced_reroute_control": expected_forced_reroute,
        "identity_tolerance": float(gate_config.get("identity_tolerance", 1e-5)),
        "authorization_env": str(profile["authorization_env"]),
        "gate_config_hash": stable_hash(dict(gate_config), length=64),
        "stage_config_hashes": {
            key: stable_hash(value, length=64) for key, value in stage_payloads.items()
        },
        "prerequisite_smoke_review": smoke_prerequisite,
        "total_block_count": model_config.expected_total_block_count,
        "router_semantics_id": str(profile["router_semantics_id"]),
        "router_block_indices": router_layers,
        "endpoint_layers": endpoint_layers,
        "recapture_site_kinds": list(RECAPTURE_SITE_KINDS),
        "endpoint_site_kinds": list(ENDPOINT_SITE_KINDS),
        "stage_paths": {key: str(value) for key, value in stage_paths.items()},
        "stage_configs": stage_payloads,
        "artifact_contract": dict(artifact_contract),
    }


def _selection_valid(selection: Mapping[str, Any]) -> bool:
    unhashed = dict(selection)
    recorded = unhashed.pop("manifest_hash", None)
    return bool(recorded) and recorded == stable_hash(unhashed, length=64)


def _path_matches_contract(value: Any, expected: str) -> bool:
    observed = Path(str(value)).as_posix()
    canonical = Path(expected).as_posix()
    return observed == canonical or observed.endswith(f"/{canonical}")


def _scores_finite(row: Mapping[str, Any]) -> bool:
    scores = row.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != {"CORRECT", "INCORRECT"}:
        return False
    try:
        return all(
            isinstance(scores[candidate], Mapping)
            and scores[candidate].get("candidate") == candidate
            and int(scores[candidate]["token_count"]) > 0
            and math.isfinite(float(scores[candidate][field]))
            for candidate in ("CORRECT", "INCORRECT")
            for field in ("mean_logprob", "total_logprob")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _behavior_row_valid(row: Mapping[str, Any]) -> bool:
    if not _scores_finite(row):
        return False
    try:
        scores = cast(Mapping[str, Mapping[str, Any]], row["scores"])
        recorded_margin = float(row["margin_incorrect_minus_correct"])
        recorded_mean_margin = float(row["margin_incorrect_minus_correct_mean"])
        recorded_total_margin = float(row["margin_incorrect_minus_correct_total"])
        score_mean_margin = float(scores["INCORRECT"]["mean_logprob"]) - float(
            scores["CORRECT"]["mean_logprob"]
        )
        score_total_margin = float(scores["INCORRECT"]["total_logprob"]) - float(
            scores["CORRECT"]["total_logprob"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    expected_prediction = "INCORRECT" if recorded_margin > 0.0 else "CORRECT"
    return (
        math.isfinite(recorded_margin)
        and math.isfinite(recorded_mean_margin)
        and math.isfinite(recorded_total_margin)
        and abs(recorded_margin - score_mean_margin) <= 1e-12
        and abs(recorded_mean_margin - score_mean_margin) <= 1e-12
        and abs(recorded_total_margin - score_total_margin) <= 1e-12
        and row.get("primary_score") == "mean_logprob"
        and row.get("prediction") == expected_prediction
        and row.get("gold")
        == ("CORRECT" if row.get("variant") == "clean" else "INCORRECT")
    )


def _score_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    if not _scores_finite({"scores": left}) or not _scores_finite({"scores": right}):
        return float("inf")
    values = []
    for candidate in ("CORRECT", "INCORRECT"):
        for field in ("mean_logprob", "total_logprob"):
            values.append(
                abs(float(left[candidate][field]) - float(right[candidate][field]))
            )
    return max(values, default=0.0)


def _activation_path(run: Path, value: Any) -> Path | None:
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [run / path, run / path.name, run / "activations" / path.name]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _activation_contract_valid(
    run: Path,
    row: Mapping[str, Any],
    *,
    expected_sites: set[str],
) -> tuple[bool, str | None]:
    path = _activation_path(run, row.get("activation_file"))
    arrays = row.get("arrays")
    if path is None:
        return False, "activation file is missing"
    if not isinstance(arrays, Mapping) or set(arrays) != expected_sites:
        return False, "manifest site keys do not match"
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError:
        return False, "safetensors is required to review recapture artifacts"
    try:
        with safe_open(path, framework="numpy") as handle:
            if set(handle.keys()) != expected_sites:
                return False, "safetensors keys do not match"
            for key in expected_sites:
                recorded_shape = [int(value) for value in arrays[key]]
                if list(handle.get_tensor(key).shape) != recorded_shape:
                    return False, f"safetensors shape mismatch: {key}"
    except (OSError, RuntimeError, SafetensorError, TypeError, ValueError) as exc:
        return False, f"invalid safetensors: {exc}"
    return True, None


def _prompt_inputs_contract(
    manifest: Mapping[str, Any],
    *,
    pair_ids: list[str],
) -> tuple[bool, dict[tuple[str, str], Mapping[str, Any]]]:
    rows = manifest.get("prompt_inputs")
    if not isinstance(rows, list):
        return False, {}
    mapping: dict[tuple[str, str], Mapping[str, Any]] = {}
    valid = (
        [str(row.get("pair_id")) for row in rows if isinstance(row, Mapping)]
        == pair_ids
        and manifest.get("prompt_inputs_hash") == stable_hash(rows, length=64)
    )
    for row in rows:
        if not isinstance(row, Mapping):
            valid = False
            continue
        pair_id = str(row.get("pair_id"))
        for receiver in ("clean", "corrupt"):
            provenance = row.get(receiver)
            provenance_valid = (
                isinstance(provenance, Mapping)
                and provenance.get("input_format") == "chat"
                and bool(provenance.get("raw_prompt_hash"))
                and bool(provenance.get("rendered_prompt_hash"))
                and bool(provenance.get("input_ids_hash"))
            )
            valid = valid and provenance_valid
            if isinstance(provenance, Mapping):
                mapping[(pair_id, receiver)] = provenance
    return valid and len(mapping) == 2 * len(pair_ids), mapping


def _manipulation_valid(
    manipulation: Any,
    *,
    expected_sites: set[str],
    router_sites: set[str],
    simultaneous: bool,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(manipulation, Mapping):
        return False, {"reason": "missing manipulation mapping"}
    candidates = manipulation.get("candidate_diagnostics")
    candidate_mapping = candidates if isinstance(candidates, Mapping) else {}
    aggregate_hooks = manipulation.get("hook_fire_counts")
    aggregate_hook_mapping = aggregate_hooks if isinstance(aggregate_hooks, Mapping) else {}
    summary_ok = (
        manipulation.get("schema_version") == 1
        and manipulation.get("candidate_count") == 2
        and manipulation.get("candidate_diagnostics_present") is True
        and set(candidate_mapping) == {"CORRECT", "INCORRECT"}
        and manipulation.get("registered_site_count") == len(expected_sites)
        and manipulation.get("all_registered_hooks_fired") is True
        and set(aggregate_hook_mapping) == expected_sites
        and all(int(value) > 0 for value in aggregate_hook_mapping.values())
    )
    candidate_ok = True
    candidate_site_metrics: list[Mapping[str, Any]] = []
    for payload in candidate_mapping.values():
        if not isinstance(payload, Mapping):
            candidate_ok = False
            continue
        hooks = payload.get("hook_fire_counts")
        hook_mapping = hooks if isinstance(hooks, Mapping) else {}
        dispatch = payload.get("dispatch_by_site")
        dispatch_mapping = dispatch if isinstance(dispatch, Mapping) else {}
        manipulation_by_site = payload.get("manipulation_by_site")
        manipulation_mapping = (
            manipulation_by_site if isinstance(manipulation_by_site, Mapping) else {}
        )
        candidate_ok = candidate_ok and (
            payload.get("simultaneous") is simultaneous
            and payload.get("registered_site_count") == len(expected_sites)
            and set(payload.get("registered_sites", [])) == expected_sites
            and set(hook_mapping) == expected_sites
            and all(int(value) > 0 for value in hook_mapping.values())
        )
        if router_sites:
            metric_values_ok = set(manipulation_mapping) == router_sites
            for metric in manipulation_mapping.values():
                if not isinstance(metric, Mapping):
                    metric_values_ok = False
                    continue
                try:
                    changed = float(metric["topk_changed_rate"])
                    overlap = float(metric["topk_overlap"])
                    kl_value = float(metric["router_kl_receiver_to_patched"])
                except (KeyError, TypeError, ValueError):
                    metric_values_ok = False
                    continue
                metric_values_ok = metric_values_ok and (
                    math.isfinite(changed)
                    and 0.0 <= changed <= 1.0
                    and math.isfinite(overlap)
                    and 0.0 <= overlap <= 1.0
                    and math.isfinite(kl_value)
                )
                candidate_site_metrics.append(metric)
            candidate_ok = candidate_ok and (
                set(dispatch_mapping) == router_sites
                and metric_values_ok
                and all(
                    isinstance(value, Mapping)
                    and value.get("downstream_dispatch_verified") is True
                    for value in dispatch_mapping.values()
                )
                and payload.get("all_router_dispatch_verified") is True
            )
        else:
            candidate_ok = candidate_ok and not dispatch_mapping and not manipulation_mapping
    if router_sites:
        recomputed_summary: dict[str, float] = {}
        for field in (
            "topk_changed_rate",
            "topk_overlap",
            "router_kl_receiver_to_patched",
        ):
            values = [float(metric[field]) for metric in candidate_site_metrics]
            recomputed_summary[field] = (
                float("nan") if not values else float(sum(values) / len(values))
            )
        summaries_match = all(
            manipulation.get(field) is not None
            and math.isfinite(recomputed_summary[field])
            and math.isfinite(float(manipulation[field]))
            and abs(float(manipulation[field]) - recomputed_summary[field]) <= 1e-12
            for field in recomputed_summary
        )
        summary_ok = summary_ok and (
            manipulation.get("router_dispatch_diagnostics_expected") is True
            and manipulation.get("router_dispatch_diagnostics_present") is True
            and manipulation.get("router_dispatch_exact_site_coverage") is True
            and manipulation.get("all_router_dispatch_verified") is True
            and summaries_match
        )
    else:
        summary_ok = summary_ok and (
            manipulation.get("router_dispatch_diagnostics_expected") is False
            and manipulation.get("router_dispatch_diagnostics_present") is False
            and manipulation.get("router_dispatch_exact_site_coverage") is None
            and manipulation.get("all_router_dispatch_verified") is None
        )
    return bool(summary_ok and candidate_ok), {
        "summary_ok": bool(summary_ok),
        "candidate_ok": bool(candidate_ok),
        "expected_sites": sorted(expected_sites),
        "router_sites": sorted(router_sites),
    }


def common_core_smoke_review(
    *,
    contract: Mapping[str, Any],
    runtime_report: Path,
    behavior_run: Path,
    recapture_run: Path,
    router_run: Path,
    endpoint_run: Path,
) -> dict[str, Any]:
    """Review four common-core stages plus runtime preflight without model weights."""

    checks: list[dict[str, Any]] = []
    run_role = str(contract.get("run_role", "smoke"))
    campaign = contract.get("campaign_policy") == "glm_common_core_20260907_v1"
    execution_source_sha256 = contract.get("execution_source_sha256")
    expected_pairs = int(contract["expected_pairs"])
    expected_behavior_pairs = int(contract["expected_behavior_pairs"])
    prerequisite = contract.get("prerequisite_smoke_review")

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    review_contract = {
        "run_role": run_role,
        "campaign_policy": contract.get("campaign_policy"),
        "execution_source_sha256": execution_source_sha256,
        "model_profile": contract["model_profile"],
        "adapter_alias": contract["adapter_alias"],
        "model_id": contract["model_id"],
        "revision": contract["revision"],
        "seed": int(contract["seed"]),
        "expected_pairs": expected_pairs,
        "expected_behavior_pairs": expected_behavior_pairs,
        "split_partition": contract["split_partition"],
        "identity_tolerance": float(contract["identity_tolerance"]),
        "router_dispatch_site_kinds": contract["artifact_contract"][
            "require_router_dispatch_for_site_kinds"
        ],
        "input_provenance_required": True,
        "validation_accessed": False,
    }
    contract_hashes = {
        "gate_config": contract["gate_config_hash"],
        "split_manifest": contract["split_manifest_hash"],
        "stage_configs": contract["stage_config_hashes"],
        "prerequisite_smoke_review": (
            prerequisite.get("sha256") if isinstance(prerequisite, Mapping) else None
        ),
    }

    stage_runs = {
        "behavior": behavior_run,
        "recapture": recapture_run,
        "router": router_run,
        "endpoint": endpoint_run,
    }
    required = [runtime_report]
    for name, run in stage_runs.items():
        required.extend([run / "environment.json", run / "selection_manifest.json"])
        if name == "behavior":
            required.extend(
                [run / "predictions.jsonl", run / "metrics.json", run / "data_manifest.json"]
            )
        elif name == "recapture":
            required.extend(
                [run / "data_manifest.jsonl", run / "metrics.json", run / "config.resolved.yaml"]
            )
        else:
            required.extend(
                [
                    run / "predictions.jsonl",
                    run / "metrics.json",
                    run / "data_manifest.json",
                    run / "config.resolved.yaml",
                ]
            )
    missing = [str(path) for path in required if not path.is_file()]
    check("required artifacts", not missing, missing)
    if missing:
        return {
            "schema_version": 1,
            "passed": False,
            **review_contract,
            "hashes": contract_hashes,
            "prerequisite_smoke_review": prerequisite,
            "checks": checks,
        }

    runtime = _read_json(runtime_report)
    environments = {
        name: _read_json(run / "environment.json") for name, run in stage_runs.items()
    }
    selections = {
        name: _read_json(run / "selection_manifest.json") for name, run in stage_runs.items()
    }
    behavior_rows = _read_jsonl(behavior_run / "predictions.jsonl")
    behavior_metrics = _read_json(behavior_run / "metrics.json")
    behavior_manifest = _read_json(behavior_run / "data_manifest.json")
    recapture_rows = _read_jsonl(recapture_run / "data_manifest.jsonl")
    router_rows = _read_jsonl(router_run / "predictions.jsonl")
    endpoint_rows = _read_jsonl(endpoint_run / "predictions.jsonl")
    router_manifest = _read_json(router_run / "data_manifest.json")
    endpoint_manifest = _read_json(endpoint_run / "data_manifest.json")
    resolved_configs = {
        "recapture": _read_yaml(recapture_run / "config.resolved.yaml"),
        "router": _read_yaml(router_run / "config.resolved.yaml"),
        "endpoint": _read_yaml(endpoint_run / "config.resolved.yaml"),
    }

    seed = int(contract["seed"])
    revision = str(contract["revision"])
    adapter_alias = str(contract["adapter_alias"])
    adapter_class = str(contract["adapter_class"])
    model_id = str(contract["model_id"])
    router_layers = [int(value) for value in contract["router_block_indices"]]
    endpoint_layers = [int(value) for value in contract["endpoint_layers"]]
    router_semantics_id = str(contract["router_semantics_id"])
    total_block_count = int(contract["total_block_count"])
    split_manifest_hash = str(contract["split_manifest_hash"])
    expected_pair_ids = [str(value) for value in contract["expected_pair_ids"]]
    expected_group_ids = [
        str(value) for value in contract["expected_discovery_group_ids"]
    ]
    tolerance = float(contract["identity_tolerance"])

    if run_role == "full_discovery":
        check(
            "passed n=8 smoke prerequisite",
            isinstance(prerequisite, Mapping)
            and prerequisite.get("passed") is True
            and prerequisite.get("expected_pairs") == 8
            and prerequisite.get("model_profile") == contract["model_profile"]
            and prerequisite.get("validation_accessed") is False
            and bool(prerequisite.get("sha256"))
            and (campaign or bool(prerequisite.get("commit"))),
            prerequisite,
        )

    runtime_layout = runtime.get("moe_layout", {})
    runtime_environment = runtime.get("environment", {})
    runtime_inspect = runtime.get("runtime_inspect", {})
    check(
        "runtime preflight",
        runtime.get("passed") is True
        and runtime_inspect.get("layout_consistent") is True
        and runtime_inspect.get("contract_matches", True) is True
        and runtime.get("adapter_alias") == adapter_alias
        and runtime.get("adapter_class") == adapter_class
        and runtime.get("model_id") == model_id
        and runtime.get("revision") == revision
        and runtime.get("tokenizer_revision") == revision
        and runtime.get("router_semantics_id") == router_semantics_id
        and runtime_layout.get("total_blocks") == total_block_count
        and runtime_layout.get("router_block_indices") == router_layers
        and runtime.get("validation_accessed") is False
        and runtime_environment.get("model", {}).get("revision") == revision,
        runtime,
    )
    check(
        "model-specific runtime fidelity",
        _model_specific_runtime_ok(adapter_alias, runtime_inspect),
        runtime_inspect,
    )

    selection_hashes = {
        name: selection.get("manifest_hash") for name, selection in selections.items()
    }
    selection_rows = {
        name: [
            row for row in selection.get("pairs", []) if isinstance(row, Mapping)
        ]
        for name, selection in selections.items()
    }
    reference_rows = selection_rows["router"]
    pair_ids = [str(row.get("pair_id")) for row in reference_rows]
    group_ids = [str(row.get("group_id")) for row in reference_rows]
    expected_stage_pairs = {
        "behavior": expected_behavior_pairs,
        "recapture": expected_pairs,
        "router": expected_pairs,
        "endpoint": expected_pairs,
    }
    selection_identity_ok = all(
        len(selection_rows[name]) == expected_count
        and all(
            str(row.get("pair_id")) == expected_pair_ids[index]
            and str(row.get("group_id")) == expected_group_ids[index]
            and bool(row.get("clean_sample_id"))
            and bool(row.get("corrupt_sample_id"))
            and bool(row.get("clean_content_hash"))
            and bool(row.get("corrupt_content_hash"))
            for index, row in enumerate(selection_rows[name])
        )
        for name, expected_count in expected_stage_pairs.items()
    )
    frozen_split_ok = all(
        selection.get("frozen_split", {}).get("partition") == "discovery"
        and selection.get("frozen_split", {}).get("manifest_hash")
        == split_manifest_hash
        and _path_matches_contract(
            selection.get("frozen_split", {}).get("path"),
            str(contract["split_manifest"]),
        )
        and selection.get("requested_pair_limit") == expected_stage_pairs[name]
        and selection.get("selected_pair_count") == expected_stage_pairs[name]
        and selection.get("selected_example_count") == 2 * expected_stage_pairs[name]
        for name, selection in selections.items()
    )
    causal_stages = ("recapture", "router", "endpoint")
    causal_rows_match = all(
        selection_rows[name] == reference_rows for name in causal_stages
    )
    causal_hashes_match = len(
        {selection_hashes[name] for name in causal_stages}
    ) == 1
    behavior_is_prefix = (
        selection_rows["behavior"] == reference_rows[:expected_behavior_pairs]
    )
    smoke_hash_matches = (
        expected_behavior_pairs != expected_pairs
        or selection_hashes["behavior"] == selection_hashes["router"]
    )
    check(
        "same valid discovery selection",
        all(_selection_valid(selection) for selection in selections.values())
        and causal_hashes_match
        and causal_rows_match
        and behavior_is_prefix
        and smoke_hash_matches
        and pair_ids == expected_pair_ids
        and group_ids == expected_group_ids
        and selection_identity_ok
        and frozen_split_ok,
        {
            "hashes": selection_hashes,
            "behavior_pairs": len(selection_rows["behavior"]),
            "causal_pairs": len(reference_rows),
            "pairs": pair_ids,
            "groups": group_ids,
            "split_manifest_hash": split_manifest_hash,
        },
    )

    git_states = [environment.get("git") for environment in environments.values()]
    environment_failures = []
    for stage_name, environment in environments.items():
        model = environment.get("model", {})
        if not (
            environment.get("seed") == seed
            and environment.get("split_partition") == "discovery"
            and environment.get("validation_accessed") is False
            and model.get("alias") == adapter_alias
            and model.get("model_id") == model_id
            and model.get("revision") == revision
            and model.get("tokenizer_revision") == revision
        ):
            environment_failures.append(stage_name)
    check(
        "same clean model environment and no validation",
        not environment_failures
        and all(state == git_states[0] for state in git_states)
        and isinstance(git_states[0], Mapping)
        and (campaign or git_states[0].get("dirty") is False)
        and (not campaign or (
            isinstance(execution_source_sha256, str)
            and len(execution_source_sha256) == 64
            and all(environment.get("execution_source_sha256") == execution_source_sha256
                    for environment in environments.values())
            and runtime_environment.get("execution_source_sha256") == execution_source_sha256
            and runtime_environment.get("cuda", {}).get("cuda_available") is True
        ))
        and runtime_environment.get("git") == git_states[0],
        {"environment_failures": environment_failures, "git": git_states},
    )

    resolved_failures = []
    canonical_stage_configs = contract["stage_configs"]
    for stage, resolved in resolved_configs.items():
        expected_config = canonical_stage_configs[stage]
        mismatched_fields = [
            key
            for key, value in expected_config.items()
            if key not in {"output_dir", "split_manifest"}
            and resolved.get(key) != value
        ]
        split_matches = _path_matches_contract(
            resolved.get("split_manifest"), str(contract["split_manifest"])
        )
        if mismatched_fields or not split_matches:
            resolved_failures.append(
                {
                    "stage": stage,
                    "mismatched_fields": mismatched_fields,
                    "split_matches": split_matches,
                }
            )
    check(
        "resolved stage seeds and discovery-only manifests",
        not resolved_failures
        and behavior_manifest.get("validation_accessed") is False
        and router_manifest.get("validation_accessed") is False
        and endpoint_manifest.get("validation_accessed") is False,
        {
            "resolved_failures": resolved_failures,
            "manifest_validation": {
                "behavior": behavior_manifest.get("validation_accessed"),
                "router": router_manifest.get("validation_accessed"),
                "endpoint": endpoint_manifest.get("validation_accessed"),
            },
        },
    )

    behavior_counts = Counter(
        (str(row.get("pair_id")), str(row.get("variant"))) for row in behavior_rows
    )
    behavior_pair_ids = pair_ids[:expected_behavior_pairs]
    expected_behavior = Counter(
        (pair_id, variant)
        for pair_id in behavior_pair_ids
        for variant in ("clean", "corrupt")
    )
    try:
        recomputed_behavior = binary_verification_metrics(behavior_rows)
    except (KeyError, TypeError, ValueError):
        recomputed_behavior = {}
    recorded_auroc = behavior_metrics.get("auroc")
    recomputed_auroc = recomputed_behavior.get("auroc")
    behavior_metrics_match = all(
        behavior_metrics.get(key) == recomputed_behavior.get(key)
        for key in ("n", "positives", "negatives")
    ) and all(
        behavior_metrics.get(key) is not None
        and recomputed_behavior.get(key) is not None
        and math.isfinite(float(behavior_metrics[key]))
        and math.isfinite(float(recomputed_behavior[key]))
        and abs(float(behavior_metrics[key]) - float(recomputed_behavior[key])) <= 1e-12
        for key in ("accuracy", "auroc", "mean_margin_incorrect_minus_correct")
    )
    behavior_identity_ok = all(
        str(row.get("group_id")) == expected_group_ids[expected_pair_ids.index(pair_id)]
        and str(row.get("sample_id"))
        == str(
            reference_rows[expected_pair_ids.index(pair_id)].get(
                f"{row.get('variant')}_sample_id"
            )
        )
        for row in behavior_rows
        if (pair_id := str(row.get("pair_id"))) in behavior_pair_ids
    )
    check(
        "behavior exact coverage and recomputed capability",
        len(behavior_rows) == 2 * expected_behavior_pairs
        and behavior_counts == expected_behavior
        and behavior_identity_ok
        and all(_behavior_row_valid(row) for row in behavior_rows)
        and behavior_metrics.get("n") == len(behavior_rows)
        and recomputed_behavior.get("n") == len(behavior_rows)
        and recomputed_behavior.get("positives") == expected_behavior_pairs
        and recomputed_behavior.get("negatives") == expected_behavior_pairs
        and recorded_auroc is not None
        and recomputed_auroc is not None
        and math.isfinite(float(recorded_auroc))
        and math.isfinite(float(recomputed_auroc))
        and behavior_metrics_match
        and (not contract.get("behavior_is_gate", True)
             or float(recomputed_auroc) >= float(contract["behavior_min_auroc"])),
        {
            "rows": len(behavior_rows),
            "expected_rows": 2 * expected_behavior_pairs,
            "recorded_auroc": recorded_auroc,
            "recomputed_auroc": recomputed_auroc,
            "minimum": contract["behavior_min_auroc"],
            "capability_threshold_is_gate": contract.get("behavior_is_gate", True),
        },
    )

    recapture_site_keys = {
        f"layer_{layer}.{kind}"
        for layer in router_layers
        for kind in contract["recapture_site_kinds"]
    }
    recapture_contract = environments["recapture"].get("capture_site_contract", {})
    recapture_contract_sites = {
        str(row.get("site_key")) for row in recapture_contract.get("sites", [])
    }
    selection_by_pair = {str(row.get("pair_id")): row for row in reference_rows}
    expected_recapture = Counter(
        (pair_id, variant) for pair_id in pair_ids for variant in ("clean", "corrupt")
    )
    observed_recapture = Counter(
        (str(row.get("pair_id")), str(row.get("variant")))
        for row in recapture_rows
    )
    recapture_failures = []
    recapture_activation_paths: list[Path] = []
    recapture_anchors: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in recapture_rows:
        pair_id = str(row.get("pair_id"))
        variant = str(row.get("variant"))
        selection_row = selection_by_pair.get(pair_id, {})
        activation_ok, activation_error = _activation_contract_valid(
            recapture_run,
            row,
            expected_sites=recapture_site_keys,
        )
        activation_path = _activation_path(recapture_run, row.get("activation_file"))
        expected_activation_name = (
            stable_hash(
                f"{row.get('sample_id')}:step_binary:{row.get('boundary_step')}"
            )
            + ".safetensors"
        )
        if activation_path is not None:
            recapture_activation_paths.append(activation_path.resolve())
        recapture_anchors[(pair_id, variant)] = row
        if (
            not row.get("input_ids_hash")
            or not row.get("prompt_hash")
            or row.get("prompt_mode") != "step_binary"
            or row.get("input_format") != "chat"
            or row.get("boundary_strategy") != "paired_first_error"
            or row.get("group_id") != selection_row.get("group_id")
            or row.get("sample_id") != selection_row.get(f"{variant}_sample_id")
            or row.get("content_hash") != selection_row.get(f"{variant}_content_hash")
            or activation_path is None
            or activation_path.name != expected_activation_name
            or not activation_ok
        ):
            recapture_failures.append(
                {"sample_id": str(row.get("sample_id")), "activation": activation_error}
            )
    check(
        "recapture exact site and activation contract",
        len(recapture_rows) == 2 * expected_pairs
        and observed_recapture == expected_recapture
        and len(set(recapture_activation_paths)) == 2 * expected_pairs
        and not recapture_failures
        and recapture_contract.get("router_semantics_id") == router_semantics_id
        and recapture_contract.get("moe_layout", {}).get("total_blocks")
        == total_block_count
        and recapture_contract.get("moe_layout", {}).get("router_block_indices")
        == router_layers
        and recapture_contract_sites == recapture_site_keys,
        {
            "rows": len(recapture_rows),
            "expected_rows": 2 * expected_pairs,
            "unique_activation_files": len(set(recapture_activation_paths)),
            "failures": recapture_failures[:5],
        },
    )

    router_site_keys = {f"layer_{layer}.router_logits" for layer in router_layers}
    endpoint_site_keys = {
        f"layer_{layer}.{kind}"
        for layer in endpoint_layers
        for kind in contract["endpoint_site_kinds"]
    }
    stage_site_contract_failures = []
    for stage_name, expected_sites in (
        ("router", router_site_keys),
        ("endpoint", endpoint_site_keys),
    ):
        site_contract = environments[stage_name].get("capture_site_contract", {})
        observed_sites = {
            str(row.get("site_key")) for row in site_contract.get("sites", [])
        }
        if not (
            site_contract.get("router_semantics_id") == router_semantics_id
            and site_contract.get("moe_layout", {}).get("total_blocks")
            == total_block_count
            and site_contract.get("moe_layout", {}).get("router_block_indices")
            == router_layers
            and observed_sites == expected_sites
        ):
            stage_site_contract_failures.append(stage_name)
    check(
        "patch stage site topology and semantics",
        not stage_site_contract_failures,
        stage_site_contract_failures,
    )
    router_prompt_inputs_ok, router_prompt_inputs = _prompt_inputs_contract(
        router_manifest,
        pair_ids=pair_ids,
    )
    endpoint_prompt_inputs_ok, endpoint_prompt_inputs = _prompt_inputs_contract(
        endpoint_manifest,
        pair_ids=pair_ids,
    )

    def cross_stage_boundary_failures(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        failures = []
        for row in rows:
            pair_id = str(row.get("pair_id"))
            receiver = str(row.get("receiver_variant"))
            anchor = recapture_anchors.get((pair_id, receiver))
            provenance = row.get("input_provenance")
            pre_error = str(row.get("treatment", "")).startswith("pre_error_")
            try:
                same_boundary = (
                    anchor is not None
                    and int(str(row.get("boundary_step")))
                    == int(str(anchor.get("boundary_step")))
                )
                token_ok = anchor is not None and (
                    int(str(row.get("token_position")))
                    < int(str(anchor.get("token_position")))
                    if pre_error
                    else int(str(row.get("token_position")))
                    == int(str(anchor.get("token_position")))
                )
            except (TypeError, ValueError):
                same_boundary = False
                token_ok = False
            prompt_ok = (
                anchor is not None
                and isinstance(provenance, Mapping)
                and str(provenance.get("rendered_prompt_hash", "")).startswith(
                    str(anchor.get("prompt_hash", ""))
                )
                and str(provenance.get("input_ids_hash", "")).startswith(
                    str(anchor.get("input_ids_hash", ""))
                )
            )
            if not (same_boundary and token_ok and prompt_ok):
                failures.append(
                    {
                        "pair_id": pair_id,
                        "receiver": receiver,
                        "treatment": row.get("treatment"),
                        "same_boundary": same_boundary,
                        "token_ok": token_ok,
                        "prompt_ok": prompt_ok,
                    }
                )
        return failures

    def baselines(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row["pair_id"]), str(row["receiver_variant"])): row
            for row in rows
            if row.get("treatment") == "unpatched"
        }

    def identity_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        baseline_rows = baselines(rows)
        failures = []
        for row in rows:
            is_identity = row.get("treatment") in {"unpatched_rerun", "self_patch"} or (
                row.get("treatment") in {"clean_to_corrupt", "corrupt_to_clean"}
                and float(row.get("metadata", {}).get("alpha", -1.0)) == 0.0
            )
            if not is_identity:
                continue
            baseline = baseline_rows.get(
                (str(row.get("pair_id")), str(row.get("receiver_variant")))
            )
            delta = (
                float("inf")
                if baseline is None
                else _score_delta(row.get("scores", {}), baseline.get("scores", {}))
            )
            if delta > tolerance:
                failures.append({"pair_id": row.get("pair_id"), "treatment": row.get("treatment"), "delta": delta})
        return failures

    def baseline_coverage(rows: list[dict[str, Any]]) -> bool:
        observed = Counter(
            (
                str(row.get("pair_id")),
                str(row.get("receiver_variant")),
                str(row.get("treatment")),
                row.get("donor_variant"),
            )
            for row in rows
            if row.get("treatment") in {"unpatched", "unpatched_rerun"}
        )
        expected = Counter(
            (pair_id, receiver, treatment, None)
            for pair_id in pair_ids
            for receiver in ("clean", "corrupt")
            for treatment in ("unpatched", "unpatched_rerun")
        )
        return observed == expected

    def input_provenance_failures(
        rows: list[dict[str, Any]],
        expected: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        failures = []
        for row in rows:
            provenance = row.get("input_provenance")
            key = (str(row.get("pair_id")), str(row.get("receiver_variant")))
            valid = (
                isinstance(provenance, Mapping)
                and provenance.get("input_format") == "chat"
                and bool(provenance.get("raw_prompt_hash"))
                and bool(provenance.get("rendered_prompt_hash"))
                and bool(provenance.get("input_ids_hash"))
                and provenance == expected.get(key)
            )
            if not valid:
                failures.append(
                    {"pair_id": key[0], "receiver": key[1], "treatment": row.get("treatment")}
                )
        return failures

    router_diag_failures = []
    for row in router_rows:
        if row.get("treatment") in {"unpatched", "unpatched_rerun"}:
            continue
        valid, details = _manipulation_valid(
            row.get("metadata", {}).get("manipulation"),
            expected_sites=router_site_keys,
            router_sites=router_site_keys,
            simultaneous=True,
        )
        if not valid:
            router_diag_failures.append(
                {"pair_id": row.get("pair_id"), "treatment": row.get("treatment"), **details}
            )
    router_identity_failures = identity_failures(router_rows)
    router_provenance_failures = input_provenance_failures(
        router_rows,
        router_prompt_inputs,
    )
    router_score_failures = [
        index for index, row in enumerate(router_rows) if not _scores_finite(row)
    ]
    router_boundary_failures = cross_stage_boundary_failures(router_rows)
    required_router_descriptors = Counter(
        (pair_id, treatment, receiver, donor, alpha)
        for pair_id in pair_ids
        for treatment, receiver, donor, alphas in (
            ("self_patch", "clean", "clean", (None,)),
            ("self_patch", "corrupt", "corrupt", (None,)),
            ("corrupt_to_clean", "clean", "corrupt", (0.0, 0.5, 1.0, 2.0)),
            ("clean_to_corrupt", "corrupt", "clean", (0.0, 0.5, 1.0, 2.0)),
        )
        for alpha in alphas
    )
    observed_router_descriptors = Counter(
        (
            str(row.get("pair_id")),
            str(row.get("treatment")),
            str(row.get("receiver_variant")),
            row.get("donor_variant"),
            None
            if row.get("treatment") == "self_patch"
            else float(row.get("metadata", {}).get("alpha", -1.0)),
        )
        for row in router_rows
        if row.get("treatment") in {"self_patch", "clean_to_corrupt", "corrupt_to_clean"}
    )
    router_bundle_row_count = sum(
        1
        for row in router_rows
        if row.get("treatment")
        in {"self_patch", "clean_to_corrupt", "corrupt_to_clean"}
    )
    check(
        "router rows, identity, hooks, and native dispatch",
        baseline_coverage(router_rows)
        and router_prompt_inputs_ok
        and required_router_descriptors == observed_router_descriptors
        and not router_diag_failures
        and not router_identity_failures
        and not router_provenance_failures
        and not router_score_failures
        and not router_boundary_failures,
        {
            "diagnostic_failures": router_diag_failures[:5],
            "identity_failures": router_identity_failures[:5],
            "provenance_failures": router_provenance_failures[:5],
            "prompt_inputs_contract": router_prompt_inputs_ok,
            "nonfinite_score_rows": router_score_failures[:5],
            "cross_stage_boundary_failures": router_boundary_failures[:5],
        },
    )
    forced_treatments = {
        "forced_reroute_clean_receiver": "clean",
        "forced_reroute_corrupt_receiver": "corrupt",
    }
    forced_rows = [
        row
        for row in router_rows
        if row.get("treatment") in forced_treatments
    ]
    forced_bundle_label = f"bundle.router_logits[{len(router_site_keys)}]"
    check(
        "forced router controls change actual top-k",
        all(
            row.get("metadata", {}).get("control") == "forced_reroute_smoke_only"
            and row.get("receiver_variant") == forced_treatments[row["treatment"]]
            and row.get("donor_variant") == "forced"
            and row.get("site") == forced_bundle_label
            and float(
                row.get("metadata", {})
                .get("manipulation", {})
                .get("topk_changed_rate")
                or 0.0
            )
            > 0.0
            for row in forced_rows
        ),
        {"rows": len(forced_rows)},
    )
    forced_conditional_failures = []
    for pair_id in (pair_ids if contract.get("forced_reroute_control", run_role == "smoke") else []):
        for direction, forced_treatment in (
            ("clean_to_corrupt", "forced_reroute_corrupt_receiver"),
            ("corrupt_to_clean", "forced_reroute_clean_receiver"),
        ):
            dose_rows = [
                row
                for row in router_rows
                if row.get("pair_id") == pair_id and row.get("treatment") == direction
            ]
            max_change = max(
                (
                    float(
                        row.get("metadata", {})
                        .get("manipulation", {})
                        .get("topk_changed_rate")
                        or 0.0
                    )
                    for row in dose_rows
                ),
                default=0.0,
            )
            observed_forced = [
                row
                for row in router_rows
                if row.get("pair_id") == pair_id
                and row.get("treatment") == forced_treatment
            ]
            needs_forced = max_change <= 0.01
            if (needs_forced and len(observed_forced) != 1) or (
                not needs_forced and observed_forced
            ):
                forced_conditional_failures.append(
                    {
                        "pair_id": pair_id,
                        "direction": direction,
                        "max_change": max_change,
                        "forced_rows": len(observed_forced),
                    }
                )
    check(
        "forced router controls are conditional",
        not forced_conditional_failures,
        forced_conditional_failures[:5],
    )
    if run_role == "full_discovery":
        forced_marker_rows = [
            row
            for row in router_rows
            if row.get("metadata", {}).get("control") == "forced_reroute_smoke_only"
        ]
        check(
            "full router exact rows and no smoke-only forced controls",
            len(router_rows) == 14 * expected_pairs
            and router_bundle_row_count == 10 * expected_pairs
            and not forced_rows
            and not forced_marker_rows,
            {
                "rows": len(router_rows),
                "expected_rows": 14 * expected_pairs,
                "bundle_descriptor_rows": router_bundle_row_count,
                "expected_bundle_descriptor_rows": 10 * expected_pairs,
                "forced_rows": len(forced_rows),
                "forced_marker_rows": len(forced_marker_rows),
            },
        )

    export = router_manifest.get("router_weight_export")
    export_error = None
    matrices: dict[int, Any] = {}
    loaded_export: dict[str, Any] = {}
    if isinstance(export, Mapping) and export.get("matrix_file"):
        try:
            matrices, loaded_export = load_router_weight_export(
                router_run / str(export["matrix_file"])
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            export_error = str(exc)
    check(
        "router weight schema-v2 topology and hash",
        isinstance(export, Mapping)
        and loaded_export == export
        and export.get("schema_version") == 2
        and export.get("adapter_alias") == adapter_alias
        and export.get("model_id") == model_id
        and export.get("model_revision") == revision
        and export.get("tokenizer_revision") == revision
        and export.get("total_blocks") == total_block_count
        and export.get("num_router_blocks") == len(router_layers)
        and export.get("router_block_indices") == router_layers
        and export.get("router_semantics_id") == router_semantics_id
        and set(matrices) == set(router_layers),
        {"manifest": export, "error": export_error},
    )

    endpoint_diag_failures = []
    endpoint_required = Counter(
        (pair_id, site, treatment, receiver, donor)
        for pair_id in pair_ids
        for site in endpoint_site_keys
        for treatment, receiver, donor in (
            ("self_patch", "clean", "clean"),
            ("self_patch", "corrupt", "corrupt"),
            ("corrupt_to_clean", "clean", "corrupt"),
            ("clean_to_corrupt", "corrupt", "clean"),
            ("norm_random_delta", "clean", "random"),
            ("norm_random_delta", "corrupt", "random"),
            ("random_matched_corrupt_to_clean", "clean", "matched_corrupt"),
            ("random_matched_clean_to_corrupt", "corrupt", "matched_clean"),
        )
    )
    known_endpoint_treatments = {
        "unpatched",
        "unpatched_rerun",
        "self_patch",
        "corrupt_to_clean",
        "clean_to_corrupt",
        "norm_random_delta",
        "random_matched_corrupt_to_clean",
        "random_matched_clean_to_corrupt",
        "pre_error_corrupt_to_clean",
        "pre_error_clean_to_corrupt",
    }
    unknown_endpoint_treatments = [
        {"row": index, "treatment": row.get("treatment")}
        for index, row in enumerate(endpoint_rows)
        if row.get("treatment") not in known_endpoint_treatments
    ]
    endpoint_observed: Counter[tuple[str, str, str, str, Any]] = Counter()
    for row in endpoint_rows:
        if row.get("treatment") in {"unpatched", "unpatched_rerun"}:
            continue
        site = str(row.get("site"))
        if row.get("treatment") in {
            "self_patch",
            "corrupt_to_clean",
            "clean_to_corrupt",
            "norm_random_delta",
            "random_matched_corrupt_to_clean",
            "random_matched_clean_to_corrupt",
        }:
            endpoint_observed[
                (
                    str(row.get("pair_id")),
                    site,
                    str(row.get("treatment")),
                    str(row.get("receiver_variant")),
                    row.get("donor_variant"),
                )
            ] += 1
        router_expected = site.endswith(".router_logits")
        valid, details = _manipulation_valid(
            row.get("metadata", {}).get("manipulation"),
            expected_sites={site},
            router_sites={site} if router_expected else set(),
            simultaneous=False,
        )
        if not valid:
            endpoint_diag_failures.append(
                {"pair_id": row.get("pair_id"), "site": site, "treatment": row.get("treatment"), **details}
            )
    endpoint_identity_failures = identity_failures(endpoint_rows)
    endpoint_provenance_failures = input_provenance_failures(
        endpoint_rows,
        endpoint_prompt_inputs,
    )
    endpoint_score_failures = [
        index for index, row in enumerate(endpoint_rows) if not _scores_finite(row)
    ]
    endpoint_boundary_failures = cross_stage_boundary_failures(endpoint_rows)
    matched_control_failures = [
        {
            "pair_id": row.get("pair_id"),
            "control_pair_id": row.get("metadata", {}).get("control_pair_id"),
        }
        for row in endpoint_rows
        if str(row.get("treatment", "")).startswith("random_matched_")
        and (
            str(row.get("metadata", {}).get("control_pair_id")) not in pair_ids
            or str(row.get("metadata", {}).get("control_pair_id"))
            == str(row.get("pair_id"))
        )
    ]
    pre_error_pair_ids = [
        pair_id
        for pair_id in pair_ids
        if contract.get("pre_error_control", True) and int(
            recapture_anchors.get((pair_id, "corrupt"), {}).get("boundary_step", 0)
        )
        > 0
    ]
    pre_error_required = Counter(
        (pair_id, site, treatment, receiver, donor)
        for pair_id in pre_error_pair_ids
        for site in endpoint_site_keys
        for treatment, receiver, donor in (
            ("pre_error_corrupt_to_clean", "clean", "corrupt"),
            ("pre_error_clean_to_corrupt", "corrupt", "clean"),
        )
    )
    pre_error_observed = Counter(
        (
            str(row.get("pair_id")),
            str(row.get("site")),
            str(row.get("treatment")),
            str(row.get("receiver_variant")),
            row.get("donor_variant"),
        )
        for row in endpoint_rows
        if str(row.get("treatment", "")).startswith("pre_error_")
    )
    endpoint_base_rows = expected_pairs * (4 + 8 * len(endpoint_site_keys))
    expected_endpoint_rows = endpoint_base_rows + (
        2 * len(endpoint_site_keys) * len(pre_error_pair_ids)
    )
    check(
        "endpoint rows, identity, hooks, controls, and router dispatch",
        (
            run_role != "full_discovery"
            or (
                len(endpoint_rows) == expected_endpoint_rows
                and not unknown_endpoint_treatments
            )
        )
        and baseline_coverage(endpoint_rows)
        and endpoint_prompt_inputs_ok
        and set(endpoint_manifest.get("sites", [])) == endpoint_site_keys
        and endpoint_observed == endpoint_required
        and pre_error_observed == pre_error_required
        and not endpoint_diag_failures
        and not endpoint_identity_failures
        and not endpoint_provenance_failures
        and not endpoint_score_failures
        and not endpoint_boundary_failures
        and not matched_control_failures,
        {
            "diagnostic_failures": endpoint_diag_failures[:5],
            "identity_failures": endpoint_identity_failures[:5],
            "provenance_failures": endpoint_provenance_failures[:5],
            "prompt_inputs_contract": endpoint_prompt_inputs_ok,
            "nonfinite_score_rows": endpoint_score_failures[:5],
            "cross_stage_boundary_failures": endpoint_boundary_failures[:5],
            "matched_control_failures": matched_control_failures[:5],
            "rows": len(endpoint_rows),
            "expected_rows": expected_endpoint_rows,
            "base_expected_rows": endpoint_base_rows,
            "pre_error_pairs": len(pre_error_pair_ids),
            "unknown_treatments": unknown_endpoint_treatments[:5],
            "site_count": len(endpoint_manifest.get("sites", [])),
        },
    )

    prediction_hashes = {
        "behavior": stable_hash(behavior_rows, length=64),
        "router": stable_hash(router_rows, length=64),
        "endpoint": stable_hash(endpoint_rows, length=64),
    }
    check_results = {str(item["name"]): bool(item["passed"]) for item in checks}
    identity_dispatch_provenance = {
        "identity_tolerance": tolerance,
        "router_dispatch_site_kinds": contract["artifact_contract"][
            "require_router_dispatch_for_site_kinds"
        ],
        "input_provenance_required": True,
        "router_identity_dispatch_passed": check_results.get(
            "router rows, identity, hooks, and native dispatch", False
        ),
        "endpoint_identity_dispatch_passed": check_results.get(
            "endpoint rows, identity, hooks, controls, and router dispatch", False
        ),
    }
    hashes = {
        **contract_hashes,
        "selection_manifests": selection_hashes,
        "predictions": prediction_hashes,
    }
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        **review_contract,
        "hashes": hashes,
        "prediction_hashes": prediction_hashes,
        "identity_dispatch_provenance": identity_dispatch_provenance,
        "provenance": {
            "git": git_states[0],
            "model_id": model_id,
            "revision": revision,
            "tokenizer_revision": revision,
            "adapter_alias": adapter_alias,
            "adapter_class": adapter_class,
            "router_semantics_id": router_semantics_id,
            "prerequisite_smoke_review": prerequisite,
        },
        "prerequisite_smoke_review": prerequisite,
        "checks": checks,
    }
