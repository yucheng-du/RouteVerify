"""Artifact-only reviewer for the independent GPT-OSS preflight package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from routeverify.data.schema import stable_hash
from routeverify.instrumentation.storage import atomic_write_json


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    deltas: list[float] = []
    for candidate in ("CORRECT", "INCORRECT"):
        left_score = left["scores"][candidate]
        right_score = right["scores"][candidate]
        for field in ("mean_logprob", "total_logprob"):
            deltas.append(abs(float(left_score[field]) - float(right_score[field])))
    return max(deltas, default=0.0)


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_harmony_render(
    render: dict[str, Any],
    *,
    text: Any,
    expected_hash: Any,
    expected_template_sha256: str,
    expected_reasoning_effort: str,
) -> tuple[bool, str]:
    if not isinstance(text, str) or not text:
        return False, "missing rendered_prompt_text"
    actual_hash = stable_hash({"prompt": text}, length=64)
    if actual_hash != expected_hash:
        return False, f"rendered text/hash mismatch: {actual_hash} != {expected_hash}"
    if render.get("rendered_prompt_hash") != expected_hash:
        return False, "adapter and boundary rendered hashes differ"
    if render.get("rendered_prompt_sha256") != hashlib.sha256(text.encode()).hexdigest():
        return False, "rendered prompt SHA256 mismatch"
    if render.get("render_family") != "openai_harmony":
        return False, "render family is not openai_harmony"
    if render.get("chat_template_sha256") != expected_template_sha256:
        return False, "Harmony template hash mismatch"
    if render.get("reasoning_effort") != expected_reasoning_effort:
        return False, "Harmony reasoning effort mismatch"
    if render.get("candidate_prefix_length") != len(text):
        return False, "Harmony candidate prefix length mismatch"
    rendered_date = render.get("rendered_date")
    if not isinstance(rendered_date, str) or _DATE_PATTERN.fullmatch(rendered_date) is None:
        return False, "Harmony dynamic date was not recorded"
    suffix = render.get("candidate_suffix")
    if suffix != "<|return|>":
        return False, f"unexpected Harmony candidate suffix: {suffix!r}"
    contracts = render.get("candidate_contracts")
    if not isinstance(contracts, dict) or set(contracts) != {"CORRECT", "INCORRECT"}:
        return False, "Harmony candidate contracts are incomplete"
    for candidate, contract in contracts.items():
        if not isinstance(contract, dict):
            return False, f"invalid candidate contract: {candidate}"
        expected_span = [len(text), len(text) + len(candidate)]
        if contract.get("target_char_span") != expected_span:
            return False, f"candidate span mismatch: {candidate}"
        full_render = f"{text}{candidate}{suffix}"
        if contract.get("full_render_sha256") != hashlib.sha256(full_render.encode()).hexdigest():
            return False, f"candidate render SHA256 mismatch: {candidate}"
    return True, actual_hash


def _check_render_record(
    record: dict[str, Any],
    *,
    expected_template_sha256: str,
    expected_reasoning_effort: str,
) -> tuple[bool, str]:
    adapter_render = record.get("adapter_render")
    if not isinstance(adapter_render, dict):
        return False, "missing adapter_render contract"
    return _check_harmony_render(
        adapter_render,
        text=record.get("rendered_prompt_text"),
        expected_hash=record.get("rendered_prompt_hash"),
        expected_template_sha256=expected_template_sha256,
        expected_reasoning_effort=expected_reasoning_effort,
    )


def gpt_oss_preflight_review(*, run_root: Path, protocol_path: Path) -> dict[str, Any]:
    """Review behavior → runtime fidelity → n=8 patch artifacts without weights."""

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("GPT-OSS protocol must be a mapping")
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    required = [run_root / str(relative) for relative in protocol["package"]["required_files"]]
    missing = [str(path.relative_to(run_root)) for path in required if not path.is_file()]
    record("required_files", not missing, {"missing": missing})
    if missing:
        return {
            "schema_version": 1,
            "protocol_id": protocol.get("protocol_id"),
            "passed": False,
            "decision": "NO_GO",
            "checks": checks,
        }

    model_contract = protocol["model"]
    data_contract = protocol["data"]
    behavior_contract = protocol["behavior"]
    hook_contract = protocol["runtime_hook_fidelity"]
    patch_contract = protocol["patch_smoke"]

    provenance = _read_json(run_root / "provenance_manifest.json")
    run_manifest = _read_json(run_root / "run_manifest.json")
    environment = _read_json(run_root / "environment_manifest.json")
    record(
        "no_validation",
        provenance.get("validation_accessed") is False
        and run_manifest.get("validation_accessed") is False
        and data_contract.get("split_partition") == "discovery",
        {
            "provenance": provenance.get("validation_accessed"),
            "run": run_manifest.get("validation_accessed"),
            "partition": data_contract.get("split_partition"),
        },
    )
    code = provenance.get("code", {})
    record(
        "clean_code_commit",
        isinstance(code.get("commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", code["commit"]) is not None
        and code.get("dirty") is False,
        code,
    )
    record(
        "execution_contract",
        provenance.get("execution_order") == protocol.get("execution_order")
        and run_manifest.get("profile") == protocol.get("protocol_id")
        and run_manifest.get("code_commit") == code.get("commit")
        and run_manifest.get("exit_code") == 0
        and isinstance(run_manifest.get("success"), bool),
        {
            "provenance_order": provenance.get("execution_order"),
            "protocol_order": protocol.get("execution_order"),
            "profile": run_manifest.get("profile"),
            "code_commit": run_manifest.get("code_commit"),
            "exit_code": run_manifest.get("exit_code"),
            "success_at_review": run_manifest.get("success"),
        },
    )
    record(
        "transformers_version",
        str(environment.get("transformers")) == str(model_contract["transformers_version"]),
        {
            "actual": environment.get("transformers"),
            "expected": model_contract["transformers_version"],
        },
    )
    gpu = environment.get("gpu", {})
    gpu_name = str(gpu.get("name", ""))
    record(
        "gpu_environment",
        environment.get("validation_accessed") is False
        and environment.get("git_porcelain") == []
        and ("A100" in gpu_name.upper() or "H100" in gpu_name.upper())
        and float(gpu.get("vram_gb", 0.0)) >= 39.0,
        {
            "validation_accessed": environment.get("validation_accessed"),
            "git_porcelain": environment.get("git_porcelain"),
            "gpu": gpu,
        },
    )

    behavior_dir = run_root / "behavior"
    behavior_metrics = _read_json(behavior_dir / "metrics.json")
    behavior_rows = _read_jsonl(behavior_dir / "predictions.jsonl")
    behavior_manifest = _read_json(behavior_dir / "data_manifest.json")
    behavior_environment = _read_json(behavior_dir / "environment.json")
    behavior_selection = _read_json(behavior_dir / "selection_manifest.json")
    behavior_model = behavior_environment.get("model", {})
    record(
        "behavior_model_revision",
        behavior_model.get("model_id") == model_contract["model_id"]
        and behavior_model.get("revision") == model_contract["revision"]
        and behavior_model.get("tokenizer_revision") == model_contract["tokenizer_revision"],
        behavior_model,
    )
    variants = {str(row.get("variant")) for row in behavior_rows}
    behavior_pairs = {
        str(row["pair_id"]) for row in behavior_rows if row.get("pair_id") is not None
    }
    behavior_pair_variants = Counter(
        (str(row.get("pair_id")), str(row.get("variant"))) for row in behavior_rows
    )
    expected_behavior_keys = {
        (pair_id, variant)
        for pair_id in behavior_pairs
        for variant in behavior_contract["required_variants"]
    }
    manifest_behavior_pair_list = [
        str(pair_id) for pair_id in behavior_manifest.get("pair_ids", [])
    ]
    manifest_behavior_pairs = set(manifest_behavior_pair_list)
    record(
        "behavior_rows_and_pairs",
        len(behavior_rows) == int(behavior_contract["expected_rows"])
        and len(behavior_pairs) == int(data_contract["expected_pairs"])
        and variants == set(behavior_contract["required_variants"])
        and set(behavior_pair_variants) == expected_behavior_keys
        and all(count == 1 for count in behavior_pair_variants.values())
        and manifest_behavior_pairs == behavior_pairs
        and len(manifest_behavior_pair_list) == len(manifest_behavior_pairs)
        and int(behavior_manifest.get("row_count", -1)) == len(behavior_rows),
        {
            "rows": len(behavior_rows),
            "pairs": len(behavior_pairs),
            "variants": sorted(variants),
            "duplicate_pair_variants": sorted(
                f"{pair_id}:{variant}"
                for (pair_id, variant), count in behavior_pair_variants.items()
                if count != 1
            ),
            "manifest_pairs_match": manifest_behavior_pairs == behavior_pairs,
            "manifest_row_count": behavior_manifest.get("row_count"),
        },
    )
    record(
        "behavior_no_validation",
        behavior_manifest.get("validation_accessed") is False
        and behavior_environment.get("validation_accessed") is False
        and behavior_environment.get("split_partition") == "discovery",
        {
            "manifest": behavior_manifest.get("validation_accessed"),
            "environment": behavior_environment.get("validation_accessed"),
            "partition": behavior_environment.get("split_partition"),
        },
    )
    auroc = behavior_metrics.get("auroc")
    accuracy = behavior_metrics.get("accuracy")
    record(
        "behavior_gate",
        auroc is not None
        and accuracy is not None
        and float(auroc) >= float(behavior_contract["min_auroc"])
        and float(accuracy) >= float(behavior_contract["min_accuracy"]),
        {"auroc": auroc, "accuracy": accuracy},
    )
    behavior_render_failures: list[str] = []
    behavior_render_hashes: dict[tuple[str, str], str] = {}
    for row in behavior_rows:
        render = row.get("input_provenance")
        if not isinstance(render, dict):
            behavior_render_failures.append(str(row.get("sample_id")))
            continue
        ok, detail = _check_harmony_render(
            render,
            text=render.get("rendered_prompt_text"),
            expected_hash=render.get("rendered_prompt_hash"),
            expected_template_sha256=str(model_contract["chat_template_sha256"]),
            expected_reasoning_effort=str(model_contract["harmony_reasoning_effort"]),
        )
        if not ok:
            behavior_render_failures.append(f"{row.get('sample_id')}:{detail}")
        else:
            behavior_render_hashes[(str(row.get("pair_id")), str(row.get("variant")))] = detail
    record(
        "behavior_harmony_renders",
        not behavior_render_failures,
        {"failed_samples": behavior_render_failures},
    )

    runtime_dir = run_root / "runtime"
    runtime_inspect = _read_json(runtime_dir / "runtime_inspect.json")
    hook = _read_json(runtime_dir / "hook_fidelity.json")
    runtime_manifest = _read_json(runtime_dir / "data_manifest.json")
    runtime_environment = _read_json(runtime_dir / "environment.json")
    runtime_selection = _read_json(runtime_dir / "selection_manifest.json")
    record(
        "runtime_no_validation",
        hook.get("validation_accessed") is False
        and runtime_manifest.get("validation_accessed") is False
        and runtime_environment.get("validation_accessed") is False,
        {
            "hook": hook.get("validation_accessed"),
            "manifest": runtime_manifest.get("validation_accessed"),
            "environment": runtime_environment.get("validation_accessed"),
        },
    )
    actual = runtime_inspect.get("actual_contract", {})
    expected = model_contract["static_contract"]
    record(
        "runtime_static_contract",
        runtime_inspect.get("contract_matches") is True
        and runtime_inspect.get("layout_consistent") is True
        and actual == expected
        and runtime_inspect.get("config_contract") == expected
        and runtime_inspect.get("model_type") == model_contract["model_type"]
        and set(runtime_inspect.get("architectures", [])) == set(model_contract["architectures"]),
        {
            "actual": actual,
            "config": runtime_inspect.get("config_contract"),
            "expected": expected,
            "layout_consistent": runtime_inspect.get("layout_consistent"),
            "model_type": runtime_inspect.get("model_type"),
            "architectures": runtime_inspect.get("architectures"),
        },
    )
    modes = set(runtime_inspect.get("router_execution_modes", []))
    record(
        "runtime_router_mode",
        modes == {str(model_contract["required_router_execution_mode"])}
        and runtime_inspect.get("native_mxfp4") is True,
        sorted(modes),
    )
    package_versions = runtime_inspect.get("runtime_package_versions", {})
    kernel_load = runtime_inspect.get("mxfp4_kernel_load", {})
    kernel_preflight = environment.get("mxfp4_kernel_preflight", {})
    locked_versions = environment.get("locked_runtime_versions", {})
    packaged_lock = tomllib.loads((run_root / "configs/uv.lock").read_text(encoding="utf-8"))
    packaged_lock_versions = {
        str(package["name"]): str(package["version"])
        for package in packaged_lock.get("package", [])
        if isinstance(package, dict) and "name" in package and "version" in package
    }
    packaged_pyproject = tomllib.loads(
        (run_root / "configs/pyproject.toml").read_text(encoding="utf-8")
    )
    packaged_extras = packaged_pyproject.get("project", {}).get("optional-dependencies", {})
    required_specs = {
        f"kernels=={model_contract['kernels_version']}",
        f"transformers=={model_contract['transformers_version']}",
    }
    packaged_specs_match = all(
        required_specs.issubset(set(packaged_extras.get(extra, []))) for extra in ("models", "all")
    )
    record(
        "native_mxfp4_kernel_contract",
        package_versions.get("transformers") == model_contract["transformers_version"]
        and package_versions.get("kernels") == model_contract["kernels_version"]
        and package_versions.get("triton") == model_contract["triton_version"]
        and kernel_load.get("repository") == model_contract["mxfp4_kernel_repo"]
        and kernel_load.get("revision") == model_contract["mxfp4_kernel_revision"]
        and int(kernel_load.get("matched_call_count", 0)) >= 1
        and bool(kernel_load.get("module_names"))
        and bool(kernel_load.get("module_files"))
        and all(value != "unknown" for value in kernel_load.get("module_files", []))
        and kernel_load.get("restored") is True
        and kernel_preflight.get("repository") == model_contract["mxfp4_kernel_repo"]
        and kernel_preflight.get("revision") == model_contract["mxfp4_kernel_revision"]
        and kernel_preflight.get("module_name") != "unknown"
        and kernel_preflight.get("module_file") != "unknown"
        and locked_versions.get("kernels") == model_contract["kernels_version"]
        and locked_versions.get("triton") == model_contract["triton_version"]
        and packaged_lock_versions.get("transformers") == model_contract["transformers_version"]
        and packaged_lock_versions.get("kernels") == model_contract["kernels_version"]
        and packaged_lock_versions.get("triton") == model_contract["triton_version"]
        and packaged_specs_match,
        {
            "package_versions": package_versions,
            "locked_versions": locked_versions,
            "packaged_lock_versions": {
                package: packaged_lock_versions.get(package)
                for package in ("transformers", "kernels", "triton")
            },
            "packaged_specs_match": packaged_specs_match,
            "kernel_preflight": kernel_preflight,
            "kernel_load": kernel_load,
        },
    )
    record(
        "runtime_template_hash",
        runtime_inspect.get("chat_template_sha256") == model_contract["chat_template_sha256"],
        runtime_inspect.get("chat_template_sha256"),
    )
    runtime_render = hook.get("render")
    runtime_prompt = runtime_manifest.get("prompt")
    runtime_render_ok = False
    runtime_render_detail = "missing runtime render"
    runtime_prompt_ok = False
    runtime_prompt_detail = "missing runtime prompt"
    runtime_render_hash = None
    runtime_prompt_hash = None
    if isinstance(runtime_render, dict):
        runtime_render_hash = runtime_render.get("rendered_prompt_hash")
        runtime_render_ok, runtime_render_detail = _check_harmony_render(
            runtime_render,
            text=runtime_render.get("rendered_prompt_text"),
            expected_hash=runtime_render.get("rendered_prompt_hash"),
            expected_template_sha256=str(model_contract["chat_template_sha256"]),
            expected_reasoning_effort=str(model_contract["harmony_reasoning_effort"]),
        )
    if isinstance(runtime_prompt, dict):
        runtime_prompt_hash = runtime_prompt.get("rendered_prompt_hash")
        runtime_prompt_ok, runtime_prompt_detail = _check_render_record(
            runtime_prompt,
            expected_template_sha256=str(model_contract["chat_template_sha256"]),
            expected_reasoning_effort=str(model_contract["harmony_reasoning_effort"]),
        )
    record(
        "runtime_harmony_render",
        runtime_render_ok
        and runtime_prompt_ok
        and runtime_render_hash == runtime_prompt_hash
        and stable_hash(runtime_prompt, length=64) == runtime_manifest.get("prompt_hash"),
        {
            "hook": runtime_render_detail,
            "manifest": runtime_prompt_detail,
            "prompt_hash": runtime_manifest.get("prompt_hash"),
        },
    )
    capture = hook.get("capture", {})
    expected_router_keys = {
        f"layer_{index}.router_logits" for index in range(int(expected["layers"]))
    }
    sentinel_layers = tuple(int(value) for value in hook_contract["sentinel_layers"])
    expected_capture_keys = expected_router_keys | {
        f"layer_{layer_index}.{kind}"
        for layer_index in sentinel_layers
        for kind in (
            "pre_moe_residual",
            "router_input",
            "router_topk_ids",
            "router_topk_weights",
            "moe_output",
            "post_moe_residual",
        )
    }
    observed_capture_keys = {str(value) for value in capture.get("sites", [])}
    fire_counts = capture.get("metadata", {}).get("hook_fire_counts", {})
    record(
        "all_capture_intercepts_fired",
        observed_capture_keys == expected_capture_keys
        and set(fire_counts) == expected_capture_keys
        and all(int(fire_counts[key]) == 1 for key in expected_capture_keys)
        and capture.get("metadata", {}).get("all_registered_hooks_fired") is True,
        {
            "expected": len(expected_capture_keys),
            "observed": len(observed_capture_keys),
            "missing": sorted(expected_capture_keys - observed_capture_keys),
            "extra": sorted(observed_capture_keys - expected_capture_keys),
        },
    )
    identity = hook.get("self_patch_identity", {})
    identity_layer = sentinel_layers[len(sentinel_layers) // 2]
    expected_identity_keys = {
        f"layer_{identity_layer}.{kind}"
        for kind in ("post_moe_residual", "moe_output", "router_logits")
    }
    identity_sites = identity.get("sites", {})
    identity_diagnostics_ok = (
        isinstance(identity_sites, dict) and set(identity_sites) == expected_identity_keys
    )
    if identity_diagnostics_ok:
        identity_diagnostics_ok = all(
            item.get("diagnostics", {}).get("all_registered_hooks_fired") is True
            for item in identity_sites.values()
            if isinstance(item, dict)
        ) and all(isinstance(item, dict) for item in identity_sites.values())
    router_identity = (
        identity_sites.get(f"layer_{identity_layer}.router_logits", {})
        if isinstance(identity_sites, dict)
        else {}
    )
    router_identity_dispatch_ok = (
        router_identity.get("diagnostics", {}).get("all_router_dispatch_verified") is True
    )
    record(
        "capture_and_self_patch_identity",
        float(capture.get("max_abs_logit_delta", float("inf")))
        <= float(hook_contract["capture_tolerance"])
        and float(identity.get("max_abs_logit_delta", float("inf")))
        <= float(hook_contract["self_patch_identity_tolerance"])
        and identity_diagnostics_ok
        and router_identity_dispatch_ok
        and identity.get("all_registered_hooks_fired") is True
        and identity.get("all_interceptors_restored") is True
        and identity.get("router_dispatch_verified") is True,
        {
            "capture": capture.get("max_abs_logit_delta"),
            "self_patch": identity.get("max_abs_logit_delta"),
            "identity_sites": sorted(identity_sites) if isinstance(identity_sites, dict) else None,
            "identity_diagnostics_ok": identity_diagnostics_ok,
            "router_identity_dispatch_ok": router_identity_dispatch_ok,
        },
    )
    forced = hook.get("forced_router", {})
    manipulation = forced.get("manipulation", {})
    dispatch = forced.get("dispatch", {})
    native_dispatch_ok = dispatch.get("router_execution_mode") != "mxfp4_functional_linear" or (
        dispatch.get("routing_input_matches_patched_topk") is True
        and dispatch.get("routing_artifacts_consumed") is True
        and forced.get("diagnostics", {}).get("mxfp4_routing_interceptor", {}).get("restored")
        is True
    )
    record(
        "forced_router_changes_downstream_dispatch",
        float(manipulation.get("topk_changed_rate", 0.0)) > 0.0
        and dispatch.get("downstream_dispatch_verified") is True
        and native_dispatch_ok
        and float(forced.get("max_abs_final_logit_delta", 0.0)) > 0.0,
        {
            "topk_changed_rate": manipulation.get("topk_changed_rate"),
            "dispatch": dispatch,
            "native_dispatch_ok": native_dispatch_ok,
            "final_logit_delta": forced.get("max_abs_final_logit_delta"),
        },
    )
    restoration = hook.get("restoration", {})
    record(
        "hook_restoration",
        restoration.get("functional_linear_restored_after_capture") is True
        and restoration.get("functional_linear_restored_final") is True
        and float(restoration.get("post_patch_rerun_max_abs_logit_delta", float("inf")))
        <= float(hook_contract["capture_tolerance"]),
        restoration,
    )
    record("runtime_hook_stage_pass", hook.get("passed") is True, hook.get("passed"))

    patch_dir = run_root / "patch"
    patch_manifest = _read_json(patch_dir / "data_manifest.json")
    patch_rows = _read_jsonl(patch_dir / "predictions.jsonl")
    patch_environment = _read_json(patch_dir / "environment.json")
    patch_selection = _read_json(patch_dir / "selection_manifest.json")
    patch_model = patch_environment.get("model", {})
    patch_pair_id_list = [str(value) for value in patch_manifest.get("pair_ids", [])]
    patch_pair_ids = set(patch_pair_id_list)
    runtime_pair_id_list = [str(value) for value in runtime_manifest.get("pair_ids", [])]
    runtime_pair_ids = set(runtime_pair_id_list)
    record(
        "cross_stage_pairs",
        len(patch_pair_ids) == int(patch_contract["expected_pairs"])
        and len(patch_pair_id_list) == len(patch_pair_ids)
        and int(patch_manifest.get("num_pairs", -1)) == len(patch_pair_ids)
        and patch_pair_ids == behavior_pairs
        and runtime_pair_ids == behavior_pairs
        and len(runtime_pair_id_list) == len(runtime_pair_ids)
        and int(runtime_manifest.get("num_pairs", -1)) == len(runtime_pair_ids)
        and set(behavior_manifest.get("pair_ids", [])) == behavior_pairs,
        {
            "behavior": sorted(behavior_pairs),
            "runtime": sorted(runtime_pair_ids),
            "patch": sorted(patch_pair_ids),
        },
    )
    record(
        "patch_no_validation",
        patch_manifest.get("validation_accessed") is False
        and patch_environment.get("validation_accessed") is False
        and patch_environment.get("split_partition") == "discovery",
        {
            "manifest": patch_manifest.get("validation_accessed"),
            "environment": patch_environment.get("validation_accessed"),
            "partition": patch_environment.get("split_partition"),
        },
    )
    stage_git = {
        "behavior": behavior_environment.get("git", {}),
        "runtime": runtime_environment.get("git", {}),
        "patch": patch_environment.get("git", {}),
    }
    stage_seeds = {
        "top_level": environment.get("seed"),
        "behavior": behavior_environment.get("seed"),
        "runtime": runtime_environment.get("seed"),
        "patch": patch_environment.get("seed"),
    }
    record(
        "stage_code_linkage",
        all(
            value.get("available") is True
            and value.get("commit") == code.get("commit")
            and value.get("dirty") is False
            for value in stage_git.values()
        )
        and all(value == data_contract["seed"] for value in stage_seeds.values()),
        {"git": stage_git, "seeds": stage_seeds},
    )
    record(
        "patch_model_revision",
        patch_model.get("model_id") == model_contract["model_id"]
        and patch_model.get("revision") == model_contract["revision"]
        and patch_model.get("tokenizer_revision") == model_contract["tokenizer_revision"],
        patch_model,
    )
    observed_site_list = [str(value) for value in patch_manifest.get("sites", [])]
    observed_sites = set(observed_site_list)
    record(
        "patch_sites",
        observed_sites == set(patch_contract["expected_sites"])
        and len(observed_site_list) == len(observed_sites),
        sorted(observed_sites),
    )
    observed_treatments = {str(row.get("treatment")) for row in patch_rows}
    record(
        "patch_treatments",
        set(patch_contract["required_treatments"]).issubset(observed_treatments),
        sorted(observed_treatments),
    )

    required_patch_rows: set[tuple[str, str, str, str | None]] = {
        (pair_id, variant, treatment, None)
        for pair_id in patch_pair_ids
        for variant in ("clean", "corrupt")
        for treatment in ("unpatched", "unpatched_rerun")
    }
    for pair_id in patch_pair_ids:
        for site in patch_contract["expected_sites"]:
            required_patch_rows.update(
                {
                    (pair_id, "clean", "self_patch", str(site)),
                    (pair_id, "corrupt", "self_patch", str(site)),
                    (pair_id, "corrupt", "clean_to_corrupt", str(site)),
                    (pair_id, "clean", "corrupt_to_clean", str(site)),
                    (pair_id, "clean", "norm_random_delta", str(site)),
                    (pair_id, "corrupt", "norm_random_delta", str(site)),
                }
            )
    patch_row_counts = Counter(
        (
            str(row.get("pair_id")),
            str(row.get("receiver_variant")),
            str(row.get("treatment")),
            None if row.get("site") is None else str(row.get("site")),
        )
        for row in patch_rows
    )
    missing_required_patch_rows = sorted(required_patch_rows - set(patch_row_counts))
    duplicate_required_patch_rows = sorted(
        key for key in required_patch_rows if patch_row_counts[key] > 1
    )
    record(
        "patch_required_row_coverage",
        not missing_required_patch_rows and not duplicate_required_patch_rows,
        {
            "required": len(required_patch_rows),
            "observed_required": len(required_patch_rows & set(patch_row_counts)),
            "missing": missing_required_patch_rows[:20],
            "missing_count": len(missing_required_patch_rows),
            "duplicates": duplicate_required_patch_rows[:20],
            "duplicate_count": len(duplicate_required_patch_rows),
        },
    )

    baselines = {
        (str(row["pair_id"]), str(row["receiver_variant"])): row
        for row in patch_rows
        if row.get("treatment") == "unpatched"
    }
    identity_deltas: list[float] = []
    rerun_deltas: list[float] = []
    missing_baselines: list[str] = []
    for row in patch_rows:
        if row.get("treatment") == "unpatched_rerun":
            key = (str(row["pair_id"]), str(row["receiver_variant"]))
            baseline = baselines.get(key)
            if baseline is None:
                missing_baselines.append(f"{key[0]}:{key[1]}:rerun")
            else:
                rerun_deltas.append(_score_delta(baseline, row))
        if row.get("treatment") != "self_patch":
            continue
        key = (str(row["pair_id"]), str(row["receiver_variant"]))
        baseline = baselines.get(key)
        if baseline is None:
            missing_baselines.append(f"{key[0]}:{key[1]}")
        else:
            identity_deltas.append(_score_delta(baseline, row))
    max_identity = max(identity_deltas, default=float("inf"))
    max_rerun = max(rerun_deltas, default=float("inf"))
    expected_identity_rows = (
        int(patch_contract["expected_pairs"]) * len(patch_contract["expected_sites"]) * 2
    )
    record(
        "patch_self_identity",
        not missing_baselines
        and len(identity_deltas) == expected_identity_rows
        and len(rerun_deltas) == int(patch_contract["expected_pairs"]) * 2
        and all(math.isfinite(value) for value in identity_deltas + rerun_deltas)
        and max_identity <= float(patch_contract["self_patch_identity_tolerance"])
        and max_rerun <= float(patch_contract["self_patch_identity_tolerance"]),
        {
            "rows": len(identity_deltas),
            "expected_rows": expected_identity_rows,
            "max_delta": max_identity,
            "rerun_rows": len(rerun_deltas),
            "max_rerun_delta": max_rerun,
            "missing_baselines": missing_baselines,
        },
    )

    prompt_inputs = patch_manifest.get("prompt_inputs", [])
    prompt_hash_ok = stable_hash(prompt_inputs, length=64) == patch_manifest.get(
        "prompt_inputs_hash"
    )
    prompt_pair_ids = [str(item.get("pair_id")) for item in prompt_inputs]
    prompt_pair_coverage_ok = (
        len(prompt_pair_ids) == len(patch_pair_ids)
        and len(prompt_pair_ids) == len(set(prompt_pair_ids))
        and set(prompt_pair_ids) == patch_pair_ids
    )
    render_failures: list[str] = []
    render_hashes: dict[tuple[str, str], str] = {}
    for item in prompt_inputs:
        pair_id = str(item.get("pair_id"))
        for variant in ("clean", "corrupt"):
            ok, detail = _check_render_record(
                item.get(variant, {}),
                expected_template_sha256=str(model_contract["chat_template_sha256"]),
                expected_reasoning_effort=str(model_contract["harmony_reasoning_effort"]),
            )
            if not ok:
                render_failures.append(f"{pair_id}:{variant}:{detail}")
            else:
                render_hashes[(pair_id, variant)] = detail
    row_render_failures: list[str] = []
    for row in patch_rows:
        key = (str(row.get("pair_id")), str(row.get("receiver_variant")))
        row_provenance = row.get("input_provenance", {})
        if row_provenance.get("rendered_prompt_hash") != render_hashes.get(key):
            row_render_failures.append(f"{key[0]}:{key[1]}:{row.get('treatment')}")
    record(
        "patch_render_reuse",
        prompt_hash_ok
        and prompt_pair_coverage_ok
        and not render_failures
        and not row_render_failures,
        {
            "prompt_inputs_hash_ok": prompt_hash_ok,
            "prompt_pair_coverage_ok": prompt_pair_coverage_ok,
            "manifest_failures": render_failures,
            "row_failures": row_render_failures[:20],
            "row_failure_count": len(row_render_failures),
        },
    )
    behavior_patch_corrupt_failures = sorted(
        pair_id
        for pair_id in patch_pair_ids
        if behavior_render_hashes.get((pair_id, "corrupt"))
        != render_hashes.get((pair_id, "corrupt"))
    )
    record(
        "behavior_patch_corrupt_render_linkage",
        not behavior_patch_corrupt_failures,
        {"failed_pairs": behavior_patch_corrupt_failures},
    )

    runtime_pair_id = str(runtime_manifest.get("runtime_pair_id"))
    runtime_patch_record = next(
        (item for item in prompt_inputs if str(item.get("pair_id")) == runtime_pair_id),
        None,
    )
    record(
        "runtime_patch_render_linkage",
        isinstance(runtime_patch_record, dict)
        and runtime_patch_record.get("corrupt", {}).get("rendered_prompt_hash")
        == runtime_prompt_hash,
        {
            "runtime_pair_id": runtime_pair_id,
            "runtime_hash": runtime_prompt_hash,
            "patch_hash": (
                None
                if not isinstance(runtime_patch_record, dict)
                else runtime_patch_record.get("corrupt", {}).get("rendered_prompt_hash")
            ),
        },
    )

    behavior_selection_hash = behavior_manifest.get("selection_manifest_hash")
    patch_selection_hash = patch_manifest.get("selection_manifest_hash")
    record(
        "selection_hash_linkage",
        isinstance(behavior_selection_hash, str)
        and behavior_selection_hash == patch_selection_hash
        and behavior_selection_hash == runtime_manifest.get("selection_manifest_hash")
        and behavior_selection.get("manifest_hash") == behavior_selection_hash
        and runtime_selection.get("manifest_hash") == behavior_selection_hash
        and patch_selection.get("manifest_hash") == behavior_selection_hash,
        {
            "behavior": behavior_selection_hash,
            "runtime": runtime_manifest.get("selection_manifest_hash"),
            "patch": patch_selection_hash,
        },
    )

    def validate_hash_manifest(
        section: str,
        expected_paths: set[str],
    ) -> tuple[bool, dict[str, Any]]:
        manifest = provenance.get(section, {})
        if not isinstance(manifest, dict):
            return False, {"reason": "section is not a mapping"}
        observed_paths: set[str] = set()
        bad_entries: list[str] = []
        for label, item in manifest.items():
            if not isinstance(item, dict):
                bad_entries.append(str(label))
                continue
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                bad_entries.append(str(label))
                continue
            observed_paths.add(relative.as_posix())
            path = run_root / relative
            if not path.is_file() or _sha256(path) != item.get("sha256"):
                bad_entries.append(str(label))
        missing_paths = sorted(expected_paths - observed_paths)
        return not bad_entries and not missing_paths, {
            "bad_entries": sorted(bad_entries),
            "missing_paths": missing_paths,
            "observed_paths": sorted(observed_paths),
        }

    config_paths = {
        "protocol.yaml",
        "configs/model.yaml",
        "configs/data.yaml",
        "configs/patch.source.yaml",
        "configs/patch.runtime.yaml",
        "configs/discovery_validation_split.json",
        "configs/pyproject.toml",
        "configs/uv.lock",
    }
    config_hashes_ok, config_hash_detail = validate_hash_manifest("config_hashes", config_paths)
    record("config_hashes", config_hashes_ok, config_hash_detail)
    artifact_paths = {
        "behavior/metrics.json",
        "behavior/predictions.jsonl",
        "behavior/data_manifest.json",
        "behavior/environment.json",
        "behavior/selection_manifest.json",
        "runtime/runtime_inspect.json",
        "runtime/hook_fidelity.json",
        "runtime/data_manifest.json",
        "runtime/environment.json",
        "runtime/selection_manifest.json",
        "runtime/model_discovery.json",
        "runtime/config.resolved.yaml",
        "patch/data_manifest.json",
        "patch/predictions.jsonl",
        "patch/metrics.json",
        "patch/environment.json",
        "patch/selection_manifest.json",
        "patch/model_discovery.json",
        "patch/config.resolved.yaml",
    }
    artifact_hashes_ok, artifact_hash_detail = validate_hash_manifest(
        "artifact_hashes", artifact_paths
    )
    record("artifact_hashes", artifact_hashes_ok, artifact_hash_detail)

    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": 1,
        "protocol_id": protocol.get("protocol_id"),
        "passed": passed,
        "decision": "PASS" if passed else "NO_GO",
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_count": sum(bool(check["passed"]) for check in checks),
            "failed": [check["name"] for check in checks if not check["passed"]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = gpt_oss_preflight_review(
        run_root=args.run_root,
        protocol_path=args.protocol,
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
