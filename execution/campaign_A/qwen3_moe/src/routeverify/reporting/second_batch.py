"""Strict local review and prerequisite gates for second-batch GPU artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from routeverify.data.schema import stable_hash
from routeverify.eval.verification import binary_verification_metrics
from routeverify.instrumentation.router_weights import load_router_weight_export

FROZEN_OLMOE_ALPHAS = (0.0, 0.5, 1.0, 2.0)
FROZEN_OLMOE_LAYERS = tuple(range(16))
FROZEN_OLMOE_CONTROLS = (
    "unpatched",
    "unpatched_rerun",
    "self_patch",
    "alpha_zero_identity",
)
OLMOE_SMOKE_REQUIRED_CHECKS = frozenset(
    {
        "required artifacts",
        "pair count",
        "behavior capability",
        "behavior exact pair coverage and recomputed AUROC",
        "unpatched rerun identity",
        "self-patch identity",
        "alpha=0 identity",
        "identity row coverage",
        "registered controls",
        "registered frozen protocol",
        "resolved frozen experiment config",
        "alpha dose-response coverage",
        "frozen smoke exact row contract",
        "paired boundary/token consistency",
        "simultaneous hook coverage",
        "routing manipulation",
        "forced reroute is conditional",
        "forced reroute changes actual top-k",
        "selection manifest hash",
        "patch selection IDs",
        "patch frozen discovery prefix",
        "behavior/patch frozen discovery split",
        "behavior selection provenance",
        "behavior frozen discovery prefix",
        "rendered input provenance hash",
        "frozen rendered input coverage",
        "model/data/code/config provenance",
        "frozen contract same clean code/environment",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _score_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    values = []
    for candidate in ("CORRECT", "INCORRECT"):
        for field in ("mean_logprob", "total_logprob"):
            values.append(abs(float(left[candidate][field]) - float(right[candidate][field])))
    return max(values, default=0.0)


def second_batch_smoke_review(
    *,
    patch_run: Path,
    behavior_run: Path,
    model_profile: str,
    expected_pairs: int,
    expected_hooks: int,
    recapture_run: Path | None = None,
    tolerance: float = 1e-5,
    min_behavior_auroc: float = 0.55,
    expected_behavior_pairs: int | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    required = [
        patch_run / "predictions.jsonl",
        patch_run / "data_manifest.json",
        patch_run / "selection_manifest.json",
        patch_run / "environment.json",
        patch_run / "config.resolved.yaml",
        patch_run / "metrics.json",
        behavior_run / "metrics.json",
        behavior_run / "environment.json",
        behavior_run / "predictions.jsonl",
        behavior_run / "selection_manifest.json",
        behavior_run / "data_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    check("required artifacts", not missing, {"missing": missing})
    if missing:
        return {"schema_version": 1, "passed": False, "checks": checks}

    rows = _read_jsonl(patch_run / "predictions.jsonl")
    manifest = _read_json(patch_run / "data_manifest.json")
    selection = _read_json(patch_run / "selection_manifest.json")
    behavior = _read_json(behavior_run / "metrics.json")
    patch_environment = _read_json(patch_run / "environment.json")
    behavior_environment = _read_json(behavior_run / "environment.json")
    behavior_selection = _read_json(behavior_run / "selection_manifest.json")
    behavior_manifest = _read_json(behavior_run / "data_manifest.json")
    behavior_rows = _read_jsonl(behavior_run / "predictions.jsonl")
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    check(
        "pair count",
        len(pair_ids) == expected_pairs and manifest.get("num_pairs") == expected_pairs,
        {"rows": len(rows), "pair_ids": len(pair_ids), "manifest": manifest.get("num_pairs")},
    )
    check(
        "behavior capability",
        behavior.get("auroc") is not None and float(behavior["auroc"]) >= min_behavior_auroc,
        {"auroc": behavior.get("auroc"), "minimum": min_behavior_auroc},
    )
    if expected_behavior_pairs is not None:
        recomputed_behavior = binary_verification_metrics(behavior_rows)
        behavior_pair_ids = {
            str(row.get("pair_id")) for row in behavior_rows if row.get("pair_id")
        }
        selected_behavior_pair_ids = [
            str(row.get("pair_id"))
            for row in behavior_selection.get("pairs", [])
        ]
        observed_variants = Counter(
            (str(row.get("pair_id")), str(row.get("variant")))
            for row in behavior_rows
        )
        expected_variants = Counter(
            (pair_id, variant)
            for pair_id in selected_behavior_pair_ids
            for variant in ("clean", "corrupt")
        )
        recorded_auroc = behavior.get("auroc")
        recomputed_auroc = recomputed_behavior.get("auroc")
        check(
            "behavior exact pair coverage and recomputed AUROC",
            len(behavior_rows) == 2 * expected_behavior_pairs
            and len(behavior_pair_ids) == expected_behavior_pairs
            and len(selected_behavior_pair_ids) == expected_behavior_pairs
            and observed_variants == expected_variants
            and recomputed_behavior.get("n") == 2 * expected_behavior_pairs
            and recomputed_behavior.get("positives") == expected_behavior_pairs
            and recomputed_behavior.get("negatives") == expected_behavior_pairs
            and recorded_auroc is not None
            and recomputed_auroc is not None
            and abs(float(recorded_auroc) - float(recomputed_auroc)) <= 1e-12
            and float(recomputed_auroc) >= min_behavior_auroc,
            {
                "rows": len(behavior_rows),
                "pairs": len(behavior_pair_ids),
                "expected_pairs": expected_behavior_pairs,
                "recorded_auroc": recorded_auroc,
                "recomputed_auroc": recomputed_auroc,
            },
        )

    baselines = {
        (str(row["pair_id"]), str(row["receiver_variant"])): row
        for row in rows
        if row["treatment"] == "unpatched"
    }
    reruns = [row for row in rows if row["treatment"] == "unpatched_rerun"]
    self_rows = [row for row in rows if row["treatment"] == "self_patch"]
    alpha_zero = [
        row
        for row in rows
        if row["treatment"] in {"clean_to_corrupt", "corrupt_to_clean"}
        and float(row.get("metadata", {}).get("alpha", -1.0)) == 0.0
    ]

    def max_identity_delta(identity_rows: list[dict[str, Any]]) -> float:
        deltas = []
        for row in identity_rows:
            baseline = baselines.get((str(row["pair_id"]), str(row["receiver_variant"])))
            if baseline is None:
                return float("inf")
            deltas.append(_score_delta(row["scores"], baseline["scores"]))
        return max(deltas, default=float("inf"))

    rerun_delta = max_identity_delta(reruns)
    self_delta = max_identity_delta(self_rows)
    alpha_zero_delta = max_identity_delta(alpha_zero)
    check("unpatched rerun identity", rerun_delta <= tolerance, rerun_delta)
    check("self-patch identity", self_delta <= tolerance, self_delta)
    check("alpha=0 identity", alpha_zero_delta <= tolerance, alpha_zero_delta)
    check(
        "identity row coverage",
        len(reruns) == 2 * expected_pairs
        and len(self_rows) == 2 * expected_pairs
        and len(alpha_zero) == 2 * expected_pairs,
        {
            "rerun": len(reruns),
            "self": len(self_rows),
            "alpha_zero": len(alpha_zero),
        },
    )
    configured_alphas = [
        float(value)
        for value in manifest.get("protocol", {}).get(
            "alphas", [0.0, 0.5, 1.0, 2.0]
        )
    ]
    forced_enabled = bool(manifest.get("protocol", {}).get("forced_reroute_control"))
    required_controls = {
        "unpatched",
        "unpatched_rerun",
        "self_patch",
        "alpha_zero_identity",
    }
    if forced_enabled:
        required_controls.add("forced_reroute_if_needed")
    recorded_controls = set(manifest.get("protocol", {}).get("controls", []))
    check(
        "registered controls",
        required_controls <= recorded_controls,
        {"recorded": sorted(recorded_controls)},
    )
    if contract is not None:
        frozen_protocol = cast(Mapping[str, Any], contract["protocol"])
        check(
            "registered frozen protocol",
            configured_alphas
            == [float(value) for value in frozen_protocol["alphas"]]
            and forced_enabled is bool(frozen_protocol["forced_reroute_control"])
            and list(manifest.get("protocol", {}).get("controls", []))
            == list(frozen_protocol["controls"])
            and all(
                manifest.get("protocol", {}).get(field) == frozen_protocol[field]
                for field in (
                    "prompt_mode",
                    "input_format",
                    "boundary_strategy",
                    "patch_mode",
                    "split_partition",
                    "sites",
                    "layers",
                )
            ),
            {
                "alphas": configured_alphas,
                "forced_reroute_control": forced_enabled,
                "controls": list(manifest.get("protocol", {}).get("controls", [])),
            },
        )
        resolved_manifest = manifest.get("resolved_config")
        resolved_file = yaml.safe_load(
            (patch_run / "config.resolved.yaml").read_text(encoding="utf-8")
        )
        experiment = cast(Mapping[str, Any], contract["experiment_config"])
        science_fields = (
            "model",
            "data",
            "seed",
            "limit",
            "prompt_mode",
            "input_format",
            "boundary_strategy",
            "patch_mode",
            "alphas",
            "forced_reroute_control",
            "controls",
            "split_manifest",
            "split_partition",
            "expected_router_hooks",
            "sites",
            "layers",
        )
        check(
            "resolved frozen experiment config",
            isinstance(resolved_manifest, Mapping)
            and resolved_file == resolved_manifest
            and all(
                resolved_manifest.get(field) == experiment.get(field)
                for field in science_fields
            )
            and resolved_manifest.get("actual_protocol")
            == manifest.get("protocol"),
            {
                "file_matches_manifest": resolved_file == resolved_manifest,
                "experiment_config_sha256": contract.get(
                    "experiment_config_sha256"
                ),
            },
        )
    missing_doses: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        for treatment in ("clean_to_corrupt", "corrupt_to_clean"):
            observed = sorted(
                float(row.get("metadata", {}).get("alpha"))
                for row in rows
                if row["pair_id"] == pair_id
                and row["treatment"] == treatment
                and row.get("metadata", {}).get("alpha") is not None
            )
            if observed != sorted(configured_alphas):
                missing_doses.append(
                    {
                        "pair_id": pair_id,
                        "treatment": treatment,
                        "observed": observed,
                    }
                )
    check(
        "alpha dose-response coverage",
        not missing_doses,
        {"alphas": configured_alphas, "missing": missing_doses[:5]},
    )
    if contract is not None:
        bundle_site = f"bundle.router_logits[{expected_hooks}]"
        base_descriptors = _expected_discovery_descriptors(
            pair_ids,
            alphas=[float(value) for value in contract["protocol"]["alphas"]],
            bundle_site=bundle_site,
        )
        observed_descriptors = Counter(_row_descriptor(row) for row in rows)
        missing_base = base_descriptors - observed_descriptors
        extra = observed_descriptors - base_descriptors
        allowed_extra: set[tuple[Any, ...]] = set()
        if contract.get("run_role") == "smoke":
            for pair_id in pair_ids:
                allowed_extra.update(
                    {
                        (
                            pair_id,
                            "forced_reroute_clean_receiver",
                            "clean",
                            "forced",
                            None,
                            bundle_site,
                        ),
                        (
                            pair_id,
                            "forced_reroute_corrupt_receiver",
                            "corrupt",
                            "forced",
                            None,
                            bundle_site,
                        ),
                    }
                )
            arm_controls_ok = all(
                (
                    row.get("metadata", {}).get("control") == "dose_response"
                    if row.get("treatment")
                    in {"clean_to_corrupt", "corrupt_to_clean"}
                    else row.get("metadata", {}).get("control")
                    == "forced_reroute_smoke_only"
                    if str(row.get("treatment", "")).startswith("forced_reroute")
                    else True
                )
                for row in rows
            )
            extra_ok = all(
                descriptor in allowed_extra and count == 1
                for descriptor, count in extra.items()
            )
            patch_metrics = _read_json(patch_run / "metrics.json")
            excluded_rows_ok = (
                patch_metrics.get("excluded_smoke_control_rows")
                == sum(extra.values())
            )
            check(
                "frozen smoke exact row contract",
                not missing_base
                and extra_ok
                and arm_controls_ok
                and excluded_rows_ok
                and len(rows) == 14 * expected_pairs + sum(extra.values()),
                {
                    "rows": len(rows),
                    "base_rows": 14 * expected_pairs,
                    "forced_rows": sum(extra.values()),
                    "missing_base": sum(missing_base.values()),
                    "unexpected_rows": sum(
                        count
                        for descriptor, count in extra.items()
                        if descriptor not in allowed_extra or count != 1
                    ),
                    "controls_ok": arm_controls_ok,
                    "excluded_rows_ok": excluded_rows_ok,
                },
            )
        pair_consistency_failures: list[dict[str, Any]] = []
        for pair_id in pair_ids:
            pair_rows = [
                row for row in rows if str(row.get("pair_id")) == pair_id
            ]
            boundaries = {row.get("boundary_step") for row in pair_rows}
            for receiver in ("clean", "corrupt"):
                receiver_rows = [
                    row
                    for row in pair_rows
                    if row.get("receiver_variant") == receiver
                ]
                token_positions = {
                    row.get("token_position") for row in receiver_rows
                }
                provenances = {
                    stable_hash(row.get("input_provenance", {}), length=64)
                    for row in receiver_rows
                }
                if (
                    len(boundaries) != 1
                    or len(token_positions) != 1
                    or len(provenances) != 1
                ):
                    pair_consistency_failures.append(
                        {
                            "pair_id": pair_id,
                            "receiver": receiver,
                            "boundaries": len(boundaries),
                            "token_positions": len(token_positions),
                            "input_provenances": len(provenances),
                        }
                    )
        check(
            "paired boundary/token consistency",
            not pair_consistency_failures,
            pair_consistency_failures[:5],
        )

    bundle_rows = [
        row for row in rows if row.get("metadata", {}).get("patch_mode") == "simultaneous_bundle"
    ]
    hook_failures: list[dict[str, Any]] = []
    changed_rates: list[float] = []
    for row in bundle_rows:
        manipulation = row["metadata"].get("manipulation", {})
        expected_site_keys = None
        candidate_diagnostics_ok = True
        if contract is not None:
            expected_site_keys = {
                f"layer_{int(layer)}.router_logits"
                for layer in contract["protocol"]["layers"]
            }
            candidate_diagnostics = manipulation.get(
                "candidate_diagnostics", {}
            )
            candidate_diagnostics_ok = (
                manipulation.get("candidate_count") == 2
                and isinstance(candidate_diagnostics, Mapping)
                and set(candidate_diagnostics) == {"CORRECT", "INCORRECT"}
            )
            if isinstance(candidate_diagnostics, Mapping):
                for candidate_payload in candidate_diagnostics.values():
                    candidate_diagnostics_ok = candidate_diagnostics_ok and (
                        isinstance(candidate_payload, Mapping)
                        and candidate_payload.get("simultaneous") is True
                        and candidate_payload.get("registered_site_count")
                        == expected_hooks
                        and set(candidate_payload.get("registered_sites", []))
                        == expected_site_keys
                        and set(candidate_payload.get("hook_fire_counts", {}))
                        == expected_site_keys
                        and all(
                            int(value) > 0
                            for value in candidate_payload.get(
                                "hook_fire_counts", {}
                            ).values()
                        )
                        and set(
                            candidate_payload.get("manipulation_by_site", {})
                        )
                        == expected_site_keys
                    )
        if (
            manipulation.get("registered_site_count") != expected_hooks
            or not manipulation.get("all_registered_hooks_fired")
            or len(manipulation.get("hook_fire_counts", {})) != expected_hooks
            or any(int(count) <= 0 for count in manipulation.get("hook_fire_counts", {}).values())
            or (
                expected_site_keys is not None
                and set(manipulation.get("hook_fire_counts", {}))
                != expected_site_keys
            )
            or not candidate_diagnostics_ok
        ):
            hook_failures.append(
                {
                    "pair_id": row["pair_id"],
                    "treatment": row["treatment"],
                    "registered_site_count": manipulation.get("registered_site_count"),
                }
            )
        if manipulation.get("topk_changed_rate") is not None:
            changed_rates.append(float(manipulation["topk_changed_rate"]))
    check(
        "simultaneous hook coverage",
        bool(bundle_rows) and not hook_failures,
        {"bundle_rows": len(bundle_rows), "failures": hook_failures[:5]},
    )
    check(
        "routing manipulation",
        bool(changed_rates) and (max(changed_rates) > 0.0 or not forced_enabled),
        {
            "max_topk_changed_rate": max(changed_rates, default=0.0),
            "forced_control_enabled": forced_enabled,
        },
    )
    conditional_control_failures: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        for direction, forced_treatment in (
            ("clean_to_corrupt", "forced_reroute_corrupt_receiver"),
            ("corrupt_to_clean", "forced_reroute_clean_receiver"),
        ):
            dose_changes = [
                float(row["metadata"]["manipulation"]["topk_changed_rate"] or 0.0)
                for row in rows
                if row["pair_id"] == pair_id and row["treatment"] == direction
            ]
            forced_rows = [
                row
                for row in rows
                if row["pair_id"] == pair_id
                and row["treatment"] == forced_treatment
            ]
            needs_forced = max(dose_changes, default=0.0) <= 0.01
            forced_changes_route = any(
                float(row["metadata"]["manipulation"]["topk_changed_rate"] or 0.0)
                > 0.0
                for row in forced_rows
            )
            if (needs_forced and not forced_changes_route) or (
                not needs_forced and forced_rows
            ):
                conditional_control_failures.append(
                    {
                        "pair_id": pair_id,
                        "direction": direction,
                        "max_dose_change": max(dose_changes, default=0.0),
                        "forced_rows": len(forced_rows),
                    }
                )
    if forced_enabled:
        check(
            "forced reroute is conditional",
            not conditional_control_failures,
            conditional_control_failures[:5],
        )
        forced_rows = [
            row
            for row in rows
            if row.get("metadata", {}).get("control") == "forced_reroute_smoke_only"
        ]
        check(
            "forced reroute changes actual top-k",
            not forced_rows
            or all(
                float(row["metadata"]["manipulation"].get("topk_changed_rate") or 0.0)
                > 0.0
                for row in forced_rows
            ),
            {"rows": len(forced_rows)},
        )

    unhashed_selection = dict(selection)
    selection_hash = unhashed_selection.pop("manifest_hash", None)
    check(
        "selection manifest hash",
        selection_hash == stable_hash(unhashed_selection, length=64)
        and manifest.get("selection_manifest_hash") == selection_hash,
        {
            "recorded": selection_hash,
            "referenced": manifest.get("selection_manifest_hash"),
        },
    )
    def selection_identity(
        selection_payload: dict[str, Any],
    ) -> tuple[set[str], set[str], str | None, str | None, str | None]:
        pairs = selection_payload.get("pairs", [])
        frozen = selection_payload.get("frozen_split")
        if not isinstance(frozen, dict):
            frozen = {}
        return (
            {str(row.get("pair_id")) for row in pairs if row.get("pair_id")},
            {str(row.get("group_id")) for row in pairs if row.get("group_id")},
            frozen.get("partition"),
            frozen.get("manifest_hash"),
            selection_payload.get("manifest_hash"),
        )

    behavior_unhashed = dict(behavior_selection)
    behavior_selection_hash = behavior_unhashed.pop("manifest_hash", None)
    behavior_hash_ok = behavior_selection_hash == stable_hash(
        behavior_unhashed, length=64
    )
    patch_identity = selection_identity(selection)
    behavior_identity = selection_identity(behavior_selection)
    check(
        "patch selection IDs",
        set(str(value) for value in manifest.get("pair_ids", [])) == patch_identity[0]
        and set(str(value) for value in manifest.get("group_ids", []))
        == patch_identity[1],
        {
            "manifest_pairs": len(manifest.get("pair_ids", [])),
            "selection_pairs": len(patch_identity[0]),
        },
    )
    if contract is not None:
        selection_pairs = selection.get("pairs", [])
        ordered_pair_ids = [str(row.get("pair_id")) for row in selection_pairs]
        ordered_group_ids = [str(row.get("group_id")) for row in selection_pairs]
        expected_groups = list(contract.get("_discovery_group_ids", []))[
            :expected_pairs
        ]
        check(
            "patch frozen discovery prefix",
            _selection_manifest_is_valid(selection)
            and len(selection_pairs) == expected_pairs
            and len(set(ordered_pair_ids)) == expected_pairs
            and len(set(ordered_group_ids)) == expected_pairs
            and ordered_group_ids == expected_groups
            and selection.get("requested_pair_limit") == expected_pairs
            and selection.get("available_pair_count")
            == len(contract.get("_discovery_group_ids", []))
            and selection.get("selected_pair_count") == expected_pairs
            and selection.get("selected_example_count") == 2 * expected_pairs
            and manifest.get("pair_ids") == ordered_pair_ids
            and manifest.get("group_ids") == ordered_group_ids,
            {
                "pairs": len(selection_pairs),
                "expected_pairs": expected_pairs,
                "ordered_prefix_match": ordered_group_ids == expected_groups,
            },
        )
    split_ok = (
        patch_identity[2] == "discovery"
        and behavior_identity[2] == "discovery"
        and patch_identity[3]
        and patch_identity[3] == behavior_identity[3]
        and (
            patch_identity[0] <= behavior_identity[0]
            or behavior_identity[0] <= patch_identity[0]
        )
        and (
            patch_identity[1] <= behavior_identity[1]
            or behavior_identity[1] <= patch_identity[1]
        )
    )
    check(
        "behavior/patch frozen discovery split",
        bool(behavior_hash_ok and split_ok),
        {
            "patch_partition": patch_identity[2],
            "behavior_partition": behavior_identity[2],
            "split_hash_match": patch_identity[3] == behavior_identity[3],
            "patch_pairs": len(patch_identity[0]),
            "behavior_pairs": len(behavior_identity[0]),
        },
    )
    behavior_rows_pairs = {
        str(row.get("pair_id")) for row in behavior_rows if row.get("pair_id")
    }
    behavior_rows_groups = {
        str(row.get("group_id")) for row in behavior_rows if row.get("group_id")
    }
    check(
        "behavior selection provenance",
        behavior_manifest.get("selection_manifest_hash") == behavior_selection_hash
        and behavior_manifest.get("selection_rows_hash")
        == behavior_environment.get("data_provenance", {})
        .get("selection", {})
        .get("rows_hash")
        and behavior_manifest.get("selected_rows_hash")
        == stable_hash(
            behavior_environment.get("data_provenance", {})
            .get("selection", {})
            .get("rows", []),
            length=64,
        )
        and behavior_manifest.get("pair_ids") == sorted(behavior_rows_pairs)
        and behavior_manifest.get("group_ids") == sorted(behavior_rows_groups)
        and behavior_rows_pairs == behavior_identity[0]
        and behavior_rows_groups == behavior_identity[1],
        behavior_manifest,
    )
    if contract is not None and expected_behavior_pairs is not None:
        expected_behavior_groups = list(
            contract.get("_discovery_group_ids", [])
        )[:expected_behavior_pairs]
        behavior_selection_pairs = behavior_selection.get("pairs", [])
        observed_behavior_groups = [
            str(row.get("group_id")) for row in behavior_selection_pairs
        ]
        check(
            "behavior frozen discovery prefix",
            _selection_manifest_is_valid(behavior_selection)
            and len(behavior_selection_pairs) == expected_behavior_pairs
            and observed_behavior_groups == expected_behavior_groups
            and behavior_selection.get("requested_pair_limit")
            == expected_behavior_pairs
            and behavior_selection.get("available_pair_count")
            == len(contract.get("_discovery_group_ids", []))
            and behavior_selection.get("selected_pair_count")
            == expected_behavior_pairs
            and behavior_selection.get("selected_example_count")
            == 2 * expected_behavior_pairs,
            {
                "pairs": len(behavior_selection_pairs),
                "expected_pairs": expected_behavior_pairs,
                "ordered_prefix_match": observed_behavior_groups
                == expected_behavior_groups,
            },
        )
    prompt_inputs = manifest.get("prompt_inputs")
    prompt_input_rows = prompt_inputs if isinstance(prompt_inputs, list) else []
    check(
        "rendered input provenance hash",
        isinstance(prompt_inputs, list)
        and manifest.get("prompt_inputs_hash") == stable_hash(prompt_inputs, length=64),
        manifest.get("prompt_inputs_hash"),
    )
    if contract is not None:
        ordered_selection_pair_ids = [
            str(row.get("pair_id")) for row in selection.get("pairs", [])
        ]
        prompt_by_pair = {
            str(row.get("pair_id")): row
            for row in prompt_input_rows
            if isinstance(row, Mapping)
        }
        input_failures: list[dict[str, Any]] = []
        for row in rows:
            pair_id = str(row.get("pair_id"))
            receiver = str(row.get("receiver_variant"))
            provenance = row.get("input_provenance", {})
            prompt_row = prompt_by_pair.get(pair_id, {})
            if (
                prompt_row.get(receiver) != provenance
                or provenance.get("prompt_mode")
                != contract["protocol"]["prompt_mode"]
                or provenance.get("input_format")
                != contract["protocol"]["input_format"]
                or not provenance.get("raw_prompt_hash")
                or not provenance.get("rendered_prompt_hash")
                or not provenance.get("input_ids_hash")
            ):
                input_failures.append(
                    {
                        "pair_id": pair_id,
                        "receiver": receiver,
                        "treatment": row.get("treatment"),
                    }
                )
        check(
            "frozen rendered input coverage",
            len(prompt_input_rows) == expected_pairs
            and list(prompt_by_pair) == ordered_selection_pair_ids
            and not input_failures,
            {
                "prompt_inputs": len(prompt_input_rows),
                "expected_pairs": expected_pairs,
                "failures": input_failures[:5],
            },
        )
    behavior_model = behavior_environment.get("model", {})
    patch_model = patch_environment.get("model", {})
    provenance_ok = (
        bool(manifest.get("code", {}).get("commit"))
        and manifest.get("code", {}).get("dirty") is False
        and bool(manifest.get("command", {}).get("argv"))
        and bool(manifest.get("resolved_config"))
        and bool(manifest.get("dataset_source", {}).get("requested_revision"))
        and bool(patch_model.get("revision"))
        and behavior_model.get("model_id") == patch_model.get("model_id")
        and behavior_model.get("revision") == patch_model.get("revision")
        and bool(behavior_environment.get("data_provenance"))
    )
    check("model/data/code/config provenance", provenance_ok, provenance_ok)
    if contract is not None:
        patch_git = patch_environment.get("git", {})
        behavior_git = behavior_environment.get("git", {})
        check(
            "frozen contract same clean code/environment",
            patch_git.get("commit")
            and patch_git == behavior_git
            and patch_git.get("dirty") is False
            and manifest.get("code") == patch_git
            and patch_environment.get("model")
            == behavior_environment.get("model")
            and patch_environment.get("data")
            == behavior_environment.get("data")
            and patch_environment.get("files")
            == behavior_environment.get("files"),
            {
                "patch_commit": patch_git.get("commit"),
                "behavior_commit": behavior_git.get("commit"),
                "patch_dirty": patch_git.get("dirty"),
                "behavior_dirty": behavior_git.get("dirty"),
            },
        )
    if model_profile == "qwen":
        export = manifest.get("router_weight_export")
        export_error = None
        exported_matrices: dict[int, Any] = {}
        loaded_export: dict[str, Any] = {}
        if isinstance(export, dict) and export.get("matrix_file"):
            try:
                exported_matrices, loaded_export = load_router_weight_export(
                    patch_run / str(export["matrix_file"])
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                export_error = str(exc)
        check(
            "Qwen router weight export",
            isinstance(export, dict)
            and export.get("num_layers") == expected_hooks
            and bool(export.get("aggregate_matrix_sha256"))
            and loaded_export == export
            and len(exported_matrices) == expected_hooks,
            {"manifest": export, "error": export_error},
        )
        recapture_required = [
            recapture_run / "data_manifest.jsonl",
            recapture_run / "environment.json",
            recapture_run / "selection_manifest.json",
            recapture_run / "config.resolved.yaml",
        ] if recapture_run is not None else []
        recapture_missing = [
            str(path) for path in recapture_required if not path.is_file()
        ]
        recapture_rows = (
            _read_jsonl(recapture_run / "data_manifest.jsonl")
            if recapture_run is not None and not recapture_missing
            else []
        )
        recapture_selection = (
            _read_json(recapture_run / "selection_manifest.json")
            if recapture_run is not None and not recapture_missing
            else {}
        )
        recapture_environment = (
            _read_json(recapture_run / "environment.json")
            if recapture_run is not None and not recapture_missing
            else {}
        )
        recapture_identity = selection_identity(recapture_selection)
        recapture_unhashed = dict(recapture_selection)
        recapture_recorded_hash = recapture_unhashed.pop("manifest_hash", None)
        recapture_hash_ok = recapture_recorded_hash == stable_hash(
            recapture_unhashed, length=64
        )
        recapture_activation_missing = []
        if recapture_run is not None:
            for row in recapture_rows:
                value = Path(str(row.get("activation_file", "")))
                candidates = (
                    [value]
                    if value.is_absolute()
                    else [
                        value,
                        recapture_run / value,
                        recapture_run / value.name,
                        recapture_run / "activations" / value.name,
                    ]
                )
                if not any(candidate.is_file() for candidate in candidates):
                    recapture_activation_missing.append(str(value))
        recapture_model = recapture_environment.get("model", {})
        recapture_row_pairs = {
            str(row.get("pair_id")) for row in recapture_rows if row.get("pair_id")
        }
        recapture_row_groups = {
            str(row.get("group_id")) for row in recapture_rows if row.get("group_id")
        }
        recapture_ok = (
            recapture_run is not None
            and not recapture_missing
            and len(recapture_rows) == expected_pairs * 2
            and all(row.get("input_format") == "chat" for row in recapture_rows)
            and all(
                row.get("boundary_strategy") == "paired_first_error"
                for row in recapture_rows
            )
            and all(
                len(row.get("arrays", {})) == expected_hooks * 4
                for row in recapture_rows
            )
            and all(row.get("input_ids_hash") for row in recapture_rows)
            and not recapture_activation_missing
            and recapture_hash_ok
            and recapture_identity == patch_identity
            and recapture_row_pairs == patch_identity[0]
            and recapture_row_groups == patch_identity[1]
            and recapture_model.get("model_id") == patch_model.get("model_id")
            and recapture_model.get("revision") == patch_model.get("revision")
            and bool(recapture_environment.get("data_provenance"))
        )
        check(
            "Qwen split-first paired recapture",
            recapture_ok,
            {
                "run": None if recapture_run is None else str(recapture_run),
                "missing": recapture_missing,
                "rows": len(recapture_rows),
                "missing_activations": recapture_activation_missing[:5],
                "same_selection": recapture_identity == patch_identity,
            },
        )
    prediction_hash = hashlib.sha256((patch_run / "predictions.jsonl").read_bytes()).hexdigest()
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "model_profile": model_profile,
        "expected_pairs": expected_pairs,
        "expected_hooks": expected_hooks,
        "prediction_rows": len(rows),
        "predictions_sha256": prediction_hash,
        "checks": checks,
    }


def build_olmoe_review_contract(
    *,
    gate_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    run_role: str,
    gate_config_sha256: str | None = None,
    experiment_config_sha256: str | None = None,
    experiment_config_path: str | None = None,
) -> dict[str, Any]:
    """Bind a reviewed OLMoE experiment config to the current machine gate."""

    role_key = {"smoke": "smoke", "frozen_discovery": "discovery"}.get(run_role)
    if role_key is None:
        raise ValueError("run_role must be smoke or frozen_discovery")
    behavior_gate = gate_config.get("behavior", {})
    identity_gate = gate_config.get("identity", {})
    if (
        behavior_gate.get("metric") != "auroc"
        or float(behavior_gate.get("minimum", -1.0)) != 0.55
        or float(
            identity_gate.get("candidate_score_max_abs_delta", -1.0)
        )
        != 1.0e-5
    ):
        raise ValueError(
            "gate config drifts from behavior AUROC>=0.55 or identity<=1e-5"
        )
    profile = gate_config.get("profiles", {}).get("olmoe", {})
    role = profile.get(role_key, {})
    if not isinstance(profile, Mapping) or not isinstance(role, Mapping):
        raise ValueError(f"gate config is missing profiles.olmoe.{role_key}")
    required = {
        "expected_pairs",
        "expected_behavior_pairs",
        "alphas",
        "forced_reroute_control",
        "controls",
        "prompt_mode",
        "input_format",
        "boundary_strategy",
        "patch_mode",
        "sites",
        "layers",
        "split_partition",
    }
    if run_role == "frozen_discovery":
        required |= {"expected_prediction_rows", "expected_bundle_rows"}
    missing = sorted(required - set(role))
    if missing:
        raise ValueError(
            f"gate config profiles.olmoe.{role_key} is missing: {missing}"
        )
    expected_hooks = profile.get("expected_router_hooks")
    if expected_hooks is None:
        raise ValueError("gate config is missing profiles.olmoe.expected_router_hooks")
    invariant = {
        "expected_pairs": 8 if run_role == "smoke" else 290,
        "expected_behavior_pairs": 64,
        "alphas": list(FROZEN_OLMOE_ALPHAS),
        "forced_reroute_control": run_role == "smoke",
        "controls": list(FROZEN_OLMOE_CONTROLS)
        + (["forced_reroute_if_needed"] if run_role == "smoke" else []),
        "prompt_mode": "step_binary",
        "input_format": "chat",
        "boundary_strategy": "paired_first_error",
        "patch_mode": "simultaneous_bundle",
        "sites": ["router_logits"],
        "layers": list(FROZEN_OLMOE_LAYERS),
        "split_partition": "discovery",
    }
    if run_role == "frozen_discovery":
        invariant.update(
            {
                "expected_prediction_rows": 4060,
                "expected_bundle_rows": 2900,
            }
        )
    invariant_mismatches = {
        field: {"required": expected, "gate": role.get(field)}
        for field, expected in invariant.items()
        if role.get(field) != expected
    }
    if int(expected_hooks) != 16:
        invariant_mismatches["expected_router_hooks"] = {
            "required": 16,
            "gate": expected_hooks,
        }
    if invariant_mismatches:
        raise ValueError(
            f"gate config drifts from the frozen OLMoE contract: {invariant_mismatches}"
        )

    protocol_fields = (
        "alphas",
        "forced_reroute_control",
        "controls",
        "prompt_mode",
        "input_format",
        "boundary_strategy",
        "patch_mode",
        "sites",
        "layers",
        "split_partition",
    )
    mismatches: dict[str, Any] = {}
    for field in protocol_fields:
        expected = role[field]
        observed = experiment_config.get(field)
        if field == "alphas":
            expected = [float(value) for value in expected]
            observed = [float(value) for value in (observed or [])]
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    scalar_expectations = {
        "limit": int(role["expected_pairs"]),
        "expected_router_hooks": int(expected_hooks),
    }
    for field, expected in scalar_expectations.items():
        observed = experiment_config.get(field)
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(f"experiment config differs from frozen gate contract: {mismatches}")

    split_payload = dict(split_manifest)
    split_recorded_hash = split_payload.pop("manifest_hash", None)
    split_hash_ok = split_recorded_hash == stable_hash(split_payload, length=64)
    discovery_group_ids = [
        str(value) for value in split_manifest.get("discovery_group_ids", [])
    ]
    discovery_hash_ok = split_manifest.get("discovery_groups_hash") == stable_hash(
        discovery_group_ids, length=64
    )
    if not split_hash_ok or not discovery_hash_ok or not discovery_group_ids:
        raise ValueError("frozen split manifest hash/selection is invalid")
    if len(discovery_group_ids) < int(role["expected_pairs"]):
        raise ValueError("frozen discovery partition is smaller than expected_pairs")

    return {
        "schema_version": 1,
        "run_role": run_role,
        "model_profile": "olmoe",
        "expected_pairs": int(role["expected_pairs"]),
        "expected_behavior_pairs": int(role["expected_behavior_pairs"]),
        "expected_prediction_rows": (
            None
            if role.get("expected_prediction_rows") is None
            else int(role["expected_prediction_rows"])
        ),
        "expected_bundle_rows": (
            None
            if role.get("expected_bundle_rows") is None
            else int(role["expected_bundle_rows"])
        ),
        "expected_hooks": int(expected_hooks),
        "protocol": {
            field: (
                [float(value) for value in role[field]]
                if field == "alphas"
                else role[field]
            )
            for field in protocol_fields
        },
        "experiment_config": dict(experiment_config),
        "gate_config_sha256": gate_config_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "experiment_config_path": experiment_config_path,
        "split_manifest_hash": split_recorded_hash,
        "_discovery_group_ids": discovery_group_ids,
    }


def _public_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in contract.items()
        if not str(key).startswith("_")
    }


def _selection_manifest_is_valid(selection: Mapping[str, Any]) -> bool:
    payload = dict(selection)
    recorded = payload.pop("manifest_hash", None)
    pairs = selection.get("pairs")
    return (
        isinstance(pairs, list)
        and recorded == stable_hash(payload, length=64)
        and selection.get("pairs_hash") == stable_hash(pairs, length=64)
    )


def _row_descriptor(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    alpha = metadata.get("alpha")
    return (
        str(row.get("pair_id")),
        row.get("treatment"),
        row.get("receiver_variant"),
        row.get("donor_variant"),
        None if alpha is None else float(alpha),
        row.get("site"),
    )


def _expected_discovery_descriptors(
    pair_ids: list[str],
    *,
    alphas: list[float],
    bundle_site: str,
) -> Counter[tuple[Any, ...]]:
    expected: Counter[tuple[Any, ...]] = Counter()
    for pair_id in pair_ids:
        expected.update(
            {
                (pair_id, "unpatched", "clean", None, None, None): 1,
                (pair_id, "unpatched", "corrupt", None, None, None): 1,
                (pair_id, "unpatched_rerun", "clean", None, None, None): 1,
                (pair_id, "unpatched_rerun", "corrupt", None, None, None): 1,
                (pair_id, "self_patch", "clean", "clean", None, bundle_site): 1,
                (pair_id, "self_patch", "corrupt", "corrupt", None, bundle_site): 1,
            }
        )
        for alpha in alphas:
            expected[
                (
                    pair_id,
                    "clean_to_corrupt",
                    "corrupt",
                    "clean",
                    alpha,
                    bundle_site,
                )
            ] += 1
            expected[
                (
                    pair_id,
                    "corrupt_to_clean",
                    "clean",
                    "corrupt",
                    alpha,
                    bundle_site,
                )
            ] += 1
    return expected


def second_batch_frozen_discovery_review(
    *,
    patch_run: Path,
    behavior_run: Path,
    contract: Mapping[str, Any],
    tolerance: float = 1e-5,
    min_behavior_auroc: float = 0.55,
) -> dict[str, Any]:
    """Strictly review the pre-registered OLMoE n=290 discovery artifact."""

    expected_pairs = int(contract["expected_pairs"])
    expected_hooks = int(contract["expected_hooks"])
    payload = second_batch_smoke_review(
        patch_run=patch_run,
        behavior_run=behavior_run,
        model_profile="olmoe",
        expected_pairs=expected_pairs,
        expected_hooks=expected_hooks,
        tolerance=tolerance,
        min_behavior_auroc=min_behavior_auroc,
        expected_behavior_pairs=int(contract["expected_behavior_pairs"]),
        contract=contract,
    )
    checks = cast(list[dict[str, Any]], payload["checks"])

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    payload["run_role"] = "frozen_discovery"
    payload["expected_behavior_pairs"] = int(contract["expected_behavior_pairs"])
    payload["contract"] = _public_contract(contract)
    if not any(item["name"] == "required artifacts" and item["passed"] for item in checks):
        payload["passed"] = False
        return payload

    rows = _read_jsonl(patch_run / "predictions.jsonl")
    manifest = _read_json(patch_run / "data_manifest.json")
    selection = _read_json(patch_run / "selection_manifest.json")
    patch_environment = _read_json(patch_run / "environment.json")
    resolved_file = yaml.safe_load(
        (patch_run / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    behavior_rows = _read_jsonl(behavior_run / "predictions.jsonl")
    behavior_metrics = _read_json(behavior_run / "metrics.json")
    behavior_manifest = _read_json(behavior_run / "data_manifest.json")
    behavior_selection = _read_json(behavior_run / "selection_manifest.json")
    behavior_environment = _read_json(behavior_run / "environment.json")

    protocol = cast(dict[str, Any], contract["protocol"])
    expected_protocol = {
        **protocol,
        "expected_router_hooks": expected_hooks,
        "split_manifest": contract["experiment_config"].get("split_manifest"),
    }
    observed_protocol = manifest.get("protocol", {})
    protocol_ok = all(observed_protocol.get(key) == value for key, value in expected_protocol.items())
    experiment = cast(dict[str, Any], contract["experiment_config"])
    resolved_manifest = manifest.get("resolved_config")
    science_fields = (
        "model",
        "data",
        "seed",
        "limit",
        "prompt_mode",
        "input_format",
        "boundary_strategy",
        "patch_mode",
        "alphas",
        "forced_reroute_control",
        "controls",
        "split_manifest",
        "split_partition",
        "expected_router_hooks",
        "sites",
        "layers",
    )
    resolved_ok = isinstance(resolved_manifest, Mapping) and all(
        resolved_manifest.get(field) == experiment.get(field) for field in science_fields
    )
    check(
        "frozen discovery protocol/config",
        protocol_ok and resolved_ok and resolved_file == resolved_manifest,
        {
            "protocol_match": protocol_ok,
            "resolved_match": resolved_ok,
            "resolved_file_match": resolved_file == resolved_manifest,
        },
    )

    selection_pairs = selection.get("pairs", [])
    pair_ids = [str(row.get("pair_id")) for row in selection_pairs]
    group_ids = [str(row.get("group_id")) for row in selection_pairs]
    frozen_groups = cast(list[str], contract["_discovery_group_ids"])
    selection_ok = (
        _selection_manifest_is_valid(selection)
        and len(selection_pairs) == expected_pairs
        and len(set(pair_ids)) == expected_pairs
        and len(set(group_ids)) == expected_pairs
        and group_ids == frozen_groups[:expected_pairs]
        and selection.get("selection_unit") == "clean_corrupt_pair"
        and selection.get("requested_pair_limit") == expected_pairs
        and selection.get("available_pair_count") == len(frozen_groups)
        and selection.get("selected_pair_count") == expected_pairs
        and selection.get("selected_example_count") == 2 * expected_pairs
        and selection.get("frozen_split", {}).get("partition") == "discovery"
        and selection.get("frozen_split", {}).get("manifest_hash")
        == contract.get("split_manifest_hash")
        and manifest.get("pair_ids") == pair_ids
        and manifest.get("group_ids") == group_ids
    )
    check(
        "frozen discovery ordered pair selection",
        selection_ok,
        {
            "pairs": len(selection_pairs),
            "unique_pairs": len(set(pair_ids)),
            "unique_groups": len(set(group_ids)),
            "expected_split_hash": contract.get("split_manifest_hash"),
        },
    )

    expected_rows = int(contract["expected_prediction_rows"])
    expected_bundle_rows = int(contract["expected_bundle_rows"])
    layers = [int(value) for value in protocol["layers"]]
    expected_site_keys = {
        f"layer_{layer}.router_logits" for layer in layers
    }
    bundle_site = f"bundle.router_logits[{expected_hooks}]"
    observed_descriptors = Counter(_row_descriptor(row) for row in rows)
    expected_descriptors = _expected_discovery_descriptors(
        pair_ids,
        alphas=[float(value) for value in protocol["alphas"]],
        bundle_site=bundle_site,
    )
    bundle_rows = [
        row
        for row in rows
        if row.get("metadata", {}).get("patch_mode") == "simultaneous_bundle"
    ]
    forced_rows = [
        row
        for row in rows
        if str(row.get("treatment", "")).startswith("forced_reroute")
        or row.get("metadata", {}).get("control") == "forced_reroute_smoke_only"
    ]
    dose_controls_ok = all(
        row.get("metadata", {}).get("control") == "dose_response"
        for row in rows
        if row.get("treatment") in {"clean_to_corrupt", "corrupt_to_clean"}
    )
    row_contract_ok = (
        len(rows) == expected_rows
        and len(bundle_rows) == expected_bundle_rows
        and observed_descriptors == expected_descriptors
        and not forced_rows
        and dose_controls_ok
    )
    check(
        "frozen discovery exact row contract",
        row_contract_ok,
        {
            "rows": len(rows),
            "expected_rows": expected_rows,
            "bundle_rows": len(bundle_rows),
            "expected_bundle_rows": expected_bundle_rows,
            "unexpected_or_missing_descriptors": sum(
                (observed_descriptors - expected_descriptors).values()
            )
            + sum((expected_descriptors - observed_descriptors).values()),
            "forced_rows": len(forced_rows),
            "dose_controls_ok": dose_controls_ok,
        },
    )

    hook_failures: list[dict[str, Any]] = []
    for row in bundle_rows:
        manipulation = row.get("metadata", {}).get("manipulation", {})
        aggregate_keys = set(manipulation.get("hook_fire_counts", {}))
        candidates = manipulation.get("candidate_diagnostics", {})
        candidate_ok = set(candidates) == {"CORRECT", "INCORRECT"}
        for candidate_diagnostics in candidates.values():
            candidate_ok = candidate_ok and (
                candidate_diagnostics.get("simultaneous") is True
                and candidate_diagnostics.get("registered_site_count") == expected_hooks
                and set(candidate_diagnostics.get("registered_sites", []))
                == expected_site_keys
                and set(candidate_diagnostics.get("hook_fire_counts", {}))
                == expected_site_keys
                and all(
                    int(value) > 0
                    for value in candidate_diagnostics.get(
                        "hook_fire_counts", {}
                    ).values()
                )
                and set(candidate_diagnostics.get("manipulation_by_site", {}))
                == expected_site_keys
            )
        if (
            manipulation.get("candidate_count") != 2
            or manipulation.get("registered_site_count") != expected_hooks
            or manipulation.get("all_registered_hooks_fired") is not True
            or aggregate_keys != expected_site_keys
            or any(
                int(value) <= 0
                for value in manipulation.get("hook_fire_counts", {}).values()
            )
            or not candidate_ok
        ):
            hook_failures.append(
                {"pair_id": row.get("pair_id"), "treatment": row.get("treatment")}
            )
    check(
        "frozen discovery exact simultaneous hooks",
        len(bundle_rows) == expected_bundle_rows and not hook_failures,
        {"expected_sites": sorted(expected_site_keys), "failures": hook_failures[:5]},
    )

    prompt_inputs = manifest.get("prompt_inputs", [])
    prompt_by_pair = {
        str(row.get("pair_id")): row for row in prompt_inputs if isinstance(row, Mapping)
    }
    input_failures = []
    for row in rows:
        pair_id = str(row.get("pair_id"))
        receiver = str(row.get("receiver_variant"))
        provenance = row.get("input_provenance", {})
        expected_provenance = prompt_by_pair.get(pair_id, {}).get(receiver)
        required_hashes = (
            provenance.get("raw_prompt_hash"),
            provenance.get("rendered_prompt_hash"),
            provenance.get("input_ids_hash"),
        )
        if (
            expected_provenance != provenance
            or provenance.get("prompt_mode") != protocol["prompt_mode"]
            or provenance.get("input_format") != protocol["input_format"]
            or not all(required_hashes)
        ):
            input_failures.append(
                {"pair_id": pair_id, "receiver": receiver, "treatment": row.get("treatment")}
            )
    check(
        "frozen discovery rendered input coverage",
        len(prompt_inputs) == expected_pairs
        and list(prompt_by_pair) == pair_ids
        and not input_failures,
        {
            "prompt_inputs": len(prompt_inputs),
            "failures": input_failures[:5],
        },
    )

    expected_behavior_pairs = int(contract["expected_behavior_pairs"])
    behavior_selection_pairs = behavior_selection.get("pairs", [])
    behavior_pair_ids = [str(row.get("pair_id")) for row in behavior_selection_pairs]
    behavior_group_ids = [str(row.get("group_id")) for row in behavior_selection_pairs]
    variants = Counter(
        (str(row.get("pair_id")), str(row.get("variant"))) for row in behavior_rows
    )
    expected_variants = Counter(
        (pair_id, variant)
        for pair_id in pair_ids[:expected_behavior_pairs]
        for variant in ("clean", "corrupt")
    )
    behavior_selection_ok = (
        _selection_manifest_is_valid(behavior_selection)
        and len(behavior_selection_pairs) == expected_behavior_pairs
        and behavior_pair_ids == pair_ids[:expected_behavior_pairs]
        and behavior_group_ids == group_ids[:expected_behavior_pairs]
        and behavior_selection.get("requested_pair_limit") == expected_behavior_pairs
        and behavior_selection.get("available_pair_count") == len(frozen_groups)
        and behavior_selection.get("selected_pair_count") == expected_behavior_pairs
        and behavior_selection.get("selected_example_count")
        == 2 * expected_behavior_pairs
        and len(behavior_rows) == 2 * expected_behavior_pairs
        and variants == expected_variants
        and behavior_manifest.get("row_count") == 2 * expected_behavior_pairs
    )
    check(
        "frozen discovery behavior 64-pair coverage",
        behavior_selection_ok,
        {
            "pairs": len(behavior_selection_pairs),
            "rows": len(behavior_rows),
            "expected_pairs": expected_behavior_pairs,
        },
    )

    recomputed_behavior = binary_verification_metrics(behavior_rows)
    recorded_auroc = behavior_metrics.get("auroc")
    recomputed_auroc = recomputed_behavior.get("auroc")
    behavior_metric_ok = (
        recomputed_behavior.get("n") == 2 * expected_behavior_pairs
        and recomputed_behavior.get("positives") == expected_behavior_pairs
        and recomputed_behavior.get("negatives") == expected_behavior_pairs
        and recorded_auroc is not None
        and recomputed_auroc is not None
        and abs(float(recorded_auroc) - float(recomputed_auroc)) <= 1e-12
        and float(recomputed_auroc) >= min_behavior_auroc
    )
    check(
        "frozen discovery recomputed behavior AUROC",
        behavior_metric_ok,
        {
            "recorded": recorded_auroc,
            "recomputed": recomputed_auroc,
            "minimum": min_behavior_auroc,
            "n": recomputed_behavior.get("n"),
            "positives": recomputed_behavior.get("positives"),
            "negatives": recomputed_behavior.get("negatives"),
        },
    )

    patch_git = patch_environment.get("git", {})
    behavior_git = behavior_environment.get("git", {})
    check(
        "frozen discovery same clean code/environment",
        patch_git.get("commit")
        and patch_git == behavior_git
        and patch_git.get("dirty") is False
        and patch_environment.get("model") == behavior_environment.get("model")
        and patch_environment.get("data") == behavior_environment.get("data")
        and patch_environment.get("files") == behavior_environment.get("files"),
        {
            "commit": patch_git.get("commit"),
            "dirty": patch_git.get("dirty"),
            "behavior_commit": behavior_git.get("commit"),
        },
    )
    patch_metrics = _read_json(patch_run / "metrics.json")
    check(
        "frozen discovery excludes no primary rows",
        patch_metrics.get("excluded_smoke_control_rows") == 0,
        {"excluded_smoke_control_rows": patch_metrics.get("excluded_smoke_control_rows")},
    )

    payload["passed"] = all(item["passed"] for item in checks)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256sums(run_root: Path, sums_path: Path) -> tuple[bool, dict[str, Any]]:
    failures: list[str] = []
    verified = 0
    seen: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed:{line[:80]}")
            continue
        relative = Path(relative_text)
        if (
            len(expected) != 64
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            failures.append(f"unsafe:{relative_text}")
            continue
        seen.add(relative_text)
        target = run_root / relative
        if not target.is_file() or _sha256_file(target) != expected:
            failures.append(relative_text)
            continue
        verified += 1
    excluded = {"SHA256SUMS", "SUCCESS", "FAILED"}
    actual_files = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.relative_to(run_root).as_posix() not in excluded
    }
    unlisted = sorted(actual_files - seen)
    nonexistent = sorted(seen - actual_files)
    critical = {
        "run.log",
        "run_manifest.json",
        "second_batch_review.json",
        "patch/predictions.jsonl",
        "behavior/predictions.jsonl",
    }
    missing_critical = sorted(critical - seen)
    return bool(
        verified
        and not failures
        and not unlisted
        and not nonexistent
        and not missing_critical
    ), {
        "verified_files": verified,
        "failures": failures[:10],
        "unlisted_files": unlisted[:10],
        "nonexistent_entries": nonexistent[:10],
        "missing_critical": missing_critical,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved.is_relative_to(right_resolved)
        or right_resolved.is_relative_to(left_resolved)
    )


def _verify_package_archive(
    package_path: Path,
    *,
    run_root: Path,
    run_id: str,
) -> tuple[bool, dict[str, Any]]:
    """Bind the external tgz byte-for-byte to the finalized smoke root."""

    excluded = {"SUCCESS", "FAILED"}
    expected_files = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.relative_to(run_root).as_posix() not in excluded
    }
    observed_files: set[str] = set()
    failures: list[str] = []
    seen_members: set[str] = set()
    top_levels: set[str] = set()
    try:
        with tarfile.open(package_path, "r:gz") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if (
                    member.name in seen_members
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or not member_path.parts
                ):
                    failures.append(f"unsafe-or-duplicate:{member.name}")
                    continue
                seen_members.add(member.name)
                top_levels.add(member_path.parts[0])
                if not (member.isdir() or member.isreg()):
                    failures.append(f"unsupported-member:{member.name}")
                    continue
                if member.isdir():
                    continue
                if member_path.parts[0] != run_id or len(member_path.parts) < 2:
                    failures.append(f"wrong-top-level:{member.name}")
                    continue
                relative = Path(*member_path.parts[1:]).as_posix()
                observed_files.add(relative)
                source = archive.extractfile(member)
                root_file = run_root / relative
                if source is None or not root_file.is_file():
                    failures.append(f"missing-root-file:{relative}")
                    continue
                archived_bytes = source.read()
                if hashlib.sha256(archived_bytes).hexdigest() != _sha256_file(
                    root_file
                ):
                    failures.append(f"content-mismatch:{relative}")
    except (OSError, tarfile.TarError) as exc:
        failures.append(f"archive-error:{exc}")
    missing = sorted(expected_files - observed_files)
    extra = sorted(observed_files - expected_files)
    return (
        not failures
        and top_levels == {run_id}
        and not missing
        and not extra,
        {
            "top_levels": sorted(top_levels),
            "expected_files": len(expected_files),
            "observed_files": len(observed_files),
            "missing": missing[:10],
            "extra": extra[:10],
            "failures": failures[:10],
        },
    )


def olmoe_discovery_prerequisite_review(
    *,
    smoke_review_path: Path,
    gate_config_path: Path,
    next_run_id: str,
    next_run_root: Path,
    next_package_path: Path,
    current_commit: str | None,
    current_dirty: bool | None,
    current_environment: Mapping[str, str | None],
) -> dict[str, Any]:
    """Verify that one intact same-Pod OLMoE smoke authorizes discovery."""

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    smoke_review_path = smoke_review_path.resolve()
    smoke_root = smoke_review_path.parent
    manifest_path = smoke_root / "run_manifest.json"
    sums_path = smoke_root / "SHA256SUMS"
    required_paths = [
        smoke_review_path,
        manifest_path,
        sums_path,
        smoke_root / "SUCCESS",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    check(
        "smoke prerequisite files",
        smoke_review_path.name == "second_batch_review.json"
        and not missing
        and not (smoke_root / "FAILED").exists(),
        {"missing": missing, "failed_marker": (smoke_root / "FAILED").exists()},
    )
    if missing:
        return {
            "schema_version": 1,
            "passed": False,
            "smoke_review": str(smoke_review_path),
            "checks": checks,
        }

    review = _read_json(smoke_review_path)
    manifest = _read_json(manifest_path)
    named_checks = review.get("checks", [])
    names = [str(item.get("name")) for item in named_checks]
    review_contract = review.get("contract", {})
    check(
        "machine smoke PASS",
        review.get("passed") is True
        and review.get("model_profile") == "olmoe"
        and review.get("run_role") == "smoke"
        and review.get("expected_pairs") == 8
        and review.get("expected_hooks") == 16
        and review.get("expected_behavior_pairs") == 64
        and review_contract.get("run_role") == "smoke"
        and review_contract.get("expected_pairs") == 8
        and review_contract.get("expected_behavior_pairs") == 64
        and review_contract.get("expected_hooks") == 16
        and len(names) == len(set(names))
        and OLMOE_SMOKE_REQUIRED_CHECKS <= set(names)
        and all(item.get("passed") is True for item in named_checks),
        {
            "passed": review.get("passed"),
            "model_profile": review.get("model_profile"),
            "run_role": review.get("run_role"),
            "expected_pairs": review.get("expected_pairs"),
            "expected_behavior_pairs": review.get("expected_behavior_pairs"),
            "expected_hooks": review.get("expected_hooks"),
            "missing_checks": sorted(OLMOE_SMOKE_REQUIRED_CHECKS - set(names)),
        },
    )

    gate_sha256 = _sha256_file(gate_config_path)
    experiment_path_value = review_contract.get("experiment_config_path")
    experiment_path = (
        None if not experiment_path_value else Path(str(experiment_path_value))
    )
    experiment_hash_ok = (
        experiment_path is not None
        and experiment_path.is_file()
        and review_contract.get("experiment_config_sha256")
        == _sha256_file(experiment_path)
    )
    check(
        "smoke gate/experiment contract hashes",
        review.get("gate_config", {}).get("sha256") == gate_sha256
        and review_contract.get("gate_config_sha256") == gate_sha256
        and experiment_hash_ok,
        {
            "gate_recorded": review.get("gate_config", {}).get("sha256"),
            "gate_contract": review_contract.get("gate_config_sha256"),
            "gate_current": gate_sha256,
            "experiment_path": experiment_path_value,
            "experiment_recorded": review_contract.get(
                "experiment_config_sha256"
            ),
            "experiment_current": (
                None
                if experiment_path is None or not experiment_path.is_file()
                else _sha256_file(experiment_path)
            ),
        },
    )
    sums_ok, sums_details = _verify_sha256sums(smoke_root, sums_path)
    check("smoke SHA256SUMS", sums_ok, sums_details)

    package_value = manifest.get("package_path")
    package_path = Path(str(package_value)).resolve() if package_value else None
    checksum_path = None if package_path is None else Path(f"{package_path}.sha256")
    package_ok = False
    package_recorded_hash = None
    archive_ok = False
    archive_details: dict[str, Any] = {}
    if package_path is not None and checksum_path is not None:
        if package_path.is_file() and checksum_path.is_file():
            checksum_line = checksum_path.read_text(encoding="utf-8").strip()
            pieces = checksum_line.split("  ", 1)
            if len(pieces) == 2:
                package_recorded_hash = pieces[0]
                package_ok = (
                    pieces[1] == package_path.name
                    and package_recorded_hash == _sha256_file(package_path)
                )
                if package_ok and manifest.get("run_id"):
                    archive_ok, archive_details = _verify_package_archive(
                        package_path,
                        run_root=smoke_root,
                        run_id=str(manifest["run_id"]),
                    )
    check(
        "smoke package hash",
        package_ok and archive_ok,
        {
            "package": None if package_path is None else str(package_path),
            "sha256": package_recorded_hash,
            "archive": archive_details,
        },
    )

    manifest_environment = manifest.get("environment", {})
    environment_keys = (
        "RUNPOD_POD_ID",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
    )
    environment_match = all(
        current_environment.get(key)
        and manifest_environment.get(key) == current_environment.get(key)
        for key in environment_keys
    )
    check(
        "same pod and Hugging Face caches",
        environment_match,
        {
            key: {
                "smoke": manifest_environment.get(key),
                "current": current_environment.get(key),
            }
            for key in environment_keys
        },
    )
    check(
        "same clean code commit",
        bool(current_commit)
        and current_dirty is False
        and manifest.get("code_commit") == current_commit
        and manifest.get("success") is True
        and manifest.get("exit_code") == 0
        and manifest.get("profile") == "olmoe_smoke",
        {
            "smoke_commit": manifest.get("code_commit"),
            "current_commit": current_commit,
            "current_dirty": current_dirty,
            "profile": manifest.get("profile"),
            "success": manifest.get("success"),
        },
    )

    smoke_run_id = manifest.get("run_id")
    manifest_root_value = manifest.get("run_root")
    smoke_identity_ok = (
        bool(smoke_run_id)
        and smoke_root.name == smoke_run_id
        and manifest_root_value
        and Path(str(manifest_root_value)).resolve() == smoke_root
        and package_path is not None
        and package_path.name == f"{smoke_run_id}.tgz"
        and not _paths_overlap(smoke_root, package_path)
    )
    check(
        "smoke run identity",
        bool(smoke_identity_ok),
        {
            "run_id": smoke_run_id,
            "review_root": str(smoke_root),
            "manifest_root": manifest_root_value,
            "package": None if package_path is None else str(package_path),
        },
    )
    next_paths_absent = all(
        not path.exists()
        for path in (
            next_run_root,
            next_package_path,
            Path(f"{next_package_path}.sha256"),
            Path(f"{next_package_path}.partial"),
            Path(f"{next_package_path}.sha256.partial"),
        )
    )
    next_identity_ok = (
        bool(next_run_id)
        and smoke_run_id
        and next_run_id != smoke_run_id
        and next_run_root.name == next_run_id
        and next_package_path.name == f"{next_run_id}.tgz"
        and next_run_root.is_absolute()
        and next_package_path.is_absolute()
        and not _paths_overlap(next_run_root, smoke_root)
        and package_path is not None
        and not _paths_overlap(next_run_root, package_path)
        and not _paths_overlap(next_package_path, smoke_root)
        and not _paths_overlap(next_package_path, package_path)
        and not _paths_overlap(next_run_root, next_package_path)
        and next_paths_absent
    )
    check(
        "new discovery run identity",
        bool(next_identity_ok),
        {
            "smoke_run_id": smoke_run_id,
            "next_run_id": next_run_id,
            "next_run_root": str(next_run_root),
            "next_package": str(next_package_path),
            "paths_absent": next_paths_absent,
        },
    )
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "passed": passed,
        "smoke": {
            "review_path": str(smoke_review_path),
            "run_root": str(smoke_root),
            "run_id": smoke_run_id,
            "package_path": None if package_path is None else str(package_path),
            "package_sha256": package_recorded_hash,
            "code_commit": manifest.get("code_commit"),
            "pod_id": manifest_environment.get("RUNPOD_POD_ID"),
        },
        "discovery": {
            "run_id": next_run_id,
            "run_root": str(next_run_root),
            "package_path": str(next_package_path),
        },
        "gate_config": {
            "path": str(gate_config_path),
            "sha256": gate_sha256,
        },
        "checks": checks,
    }
