"""Machine review for the narrow OLMoE residual-discovery workflow.

The module intentionally has no dependency on any validation artifact.  Its
inputs are the committed, discovery-only CPU-analysis pair order and one of the
three versioned residual run roots.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from routeverify.data.schema import stable_hash
from routeverify.instrumentation.storage import atomic_write_json

PROFILE_SMOKE = "residual_smoke"
PROFILE_SWEEP = "residual_sweep"
PROFILE_CONTROLS = "residual_controls"
PROFILE_SAME_VARIANT = "residual_same_variant"
SWEEP_PROFILES = frozenset({PROFILE_SMOKE, PROFILE_SWEEP})
TREATMENT_DIRECTIONS = ("corrupt_to_clean", "clean_to_corrupt")
SAME_VARIANT_CONFIG_ID = "olmoe_same_variant_donor_v1_20260808"
SAME_VARIANT_TREATMENTS = (
    "random_matched_clean_to_clean",
    "random_matched_corrupt_to_corrupt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(cast(dict[str, Any], value))
    return rows


def _margin(row: Mapping[str, Any]) -> float:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("row is missing candidate scores")
    try:
        correct = cast(Mapping[str, Any], scores["CORRECT"])
        incorrect = cast(Mapping[str, Any], scores["INCORRECT"])
        derived = float(incorrect["mean_logprob"]) - float(correct["mean_logprob"])
        reported = float(row["margin_incorrect_minus_correct_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("row has incomplete candidate scores/margin") from exc
    if not np.isfinite(derived) or not np.isfinite(reported):
        raise ValueError("row candidate-derived/reported margin is non-finite")
    if abs(derived - reported) > 1e-12:
        raise ValueError("reported mean-logprob margin differs from candidate scores")
    return derived


def _score_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_scores = left.get("scores")
    right_scores = right.get("scores")
    if not isinstance(left_scores, Mapping) or not isinstance(right_scores, Mapping):
        return float("inf")
    deltas: list[float] = []
    for candidate in ("CORRECT", "INCORRECT"):
        left_candidate = left_scores.get(candidate)
        right_candidate = right_scores.get(candidate)
        if not isinstance(left_candidate, Mapping) or not isinstance(right_candidate, Mapping):
            return float("inf")
        for field in ("mean_logprob", "total_logprob"):
            try:
                delta = abs(float(left_candidate[field]) - float(right_candidate[field]))
            except (KeyError, TypeError, ValueError):
                return float("inf")
            if not np.isfinite(delta):
                return float("inf")
            deltas.append(delta)
    return max(deltas, default=float("inf"))


def _site_layer(row: Mapping[str, Any]) -> int | None:
    site = row.get("site")
    if site is None:
        return None
    text = str(site)
    prefix = "layer_"
    suffix = ".post_moe_residual"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return None
    try:
        return int(text[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, int | None]:
    return (
        str(row.get("pair_id")),
        str(row.get("receiver_variant")),
        str(row.get("treatment")),
        _site_layer(row),
    )


def _row_signature(
    row: Mapping[str, Any],
) -> tuple[str, str, str | None, str, int | None]:
    donor = row.get("donor_variant")
    return (
        str(row.get("pair_id")),
        str(row.get("receiver_variant")),
        None if donor is None else str(donor),
        str(row.get("treatment")),
        _site_layer(row),
    )


def _exact_row_signature_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    layers: Sequence[int],
    controls_layer: int | None,
) -> dict[str, Any]:
    expected: Counter[tuple[str, str, str | None, str, int | None]] = Counter()
    for pair_id in pair_ids:
        expected.update(
            {
                (pair_id, "clean", None, "unpatched", None): 1,
                (pair_id, "corrupt", None, "unpatched", None): 1,
                (pair_id, "clean", None, "unpatched_rerun", None): 1,
                (pair_id, "corrupt", None, "unpatched_rerun", None): 1,
            }
        )
        for layer in layers:
            expected.update(
                {
                    (pair_id, "clean", "clean", "self_patch", layer): 1,
                    (pair_id, "corrupt", "corrupt", "self_patch", layer): 1,
                    (pair_id, "clean", "corrupt", "corrupt_to_clean", layer): 1,
                    (pair_id, "corrupt", "clean", "clean_to_corrupt", layer): 1,
                }
            )
        if controls_layer is not None:
            expected.update(
                {
                    (
                        pair_id,
                        "clean",
                        "random",
                        "norm_random_delta",
                        controls_layer,
                    ): 1,
                    (
                        pair_id,
                        "corrupt",
                        "random",
                        "norm_random_delta",
                        controls_layer,
                    ): 1,
                    (
                        pair_id,
                        "clean",
                        "matched_corrupt",
                        "random_matched_corrupt_to_clean",
                        controls_layer,
                    ): 1,
                    (
                        pair_id,
                        "corrupt",
                        "matched_clean",
                        "random_matched_clean_to_corrupt",
                        controls_layer,
                    ): 1,
                }
            )
    observed = Counter(_row_signature(row) for row in rows)
    missing = list((expected - observed).items())
    unexpected = list((observed - expected).items())
    return {
        "passed": observed == expected,
        "expected_rows": sum(expected.values()),
        "observed_rows": len(rows),
        "missing": [
            {"signature": list(signature), "count": count} for signature, count in missing[:10]
        ],
        "unexpected": [
            {"signature": list(signature), "count": count} for signature, count in unexpected[:10]
        ],
    }


def _same_variant_row_signature_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    selected_layer: int,
) -> dict[str, Any]:
    expected: Counter[tuple[str, str, str | None, str, int | None]] = Counter()
    for pair_id in pair_ids:
        expected.update(
            {
                (pair_id, "clean", None, "unpatched", None): 1,
                (pair_id, "corrupt", None, "unpatched", None): 1,
                (pair_id, "clean", None, "unpatched_rerun", None): 1,
                (pair_id, "corrupt", None, "unpatched_rerun", None): 1,
                (pair_id, "clean", "clean", "self_patch", selected_layer): 1,
                (pair_id, "corrupt", "corrupt", "self_patch", selected_layer): 1,
                (
                    pair_id,
                    "clean",
                    "matched_corrupt",
                    "random_matched_corrupt_to_clean",
                    selected_layer,
                ): 1,
                (
                    pair_id,
                    "corrupt",
                    "matched_clean",
                    "random_matched_clean_to_corrupt",
                    selected_layer,
                ): 1,
                (
                    pair_id,
                    "clean",
                    "matched_clean",
                    "random_matched_clean_to_clean",
                    selected_layer,
                ): 1,
                (
                    pair_id,
                    "corrupt",
                    "matched_corrupt",
                    "random_matched_corrupt_to_corrupt",
                    selected_layer,
                ): 1,
            }
        )
    observed = Counter(_row_signature(row) for row in rows)
    missing = list((expected - observed).items())
    unexpected = list((observed - expected).items())
    return {
        "passed": observed == expected,
        "expected_rows": sum(expected.values()),
        "observed_rows": len(rows),
        "missing": [
            {"signature": list(signature), "count": count} for signature, count in missing[:10]
        ],
        "unexpected": [
            {"signature": list(signature), "count": count}
            for signature, count in unexpected[:10]
        ],
    }


def _ordered_pair_ids(selection: Mapping[str, Any]) -> list[str]:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list):
        return []
    return [
        str(row.get("pair_id")) for row in pairs if isinstance(row, Mapping) and row.get("pair_id")
    ]


def _pair_order_sha256(pair_ids: Sequence[str]) -> str:
    payload = "".join(f"{pair_id}\n" for pair_id in pair_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _selection_is_valid(
    selection: Mapping[str, Any],
    *,
    expected_pairs: int,
    expected_full_order_sha256: str,
) -> bool:
    payload = dict(selection)
    recorded_hash = payload.pop("manifest_hash", None)
    pair_ids = _ordered_pair_ids(selection)
    return (
        recorded_hash == stable_hash(payload, length=64)
        and len(pair_ids) == expected_pairs
        and len(set(pair_ids)) == expected_pairs
        and selection.get("selected_pair_count") == expected_pairs
        and selection.get("source_pair_count") == 290
        and selection.get("source_pair_order_sha256") == expected_full_order_sha256
        and selection.get("selected_pair_order_sha256") == _pair_order_sha256(pair_ids)
        and (expected_pairs != 290 or _pair_order_sha256(pair_ids) == expected_full_order_sha256)
        and selection.get("selection_rule") == "prefix_of_committed_discovery_cpu_pair_order"
    )


def _expected_matched_edges(
    selection: Mapping[str, Any], pair_ids: Sequence[str], *, seed: int
) -> list[dict[str, Any]] | None:
    pairs = selection.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(pair_ids):
        return None
    metadata: dict[str, tuple[str | None, int]] = {}
    for expected_pair_id, row in zip(pair_ids, pairs, strict=True):
        if not isinstance(row, Mapping) or str(row.get("pair_id")) != expected_pair_id:
            return None
        error_type = row.get("corrupt_error_type")
        first_error = row.get("corrupt_first_error")
        if (
            (error_type is not None and not isinstance(error_type, str))
            or not isinstance(first_error, int)
            or isinstance(first_error, bool)
            or first_error < 0
        ):
            return None
        metadata[expected_pair_id] = (error_type, first_error)
    edges: list[dict[str, Any]] = []
    for target_pair_id in pair_ids:
        target_type, target_step = metadata[target_pair_id]

        def rank(
            source_pair_id: str,
            *,
            target_pair_id: str = target_pair_id,
            target_type: str | None = target_type,
            target_step: int = target_step,
        ) -> tuple[int, int, str]:
            source_type, source_step = metadata[source_pair_id]
            tie = stable_hash(
                {
                    "domain": "olmoe_residual_discovery_v2_matched_donor",
                    "seed": seed,
                    "target_pair_id": target_pair_id,
                    "source_pair_id": source_pair_id,
                },
                length=64,
            )
            return int(source_type != target_type), abs(target_step - source_step), tie

        candidates = [pair_id for pair_id in pair_ids if pair_id != target_pair_id]
        if not candidates:
            return None
        source_pair_id = min(candidates, key=rank)
        source_type, source_step = metadata[source_pair_id]
        edges.append(
            {
                "target_pair_id": target_pair_id,
                "source_pair_id": source_pair_id,
                "target_error_type": target_type,
                "source_error_type": source_type,
                "same_error_type": target_type == source_type,
                "corrupt_first_error_step_distance": abs(target_step - source_step),
            }
        )
    return edges


def _pair_registry_matches_selection(registry: Mapping[str, Any], pair_ids: Sequence[str]) -> bool:
    try:
        pair_effects = Path(str(registry["pair_effects_path"]))
        analysis_path = Path(str(registry["analysis_path"]))
        if (
            not pair_effects.is_file()
            or not analysis_path.is_file()
            or sha256_file(pair_effects) != registry["pair_effects_sha256"]
            or sha256_file(analysis_path) != registry["analysis_sha256"]
        ):
            return False
        registered: list[str] = []
        seen: set[str] = set()
        for row in _read_jsonl(pair_effects):
            pair_id = str(row.get("pair_id"))
            if not pair_id or pair_id == "None":
                return False
            if pair_id not in seen:
                seen.add(pair_id)
                registered.append(pair_id)
        analysis = _read_json(analysis_path)
        source = analysis.get("source")
        return bool(
            isinstance(source, Mapping)
            and len(registered) == registry["expected_pair_count"]
            and _pair_order_sha256(registered) == registry["ordered_pair_ids_sha256"]
            and list(pair_ids) == registered[: len(pair_ids)]
            and source.get("run_id") == registry["reference_discovery_run_id"]
            and source.get("predictions_sha256")
            == registry["referenced_discovery_predictions_sha256"]
            and source.get("data_manifest_sha256")
            == registry["referenced_discovery_data_manifest_sha256"]
            and source.get("pair_order_sha256") == registry["ordered_pair_ids_sha256"]
            and source.get("num_pairs") == registry["expected_pair_count"]
            and source.get("source_hash_verification", {}).get("status") == "passed"
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _runtime_environment_matches(snapshot: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    runtime = snapshot.get("runtime_env")
    shared_keys = ("RUNPOD_POD_ID", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
    return bool(
        isinstance(runtime, Mapping)
        and snapshot.get("residual_v2_runtime") == expected
        and all(expected.get(key) and runtime.get(key) == expected.get(key) for key in shared_keys)
    )


def _shared_runtime_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    shared_keys = ("RUNPOD_POD_ID", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
    return all(left.get(key) and left.get(key) == right.get(key) for key in shared_keys)


def _current_linkage_ok(
    linkage: Mapping[str, Any],
    *,
    run_root: Path,
    profile: str,
    protocol_id: str,
    protocol_sha256: str,
    code_commit: str,
    root_environment: Mapping[str, Any],
    patch_environment: Mapping[str, Any],
) -> bool:
    environment = linkage.get("environment")
    if not isinstance(environment, Mapping):
        return False
    required_environment = {
        "RUNPOD_POD_ID",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "RESIDUAL_V2_RUN_ID",
        "RESIDUAL_V2_RUN_ROOT",
        "RESIDUAL_V2_PACKAGE_PATH",
    }
    try:
        linked_root = Path(str(linkage["run_root"]))
        package = Path(str(linkage["package_path"]))
        run_id = str(linkage["run_id"])
        return bool(
            linkage.get("schema_version") == 2
            and linkage.get("profile") == profile
            and linkage.get("protocol_id") == protocol_id
            and linkage.get("protocol_sha256") == protocol_sha256
            and linkage.get("code_commit") == code_commit
            and linkage.get("validation_accessed") is False
            and set(environment) == required_environment
            and all(environment.get(key) for key in required_environment)
            and run_id == run_root.name
            and environment.get("RESIDUAL_V2_RUN_ID") == run_id
            and Path(str(environment.get("RESIDUAL_V2_RUN_ROOT"))).resolve() == run_root.resolve()
            and linked_root.is_absolute()
            and linked_root.resolve() == run_root.resolve()
            and package.is_absolute()
            and package.name == f"{run_id}.tgz"
            and Path(str(environment.get("RESIDUAL_V2_PACKAGE_PATH"))).resolve()
            == package.resolve()
            and run_root.resolve() not in package.resolve().parents
            and _runtime_environment_matches(root_environment, environment)
            and _runtime_environment_matches(patch_environment, environment)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _linked_prerequisite_files_ok(
    prerequisite: Mapping[str, Any],
    *,
    require_winner: bool,
    protocol_id: str,
    code_commit: str,
    environment: Mapping[str, Any],
    ordered_pair_ids_sha256: str,
) -> bool:
    try:
        review_path = Path(str(prerequisite["review_path"]))
        run_root = Path(str(prerequisite["run_root"]))
        run_manifest = run_root / "run_manifest.json"
        sha256sums = run_root / "SHA256SUMS"
        root_environment_path = run_root / "environment_manifest.json"
        patch_environment_path = run_root / "patch/environment.json"
        package = Path(str(prerequisite["package_path"]))
        sidecar = Path(str(prerequisite["package_sidecar_path"]))
        if (
            not all(
                path.is_file()
                for path in (
                    review_path,
                    run_manifest,
                    sha256sums,
                    root_environment_path,
                    patch_environment_path,
                    package,
                    sidecar,
                    run_root / "SUCCESS",
                )
            )
            or (run_root / "FAILED").exists()
        ):
            return False
        review = _read_json(review_path)
        manifest = _read_json(run_manifest)
        root_environment = _read_json(root_environment_path)
        patch_environment = _read_json(patch_environment_path)
        root_git = root_environment.get("git")
        patch_git = patch_environment.get("git")
        sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
        prior_profile = str(prerequisite.get("profile"))
        run_id = str(prerequisite.get("run_id"))
        prior_environment = prerequisite.get("environment")
        ok = (
            sha256_file(review_path) == prerequisite.get("review_sha256")
            and sha256_file(run_manifest) == prerequisite.get("run_manifest_sha256")
            and Path(str(prerequisite.get("sha256sums_path"))).resolve() == sha256sums.resolve()
            and sha256_file(sha256sums) == prerequisite.get("sha256sums_sha256")
            and sha256_file(package) == prerequisite.get("package_sha256")
            and sha256_file(sidecar) == prerequisite.get("package_sidecar_sha256")
            and len(sidecar_parts) == 2
            and sidecar_parts[0] == prerequisite.get("package_sha256")
            and sidecar_parts[1] == package.name
            and isinstance(prior_environment, Mapping)
            and _shared_runtime_matches(prior_environment, environment)
            and review.get("integrity_passed") is True
            and review.get("completed") is True
            and review.get("profile") == prior_profile
            and review.get("protocol_id") == protocol_id
            and review.get("run_id") == run_id
            and manifest.get("success") is True
            and manifest.get("exit_code") == 0
            and manifest.get("profile") == prior_profile
            and manifest.get("run_id") == run_id
            and Path(str(manifest.get("run_root"))).resolve() == run_root.resolve()
            and Path(str(manifest.get("package_path"))).resolve() == package.resolve()
            and manifest.get("code_commit") == code_commit
            and manifest.get("environment") == prior_environment
            and manifest.get("review") == review
            and run_root.name == run_id
            and package.name == f"{run_id}.tgz"
            and prior_environment.get("RESIDUAL_V2_RUN_ID") == run_id
            and Path(str(prior_environment.get("RESIDUAL_V2_RUN_ROOT"))).resolve()
            == run_root.resolve()
            and Path(str(prior_environment.get("RESIDUAL_V2_PACKAGE_PATH"))).resolve()
            == package.resolve()
            and isinstance(root_git, Mapping)
            and isinstance(patch_git, Mapping)
            and root_git.get("commit") == code_commit
            and root_git.get("dirty") is False
            and patch_git == root_git
            and _runtime_environment_matches(root_environment, prior_environment)
            and _runtime_environment_matches(patch_environment, prior_environment)
            and root_environment.get("validation_accessed") is False
            and patch_environment.get("validation_accessed") is False
        )
        if require_winner:
            winner = Path(str(prerequisite["winner_manifest_path"]))
            winner_payload = _read_json(winner) if winner.is_file() else {}
            ok = bool(
                ok
                and winner.is_file()
                and sha256_file(winner) == prerequisite.get("winner_manifest_sha256")
                and winner_payload.get("protocol_id") == protocol_id
                and winner_payload.get("profile") == PROFILE_SWEEP
                and winner_payload.get("selection_role") == "discovery_only"
                and winner_payload.get("sweep_run_id") == run_id
                and winner_payload.get("ordered_pair_ids_sha256") == ordered_pair_ids_sha256
                and winner_payload.get("selected_layer") == prerequisite.get("selected_layer")
            )
        return bool(ok)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _hook_diagnostics_ok(row: Mapping[str, Any], layer: int) -> bool:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    diagnostics = metadata.get("patch_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    site = f"layer_{layer}.post_moe_residual"
    candidates = diagnostics.get("candidates")
    return (
        diagnostics.get("candidate_count") == 2
        and diagnostics.get("registered_site") == site
        and diagnostics.get("all_hooks_fired") is True
        and isinstance(candidates, Mapping)
        and set(candidates) == {"CORRECT", "INCORRECT"}
        and all(
            isinstance(value, Mapping)
            and value.get("registered_site_count") == 1
            and value.get("registered_sites") == [site]
            and value.get("all_registered_hooks_fired") is True
            and int(value.get("hook_fire_counts", {}).get(site, 0)) > 0
            for value in candidates.values()
        )
    )


def _generic_row_review(
    rows: Sequence[Mapping[str, Any]], pair_ids: Sequence[str]
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    allowed_pairs = set(pair_ids)
    for index, row in enumerate(rows):
        scores = row.get("scores")
        scores_ok = isinstance(scores, Mapping) and set(scores) == {
            "CORRECT",
            "INCORRECT",
        }
        if scores_ok:
            score_mapping = cast(Mapping[str, Any], scores)
            for candidate in ("CORRECT", "INCORRECT"):
                value = score_mapping.get(candidate)
                try:
                    candidate_ok = bool(
                        isinstance(value, Mapping)
                        and value.get("candidate") == candidate
                        and isinstance(value.get("token_count"), int)
                        and not isinstance(value.get("token_count"), bool)
                        and int(value["token_count"]) > 0
                        and all(
                            np.isfinite(float(value[field]))
                            for field in ("mean_logprob", "total_logprob")
                        )
                        and abs(
                            float(value["mean_logprob"])
                            - float(value["total_logprob"]) / int(value["token_count"])
                        )
                        <= 1e-12
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_ok = False
                scores_ok = bool(scores_ok and candidate_ok)
            if scores_ok:
                try:
                    correct = cast(Mapping[str, Any], score_mapping["CORRECT"])
                    incorrect = cast(Mapping[str, Any], score_mapping["INCORRECT"])
                    expected_mean = float(incorrect["mean_logprob"]) - float(
                        correct["mean_logprob"]
                    )
                    expected_total = float(incorrect["total_logprob"]) - float(
                        correct["total_logprob"]
                    )
                    scores_ok = bool(
                        abs(float(row["margin_incorrect_minus_correct_mean"]) - expected_mean)
                        <= 1e-12
                        and abs(float(row["margin_incorrect_minus_correct_total"]) - expected_total)
                        <= 1e-12
                        and row.get("primary_score") == "mean_logprob"
                    )
                except (KeyError, TypeError, ValueError):
                    scores_ok = False
        boundary = row.get("boundary_step")
        token = row.get("token_position")
        provenance = row.get("input_provenance")
        provenance_ok = isinstance(provenance, Mapping) and all(
            provenance.get(field)
            for field in ("raw_prompt_hash", "rendered_prompt_hash", "input_ids_hash")
        )
        row_ok = (
            str(row.get("pair_id")) in allowed_pairs
            and row.get("receiver_variant") in {"clean", "corrupt"}
            and isinstance(boundary, int)
            and not isinstance(boundary, bool)
            and boundary >= 0
            and isinstance(token, int)
            and not isinstance(token, bool)
            and token >= 0
            and scores_ok
            and provenance_ok
        )
        if not row_ok:
            failures.append(
                {
                    "row_index": index,
                    "pair_id": row.get("pair_id"),
                    "treatment": row.get("treatment"),
                }
            )
    return {
        "passed": not failures,
        "rows": len(rows),
        "failures": failures[:10],
    }


def _norm_random_metadata_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    selected_layer: int,
    seed: int,
) -> dict[str, Any]:
    """Verify the frozen per-candidate float64 norm-random construction record."""

    domain = "olmoe_residual_discovery_v2_norm_random"
    expected_units = {
        (pair_id, receiver) for pair_id in pair_ids for receiver in ("clean", "corrupt")
    }
    observed: Counter[tuple[str, str]] = Counter()
    failures: list[dict[str, Any]] = []
    random_rows = [row for row in rows if row.get("treatment") == "norm_random_delta"]
    for row in random_rows:
        pair_id = str(row.get("pair_id"))
        receiver = str(row.get("receiver_variant"))
        observed[(pair_id, receiver)] += 1
        metadata = row.get("metadata")
        randomization = metadata.get("randomization") if isinstance(metadata, Mapping) else None
        row_ok = bool(
            (pair_id, receiver) in expected_units
            and _site_layer(row) == selected_layer
            and isinstance(metadata, Mapping)
            and metadata.get("control") == "winner_only_norm_random"
            and isinstance(randomization, Mapping)
            and set(randomization) == {"CORRECT", "INCORRECT"}
        )
        if row_ok:
            for candidate in ("CORRECT", "INCORRECT"):
                record = cast(Mapping[str, Any], randomization).get(candidate)
                unit_key = {
                    "pair_id": pair_id,
                    "receiver": receiver,
                    "candidate": candidate,
                    "layer": selected_layer,
                }
                seed_payload = {
                    "domain": domain,
                    "seed": seed,
                    "unit_key": unit_key,
                }
                expected_digest = hashlib.sha256(
                    json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                try:
                    target = float(cast(Mapping[str, Any], record)["target_delta_l2_float64"])
                    constructed = float(
                        cast(Mapping[str, Any], record)["constructed_delta_l2_float64"]
                    )
                    relative = float(
                        cast(Mapping[str, Any], record)["construction_relative_l2_error"]
                    )
                    recomputed_relative = (
                        0.0 if target == 0.0 else abs(constructed - target) / target
                    )
                    candidate_ok = bool(
                        isinstance(record, Mapping)
                        and record.get("domain") == domain
                        and record.get("seed") == seed
                        and record.get("unit_key") == unit_key
                        and record.get("unit_key_sha256") == expected_digest
                        and record.get("construction_dtype") == "float64"
                        and record.get("execution_cast")
                        == "deferred_once_to_model_target_dtype_by_patch_adapter"
                        and record.get("redraw") is False
                        and np.isfinite(target)
                        and np.isfinite(constructed)
                        and np.isfinite(relative)
                        and target >= 0.0
                        and constructed >= 0.0
                        and relative >= 0.0
                        and relative <= 1.0e-12
                        and abs(relative - recomputed_relative) <= 1.0e-15
                        and (target != 0.0 or constructed == 0.0)
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_ok = False
                row_ok = bool(row_ok and candidate_ok)
        if not row_ok:
            failures.append({"pair_id": pair_id, "receiver_variant": receiver})
    exact_units = set(observed) == expected_units and all(
        observed[unit] == 1 for unit in expected_units
    )
    return {
        "passed": exact_units and not failures,
        "expected_rows": len(expected_units),
        "observed_rows": len(random_rows),
        "exact_unit_coverage_passed": exact_units,
        "failures": failures[:10],
    }


def _bootstrap_means(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    ci_alpha: float,
    chunk_size: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-pair ordinary percentile intervals for metric-by-pair values."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("bootstrap requires metric-by-pair values with at least two pairs")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("bootstrap values must all be finite")
    if resamples < 2 or not 0.0 < ci_alpha < 1.0:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    replicates = np.empty((matrix.shape[0], resamples), dtype=np.float64)
    pair_count = matrix.shape[1]
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, pair_count, size=(stop - start, pair_count))
        replicates[:, start:stop] = matrix[:, indices].mean(axis=2)
    quantiles = np.quantile(
        replicates,
        [ci_alpha / 2.0, 1.0 - ci_alpha / 2.0],
        axis=1,
        method="linear",
    )
    return quantiles[0], quantiles[1]


def _baseline_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    baselines: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("treatment") != "unpatched":
            continue
        key = (str(row.get("pair_id")), str(row.get("receiver_variant")))
        if key in baselines:
            raise ValueError(f"duplicate baseline row: {key}")
        baselines[key] = row
    return baselines


def _effect(
    row: Mapping[str, Any],
    baselines: Mapping[tuple[str, str], Mapping[str, Any]],
) -> float:
    key = (str(row.get("pair_id")), str(row.get("receiver_variant")))
    baseline = baselines.get(key)
    if baseline is None:
        raise ValueError(f"missing receiver baseline for {key}")
    return _margin(row) - _margin(baseline)


def _identity_and_alignment_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    expected_self_layers: Sequence[int],
    tolerance: float,
) -> dict[str, Any]:
    baselines = _baseline_map(rows)
    expected_baselines = 2 * len(pair_ids)
    reruns = [row for row in rows if row.get("treatment") == "unpatched_rerun"]
    self_rows = [row for row in rows if row.get("treatment") == "self_patch"]
    max_rerun = 0.0
    max_self = 0.0
    alignment_failures: list[dict[str, Any]] = []
    for row in rows:
        receiver = str(row.get("receiver_variant"))
        pair_id = str(row.get("pair_id"))
        baseline = baselines.get((pair_id, receiver))
        if baseline is None:
            alignment_failures.append({"pair_id": pair_id, "receiver": receiver})
            continue
        if (
            row.get("boundary_step") != baseline.get("boundary_step")
            or row.get("token_position") != baseline.get("token_position")
            or row.get("input_provenance") != baseline.get("input_provenance")
        ):
            alignment_failures.append(
                {
                    "pair_id": pair_id,
                    "receiver": receiver,
                    "treatment": row.get("treatment"),
                    "site": row.get("site"),
                }
            )
    for row in reruns:
        baseline = baselines.get((str(row.get("pair_id")), str(row.get("receiver_variant"))))
        max_rerun = max(
            max_rerun,
            float("inf") if baseline is None else _score_delta(row, baseline),
        )
    for row in self_rows:
        baseline = baselines.get((str(row.get("pair_id")), str(row.get("receiver_variant"))))
        max_self = max(
            max_self,
            float("inf") if baseline is None else _score_delta(row, baseline),
        )
    expected_self = 2 * len(pair_ids) * len(expected_self_layers)
    expected_self_keys = {
        (pair_id, layer, receiver)
        for pair_id in pair_ids
        for layer in expected_self_layers
        for receiver in ("clean", "corrupt")
    }
    observed_self = Counter(
        (
            str(row.get("pair_id")),
            _site_layer(row),
            str(row.get("receiver_variant")),
        )
        for row in self_rows
    )
    exact_self = set(observed_self) == expected_self_keys and all(
        observed_self[key] == 1 for key in expected_self_keys
    )
    return {
        "passed": (
            len(baselines) == expected_baselines
            and len(reruns) == expected_baselines
            and exact_self
            and max_rerun <= tolerance
            and max_self <= tolerance
            and not alignment_failures
        ),
        "baseline_rows": len(baselines),
        "rerun_rows": len(reruns),
        "self_rows": len(self_rows),
        "expected_self_rows": expected_self,
        "max_unpatched_rerun_candidate_score_delta": max_rerun,
        "max_self_patch_candidate_score_delta": max_self,
        "tolerance": tolerance,
        "boundary_token_failures": alignment_failures[:10],
    }


def _sweep_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    pair_ids: Sequence[str],
    layers: Sequence[int],
    eligible_layers: Sequence[int],
    tolerance: float,
    seed: int,
    resamples: int,
    ci_alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    eligible_set = set(eligible_layers)
    if not eligible_set or not eligible_set.issubset(layer_index):
        raise ValueError("winner-eligible layers must be a non-empty subset of measured layers")
    direct = np.full((len(layers), 2, len(pair_ids)), np.nan, dtype=np.float64)
    direct_counts: Counter[tuple[int, str]] = Counter()
    hook_failures: list[dict[str, Any]] = []
    baselines = _baseline_map(rows)
    for row in rows:
        treatment = str(row.get("treatment"))
        layer = _site_layer(row)
        if treatment not in {"unpatched", "unpatched_rerun"} and (
            layer is None or not _hook_diagnostics_ok(row, layer)
        ):
            hook_failures.append(
                {
                    "pair_id": row.get("pair_id"),
                    "treatment": treatment,
                    "layer": layer,
                }
            )
        if treatment not in TREATMENT_DIRECTIONS:
            continue
        pair_id = str(row.get("pair_id"))
        direction_index = 0 if treatment == "corrupt_to_clean" else 1
        if layer not in layer_index or pair_id not in pair_index:
            continue
        direct_counts[(layer, treatment)] += 1
        direct[layer_index[layer], direction_index, pair_index[pair_id]] = _effect(row, baselines)
    complete_direct = np.all(np.isfinite(direct)) and all(
        direct_counts[(layer, treatment)] == len(pair_ids)
        for layer in layers
        for treatment in TREATMENT_DIRECTIONS
    )
    symmetric = 0.5 * (direct[:, 0, :] - direct[:, 1, :])
    metrics = np.concatenate((direct[:, 0, :], direct[:, 1, :], symmetric), axis=0)
    lower, upper = _bootstrap_means(
        metrics,
        seed=seed,
        resamples=resamples,
        ci_alpha=ci_alpha,
    )
    layer_table: list[dict[str, Any]] = []
    count = len(layers)
    for index, layer in enumerate(layers):
        self_by_receiver: dict[str, list[float]] = {"clean": [], "corrupt": []}
        for row in rows:
            if row.get("treatment") != "self_patch" or _site_layer(row) != layer:
                continue
            receiver = str(row.get("receiver_variant"))
            baseline = baselines.get((str(row.get("pair_id")), receiver))
            if receiver in self_by_receiver and baseline is not None:
                self_by_receiver[receiver].append(_score_delta(row, baseline))
        clean_self_max = max(self_by_receiver["clean"], default=float("inf"))
        corrupt_self_max = max(self_by_receiver["corrupt"], default=float("inf"))
        layer_identity_passed = (
            len(self_by_receiver["clean"]) == len(pair_ids)
            and len(self_by_receiver["corrupt"]) == len(pair_ids)
            and max(clean_self_max, corrupt_self_max) <= tolerance
        )
        layer_table.append(
            {
                "layer_index": layer,
                "winner_eligible": layer in eligible_set,
                "pair_count": len(pair_ids),
                "delta_corrupt_to_clean": {
                    "mean": float(direct[index, 0].mean()),
                    "ci_lower": float(lower[index]),
                    "ci_upper": float(upper[index]),
                },
                "delta_clean_to_corrupt": {
                    "mean": float(direct[index, 1].mean()),
                    "ci_lower": float(lower[count + index]),
                    "ci_upper": float(upper[count + index]),
                },
                "T_direct": {
                    "mean": float(symmetric[index].mean()),
                    "ci_lower": float(lower[2 * count + index]),
                    "ci_upper": float(upper[2 * count + index]),
                },
                "identity_review": {
                    "clean_self_candidate_score_max_abs_delta": clean_self_max,
                    "corrupt_self_candidate_score_max_abs_delta": corrupt_self_max,
                    "tolerance": tolerance,
                    "passed": layer_identity_passed,
                },
            }
        )
    means = {
        int(row["layer_index"]): float(row["T_direct"]["mean"])
        for row in layer_table
        if int(row["layer_index"]) in eligible_set
    }
    maximum = max(means.values())
    exact_ties = [layer for layer in eligible_layers if means[int(layer)] == maximum]
    selected_layer = min(exact_ties)
    expected_control_rows = 4 * len(pair_ids) if profile == PROFILE_SMOKE else 0
    control_rows = [
        row
        for row in rows
        if row.get("treatment")
        in {
            "norm_random_delta",
            "random_matched_corrupt_to_clean",
            "random_matched_clean_to_corrupt",
        }
    ]
    smoke_controls_ok = (
        not control_rows
        if profile == PROFILE_SWEEP
        else len(control_rows) == expected_control_rows
        and {_site_layer(row) for row in control_rows} == {selected_layer}
        and Counter(str(row.get("treatment")) for row in control_rows)
        == Counter(
            {
                "norm_random_delta": 2 * len(pair_ids),
                "random_matched_corrupt_to_clean": len(pair_ids),
                "random_matched_clean_to_corrupt": len(pair_ids),
            }
        )
    )
    norm_random = (
        _norm_random_metadata_review(
            rows,
            pair_ids=pair_ids,
            selected_layer=selected_layer,
            seed=seed,
        )
        if profile == PROFILE_SMOKE
        else {
            "passed": not any(row.get("treatment") == "norm_random_delta" for row in rows),
            "expected_rows": 0,
            "observed_rows": sum(row.get("treatment") == "norm_random_delta" for row in rows),
            "exact_unit_coverage_passed": True,
            "failures": [],
        }
    )
    identity = _identity_and_alignment_review(
        rows,
        pair_ids=pair_ids,
        expected_self_layers=layers,
        tolerance=tolerance,
    )
    row_signatures = _exact_row_signature_review(
        rows,
        pair_ids=pair_ids,
        layers=layers,
        controls_layer=selected_layer if profile == PROFILE_SMOKE else None,
    )
    expected_rows = (72 if profile == PROFILE_SMOKE else 68) * len(pair_ids)
    integrity = (
        len(rows) == expected_rows
        and complete_direct
        and not hook_failures
        and smoke_controls_ok
        and norm_random["passed"]
        and identity["passed"]
        and row_signatures["passed"]
    )
    review = {
        "integrity_passed": integrity,
        "expected_prediction_rows": expected_rows,
        "observed_prediction_rows": len(rows),
        "direct_coverage_passed": complete_direct,
        "hook_coverage_passed": not hook_failures,
        "hook_failures": hook_failures[:10],
        "identity": identity,
        "exact_row_signatures": row_signatures,
        "controls_scope_passed": smoke_controls_ok,
        "norm_random_metadata": norm_random,
        "selected_layer": selected_layer,
        "winner_eligible_layers": list(eligible_layers),
        "selection_statistic": maximum,
        "exact_tie_set": exact_ties,
    }
    winner = {
        "selected_layer": selected_layer,
        "winner_eligible_layers": list(eligible_layers),
        "selection_statistic": maximum,
        "exact_tie_set": exact_ties,
    }
    return review, layer_table, winner


def _controls_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    selected_layer: int,
    tolerance: float,
    seed: int,
    resamples: int,
    ci_alpha: float,
) -> dict[str, Any]:
    baselines = _baseline_map(rows)
    keyed: dict[tuple[str, str, str, int | None], Mapping[str, Any]] = {}
    duplicates: list[tuple[str, str, str, int | None]] = []
    hook_failures: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        if key in keyed:
            duplicates.append(key)
        keyed[key] = row
        if row.get("treatment") not in {
            "unpatched",
            "unpatched_rerun",
        } and not _hook_diagnostics_ok(row, selected_layer):
            hook_failures.append(
                {
                    "pair_id": row.get("pair_id"),
                    "treatment": row.get("treatment"),
                }
            )
    values = np.full((3, len(pair_ids)), np.nan, dtype=np.float64)
    missing: list[dict[str, Any]] = []
    for index, pair_id in enumerate(pair_ids):
        try:
            direct_clean_receiver = _effect(
                keyed[(pair_id, "clean", "corrupt_to_clean", selected_layer)],
                baselines,
            )
            direct_corrupt_receiver = _effect(
                keyed[(pair_id, "corrupt", "clean_to_corrupt", selected_layer)],
                baselines,
            )
            random_clean = _effect(
                keyed[(pair_id, "clean", "norm_random_delta", selected_layer)],
                baselines,
            )
            random_corrupt = _effect(
                keyed[(pair_id, "corrupt", "norm_random_delta", selected_layer)],
                baselines,
            )
            matched_clean = _effect(
                keyed[
                    (
                        pair_id,
                        "clean",
                        "random_matched_corrupt_to_clean",
                        selected_layer,
                    )
                ],
                baselines,
            )
            matched_corrupt = _effect(
                keyed[
                    (
                        pair_id,
                        "corrupt",
                        "random_matched_clean_to_corrupt",
                        selected_layer,
                    )
                ],
                baselines,
            )
        except KeyError as exc:
            missing.append({"pair_id": pair_id, "missing": str(exc)})
            continue
        values[0, index] = 0.5 * (direct_clean_receiver - direct_corrupt_receiver)
        values[1, index] = 0.5 * (
            (direct_clean_receiver - random_clean) - (direct_corrupt_receiver - random_corrupt)
        )
        values[2, index] = 0.5 * (
            (direct_clean_receiver - matched_clean) - (direct_corrupt_receiver - matched_corrupt)
        )
    complete = np.all(np.isfinite(values)) and not missing
    lower, upper = _bootstrap_means(
        values,
        seed=seed,
        resamples=resamples,
        ci_alpha=ci_alpha,
    )
    names = ("T_direct", "T_norm", "T_match")
    estimates = {
        name: {
            "mean": float(values[index].mean()),
            "ci_lower": float(lower[index]),
            "ci_upper": float(upper[index]),
            "pair_count": len(pair_ids),
        }
        for index, name in enumerate(names)
    }
    identity = _identity_and_alignment_review(
        rows,
        pair_ids=pair_ids,
        expected_self_layers=[selected_layer],
        tolerance=tolerance,
    )
    treatment_counts = Counter(str(row.get("treatment")) for row in rows)
    expected_counts = Counter(
        {
            "unpatched": 2 * len(pair_ids),
            "unpatched_rerun": 2 * len(pair_ids),
            "self_patch": 2 * len(pair_ids),
            "corrupt_to_clean": len(pair_ids),
            "clean_to_corrupt": len(pair_ids),
            "norm_random_delta": 2 * len(pair_ids),
            "random_matched_corrupt_to_clean": len(pair_ids),
            "random_matched_clean_to_corrupt": len(pair_ids),
        }
    )
    exact_rows = (
        len(rows) == 12 * len(pair_ids)
        and treatment_counts == expected_counts
        and not duplicates
        and all(
            row.get("treatment") in {"unpatched", "unpatched_rerun"}
            or _site_layer(row) == selected_layer
            for row in rows
        )
    )
    norm_random = _norm_random_metadata_review(
        rows,
        pair_ids=pair_ids,
        selected_layer=selected_layer,
        seed=seed,
    )
    row_signatures = _exact_row_signature_review(
        rows,
        pair_ids=pair_ids,
        layers=[selected_layer],
        controls_layer=selected_layer,
    )
    integrity = (
        complete
        and exact_rows
        and identity["passed"]
        and not hook_failures
        and norm_random["passed"]
        and row_signatures["passed"]
    )
    gate4_passed = bool(
        integrity
        and estimates["T_direct"]["mean"] > 0.0
        and estimates["T_direct"]["ci_lower"] > 0.0
        and estimates["T_norm"]["mean"] > 0.0
        and estimates["T_norm"]["ci_lower"] > 0.0
    )
    return {
        "integrity_passed": integrity,
        "expected_prediction_rows": 12 * len(pair_ids),
        "observed_prediction_rows": len(rows),
        "exact_row_contract_passed": exact_rows,
        "complete_estimands": complete,
        "missing_rows": missing[:10],
        "hook_coverage_passed": not hook_failures,
        "hook_failures": hook_failures[:10],
        "identity": identity,
        "exact_row_signatures": row_signatures,
        "norm_random_metadata": norm_random,
        "selected_layer": selected_layer,
        "estimands": estimates,
        "gate4": {
            "passed": gate4_passed,
            "decision": "PASS" if gate4_passed else "NO-GO",
            "rule": "T_direct and T_norm means and ordinary pair-bootstrap 95% lower bounds are all > 0",
            "secondary_specificity_diagnostic": {
                "estimand": "T_match",
                "required_to_report": True,
                "hard_pass_requirement": False,
            },
        },
    }


def _same_variant_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_ids: Sequence[str],
    selected_layer: int,
    tolerance: float,
    seed: int,
    resamples: int,
    ci_alpha: float,
) -> dict[str, Any]:
    baselines = _baseline_map(rows)
    keyed: dict[tuple[str, str, str, int | None], Mapping[str, Any]] = {}
    duplicates: list[tuple[str, str, str, int | None]] = []
    hook_failures: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        if key in keyed:
            duplicates.append(key)
        keyed[key] = row
        if row.get("treatment") not in {
            "unpatched",
            "unpatched_rerun",
        } and not _hook_diagnostics_ok(row, selected_layer):
            hook_failures.append(
                {"pair_id": row.get("pair_id"), "treatment": row.get("treatment")}
            )

    values = np.full((5, len(pair_ids)), np.nan, dtype=np.float64)
    missing: list[dict[str, Any]] = []
    treatments = (
        ("clean", "random_matched_corrupt_to_clean"),
        ("clean", "random_matched_clean_to_clean"),
        ("corrupt", "random_matched_clean_to_corrupt"),
        ("corrupt", "random_matched_corrupt_to_corrupt"),
    )
    for index, pair_id in enumerate(pair_ids):
        try:
            for treatment_index, (receiver, treatment) in enumerate(treatments):
                values[treatment_index, index] = _effect(
                    keyed[(pair_id, receiver, treatment, selected_layer)],
                    baselines,
                )
        except KeyError as exc:
            missing.append({"pair_id": pair_id, "missing": str(exc)})
            continue
        values[4, index] = 0.5 * (
            (values[0, index] - values[1, index])
            - (values[2, index] - values[3, index])
        )

    complete = bool(np.all(np.isfinite(values)) and not missing)
    lower, upper = _bootstrap_means(
        values,
        seed=seed,
        resamples=resamples,
        ci_alpha=ci_alpha,
    )
    names = (
        "matched_corrupt_to_clean",
        "matched_clean_to_clean",
        "matched_clean_to_corrupt",
        "matched_corrupt_to_corrupt",
        "T_variant",
    )
    estimates = {
        name: {
            "mean": float(values[index].mean()),
            "ci_lower": float(lower[index]),
            "ci_upper": float(upper[index]),
            "pair_count": len(pair_ids),
        }
        for index, name in enumerate(names)
    }
    identity = _identity_and_alignment_review(
        rows,
        pair_ids=pair_ids,
        expected_self_layers=[selected_layer],
        tolerance=tolerance,
    )
    expected_counts = Counter(
        {
            "unpatched": 2 * len(pair_ids),
            "unpatched_rerun": 2 * len(pair_ids),
            "self_patch": 2 * len(pair_ids),
            "random_matched_corrupt_to_clean": len(pair_ids),
            "random_matched_clean_to_corrupt": len(pair_ids),
            "random_matched_clean_to_clean": len(pair_ids),
            "random_matched_corrupt_to_corrupt": len(pair_ids),
        }
    )
    treatment_counts = Counter(str(row.get("treatment")) for row in rows)
    row_signatures = _same_variant_row_signature_review(
        rows,
        pair_ids=pair_ids,
        selected_layer=selected_layer,
    )
    exact_rows = bool(
        len(rows) == 10 * len(pair_ids)
        and treatment_counts == expected_counts
        and not duplicates
        and row_signatures["passed"]
    )
    integrity = bool(
        complete
        and exact_rows
        and identity["passed"]
        and not hook_failures
    )
    return {
        "integrity_passed": integrity,
        "expected_prediction_rows": 10 * len(pair_ids),
        "observed_prediction_rows": len(rows),
        "exact_row_contract_passed": exact_rows,
        "complete_estimands": complete,
        "missing_rows": missing[:10],
        "hook_coverage_passed": not hook_failures,
        "hook_failures": hook_failures[:10],
        "identity": identity,
        "exact_row_signatures": row_signatures,
        "selected_layer": selected_layer,
        "estimands": estimates,
        "same_variant_specificity": {
            "estimand": "T_variant",
            "role": "exploratory_specificity_diagnostic",
            "hard_gate_requirement": False,
        },
    }


def _same_variant_donor_contract(
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    protocol_id: str,
    protocol_sha256: str | None,
    profile: str,
) -> tuple[bool, bool, dict[str, Any] | None]:
    resolved_record = resolved.get("same_variant_donor")
    manifest_record = manifest.get("same_variant_donor")
    if resolved_record is None:
        return False, profile != PROFILE_SAME_VARIANT and manifest_record is None, None
    if not isinstance(resolved_record, Mapping) or not isinstance(manifest_record, Mapping):
        return False, False, None
    enabled_for_profile = profile == PROFILE_SAME_VARIANT
    config_path = Path(str(resolved_record.get("config_path")))
    anchor = resolved_record.get("frozen_anchor")
    config_payload: Mapping[str, Any] = {}
    if config_path.is_file():
        loaded_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded_config, Mapping):
            config_payload = loaded_config
    configured_anchor = config_payload.get("frozen_anchor")
    passed = bool(
        dict(resolved_record) == dict(manifest_record)
        and resolved_record.get("schema_version") == 1
        and resolved_record.get("config_id") == SAME_VARIANT_CONFIG_ID
        and resolved_record.get("base_protocol_id") == protocol_id
        and protocol_sha256 is not None
        and resolved_record.get("base_protocol_sha256") == protocol_sha256
        and resolved_record.get("enabled") is True
        and resolved_record.get("profile") == PROFILE_SAME_VARIANT
        and resolved_record.get("expected_pairs") == 290
        and resolved_record.get("expected_prediction_rows") == 2900
        and resolved_record.get("hard_gate") is False
        and enabled_for_profile
        and isinstance(anchor, Mapping)
        and isinstance(configured_anchor, Mapping)
        and dict(anchor) == dict(configured_anchor)
        and anchor.get("selected_layer") == 8
        and anchor.get("winner_manifest_sha256")
        == manifest.get("source_winner_manifest_sha256")
        and manifest.get("selected_layer") == anchor.get("selected_layer")
        and manifest.get("sweep_winner_sha256")
        == anchor.get("winner_manifest_sha256")
        and manifest.get("sweep_run_id") == anchor.get("sweep_run_id")
        and manifest.get("sweep_ordered_pair_ids_sha256")
        == anchor.get("ordered_pair_ids_sha256")
        and config_path.is_file()
        and resolved_record.get("config_sha256") == sha256_file(config_path)
        and config_payload.get("config_id") == SAME_VARIANT_CONFIG_ID
        and config_payload.get("enabled") is True
    )
    review = {
        "configured": True,
        "profile": PROFILE_SAME_VARIANT,
        "config_id": resolved_record.get("config_id"),
        "config_sha256": resolved_record.get("config_sha256"),
        "hard_gate": False,
        "passed": passed,
    }
    return enabled_for_profile, passed, review


def review_residual_v2_run(
    *,
    run_root: Path,
    protocol: Mapping[str, Any],
    profile: str,
    selected_layer: int | None = None,
    protocol_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Review one residual-v3 stage and return payload plus generated artifacts."""

    if profile not in {
        PROFILE_SMOKE,
        PROFILE_SWEEP,
        PROFILE_CONTROLS,
        PROFILE_SAME_VARIANT,
    }:
        raise ValueError(f"unsupported residual-v3 profile: {profile}")
    patch_run = run_root / "patch"
    required = [
        run_root / "environment_manifest.json",
        run_root / "stage_linkage_manifest.json",
        patch_run / "predictions.jsonl",
        patch_run / "data_manifest.json",
        patch_run / "selection_manifest.json",
        patch_run / "environment.json",
        patch_run / "config.resolved.yaml",
    ]
    if profile in {PROFILE_SMOKE, PROFILE_CONTROLS, PROFILE_SAME_VARIANT}:
        required.append(patch_run / "matched_donor_manifest.json")
    if profile == PROFILE_SAME_VARIANT:
        required.append(patch_run / "source_winner_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        payload = {
            "schema_version": 2,
            "profile": profile,
            "integrity_passed": False,
            "passed": False,
            "completed": False,
            "missing_artifacts": missing,
        }
        return payload, {}

    rows = _read_jsonl(patch_run / "predictions.jsonl")
    manifest = _read_json(patch_run / "data_manifest.json")
    selection = _read_json(patch_run / "selection_manifest.json")
    patch_environment = _read_json(patch_run / "environment.json")
    root_environment = _read_json(run_root / "environment_manifest.json")
    linkage = _read_json(run_root / "stage_linkage_manifest.json")
    resolved = yaml.safe_load((patch_run / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(resolved, Mapping):
        resolved = {}
    same_variant_enabled, same_variant_config_ok, same_variant_review = (
        _same_variant_donor_contract(
            resolved,
            manifest,
            protocol_id=str(protocol.get("protocol_id")),
            protocol_sha256=protocol_sha256,
            profile=profile,
        )
    )
    pair_ids = _ordered_pair_ids(selection)
    profiles = cast(Mapping[str, Any], protocol["profiles"])
    expected_pairs = (
        290
        if profile == PROFILE_SAME_VARIANT
        else int(cast(Mapping[str, Any], profiles[profile])["expected_pairs"])
    )
    layers = [int(value) for value in protocol["assay"]["layers"]]
    eligible_layers = [int(value) for value in protocol["layer_selection"]["eligible_layers"]]
    tolerance = float(protocol["identity_controls"]["candidate_score_max_abs_delta"])
    bootstrap = cast(Mapping[str, Any], protocol["inference"]["bootstrap"])
    seed = int(bootstrap["seed"])
    resamples = int(bootstrap["resamples"])
    ci_alpha = 1.0 - float(bootstrap["confidence_level"])
    registry = cast(Mapping[str, Any], protocol["source_contract"]["pair_registry"])
    expected_order_sha = str(registry["ordered_pair_ids_sha256"])
    selection_ok = _selection_is_valid(
        selection,
        expected_pairs=expected_pairs,
        expected_full_order_sha256=expected_order_sha,
    ) and _pair_registry_matches_selection(registry, pair_ids)
    prediction_hash = sha256_file(patch_run / "predictions.jsonl")
    manifest_ok = (
        manifest.get("schema_version") == 2
        and manifest.get("profile") == profile
        and manifest.get("protocol_id") == protocol.get("protocol_id")
        and protocol_sha256 is not None
        and manifest.get("protocol_sha256") == protocol_sha256
        and manifest.get("num_pairs") == expected_pairs
        and manifest.get("pair_ids") == pair_ids
        and manifest.get("predictions_sha256") == prediction_hash
        and manifest.get("source_pair_order_sha256") == expected_order_sha
        and manifest.get("validation_accessed") is False
        and same_variant_config_ok
    )
    patch_git = patch_environment.get("git", {})
    root_git = root_environment.get("git", {})
    code_commit = str(patch_git.get("commit")) if isinstance(patch_git, Mapping) else ""
    linkage_ok = bool(
        protocol_sha256 is not None
        and isinstance(patch_git, Mapping)
        and patch_git.get("commit")
        and _current_linkage_ok(
            linkage,
            run_root=run_root,
            profile=profile,
            protocol_id=str(protocol.get("protocol_id")),
            protocol_sha256=protocol_sha256,
            code_commit=code_commit,
            root_environment=root_environment,
            patch_environment=patch_environment,
        )
    )
    provenance_ok = (
        isinstance(patch_git, Mapping)
        and isinstance(root_git, Mapping)
        and patch_git.get("commit")
        and patch_git.get("dirty") is False
        and patch_git == root_git
        and manifest.get("code") == patch_git
        and manifest.get("model") == patch_environment.get("model")
        and manifest.get("data") == patch_environment.get("data")
        and patch_environment.get("validation_accessed") is False
        and root_environment.get("validation_accessed") is False
        and resolved.get("profile") == profile
        and resolved.get("protocol_sha256") == protocol_sha256
        and resolved.get("validation_accessed") is False
        and linkage_ok
    )
    generic_rows = _generic_row_review(rows, pair_ids)

    matched_ok = True
    if profile in {PROFILE_SMOKE, PROFILE_CONTROLS, PROFILE_SAME_VARIANT}:
        matched_path = patch_run / "matched_donor_manifest.json"
        matched = _read_json(matched_path)
        matched_unhashed = dict(matched)
        matched_hash = matched_unhashed.pop("manifest_hash", None)
        edges = matched.get("edges")
        expected_edges = _expected_matched_edges(selection, pair_ids, seed=20260731)
        edges_by_target = {
            str(edge.get("target_pair_id")): str(edge.get("source_pair_id"))
            for edge in edges or []
            if isinstance(edge, Mapping)
        }
        matched_prediction_rows = [
            row
            for row in rows
            if row.get("treatment")
            in {
                "random_matched_corrupt_to_clean",
                "random_matched_clean_to_corrupt",
            }
        ]
        matched_predictions_ok = len(matched_prediction_rows) == 2 * expected_pairs and all(
            isinstance(row.get("metadata"), Mapping)
            and row["metadata"].get("control_pair_id")
            == edges_by_target.get(str(row.get("pair_id")))
            for row in matched_prediction_rows
        )
        same_variant_prediction_rows = [
            row for row in rows if row.get("treatment") in SAME_VARIANT_TREATMENTS
        ]
        expected_same_variant_rows = 2 * expected_pairs if same_variant_enabled else 0
        same_variant_predictions_ok = (
            len(same_variant_prediction_rows) == expected_same_variant_rows
            and all(
                isinstance(row.get("metadata"), Mapping)
                and row["metadata"].get("control")
                == "winner_only_same_variant_matched_donor"
                and row["metadata"].get("control_pair_id")
                == edges_by_target.get(str(row.get("pair_id")))
                for row in same_variant_prediction_rows
            )
        )
        matched_ok = bool(
            matched_hash == stable_hash(matched_unhashed, length=64)
            and isinstance(edges, list)
            and expected_edges is not None
            and edges == expected_edges
            and matched.get("schema_version") == 2
            and matched.get("domain") == "olmoe_residual_discovery_v2_matched_donor"
            and matched.get("seed") == 20260731
            and matched.get("source_selection_granularity") == "once_per_target_pair"
            and matched.get("matching_with_replacement_across_targets") is True
            and matched.get("edges_hash") == stable_hash(edges, length=64)
            and manifest.get("matched_donor_manifest_sha256") == sha256_file(matched_path)
            and matched_predictions_ok
            and same_variant_predictions_ok
        )

    source_winner_ok = True
    if profile == PROFILE_SAME_VARIANT:
        manifest_selected_layer = manifest.get("selected_layer")
        if selected_layer is None and isinstance(manifest_selected_layer, int):
            selected_layer = manifest_selected_layer
        source_winner_path = patch_run / "source_winner_manifest.json"
        source_winner = _read_json(source_winner_path)
        same_variant_record = resolved.get("same_variant_donor")
        anchor = (
            same_variant_record.get("frozen_anchor")
            if isinstance(same_variant_record, Mapping)
            else None
        )
        source_winner_ok = bool(
            isinstance(anchor, Mapping)
            and sha256_file(source_winner_path) == anchor.get("winner_manifest_sha256")
            and manifest.get("source_winner_manifest_sha256")
            == anchor.get("winner_manifest_sha256")
            and source_winner.get("protocol_id") == protocol.get("protocol_id")
            and source_winner.get("profile") == PROFILE_SWEEP
            and source_winner.get("selection_role") == "discovery_only"
            and source_winner.get("selected_layer") == anchor.get("selected_layer")
            and source_winner.get("sweep_run_id") == anchor.get("sweep_run_id")
            and source_winner.get("sweep_predictions_sha256")
            == anchor.get("sweep_predictions_sha256")
            and source_winner.get("ordered_pair_ids_sha256")
            == anchor.get("ordered_pair_ids_sha256")
        )

    prerequisite = linkage.get("prerequisite")
    linkage_environment = linkage.get("environment")
    if profile in {PROFILE_SMOKE, PROFILE_SAME_VARIANT}:
        transition_linkage_ok = prerequisite is None
    elif profile == PROFILE_SWEEP:
        transition_linkage_ok = bool(
            isinstance(prerequisite, Mapping)
            and isinstance(linkage_environment, Mapping)
            and prerequisite.get("profile") == PROFILE_SMOKE
            and prerequisite.get("integrity_passed") is True
            and _linked_prerequisite_files_ok(
                prerequisite,
                require_winner=False,
                protocol_id=str(protocol["protocol_id"]),
                code_commit=code_commit,
                environment=linkage_environment,
                ordered_pair_ids_sha256=expected_order_sha,
            )
        )
    elif profile == PROFILE_CONTROLS:
        selected_layer_eligible = selected_layer in eligible_layers
        transition_linkage_ok = bool(
            selected_layer_eligible
            and isinstance(prerequisite, Mapping)
            and isinstance(linkage_environment, Mapping)
            and prerequisite.get("profile") == PROFILE_SWEEP
            and prerequisite.get("integrity_passed") is True
            and prerequisite.get("winner_manifest_sha256") == manifest.get("sweep_winner_sha256")
            and prerequisite.get("selected_layer") == selected_layer
            and prerequisite.get("sweep_run_id") == manifest.get("sweep_run_id")
            and prerequisite.get("ordered_pair_ids_sha256") == expected_order_sha
            and manifest.get("sweep_ordered_pair_ids_sha256") == expected_order_sha
            and manifest.get("selected_layer") == selected_layer
            and _linked_prerequisite_files_ok(
                prerequisite,
                require_winner=True,
                protocol_id=str(protocol["protocol_id"]),
                code_commit=code_commit,
                environment=linkage_environment,
                ordered_pair_ids_sha256=expected_order_sha,
            )
        )

    generated: dict[str, Any] = {}
    if profile in SWEEP_PROFILES:
        try:
            stage, layer_table, winner = _sweep_review(
                rows,
                profile=profile,
                pair_ids=pair_ids,
                layers=layers,
                eligible_layers=eligible_layers,
                tolerance=tolerance,
                seed=seed,
                resamples=resamples,
                ci_alpha=ci_alpha,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            stage = {
                "integrity_passed": False,
                "review_error": str(exc),
                "selected_layer": None,
            }
            layer_table = []
            winner = {}
        integrity = bool(
            stage["integrity_passed"]
            and selection_ok
            and manifest_ok
            and provenance_ok
            and generic_rows["passed"]
            and matched_ok
            and transition_linkage_ok
        )
        stage["integrity_passed"] = integrity
        all_layer_payload = {
            "schema_version": 2,
            "protocol_id": protocol["protocol_id"],
            "profile": profile,
            "pair_count": expected_pairs,
            "bootstrap": {
                "method": "ordinary_pair_percentile",
                "resamples": resamples,
                "seed": seed,
                "confidence_level": 1.0 - ci_alpha,
                "shared_pair_indices": True,
            },
            "layers": layer_table,
        }
        atomic_write_json(run_root / "all_layer_table.json", all_layer_payload)
        winner_payload = {
            "schema_version": 2,
            "protocol_id": protocol["protocol_id"],
            "profile": profile,
            "run_id": run_root.name,
            "sweep_run_id": run_root.name,
            "sweep_predictions_sha256": prediction_hash,
            "source_pair_order_sha256": expected_order_sha,
            "ordered_pair_ids_sha256": _pair_order_sha256(pair_ids),
            "all_layer_table_sha256": sha256_file(run_root / "all_layer_table.json"),
            **winner,
            "selection_role": "engineering_only" if profile == PROFILE_SMOKE else "discovery_only",
        }
        if integrity:
            atomic_write_json(run_root / "winner_manifest.json", winner_payload)
        generated = {
            "all_layer_table": all_layer_payload,
            "winner_manifest": winner_payload,
        }
    elif profile == PROFILE_CONTROLS:
        if selected_layer is None:
            raise ValueError("selected_layer is required for residual_controls review")
        try:
            stage = _controls_review(
                rows,
                pair_ids=pair_ids,
                selected_layer=selected_layer,
                tolerance=tolerance,
                seed=seed,
                resamples=resamples,
                ci_alpha=ci_alpha,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            stage = {
                "integrity_passed": False,
                "review_error": str(exc),
                "gate4": {"passed": False, "decision": "INVALID"},
            }
        integrity = bool(
            stage["integrity_passed"]
            and selected_layer_eligible
            and selection_ok
            and manifest_ok
            and provenance_ok
            and generic_rows["passed"]
            and matched_ok
            and transition_linkage_ok
        )
        stage["integrity_passed"] = integrity
        stage["selected_layer_eligible"] = selected_layer_eligible
        if not integrity:
            stage["gate4"]["passed"] = False
            stage["gate4"]["decision"] = "INVALID"
    else:
        assert profile == PROFILE_SAME_VARIANT
        if selected_layer is None:
            raise ValueError("selected_layer is required for residual_same_variant review")
        try:
            stage = _same_variant_review(
                rows,
                pair_ids=pair_ids,
                selected_layer=selected_layer,
                tolerance=tolerance,
                seed=seed,
                resamples=resamples,
                ci_alpha=ci_alpha,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            stage = {
                "integrity_passed": False,
                "review_error": str(exc),
            }
        integrity = bool(
            stage["integrity_passed"]
            and selected_layer == 8
            and selection_ok
            and manifest_ok
            and provenance_ok
            and generic_rows["passed"]
            and matched_ok
            and source_winner_ok
            and transition_linkage_ok
        )
        stage["integrity_passed"] = integrity
        stage["source_winner_manifest_passed"] = source_winner_ok

    payload = {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "profile": profile,
        "run_id": run_root.name,
        "expected_pairs": expected_pairs,
        "predictions_sha256": prediction_hash,
        "selection_manifest_sha256": sha256_file(patch_run / "selection_manifest.json"),
        "selection_contract_passed": selection_ok,
        "data_manifest_contract_passed": manifest_ok,
        "provenance_contract_passed": provenance_ok,
        "current_stage_linkage_passed": linkage_ok,
        "generic_row_contract": generic_rows,
        "matched_donor_manifest_passed": matched_ok,
        "transition_prerequisite_linkage_passed": transition_linkage_ok,
        **stage,
    }
    if same_variant_review is not None:
        payload["same_variant_donor"] = same_variant_review
    payload["passed"] = bool(payload["integrity_passed"])
    payload["completed"] = bool(payload["integrity_passed"])
    if profile == PROFILE_CONTROLS:
        gate4 = cast(Mapping[str, Any], payload["gate4"])
        payload["scientific_decision"] = gate4["decision"]
    return payload, generated


def write_residual_v2_review(
    *,
    run_root: Path,
    output: Path,
    protocol: Mapping[str, Any],
    profile: str,
    selected_layer: int | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    payload, _generated = review_residual_v2_run(
        run_root=run_root,
        protocol=protocol,
        profile=profile,
        selected_layer=selected_layer,
        protocol_sha256=protocol_sha256,
    )
    atomic_write_json(output, payload)
    return payload
