"""Narrow three-stage OLMoE residual discovery execution path.

This module deliberately does not call the legacy ``intervene patch``
single-site sweep.  That path also runs pre-error and all-layer control arms.
The shared engine runs the versioned residual protocol's registered direct
sweep and winner-only controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from routeverify.cli import (
    _load_prmbench_selection_from_hf,
    _render_and_boundary_token,
    _score_row,
    _selected_sites,
    _site_key,
    _tensor_to_numpy,
)
from routeverify.config import (
    environment_snapshot,
    git_state,
    load_data_config,
    load_model_config,
    load_yaml,
)
from routeverify.data.prmbench import PRMBenchPair
from routeverify.data.prompts import STEP_BINARY_CANDIDATES
from routeverify.data.schema import stable_hash
from routeverify.instrumentation.sites import CaptureSpec, LayerSite, PatchBundle, PatchSpec
from routeverify.instrumentation.storage import atomic_write_json, atomic_write_jsonl
from routeverify.models.registry import create_adapter
from routeverify.reporting.residual_v2 import (
    PROFILE_CONTROLS,
    PROFILE_SAME_VARIANT,
    PROFILE_SMOKE,
    PROFILE_SWEEP,
    sha256_file,
    write_residual_v2_review,
)

DEFAULT_PROTOCOL = Path("configs/analysis/olmoe_residual_discovery_v3.yaml")
DEFAULT_MODEL = Path("configs/models/olmoe.yaml")
DEFAULT_DATA = Path("configs/data/prmbench.yaml")
SAME_VARIANT_CONFIG_ID = "olmoe_same_variant_donor_v1_20260808"
RESIDUAL_RUNTIME_ENV_KEYS = (
    "RUNPOD_POD_ID",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "RESIDUAL_V2_RUN_ID",
    "RESIDUAL_V2_RUN_ROOT",
    "RESIDUAL_V2_PACKAGE_PATH",
)


def _residual_runtime_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in RESIDUAL_RUNTIME_ENV_KEYS}


def _shared_runtime_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) and left.get(key) == right.get(key)
        for key in ("RUNPOD_POD_ID", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _pair_order_sha256(pair_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in pair_ids).encode("utf-8")
    ).hexdigest()


def _pair_id(pair: PRMBenchPair) -> str:
    return pair.clean.pair_id or pair.corrupt.pair_id or pair.clean.sample_id


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 2,
        "protocol_id": "olmoe_residual_discovery_v3_20260803",
    }
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ValueError(f"protocol {field} must be {value!r}")
    if protocol.get("access_policy", {}).get("allowed_partition") != "discovery":
        raise ValueError("residual v3 is discovery-only")
    if protocol.get("access_policy", {}).get("validation_access") != "forbidden":
        raise ValueError("residual v3 must keep validation forbidden")
    assay = protocol.get("assay", {})
    if (
        float(assay.get("alpha", -1.0)) != 1.0
        or assay.get("site_kind") != "post_moe_residual"
        or assay.get("patch_mode") != "single_layer"
        or assay.get("layers") != list(range(16))
    ):
        raise ValueError("residual v3 assay drifted from alpha=1 single-layer 16-layer sweep")
    layer_selection = protocol.get("layer_selection", {})
    if layer_selection.get("eligible_layers") != list(range(15)) or layer_selection.get(
        "built_in_null_layers"
    ) != [15]:
        raise ValueError("residual v3 must measure layer 15 but restrict winners to layers 0-14")
    profiles = protocol.get("profiles", {})
    required_profiles = {
        PROFILE_SMOKE: (8, 576),
        PROFILE_SWEEP: (290, 19720),
        PROFILE_CONTROLS: (290, 3480),
    }
    for profile, (pairs, rows) in required_profiles.items():
        contract = profiles.get(profile, {})
        if (
            contract.get("expected_pairs") != pairs
            or contract.get("expected_prediction_rows") != rows
        ):
            raise ValueError(f"protocol profile drift: {profile}")
    bootstrap = protocol.get("inference", {}).get("bootstrap", {})
    if (
        bootstrap.get("method") != "ordinary_pair_percentile"
        or bootstrap.get("resamples") != 50_000
        or bootstrap.get("seed") != 20260731
        or float(bootstrap.get("confidence_level", 0.0)) != 0.95
    ):
        raise ValueError("residual v3 bootstrap contract drift")
    gate4 = protocol.get("gate4", {})
    if gate4.get("pass_requires_all") != [
        "T_direct_mean_strictly_greater_than_zero",
        "T_direct_pair_bootstrap_95_ci_lower_strictly_greater_than_zero",
        "T_norm_mean_strictly_greater_than_zero",
        "T_norm_pair_bootstrap_95_ci_lower_strictly_greater_than_zero",
    ]:
        raise ValueError("residual v3 Gate 4 primary-sensitivity contract drift")
    t_match = gate4.get("secondary_specificity_diagnostic", {})
    if (
        t_match.get("estimand") != "T_match"
        or t_match.get("required_to_report") is not True
        or t_match.get("hard_pass_requirement") is not False
    ):
        raise ValueError("residual v3 must report T_match without making it a hard gate")


def _load_same_variant_donor_config(
    path: Path | None,
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
) -> dict[str, Any] | None:
    """Load the optional exploratory donor control without changing the v3 protocol."""

    if path is None:
        return None
    config = load_yaml(path)
    base = config.get("base_protocol", {})
    treatments = config.get("treatments", {})
    matching = config.get("matching", {})
    diagnostic = config.get("diagnostic", {})
    protocol_sha = sha256_file(protocol_path)
    profile = config.get("profile", {})
    anchor = config.get("frozen_anchor", {})
    if (
        config.get("schema_version") != 1
        or config.get("config_id") != SAME_VARIANT_CONFIG_ID
        or config.get("status") != "exploratory_optional_control"
        or config.get("enabled") is not True
        or base.get("protocol_id") != protocol.get("protocol_id")
        or base.get("protocol_sha256") != protocol_sha
        or profile.get("name") != PROFILE_SAME_VARIANT
        or profile.get("expected_pairs") != 290
        or profile.get("expected_prediction_rows") != 2900
        or profile.get("rows_per_pair", {}).get("total") != 10
        or anchor
        != {
            "selected_layer": 8,
            "sweep_run_id": "olmoe_residual_v3_sweep_20260807T064828Z",
            "ordered_pair_ids_sha256": (
                "9a8552534796a6eb1ce8d6898d49398aa1e723b3ccf1db7f64c671359e190b4a"
            ),
            "sweep_predictions_sha256": (
                "e5287f8ac570b64bd69ea2e7df9c0e24b0fc2ec666ca3064d07ba5d6f19c7f21"
            ),
            "winner_manifest_sha256": (
                "1f71a4b24cffd013f4dfde8c095d648983ea18e4359e69d4c3f5c6b54999ce62"
            ),
        }
        or matching
        != {
            "source": "existing_matched_donor_edge",
            "same_source_pair_for_both_variants": True,
            "exclude_target_pair": True,
        }
        or treatments
        != {
            "clean_receiver": "random_matched_clean_to_clean",
            "corrupt_receiver": "random_matched_corrupt_to_corrupt",
        }
        or diagnostic.get("symbol") != "T_variant"
        or diagnostic.get("role") != "exploratory_specificity_diagnostic"
        or diagnostic.get("hard_gate") is not False
    ):
        raise ValueError("same-variant donor config drifted from its exploratory v1 contract")
    return {
        "schema_version": 1,
        "config_id": SAME_VARIANT_CONFIG_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "base_protocol_id": str(protocol["protocol_id"]),
        "base_protocol_sha256": protocol_sha,
        "enabled": True,
        "profile": PROFILE_SAME_VARIANT,
        "expected_pairs": 290,
        "expected_prediction_rows": 2900,
        "frozen_anchor": dict(anchor),
        "hard_gate": False,
    }


def _load_frozen_pair_order(
    protocol: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    registry = cast(Mapping[str, Any], protocol["source_contract"]["pair_registry"])
    pair_effects = Path(str(registry["pair_effects_path"]))
    analysis_path = Path(str(registry["analysis_path"]))
    if not pair_effects.is_file() or not analysis_path.is_file():
        raise FileNotFoundError("committed residual-v3 pair registry artifacts are missing")
    pair_effects_sha = sha256_file(pair_effects)
    analysis_sha = sha256_file(analysis_path)
    if pair_effects_sha != registry["pair_effects_sha256"]:
        raise ValueError("committed pair_effects.jsonl SHA256 mismatch")
    if analysis_sha != registry["analysis_sha256"]:
        raise ValueError("committed analysis.json SHA256 mismatch")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    source = analysis.get("source", {})
    if (
        source.get("num_pairs") != registry["expected_pair_count"]
        or source.get("pair_order_sha256") != registry["ordered_pair_ids_sha256"]
        or source.get("data_manifest_sha256")
        != registry["referenced_discovery_data_manifest_sha256"]
        or source.get("predictions_sha256") != registry["referenced_discovery_predictions_sha256"]
        or source.get("run_id") != registry["reference_discovery_run_id"]
        or source.get("source_hash_verification", {}).get("status") != "passed"
    ):
        raise ValueError("analysis.json does not bind the registered discovery source")
    pair_ids: list[str] = []
    seen: set[str] = set()
    with pair_effects.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id") if isinstance(row, dict) else None
            if not pair_id:
                raise ValueError(f"missing pair_id at {pair_effects}:{line_number}")
            pair_id = str(pair_id)
            if pair_id not in seen:
                seen.add(pair_id)
                pair_ids.append(pair_id)
    if (
        len(pair_ids) != int(registry["expected_pair_count"])
        or _pair_order_sha256(pair_ids) != registry["ordered_pair_ids_sha256"]
    ):
        raise ValueError("committed first-occurrence pair order does not match frozen SHA256")
    evidence = {
        "pair_effects_path": str(pair_effects),
        "pair_effects_sha256": pair_effects_sha,
        "analysis_path": str(analysis_path),
        "analysis_sha256": analysis_sha,
        "source_data_manifest_sha256": source["data_manifest_sha256"],
        "source_predictions_sha256": source["predictions_sha256"],
        "source_run_id": source["run_id"],
        "pair_count": len(pair_ids),
        "pair_order_sha256": _pair_order_sha256(pair_ids),
        "serialization": "utf8_pair_ids_joined_by_newline_with_final_newline",
    }
    return pair_ids, evidence


def _select_frozen_pairs(
    *,
    data_config: Any,
    pair_ids: Sequence[str],
    limit: int,
    seed: int,
    registry_evidence: Mapping[str, Any],
) -> tuple[list[PRMBenchPair], dict[str, Any], dict[str, Any]]:
    loaded = _load_prmbench_selection_from_hf(data_config, limit=None, seed=seed)
    by_id = {_pair_id(pair): pair for pair in loaded.pairs}
    missing = [pair_id for pair_id in pair_ids if pair_id not in by_id]
    if missing:
        raise ValueError(f"registered discovery pairs missing from pinned dataset: {missing[:5]}")
    selected_ids = list(pair_ids[:limit])
    selected = [by_id[pair_id] for pair_id in selected_ids]
    pair_rows = [
        {
            "pair_id": _pair_id(pair),
            "group_id": pair.clean.group_id,
            "clean_sample_id": pair.clean.sample_id,
            "corrupt_sample_id": pair.corrupt.sample_id,
            "clean_content_hash": pair.clean.content_hash,
            "corrupt_content_hash": pair.corrupt.content_hash,
            "corrupt_error_type": pair.corrupt.error_type,
            "corrupt_first_error": pair.corrupt.first_error,
        }
        for pair in selected
    ]
    selection: dict[str, Any] = {
        "schema_version": 2,
        "selection_unit": "clean_corrupt_pair",
        "selection_rule": "prefix_of_committed_discovery_cpu_pair_order",
        "selected_pair_count": len(selected),
        "selected_example_count": 2 * len(selected),
        "source_pair_count": len(pair_ids),
        "source_pair_order_sha256": _pair_order_sha256(pair_ids),
        "selected_pair_order_sha256": _pair_order_sha256(selected_ids),
        "natural_resort_applied": False,
        "pairs": pair_rows,
        "pairs_hash": stable_hash(pair_rows, length=64),
        "registry": dict(registry_evidence),
    }
    selection["manifest_hash"] = stable_hash(selection, length=64)
    provenance = {
        **loaded.source_provenance,
        "pair_registry": dict(registry_evidence),
        "selection_manifest": selection,
    }
    return selected, selection, provenance


def norm_matched_random_delta_replacement_v2(
    receiver: Any,
    donor: Any,
    *,
    seed: int,
    unit_key: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the frozen float64 norm-random delta, then cast once."""

    receiver_array = _tensor_to_numpy(receiver)
    donor_array = _tensor_to_numpy(donor)
    if receiver_array.shape != donor_array.shape:
        raise ValueError(
            f"receiver/donor shapes differ: {receiver_array.shape} vs {donor_array.shape}"
        )
    receiver64 = np.asarray(receiver_array, dtype=np.float64)
    donor64 = np.asarray(donor_array, dtype=np.float64)
    target_delta = donor64 - receiver64
    target_norm = float(np.linalg.norm(target_delta.reshape(-1), ord=2))
    seed_payload = {
        "domain": "olmoe_residual_discovery_v2_norm_random",
        "seed": int(seed),
        "unit_key": dict(unit_key),
    }
    seed_digest = hashlib.sha256(
        json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rng_seed = int(seed_digest[:16], 16)
    if target_norm == 0.0:
        delta64 = np.zeros_like(target_delta, dtype=np.float64)
    else:
        gaussian = np.random.default_rng(rng_seed).standard_normal(target_delta.shape)
        gaussian_norm = float(np.linalg.norm(gaussian.reshape(-1), ord=2))
        if gaussian_norm == 0.0:
            raise RuntimeError("deterministic Gaussian realization has zero norm")
        delta64 = gaussian * (target_norm / gaussian_norm)
    constructed_norm = float(np.linalg.norm(delta64.reshape(-1), ord=2))
    relative_error = (
        0.0 if target_norm == 0.0 else abs(constructed_norm - target_norm) / target_norm
    )
    if relative_error > 1.0e-12:
        raise RuntimeError(
            f"float64 norm-random construction relative error {relative_error} exceeds 1e-12"
        )
    replacement64 = receiver64 + delta64
    return replacement64, {
        "domain": seed_payload["domain"],
        "seed": seed,
        "unit_key": dict(unit_key),
        "unit_key_sha256": seed_digest,
        "target_delta_l2_float64": target_norm,
        "constructed_delta_l2_float64": constructed_norm,
        "construction_relative_l2_error": relative_error,
        "construction_dtype": "float64",
        "execution_cast": "deferred_once_to_model_target_dtype_by_patch_adapter",
        "redraw": False,
    }


def _matched_source_indices(pairs: Sequence[PRMBenchPair], *, seed: int) -> list[int]:
    if len(pairs) < 2:
        raise ValueError("matched donor control requires at least two pairs")
    selected: list[int] = []
    for target_index, target in enumerate(pairs):
        target_error = target.corrupt.first_error
        target_type = target.corrupt.error_type

        def rank(
            source_index: int,
            *,
            target: PRMBenchPair = target,
            target_error: int | None = target_error,
            target_type: str | None = target_type,
        ) -> tuple[int, int, str]:
            source = pairs[source_index]
            source_error = source.corrupt.first_error
            distance = 10**9
            if target_error is not None and source_error is not None:
                distance = abs(target_error - source_error)
            tie = stable_hash(
                {
                    "domain": "olmoe_residual_discovery_v2_matched_donor",
                    "seed": seed,
                    "target_pair_id": _pair_id(target),
                    "source_pair_id": _pair_id(source),
                },
                length=64,
            )
            return int(source.corrupt.error_type != target_type), distance, tie

        candidates = [index for index in range(len(pairs)) if index != target_index]
        selected.append(min(candidates, key=rank))
    return selected


def _matched_manifest(
    pairs: Sequence[PRMBenchPair], source_indices: Sequence[int], *, seed: int
) -> dict[str, Any]:
    edges = []
    for target, source_index in zip(pairs, source_indices, strict=True):
        source = pairs[source_index]
        target_step = target.corrupt.first_error
        source_step = source.corrupt.first_error
        edges.append(
            {
                "target_pair_id": _pair_id(target),
                "source_pair_id": _pair_id(source),
                "target_error_type": target.corrupt.error_type,
                "source_error_type": source.corrupt.error_type,
                "same_error_type": target.corrupt.error_type == source.corrupt.error_type,
                "corrupt_first_error_step_distance": (
                    None
                    if target_step is None or source_step is None
                    else abs(target_step - source_step)
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "domain": "olmoe_residual_discovery_v2_matched_donor",
        "seed": seed,
        "source_selection_granularity": "once_per_target_pair",
        "matching_with_replacement_across_targets": True,
        "edges": edges,
        "edges_hash": stable_hash(edges, length=64),
    }
    payload["manifest_hash"] = stable_hash(payload, length=64)
    return payload


def _summarize_patch_diagnostics(
    diagnostics: Mapping[str, Mapping[str, Any]], site: LayerSite
) -> dict[str, Any]:
    key = _site_key(site)
    return {
        "candidate_count": len(diagnostics),
        "registered_site": key,
        "all_hooks_fired": all(
            value.get("registered_site_count") == 1
            and value.get("registered_sites") == [key]
            and value.get("all_registered_hooks_fired") is True
            and int(value.get("hook_fire_counts", {}).get(key, 0)) > 0
            for value in diagnostics.values()
        ),
        "candidates": dict(diagnostics),
    }


def _score_with_replacements(
    adapter: Any,
    *,
    prompt: str,
    candidates: Sequence[str],
    site: LayerSite,
    token_position: int,
    replacements: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundles = {
        candidate: PatchBundle(
            (
                PatchSpec(
                    site=site,
                    token_position=token_position,
                    replacement=replacements[candidate],
                    metadata={"alpha": 1.0, "residual_v2": True},
                ),
            )
        )
        for candidate in candidates
    }
    scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
        prompt,
        list(candidates),
        bundles_by_candidate=bundles,
    )
    return scores, _summarize_patch_diagnostics(diagnostics, site)


@dataclass(frozen=True)
class _PreparedPair:
    pair: PRMBenchPair
    pair_id: str
    boundary_step: int
    clean_prompt: Any
    corrupt_prompt: Any
    clean_capture: Mapping[str, Any]
    corrupt_capture: Mapping[str, Any]


def _prepare_pair(
    adapter: Any,
    pair: PRMBenchPair,
    *,
    sites: tuple[LayerSite, ...],
    candidates: list[str],
) -> _PreparedPair:
    boundary = pair.corrupt.first_error
    if boundary is None:
        raise ValueError(f"registered corrupt pair has no first_error: {_pair_id(pair)}")
    clean_prompt = _render_and_boundary_token(
        adapter,
        pair.clean,
        boundary_step=boundary,
        prompt_mode="step_binary",
        input_format="chat",
    )
    corrupt_prompt = _render_and_boundary_token(
        adapter,
        pair.corrupt,
        boundary_step=boundary,
        prompt_mode="step_binary",
        input_format="chat",
    )
    clean_capture = adapter.capture_candidate_activations(
        clean_prompt.raw_render.prompt,
        candidates,
        CaptureSpec(
            sites=sites,
            token_positions=(clean_prompt.boundary_token,),
            dtype="float32",
        ),
    )
    corrupt_capture = adapter.capture_candidate_activations(
        corrupt_prompt.raw_render.prompt,
        candidates,
        CaptureSpec(
            sites=sites,
            token_positions=(corrupt_prompt.boundary_token,),
            dtype="float32",
        ),
    )
    return _PreparedPair(
        pair=pair,
        pair_id=_pair_id(pair),
        boundary_step=boundary,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        clean_capture=clean_capture,
        corrupt_capture=corrupt_capture,
    )


def _capture_replacements(
    captures: Mapping[str, Any], candidates: Sequence[str], site: LayerSite
) -> dict[str, Any]:
    key = _site_key(site)
    return {candidate: captures[candidate].arrays[key] for candidate in candidates}


def _add_input_provenance(rows: list[dict[str, Any]], prepared: _PreparedPair) -> None:
    for row in rows:
        receiver = str(row["receiver_variant"])
        row["input_provenance"] = (
            prepared.clean_prompt.provenance
            if receiver == "clean"
            else prepared.corrupt_prompt.provenance
        )


def _baseline_rows(
    adapter: Any, prepared: _PreparedPair, candidates: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receiver, prompt in (
        ("clean", prepared.clean_prompt),
        ("corrupt", prepared.corrupt_prompt),
    ):
        first = adapter.score_candidates(prompt.raw_render.prompt, candidates)
        rerun = adapter.score_candidates(prompt.raw_render.prompt, candidates)
        for treatment, scores in (("unpatched", first), ("unpatched_rerun", rerun)):
            rows.append(
                _score_row(
                    pair_id=prepared.pair_id,
                    receiver_variant=receiver,
                    donor_variant=None,
                    treatment=treatment,
                    site=None,
                    scores=scores,
                    boundary_step=prepared.boundary_step,
                    token_position=prompt.boundary_token,
                    metadata={"profile_path": "residual_v2"},
                )
            )
    _add_input_provenance(rows, prepared)
    return rows


def _direct_and_self_rows(
    adapter: Any,
    prepared: _PreparedPair,
    *,
    sites: Sequence[LayerSite],
    candidates: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for site in sites:
        arms = (
            (
                "clean",
                "clean",
                "self_patch",
                prepared.clean_prompt,
                prepared.clean_capture,
            ),
            (
                "corrupt",
                "corrupt",
                "self_patch",
                prepared.corrupt_prompt,
                prepared.corrupt_capture,
            ),
            (
                "clean",
                "corrupt",
                "corrupt_to_clean",
                prepared.clean_prompt,
                prepared.corrupt_capture,
            ),
            (
                "corrupt",
                "clean",
                "clean_to_corrupt",
                prepared.corrupt_prompt,
                prepared.clean_capture,
            ),
        )
        for receiver, donor, treatment, prompt, donor_capture in arms:
            scores, diagnostics = _score_with_replacements(
                adapter,
                prompt=prompt.raw_render.prompt,
                candidates=candidates,
                site=site,
                token_position=prompt.boundary_token,
                replacements=_capture_replacements(donor_capture, candidates, site),
            )
            rows.append(
                _score_row(
                    pair_id=prepared.pair_id,
                    receiver_variant=receiver,
                    donor_variant=donor,
                    treatment=treatment,
                    site=site,
                    scores=scores,
                    boundary_step=prepared.boundary_step,
                    token_position=prompt.boundary_token,
                    metadata={
                        "profile_path": "residual_v2",
                        "alpha": 1.0,
                        "patch_mode": "single_layer",
                        "patch_diagnostics": diagnostics,
                    },
                )
            )
    _add_input_provenance(rows, prepared)
    return rows


def _self_rows(
    adapter: Any,
    prepared: _PreparedPair,
    *,
    site: LayerSite,
    candidates: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receiver, prompt, donor_capture in (
        ("clean", prepared.clean_prompt, prepared.clean_capture),
        ("corrupt", prepared.corrupt_prompt, prepared.corrupt_capture),
    ):
        scores, diagnostics = _score_with_replacements(
            adapter,
            prompt=prompt.raw_render.prompt,
            candidates=candidates,
            site=site,
            token_position=prompt.boundary_token,
            replacements=_capture_replacements(donor_capture, candidates, site),
        )
        rows.append(
            _score_row(
                pair_id=prepared.pair_id,
                receiver_variant=receiver,
                donor_variant=receiver,
                treatment="self_patch",
                site=site,
                scores=scores,
                boundary_step=prepared.boundary_step,
                token_position=prompt.boundary_token,
                metadata={
                    "profile_path": "residual_v2",
                    "alpha": 1.0,
                    "patch_mode": "single_layer",
                    "patch_diagnostics": diagnostics,
                },
            )
        )
    _add_input_provenance(rows, prepared)
    return rows


def _quick_select_layer(rows: Sequence[Mapping[str, Any]], eligible_layers: Sequence[int]) -> int:
    if not eligible_layers:
        raise ValueError("winner selection requires at least one eligible layer")
    baselines = {
        (str(row["pair_id"]), str(row["receiver_variant"])): float(
            row["margin_incorrect_minus_correct_mean"]
        )
        for row in rows
        if row.get("treatment") == "unpatched"
    }
    by_layer: dict[int, list[float]] = {layer: [] for layer in eligible_layers}
    effects: dict[tuple[str, int, str], float] = {}
    for row in rows:
        treatment = str(row.get("treatment"))
        site = str(row.get("site"))
        if treatment not in {"corrupt_to_clean", "clean_to_corrupt"}:
            continue
        layer = int(site.split(".", 1)[0].removeprefix("layer_"))
        effects[(str(row["pair_id"]), layer, treatment)] = (
            float(row["margin_incorrect_minus_correct_mean"])
            - baselines[(str(row["pair_id"]), str(row["receiver_variant"]))]
        )
    pair_ids: list[str] = []
    seen_pair_ids: set[str] = set()
    for row in rows:
        if row.get("treatment") not in {"corrupt_to_clean", "clean_to_corrupt"}:
            continue
        pair_id = str(row.get("pair_id"))
        if pair_id not in seen_pair_ids:
            seen_pair_ids.add(pair_id)
            pair_ids.append(pair_id)
    for layer in eligible_layers:
        for pair_id in pair_ids:
            by_layer[layer].append(
                0.5
                * (
                    effects[(pair_id, layer, "corrupt_to_clean")]
                    - effects[(pair_id, layer, "clean_to_corrupt")]
                )
            )
    means = {layer: float(np.mean(values, dtype=np.float64)) for layer, values in by_layer.items()}
    maximum = max(means.values())
    return min(layer for layer, value in means.items() if value == maximum)


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _log_pair_progress(
    *,
    profile: str,
    phase: str,
    completed: int,
    total: int,
    started_at: float,
) -> None:
    elapsed = max(0.0, time.monotonic() - started_at)
    eta = elapsed * (total - completed) / completed if completed else 0.0
    print(
        f"[{profile}:{phase}] completed={completed}/{total} "
        f"elapsed={_format_duration(elapsed)} ETA={_format_duration(eta)}",
        flush=True,
    )


def _matched_donor_rows(
    adapter: Any,
    prepared: _PreparedPair,
    matched_prepared: _PreparedPair,
    *,
    site: LayerSite,
    candidates: list[str],
    include_same_variant_donor: bool = False,
) -> list[dict[str, Any]]:
    matched_arms = [
        (
            "clean",
            "matched_corrupt",
            "random_matched_corrupt_to_clean",
            prepared.clean_prompt,
            matched_prepared.corrupt_capture,
        ),
        (
            "corrupt",
            "matched_clean",
            "random_matched_clean_to_corrupt",
            prepared.corrupt_prompt,
            matched_prepared.clean_capture,
        ),
    ]
    if include_same_variant_donor:
        matched_arms.extend(
            [
                (
                    "clean",
                    "matched_clean",
                    "random_matched_clean_to_clean",
                    prepared.clean_prompt,
                    matched_prepared.clean_capture,
                ),
                (
                    "corrupt",
                    "matched_corrupt",
                    "random_matched_corrupt_to_corrupt",
                    prepared.corrupt_prompt,
                    matched_prepared.corrupt_capture,
                ),
            ]
        )
    rows: list[dict[str, Any]] = []
    for receiver, donor, treatment, prompt, donor_capture in matched_arms:
        scores, diagnostics = _score_with_replacements(
            adapter,
            prompt=prompt.raw_render.prompt,
            candidates=candidates,
            site=site,
            token_position=prompt.boundary_token,
            replacements=_capture_replacements(donor_capture, candidates, site),
        )
        rows.append(
            _score_row(
                pair_id=prepared.pair_id,
                receiver_variant=receiver,
                donor_variant=donor,
                treatment=treatment,
                site=site,
                scores=scores,
                boundary_step=prepared.boundary_step,
                token_position=prompt.boundary_token,
                metadata={
                    "profile_path": "residual_v2",
                    "control": (
                        "winner_only_same_variant_matched_donor"
                        if treatment
                        in {
                            "random_matched_clean_to_clean",
                            "random_matched_corrupt_to_corrupt",
                        }
                        else "winner_only_matched_donor"
                    ),
                    "control_pair_id": matched_prepared.pair_id,
                    "patch_diagnostics": diagnostics,
                },
            )
        )
    _add_input_provenance(rows, prepared)
    return rows


def _control_rows(
    adapter: Any,
    prepared: _PreparedPair,
    matched_prepared: _PreparedPair,
    *,
    site: LayerSite,
    candidates: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    key = _site_key(site)
    random_metadata: dict[str, dict[str, Any]] = {}
    for receiver, prompt, receiver_capture, direct_donor_capture in (
        (
            "clean",
            prepared.clean_prompt,
            prepared.clean_capture,
            prepared.corrupt_capture,
        ),
        (
            "corrupt",
            prepared.corrupt_prompt,
            prepared.corrupt_capture,
            prepared.clean_capture,
        ),
    ):
        replacements: dict[str, Any] = {}
        candidate_metadata: dict[str, Any] = {}
        for candidate in candidates:
            replacement, metadata = norm_matched_random_delta_replacement_v2(
                receiver_capture[candidate].arrays[key],
                direct_donor_capture[candidate].arrays[key],
                seed=seed,
                unit_key={
                    "pair_id": prepared.pair_id,
                    "receiver": receiver,
                    "candidate": candidate,
                    "layer": site.layer_index,
                },
            )
            replacements[candidate] = replacement
            candidate_metadata[candidate] = metadata
        scores, diagnostics = _score_with_replacements(
            adapter,
            prompt=prompt.raw_render.prompt,
            candidates=candidates,
            site=site,
            token_position=prompt.boundary_token,
            replacements=replacements,
        )
        random_metadata[receiver] = candidate_metadata
        rows.append(
            _score_row(
                pair_id=prepared.pair_id,
                receiver_variant=receiver,
                donor_variant="random",
                treatment="norm_random_delta",
                site=site,
                scores=scores,
                boundary_step=prepared.boundary_step,
                token_position=prompt.boundary_token,
                metadata={
                    "profile_path": "residual_v2",
                    "control": "winner_only_norm_random",
                    "randomization": candidate_metadata,
                    "patch_diagnostics": diagnostics,
                },
            )
        )
    rows.extend(
        _matched_donor_rows(
            adapter,
            prepared,
            matched_prepared,
            site=site,
            candidates=candidates,
        )
    )
    _add_input_provenance(rows, prepared)
    return rows


def run_residual_v2(
    *,
    protocol_path: Path,
    profile: str,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL,
    data_path: Path = DEFAULT_DATA,
    sweep_winner_path: Path | None = None,
    same_variant_config_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_yaml(protocol_path)
    _validate_protocol(protocol)
    same_variant_config = _load_same_variant_donor_config(
        same_variant_config_path,
        protocol=protocol,
        protocol_path=protocol_path,
    )
    supported_profiles = {
        PROFILE_SMOKE,
        PROFILE_SWEEP,
        PROFILE_CONTROLS,
        PROFILE_SAME_VARIANT,
    }
    if profile not in supported_profiles:
        raise ValueError(f"unsupported profile: {profile}")
    if profile == PROFILE_SAME_VARIANT and same_variant_config is None:
        raise ValueError("residual_same_variant requires --same-variant-config")
    if profile != PROFILE_SAME_VARIANT and same_variant_config is not None:
        raise ValueError("same-variant config is only valid for residual_same_variant")
    if output_dir.exists():
        raise FileExistsError(f"residual v3 refuses an existing output_dir: {output_dir}")
    expected_pairs = (
        int(same_variant_config["expected_pairs"])
        if same_variant_config is not None
        else int(protocol["profiles"][profile]["expected_pairs"])
    )
    seed = 20260730
    ordered_ids, registry_evidence = _load_frozen_pair_order(protocol)
    model_config = load_model_config(model_path).model_copy(update={"load_weights": True})
    data_config = load_data_config(data_path)
    pairs, selection, data_provenance = _select_frozen_pairs(
        data_config=data_config,
        pair_ids=ordered_ids,
        limit=expected_pairs,
        seed=seed,
        registry_evidence=registry_evidence,
    )
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    layers = [int(value) for value in protocol["assay"]["layers"]]
    eligible_layers = [int(value) for value in protocol["layer_selection"]["eligible_layers"]]
    sites = tuple(
        site
        for site in _selected_sites(adapter, ["post_moe_residual"], layers=layers)
        if site.kind == "post_moe_residual"
    )
    if len(sites) != 16 or [site.layer_index for site in sites] != layers:
        raise RuntimeError("residual v3 requires all 16 ordered post-MoE residual sites")

    selected_layer: int | None = None
    sweep_winner_sha: str | None = None
    sweep_run_id: str | None = None
    sweep_ordered_pair_ids_sha256: str | None = None
    if profile in {PROFILE_CONTROLS, PROFILE_SAME_VARIANT}:
        if sweep_winner_path is None or not sweep_winner_path.is_file():
            raise ValueError(f"{profile} requires a finalized sweep winner manifest")
        winner = json.loads(sweep_winner_path.read_text(encoding="utf-8"))
        if (
            winner.get("protocol_id") != protocol["protocol_id"]
            or winner.get("profile") != PROFILE_SWEEP
            or winner.get("selection_role") != "discovery_only"
            or not winner.get("sweep_run_id")
            or winner.get("ordered_pair_ids_sha256") != _pair_order_sha256(ordered_ids)
        ):
            raise ValueError("sweep winner manifest does not match residual-v3 discovery")
        selected_layer = int(winner["selected_layer"])
        sweep_run_id = str(winner["sweep_run_id"])
        sweep_ordered_pair_ids_sha256 = str(winner["ordered_pair_ids_sha256"])
        if selected_layer not in eligible_layers:
            raise ValueError("sweep winner selected_layer is outside eligible layers 0-14")
        sweep_winner_sha = sha256_file(sweep_winner_path)
        if same_variant_config is not None:
            anchor = same_variant_config["frozen_anchor"]
            if (
                sweep_winner_sha != anchor["winner_manifest_sha256"]
                or selected_layer != anchor["selected_layer"]
                or sweep_run_id != anchor["sweep_run_id"]
                or winner.get("sweep_predictions_sha256")
                != anchor["sweep_predictions_sha256"]
                or sweep_ordered_pair_ids_sha256
                != anchor["ordered_pair_ids_sha256"]
            ):
                raise ValueError("same-variant run is not anchored to the frozen v3 winner")

    output_dir.mkdir(parents=True)
    atomic_write_json(output_dir / "selection_manifest.json", selection)
    environment = environment_snapshot(
        model=model_config,
        data=data_config,
        data_provenance=data_provenance,
        seed=seed,
    )
    environment["validation_accessed"] = False
    environment["residual_v2_runtime"] = _residual_runtime_environment()
    environment["protocol"] = {
        "path": str(protocol_path),
        "sha256": sha256_file(protocol_path),
        "protocol_id": protocol["protocol_id"],
    }
    atomic_write_json(output_dir / "environment.json", environment)
    resolved = {
        "schema_version": 2,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol": protocol,
        "profile": profile,
        "model_path": str(model_path),
        "model": model_config.model_dump(mode="json"),
        "data_path": str(data_path),
        "data": data_config.model_dump(mode="json"),
        "pair_registry": registry_evidence,
        "selected_layer": selected_layer,
        "sweep_winner_path": None if sweep_winner_path is None else str(sweep_winner_path),
        "sweep_winner_sha256": sweep_winner_sha,
        "sweep_run_id": sweep_run_id,
        "sweep_ordered_pair_ids_sha256": sweep_ordered_pair_ids_sha256,
        "validation_accessed": False,
    }
    if same_variant_config is not None:
        resolved["same_variant_donor"] = same_variant_config
        atomic_write_json(output_dir / "source_winner_manifest.json", winner)
    _atomic_write_text(
        output_dir / "config.resolved.yaml",
        yaml.safe_dump(resolved, sort_keys=True),
    )

    candidates = list(STEP_BINARY_CANDIDATES)
    rows: list[dict[str, Any]] = []
    prepared_cache: dict[int, _PreparedPair] = {}

    def prepared(index: int, selected_sites: tuple[LayerSite, ...]) -> _PreparedPair:
        cached = prepared_cache.get(index)
        if cached is not None:
            return cached
        value = _prepare_pair(
            adapter,
            pairs[index],
            sites=selected_sites,
            candidates=candidates,
        )
        prepared_cache[index] = value
        return value

    if profile in {PROFILE_SMOKE, PROFILE_SWEEP}:
        phase_started_at = time.monotonic()
        for index in range(len(pairs)):
            item = prepared(index, sites)
            rows.extend(_baseline_rows(adapter, item, candidates))
            rows.extend(
                _direct_and_self_rows(
                    adapter,
                    item,
                    sites=sites,
                    candidates=candidates,
                )
            )
            completed = index + 1
            if completed % 10 == 0 or completed == len(pairs):
                _log_pair_progress(
                    profile=profile,
                    phase="sweep",
                    completed=completed,
                    total=len(pairs),
                    started_at=phase_started_at,
                )
        selected_layer = _quick_select_layer(rows, eligible_layers)
        if profile == PROFILE_SMOKE:
            source_indices = _matched_source_indices(pairs, seed=20260731)
            matched = _matched_manifest(pairs, source_indices, seed=20260731)
            atomic_write_json(output_dir / "matched_donor_manifest.json", matched)
            selected_site = sites[layers.index(selected_layer)]
            controls_started_at = time.monotonic()
            for index, source_index in enumerate(source_indices):
                target_item = prepared(index, sites)
                source_item = prepared(source_index, sites)
                rows.extend(
                    _control_rows(
                        adapter,
                        target_item,
                        source_item,
                        site=selected_site,
                        candidates=candidates,
                        seed=20260731,
                    )
                )
                completed = index + 1
                if completed % 10 == 0 or completed == len(source_indices):
                    _log_pair_progress(
                        profile=profile,
                        phase="controls",
                        completed=completed,
                        total=len(source_indices),
                        started_at=controls_started_at,
                    )
            atomic_write_json(
                output_dir / "smoke_control_selection.json",
                {
                    "schema_version": 2,
                    "protocol_id": protocol["protocol_id"],
                    "selected_layer": selected_layer,
                    "selection_rule": "max_pair_mean_T_direct_exact_tie_lowest_layer",
                    "scientific_role": "none_engineering_only",
                },
            )
    elif profile == PROFILE_CONTROLS:
        assert selected_layer is not None
        selected_site = sites[layers.index(selected_layer)]
        selected_sites = (selected_site,)
        source_indices = _matched_source_indices(pairs, seed=20260731)
        matched = _matched_manifest(pairs, source_indices, seed=20260731)
        atomic_write_json(output_dir / "matched_donor_manifest.json", matched)
        controls_started_at = time.monotonic()
        for index, source_index in enumerate(source_indices):
            target_item = prepared(index, selected_sites)
            source_item = prepared(source_index, selected_sites)
            rows.extend(_baseline_rows(adapter, target_item, candidates))
            rows.extend(
                _direct_and_self_rows(
                    adapter,
                    target_item,
                    sites=selected_sites,
                    candidates=candidates,
                )
            )
            rows.extend(
                _control_rows(
                    adapter,
                    target_item,
                    source_item,
                    site=selected_site,
                    candidates=candidates,
                    seed=20260731,
                )
            )
            completed = index + 1
            if completed % 10 == 0 or completed == len(source_indices):
                _log_pair_progress(
                    profile=profile,
                    phase="controls",
                    completed=completed,
                    total=len(source_indices),
                    started_at=controls_started_at,
                )
    else:
        assert profile == PROFILE_SAME_VARIANT
        assert selected_layer is not None
        selected_site = sites[layers.index(selected_layer)]
        selected_sites = (selected_site,)
        source_indices = _matched_source_indices(pairs, seed=20260731)
        matched = _matched_manifest(pairs, source_indices, seed=20260731)
        atomic_write_json(output_dir / "matched_donor_manifest.json", matched)
        controls_started_at = time.monotonic()
        for index, source_index in enumerate(source_indices):
            target_item = prepared(index, selected_sites)
            source_item = prepared(source_index, selected_sites)
            rows.extend(_baseline_rows(adapter, target_item, candidates))
            rows.extend(
                _self_rows(
                    adapter,
                    target_item,
                    site=selected_site,
                    candidates=candidates,
                )
            )
            rows.extend(
                _matched_donor_rows(
                    adapter,
                    target_item,
                    source_item,
                    site=selected_site,
                    candidates=candidates,
                    include_same_variant_donor=True,
                )
            )
            completed = index + 1
            if completed % 10 == 0 or completed == len(source_indices):
                _log_pair_progress(
                    profile=profile,
                    phase="same_variant_controls",
                    completed=completed,
                    total=len(source_indices),
                    started_at=controls_started_at,
                )

    atomic_write_jsonl(output_dir / "predictions.jsonl", rows)
    predictions_sha = sha256_file(output_dir / "predictions.jsonl")
    pair_ids = [_pair_id(pair) for pair in pairs]
    manifest = {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "profile": profile,
        "num_pairs": len(pairs),
        "pair_ids": pair_ids,
        "group_ids": [pair.clean.group_id for pair in pairs],
        "source_pair_order_sha256": registry_evidence["pair_order_sha256"],
        "selected_pair_order_sha256": _pair_order_sha256(pair_ids),
        "selection_manifest_hash": selection["manifest_hash"],
        "prediction_rows": len(rows),
        "predictions_sha256": predictions_sha,
        "layers": layers if profile in {PROFILE_SMOKE, PROFILE_SWEEP} else [selected_layer],
        "winner_eligible_layers": eligible_layers,
        "selected_layer": selected_layer,
        "sweep_winner_sha256": sweep_winner_sha,
        "sweep_run_id": sweep_run_id,
        "sweep_ordered_pair_ids_sha256": sweep_ordered_pair_ids_sha256,
        "matched_donor_manifest_sha256": (
            sha256_file(output_dir / "matched_donor_manifest.json")
            if profile in {PROFILE_SMOKE, PROFILE_CONTROLS, PROFILE_SAME_VARIANT}
            else None
        ),
        "code": environment["git"],
        "model": environment["model"],
        "data": environment["data"],
        "validation_accessed": False,
    }
    if same_variant_config is not None:
        manifest["same_variant_donor"] = same_variant_config
        manifest["source_winner_manifest_sha256"] = sha256_file(
            output_dir / "source_winner_manifest.json"
        )
    atomic_write_json(output_dir / "data_manifest.json", manifest)
    metrics = {
        "schema_version": 2,
        "profile": profile,
        "num_pairs": len(pairs),
        "prediction_rows": len(rows),
        "selected_layer": selected_layer,
        "treatment_counts": {
            treatment: sum(row["treatment"] == treatment for row in rows)
            for treatment in sorted({str(row["treatment"]) for row in rows})
        },
    }
    atomic_write_json(output_dir / "metrics.json", metrics)
    return manifest


def _main_run(args: argparse.Namespace) -> int:
    payload = run_residual_v2(
        protocol_path=args.protocol,
        profile=args.profile,
        output_dir=args.output_dir,
        model_path=args.model,
        data_path=args.data,
        sweep_winner_path=args.sweep_winner,
        same_variant_config_path=args.same_variant_config,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _main_review(args: argparse.Namespace) -> int:
    protocol = load_yaml(args.protocol)
    _validate_protocol(protocol)
    payload = write_residual_v2_review(
        run_root=args.run_root,
        output=args.output,
        protocol=protocol,
        profile=args.profile,
        selected_layer=args.selected_layer,
        protocol_sha256=sha256_file(args.protocol),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("integrity_passed") is True else 1


def _verify_sha256sums(run_root: Path) -> dict[str, Any]:
    sums_path = run_root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ValueError(f"missing prerequisite SHA256SUMS: {sums_path}")
    seen: set[str] = set()
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split("  ", 1)
        if len(pieces) != 2:
            raise ValueError("malformed prerequisite SHA256SUMS")
        expected, relative_text = pieces
        relative = Path(relative_text)
        if (
            len(expected) != 64
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise ValueError("unsafe prerequisite SHA256SUMS entry")
        target = run_root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"prerequisite SHA256 mismatch: {relative_text}")
        seen.add(relative_text)
    excluded = {"SHA256SUMS", "SUCCESS", "FAILED"}
    actual = {
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and path.relative_to(run_root).as_posix() not in excluded
    }
    if not seen or seen != actual:
        raise ValueError("prerequisite SHA256SUMS does not cover all and only run files")
    return {
        "sha256sums_path": str(sums_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "sha256sums_verified_files": len(seen),
    }


def _package_evidence(package: Path) -> dict[str, Any]:
    sidecar = Path(f"{package}.sha256")
    if not package.is_file() or not sidecar.is_file():
        raise ValueError("prerequisite package or sidecar is missing")
    pieces = sidecar.read_text(encoding="utf-8").strip().split("  ", 1)
    package_sha = sha256_file(package)
    if len(pieces) != 2 or pieces[0] != package_sha or pieces[1] != package.name:
        raise ValueError("prerequisite package sidecar mismatch")
    return {
        "package_path": str(package),
        "package_sha256": package_sha,
        "package_sidecar_path": str(sidecar),
        "package_sidecar_sha256": sha256_file(sidecar),
    }


def build_stage_linkage(
    *,
    profile: str,
    protocol_path: Path,
    next_run_id: str,
    next_run_root: Path,
    next_package: Path,
    prerequisite_review: Path | None,
) -> dict[str, Any]:
    protocol = load_yaml(protocol_path)
    _validate_protocol(protocol)
    eligible_layers = {int(value) for value in protocol["layer_selection"]["eligible_layers"]}
    state = git_state()
    commit = state.get("commit")
    if not commit or state.get("dirty") is not False:
        raise ValueError("residual-v3 execution checkout must be clean")
    if (
        next_run_root.exists()
        or next_package.exists()
        or Path(f"{next_package}.sha256").exists()
        or next_run_root.name != next_run_id
        or next_package.name != f"{next_run_id}.tgz"
    ):
        raise ValueError("next residual-v3 run identity is not fresh and exact")
    current_environment = _residual_runtime_environment()
    if (
        not all(current_environment.values())
        or current_environment["RESIDUAL_V2_RUN_ID"] != next_run_id
        or Path(str(current_environment["RESIDUAL_V2_RUN_ROOT"])).resolve()
        != next_run_root.resolve()
        or Path(str(current_environment["RESIDUAL_V2_PACKAGE_PATH"])).resolve()
        != next_package.resolve()
    ):
        raise ValueError("same-Pod and Hugging Face cache environment is incomplete")
    expected_prior = {
        PROFILE_SMOKE: None,
        PROFILE_SWEEP: PROFILE_SMOKE,
        PROFILE_CONTROLS: PROFILE_SWEEP,
        PROFILE_SAME_VARIANT: None,
    }[profile]
    prerequisite: dict[str, Any] | None = None
    if expected_prior is None:
        if prerequisite_review is not None:
            raise ValueError(f"{profile} must not have a prerequisite review")
    else:
        if (
            prerequisite_review is None
            or not prerequisite_review.is_file()
            or prerequisite_review.name != "machine_review.json"
        ):
            raise ValueError(f"{profile} requires a finalized {expected_prior} review")
        prior_root = prerequisite_review.resolve().parent
        review = json.loads(prerequisite_review.read_text(encoding="utf-8"))
        manifest_path = prior_root / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("prerequisite run manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        prior_environment = manifest.get("environment")
        prior_root_environment_path = prior_root / "environment_manifest.json"
        prior_patch_environment_path = prior_root / "patch/environment.json"
        if not prior_root_environment_path.is_file() or not prior_patch_environment_path.is_file():
            raise ValueError("prerequisite environment manifests are missing")
        prior_root_environment = json.loads(prior_root_environment_path.read_text(encoding="utf-8"))
        prior_patch_environment = json.loads(
            prior_patch_environment_path.read_text(encoding="utf-8")
        )
        if (
            review.get("integrity_passed") is not True
            or review.get("completed") is not True
            or review.get("profile") != expected_prior
            or review.get("protocol_id") != protocol["protocol_id"]
            or review.get("run_id") != run_id
            or manifest.get("success") is not True
            or manifest.get("exit_code") != 0
            or manifest.get("profile") != expected_prior
            or run_id != prior_root.name
            or Path(str(manifest.get("run_root"))).resolve() != prior_root.resolve()
            or manifest.get("review") != review
            or manifest.get("code_commit") != commit
            or not isinstance(prior_environment, Mapping)
            or not _shared_runtime_matches(prior_environment, current_environment)
            or prior_root_environment.get("residual_v2_runtime") != prior_environment
            or prior_patch_environment.get("residual_v2_runtime") != prior_environment
            or prior_root_environment.get("git") != prior_patch_environment.get("git")
            or prior_root_environment.get("git", {}).get("commit") != commit
            or prior_root_environment.get("git", {}).get("dirty") is not False
            or not (prior_root / "SUCCESS").is_file()
            or (prior_root / "FAILED").exists()
        ):
            raise ValueError("prerequisite review/run identity is not an integrity PASS")
        sums = _verify_sha256sums(prior_root)
        package_value = manifest.get("package_path")
        if not package_value:
            raise ValueError("prerequisite run manifest has no package path")
        package = _package_evidence(Path(str(package_value)).resolve())
        if (
            Path(str(package_value)).resolve().name != f"{run_id}.tgz"
            or Path(str(package_value)).resolve() != Path(str(package["package_path"])).resolve()
        ):
            raise ValueError("prerequisite package identity is invalid")
        prerequisite = {
            "profile": expected_prior,
            "run_id": manifest.get("run_id"),
            "run_root": str(prior_root),
            "review_path": str(prerequisite_review.resolve()),
            "review_sha256": sha256_file(prerequisite_review),
            "run_manifest_sha256": sha256_file(manifest_path),
            "integrity_passed": True,
            "environment": prior_environment,
            **sums,
            **package,
        }
        if expected_prior == PROFILE_SWEEP:
            winner_path = prior_root / "winner_manifest.json"
            if not winner_path.is_file():
                raise ValueError("sweep prerequisite winner manifest is missing")
            winner = json.loads(winner_path.read_text(encoding="utf-8"))
            if (
                winner.get("profile") != PROFILE_SWEEP
                or winner.get("protocol_id") != protocol["protocol_id"]
                or winner.get("selection_role") != "discovery_only"
                or winner.get("sweep_run_id") != run_id
                or winner.get("ordered_pair_ids_sha256")
                != protocol["source_contract"]["pair_registry"]["ordered_pair_ids_sha256"]
                or winner.get("sweep_predictions_sha256") != review.get("predictions_sha256")
                or int(winner.get("selected_layer", -1)) not in eligible_layers
            ):
                raise ValueError("sweep winner manifest is invalid")
            prerequisite.update(
                {
                    "winner_manifest_path": str(winner_path),
                    "winner_manifest_sha256": sha256_file(winner_path),
                    "selected_layer": int(winner["selected_layer"]),
                    "sweep_run_id": str(winner["sweep_run_id"]),
                    "ordered_pair_ids_sha256": str(winner["ordered_pair_ids_sha256"]),
                }
            )
    return {
        "schema_version": 2,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "profile": profile,
        "run_id": next_run_id,
        "run_root": str(next_run_root),
        "package_path": str(next_package),
        "code_commit": commit,
        "environment": current_environment,
        "prerequisite": prerequisite,
        "validation_accessed": False,
    }


def _main_link(args: argparse.Namespace) -> int:
    payload = build_stage_linkage(
        profile=args.profile,
        protocol_path=args.protocol,
        next_run_id=args.next_run_id,
        next_run_root=args.next_run_root,
        next_package=args.next_package,
        prerequisite_review=args.prerequisite_review,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite linkage output: {args.output}")
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--profile",
        required=True,
        choices=(PROFILE_SMOKE, PROFILE_SWEEP, PROFILE_CONTROLS, PROFILE_SAME_VARIANT),
    )
    run_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    run_parser.add_argument("--sweep-winner", type=Path)
    run_parser.add_argument(
        "--same-variant-config",
        type=Path,
        help="explicitly opt in to a versioned exploratory same-variant donor config",
    )
    run_parser.set_defaults(handler=_main_run)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument(
        "--profile",
        required=True,
        choices=(PROFILE_SMOKE, PROFILE_SWEEP, PROFILE_CONTROLS, PROFILE_SAME_VARIANT),
    )
    review_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    review_parser.add_argument("--run-root", type=Path, required=True)
    review_parser.add_argument("--output", type=Path, required=True)
    review_parser.add_argument("--selected-layer", type=int)
    review_parser.set_defaults(handler=_main_review)

    link_parser = subparsers.add_parser("link")
    link_parser.add_argument(
        "--profile",
        required=True,
        choices=(PROFILE_SMOKE, PROFILE_SWEEP, PROFILE_CONTROLS, PROFILE_SAME_VARIANT),
    )
    link_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    link_parser.add_argument("--next-run-id", required=True)
    link_parser.add_argument("--next-run-root", type=Path, required=True)
    link_parser.add_argument("--next-package", type=Path, required=True)
    link_parser.add_argument("--prerequisite-review", type=Path)
    link_parser.add_argument("--output", type=Path, required=True)
    link_parser.set_defaults(handler=_main_link)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
