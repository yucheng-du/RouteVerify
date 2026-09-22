"""Same-run donor-role contrasts and last-step-token all-layer output bundles.

This prospective runner intentionally leaves historical patch protocols intact.
Every D cell is freshly scored under one loaded adapter; historical rows can
never supply a missing cell. The donor pool remains fixed when smoke receivers
are limited. Run ``python -m routeverify.donor_contrast --help`` for entry points.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from routeverify.data.prmbench import (
    PRMBenchPair,
    audit_prmbench_example_pairs,
    prmbench_source_problem_components,
)
from routeverify.data.schema import normalize_text, stable_hash, trace_example_from_json
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchSpec,
    site_key,
)

ROLES = ("clean", "corrupt")
CANDIDATES = ["CORRECT", "INCORRECT"]
KINDS = ("router_logits", "moe_output", "post_moe_residual")


class DonorBundleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Path
    data: Path
    split_manifest: Path
    split_partition: Literal["discovery"] = "discovery"
    selection_manifest: Path = Path(
        "docs/results/gpt_oss_full_20260809/full_selection_manifest.json"
    )
    source_component_manifest: Path = Path(
        "docs/results/qwen3_cpu_corrections_v1_20260808/source_problem_manifest.json"
    )
    expected_pairs: int = Field(default=290, ge=2)
    anchor_layer: int = Field(ge=0)
    expected_bundle_sites: int = Field(ge=1)
    input_format: Literal["raw", "chat"] = "chat"
    mode: Literal["d", "bundle", "both"] = "both"
    d_sites: list[Literal["router_logits", "moe_output", "post_moe_residual"]] = Field(
        default_factory=lambda: list(KINDS), min_length=1
    )
    bundle_kind: Literal["moe_output", "post_moe_residual"] = "moe_output"
    seed: int = Field(default=0, ge=0)
    output_dir: Path
    identity_tolerance: float = Field(default=1e-5, ge=0)
    extended_diagnostics: bool = False


def _pair_id(pair: PRMBenchPair) -> str:
    return str(pair.clean.pair_id or pair.corrupt.pair_id or pair.clean.sample_id)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def matching_manifest(
    pairs: list[PRMBenchPair], *, seed: int, source_components: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    """Match by type, first-error distance, seeded hash; exclude source components."""
    components = dict(source_components or prmbench_source_problem_components(pairs))
    if len({_pair_id(pair) for pair in pairs}) != len(pairs):
        raise ValueError("duplicate pair IDs in donor pool")
    if any(pair.clean.group_id not in components for pair in pairs):
        raise ValueError("source component is missing for a donor-pool pair")
    edges = []
    for current in pairs:
        if current.corrupt.first_error is None:
            raise ValueError("D requires an eligible corrupt first-error boundary")
        current_component = components[current.clean.group_id]
        candidates = [
            pair
            for pair in pairs
            if _pair_id(pair) != _pair_id(current)
            and components[pair.clean.group_id] != current_component
        ]
        if not candidates:
            raise ValueError(f"no cross-source donor available for {_pair_id(current)}")

        def rank(candidate: PRMBenchPair, current: PRMBenchPair = current) -> tuple[int, int, str]:
            if candidate.corrupt.first_error is None:
                raise ValueError("donor has no first-error boundary")
            return (
                int(candidate.corrupt.error_type != current.corrupt.error_type),
                abs(candidate.corrupt.first_error - current.corrupt.first_error),
                stable_hash(
                    {"seed": seed, "current": _pair_id(current), "candidate": _pair_id(candidate)}
                ),
            )

        donor = min(candidates, key=rank)
        mismatch, distance, tie = rank(donor)
        edges.append(
            {
                "pair_id": _pair_id(current),
                "donor_pair_id": _pair_id(donor),
                "receiver_group_id": current.clean.group_id,
                "donor_group_id": donor.clean.group_id,
                "receiver_source_component": current_component,
                "donor_source_component": components[donor.clean.group_id],
                "same_source_component": False,
                "same_error_type": mismatch == 0,
                "error_position_distance": distance,
                "seed_tie_hash": tie,
                "receiver_error_type": current.corrupt.error_type,
                "donor_error_type": donor.corrupt.error_type,
                "receiver_first_error": current.corrupt.first_error,
                "donor_first_error": donor.corrupt.first_error,
                "donor_sample_ids": {role: getattr(donor, role).sample_id for role in ROLES},
                "donor_content_hashes": {role: getattr(donor, role).content_hash for role in ROLES},
            }
        )
    return edges


def planned_counts(
    n_pairs: int, *, n_d_sites: int = 3, mode: str = "both", extended_diagnostics: bool = False
) -> dict[str, int]:
    """Count score rows, two candidate forwards per row, and uncached captures."""
    if mode not in {"d", "bundle", "both"}:
        raise ValueError("mode must be d, bundle or both")
    d_primary = 4 * n_d_sites * n_pairs if mode in {"d", "both"} else 0
    d_identity = (
        (4 if extended_diagnostics else 2) * n_d_sites * n_pairs if mode in {"d", "both"} else 0
    )
    bundle = (12 if extended_diagnostics else 8) * n_pairs if mode in {"bundle", "both"} else 0
    baseline = 4 * n_pairs
    rows = d_primary + d_identity + bundle + baseline
    return {
        "receiver_pairs": n_pairs,
        "d_primary_rows": d_primary,
        "d_primary_scoring_forwards": d_primary * 2,
        "d_identity_rows": d_identity,
        "bundle_rows_including_controls": bundle,
        "shared_baseline_rerun_rows": baseline,
        "scoring_rows": rows,
        "scoring_forwards": rows * 2,
        "capture_forwards": 8 * n_pairs,
        "total_model_forwards": rows * 2 + 8 * n_pairs,
    }


@dataclass
class _Captured:
    pair_id: str
    role: str
    prompt: str
    boundary: int
    token: int
    identity: dict[str, Any]
    captures: dict[str, Any]


def _capture_member(
    adapter: Any,
    pair: PRMBenchPair,
    role: str,
    sites: tuple[LayerSite, ...],
    *,
    input_format: str,
    component: str,
) -> _Captured:
    from routeverify.cli import _render_and_boundary_token

    example = getattr(pair, role)
    boundary = pair.corrupt.first_error
    if boundary is None:
        raise ValueError("pair has no error boundary")
    render = _render_and_boundary_token(
        adapter,
        example,
        boundary_step=boundary,
        prompt_mode="step_binary",
        input_format=input_format,
    )
    candidate_inputs = {}
    for candidate in CANDIDATES:
        inputs, start, end, targets = adapter._encode_prompt_candidate(
            render.raw_render.prompt, candidate
        )
        # This private adapter API is already shared by all seven candidate scorers.
        ids = inputs["input_ids"].detach().cpu().tolist()
        if not 0 <= render.boundary_token < start:
            raise ValueError("patch boundary must precede the scored candidate span")
        candidate_inputs[candidate] = {
            "input_ids_sha256": stable_hash(ids, length=64),
            "input_shape": list(inputs["input_ids"].shape),
            "target_token_start": int(start),
            "target_token_end": int(end),
            "target_ids": [int(token) for token in targets],
        }
    identity = {
        "sample_id": example.sample_id,
        "content_hash": example.content_hash,
        "source_component": component,
        "input_sha256": stable_hash(candidate_inputs, length=64),
        "candidate_inputs": candidate_inputs,
        "boundary_step": boundary,
        "token_position": render.boundary_token,
        "prompt_provenance": render.provenance,
    }
    captures = adapter.capture_candidate_activations(
        render.raw_render.prompt,
        CANDIDATES,
        CaptureSpec(sites=sites, token_positions=(render.boundary_token,), dtype="float32"),
    )
    expected = {site_key(site) for site in sites}
    if set(captures) != set(CANDIDATES):
        raise ValueError("capture must contain both candidate-conditioned inputs")
    for capture in captures.values():
        if not expected.issubset(capture.arrays):
            raise ValueError("capture is missing selected interfaces")
        if any(not np.isfinite(_numpy(capture.arrays[key])).all() for key in expected):
            raise ValueError("nonfinite captured activation")
    return _Captured(
        _pair_id(pair),
        role,
        render.raw_render.prompt,
        boundary,
        render.boundary_token,
        identity,
        captures,
    )


def _bundles(
    receiver: _Captured, donor: _Captured, sites: tuple[LayerSite, ...], *, control: str, seed: int
) -> tuple[dict[str, PatchBundle], dict[str, Any]]:
    from routeverify.cli import _norm_matched_random_array, _norm_matched_random_delta_replacement

    bundles = {}
    replacement_metrics = {}
    for candidate in CANDIDATES:
        patches = []
        metrics = {}
        for site in sites:
            key = site_key(site)
            source = _numpy(donor.captures[candidate].arrays[key])
            target = _numpy(receiver.captures[candidate].arrays[key])
            if source.shape != target.shape:
                raise ValueError(f"receiver/donor activation shapes differ at {key}")
            salt = f"{receiver.pair_id}:{receiver.role}:{donor.pair_id}:{donor.role}:{key}:{candidate}:{control}"
            if control == "alpha_zero_identity":
                replacement = target
            elif control == "norm_random":
                replacement = _norm_matched_random_array(source, seed=seed, salt=salt)
            elif control == "norm_random_delta":
                replacement = _norm_matched_random_delta_replacement(
                    target, source, seed=seed, salt=salt
                )
            else:
                replacement = source
            metrics[key] = {
                "shape": list(target.shape),
                "donor_norm": float(np.linalg.norm(source)),
                "receiver_norm": float(np.linalg.norm(target)),
                "donor_receiver_delta_norm": float(np.linalg.norm(source - target)),
                "replacement_norm": float(np.linalg.norm(replacement)),
                "applied_delta_norm": float(np.linalg.norm(replacement - target)),
            }
            patches.append(
                PatchSpec(site=site, token_position=receiver.token, replacement=replacement)
            )
        bundles[candidate] = PatchBundle(tuple(patches), metadata={"control": control})
        replacement_metrics[candidate] = metrics
    return bundles, replacement_metrics


def _check_diagnostics(diagnostics: Mapping[str, Any], sites: tuple[LayerSite, ...]) -> None:
    expected = {site_key(site) for site in sites}
    if set(diagnostics) != set(CANDIDATES):
        raise ValueError("patch diagnostics missing a candidate")
    for candidate, diagnostic in diagnostics.items():
        if (
            set(diagnostic.get("registered_sites", [])) != expected
            or diagnostic.get("registered_site_count") != len(sites)
            or diagnostic.get("all_registered_hooks_fired") is not True
            or any(diagnostic.get("hook_fire_counts", {}).get(key, 0) < 1 for key in expected)
        ):
            raise ValueError(f"incomplete patch hook coverage for {candidate}")
        # Router dispatch provenance is retained verbatim. Output bundles do not
        # inherit router/top-k criteria; zero effects do not fail technical gates.


def _row(
    run_id: str,
    receiver: _Captured,
    donor: _Captured | None,
    *,
    treatment: str,
    site: str | None,
    scores: Any,
    panel: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from routeverify.cli import _score_row

    if set(scores) != set(CANDIDATES):
        raise ValueError("both candidate scores are required")
    for candidate, score in scores.items():
        expected_tokens = len(receiver.identity["candidate_inputs"][candidate]["target_ids"])
        if (
            score.token_count != expected_tokens
            or score.token_count < 1
            or not np.isfinite([score.mean_logprob, score.total_logprob]).all()
            or not np.isclose(score.total_logprob, score.mean_logprob * score.token_count)
        ):
            raise ValueError("invalid candidate score or changed target tokenization")
    payload = _score_row(
        pair_id=receiver.pair_id,
        receiver_variant=receiver.role,
        donor_variant=None if donor is None else donor.role,
        treatment=treatment,
        site=site,
        scores=scores,
        boundary_step=receiver.boundary,
        token_position=receiver.token,
        metadata={
            "panel": panel,
            "receiver_identity": receiver.identity,
            "donor_identity": None if donor is None else donor.identity,
            **(metadata or {}),
        },
    )
    payload.update(
        {
            "run_id": run_id,
            "donor_pair_id": None if donor is None else donor.pair_id,
            "source_component": receiver.identity["source_component"],
            "input_provenance": receiver.identity["prompt_provenance"],
        }
    )
    return payload


def _validate_identity(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    baselines = {
        (row["pair_id"], row["receiver_variant"]): row
        for row in rows
        if row["treatment"] == "unpatched"
    }
    maximum = 0.0
    tested = 0
    for row in rows:
        if row["treatment"] not in {"unpatched_rerun", "self_patch", "alpha_zero_identity"}:
            continue
        baseline = baselines[(row["pair_id"], row["receiver_variant"])]
        for candidate in CANDIDATES:
            for field in ("mean_logprob", "total_logprob"):
                delta = abs(row["scores"][candidate][field] - baseline["scores"][candidate][field])
                maximum = max(maximum, delta)
        tested += 1
    if maximum > tolerance:
        raise ValueError(f"identity candidate-score delta {maximum:.6g} exceeds {tolerance:.6g}")
    return {
        "passed": True,
        "checked_rows": tested,
        "max_candidate_score_delta": maximum,
        "tolerance": tolerance,
        "criteria": "both candidates, mean and total logprob",
    }


def analyze_run(
    output_dir: Path, *, n_resamples: int = 1000, seed: int | None = None
) -> dict[str, Any]:
    from routeverify.analysis.donor_contrast import analyze_bundle_rows, analyze_donor_rows

    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text().splitlines()
        if line
    ]
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(rows) != manifest["expected_counts"]["scoring_rows"]:
            raise ValueError("saved run is incomplete: score row count differs from run contract")
        if {row["run_id"] for row in rows} != {manifest["run_id"]}:
            raise ValueError("saved prediction run IDs differ from run manifest")
        if seed is None:
            seed = int(manifest["seed"])
    seed = 0 if seed is None else seed
    d_rows = [row for row in rows if row["treatment"] == "donor_role"]
    results = {}
    if d_rows:
        results["d"] = analyze_donor_rows(d_rows, n_resamples=n_resamples, seed=seed)
        _write_json(output_dir / "donor_contrast.json", results["d"])
        _write_json(
            output_dir / "donor_contrast_component.json",
            analyze_donor_rows(d_rows, n_resamples=n_resamples, seed=seed, cluster_by_source=True),
        )
    bundle_rows = [row for row in rows if row["metadata"]["panel"] in {"shared", "bundle"}]
    if any(row["metadata"]["panel"] == "bundle" for row in bundle_rows):
        results["bundle"] = analyze_bundle_rows(bundle_rows, n_resamples=n_resamples, seed=seed)
        _write_json(output_dir / "bundle_summary.json", results["bundle"])
    return results


def run_donor_bundle(
    adapter: Any,
    pairs: list[PRMBenchPair],
    *,
    d_sites: tuple[LayerSite, ...],
    bundle_sites: tuple[LayerSite, ...],
    output_dir: Path,
    mode: str = "both",
    seed: int = 0,
    input_format: str = "chat",
    identity_tolerance: float = 1e-5,
    source_components: Mapping[str, str] | None = None,
    receiver_limit: int | None = None,
    n_resamples: int = 1000,
    extended_diagnostics: bool = False,
    run_metadata: Mapping[str, Any] | None = None,
    runtime_label: str = "adapter_supplied_no_checkpoint_assertion",
) -> dict[str, Any]:
    """Run all four D cells and complete bundle controls using one loaded adapter.

    This function never loads/reloads a model. All captures use each member's
    own first-error-step final token under its exact candidate-scoring inputs.
    Memory is bounded to the current receiver and one external donor pair.
    """
    if mode not in {"d", "bundle", "both"}:
        raise ValueError("mode must be d, bundle or both")
    if receiver_limit is not None and receiver_limit < 1:
        raise ValueError("receiver_limit must be positive")
    if seed < 0 or n_resamples < 2:
        raise ValueError("seed must be nonnegative and n_resamples at least two")
    if identity_tolerance < 0 or not np.isfinite(identity_tolerance):
        raise ValueError("identity tolerance must be finite and nonnegative")
    if input_format not in {"raw", "chat"}:
        raise ValueError("input_format must be raw or chat")
    formatter = callable(getattr(adapter, "chat_completion_render", None))
    if (input_format == "chat") != formatter:
        raise ValueError("input_format disagrees with adapter candidate scoring render")
    if mode in {"d", "both"} and (not d_sites or any(site.kind not in KINDS for site in d_sites)):
        raise ValueError("D requires supported single sites")
    if mode in {"bundle", "both"}:
        if not bundle_sites or len({site.kind for site in bundle_sites}) != 1:
            raise ValueError("bundle requires a nonempty homogeneous interface")
        if bundle_sites[0].kind not in {"moe_output", "post_moe_residual"}:
            raise ValueError("this supplemental bundle requires output or residual sites")
    selected_sites = tuple(
        {
            site_key(site): site
            for site in (
                *(d_sites if mode in {"d", "both"} else ()),
                *(bundle_sites if mode in {"bundle", "both"} else ()),
            )
        }.values()
    )
    if len({site_key(site) for site in d_sites}) != len(d_sites):
        raise ValueError("duplicate D sites")
    if len({site_key(site) for site in bundle_sites}) != len(bundle_sites):
        raise ValueError("duplicate bundle sites")
    components = dict(source_components or prmbench_source_problem_components(pairs))
    edges = matching_manifest(pairs, seed=seed, source_components=components)
    by_id = {_pair_id(pair): pair for pair in pairs}
    edge_by_id = {edge["pair_id"]: edge for edge in edges}
    receivers = pairs if receiver_limit is None else pairs[:receiver_limit]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (run_metadata or {}).items():
        if filename not in {
            "config.resolved.json",
            "environment.json",
            "selection_manifest.json",
            "model_discovery.json",
        }:
            raise ValueError(f"unsupported run metadata file: {filename}")
        _write_json(output_dir / filename, payload)
    run_id = str(uuid.uuid4())
    started = time.monotonic()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "mode": mode,
        "diagnostics_profile": "extended" if extended_diagnostics else "core",
        "adapter_alias": getattr(adapter, "alias", None),
        "adapter_model_id": getattr(adapter, "model_id", None),
        "runtime_label": runtime_label,
        "same_loaded_adapter_all_cells": True,
        "historical_score_reuse": False,
        "patch_position": "last_token_of_first_error_step",
        "seed": seed,
        "donor_pool_pairs": len(pairs),
        "receiver_pairs": len(receivers),
        "receiver_pair_ids": [_pair_id(pair) for pair in receivers],
        "d_sites": [site_key(site) for site in d_sites] if mode in {"d", "both"} else [],
        "bundle_sites": [site_key(site) for site in bundle_sites]
        if mode in {"bundle", "both"}
        else [],
        "expected_counts": planned_counts(
            len(receivers),
            n_d_sites=len(d_sites),
            mode=mode,
            extended_diagnostics=extended_diagnostics,
        ),
        "completed_pairs": 0,
        "completed_score_rows": 0,
        "claim_boundary": "D tests donor-role directionality under this construction; positive D is not proof of a semantic correctness code. Bundle results do not gate windows.",
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(
        output_dir / "matched_donor_manifest.json",
        {
            "run_id": run_id,
            "seed": seed,
            "pool_pairs": len(pairs),
            "ranking": ["error_type_mismatch", "first_error_absolute_distance", "seeded_pair_hash"],
            "same_source_excluded": True,
            "edges": edges,
        },
    )
    all_rows = []
    capture_forwards = 0
    try:
        with (output_dir / "predictions.jsonl").open("x") as stream:
            for pair in receivers:
                external_pair = by_id[edge_by_id[_pair_id(pair)]["donor_pair_id"]]
                captured = {}
                for subject in (pair, external_pair):
                    for role in ROLES:
                        captured[(_pair_id(subject), role)] = _capture_member(
                            adapter,
                            subject,
                            role,
                            selected_sites,
                            input_format=input_format,
                            component=components[subject.clean.group_id],
                        )
                        capture_forwards += 2
                local = {role: captured[(_pair_id(pair), role)] for role in ROLES}
                external = {role: captured[(_pair_id(external_pair), role)] for role in ROLES}
                pair_rows = []

                def score_patch(
                    receiver: _Captured,
                    donor: _Captured,
                    sites: tuple[LayerSite, ...],
                    treatment: str,
                    panel: str,
                    pair_rows: list[dict[str, Any]] = pair_rows,
                ) -> None:
                    bundles, metrics = _bundles(
                        receiver, donor, sites, control=treatment, seed=seed
                    )
                    scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
                        receiver.prompt, CANDIDATES, bundles_by_candidate=bundles
                    )
                    _check_diagnostics(diagnostics, sites)
                    label = (
                        site_key(sites[0])
                        if panel == "d"
                        else f"bundle.{sites[0].kind}[{len(sites)}]"
                    )
                    pair_rows.append(
                        _row(
                            run_id,
                            receiver,
                            donor,
                            treatment=treatment,
                            site=label,
                            scores=scores,
                            panel=panel,
                            metadata={
                                "patch_mode": "single_site_sweep"
                                if panel == "d"
                                else "simultaneous_bundle",
                                "patch_sites": [site_key(site) for site in sites],
                                "patch_diagnostics": diagnostics,
                                "replacement_metrics": metrics,
                                "alpha": 0.0 if treatment == "alpha_zero_identity" else 1.0,
                            },
                        )
                    )

                for role in ROLES:
                    receiver = local[role]
                    for treatment in ("unpatched", "unpatched_rerun"):
                        pair_rows.append(
                            _row(
                                run_id,
                                receiver,
                                None,
                                treatment=treatment,
                                site=None,
                                scores=adapter.score_candidates(receiver.prompt, CANDIDATES),
                                panel="shared",
                            )
                        )
                if mode in {"d", "both"}:
                    for site in d_sites:
                        for role in ROLES:
                            receiver = local[role]
                            for donor_role in ROLES:
                                score_patch(
                                    receiver, external[donor_role], (site,), "donor_role", "d"
                                )
                            score_patch(receiver, receiver, (site,), "self_patch", "d")
                            if extended_diagnostics:
                                score_patch(
                                    receiver,
                                    external["corrupt"],
                                    (site,),
                                    "alpha_zero_identity",
                                    "d",
                                )
                if mode in {"bundle", "both"}:
                    for role, other in (("clean", "corrupt"), ("corrupt", "clean")):
                        receiver, exact_donor = local[role], local[other]
                        direction = f"{other}_to_{role}"
                        score_patch(receiver, exact_donor, bundle_sites, direction, "bundle")
                        score_patch(receiver, receiver, bundle_sites, "self_patch", "bundle")
                        if extended_diagnostics:
                            score_patch(
                                receiver, exact_donor, bundle_sites, "alpha_zero_identity", "bundle"
                            )
                        score_patch(
                            receiver,
                            external[other],
                            bundle_sites,
                            f"matched_{direction}",
                            "bundle",
                        )
                        if extended_diagnostics:
                            score_patch(
                                receiver, exact_donor, bundle_sites, "norm_random", "bundle"
                            )
                        score_patch(
                            receiver, exact_donor, bundle_sites, "norm_random_delta", "bundle"
                        )
                # Preserve evidence even when a numerical identity gate fails.
                for row in pair_rows:
                    row["metadata"]["diagnostics_profile"] = manifest["diagnostics_profile"]
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                _validate_identity(pair_rows, identity_tolerance)
                all_rows.extend(pair_rows)
                manifest.update(
                    completed_pairs=manifest["completed_pairs"] + 1,
                    completed_score_rows=len(all_rows),
                    capture_forwards=capture_forwards,
                )
                _write_json(output_dir / "run_manifest.json", manifest)
                print(
                    f"{manifest['completed_pairs']}/{len(receivers)} pairs, {len(all_rows)} score rows",
                    flush=True,
                )
        manifest["identity"] = _validate_identity(all_rows, identity_tolerance)
        manifest["actual_counts"] = {
            "scoring_rows": len(all_rows),
            "scoring_forwards": len(all_rows) * 2,
            "capture_forwards": capture_forwards,
            "total_model_forwards": len(all_rows) * 2 + capture_forwards,
            "rows_by_panel_treatment": dict(
                Counter(f"{row['metadata']['panel']}:{row['treatment']}" for row in all_rows)
            ),
        }
        expected = manifest["expected_counts"]
        if (
            len(all_rows) != expected["scoring_rows"]
            or capture_forwards != expected["capture_forwards"]
        ):
            raise RuntimeError("observed row/capture counts differ from run plan")
        analyze_run(output_dir, n_resamples=n_resamples, seed=seed)
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic() - started,
        )
        _write_json(output_dir / "run_manifest.json", manifest)
        raise
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def load_discovery_pairs(
    config: DonorBundleConfig, *, pairs_jsonl: Path | None = None
) -> tuple[list[PRMBenchPair], dict[str, str], dict[str, Any]]:
    """Verify full discovery membership and content before limiting receivers/loading weights."""
    from routeverify.cli import _load_prmbench_selection_from_hf, _sha256_file
    from routeverify.config import load_data_config
    from routeverify.data.split_gate import select_partition_group_ids

    selected, _, _ = select_partition_group_ids(config.split_manifest, partition="discovery")
    selection = json.loads(config.selection_manifest.read_text())
    pinned = {row["group_id"]: row for row in selection["pairs"]}
    if len(selected) != config.expected_pairs or set(pinned) != set(selected):
        raise ValueError("selection manifest must pin the full expected discovery pool")
    if pairs_jsonl is None:
        loaded = _load_prmbench_selection_from_hf(
            load_data_config(config.data), limit=None, seed=config.seed
        )
        pairs = loaded.pairs
        source = loaded.source_provenance
    else:
        examples = []
        for line in pairs_jsonl.read_text().splitlines():
            if line:
                row = json.loads(line)
                if row.get("group_id") in pinned:
                    examples.append(trace_example_from_json(row))
        pairs, audit = audit_prmbench_example_pairs(examples)
        if len(examples) != 2 * len(pairs):
            raise ValueError(
                f"local discovery rows contain excluded or duplicate examples: {audit}"
            )
        source = {
            "source_type": "local_normalized_discovery",
            "path": str(pairs_jsonl),
            "sha256": _sha256_file(pairs_jsonl),
        }
    by_group = {pair.clean.group_id: pair for pair in pairs}
    missing = set(selected) - set(by_group)
    if missing:
        raise ValueError(
            f"full discovery pool missing {len(missing)} pairs; local 20-pair pilot is insufficient"
        )
    pairs = [by_group[group] for group in selected]
    for pair in pairs:
        reference = pinned[pair.clean.group_id]
        for role in ROLES:
            example = getattr(pair, role)
            if (
                example.content_hash != reference[f"{role}_content_hash"]
                or example.sample_id != reference[f"{role}_sample_id"]
            ):
                raise ValueError(f"discovery input identity mismatch for {example.sample_id}")
    component_payload = json.loads(config.source_component_manifest.read_text())
    component_rows = {
        row["pair_group_id"]: row
        for row in component_payload["pairs"]
        if row["pair_group_id"] in pinned
    }
    components = {}
    for pair in pairs:
        row = component_rows[pair.clean.group_id]
        if (
            row["historical_partition"] != "discovery"
            or row["source_idx"] != pair.clean.metadata.get("source_idx")
            or row["error_type"] != pair.corrupt.error_type
            or row["normalized_problem_hash"]
            != stable_hash(normalize_text(pair.clean.problem), length=64)
        ):
            raise ValueError("source component manifest identity mismatch")
        components[pair.clean.group_id] = row["source_problem_id"]
    provenance = {
        "source": source,
        "selected_pair_count": len(pairs),
        "pairs": selection["pairs"],
        "selection_manifest_sha256": _sha256_file(config.selection_manifest),
        "split_manifest_sha256": _sha256_file(config.split_manifest),
        "source_component_manifest_sha256": _sha256_file(config.source_component_manifest),
        "validation_experiment_accessed": False,
        "full_pool_retained_for_smoke_matching": True,
    }
    return pairs, components, provenance


def main(argv: list[str] | None = None) -> None:
    from routeverify.cli import _capture_site_contract, _sha256_file
    from routeverify.config import environment_snapshot, load_model_config, load_yaml
    from routeverify.models.registry import create_adapter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("d", "bundle", "both"))
    parser.add_argument(
        "--limit", type=int, help="Receiver limit only; full discovery donor pool retained"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pairs-jsonl", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and report counts; no data or weights loaded",
    )
    parser.add_argument(
        "--analyze", type=Path, help="Reanalyze an existing predictions.jsonl run directory"
    )
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument(
        "--extended-diagnostics",
        action="store_true",
        help="Opt in to alpha=0 and absolute-state norm controls (e.g. technical smoke)",
    )
    args = parser.parse_args(argv)
    if args.n_resamples < 2 or (args.limit is not None and args.limit < 1):
        parser.error("limit must be positive and n-resamples at least two")
    if args.analyze is not None:
        analyze_run(args.analyze, n_resamples=args.n_resamples)
        return
    if args.config is None:
        parser.error("--config is required unless --analyze is used")
    config = DonorBundleConfig.model_validate(load_yaml(args.config))
    if args.mode:
        config.mode = args.mode
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.extended_diagnostics:
        config.extended_diagnostics = True
    if len(config.d_sites) != len(set(config.d_sites)):
        parser.error("duplicate D site kinds")
    model = load_model_config(config.model)
    plan = {
        "config": config.model_dump(mode="json"),
        "counts": planned_counts(
            min(args.limit or config.expected_pairs, config.expected_pairs),
            n_d_sites=len(config.d_sites),
            mode=config.mode,
            extended_diagnostics=config.extended_diagnostics,
        ),
        "checkpoint_forward_status": "not_run",
        "weights_loaded": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError(f"refusing nonempty output directory: {config.output_dir}")
    pairs, components, provenance = load_discovery_pairs(config, pairs_jsonl=args.pairs_jsonl)
    # The CLI is the sole load call. The runner and analyzer never reload weights.
    adapter = create_adapter(model.alias)
    model = model.model_copy(update={"load_weights": True})
    adapter.load(model)
    available = tuple(adapter.discover_layers())
    d_sites = tuple(
        site
        for site in available
        if site.layer_index == config.anchor_layer and site.kind in config.d_sites
    )
    bundle_sites = tuple(site for site in available if site.kind == config.bundle_kind)
    if config.mode in {"d", "both"} and len(d_sites) != len(config.d_sites):
        raise ValueError("adapter did not expose the requested D anchor interfaces")
    if config.mode in {"bundle", "both"} and len(bundle_sites) != config.expected_bundle_sites:
        raise ValueError("adapter all-layer output site count differs from the model contract")
    actual_receivers = pairs if args.limit is None else pairs[: args.limit]
    provenance["receiver_pair_ids"] = [_pair_id(pair) for pair in actual_receivers]
    provenance["receiver_pair_count"] = len(actual_receivers)
    environment = environment_snapshot(model=model, seed=config.seed)
    environment["capture_site_contract"] = _capture_site_contract(
        adapter, tuple({site_key(site): site for site in (*d_sites, *bundle_sites)}.values())
    )
    environment["runner_source_sha256"] = _sha256_file(Path(__file__))
    manifest = run_donor_bundle(
        adapter,
        pairs,
        d_sites=d_sites,
        bundle_sites=bundle_sites,
        output_dir=config.output_dir,
        mode=config.mode,
        seed=config.seed,
        input_format=config.input_format,
        identity_tolerance=config.identity_tolerance,
        source_components=components,
        receiver_limit=args.limit,
        n_resamples=args.n_resamples,
        extended_diagnostics=config.extended_diagnostics,
        runtime_label="checkpoint_loaded_by_cli",
        run_metadata={
            "selection_manifest.json": provenance,
            "config.resolved.json": {
                **config.model_dump(mode="json"),
                "resolved_model": model.model_dump(mode="json"),
                "command": sys.argv,
                "receiver_limit": args.limit,
                "config_source_sha256": _sha256_file(args.config),
            },
            "environment.json": environment,
            "model_discovery.json": adapter.discovery_report.to_json(),
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
