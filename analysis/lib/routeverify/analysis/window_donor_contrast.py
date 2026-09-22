"""Direct D for full-step, all-router, fixed external-donor interventions.

This is not an analysis of the opposite-role matched control T. Its four rows
keep the receiver fixed while switching roles of the same external donor pair.
Source components, rather than receiver roles or reused donors, are resampled.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from routeverify.analysis.bootstrap import (
    paired_cluster_bootstrap_indices,
    paired_cluster_indices_sha256,
)
from routeverify.analysis.donor_contrast import analyze_donor_rows

_ROLES = ("clean", "corrupt")
_CANDIDATES = ("CORRECT", "INCORRECT")
_ESTIMANDS = (
    "d_clean_mean", "d_corrupt_mean", "d_common_mean",
    "d_clean_total", "d_corrupt_total", "d_common_total",
)
_Panel = tuple[str, str]


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"requires nonempty {name}")
    return value


def _field(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    metadata = row.get("metadata", {})
    nested = metadata.get(key, default) if isinstance(metadata, Mapping) else default
    if key in row and isinstance(metadata, Mapping) and key in metadata and row[key] != nested:
        raise ValueError(f"top-level {key} disagrees with metadata.{key}")
    return row.get(key, nested)


def _role(value: Any) -> str:
    if value == "error":
        return "corrupt"
    if not isinstance(value, str) or value not in _ROLES:
        raise ValueError("receiver and donor variants must be clean, corrupt, or error")
    return value


def _panel(row: Mapping[str, Any]) -> _Panel:
    return (
        _string(_field(row, "contract_id"), "contract_id"),
        _string(_field(row, "execution_version"), "execution_version"),
    )


def _check_candidate_inputs(row: Mapping[str, Any]) -> None:
    """Require actual scoring inputs and coordinate evidence for both roles."""
    metadata = row["metadata"]
    for field in ("receiver_identity", "donor_identity"):
        identity = metadata.get(field)
        if not isinstance(identity, Mapping):
            raise ValueError(f"requires metadata.{field}")
        inputs = identity.get("candidate_inputs")
        if not isinstance(inputs, Mapping) or set(inputs) != set(_CANDIDATES):
            raise ValueError(f"{field} requires both actual candidate_inputs")
        for candidate, payload in inputs.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"{field} candidate input must be a mapping")
            ids = payload.get("input_ids")
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                ids = ids[0]
            targets = payload.get("target_ids")
            if (not isinstance(ids, list) or not ids
                    or any(type(token) is not int or token < 0 for token in ids)
                    or not isinstance(targets, list) or not targets
                    or any(type(token) is not int or token < 0 for token in targets)):
                raise ValueError(f"{field} requires actual input_ids and target_ids")
            start, end = payload.get("target_token_start"), payload.get("target_token_end")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(ids) or ids[start:end] != targets):
                raise ValueError(f"{field} candidate target positions disagree with input_ids")
            _string(payload.get("input_ids_sha256"), f"{field} input_ids_sha256")
            if field == "receiver_identity":
                scores = row.get("scores", {})
                if not isinstance(scores, Mapping) or not isinstance(scores.get(candidate), Mapping):
                    raise ValueError("requires both candidate scores")
                if scores[candidate].get("token_count") != len(targets):
                    raise ValueError("candidate token_count disagrees with actual target_ids")


def _prepare(row: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(row))
    if _field(result, "window") != "full_step":
        raise ValueError("window donor D requires window='full_step'; single-token results cannot be widened")
    if _field(result, "arm") != "all_router" or result.get("site") != "all_router":
        raise ValueError("window donor D requires arm and site 'all_router'")
    alpha = _field(result, "alpha")
    if isinstance(alpha, (bool, str)) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha != 1:
        raise ValueError("full-step all-router D requires alpha=1")
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("requires metadata identities and window provenance")
    sites = _field(result, "target_sites")
    if (not isinstance(sites, (list, tuple)) or not sites
            or any(not isinstance(site, str) or not site.endswith(".router_logits") for site in sites)
            or len(set(sites)) != len(sites)):
        raise ValueError("target_sites must enumerate unique adapter-defined routed sites")
    geometry = _field(result, "geometry")
    if not isinstance(geometry, Mapping) or geometry.get("name", "full_step") != "full_step":
        raise ValueError("requires full_step geometry provenance")
    for key in ("receiver_variant", "donor_variant"):
        result[key] = _role(result.get(key))
    for name in ("receiver_identity", "donor_identity"):
        identity = metadata.get(name)
        if isinstance(identity, dict) and "variant" in identity:
            identity["variant"] = _role(identity["variant"])
    _check_candidate_inputs(result)
    return result


def _source_bootstrap(records: list[dict[str, Any]], *, seed: int, n_resamples: int, alpha: float) -> dict[str, Any]:
    values = np.asarray([[record[key] for key in _ESTIMANDS] for record in records], dtype=float)
    components = sorted({record["source_component"] for record in records})
    members = [[i for i, record in enumerate(records) if record["source_component"] == component]
               for component in components]
    sums = np.asarray([values[member].sum(axis=0) for member in members])
    sizes = np.asarray([len(member) for member in members])
    indices = paired_cluster_bootstrap_indices(len(components), seed=seed, n_resamples=n_resamples)
    replicates = np.empty((n_resamples, len(_ESTIMANDS)))
    chunk_size = max(1, 2_000_000 // (len(components) * len(_ESTIMANDS)))
    for start in range(0, n_resamples, chunk_size):
        draws = indices[start:start + chunk_size]
        replicates[start:start + chunk_size] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    low, high = np.quantile(replicates, [alpha / 2, 1 - alpha / 2], axis=0)
    result = {
        "method": "receiver_source_component_percentile",
        "draw_method": "sample_receiver_components_with_replacement_keep_all_member_pairs",
        "weighting": "receiver-pair weighted, including within each resample",
        "estimands": list(_ESTIMANDS), "rng": "numpy.random.Generator(PCG64)",
        "seed": seed, "n_resamples": n_resamples, "confidence_level": 1 - alpha,
        "n_pairs": len(records), "n_clusters": len(components),
        "source_components": components, "cluster_pair_counts": sizes.tolist(),
        "indices_sha256": paired_cluster_indices_sha256(indices),
        "point": values.mean(axis=0).tolist(), "ci_low": low.tolist(), "ci_high": high.tolist(),
        "bootstrap_standard_error": np.std(replicates, axis=0, ddof=1).tolist(),
        "bootstrap_covariance": np.cov(replicates, rowvar=False, ddof=1).tolist(),
    }
    return result


def _intervals(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        metric: {
            role: {key: result[source][offset + index] for key, source in (
                ("mean", "point"), ("ci_low", "ci_low"), ("ci_high", "ci_high"),
                ("bootstrap_standard_error", "bootstrap_standard_error"))}
            for index, role in enumerate(("clean_receiver", "corrupt_receiver", "common"))
        }
        for offset, metric in ((0, "mean_logprob"), (3, "total_logprob"))
    }


def analyze_window_donor_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    eligibility_rows: Iterable[Mapping[str, Any]] | None = None,
    expected_pair_ids: Iterable[str] | None = None,
    n_resamples: int = 50_000,
    seed: int = 20260908,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Validate and estimate direct D without doubling the receiver-pair N.

    D rows use the existing donor-role CandidateScore schema and must declare
    ``window='full_step'``, ``arm=site='all_router'``, ``alpha=1``, contract_id,
    execution_version and the actual loaded run_id. Identities retain complete
    candidate inputs. ``metadata`` holds geometry and target_sites evidence.
    All four combinations for a pair must share a loaded run; completed pairs
    from resumed runs can pool only within one contract and execution version.

    Optional eligibility rows declare one pair's eligible boolean, component,
    donor pair/component and reason(s). Ineligible pairs must have no D rows;
    eligible pairs must have all four. With expected_pair_ids, every selected
    pair must be measured or explicitly ineligible. Without that manifest, a
    completely absent pair cannot be discovered, and coverage is labelled so.
    Multiple execution versions are separate panels; unversioned eligibility
    can only describe an unambiguous single panel. Controls never enter D.

    The main percentile intervals resample receiver source components, keeping
    all member pairs and their two receiver roles jointly; each replicate is a
    pair-weighted mean. Matching remains fixed. Mean candidate logprob is the
    primary score and total candidate logprob is a sensitivity analysis.
    """
    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    expected = None if expected_pair_ids is None else list(expected_pair_ids)
    if expected is not None:
        if any(not isinstance(pair, str) or not pair for pair in expected) or len(set(expected)) != len(expected):
            raise ValueError("expected_pair_ids must be unique nonempty strings")
    groups: dict[_Panel, list[dict[str, Any]]] = defaultdict(list)
    ignored: Counter[str] = Counter()
    input_count = 0
    for row in rows:
        input_count += 1
        if not isinstance(row, Mapping):
            raise TypeError("score rows must be mappings")
        if row.get("treatment") != "donor_role":
            ignored[str(row.get("treatment", "<missing>"))] += 1
            # Controls also identify an entirely ineligible execution panel.
            if _field(row, "contract_id") is not None and _field(row, "execution_version") is not None:
                groups.setdefault(_panel(row), [])
            continue
        groups[_panel(row)].append(_prepare(row))
    eligibility = [copy.deepcopy(dict(row)) for row in (eligibility_rows or [])]
    for row in eligibility:
        if _field(row, "contract_id") is not None or _field(row, "execution_version") is not None:
            groups.setdefault(_panel(row), [])
    if not groups:
        raise ValueError("no versioned window donor-role execution found")
    eligibility_by_panel: dict[_Panel, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eligibility:
        if _field(row, "contract_id") is None and _field(row, "execution_version") is None:
            if len(groups) != 1:
                raise ValueError("eligibility rows require contract_id/execution_version for multiple panels")
            panel = next(iter(groups))
        else:
            panel = _panel(row)
        pair = _string(row.get("pair_id"), "eligibility pair_id")
        if type(row.get("eligible")) is not bool:
            raise ValueError("eligibility eligible must be a boolean")
        if pair in eligibility_by_panel[panel]:
            raise ValueError(f"duplicate eligibility for {panel}/{pair}")
        component = row.get("source_component", row.get("component"))
        _string(component, "eligibility source component")
        if "source_component" in row and "component" in row and row["source_component"] != row["component"]:
            raise ValueError("eligibility component fields disagree")
        row["source_component"] = component
        if not row["eligible"] and not (row.get("reason") or row.get("reasons") or row.get("reason_codes")):
            raise ValueError("ineligible pairs require a recorded reason")
        eligibility_by_panel[panel][pair] = row
    per_pair: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for panel, panel_rows in sorted(groups.items()):
        loaded_runs: dict[str, str] = {}
        sites: tuple[str, ...] | None = None
        for row in panel_rows:
            pair = _string(row.get("pair_id"), "pair_id")
            run = _string(row.get("run_id"), "run_id")
            if pair in loaded_runs and loaded_runs[pair] != run:
                raise ValueError(f"pair {pair}: four donor-role rows must share one loaded run_id")
            loaded_runs[pair] = run
            row_sites = tuple(sorted(_field(row, "target_sites")))
            if sites is not None and sites != row_sites:
                raise ValueError("mixed all-router target_sites within one execution version")
            sites = row_sites
        normalized = copy.deepcopy(panel_rows)
        for row in normalized:
            # The public D validator checks global pair/role identities, donor
            # assignment, 2x2 completeness, finite scores and both margins.
            row["run_id"] = "window-panel"
        records = [] if not normalized else analyze_donor_rows(normalized, n_resamples=2)["per_pair"]
        evidence = {(row["pair_id"], row["receiver_variant"], row["donor_variant"]): row for row in panel_rows}
        for record in records:
            record.update(contract_id=panel[0], execution_version=panel[1],
                          run_id=loaded_runs[record["pair_id"]], window="full_step", arm="all_router", alpha=1)
            for receiver in _ROLES:
                for donor in _ROLES:
                    row = evidence[(record["pair_id"], receiver, donor)]
                    # Identity/candidate inputs live once per role above; keep
                    # each measured combination's geometry and native evidence.
                    excluded = {"receiver_identity", "donor_identity"}
                    record["four_combinations"][receiver][donor]["metadata"] = {
                        key: copy.deepcopy(value) for key, value in row["metadata"].items() if key not in excluded
                    }
                    for key in ("geometry", "target_sites", "patch_diagnostics", "actual_donor_positions", "replacement_diagnostics"):
                        if key in row:
                            record["four_combinations"][receiver][donor][key] = copy.deepcopy(row[key])
        measured = {record["pair_id"]: record for record in records}
        declared = eligibility_by_panel[panel]
        for pair, row in declared.items():
            if row["eligible"] != (pair in measured):
                raise ValueError(f"eligibility disagrees with measured complete pair {pair}")
            if pair in measured:
                record = measured[pair]
                for key, expected_value in (("source_component", record["source_component"]),
                                            ("donor_pair_id", record["donor_pair_id"]),
                                            ("donor_component", record["donor_source_component"])):
                    if key in row and row[key] != expected_value:
                        raise ValueError(f"eligibility {key} disagrees for pair {pair}")
        excluded_pairs = sorted(pair for pair, row in declared.items() if not row["eligible"])
        observed = set(measured) | set(excluded_pairs)
        if expected is not None and observed != set(expected):
            raise ValueError(f"selected-pair coverage mismatch: missing={sorted(set(expected) - observed)}, unexpected={sorted(observed - set(expected))}")
        components = sorted({record["source_component"] for record in records})
        summary: dict[str, Any] = {
            "contract_id": panel[0], "execution_version": panel[1],
            "window": "full_step", "arm": "all_router", "site": "all_router", "alpha": 1,
            "loaded_run_ids": sorted(set(loaded_runs.values())), "target_sites": list(sites or ()),
            "N": len(records), "K": len(components), "n_pairs": len(records), "n_source_components": len(components),
            "pair_ids": sorted(measured), "n_selected_pairs": len(observed),
            "selected_pair_coverage_verified": expected is not None,
            "n_ineligible_pairs": len(excluded_pairs), "ineligible_pair_ids": excluded_pairs,
            "eligibility_rows": [declared[pair] for pair in sorted(declared)],
            "n_unique_donor_pairs": len({record["donor_pair_id"] for record in records}),
            "donor_reuse_counts": dict(sorted(Counter(record["donor_pair_id"] for record in records).items())),
            "donor_role_rows": len(records) * 4, "scoring_forwards": len(records) * 8,
            "mean_logprob": None, "total_logprob": None, "source_component_bootstrap": None,
        }
        if records:
            bootstrap = _source_bootstrap(records, seed=seed, n_resamples=n_resamples, alpha=alpha)
            summary.update(_intervals(bootstrap))
            summary["source_component_bootstrap"] = bootstrap
        per_pair.extend(records)
        summaries.append(summary)
    return {
        "schema_version": 1, "analysis": "full_step_all_router_fixed_receiver_external_donor_role_D",
        "definition": "D = m(receiver <- error donor) - m(receiver <- clean donor)",
        "common_definition": "D_common = (D_clean_receiver + D_error_receiver) / 2 within each receiver pair",
        "margin_definition": "candidate logprob(INCORRECT) - candidate logprob(CORRECT)",
        "primary_metric": "mean_logprob", "sensitivity_metric": "total_logprob",
        "interpretation": "External donor-role directionality under this intervention; does not identify a unique correctness code.",
        "bootstrap": {
            "seed": seed, "n_resamples": n_resamples, "confidence_level": 1 - alpha,
            "unit": "receiver_source_component", "weighting": "receiver_pair_equal",
            "receiver_roles_resampled_jointly": True, "donor_assignments_attached_to_receiver_pairs": True,
            "conditional_on_observed_donor_matching": True, "intervals": "marginal two-sided percentile",
        },
        "counts": {"input_rows": input_count, "donor_role_rows": len(per_pair) * 4,
                   "expected_scoring_forwards": len(per_pair) * 8,
                   "scoring_forward_convention": "two candidates per D row; controls and capture forwards excluded",
                   "receiver_pair_panel_cells": len(per_pair), "panels": len(summaries),
                   "ignored_treatments": dict(sorted(ignored.items()))},
        "eligibility_rows": eligibility, "per_pair": per_pair, "summaries": summaries,
    }
