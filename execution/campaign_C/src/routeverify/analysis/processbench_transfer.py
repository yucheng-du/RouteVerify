"""Minimal summaries for ProcessBench natural-boundary transfer rows.

The GPU patch runner uses the repository's historical ``clean``/``corrupt``
row vocabulary internally.  ProcessBench pairs are not counterfactual edits,
so this module deliberately exposes them as ``matched_correct`` and
``first_error`` instead.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from routeverify.analysis.bootstrap import bootstrap_mean_ci

MATCHED_CORRECT = "matched_correct"
FIRST_ERROR = "first_error"
MATCHED_CORRECT_TO_FIRST_ERROR = "matched_correct_to_first_error"
FIRST_ERROR_TO_MATCHED_CORRECT = "first_error_to_matched_correct"

_ROLE_ALIASES = {
    "clean": MATCHED_CORRECT,
    MATCHED_CORRECT: MATCHED_CORRECT,
    "corrupt": FIRST_ERROR,
    FIRST_ERROR: FIRST_ERROR,
}
_DIRECTION_ALIASES = {
    "clean_to_corrupt": MATCHED_CORRECT_TO_FIRST_ERROR,
    MATCHED_CORRECT_TO_FIRST_ERROR: MATCHED_CORRECT_TO_FIRST_ERROR,
    "corrupt_to_clean": FIRST_ERROR_TO_MATCHED_CORRECT,
    FIRST_ERROR_TO_MATCHED_CORRECT: FIRST_ERROR_TO_MATCHED_CORRECT,
}
_EXPECTED_DIRECTION_ROLES = {
    MATCHED_CORRECT_TO_FIRST_ERROR: (MATCHED_CORRECT, FIRST_ERROR),
    FIRST_ERROR_TO_MATCHED_CORRECT: (FIRST_ERROR, MATCHED_CORRECT),
}
_BASELINE_TREATMENTS = {"unpatched", "unpatched_rerun"}
_REQUIRED_TREATMENTS = {*_BASELINE_TREATMENTS, "self_patch", *_DIRECTION_ALIASES}

PRIMARY_METRIC = "mean_logprob_margin_incorrect_minus_correct"
PRIMARY_UNITS = "log-probability per candidate token"


def summarize_processbench_transfer_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    identity_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Summarize the minimal symmetric natural-transfer row contract.

    Each pair must contain unpatched and rerun rows for both roles.  At every
    observed site it must also contain both self-patches and both directional
    transfers.  Other treatment rows are ignored and reported; they cannot
    silently enter the primary estimands.

    Intervals are ordinary marginal percentile bootstraps over pair-level
    effects.  No equivalence test or multiplicity adjustment is performed.
    """

    _validate_options(
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
        identity_tolerance=identity_tolerance,
    )
    materialized = list(rows)
    if not materialized:
        raise ValueError("rows must be non-empty")
    if any(not isinstance(row, Mapping) for row in materialized):
        raise TypeError("every row must be a mapping")

    baselines: dict[tuple[str, str], float] = {}
    reruns: dict[tuple[str, str], float] = {}
    self_patches: dict[tuple[str, str, str], tuple[float, Mapping[str, Any]]] = {}
    transfers: dict[tuple[str, str, str], tuple[float, Mapping[str, Any]]] = {}
    pair_ids: set[str] = set()
    sites: set[str] = set()
    ignored_treatments: Counter[str] = Counter()
    analyzed_row_count = 0

    for row_index, row in enumerate(materialized):
        raw_treatment = str(row.get("treatment", ""))
        if raw_treatment not in _REQUIRED_TREATMENTS:
            ignored_treatments[raw_treatment or "<missing>"] += 1
            continue

        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"row {row_index} has no pair_id")
        receiver = _role(row.get("receiver_variant"), label=f"row {row_index} receiver")
        margin = _mean_logprob_margin(row, row_index=row_index)
        pair_ids.add(pair_id)
        analyzed_row_count += 1

        if raw_treatment in _BASELINE_TREATMENTS:
            if row.get("site") is not None:
                raise ValueError(f"{raw_treatment} row must not name a site")
            if row.get("donor_variant") is not None:
                raise ValueError(f"{raw_treatment} row must not name a donor")
            target = baselines if raw_treatment == "unpatched" else reruns
            _insert_unique(target, (pair_id, receiver), margin, label=raw_treatment)
            continue

        site = str(row.get("site", "")).strip()
        if not site:
            raise ValueError(f"row {row_index} treatment {raw_treatment} has no site")
        sites.add(site)
        donor = _role(row.get("donor_variant"), label=f"row {row_index} donor")

        if raw_treatment == "self_patch":
            if donor != receiver:
                raise ValueError("self_patch donor and receiver roles must match")
            _insert_unique(
                self_patches,
                (pair_id, receiver, site),
                (margin, row),
                label="self_patch",
            )
            continue

        direction = _DIRECTION_ALIASES[raw_treatment]
        expected_donor, expected_receiver = _EXPECTED_DIRECTION_ROLES[direction]
        if (donor, receiver) != (expected_donor, expected_receiver):
            raise ValueError(
                f"{direction} requires donor={expected_donor} and receiver={expected_receiver}"
            )
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("alpha") is not None:
            try:
                dose = float(metadata["alpha"])
            except (TypeError, ValueError) as exc:
                raise ValueError("transfer alpha must be numeric") from exc
            if not math.isclose(dose, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("ProcessBench primary transfer rows must use alpha=1")
        _insert_unique(
            transfers,
            (pair_id, direction, site),
            (margin, row),
            label=direction,
        )

    ordered_pairs = sorted(pair_ids)
    ordered_sites = sorted(sites)
    if not ordered_sites:
        raise ValueError("no transfer sites were found")
    _validate_exact_coverage(
        pair_ids=ordered_pairs,
        sites=ordered_sites,
        baselines=baselines,
        reruns=reruns,
        self_patches=self_patches,
        transfers=transfers,
    )

    effects: dict[str, dict[str, dict[str, Any]]] = {}
    for direction in (
        MATCHED_CORRECT_TO_FIRST_ERROR,
        FIRST_ERROR_TO_MATCHED_CORRECT,
    ):
        donor, receiver = _EXPECTED_DIRECTION_ROLES[direction]
        effects[direction] = {}
        for site in ordered_sites:
            pair_effects = [
                transfers[(pair_id, direction, site)][0] - baselines[(pair_id, receiver)]
                for pair_id in ordered_pairs
            ]
            interval = bootstrap_mean_ci(
                pair_effects,
                seed=seed,
                n_resamples=n_resamples,
                alpha=alpha,
            )
            effects[direction][site] = {
                "direction": direction,
                "donor_role": donor,
                "receiver_role": receiver,
                "site": site,
                "metric": PRIMARY_METRIC,
                "units": PRIMARY_UNITS,
                "n_pairs": len(pair_effects),
                **interval,
            }

    identity = _identity_summary(
        pair_ids=ordered_pairs,
        sites=ordered_sites,
        baselines=baselines,
        reruns=reruns,
        self_patches=self_patches,
        tolerance=identity_tolerance,
    )
    router_dispatch = _router_dispatch_summary(
        pair_ids=ordered_pairs,
        sites=ordered_sites,
        self_patches=self_patches,
        transfers=transfers,
    )
    expected_analyzed_rows = len(ordered_pairs) * (4 + 4 * len(ordered_sites))
    if analyzed_row_count != expected_analyzed_rows:
        raise ValueError(
            "required row count differs from exact coverage: "
            f"observed={analyzed_row_count} expected={expected_analyzed_rows}"
        )

    return {
        "schema_version": 1,
        "analysis_id": "processbench_natural_transfer_v1",
        "role_semantics": {
            "internal_clean": MATCHED_CORRECT,
            "internal_corrupt": FIRST_ERROR,
            "counterfactual_pair_claimed": False,
        },
        "n_pairs": len(ordered_pairs),
        "n_rows": len(materialized),
        "n_analyzed_rows": analyzed_row_count,
        "expected_analyzed_rows": expected_analyzed_rows,
        "sites": ordered_sites,
        "metric": {
            "name": PRIMARY_METRIC,
            "units": PRIMARY_UNITS,
            "definition": "INCORRECT mean_logprob minus CORRECT mean_logprob",
        },
        "bootstrap": {
            "method": "ordinary_pair_percentile",
            "seed": seed,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "multiplicity_adjustment": None,
        },
        "effects": effects,
        "identity": identity,
        "router_native_dispatch": router_dispatch,
        "ignored_treatments": dict(sorted(ignored_treatments.items())),
    }


def _validate_options(
    *, seed: int, n_resamples: int, alpha: float, identity_tolerance: float
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    if not math.isfinite(identity_tolerance) or identity_tolerance < 0.0:
        raise ValueError("identity_tolerance must be finite and non-negative")


def _role(value: Any, *, label: str) -> str:
    raw = str(value)
    if raw not in _ROLE_ALIASES:
        raise ValueError(f"{label} has unsupported role: {raw}")
    return _ROLE_ALIASES[raw]


def _insert_unique(target: dict[Any, Any], key: Any, value: Any, *, label: str) -> None:
    if key in target:
        raise ValueError(f"duplicate {label} row for {key}")
    target[key] = value


def _mean_logprob_margin(row: Mapping[str, Any], *, row_index: int) -> float:
    scores = row.get("scores")
    if scores is not None:
        if not isinstance(scores, Mapping):
            raise ValueError(f"row {row_index} scores must be a mapping")
        correct = _candidate_mean(scores, "CORRECT", row_index=row_index)
        incorrect = _candidate_mean(scores, "INCORRECT", row_index=row_index)
        return incorrect - correct

    value = row.get(
        "margin_incorrect_minus_correct_mean",
        row.get("margin_incorrect_minus_correct"),
    )
    try:
        margin = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index} has no numeric mean-logprob margin") from exc
    if not math.isfinite(margin):
        raise ValueError(f"row {row_index} margin must be finite")
    return margin


def _candidate_mean(scores: Mapping[str, Any], candidate: str, *, row_index: int) -> float:
    payload = scores.get(candidate)
    if not isinstance(payload, Mapping):
        raise ValueError(f"row {row_index} scores.{candidate} must be a mapping")
    try:
        total = float(payload["total_logprob"])
        token_count = int(payload["token_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"row {row_index} scores.{candidate} lacks total_logprob/token_count"
        ) from exc
    if not math.isfinite(total) or token_count <= 0:
        raise ValueError(f"row {row_index} scores.{candidate} has an invalid total or token count")
    computed = total / token_count
    reported = payload.get("mean_logprob", computed)
    try:
        value = float(reported)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_index} scores.{candidate}.mean_logprob is invalid") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_index} scores.{candidate}.mean_logprob must be finite")
    if not math.isclose(value, computed, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(
            f"row {row_index} scores.{candidate}.mean_logprob does not match "
            "total_logprob/token_count"
        )
    return value


def _validate_exact_coverage(
    *,
    pair_ids: list[str],
    sites: list[str],
    baselines: Mapping[tuple[str, str], float],
    reruns: Mapping[tuple[str, str], float],
    self_patches: Mapping[tuple[str, str, str], Any],
    transfers: Mapping[tuple[str, str, str], Any],
) -> None:
    for pair_id in pair_ids:
        for role in (MATCHED_CORRECT, FIRST_ERROR):
            if (pair_id, role) not in baselines:
                raise ValueError(f"missing unpatched row for {(pair_id, role)}")
            if (pair_id, role) not in reruns:
                raise ValueError(f"missing unpatched_rerun row for {(pair_id, role)}")
        for site in sites:
            for role in (MATCHED_CORRECT, FIRST_ERROR):
                if (pair_id, role, site) not in self_patches:
                    raise ValueError(f"missing self_patch row for {(pair_id, role, site)}")
            for direction in (
                MATCHED_CORRECT_TO_FIRST_ERROR,
                FIRST_ERROR_TO_MATCHED_CORRECT,
            ):
                if (pair_id, direction, site) not in transfers:
                    raise ValueError(f"missing transfer row for {(pair_id, direction, site)}")


def _identity_summary(
    *,
    pair_ids: list[str],
    sites: list[str],
    baselines: Mapping[tuple[str, str], float],
    reruns: Mapping[tuple[str, str], float],
    self_patches: Mapping[tuple[str, str, str], tuple[float, Mapping[str, Any]]],
    tolerance: float,
) -> dict[str, Any]:
    rerun_summary: dict[str, dict[str, Any]] = {}
    self_summary: dict[str, dict[str, dict[str, Any]]] = {}
    all_passed = True
    for role in (MATCHED_CORRECT, FIRST_ERROR):
        summary = _absolute_delta_summary(
            [abs(reruns[(pair_id, role)] - baselines[(pair_id, role)]) for pair_id in pair_ids],
            tolerance=tolerance,
        )
        rerun_summary[role] = summary
        all_passed = all_passed and bool(summary["passed"])
    for site in sites:
        self_summary[site] = {}
        for role in (MATCHED_CORRECT, FIRST_ERROR):
            summary = _absolute_delta_summary(
                [
                    abs(self_patches[(pair_id, role, site)][0] - baselines[(pair_id, role)])
                    for pair_id in pair_ids
                ],
                tolerance=tolerance,
            )
            self_summary[site][role] = summary
            all_passed = all_passed and bool(summary["passed"])
    return {
        "tolerance": tolerance,
        "passed": all_passed,
        "unpatched_rerun_by_role": rerun_summary,
        "self_patch_by_site_role": self_summary,
    }


def _absolute_delta_summary(values: list[float], *, tolerance: float) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_abs_delta": sum(values) / len(values),
        "max_abs_delta": max(values),
        "passed": max(values) <= tolerance,
    }


def _router_dispatch_summary(
    *,
    pair_ids: list[str],
    sites: list[str],
    self_patches: Mapping[tuple[str, str, str], tuple[float, Mapping[str, Any]]],
    transfers: Mapping[tuple[str, str, str], tuple[float, Mapping[str, Any]]],
) -> dict[str, Any]:
    router_sites = [site for site in sites if site.endswith(".router_logits")]
    failures: list[dict[str, Any]] = []
    topk_values: list[float] = []
    n_rows = 0
    verified_rows = 0
    for site in router_sites:
        rows_for_site: list[tuple[str, str, Mapping[str, Any]]] = []
        for pair_id in pair_ids:
            for role in (MATCHED_CORRECT, FIRST_ERROR):
                rows_for_site.append(
                    (pair_id, f"self_patch:{role}", self_patches[(pair_id, role, site)][1])
                )
            for direction in (
                MATCHED_CORRECT_TO_FIRST_ERROR,
                FIRST_ERROR_TO_MATCHED_CORRECT,
            ):
                rows_for_site.append((pair_id, direction, transfers[(pair_id, direction, site)][1]))
        for pair_id, treatment, row in rows_for_site:
            n_rows += 1
            valid, reasons, topk = _native_dispatch_row(row, site=site)
            if valid:
                verified_rows += 1
            else:
                failures.append(
                    {
                        "pair_id": pair_id,
                        "treatment": treatment,
                        "site": site,
                        "reasons": reasons,
                    }
                )
            if topk is not None:
                topk_values.append(topk)
    topk_summary: dict[str, Any] = {
        "n_reported": len(topk_values),
        "zero_rows": sum(value == 0.0 for value in topk_values),
        "mean": None,
        "min": None,
        "max": None,
    }
    if topk_values:
        topk_summary.update(
            {
                "mean": sum(topk_values) / len(topk_values),
                "min": min(topk_values),
                "max": max(topk_values),
            }
        )
    return {
        "router_sites": router_sites,
        "required_for_router_patch_rows": True,
        "topk_change_required": False,
        "passed": not failures and bool(router_sites),
        "n_rows": n_rows,
        "verified_rows": verified_rows,
        "failures": failures,
        "topk_changed_rate": topk_summary,
    }


def _native_dispatch_row(
    row: Mapping[str, Any], *, site: str
) -> tuple[bool, list[str], float | None]:
    reasons: list[str] = []
    metadata = row.get("metadata")
    manipulation = metadata.get("manipulation") if isinstance(metadata, Mapping) else None
    if not isinstance(manipulation, Mapping):
        return False, ["missing_manipulation"], None
    required_flags = {
        "candidate_diagnostics_present": True,
        "all_registered_hooks_fired": True,
        "router_dispatch_diagnostics_expected": True,
        "router_dispatch_diagnostics_present": True,
        "router_dispatch_exact_site_coverage": True,
        "all_router_dispatch_verified": True,
    }
    for key, expected in required_flags.items():
        if manipulation.get(key) is not expected:
            reasons.append(key)
    if manipulation.get("candidate_count") != 2:
        reasons.append("candidate_count")
    if manipulation.get("registered_site_count") != 1:
        reasons.append("registered_site_count")
    aggregate_hook_counts = manipulation.get("hook_fire_counts")
    if (
        not isinstance(aggregate_hook_counts, Mapping)
        or set(aggregate_hook_counts) != {site}
        or not _is_positive_count(aggregate_hook_counts.get(site))
    ):
        reasons.append("hook_fire_counts")

    candidates = manipulation.get("candidate_diagnostics")
    if not isinstance(candidates, Mapping) or set(candidates) != {"CORRECT", "INCORRECT"}:
        reasons.append("candidate_diagnostics")
    else:
        for candidate, payload in candidates.items():
            if not isinstance(payload, Mapping):
                reasons.append(f"{candidate}:payload")
                continue
            if payload.get("registered_site_count") != 1:
                reasons.append(f"{candidate}:registered_site_count")
            if set(payload.get("registered_sites", [])) != {site}:
                reasons.append(f"{candidate}:registered_sites")
            if payload.get("all_registered_hooks_fired") is not True:
                reasons.append(f"{candidate}:all_registered_hooks_fired")
            hook_counts = payload.get("hook_fire_counts")
            if (
                not isinstance(hook_counts, Mapping)
                or set(hook_counts) != {site}
                or not _is_positive_count(hook_counts.get(site))
            ):
                reasons.append(f"{candidate}:hook_fire_counts")
            dispatch = payload.get("dispatch_by_site")
            if not isinstance(dispatch, Mapping) or set(dispatch) != {site}:
                reasons.append(f"{candidate}:dispatch_coverage")
            else:
                site_dispatch = dispatch.get(site)
                if (
                    not isinstance(site_dispatch, Mapping)
                    or site_dispatch.get("downstream_dispatch_verified") is not True
                ):
                    reasons.append(f"{candidate}:downstream_dispatch_verified")
            if payload.get("all_router_dispatch_verified") is not True:
                reasons.append(f"{candidate}:all_router_dispatch_verified")

    topk: float | None = None
    raw_topk = manipulation.get("topk_changed_rate")
    if raw_topk is not None:
        try:
            topk = float(raw_topk)
        except (TypeError, ValueError):
            reasons.append("topk_changed_rate")
        else:
            if not math.isfinite(topk) or not 0.0 <= topk <= 1.0:
                reasons.append("topk_changed_rate")
    return not reasons, sorted(set(reasons)), topk


def _is_positive_count(value: Any) -> bool:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, bool) and count > 0
