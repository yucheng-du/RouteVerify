"""Summaries for paired patching prediction rows."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from statistics import NormalDist
from typing import Any

import numpy as np

from routeverify.analysis.bootstrap import bootstrap_mean_ci

PRIMARY_MARGIN_METRIC = "mean_logprob"
SECONDARY_MARGIN_METRIC = "total_logprob"
MARGIN_UNITS = {
    PRIMARY_MARGIN_METRIC: (
        "mean target-token log-probability margin "
        "(INCORRECT minus CORRECT; log-probability per target token)"
    ),
    SECONDARY_MARGIN_METRIC: (
        "total target-sequence log-probability margin (INCORRECT minus CORRECT)"
    ),
}


@dataclass(frozen=True)
class _RowMetrics:
    mean_margin: float
    total_margin: float
    token_counts: dict[str, int] | None
    source: str

    def margin(self, metric: str) -> float:
        if metric == PRIMARY_MARGIN_METRIC:
            return self.mean_margin
        if metric == SECONDARY_MARGIN_METRIC:
            return self.total_margin
        raise ValueError(f"unknown margin metric: {metric}")


def patch_treatment_key(row: Mapping[str, Any]) -> str:
    site = row.get("site")
    key = str(row["treatment"]) if site is None else f"{row['treatment']}:{site}"
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("alpha") is not None:
        key = f"{key}:alpha={float(metadata['alpha']):g}"
    return key


def summarize_patch_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    decision_threshold: float = 0.0,
    primary_sesoi: float | None = None,
    sesoi_grid: Iterable[float] | None = None,
    sesoi_rationale: str | None = None,
    tost_alpha: float = 0.05,
) -> dict[str, Any]:
    """Build machine-readable patch metrics from prediction rows."""

    materialized = sorted(
        rows,
        key=lambda row: (
            str(row.get("pair_id")),
            str(row.get("receiver_variant")),
            "" if row.get("site") is None else str(row.get("site")),
            str(row.get("treatment")),
            int(row.get("boundary_step", -1)),
            int(row.get("token_position", -1)),
            float(
                row.get("metadata", {}).get("alpha", -1.0)
                if isinstance(row.get("metadata"), Mapping)
                else -1.0
            ),
        ),
    )
    if not materialized:
        raise ValueError("rows must be non-empty")
    if not math.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite")
    normalized_grid = _normalize_sesoi_grid(sesoi_grid)
    if primary_sesoi is not None:
        primary_sesoi = _validate_sesoi(primary_sesoi)
    if not 0.0 < tost_alpha < 0.5:
        raise ValueError("tost_alpha must be between 0 and 0.5")

    prepared = [(row, _row_metrics(row)) for row in materialized]
    by_treatment: dict[str, dict[str, list[float]]] = {
        PRIMARY_MARGIN_METRIC: {},
        SECONDARY_MARGIN_METRIC: {},
    }
    unpatched_by_receiver: dict[tuple[str, str], _RowMetrics] = {}
    for row, row_metrics in prepared:
        key = patch_treatment_key(row)
        for metric in (PRIMARY_MARGIN_METRIC, SECONDARY_MARGIN_METRIC):
            by_treatment[metric].setdefault(key, []).append(row_metrics.margin(metric))
        if row["treatment"] == "unpatched":
            baseline_key = (str(row["pair_id"]), str(row["receiver_variant"]))
            if baseline_key in unpatched_by_receiver:
                raise ValueError(f"duplicate unpatched row for {baseline_key}")
            unpatched_by_receiver[baseline_key] = row_metrics

    self_patch_deltas: dict[str, dict[str, list[float]]] = {
        PRIMARY_MARGIN_METRIC: {},
        SECONDARY_MARGIN_METRIC: {},
    }
    rerun_deltas: dict[str, dict[str, list[float]]] = {
        PRIMARY_MARGIN_METRIC: {},
        SECONDARY_MARGIN_METRIC: {},
    }
    effects_vs_unpatched: dict[str, dict[str, list[float]]] = {
        PRIMARY_MARGIN_METRIC: {},
        SECONDARY_MARGIN_METRIC: {},
    }
    effects_by_unit: dict[str, dict[tuple[str, str, str, str], float]] = {
        PRIMARY_MARGIN_METRIC: {},
        SECONDARY_MARGIN_METRIC: {},
    }
    for row, row_metrics in prepared:
        treatment = str(row["treatment"])
        site = row.get("site")
        pair_id = str(row["pair_id"])
        receiver = str(row["receiver_variant"])
        baseline_key = (pair_id, receiver)
        if baseline_key not in unpatched_by_receiver:
            raise ValueError(f"missing unpatched row for {baseline_key}")
        baseline = unpatched_by_receiver[baseline_key]

        for metric in (PRIMARY_MARGIN_METRIC, SECONDARY_MARGIN_METRIC):
            margin = row_metrics.margin(metric)
            baseline_margin = baseline.margin(metric)
            if treatment == "self_patch" and site is not None:
                self_patch_deltas[metric].setdefault(str(site), []).append(
                    abs(margin - baseline_margin)
                )
            elif treatment == "unpatched_rerun":
                rerun_deltas[metric].setdefault(receiver, []).append(abs(margin - baseline_margin))
            elif site is not None and treatment not in {
                "unpatched",
                "unpatched_rerun",
                "self_patch",
            }:
                key = patch_treatment_key(row)
                effect = margin - baseline_margin
                effects_vs_unpatched[metric].setdefault(key, []).append(effect)
                metadata = row.get("metadata")
                alpha = (
                    metadata.get("alpha")
                    if isinstance(metadata, Mapping)
                    else None
                )
                unit_treatment = (
                    treatment
                    if alpha is None
                    else f"{treatment}@alpha={float(alpha):g}"
                )
                unit_key = (pair_id, receiver, str(site), unit_treatment)
                if unit_key in effects_by_unit[metric]:
                    raise ValueError(f"duplicate patch row for {unit_key}")
                effects_by_unit[metric][unit_key] = effect

    effect_summaries_by_metric: dict[str, dict[str, dict[str, Any]]] = {}
    adjusted_summaries_by_metric: dict[str, dict[str, dict[str, Any]]] = {}
    for metric in (PRIMARY_MARGIN_METRIC, SECONDARY_MARGIN_METRIC):
        summaries = {
            key: _effect_summary(key, values, seed=seed, metric=metric)
            for key, values in sorted(effects_vs_unpatched[metric].items())
        }
        _attach_multiple_comparison_corrections(summaries)
        effect_summaries_by_metric[metric] = summaries

        adjusted = _control_adjusted_effects(effects_by_unit[metric])
        adjusted_summaries = {
            key: _effect_summary(key, values, seed=seed, metric=metric)
            for key, values in sorted(adjusted.items())
        }
        _attach_multiple_comparison_corrections(adjusted_summaries)
        adjusted_summaries_by_metric[metric] = adjusted_summaries

    primary_effects = effect_summaries_by_metric[PRIMARY_MARGIN_METRIC]
    primary_adjusted = adjusted_summaries_by_metric[PRIMARY_MARGIN_METRIC]
    secondary_effects = effect_summaries_by_metric[SECONDARY_MARGIN_METRIC]
    secondary_adjusted = adjusted_summaries_by_metric[SECONDARY_MARGIN_METRIC]
    primary_self_deltas = self_patch_deltas[PRIMARY_MARGIN_METRIC]
    primary_rerun_deltas = rerun_deltas[PRIMARY_MARGIN_METRIC]
    secondary_self_deltas = self_patch_deltas[SECONDARY_MARGIN_METRIC]
    secondary_rerun_deltas = rerun_deltas[SECONDARY_MARGIN_METRIC]

    return {
        "num_pairs": len({str(row["pair_id"]) for row in materialized}),
        "num_rows": len(materialized),
        "primary_metric": PRIMARY_MARGIN_METRIC,
        "secondary_metric": SECONDARY_MARGIN_METRIC,
        "metric_units": dict(MARGIN_UNITS),
        "decision_rule": {
            "margin": "INCORRECT mean_logprob minus CORRECT mean_logprob",
            "threshold": decision_threshold,
            "incorrect_when": "margin > threshold",
            "correct_when": "margin <= threshold",
        },
        "mean_margin_by_treatment": {
            key: sum(values) / len(values)
            for key, values in sorted(by_treatment[PRIMARY_MARGIN_METRIC].items())
        },
        "mean_mean_logprob_margin_by_treatment": {
            key: sum(values) / len(values)
            for key, values in sorted(by_treatment[PRIMARY_MARGIN_METRIC].items())
        },
        "mean_total_logprob_margin_by_treatment": {
            key: sum(values) / len(values)
            for key, values in sorted(by_treatment[SECONDARY_MARGIN_METRIC].items())
        },
        # Compatibility keys now follow the declared primary metric.
        "ate_margin_vs_unpatched": primary_effects,
        "control_adjusted_ate_margin": primary_adjusted,
        "ate_mean_logprob_margin_vs_unpatched": primary_effects,
        "control_adjusted_ate_mean_logprob_margin": primary_adjusted,
        "ate_total_logprob_margin_vs_unpatched": secondary_effects,
        "control_adjusted_ate_total_logprob_margin": secondary_adjusted,
        "max_abs_self_patch_delta_by_site": {
            key: max(values) for key, values in sorted(primary_self_deltas.items())
        },
        "max_abs_unpatched_rerun_delta_by_receiver": {
            key: max(values) for key, values in sorted(primary_rerun_deltas.items())
        },
        "max_abs_self_patch_total_logprob_delta_by_site": {
            key: max(values) for key, values in sorted(secondary_self_deltas.items())
        },
        "max_abs_unpatched_rerun_total_logprob_delta_by_receiver": {
            key: max(values) for key, values in sorted(secondary_rerun_deltas.items())
        },
        "token_count_distribution": _token_count_distribution(prepared),
        "mean_vs_total_sensitivity": _mean_vs_total_sensitivity(
            prepared,
            threshold=decision_threshold,
        ),
        "flip_rates": _flip_rates(
            prepared,
            unpatched_by_receiver,
            threshold=decision_threshold,
        ),
        "equivalence_analysis": _equivalence_analysis(
            effects_by_unit[PRIMARY_MARGIN_METRIC],
            primary_sesoi=primary_sesoi,
            sesoi_grid=normalized_grid,
            sesoi_rationale=sesoi_rationale,
            alpha=tost_alpha,
        ),
    }


def _row_metrics(row: Mapping[str, Any]) -> _RowMetrics:
    scores = row.get("scores")
    if scores is not None:
        if not isinstance(scores, Mapping):
            raise ValueError("scores must be a mapping")
        correct_total, correct_mean, correct_count = _candidate_score_values(scores, "CORRECT")
        incorrect_total, incorrect_mean, incorrect_count = _candidate_score_values(
            scores, "INCORRECT"
        )
        return _RowMetrics(
            mean_margin=incorrect_mean - correct_mean,
            total_margin=incorrect_total - correct_total,
            token_counts={"CORRECT": correct_count, "INCORRECT": incorrect_count},
            source="scores",
        )

    generic = row.get("margin_incorrect_minus_correct")
    mean_value = row.get("margin_incorrect_minus_correct_mean", generic)
    total_value = row.get("margin_incorrect_minus_correct_total", generic)
    if mean_value is None or total_value is None:
        raise ValueError(
            "row must contain scores or explicit margin_incorrect_minus_correct values"
        )
    return _RowMetrics(
        mean_margin=float(mean_value),
        total_margin=float(total_value),
        token_counts=None,
        source="legacy_margin_fields",
    )


def _candidate_score_values(
    scores: Mapping[str, Any],
    candidate: str,
) -> tuple[float, float, int]:
    payload = scores.get(candidate)
    if not isinstance(payload, Mapping):
        raise ValueError(f"scores.{candidate} must be a mapping")
    try:
        total = float(payload["total_logprob"])
        token_count = int(payload["token_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"scores.{candidate} must include numeric total_logprob and token_count"
        ) from exc
    if not math.isfinite(total):
        raise ValueError(f"scores.{candidate}.total_logprob must be finite")
    if token_count <= 0:
        raise ValueError(f"scores.{candidate}.token_count must be positive")
    mean = total / token_count
    reported_mean = payload.get("mean_logprob")
    if reported_mean is not None:
        reported = float(reported_mean)
        if not math.isfinite(reported):
            raise ValueError(f"scores.{candidate}.mean_logprob must be finite")
        if not math.isclose(reported, mean, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"scores.{candidate}.mean_logprob does not equal total_logprob/token_count"
            )
    return total, mean, token_count


def _effect_summary(
    key: str,
    values: list[float],
    *,
    seed: int,
    metric: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = bootstrap_mean_ci(values, seed=seed)
    array = np.asarray(values, dtype=float)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    summary["n"] = int(array.size)
    summary["metric"] = metric
    summary["units"] = MARGIN_UNITS[metric]
    summary["standardized_effect"] = float(summary["mean"] / std) if std else 0.0
    summary["sign_consistency"] = _sign_consistency(values)
    expected_sign = _expected_margin_sign(key)
    summary["expected_margin_sign"] = expected_sign
    summary["expected_sign_rate"] = (
        None if expected_sign is None else float(np.mean(np.sign(array) == expected_sign))
    )
    summary["p_two_sided_signflip"] = _signflip_p_value(array, seed=seed)
    return summary


def _control_adjusted_effects(
    effects_by_unit: Mapping[tuple[str, str, str, str], float],
) -> dict[str, list[float]]:
    adjusted: dict[str, list[float]] = {}
    controls_by_target = {
        "clean_to_corrupt": (
            "norm_random_delta",
            "pre_error_clean_to_corrupt",
            "random_matched_clean_to_corrupt",
        ),
        "corrupt_to_clean": (
            "norm_random_delta",
            "pre_error_corrupt_to_clean",
            "random_matched_corrupt_to_clean",
        ),
    }
    for (pair_id, receiver, site, treatment), target_effect in effects_by_unit.items():
        for control in controls_by_target.get(treatment, ()):
            control_key = (pair_id, receiver, site, control)
            if control_key not in effects_by_unit:
                continue
            adjusted_key = f"{treatment}_minus_{control}:{site}"
            adjusted.setdefault(adjusted_key, []).append(
                target_effect - effects_by_unit[control_key]
            )
    return adjusted


def _token_count_distribution(
    prepared: list[tuple[Mapping[str, Any], _RowMetrics]],
) -> dict[str, Any]:
    return {
        "scope_note": (
            "Counts are over prediction rows; the same pair/candidate can recur across "
            "treatments and sites."
        ),
        "all_prediction_rows": _token_count_scope(prepared),
        "unpatched_rows": _token_count_scope(
            [(row, metrics) for row, metrics in prepared if row["treatment"] == "unpatched"]
        ),
    }


def _token_count_scope(
    prepared: list[tuple[Mapping[str, Any], _RowMetrics]],
) -> dict[str, Any]:
    by_candidate: dict[str, list[int]] = {"CORRECT": [], "INCORRECT": []}
    pair_counts: dict[str, int] = {}
    for _row, metrics in prepared:
        if metrics.token_counts is None:
            continue
        for candidate in by_candidate:
            by_candidate[candidate].append(metrics.token_counts[candidate])
        pair_key = (
            f"CORRECT={metrics.token_counts['CORRECT']}|"
            f"INCORRECT={metrics.token_counts['INCORRECT']}"
        )
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
    return {
        "rows_with_token_counts": sum(pair_counts.values()),
        "by_candidate": {
            candidate: _integer_distribution(values) for candidate, values in by_candidate.items()
        },
        "candidate_count_pairs": dict(sorted(pair_counts.items())),
    }


def _integer_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "counts": {}, "min": None, "max": None, "mean": None}
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {
        "n": len(values),
        "counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
    }


def _mean_vs_total_sensitivity(
    prepared: list[tuple[Mapping[str, Any], _RowMetrics]],
    *,
    threshold: float,
) -> dict[str, Any]:
    score_rows = [(row, metrics) for row, metrics in prepared if metrics.source == "scores"]
    return {
        "comparison_scope": "rows with candidate scores",
        "all_score_rows": _margin_sensitivity_scope(score_rows, threshold=threshold),
        "unpatched_score_rows": _margin_sensitivity_scope(
            [(row, metrics) for row, metrics in score_rows if row["treatment"] == "unpatched"],
            threshold=threshold,
        ),
        "unpatched_by_receiver": {
            receiver: _margin_sensitivity_scope(
                [
                    (row, metrics)
                    for row, metrics in score_rows
                    if row["treatment"] == "unpatched" and str(row["receiver_variant"]) == receiver
                ],
                threshold=threshold,
            )
            for receiver in sorted(
                {
                    str(row["receiver_variant"])
                    for row, _metrics in score_rows
                    if row["treatment"] == "unpatched"
                }
            )
        },
    }


def _margin_sensitivity_scope(
    prepared: list[tuple[Mapping[str, Any], _RowMetrics]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not prepared:
        return {
            "n": 0,
            "prediction_disagreement_count": 0,
            "prediction_disagreement_rate": None,
            "mean_absolute_margin_difference": None,
            "max_absolute_margin_difference": None,
        }
    differences = [abs(metrics.mean_margin - metrics.total_margin) for _row, metrics in prepared]
    disagreements = sum(
        (metrics.mean_margin > threshold) != (metrics.total_margin > threshold)
        for _row, metrics in prepared
    )
    return {
        "n": len(prepared),
        "threshold": threshold,
        "prediction_disagreement_count": disagreements,
        "prediction_disagreement_rate": float(disagreements / len(prepared)),
        "mean_absolute_margin_difference": float(sum(differences) / len(differences)),
        "max_absolute_margin_difference": float(max(differences)),
    }


def _flip_rates(
    prepared: list[tuple[Mapping[str, Any], _RowMetrics]],
    unpatched_by_receiver: Mapping[tuple[str, str], _RowMetrics],
    *,
    threshold: float,
) -> dict[str, Any]:
    direction_specs: tuple[
        tuple[str, str, str, Callable[[float], bool], Callable[[float], bool], str],
        ...,
    ] = (
        (
            "clean_to_incorrect",
            "corrupt_to_clean",
            "clean",
            lambda baseline: baseline <= threshold,
            lambda patched: patched > threshold,
            "clean receiver is CORRECT at baseline and INCORRECT after corrupt-donor patch",
        ),
        (
            "corrupt_to_correct",
            "clean_to_corrupt",
            "corrupt",
            lambda baseline: baseline > threshold,
            lambda patched: patched <= threshold,
            "corrupt receiver is INCORRECT at baseline and CORRECT after clean-donor patch",
        ),
    )
    by_site: dict[str, dict[str, dict[str, Any]]] = {}
    for (
        label,
        treatment,
        receiver,
        baseline_eligible,
        patched_target,
        definition,
    ) in direction_specs:
        rows_by_site_alpha: dict[
            tuple[str, str], list[tuple[Mapping[str, Any], _RowMetrics]]
        ] = {}
        for row, metrics in prepared:
            site = row.get("site")
            if (
                site is not None
                and str(row["treatment"]) == treatment
                and str(row["receiver_variant"]) == receiver
            ):
                alpha = row.get("metadata", {}).get("alpha")
                alpha_key = "alpha=unspecified" if alpha is None else f"alpha={float(alpha):g}"
                rows_by_site_alpha.setdefault((str(site), alpha_key), []).append((row, metrics))
        for (site, alpha_key), site_rows in rows_by_site_alpha.items():
            eligible_count = 0
            flip_count = 0
            for row, metrics in site_rows:
                pair_id = str(row["pair_id"])
                baseline = unpatched_by_receiver[(pair_id, receiver)].mean_margin
                if baseline_eligible(baseline):
                    eligible_count += 1
                    if patched_target(metrics.mean_margin):
                        flip_count += 1
            by_site.setdefault(site, {}).setdefault(alpha_key, {})[label] = _flip_rate_payload(
                treatment=treatment,
                receiver=receiver,
                definition=definition,
                threshold=threshold,
                flip_count=flip_count,
                all_pair_count=len(site_rows),
                eligible_count=eligible_count,
            )
    return {
        "metric": PRIMARY_MARGIN_METRIC,
        "units": MARGIN_UNITS[PRIMARY_MARGIN_METRIC],
        "threshold": threshold,
        "ci_method": "95% Wilson score interval",
        "grouping": ["site", "alpha", "direction"],
        "by_site": {
            site: {alpha: by_site[site][alpha] for alpha in sorted(by_site[site])}
            for site in sorted(by_site)
        },
    }


def _flip_rate_payload(
    *,
    treatment: str,
    receiver: str,
    definition: str,
    threshold: float,
    flip_count: int,
    all_pair_count: int,
    eligible_count: int,
) -> dict[str, Any]:
    return {
        "treatment": treatment,
        "receiver_variant": receiver,
        "definition": definition,
        "threshold": threshold,
        "flip_count": flip_count,
        "all_pair_denominator": all_pair_count,
        "rate_all_pairs": (None if all_pair_count == 0 else float(flip_count / all_pair_count)),
        "ci_all_pairs": (
            None if all_pair_count == 0 else wilson_confidence_interval(flip_count, all_pair_count)
        ),
        "baseline_eligible_denominator": eligible_count,
        "rate_baseline_eligible": (
            None if eligible_count == 0 else float(flip_count / eligible_count)
        ),
        "ci_baseline_eligible": (
            None if eligible_count == 0 else wilson_confidence_interval(flip_count, eligible_count)
        ),
    }


def wilson_confidence_interval(
    successes: int,
    total: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | str]:
    """Return a two-sided Wilson score interval for one binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must satisfy 0 <= successes <= total")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    alpha = 1.0 - confidence_level
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / total + z_value**2 / (4.0 * total**2))
        / denominator
    )
    return {
        "method": "wilson",
        "confidence_level": confidence_level,
        "low": float(max(0.0, center - half_width)),
        "high": float(min(1.0, center + half_width)),
    }


def paired_tost(
    treated: list[float] | np.ndarray,
    control: list[float] | np.ndarray | None = None,
    *,
    sesoi: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired two-one-sided t tests for mean equivalence.

    The estimand is ``treated - control``. Passing ``control=None`` treats
    ``treated`` as already-paired differences. A symmetric ``sesoi`` creates
    bounds ``[-sesoi, +sesoi]``; alternatively provide both explicit bounds.
    Equivalence uses strict bounds and requires both one-sided p-values below
    ``alpha``. The reported confidence interval is therefore ``1 - 2*alpha``
    (90% for the conventional alpha=0.05 TOST).
    """

    treated_array = np.asarray(treated, dtype=float)
    if treated_array.ndim != 1:
        raise ValueError("treated must be a 1D array")
    if control is None:
        differences = treated_array
    else:
        control_array = np.asarray(control, dtype=float)
        if control_array.ndim != 1:
            raise ValueError("control must be a 1D array")
        if treated_array.shape != control_array.shape:
            raise ValueError("treated/control shapes differ")
        differences = treated_array - control_array
    if differences.size < 2:
        raise ValueError("paired TOST requires at least two pairs")
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired differences must be finite")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")

    if sesoi is not None:
        symmetric_bound = _validate_sesoi(sesoi)
        if lower_bound is not None or upper_bound is not None:
            raise ValueError("provide sesoi or explicit bounds, not both")
        lower = -symmetric_bound
        upper = symmetric_bound
    else:
        if lower_bound is None or upper_bound is None:
            raise ValueError("provide sesoi or both lower_bound and upper_bound")
        lower = float(lower_bound)
        upper = float(upper_bound)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("equivalence bounds must be finite and lower < upper")

    n_pairs = int(differences.size)
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    if bool(np.all(differences == differences[0])):
        mean_difference = float(differences[0])
        std_difference = 0.0
    standard_error = std_difference / math.sqrt(n_pairs)
    degrees_of_freedom = n_pairs - 1
    confidence_level = 1.0 - 2.0 * alpha

    if standard_error == 0.0:
        on_lower = math.isclose(mean_difference, lower, rel_tol=1e-12, abs_tol=1e-15)
        on_upper = math.isclose(mean_difference, upper, rel_tol=1e-12, abs_tol=1e-15)
        t_lower = 0.0 if on_lower else (math.inf if mean_difference > lower else -math.inf)
        t_upper = 0.0 if on_upper else (math.inf if mean_difference > upper else -math.inf)
        p_lower = 0.5 if on_lower else (0.0 if mean_difference > lower else 1.0)
        p_upper = 0.5 if on_upper else (0.0 if mean_difference < upper else 1.0)
        ci_low = mean_difference
        ci_high = mean_difference
    else:
        t_lower = (mean_difference - lower) / standard_error
        t_upper = (mean_difference - upper) / standard_error
        p_lower = _student_t_survival(t_lower, degrees_of_freedom)
        p_upper = _student_t_cdf(t_upper, degrees_of_freedom)
        critical = _student_t_ppf(1.0 - alpha, degrees_of_freedom)
        ci_low = mean_difference - critical * standard_error
        ci_high = mean_difference + critical * standard_error

    equivalent_by_p = p_lower < alpha and p_upper < alpha
    equivalent_by_ci = ci_low > lower and ci_high < upper
    return {
        "n": n_pairs,
        "effect_definition": "treated_minus_control",
        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "lower_bound": lower,
        "upper_bound": upper,
        "sesoi": (
            float(sesoi)
            if sesoi is not None
            else (float(upper) if math.isclose(-lower, upper) else None)
        ),
        "alpha": alpha,
        "confidence_level": confidence_level,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_lower": float(t_lower),
        "t_upper": float(t_upper),
        "p_lower": float(min(max(p_lower, 0.0), 1.0)),
        "p_upper": float(min(max(p_upper, 0.0), 1.0)),
        "equivalent": bool(equivalent_by_p and equivalent_by_ci),
        "equivalent_by_p_values": bool(equivalent_by_p),
        "equivalent_by_confidence_interval": bool(equivalent_by_ci),
        "decision_rule": (
            "equivalent iff p_lower < alpha and p_upper < alpha and the "
            "(1-2*alpha) CI lies strictly inside the equivalence bounds"
        ),
    }


def _equivalence_analysis(
    effects_by_unit: Mapping[tuple[str, str, str, str], float],
    *,
    primary_sesoi: float | None,
    sesoi_grid: tuple[float, ...],
    sesoi_rationale: str | None,
    alpha: float,
) -> dict[str, Any]:
    direct_treatments = {"clean_to_corrupt", "corrupt_to_clean"}
    direct: dict[str, list[tuple[str, str, float]]] = {}
    norm_adjusted: dict[str, list[tuple[str, str, float]]] = {}
    matched: dict[str, list[tuple[str, str, float]]] = {}
    for (pair_id, receiver, site, treatment), effect in effects_by_unit.items():
        if treatment not in direct_treatments:
            continue
        direct_key = f"{treatment}:{site}"
        direct.setdefault(direct_key, []).append((pair_id, receiver, effect))
        norm_control_key = (pair_id, receiver, site, "norm_random_delta")
        if norm_control_key in effects_by_unit:
            norm_key = f"{treatment}_minus_norm_random_delta:{site}"
            norm_adjusted.setdefault(norm_key, []).append(
                (pair_id, receiver, effect - effects_by_unit[norm_control_key])
            )
        matched_control = f"random_matched_{treatment}"
        control_key = (pair_id, receiver, site, matched_control)
        if control_key in effects_by_unit:
            contrast_key = f"{treatment}_minus_{matched_control}:{site}"
            matched.setdefault(contrast_key, []).append(
                (pair_id, receiver, effect - effects_by_unit[control_key])
            )

    return {
        "metric": PRIMARY_MARGIN_METRIC,
        "units": MARGIN_UNITS[PRIMARY_MARGIN_METRIC],
        "estimand_note": (
            "Direct effects are patched minus the same pair/receiver unpatched margin. "
            "Norm-adjusted direct effects additionally subtract the same-pair "
            "norm-random-delta effect. "
            "Matched-donor contrasts subtract the corresponding random-matched-donor "
            "effect within the same pair/receiver/site."
        ),
        "sesoi_config": {
            "primary": primary_sesoi,
            "sensitivity_grid": list(sesoi_grid),
            "rationale": sesoi_rationale,
            "status": "configured" if primary_sesoi is not None else "not_configured",
            "warning": (
                None
                if primary_sesoi is not None
                else "No primary SESOI configured; no primary equivalence claim is available."
            ),
        },
        "alpha": alpha,
        "confidence_level": 1.0 - 2.0 * alpha,
        "direct_effect_vs_unpatched": {
            key: _equivalence_entry(
                values,
                primary_sesoi=primary_sesoi,
                sesoi_grid=sesoi_grid,
                alpha=alpha,
            )
            for key, values in sorted(direct.items())
        },
        "direct_effect_vs_norm_random_delta": {
            key: _equivalence_entry(
                values,
                primary_sesoi=primary_sesoi,
                sesoi_grid=sesoi_grid,
                alpha=alpha,
            )
            for key, values in sorted(norm_adjusted.items())
        },
        "matched_donor_contrast": {
            key: _equivalence_entry(
                values,
                primary_sesoi=primary_sesoi,
                sesoi_grid=sesoi_grid,
                alpha=alpha,
            )
            for key, values in sorted(matched.items())
        },
    }


def _equivalence_entry(
    values: list[tuple[str, str, float]],
    *,
    primary_sesoi: float | None,
    sesoi_grid: tuple[float, ...],
    alpha: float,
) -> dict[str, Any]:
    ordered = sorted(values, key=lambda item: (item[0], item[1]))
    differences = [value for _pair_id, _receiver, value in ordered]
    return {
        "n": len(ordered),
        "paired_differences": [
            {
                "unit_id": f"{pair_id}|{receiver}",
                "pair_id": pair_id,
                "receiver_variant": receiver,
                "difference": value,
            }
            for pair_id, receiver, value in ordered
        ],
        "primary_tost": _tost_or_insufficient(
            differences,
            sesoi=primary_sesoi,
            alpha=alpha,
        ),
        "sensitivity_grid": {
            f"{bound:.12g}": _tost_or_insufficient(
                differences,
                sesoi=bound,
                alpha=alpha,
            )
            for bound in sesoi_grid
        },
    }


def _tost_or_insufficient(
    differences: list[float],
    *,
    sesoi: float | None,
    alpha: float,
) -> dict[str, Any] | None:
    if sesoi is None:
        return None
    if len(differences) < 2:
        return {
            "n": len(differences),
            "sesoi": sesoi,
            "alpha": alpha,
            "equivalent": None,
            "status": "insufficient_pairs",
        }
    return paired_tost(differences, sesoi=sesoi, alpha=alpha)


def _normalize_sesoi_grid(values: Iterable[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(sorted({_validate_sesoi(value) for value in values}))


def _validate_sesoi(value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("SESOI values must be finite and positive")
    return resolved


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if math.isnan(value):
        raise ValueError("value must not be NaN")
    if value == math.inf:
        return 1.0
    if value == -math.inf:
        return 0.0
    if value == 0.0:
        return 0.5
    degrees = float(degrees_of_freedom)
    beta_x = degrees / (degrees + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        beta_x,
        degrees / 2.0,
        0.5,
    )
    return float(1.0 - tail if value > 0.0 else tail)


def _student_t_survival(value: float, degrees_of_freedom: int) -> float:
    """Student-t survival probability without cancellation in the upper tail."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if math.isnan(value):
        raise ValueError("value must not be NaN")
    if value == math.inf:
        return 0.0
    if value == -math.inf:
        return 1.0
    if value == 0.0:
        return 0.5
    degrees = float(degrees_of_freedom)
    beta_x = degrees / (degrees + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        beta_x,
        degrees / 2.0,
        0.5,
    )
    return float(tail if value > 0.0 else 1.0 - tail)


def _student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_ppf(1.0 - probability, degrees_of_freedom)

    low = 0.0
    high = 1.0
    while _student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
        if high > 1e12:
            raise RuntimeError("failed to bracket Student-t quantile")
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return float((low + high) / 2.0)


def _regularized_incomplete_beta(x_value: float, a_value: float, b_value: float) -> float:
    if not 0.0 <= x_value <= 1.0:
        raise ValueError("x_value must be in [0, 1]")
    if a_value <= 0.0 or b_value <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x_value == 0.0:
        return 0.0
    if x_value == 1.0:
        return 1.0
    log_beta_term = (
        math.lgamma(a_value + b_value)
        - math.lgamma(a_value)
        - math.lgamma(b_value)
        + a_value * math.log(x_value)
        + b_value * math.log1p(-x_value)
    )
    beta_term = math.exp(log_beta_term)
    if x_value < (a_value + 1.0) / (a_value + b_value + 2.0):
        result = beta_term * _beta_continued_fraction(a_value, b_value, x_value) / a_value
    else:
        result = 1.0 - (
            beta_term * _beta_continued_fraction(b_value, a_value, 1.0 - x_value) / b_value
        )
    return float(min(max(result, 0.0), 1.0))


def _beta_continued_fraction(a_value: float, b_value: float, x_value: float) -> float:
    max_iterations = 200
    epsilon = 3e-14
    fp_min = 1e-300
    qab = a_value + b_value
    qap = a_value + 1.0
    qam = a_value - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x_value / qap
    if abs(d_value) < fp_min:
        d_value = fp_min
    d_value = 1.0 / d_value
    result = d_value
    for iteration in range(1, max_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration * (b_value - iteration) * x_value / ((qam + doubled) * (a_value + doubled))
        )
        d_value = 1.0 + coefficient * d_value
        if abs(d_value) < fp_min:
            d_value = fp_min
        c_value = 1.0 + coefficient / c_value
        if abs(c_value) < fp_min:
            c_value = fp_min
        d_value = 1.0 / d_value
        result *= d_value * c_value

        coefficient = -(
            (a_value + iteration)
            * (qab + iteration)
            * x_value
            / ((a_value + doubled) * (qap + doubled))
        )
        d_value = 1.0 + coefficient * d_value
        if abs(d_value) < fp_min:
            d_value = fp_min
        c_value = 1.0 + coefficient / c_value
        if abs(c_value) < fp_min:
            c_value = fp_min
        d_value = 1.0 / d_value
        delta = d_value * c_value
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RuntimeError("incomplete beta continued fraction did not converge")


def _expected_margin_sign(key: str) -> int | None:
    if key.startswith("clean_to_corrupt"):
        return -1
    if key.startswith("corrupt_to_clean"):
        return 1
    return None


def _signflip_p_value(
    values: np.ndarray,
    *,
    seed: int,
    max_exact_n: int = 16,
    n_resamples: int = 10000,
) -> float:
    """Two-sided paired sign-flip p-value for the null of zero mean effect."""

    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    tolerance = 1e-12
    if values.size <= max_exact_n:
        total = 0
        at_least_observed = 0
        for exact_signs in product((-1.0, 1.0), repeat=int(values.size)):
            total += 1
            signed_mean = abs(float(np.mean(values * np.asarray(exact_signs, dtype=float))))
            if signed_mean + tolerance >= observed:
                at_least_observed += 1
        return float(at_least_observed / total)

    rng = np.random.default_rng(seed)
    at_least_observed = 0
    for _ in range(n_resamples):
        sampled_signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.size)
        signed_mean = abs(float(np.mean(values * sampled_signs)))
        if signed_mean + tolerance >= observed:
            at_least_observed += 1
    return float((at_least_observed + 1) / (n_resamples + 1))


def _attach_multiple_comparison_corrections(effect_summaries: dict[str, dict[str, Any]]) -> None:
    p_values = {
        key: float(value["p_two_sided_signflip"]) for key, value in effect_summaries.items()
    }
    for key, q_value in _benjamini_hochberg(p_values).items():
        effect_summaries[key]["bh_q_global"] = q_value

    families: dict[str, dict[str, float]] = {}
    for key, p_value in p_values.items():
        families.setdefault(_effect_family(key), {})[key] = p_value
    for family in families.values():
        for key, q_value in _benjamini_hochberg(family).items():
            effect_summaries[key]["bh_q_family"] = q_value


def _effect_family(key: str) -> str:
    return key.split(":", 1)[0]


def _benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted_descending: dict[str, float] = {}
    running_min = 1.0
    for rank_from_end, (key, p_value) in enumerate(reversed(ordered), start=1):
        rank = total - rank_from_end + 1
        adjusted = min(running_min, p_value * total / rank)
        running_min = adjusted
        adjusted_descending[key] = float(min(max(adjusted, 0.0), 1.0))
    return adjusted_descending


def _sign_consistency(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return float(sum(value == 0 for value in values) / len(values))
    sign = 1 if mean > 0 else -1
    return float(sum((value > 0) == (sign > 0) for value in values) / len(values))


def patch_effect_markdown(metrics: Mapping[str, Any], *, max_rows: int = 40) -> str:
    """Render a compact Markdown table for patch effect summaries."""

    effects = metrics.get("ate_margin_vs_unpatched", {})
    adjusted = metrics.get("control_adjusted_ate_margin", {})
    if not isinstance(effects, Mapping) or not effects:
        return "No patch effects found.\n"
    sections = [_effect_table("Effects vs unpatched", effects, max_rows=max_rows)]
    if isinstance(adjusted, Mapping) and adjusted:
        sections.append(
            _effect_table(
                "Control-adjusted effects",
                adjusted,
                max_rows=max_rows,
            )
        )
    return "\n".join(sections)


def _effect_table(title: str, effects: Mapping[str, Any], *, max_rows: int) -> str:
    lines = [
        f"### {title}",
        "",
        "| treatment/site | n | mean delta | CI low | CI high | std effect | sign consistency | family q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sorted_items = sorted(
        effects.items(),
        key=lambda item: abs(float(item[1].get("mean", 0.0))),
        reverse=True,
    )
    for key, value in sorted_items[:max_rows]:
        lines.append(
            "| {key} | {n} | {mean:.6g} | {ci_low:.6g} | {ci_high:.6g} | {std:.6g} | {sign:.3g} | {q:.3g} |".format(
                key=key,
                n=int(value.get("n", 0)),
                mean=float(value.get("mean", 0.0)),
                ci_low=float(value.get("ci_low", 0.0)),
                ci_high=float(value.get("ci_high", 0.0)),
                std=float(value.get("standardized_effect", 0.0)),
                sign=float(value.get("sign_consistency", 0.0)),
                q=float(value.get("bh_q_family", 1.0)),
            )
        )
    return "\n".join(lines) + "\n"
