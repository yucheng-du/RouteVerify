"""Strict analysis of fixed-receiver, external-donor-role contrasts.

The primary estimand is D = margin(receiver <- corrupt donor) minus
margin(receiver <- clean donor). Both donor roles come from the same external
pair, and both receiver roles are measured in the same model-loading run.
Positive D describes donor-role directionality under this intervention; it
does not identify a correctness-specific representation.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from routeverify.analysis.bootstrap import (
    paired_cluster_bootstrap_indices,
    paired_cluster_bootstrap_mean_ci,
    paired_cluster_indices_sha256,
)

_ROLES = ("clean", "corrupt")
_COMBINATIONS = {(receiver, donor) for receiver in _ROLES for donor in _ROLES}
_IDENTITY_FIELDS = ("sample_id", "input_sha256", "source_component")
_ESTIMANDS = (
    "d_clean_mean",
    "d_corrupt_mean",
    "d_common_mean",
    "d_clean_total",
    "d_corrupt_total",
    "d_common_total",
)
_MARGIN_PREFIX = "margin_incorrect_minus_correct_"

_BUNDLE_CATEGORIES = ("exact", "matched", "norm_random", "norm_random_delta")
_BUNDLE_PROFILE_CATEGORIES = {
    "core": ("exact", "matched", "norm_random_delta"),
    "extended": _BUNDLE_CATEGORIES,
}
_BUNDLE_PROFILE_CONTROLS = {
    "core": ("unpatched_rerun", "self_patch"),
    "extended": ("unpatched_rerun", "self_patch", "alpha_zero_identity"),
}


def analyze_bundle_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Analyze complete last-token output/residual bundle panels.

    Each observed run/pair/site requires exact, matched, norm-random-delta and
    self-patch rows for each receiver role, plus shared unpatched and rerun
    baselines for both roles. The declared ``metadata.diagnostics_profile``
    (default ``core``) must agree throughout each run; ``extended`` additionally
    requires norm-random and alpha-zero rows. Candidate scores,
    input identities, source exclusion and treatment direction are validated.
    Numerical identity-control discrepancies are reported; the runner owns
    their configured tolerance gate.

    Effects use the unpatched receiver baseline. The donor-aligned bidirectional
    effect is T = (clean-receiver shift - corrupt-receiver shift) / 2. All
    category estimates and exact-minus-control contrasts share receiver-pair
    bootstrap draws, preserving the pairing and the observed donor assignment.
    Interfaces and runs are summarized separately. T is an intervention effect,
    not a test identifying a semantic representation.
    """

    _validate_options(n_resamples, seed, alpha, False)
    identities: dict[tuple[str, str, str], dict[str, Any]] = {}
    components: dict[tuple[str, str], str] = {}
    candidate_counts: dict[tuple[str, str, str], tuple[int, int]] = {}
    baselines: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    cells: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    matched_donors: dict[tuple[str, str], str] = {}
    site_specs: dict[tuple[str, str], dict[str, Any]] = {}
    profiles: dict[str, str] = {}
    ignored: Counter[str] = Counter()
    input_count = 0
    for row_index, row in enumerate(rows):
        input_count += 1
        if not isinstance(row, Mapping):
            raise TypeError(f"row {row_index} must be a mapping")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("panel") not in {"shared", "bundle"}:
            ignored[str(row.get("treatment", "<missing>"))] += 1
            continue
        run = _required_string(row, "run_id", f"row {row_index}")
        profile = metadata.get("diagnostics_profile", "core")
        if not isinstance(profile, str) or profile not in _BUNDLE_PROFILE_CATEGORIES:
            raise ValueError(f"row {row_index}: diagnostics_profile must be core or extended")
        if run in profiles and profiles[run] != profile:
            raise ValueError(f"mixed diagnostics_profile for run {run}")
        profiles[run] = profile
        pair = _required_string(row, "pair_id", f"row {row_index}")
        receiver = _role(row.get("receiver_variant"), "receiver_variant", row_index)
        receiver_identity = _identity(metadata, "receiver_identity", pair, receiver, row_index)
        _register_identity(identities, components, (run, pair, receiver), receiver_identity)
        if "source_component" in row and row["source_component"] != receiver_identity["source_component"]:
            raise ValueError(f"row {row_index}: top-level source_component disagrees with receiver identity")
        margins, counts = _validated_scores(row, row_index)
        count_key = (run, pair, receiver)
        if count_key in candidate_counts and candidate_counts[count_key] != counts:
            raise ValueError(f"mixed candidate token counts for receiver {count_key}")
        candidate_counts[count_key] = counts
        treatment = _required_string(row, "treatment", f"row {row_index}")
        record = {"margins": margins, "scores": row["scores"]}
        if metadata["panel"] == "shared":
            if treatment not in {"unpatched", "unpatched_rerun"}:
                raise ValueError(f"row {row_index}: invalid shared treatment {treatment}")
            if any(row.get(key) is not None for key in ("site", "donor_variant", "donor_pair_id")) or metadata.get("donor_identity") is not None:
                raise ValueError(f"row {row_index}: shared baseline must not name a site or donor")
            baseline_key = (run, pair, receiver, treatment)
            if baseline_key in baselines:
                raise ValueError(f"duplicate shared baseline for {baseline_key}")
            baselines[baseline_key] = record
            continue
        site = _required_string(row, "site", f"row {row_index}")
        match = re.fullmatch(r"bundle\.(moe_output|post_moe_residual)\[([1-9][0-9]*)\]", site)
        if match is None:
            raise ValueError(f"row {row_index}: invalid output/residual bundle site {site}")
        kind, count = match.group(1), int(match.group(2))
        patch_sites = metadata.get("patch_sites")
        if patch_sites is not None:
            if not isinstance(patch_sites, list) or len(patch_sites) != count or any(
                not isinstance(value, str) or re.fullmatch(rf"layer_[0-9]+\.{kind}", value) is None
                for value in patch_sites
            ) or len(set(patch_sites)) != count:
                raise ValueError(f"row {row_index}: patch_sites disagree with bundle interface/count")
        spec = {"interface_kind": kind, "n_bundle_sites": count, "patch_sites": patch_sites}
        if (run, site) in site_specs and site_specs[(run, site)] != spec:
            raise ValueError(f"mixed bundle patch_sites for {(run, site)}")
        site_specs[(run, site)] = spec
        opposite = "corrupt" if receiver == "clean" else "clean"
        direction = f"{opposite}_to_{receiver}"
        category = {direction: "exact", f"matched_{direction}": "matched"}.get(treatment, treatment)
        if category not in {*_BUNDLE_PROFILE_CATEGORIES[profile], *_BUNDLE_PROFILE_CONTROLS[profile][1:]}:
            raise ValueError(f"row {row_index}: invalid bundle treatment or receiver direction {treatment}")
        donor = _role(row.get("donor_variant"), "donor_variant", row_index)
        if donor != (receiver if category == "self_patch" else opposite):
            raise ValueError(f"row {row_index}: bundle donor role disagrees with treatment")
        donor_pair = _required_string(row, "donor_pair_id", f"row {row_index}")
        donor_identity = _identity(metadata, "donor_identity", donor_pair, donor, row_index)
        _register_identity(identities, components, (run, donor_pair, donor), donor_identity)
        if category == "matched":
            if donor_pair == pair or donor_identity["source_component"] == receiver_identity["source_component"]:
                raise ValueError(f"row {row_index}: matched donor must be an external, cross-source pair")
            if (run, pair) in matched_donors and matched_donors[(run, pair)] != donor_pair:
                raise ValueError(f"mixed matched donor pair for {(run, pair)}")
            matched_donors[(run, pair)] = donor_pair
        elif donor_pair != pair:
            raise ValueError(f"row {row_index}: {category} requires the receiver pair as donor source")
        if "alpha" in metadata and _number(metadata["alpha"], f"row {row_index} alpha") != (0.0 if category == "alpha_zero_identity" else 1.0):
            raise ValueError(f"row {row_index}: alpha disagrees with bundle treatment")
        cell_key, role_key = (run, pair, site), (receiver, category)
        if role_key in cells[cell_key]:
            raise ValueError(f"duplicate bundle row for {cell_key}, {role_key}")
        cells[cell_key][role_key] = record
    if not cells:
        raise ValueError("no bundle rows were found")
    per_pair = []
    panels: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (run, pair, site), cell in sorted(cells.items()):
        profile = profiles[run]
        categories = _BUNDLE_PROFILE_CATEGORIES[profile]
        controls = _BUNDLE_PROFILE_CONTROLS[profile]
        required = {(role, category) for role in _ROLES for category in (*categories, *controls) if category != "unpatched_rerun"}
        missing = sorted(required - cell.keys())
        if missing:
            raise ValueError(f"incomplete bundle controls for {(run, pair, site)}; missing {missing}")
        for role in _ROLES:
            for treatment in ("unpatched", "unpatched_rerun"):
                if (run, pair, role, treatment) not in baselines:
                    raise ValueError(f"missing shared baseline {(run, pair, role, treatment)}")
        record = {
            "run_id": run, "pair_id": pair, "site": site, "diagnostics_profile": profile,
            "source_component": components[(run, pair)],
            "donor_pair_id": matched_donors[(run, pair)],
            "donor_source_component": components[(run, matched_donors[(run, pair)])],
            **site_specs[(run, site)],
            "receiver_identities": {role: identities[(run, pair, role)] for role in _ROLES},
            "matched_donor_identities": {role: identities[(run, matched_donors[(run, pair)], role)] for role in _ROLES},
            "effects": {}, "contrasts": {}, "identity_controls": {},
        }
        for short, metric in (("mean", "mean_logprob"), ("total", "total_logprob")):
            effects = {}
            for category in categories:
                shifts = {
                    role: cell[(role, category)]["margins"][short] - baselines[(run, pair, role, "unpatched")]["margins"][short]
                    for role in _ROLES
                }
                effects[category] = _bundle_effect(shifts["clean"], shifts["corrupt"])
            record["effects"][metric] = effects
            record["contrasts"][metric] = {
                f"exact_minus_{category}": _bundle_effect(
                    effects["exact"]["clean_shift"] - effects[category]["clean_shift"],
                    effects["exact"]["corrupt_shift"] - effects[category]["corrupt_shift"],
                )
                for category in categories if category != "exact"
            }
            record["identity_controls"][metric] = {
                treatment: {
                    role: (baselines[(run, pair, role, treatment)] if treatment == "unpatched_rerun" else cell[(role, treatment)])["margins"][short]
                    - baselines[(run, pair, role, "unpatched")]["margins"][short]
                    for role in _ROLES
                }
                for treatment in controls
            }
        record["max_identity_candidate_score_delta"] = max(
            abs((baselines[(run, pair, role, treatment)] if treatment == "unpatched_rerun" else cell[(role, treatment)])["scores"][candidate][field]
                - baselines[(run, pair, role, "unpatched")]["scores"][candidate][field])
            for treatment in controls for role in _ROLES
            for candidate in ("CORRECT", "INCORRECT") for field in ("mean_logprob", "total_logprob")
        )
        per_pair.append(record)
        panels[(run, site)].append(record)
    orphan_baselines = {(run, pair) for run, pair, _, _ in baselines} - {(run, pair) for run, pair, _ in cells}
    if orphan_baselines:
        raise ValueError(f"shared baseline pairs have no bundle rows: {sorted(orphan_baselines)}")
    summaries = []
    for (run, site), records in sorted(panels.items()):
        categories = _BUNDLE_PROFILE_CATEGORIES[profiles[run]]
        estimands = [
            (metric, section, category, effect)
            for metric in ("mean_logprob", "total_logprob")
            for section, names in (("effects", categories), ("contrasts", tuple(f"exact_minus_{category}" for category in categories if category != "exact")))
            for category in names for effect in ("clean_shift", "corrupt_shift", "t")
        ]
        values = np.asarray([[record[section][metric][category][effect] for metric, section, category, effect in estimands] for record in records])
        indices = paired_cluster_bootstrap_indices(len(records), seed=seed, n_resamples=n_resamples)
        bootstrap = paired_cluster_bootstrap_mean_ci(values, indices=indices, seed=seed, alpha=alpha)
        summary = {
            "run_id": run, "site": site, "diagnostics_profile": profiles[run], **site_specs[(run, site)], "n_pairs": len(records),
            "pair_ids": [record["pair_id"] for record in records],
            "donor_reuse_counts": dict(sorted(Counter(record["donor_pair_id"] for record in records).items())),
            "mean_logprob": {}, "total_logprob": {},
            "max_identity_candidate_score_delta": max(record["max_identity_candidate_score_delta"] for record in records),
            "paired_bootstrap": {"estimands": [".".join(value) for value in estimands], **bootstrap},
        }
        for index, (metric, _section, category, effect) in enumerate(estimands):
            summary[metric].setdefault(category, {})[effect] = {
                "mean": bootstrap["point"][index], "ci_low": bootstrap["ci_low"][index],
                "ci_high": bootstrap["ci_high"][index],
                "bootstrap_standard_error": bootstrap["bootstrap_standard_error"][index],
            }
        summaries.append(summary)
    return {
        "schema_version": 1, "analysis": "last_token_all_layer_output_bundle",
        "definition": "shift = patched margin - unpatched receiver margin; T = (clean shift - corrupt shift) / 2",
        "primary_metric": "mean_logprob",
        "interpretation": "Donor-aligned intervention effects and paired control contrasts; no semantic-code identification.",
        "bootstrap": {"seed": seed, "n_resamples": n_resamples, "confidence_level": 1.0 - alpha, "intervals": "marginal two-sided percentile", "receiver_roles_resampled_jointly": True, "conditional_on_observed_donor_matching": True},
        "counts": {"input_rows": input_count, "bundle_rows": sum(len(cell) for cell in cells.values()), "shared_rows": len(baselines), "expected_scoring_forwards": 2 * (sum(len(cell) for cell in cells.values()) + len(baselines)), "ignored_treatments": dict(sorted(ignored.items()))},
        "per_pair": per_pair, "summaries": summaries,
    }


def _bundle_effect(clean_shift: float, corrupt_shift: float) -> dict[str, float]:
    return {"clean_shift": clean_shift, "corrupt_shift": corrupt_shift,
            "clean_aligned": clean_shift, "corrupt_aligned": -corrupt_shift,
            "t": (clean_shift - corrupt_shift) / 2.0}


def analyze_donor_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    cluster_by_source: bool = False,
) -> dict[str, Any]:
    """Validate complete 2 x 2 donor-role rows and summarize paired D.

    Required rows have ``treatment='donor_role'``, a nonempty ``run_id``,
    ``pair_id``, ``donor_pair_id``, ``site``, clean/corrupt receiver and donor
    variants, both mean and total margins, and full ``CandidateScore`` mappings
    for CORRECT and INCORRECT. ``metadata.receiver_identity`` and
    ``metadata.donor_identity`` require ``sample_id``, ``input_sha256`` and
    ``source_component``. Additional identity fields are retained and checked.

    Every observed (run, receiver pair, site) must contain exactly four rows.
    The donor pair must be external and from a different source component.
    Inputs for each (run, pair, role) must be identical everywhere, including
    when a pair is reused as a donor. Runs are always summarized separately.
    Non-donor treatments are counted and excluded, so same-batch control rows
    may be passed directly without entering D.

    Percentile intervals use receiver-pair resampling with shared draws for
    both roles and both score normalizations. The common estimate is the
    arithmetic mean of the two receiver-role D values within each pair.
    Optional source-component resampling retains all receiver pairs in each
    sampled component and computes a pair-weighted mean. Donor assignments
    stay attached to their receiver pairs; intervals are conditional on the
    observed matching, not a model of new donor selection. No simultaneous
    coverage, equivalence test or semantic-code interpretation is implied.
    """

    _validate_options(n_resamples, seed, alpha, cluster_by_source)
    grouped: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    identities: dict[tuple[str, str, str], dict[str, Any]] = {}
    components: dict[tuple[str, str], str] = {}
    candidate_counts: dict[tuple[str, str, str], tuple[int, int]] = {}
    donors: dict[tuple[str, str], str] = {}
    ignored: Counter[str] = Counter()
    input_count = 0

    for row_index, row in enumerate(rows):
        input_count += 1
        if not isinstance(row, Mapping):
            raise TypeError(f"row {row_index} must be a mapping")
        treatment = row.get("treatment")
        if treatment != "donor_role":
            ignored[str(treatment) if treatment is not None else "<missing>"] += 1
            continue
        run = _required_string(row, "run_id", f"row {row_index}")
        pair = _required_string(row, "pair_id", f"row {row_index}")
        donor_pair = _required_string(row, "donor_pair_id", f"row {row_index}")
        site = _required_string(row, "site", f"row {row_index}")
        receiver = _role(row.get("receiver_variant"), "receiver_variant", row_index)
        donor = _role(row.get("donor_variant"), "donor_variant", row_index)
        if pair == donor_pair:
            raise ValueError(f"row {row_index}: donor_pair_id must identify an external pair")
        donor_key = (run, pair)
        if donor_key in donors and donors[donor_key] != donor_pair:
            raise ValueError(f"mixed donor_pair_id for receiver {donor_key}")
        donors[donor_key] = donor_pair
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"row {row_index} requires metadata identities")
        receiver_identity = _identity(metadata, "receiver_identity", pair, receiver, row_index)
        donor_identity = _identity(metadata, "donor_identity", donor_pair, donor, row_index)
        if receiver_identity["source_component"] == donor_identity["source_component"]:
            raise ValueError(f"row {row_index}: donor and receiver share a source component")
        if "source_component" in row and row["source_component"] != receiver_identity["source_component"]:
            raise ValueError(f"row {row_index}: top-level source_component disagrees with receiver identity")
        _register_identity(identities, components, (run, pair, receiver), receiver_identity)
        _register_identity(identities, components, (run, donor_pair, donor), donor_identity)
        margins, counts = _validated_scores(row, row_index)
        count_key = (run, pair, receiver)
        if count_key in candidate_counts and candidate_counts[count_key] != counts:
            raise ValueError(f"mixed candidate token counts for receiver {count_key}")
        candidate_counts[count_key] = counts
        group_key = (run, pair, site)
        role_key = (receiver, donor)
        if role_key in grouped[group_key]:
            raise ValueError(f"duplicate donor-role row for {group_key}, {role_key}")
        grouped[group_key][role_key] = {
            "margins": margins,
            "scores": {candidate: dict(score) for candidate, score in row["scores"].items()},
            "receiver_identity": receiver_identity,
            "donor_identity": donor_identity,
        }

    if not grouped:
        raise ValueError("no donor_role rows were found")
    per_pair: list[dict[str, Any]] = []
    for (run, pair, site), combinations in sorted(grouped.items()):
        missing = sorted(_COMBINATIONS - combinations.keys())
        if missing:
            raise ValueError(f"incomplete 2x2 donor-role rows for {(run, pair, site)}; missing {missing}")
        donor_pair = donors[(run, pair)]
        record: dict[str, Any] = {
            "run_id": run,
            "pair_id": pair,
            "donor_pair_id": donor_pair,
            "site": site,
            "source_component": components[(run, pair)],
            "donor_source_component": components[(run, donor_pair)],
            "receiver_identities": {role: identities[(run, pair, role)] for role in _ROLES},
            "donor_identities": {role: identities[(run, donor_pair, role)] for role in _ROLES},
            "four_combinations": {
                receiver: {
                    donor: {
                        **combinations[(receiver, donor)]["margins"],
                        "scores": combinations[(receiver, donor)]["scores"],
                    }
                    for donor in _ROLES
                }
                for receiver in _ROLES
            },
        }
        for metric in ("mean", "total"):
            for receiver in _ROLES:
                record[f"d_{receiver}_{metric}"] = (
                    combinations[(receiver, "corrupt")]["margins"][metric]
                    - combinations[(receiver, "clean")]["margins"][metric]
                )
            record[f"d_common_{metric}"] = (
                record[f"d_clean_{metric}"] + record[f"d_corrupt_{metric}"]
            ) / 2.0
        per_pair.append(record)

    panels: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in per_pair:
        panels[(record["run_id"], record["site"])].append(record)
    summaries = []
    for (run, site), records in sorted(panels.items()):
        values = np.asarray([[record[key] for key in _ESTIMANDS] for record in records])
        indices = paired_cluster_bootstrap_indices(len(records), seed=seed, n_resamples=n_resamples)
        paired = paired_cluster_bootstrap_mean_ci(values, indices=indices, alpha=alpha, seed=seed)
        summary: dict[str, Any] = {
            "run_id": run,
            "site": site,
            "n_pairs": len(records),
            "n_source_components": len({record["source_component"] for record in records}),
            "n_unique_donor_pairs": len({record["donor_pair_id"] for record in records}),
            "donor_reuse_counts": dict(sorted(Counter(record["donor_pair_id"] for record in records).items())),
            "pair_ids": [record["pair_id"] for record in records],
            **_named_intervals(paired),
            "paired_bootstrap": {"estimands": list(_ESTIMANDS), **paired},
        }
        if cluster_by_source:
            cluster_result = _source_bootstrap(records, values, seed=seed, n_resamples=n_resamples, alpha=alpha)
            summary["source_component_bootstrap"] = {
                **_named_intervals(cluster_result),
                "estimands": list(_ESTIMANDS),
                **cluster_result,
            }
        summaries.append(summary)

    return {
        "schema_version": 1,
        "analysis": "fixed_receiver_external_donor_role_D",
        "definition": "D = m(receiver <- corrupt donor) - m(receiver <- clean donor)",
        "common_definition": "within-pair mean of clean-receiver D and corrupt-receiver D",
        "primary_metric": "mean_logprob",
        "interpretation": "Donor-role directionality under this intervention; not proof of a correctness-specific code.",
        "bootstrap": {
            "seed": seed,
            "n_resamples": n_resamples,
            "confidence_level": 1.0 - alpha,
            "receiver_roles_resampled_jointly": True,
            "donor_assignments_attached_to_receiver_pairs": True,
            "conditional_on_observed_donor_matching": True,
            "source_component_sensitivity": cluster_by_source,
            "intervals": "marginal two-sided percentile",
        },
        "counts": {
            "input_rows": input_count,
            "donor_role_rows": len(per_pair) * 4,
            "expected_scoring_forwards": len(per_pair) * 8,
            "scoring_forward_convention": "one forward per candidate, two candidates per score row; capture/control forwards excluded",
            "receiver_pair_site_cells": len(per_pair),
            "runs": len({record["run_id"] for record in per_pair}),
            "ignored_treatments": dict(sorted(ignored.items())),
        },
        "per_pair": per_pair,
        "summaries": summaries,
    }


def _validate_options(n_resamples: int, seed: int, alpha: float, cluster_by_source: bool) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not isinstance(cluster_by_source, bool):
        raise TypeError("cluster_by_source must be a boolean")


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires nonempty {key}")
    return value


def _role(value: Any, name: str, row_index: int) -> str:
    if not isinstance(value, str) or value not in _ROLES:
        raise ValueError(f"row {row_index}: {name} must be clean or corrupt")
    return value


def _identity(metadata: Mapping[str, Any], field: str, pair: str, role: str, row_index: int) -> dict[str, Any]:
    value = metadata.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"row {row_index} requires metadata.{field}")
    identity = dict(value)
    for key in _IDENTITY_FIELDS:
        _required_string(identity, key, f"row {row_index} {field}")
    for key, expected in (("pair_id", pair), ("variant", role)):
        if key in identity and identity[key] != expected:
            raise ValueError(f"row {row_index} {field}.{key} disagrees with row")
    return identity


def _register_identity(
    identities: dict[tuple[str, str, str], dict[str, Any]],
    components: dict[tuple[str, str], str],
    key: tuple[str, str, str],
    identity: dict[str, Any],
) -> None:
    if key in identities and identities[key] != identity:
        raise ValueError(f"mixed input identities for {key}")
    identities[key] = identity
    pair_key = key[:2]
    component = identity["source_component"]
    if pair_key in components and components[pair_key] != component:
        raise ValueError(f"mixed source components for pair {pair_key}")
    components[pair_key] = component


def _number(value: Any, label: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _validated_scores(row: Mapping[str, Any], row_index: int) -> tuple[dict[str, float], tuple[int, int]]:
    scores = row.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != {"CORRECT", "INCORRECT"}:
        raise ValueError(f"row {row_index} requires exactly CORRECT and INCORRECT candidate scores")
    values: dict[str, dict[str, float]] = {}
    counts = []
    for candidate in ("CORRECT", "INCORRECT"):
        score = scores[candidate]
        if not isinstance(score, Mapping) or score.get("candidate") != candidate:
            raise ValueError(f"row {row_index}: candidate score label disagrees with {candidate}")
        count = score.get("token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"row {row_index}: candidate token_count must be a positive integer")
        counts.append(count)
        mean = _number(score.get("mean_logprob"), f"row {row_index} mean_logprob")
        total = _number(score.get("total_logprob"), f"row {row_index} total_logprob")
        if not math.isclose(total, mean * count, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"row {row_index}: total_logprob disagrees with mean_logprob * token_count")
        values[candidate] = {"mean": mean, "total": total}
    margins = {}
    for metric in ("mean", "total"):
        margin = _number(row.get(_MARGIN_PREFIX + metric), f"row {row_index} {metric} margin")
        expected = values["INCORRECT"][metric] - values["CORRECT"][metric]
        if not math.isclose(margin, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"row {row_index}: {metric} margin disagrees with candidate scores")
        margins[metric] = margin
    return margins, (counts[0], counts[1])


def _named_intervals(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, metric in ((0, "mean_logprob"), (3, "total_logprob")):
        output[metric] = {
            role: {
                "mean": result["point"][offset + index],
                "ci_low": result["ci_low"][offset + index],
                "ci_high": result["ci_high"][offset + index],
                "bootstrap_standard_error": result["bootstrap_standard_error"][offset + index],
            }
            for index, role in enumerate(("clean_receiver", "corrupt_receiver", "common"))
        }
    return output


def _source_bootstrap(
    records: list[dict[str, Any]],
    values: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
    alpha: float,
) -> dict[str, Any]:
    components = sorted({record["source_component"] for record in records})
    members = [
        [index for index, record in enumerate(records) if record["source_component"] == component]
        for component in components
    ]
    sums = np.asarray([values[member].sum(axis=0) for member in members])
    sizes = np.asarray([len(member) for member in members])
    indices = paired_cluster_bootstrap_indices(len(components), seed=seed, n_resamples=n_resamples)
    replicates = np.empty((n_resamples, values.shape[1]))
    chunk_size = max(1, 2_000_000 // (len(components) * values.shape[1]))
    for start in range(0, n_resamples, chunk_size):
        draws = indices[start : start + chunk_size]
        replicates[start : start + chunk_size] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    lower, upper = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return {
        "method": "receiver_source_component_percentile",
        "draw_method": "sample_receiver_components_with_replacement_keep_all_member_pairs",
        "weighting": "receiver-pair weighted, including within each resample",
        "rng": "numpy.random.Generator(PCG64)",
        "seed": seed,
        "n_resamples": n_resamples,
        "confidence_level": 1.0 - alpha,
        "n_pairs": len(records),
        "n_clusters": len(components),
        "source_components": components,
        "cluster_pair_counts": sizes.tolist(),
        "indices_sha256": paired_cluster_indices_sha256(indices),
        "point": values.mean(axis=0).tolist(),
        "ci_low": lower.tolist(),
        "ci_high": upper.tolist(),
        "bootstrap_standard_error": np.std(replicates, axis=0, ddof=1).tolist(),
        "bootstrap_covariance": np.cov(replicates, rowvar=False, ddof=1).tolist(),
    }
