"""Deterministic CPU-only secondary analysis of frozen discovery patch rows.

This module consumes only an already-finalized discovery run. It never resolves or
opens the split manifest named by the run metadata, and it writes derived outputs to
an independent directory rather than modifying the hashed run root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from routeverify.analysis.bootstrap import (
    paired_cluster_bootstrap_indices,
    paired_cluster_indices_sha256,
)
from routeverify.instrumentation.storage import atomic_write_json, atomic_write_jsonl

DIRECTIONS = ("clean_to_corrupt", "corrupt_to_clean")
ALPHAS = (0.0, 0.5, 1.0, 2.0)
NONZERO_ALPHAS = (0.5, 1.0, 2.0)
METRICS = ("mean_logprob", "total_logprob")
EXPECTED_RECEIVER = {
    "clean_to_corrupt": "corrupt",
    "corrupt_to_clean": "clean",
}
EXPECTED_DONOR = {
    "clean_to_corrupt": "clean",
    "corrupt_to_clean": "corrupt",
}
ORIENTATION = {
    "clean_to_corrupt": -1.0,
    "corrupt_to_clean": 1.0,
}
PRIMARY_ALPHA = 1.0
EXPECTED_SITE = "bundle.router_logits[16]"
_ERROR_TYPE_PATTERN = re.compile(r"^(?P<error_type>.+?)_prm_(?:test|train)")
_SOURCE_HASHED_INPUTS = (
    "patch/predictions.jsonl",
    "patch/data_manifest.json",
    "second_batch_review.json",
    "run_manifest.json",
)
_ASSOCIATION_MIN_VALID_BOOTSTRAP_FRACTION = 0.95
_RESAMPLE_CHUNK_SIZE = 250
_IDENTITY_SCORE_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class DiscoveryEffectArtifacts:
    """Machine-readable analysis and its normalized pair-level rows."""

    analysis: dict[str, Any]
    pair_effect_rows: list[dict[str, Any]]
    source_run_root: Path


def write_discovery_effect_artifacts(
    artifacts: DiscoveryEffectArtifacts,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Atomically write one immutable derived result directory."""

    import matplotlib as mpl

    from routeverify.analysis import bootstrap as bootstrap_module
    from routeverify.analysis import discovery_plots

    output = Path(output_dir)
    source_run_root = artifacts.source_run_root.resolve(strict=True)
    resolved_output = output.resolve(strict=False)
    if resolved_output == source_run_root or source_run_root in resolved_output.parents:
        raise ValueError("discovery analysis output must be outside the frozen source run root")
    if output.exists():
        raise FileExistsError(f"refusing to reuse discovery analysis output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        figure_paths = discovery_plots.build_discovery_effect_plots(
            artifacts.analysis,
            artifacts.pair_effect_rows,
            staging / "figures",
        )
        normalized_figures = {
            name: {
                extension: str(Path(path).relative_to(staging))
                for extension, path in sorted(paths.items())
            }
            for name, paths in sorted(figure_paths.items())
        }
        analysis = dict(artifacts.analysis)
        analysis["analysis_environment"] = {
            "numpy_version": np.__version__,
            "matplotlib_version": mpl.__version__,
            "analysis_source_sha256": {
                "routeverify.analysis.bootstrap": _sha256(Path(str(bootstrap_module.__file__))),
                "routeverify.analysis.discovery_effects": _sha256(Path(__file__)),
                "routeverify.analysis.discovery_plots": _sha256(
                    Path(str(discovery_plots.__file__))
                ),
            },
            "determinism_scope": (
                "Byte-identical for identical input bytes, analysis source bytes, "
                "dependency versions, seed, and resampling parameters."
            ),
        }
        analysis["outputs"] = {
            "analysis": "analysis.json",
            "pair_effects_jsonl": "pair_effects.jsonl",
            "pair_effects_csv": "pair_effects.csv",
            "report": "report.md",
            "figures": normalized_figures,
            "sha256sums": "SHA256SUMS",
            "final_marker": "SUCCESS",
        }
        atomic_write_json(staging / "analysis.json", analysis)
        atomic_write_jsonl(staging / "pair_effects.jsonl", artifacts.pair_effect_rows)
        _atomic_write_csv(staging / "pair_effects.csv", artifacts.pair_effect_rows)
        _atomic_write_text(
            staging / "report.md",
            discovery_effect_markdown(analysis),
        )
        sums = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            if path.name == "SHA256SUMS":
                continue
            sums.append(f"{_sha256(path)}  {path.relative_to(staging).as_posix()}\n")
        _atomic_write_text(staging / "SHA256SUMS", "".join(sums))
        _atomic_write_text(staging / "SUCCESS", "")
        staging.replace(output)
        return analysis
    except BaseException:
        shutil.rmtree(staging)
        raise


def analyze_discovery_run(
    run_root: str | Path,
    *,
    seed: int = 20260731,
    n_resamples: int = 50_000,
    ci_alpha: float = 0.05,
    permutation_resamples: int = 50_000,
) -> DiscoveryEffectArtifacts:
    """Analyze one finalized OLMoE frozen-discovery run without opening validation."""

    root = Path(run_root).resolve(strict=True)
    predictions_path = root / "patch" / "predictions.jsonl"
    data_manifest_path = root / "patch" / "data_manifest.json"
    review_path = root / "second_batch_review.json"
    run_manifest_path = root / "run_manifest.json"
    required = (
        predictions_path,
        data_manifest_path,
        review_path,
        run_manifest_path,
        root / "SUCCESS",
        root / "SHA256SUMS",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"discovery run is not finalized: missing {missing}")
    if (root / "FAILED").exists():
        raise ValueError("discovery run contains FAILED")
    if (root / "SUCCESS").is_symlink():
        raise ValueError("discovery SUCCESS marker must not be a symlink")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if permutation_resamples <= 0:
        raise ValueError("permutation_resamples must be positive")
    if not 0.0 < ci_alpha < 1.0:
        raise ValueError("ci_alpha must be between 0 and 1")

    source_hash_verification = _verify_source_sha256sums(root)
    review = _read_json(review_path)
    run_manifest = _read_json(run_manifest_path)
    data_manifest = _read_json(data_manifest_path)
    if review.get("passed") is not True:
        raise ValueError("second_batch_review.json is not PASS")
    if review.get("run_role") != "frozen_discovery":
        raise ValueError("second_batch_review.json run_role is not frozen_discovery")
    if run_manifest.get("success") is not True or run_manifest.get("exit_code") != 0:
        raise ValueError("run_manifest.json is not a successful zero-exit run")
    protocol = data_manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("patch data manifest has no protocol mapping")
    if protocol.get("split_partition") != "discovery":
        raise ValueError("CPU secondary analysis only accepts the discovery partition")
    if protocol.get("alphas") != list(ALPHAS):
        raise ValueError(f"unexpected alpha contract: {protocol.get('alphas')!r}")
    if protocol.get("patch_mode") != "simultaneous_bundle":
        raise ValueError("expected simultaneous_bundle patch mode")
    if protocol.get("expected_router_hooks") != 16:
        raise ValueError("expected exactly 16 simultaneous router hooks")

    predictions_sha256 = _sha256(predictions_path)
    if review.get("predictions_sha256") != predictions_sha256:
        raise ValueError(
            "second_batch_review.json predictions_sha256 does not match predictions input"
        )
    prediction_rows = _read_jsonl(predictions_path)
    pair_effect_rows = build_pair_effect_rows(prediction_rows)
    pair_ids = sorted({str(row["pair_id"]) for row in pair_effect_rows})
    expected_pairs = int(data_manifest.get("num_pairs", -1))
    if expected_pairs != len(pair_ids):
        raise ValueError(f"manifest/prediction pair mismatch: {expected_pairs} != {len(pair_ids)}")
    if len(prediction_rows) != expected_pairs * 14:
        raise ValueError(
            f"unexpected row contract: {len(prediction_rows)} != {expected_pairs} * 14"
        )
    if review.get("expected_pairs") != len(pair_ids):
        raise ValueError(
            "second_batch_review.json expected_pairs does not match predictions input: "
            f"{review.get('expected_pairs')!r} != {len(pair_ids)}"
        )

    pair_order_hash = hashlib.sha256(("\n".join(pair_ids) + "\n").encode("utf-8")).hexdigest()
    bootstrap_indices = paired_cluster_bootstrap_indices(
        len(pair_ids),
        seed=seed,
        n_resamples=n_resamples,
    )
    bootstrap_indices_hash = paired_cluster_indices_sha256(bootstrap_indices)
    matrices = _pair_matrices(pair_effect_rows, pair_ids)

    cell_effects: dict[str, Any] = {}
    symmetric_effects: dict[str, Any] = {}
    dose_response: dict[str, Any] = {}
    for metric in METRICS:
        metric_matrix = matrices[f"effect_{metric}"]
        cell_effects[metric] = _cell_effect_summaries(
            metric_matrix,
            bootstrap_indices,
            ci_alpha=ci_alpha,
        )
        symmetric_matrix = _symmetric_matrix(metric_matrix)
        symmetric_effects[metric] = _named_alpha_summaries(
            symmetric_matrix,
            bootstrap_indices,
            ci_alpha=ci_alpha,
        )
        dose_response[metric] = _dose_response_summaries(
            metric_matrix,
            bootstrap_indices,
            ci_alpha=ci_alpha,
        )

    manipulation = _manipulation_analysis(
        matrices,
        bootstrap_indices,
        ci_alpha=ci_alpha,
        bootstrap_indices_hash=bootstrap_indices_hash,
    )
    heterogeneity_seed = _derived_seed(seed, "error_type_heterogeneity")
    error_type_heterogeneity = _error_type_analysis(
        pair_effect_rows,
        pair_ids,
        bootstrap_seed=heterogeneity_seed,
        n_resamples=n_resamples,
        ci_alpha=ci_alpha,
        permutation_resamples=permutation_resamples,
    )
    mean_total_sensitivity = _mean_total_sensitivity(
        matrices,
        cell_effects=cell_effects,
        pair_effect_rows=pair_effect_rows,
    )

    analysis = {
        "schema_version": 1,
        "analysis_role": "posthoc_cpu_secondary_on_frozen_discovery_rows",
        "validation_access": {
            "accessed": False,
            "policy": (
                "Only the finalized discovery run root was read. The split-manifest "
                "path recorded in provenance was not resolved or opened."
            ),
        },
        "source": {
            "run_id": root.name,
            "run_role": review.get("run_role"),
            "execution_commit": run_manifest.get("code_commit"),
            "num_prediction_rows": len(prediction_rows),
            "num_pairs": len(pair_ids),
            "predictions_sha256": predictions_sha256,
            "data_manifest_sha256": _sha256(data_manifest_path),
            "review_sha256": _sha256(review_path),
            "run_manifest_sha256": _sha256(run_manifest_path),
            "source_hash_verification": source_hash_verification,
            "pair_order_sha256": pair_order_hash,
            "registered_alphas": list(ALPHAS),
            "site": EXPECTED_SITE,
            "router_hook_count": 16,
        },
        "estimands": {
            "primary_outcome": (
                "mean_logprob(INCORRECT) - mean_logprob(CORRECT), nats per target token"
            ),
            "secondary_outcome": (
                "total_logprob(INCORRECT) - total_logprob(CORRECT), nats per target sequence"
            ),
            "cell_effect": ("patched margin minus the same pair/receiver unpatched margin"),
            "direction_orientation": {
                "clean_to_corrupt": (
                    "raw effect is primary reporting scale; negative is expected. "
                    "Oriented display effect is -1 times raw."
                ),
                "corrupt_to_clean": (
                    "raw effect is primary reporting scale; positive is expected. "
                    "Oriented display effect equals raw."
                ),
            },
            "symmetric_display_effect": (
                "0.5 * (corrupt_to_clean raw effect - clean_to_corrupt raw effect); "
                "exploratory here and positive in the expected transfer direction"
            ),
            "association_warning": (
                "Top-k/KL are post-treatment manipulation diagnostics. Associations "
                "are not causal mediation estimates, and the 16 layers are one joint "
                "simultaneous intervention."
            ),
        },
        "bootstrap": {
            "cluster_unit": "pair_id",
            "num_clusters": len(pair_ids),
            "seed": seed,
            "n_resamples": n_resamples,
            "ci_alpha": ci_alpha,
            "pointwise_method": "pair-cluster percentile bootstrap",
            "simultaneous_method": (
                "max-absolute standardized shared-draw bootstrap across the six "
                "nonzero direction-by-alpha cells"
            ),
            "shared_draws": True,
            "indices_sha256": bootstrap_indices_hash,
            "rng": "numpy.random.Generator(PCG64)",
            "superpopulation_caveat": (
                f"The {len(pair_ids)} frozen pairs are a finite discovery set. "
                "Bootstrap intervals model exchangeability beyond that set; they are "
                "not design-based confidence intervals from probability sampling."
            ),
        },
        "cell_effects": cell_effects,
        "symmetric_effects": symmetric_effects,
        "dose_response": dose_response,
        "manipulation": manipulation,
        "error_type_heterogeneity": error_type_heterogeneity,
        "mean_total_sensitivity": mean_total_sensitivity,
        "multiplicity": {
            "primary_family": (
                "The six nonzero mean-logprob direction-by-alpha cell effects use "
                "one shared-draw simultaneous interval family."
            ),
            "secondary_total_family": (
                "The six total-logprob cells use a separate simultaneous interval "
                "family and remain secondary sensitivity results."
            ),
            "unadjusted_exploratory_families": [
                "symmetric alpha summaries",
                "dose-response slopes and categorical contrasts",
                "manipulation diagnostic cell summaries",
                "manipulation-outcome associations and within-alpha rank correlations",
                "error-type group intervals",
                "mean-versus-total agreement summaries",
            ],
            "decision_policy": (
                "Unadjusted exploratory intervals and associations are descriptive; "
                "they are not thresholded for confirmatory decisions."
            ),
        },
        "interpretation_guards": [
            "Alpha zero, self-patch, and rerun rows are identity controls, not efficacy samples.",
            "No alpha, direction, error type, seed, or site was selected from these results.",
            "No equivalence claim is available because this run had no preregistered SESOI.",
            "Small error-type groups are descriptive only.",
            "Total-logprob is secondary and cannot replace the token-normalized primary outcome.",
        ],
    }
    return DiscoveryEffectArtifacts(
        analysis=analysis,
        pair_effect_rows=pair_effect_rows,
        source_run_root=root,
    )


def build_pair_effect_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the rectangular run contract and derive baseline-adjusted pair rows."""

    if not rows:
        raise ValueError("prediction rows must be non-empty")
    by_pair: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if not pair_id:
            raise ValueError("every prediction row must have a non-empty pair_id")
        by_pair.setdefault(pair_id, []).append(row)

    output: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(by_pair.items()):
        if len(pair_rows) != 14:
            raise ValueError(f"{pair_id} has {len(pair_rows)} rows; expected 14")
        error_type = _error_type(pair_id)
        baselines: dict[str, Mapping[str, Any]] = {}
        controls: dict[tuple[str, str], Mapping[str, Any]] = {}
        interventions: dict[tuple[str, float], Mapping[str, Any]] = {}
        for row in pair_rows:
            treatment = str(row.get("treatment"))
            receiver = str(row.get("receiver_variant"))
            if treatment == "unpatched":
                if receiver in baselines:
                    raise ValueError(f"duplicate unpatched row for {(pair_id, receiver)}")
                if row.get("site") is not None:
                    raise ValueError("unpatched rows must not declare a site")
                if row.get("donor_variant") is not None:
                    raise ValueError("unpatched rows must not declare a donor")
                baselines[receiver] = row
            elif treatment in {"unpatched_rerun", "self_patch"}:
                control_key = (treatment, receiver)
                if control_key in controls:
                    raise ValueError(
                        f"duplicate identity-control row for {(pair_id, *control_key)}"
                    )
                controls[control_key] = row
            elif treatment in DIRECTIONS:
                metadata = row.get("metadata")
                if not isinstance(metadata, Mapping) or metadata.get("alpha") is None:
                    raise ValueError(f"{pair_id} intervention row has no alpha")
                alpha = float(metadata["alpha"])
                if alpha not in ALPHAS:
                    raise ValueError(f"{pair_id} has unexpected alpha {alpha}")
                intervention_key = (treatment, alpha)
                if intervention_key in interventions:
                    raise ValueError(
                        f"duplicate intervention row for {(pair_id, *intervention_key)}"
                    )
                if receiver != EXPECTED_RECEIVER[treatment]:
                    raise ValueError(f"{pair_id} {treatment} has receiver {receiver}")
                if str(row.get("donor_variant")) != EXPECTED_DONOR[treatment]:
                    raise ValueError(f"{pair_id} {treatment} has wrong donor")
                if row.get("site") != EXPECTED_SITE:
                    raise ValueError(f"{pair_id} {treatment} has unexpected site")
                if metadata.get("control") != "dose_response":
                    raise ValueError(f"{pair_id} {treatment} is not a dose-response row")
                if metadata.get("patch_mode") != "simultaneous_bundle":
                    raise ValueError(f"{pair_id} {treatment} does not use simultaneous_bundle")
                interventions[intervention_key] = row
            else:
                raise ValueError(f"{pair_id} has unexpected treatment {treatment}")

        if set(baselines) != {"clean", "corrupt"}:
            raise ValueError(f"{pair_id} does not have both unpatched receiver rows")
        expected_controls = {
            (treatment, receiver)
            for treatment in ("unpatched_rerun", "self_patch")
            for receiver in ("clean", "corrupt")
        }
        if set(controls) != expected_controls:
            raise ValueError(f"{pair_id} identity-control contract is incomplete")
        expected_interventions = {
            (direction, alpha) for direction in DIRECTIONS for alpha in ALPHAS
        }
        if set(interventions) != expected_interventions:
            raise ValueError(f"{pair_id} intervention contract is incomplete")

        token_counts = _validate_pair_token_counts(pair_id, pair_rows)
        for receiver, baseline in baselines.items():
            for treatment in ("unpatched_rerun", "self_patch"):
                _validate_identity_control(
                    pair_id=pair_id,
                    treatment=treatment,
                    receiver=receiver,
                    baseline=baseline,
                    control=controls[(treatment, receiver)],
                )

        for direction in DIRECTIONS:
            receiver = EXPECTED_RECEIVER[direction]
            baseline = baselines[receiver]
            baseline_mean = _margin(baseline, "mean_logprob")
            baseline_total = _margin(baseline, "total_logprob")
            for alpha in ALPHAS:
                row = interventions[(direction, alpha)]
                _validate_receiver_alignment(
                    pair_id=pair_id,
                    treatment=f"{direction} alpha={alpha:g}",
                    baseline=baseline,
                    intervention=row,
                )
                patched_mean = _margin(row, "mean_logprob")
                patched_total = _margin(row, "total_logprob")
                topk, router_kl, overlap = _manipulation_values(row)
                mean_effect = patched_mean - baseline_mean
                total_effect = patched_total - baseline_total
                if alpha == 0.0:
                    candidate_score_delta = _candidate_score_max_abs_delta(
                        baseline,
                        row,
                    )
                    if (
                        candidate_score_delta > _IDENTITY_SCORE_TOLERANCE
                        or abs(mean_effect) > _IDENTITY_SCORE_TOLERANCE
                        or abs(total_effect) > _IDENTITY_SCORE_TOLERANCE
                        or abs(topk) > 1e-12
                        or abs(router_kl) > 1e-12
                        or abs(overlap - 1.0) > 1e-12
                    ):
                        raise ValueError(
                            f"{pair_id} alpha-zero identity is not exact; "
                            f"candidate_score_max_abs_delta={candidate_score_delta:.17g}"
                        )
                output.append(
                    {
                        "pair_id": pair_id,
                        "error_type": error_type,
                        "correct_token_count": token_counts["CORRECT"],
                        "incorrect_token_count": token_counts["INCORRECT"],
                        "direction": direction,
                        "receiver_variant": receiver,
                        "donor_variant": EXPECTED_DONOR[direction],
                        "alpha": alpha,
                        "site": EXPECTED_SITE,
                        "baseline_mean_logprob_margin": baseline_mean,
                        "patched_mean_logprob_margin": patched_mean,
                        "effect_mean_logprob": mean_effect,
                        "oriented_effect_mean_logprob": (ORIENTATION[direction] * mean_effect),
                        "baseline_total_logprob_margin": baseline_total,
                        "patched_total_logprob_margin": patched_total,
                        "effect_total_logprob": total_effect,
                        "oriented_effect_total_logprob": (ORIENTATION[direction] * total_effect),
                        "topk_changed_rate": topk,
                        "topk_overlap": overlap,
                        "router_kl_receiver_to_patched": router_kl,
                    }
                )
    return output


def discovery_effect_markdown(analysis: Mapping[str, Any]) -> str:
    """Render a concise narrative report from machine-readable analysis."""

    source = _require_mapping(analysis, "source")
    cell_effects = _require_mapping(analysis, "cell_effects")
    primary = _require_mapping(cell_effects, "mean_logprob")
    symmetric = _require_mapping(
        _require_mapping(analysis, "symmetric_effects"),
        "mean_logprob",
    )
    dose_response = _require_mapping(
        _require_mapping(analysis, "dose_response"),
        "mean_logprob",
    )
    heterogeneity = _require_mapping(analysis, "error_type_heterogeneity")
    heterogeneity_groups = _require_mapping(heterogeneity, "groups")
    sensitivity = _require_mapping(analysis, "mean_total_sensitivity")
    manipulation = _require_mapping(analysis, "manipulation")
    associations = _require_mapping(manipulation, "outcome_associations")
    bootstrap = _require_mapping(analysis, "bootstrap")
    confidence_percent = 100.0 * (1.0 - float(bootstrap["ci_alpha"]))
    confidence_label = f"{confidence_percent:g}% CI"
    simultaneous_exclusions = 0
    for direction in DIRECTIONS:
        direction_payload = _require_mapping(primary, direction)
        for alpha in NONZERO_ALPHAS:
            entry = _require_mapping(direction_payload, _alpha_key(alpha))
            simultaneous_exclusions += int(_interval_excludes_zero(entry["simultaneous_ci"]))

    lines = [
        "# OLMoE frozen-discovery CPU secondary analysis",
        "",
        "## Scope",
        "",
        f"- Run: `{source['run_id']}`",
        f"- Execution commit: `{source['execution_commit']}`",
        f"- Pairs / prediction rows: `{source['num_pairs']}` / `{source['num_prediction_rows']}`",
        "- Analysis role: exploratory, discovery-only, CPU-only.",
        "- Validation configuration, IDs, samples, and results were not opened.",
        "- The intervention is one joint same-boundary-token bundle across all 16 router layers.",
        "",
        "## Pair-level effects",
        "",
        (
            "Effects are patched minus the same pair/receiver unpatched margin. "
            "The primary outcome is INCORRECT minus CORRECT mean log-probability "
            "per target token."
        ),
        "",
        (
            "| direction | alpha | mean effect | "
            f"pointwise {confidence_label} | simultaneous {confidence_label} |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        direction_payload = _require_mapping(primary, direction)
        for alpha in NONZERO_ALPHAS:
            entry = _require_mapping(direction_payload, _alpha_key(alpha))
            pointwise = entry["pointwise_ci"]
            simultaneous = entry["simultaneous_ci"]
            lines.append(
                "| "
                f"{direction} | {alpha:g} | {float(entry['mean']):+.6f} | "
                f"[{float(pointwise[0]):+.6f}, {float(pointwise[1]):+.6f}] | "
                f"[{float(simultaneous[0]):+.6f}, {float(simultaneous[1]):+.6f}] |"
            )

    lines.extend(
        [
            "",
            (
                "All six simultaneous intervals include zero."
                if simultaneous_exclusions == 0
                else (f"{simultaneous_exclusions} of six simultaneous intervals exclude zero.")
            ),
            (
                "This discovery result is not an equivalence result: the run had no "
                "preregistered SESOI."
            ),
            "",
            "## Direction-symmetric effect and dose trend",
            "",
            (
                "The positive-oriented symmetric effect is "
                "`0.5 * (corrupt_to_clean - clean_to_corrupt)`. These intervals "
                "are unadjusted exploratory summaries."
            ),
            "",
            f"| alpha | symmetric mean | pointwise {confidence_label} |",
            "|---:|---:|---:|",
        ]
    )
    for alpha in NONZERO_ALPHAS:
        entry = _require_mapping(symmetric, _alpha_key(alpha))
        interval = entry["pointwise_ci"]
        lines.append(
            f"| {alpha:g} | {float(entry['mean']):+.6f} | "
            f"[{float(interval[0]):+.6f}, {float(interval[1]):+.6f}] |"
        )
    lines.extend(
        [
            "",
            (f"| direction | nonzero-alpha linear slope | pointwise {confidence_label} |"),
            "|---|---:|---:|",
        ]
    )
    for direction in DIRECTIONS:
        direction_dose = _require_mapping(dose_response, direction)
        slope = _require_mapping(direction_dose, "nonzero_alpha_linear_slope")
        interval = slope["pointwise_ci"]
        lines.append(
            f"| {direction} | {float(slope['mean']):+.6f} | "
            f"[{float(interval[0]):+.6f}, {float(interval[1]):+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Manipulation strength versus outcome",
            "",
        ]
    )
    primary_associations = _require_mapping(associations, "mean_logprob")
    for direction in DIRECTIONS:
        payload = _require_mapping(primary_associations, direction)
        topk = _require_mapping(payload, "topk_changed_rate")
        kl = _require_mapping(payload, "router_kl_receiver_to_patched")
        lines.append(
            f"- `{direction}`: within-dose standardized top-k association "
            f"`{float(topk['estimate']):+.3f}` "
            f"({confidence_label} `{_format_ci(topk['ci'])}`); rank-KL association "
            f"`{float(kl['estimate']):+.3f}` "
            f"({confidence_label} `{_format_ci(kl['ci'])}`)."
        )
    lines.extend(
        [
            "",
            "These are post-treatment diagnostic associations, not causal mediation estimates.",
            "",
            "## Error-type heterogeneity",
            "",
            (
                f"The exploratory alpha=1 symmetric-effect omnibus permutation "
                f"test returned `p={float(heterogeneity['omnibus_permutation']['p_value']):.4g}`. "
                "Groups below 20 pairs are descriptive only."
            ),
            "",
            (f"| error type | n | symmetric mean | pointwise {confidence_label} | status |"),
            "|---|---:|---:|---:|---|",
        ]
    )
    for error_type, _raw_entry in sorted(
        heterogeneity_groups.items(),
        key=lambda item: (
            -int(_require_mapping(heterogeneity_groups, str(item[0]))["n"]),
            str(item[0]),
        ),
    ):
        entry = _require_mapping(heterogeneity_groups, str(error_type))
        interval = entry["pointwise_ci"]
        lines.append(
            f"| {error_type} | {int(entry['n'])} | {float(entry['mean']):+.6f} | "
            f"[{float(interval[0]):+.6f}, {float(interval[1]):+.6f}] | "
            f"{entry['inference_status']} |"
        )
    lines.extend(
        [
            "",
            "## Mean versus total log-probability",
            "",
            (
                f"Across the six nonzero cells, ATE direction agreement was "
                f"`{sensitivity['ate_direction_agreement_count']}/6`; pair-cell sign "
                f"agreement was `{float(sensitivity['pair_cell_sign_agreement']):.3f}`; "
                "the pooled within-cell standardized Spearman association was "
                f"`{float(sensitivity['standardized_pair_effect_spearman']):.3f}`."
            ),
            "",
            "Mean log-probability remains primary. Total log-probability is a secondary "
            "sensitivity analysis in different units and cannot rescue the primary result.",
            "",
            "## Interpretation",
            "",
            "Routing diagnostics and answer-margin effects are reported without selecting an "
            "alpha, direction, error type, seed, or site from these exploratory results. "
            "Unadjusted exploratory intervals are descriptive, and validation remains sealed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _pair_matrices(
    rows: Sequence[Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    pair_index = {pair_id: index for index, pair_id in enumerate(pair_ids)}
    arm_index = {
        (direction, alpha): index
        for index, (direction, alpha) in enumerate(
            (direction, alpha) for direction in DIRECTIONS for alpha in ALPHAS
        )
    }
    matrices = {
        "effect_mean_logprob": np.full((len(pair_ids), len(arm_index)), np.nan),
        "effect_total_logprob": np.full((len(pair_ids), len(arm_index)), np.nan),
        "topk_changed_rate": np.full((len(pair_ids), len(arm_index)), np.nan),
        "router_kl_receiver_to_patched": np.full((len(pair_ids), len(arm_index)), np.nan),
    }
    for row in rows:
        pair_id = str(row["pair_id"])
        direction = str(row["direction"])
        alpha = float(row["alpha"])
        row_index = pair_index[pair_id]
        column = arm_index[(direction, alpha)]
        for key in matrices:
            if not math.isnan(float(matrices[key][row_index, column])):
                raise ValueError(f"duplicate matrix cell for {(pair_id, direction, alpha)}")
            matrices[key][row_index, column] = float(row[key])
    for key, matrix in matrices.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{key} matrix is incomplete or non-finite")
    return matrices


def _cell_effect_summaries(
    matrix: np.ndarray,
    indices: np.ndarray,
    *,
    ci_alpha: float,
) -> dict[str, Any]:
    nonzero_columns = [
        _arm_column(direction, alpha) for direction in DIRECTIONS for alpha in NONZERO_ALPHAS
    ]
    simultaneous = _matrix_mean_summaries(
        matrix[:, nonzero_columns],
        indices,
        ci_alpha=ci_alpha,
        simultaneous=True,
    )
    simultaneous_by_column = {
        column: summary for column, summary in zip(nonzero_columns, simultaneous, strict=True)
    }
    output: dict[str, Any] = {}
    for direction in DIRECTIONS:
        by_alpha: dict[str, Any] = {}
        for alpha in ALPHAS:
            column = _arm_column(direction, alpha)
            summary = _matrix_mean_summaries(
                matrix[:, [column]],
                indices,
                ci_alpha=ci_alpha,
                simultaneous=False,
            )[0]
            if alpha in NONZERO_ALPHAS:
                summary["simultaneous_ci"] = simultaneous_by_column[column]["simultaneous_ci"]
            else:
                summary["simultaneous_ci"] = None
            summary["expected_raw_sign"] = (
                "negative" if direction == "clean_to_corrupt" else "positive"
            )
            by_alpha[_alpha_key(alpha)] = summary
        output[direction] = by_alpha
    return output


def _symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    clean_to_corrupt = matrix[:, 0 : len(ALPHAS)]
    corrupt_to_clean = matrix[:, len(ALPHAS) : 2 * len(ALPHAS)]
    return np.asarray(
        0.5 * (corrupt_to_clean - clean_to_corrupt),
        dtype=float,
    )


def _named_alpha_summaries(
    matrix: np.ndarray,
    indices: np.ndarray,
    *,
    ci_alpha: float,
) -> dict[str, Any]:
    summaries = _matrix_mean_summaries(
        matrix,
        indices,
        ci_alpha=ci_alpha,
        simultaneous=False,
    )
    return {_alpha_key(alpha): summary for alpha, summary in zip(ALPHAS, summaries, strict=True)}


def _dose_response_summaries(
    matrix: np.ndarray,
    indices: np.ndarray,
    *,
    ci_alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    x = np.asarray(NONZERO_ALPHAS, dtype=float)
    centered_x = x - float(np.mean(x))
    denominator = float(np.sum(centered_x**2))
    for direction in DIRECTIONS:
        columns = [_arm_column(direction, alpha) for alpha in NONZERO_ALPHAS]
        values = matrix[:, columns]
        slopes = np.sum(values * centered_x[None, :], axis=1) / denominator
        contrasts = np.column_stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]))
        output[direction] = {
            "alpha_treated_as": "categorical_primary; linear trend is secondary",
            "multiplicity": ("unadjusted exploratory summaries; no confirmatory thresholding"),
            "nonzero_alpha_linear_slope": _matrix_mean_summaries(
                slopes[:, None],
                indices,
                ci_alpha=ci_alpha,
                simultaneous=False,
            )[0],
            "paired_categorical_contrasts": {
                "alpha_1_minus_0.5": _matrix_mean_summaries(
                    contrasts[:, [0]],
                    indices,
                    ci_alpha=ci_alpha,
                    simultaneous=False,
                )[0],
                "alpha_2_minus_1": _matrix_mean_summaries(
                    contrasts[:, [1]],
                    indices,
                    ci_alpha=ci_alpha,
                    simultaneous=False,
                )[0],
            },
        }
    return output


def _manipulation_analysis(
    matrices: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    ci_alpha: float,
    bootstrap_indices_hash: str,
) -> dict[str, Any]:
    cell_summaries: dict[str, Any] = {}
    for measure in ("topk_changed_rate", "router_kl_receiver_to_patched"):
        matrix = matrices[measure]
        by_direction: dict[str, Any] = {}
        for direction in DIRECTIONS:
            by_alpha: dict[str, Any] = {}
            for alpha in ALPHAS:
                column = _arm_column(direction, alpha)
                by_alpha[_alpha_key(alpha)] = _matrix_mean_summaries(
                    matrix[:, [column]],
                    indices,
                    ci_alpha=ci_alpha,
                    simultaneous=False,
                )[0]
            by_direction[direction] = by_alpha
        cell_summaries[measure] = by_direction

    outcome_associations: dict[str, Any] = {}
    for metric in METRICS:
        metric_output: dict[str, Any] = {}
        effect_matrix = matrices[f"effect_{metric}"]
        for direction in DIRECTIONS:
            columns = [_arm_column(direction, alpha) for alpha in NONZERO_ALPHAS]
            oriented_outcome = ORIENTATION[direction] * effect_matrix[:, columns]
            direction_output: dict[str, Any] = {}
            for measure in ("topk_changed_rate", "router_kl_receiver_to_patched"):
                manipulation_matrix = matrices[measure][:, columns]
                rank_transform = measure == "router_kl_receiver_to_patched"
                estimate = _within_alpha_zscore_pooled_correlation(
                    manipulation_matrix,
                    oriented_outcome,
                    rank_x=rank_transform,
                )
                bootstrap_values, zero_variance = _bootstrap_within_alpha_zscore_pooled_correlation(
                    manipulation_matrix,
                    oriented_outcome,
                    indices,
                    rank_x=rank_transform,
                )
                direction_output[measure] = {
                    "estimate": estimate,
                    "ci": _quantile_ci(bootstrap_values, ci_alpha),
                    "estimand": (
                        "Equal-weight pooled correlation after standardizing x and "
                        "the oriented outcome separately within each nonzero alpha."
                    ),
                    "transform": (
                        "within-alpha x midrank for KL only; then within-alpha "
                        "sample z-score (ddof=1) for x and y"
                    ),
                    "bootstrap_reestimation": (
                        "Pair-cluster resampling with KL midranks and both within-alpha "
                        "z-score transforms recomputed inside every replicate."
                    ),
                    "bootstrap_indices_sha256": bootstrap_indices_hash,
                    "bootstrap_zero_variance": zero_variance,
                    "multiplicity": (
                        "unadjusted exploratory diagnostic association; no "
                        "confirmatory thresholding"
                    ),
                    "alpha_zero_excluded": True,
                    "within_alpha_spearman": {
                        _alpha_key(alpha): _spearman(
                            manipulation_matrix[:, alpha_index],
                            oriented_outcome[:, alpha_index],
                        )
                        for alpha_index, alpha in enumerate(NONZERO_ALPHAS)
                    },
                }
            metric_output[direction] = direction_output
        outcome_associations[metric] = metric_output
    return {
        "cell_summaries": cell_summaries,
        "outcome_associations": outcome_associations,
        "causal_interpretation": (
            "forbidden: manipulation diagnostics are post-treatment and associations "
            "do not identify mediation"
        ),
    }


def _error_type_analysis(
    rows: Sequence[Mapping[str, Any]],
    pair_ids: Sequence[str],
    *,
    bootstrap_seed: int,
    n_resamples: int,
    ci_alpha: float,
    permutation_resamples: int,
) -> dict[str, Any]:
    by_pair: dict[str, dict[tuple[str, float], float]] = {}
    error_types: dict[str, str] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        error_types[pair_id] = str(row["error_type"])
        by_pair.setdefault(pair_id, {})[(str(row["direction"]), float(row["alpha"]))] = float(
            row["effect_mean_logprob"]
        )
    values = np.asarray(
        [
            0.5
            * (
                by_pair[pair_id][("corrupt_to_clean", PRIMARY_ALPHA)]
                - by_pair[pair_id][("clean_to_corrupt", PRIMARY_ALPHA)]
            )
            for pair_id in pair_ids
        ],
        dtype=float,
    )
    labels = np.asarray([error_types[pair_id] for pair_id in pair_ids], dtype=object)
    omnibus = _permutation_oneway(
        values,
        labels,
        seed=bootstrap_seed,
        n_resamples=permutation_resamples,
    )
    groups: dict[str, Any] = {}
    for error_type in sorted(set(str(label) for label in labels)):
        group_values = values[labels == error_type]
        group_seed = _derived_seed(bootstrap_seed, error_type)
        group_indices = paired_cluster_bootstrap_indices(
            int(group_values.size),
            seed=group_seed,
            n_resamples=n_resamples,
        )
        summary = _matrix_mean_summaries(
            group_values[:, None],
            group_indices,
            ci_alpha=ci_alpha,
            simultaneous=False,
        )[0]
        summary["inference_status"] = (
            "exploratory_interval" if group_values.size >= 20 else "descriptive_only_small_n"
        )
        summary["bootstrap_seed"] = group_seed
        summary["bootstrap_indices_sha256"] = paired_cluster_indices_sha256(group_indices)
        summary["multiplicity"] = (
            "unadjusted exploratory group interval; no confirmatory thresholding"
        )
        groups[error_type] = summary
    return {
        "estimand": (
            "alpha=1 symmetric expected-direction mean-logprob effect, "
            "0.5*(corrupt_to_clean - clean_to_corrupt)"
        ),
        "omnibus_permutation": omnibus,
        "groups": groups,
        "bootstrap_seed_derivation": (
            "sha256('<global_seed>:error_type_heterogeneity') followed by "
            "sha256('<heterogeneity_seed>:<error_type>'), first 8 bytes modulo 2^32"
        ),
        "multiplicity": (
            "The omnibus is one exploratory test; group intervals are unadjusted and descriptive."
        ),
        "small_group_threshold": 20,
        "error_type_source": (
            "parsed without loss from the frozen discovery pair_id prefix before "
            "'_prm_test' or '_prm_train'; source spelling is preserved"
        ),
    }


def _mean_total_sensitivity(
    matrices: Mapping[str, np.ndarray],
    *,
    cell_effects: Mapping[str, Any],
    pair_effect_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    columns = [
        _arm_column(direction, alpha) for direction in DIRECTIONS for alpha in NONZERO_ALPHAS
    ]
    mean_values = matrices["effect_mean_logprob"][:, columns]
    total_values = matrices["effect_total_logprob"][:, columns]
    sign_agreement = np.sign(mean_values) == np.sign(total_values)
    standardized_mean = _column_standardize(mean_values)
    standardized_total = _column_standardize(total_values)
    mean_cells = _require_mapping(cell_effects, "mean_logprob")
    total_cells = _require_mapping(cell_effects, "total_logprob")
    token_counts_by_pair: dict[str, tuple[int, int]] = {}
    for row in pair_effect_rows:
        pair_id = str(row["pair_id"])
        counts = (
            int(row["correct_token_count"]),
            int(row["incorrect_token_count"]),
        )
        previous = token_counts_by_pair.setdefault(pair_id, counts)
        if previous != counts:
            raise ValueError(f"{pair_id} has inconsistent derived token counts")
    cell_results: dict[str, Any] = {}
    ate_agreement = 0
    conclusion_agreement = 0
    for direction in DIRECTIONS:
        for alpha in NONZERO_ALPHAS:
            mean_entry = _require_mapping(
                _require_mapping(mean_cells, direction),
                _alpha_key(alpha),
            )
            total_entry = _require_mapping(
                _require_mapping(total_cells, direction),
                _alpha_key(alpha),
            )
            mean_sign = int(np.sign(float(mean_entry["mean"])))
            total_sign = int(np.sign(float(total_entry["mean"])))
            mean_relation = _ci_relation(mean_entry["pointwise_ci"])
            total_relation = _ci_relation(total_entry["pointwise_ci"])
            ate_agreement += int(mean_sign == total_sign)
            conclusion_agreement += int(mean_relation == total_relation)
            cell_results[f"{direction}|alpha={alpha:g}"] = {
                "mean_logprob_ate": mean_entry["mean"],
                "total_logprob_ate": total_entry["mean"],
                "ate_direction_agrees": mean_sign == total_sign,
                "mean_logprob_ci_relation_to_zero": mean_relation,
                "total_logprob_ci_relation_to_zero": total_relation,
                "ci_conclusion_agrees": mean_relation == total_relation,
                "pair_sign_agreement": float(
                    np.mean(
                        np.sign(matrices["effect_mean_logprob"][:, _arm_column(direction, alpha)])
                        == np.sign(
                            matrices["effect_total_logprob"][:, _arm_column(direction, alpha)]
                        )
                    )
                ),
                "pair_effect_spearman": _spearman(
                    matrices["effect_mean_logprob"][:, _arm_column(direction, alpha)],
                    matrices["effect_total_logprob"][:, _arm_column(direction, alpha)],
                ),
            }
    return {
        "primary_metric": "mean_logprob",
        "secondary_metric": "total_logprob",
        "raw_units_are_not_comparable": True,
        "token_count_contract": {
            "within_pair_candidate_invariant_across_treatments": True,
            "num_pairs": len(token_counts_by_pair),
            "pairs_with_different_correct_incorrect_lengths": sum(
                correct != incorrect for correct, incorrect in token_counts_by_pair.values()
            ),
        },
        "ate_direction_agreement_count": ate_agreement,
        "ci_conclusion_agreement_count": conclusion_agreement,
        "pair_cell_sign_agreement": float(np.mean(sign_agreement)),
        "standardized_pair_effect_spearman": _spearman(
            standardized_mean.ravel(),
            standardized_total.ravel(),
        ),
        "inference_status": (
            "descriptive repeated pair-cell sensitivity summaries; no multiplicity "
            "adjustment or independent-row interpretation"
        ),
        "cells": cell_results,
    }


def _matrix_mean_summaries(
    matrix: np.ndarray,
    indices: np.ndarray,
    *,
    ci_alpha: float,
    simultaneous: bool,
) -> list[dict[str, Any]]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("matrix must be a non-empty [pairs, estimands] array")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    if indices.ndim != 2 or indices.shape[1] != values.shape[0]:
        raise ValueError("bootstrap indices do not match the pair dimension")
    point = np.mean(values, axis=0)
    bootstrap_means = np.empty((indices.shape[0], values.shape[1]), dtype=float)
    for start in range(0, indices.shape[0], _RESAMPLE_CHUNK_SIZE):
        stop = min(start + _RESAMPLE_CHUNK_SIZE, indices.shape[0])
        bootstrap_means[start:stop] = np.mean(
            values[indices[start:stop]],
            axis=1,
        )
    low = np.quantile(bootstrap_means, ci_alpha / 2.0, axis=0)
    high = np.quantile(bootstrap_means, 1.0 - ci_alpha / 2.0, axis=0)
    simultaneous_intervals: list[tuple[float, float] | None] = [
        None for _ in range(values.shape[1])
    ]
    if simultaneous:
        bootstrap_se = np.std(bootstrap_means, axis=0, ddof=1)
        if np.any(bootstrap_se <= 0.0):
            raise ValueError("simultaneous bootstrap requires positive standard errors")
        standardized = np.abs((bootstrap_means - point[None, :]) / bootstrap_se[None, :])
        critical = float(np.quantile(np.max(standardized, axis=1), 1.0 - ci_alpha))
        simultaneous_intervals = [
            (
                float(point[index] - critical * bootstrap_se[index]),
                float(point[index] + critical * bootstrap_se[index]),
            )
            for index in range(values.shape[1])
        ]
    summaries: list[dict[str, Any]] = []
    for column in range(values.shape[1]):
        sample_std = float(np.std(values[:, column], ddof=1)) if values.shape[0] > 1 else 0.0
        simultaneous_interval = simultaneous_intervals[column]
        summaries.append(
            {
                "n": int(values.shape[0]),
                "mean": float(point[column]),
                "sample_std": sample_std,
                "standardized_effect": (
                    float(point[column] / sample_std) if sample_std > 0.0 else 0.0
                ),
                "pointwise_ci": [float(low[column]), float(high[column])],
                "simultaneous_ci": (
                    None if simultaneous_interval is None else list(simultaneous_interval)
                ),
            }
        )
    return summaries


def _bootstrap_within_alpha_zscore_pooled_correlation(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    rank_x: bool,
    chunk_size: int = _RESAMPLE_CHUNK_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-estimate the plotted within-alpha z-score correlation per draw."""

    output = np.empty(indices.shape[0], dtype=float)
    output.fill(np.nan)
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        chunk = indices[start:stop]
        x_sample = x[chunk, :]
        y_sample = y[chunk, :]
        if rank_x:
            x_sample = np.stack(
                [_row_midranks(x_sample[:, :, column]) for column in range(x_sample.shape[2])],
                axis=2,
            )
        x_centered = x_sample - np.mean(x_sample, axis=1, keepdims=True)
        y_centered = y_sample - np.mean(y_sample, axis=1, keepdims=True)
        x_sum_squares = np.sum(x_centered**2, axis=1)
        y_sum_squares = np.sum(y_centered**2, axis=1)
        valid = np.all(
            (x_sum_squares > 0.0) & (y_sum_squares > 0.0),
            axis=1,
        )
        if not np.any(valid):
            continue
        x_standardized = x_centered[valid] / np.sqrt(
            x_sum_squares[valid, None, :] / (x.shape[0] - 1)
        )
        y_standardized = y_centered[valid] / np.sqrt(
            y_sum_squares[valid, None, :] / (y.shape[0] - 1)
        )
        numerator = np.sum(x_standardized * y_standardized, axis=(1, 2))
        denominator = np.sqrt(
            np.sum(x_standardized**2, axis=(1, 2)) * np.sum(y_standardized**2, axis=(1, 2))
        )
        chunk_output = output[start:stop]
        chunk_output[valid] = numerator / denominator

    valid_output = output[np.isfinite(output)]
    minimum_valid = max(
        2,
        math.ceil(indices.shape[0] * _ASSOCIATION_MIN_VALID_BOOTSTRAP_FRACTION),
    )
    if valid_output.size < minimum_valid:
        raise ValueError(
            "association bootstrap has too many zero-variance replicates: "
            f"{valid_output.size}/{indices.shape[0]} valid; require {minimum_valid}"
        )
    excluded = int(indices.shape[0] - valid_output.size)
    return valid_output, {
        "requested_resamples": int(indices.shape[0]),
        "valid_resamples": int(valid_output.size),
        "excluded_zero_variance_resamples": excluded,
        "minimum_valid_fraction": _ASSOCIATION_MIN_VALID_BOOTSTRAP_FRACTION,
        "policy": (
            "Observed data must vary within every alpha. A bootstrap replicate "
            "with zero x or y variance in any alpha is excluded; analysis fails "
            "if fewer than 95% of requested replicates remain."
        ),
    }


def _within_alpha_zscore_pooled_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rank_x: bool,
) -> float:
    transformed_x = _column_midranks(x) if rank_x else np.asarray(x, dtype=float)
    x_standardized = _column_standardize(transformed_x)
    y_standardized = _column_standardize(y)
    denominator = float(np.sqrt(np.sum(x_standardized**2) * np.sum(y_standardized**2)))
    if denominator <= 0.0:
        raise ValueError("association inputs must vary within alpha cells")
    return float(np.sum(x_standardized * y_standardized) / denominator)


def _row_midranks(values: np.ndarray) -> np.ndarray:
    """Compute stable average ranks independently for each matrix row."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("row-wise ranks require a non-empty 2D matrix")
    order = np.argsort(array, axis=1, kind="stable")
    sorted_values = np.take_along_axis(array, order, axis=1)
    positions = np.arange(array.shape[1], dtype=np.int64)[None, :]

    starts = np.ones_like(sorted_values, dtype=bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    start_positions = np.maximum.accumulate(
        np.where(starts, positions, 0),
        axis=1,
    )

    ends = np.ones_like(sorted_values, dtype=bool)
    ends[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
    end_positions = np.minimum.accumulate(
        np.where(ends, positions, array.shape[1] - 1)[:, ::-1],
        axis=1,
    )[:, ::-1]
    sorted_ranks = 0.5 * (start_positions + end_positions) + 1.0
    ranks = np.empty_like(sorted_ranks, dtype=float)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def _column_midranks(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = _midranks(values[:, column])
    return output


def _column_standardize(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, ddof=1, keepdims=True)
    if np.any(std <= 0.0):
        raise ValueError("cannot standardize a zero-variance cell")
    return np.asarray((values - means) / std, dtype=float)


def _midranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x_ranks = _midranks(np.asarray(x, dtype=float))
    y_ranks = _midranks(np.asarray(y, dtype=float))
    x_centered = x_ranks - float(np.mean(x_ranks))
    y_centered = y_ranks - float(np.mean(y_ranks))
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denominator == 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def _permutation_oneway(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    observed = _oneway_f(values, labels)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_resamples):
        permuted = rng.permutation(labels)
        exceedances += int(_oneway_f(values, permuted) >= observed)
    return {
        "statistic": observed,
        "p_value": float((exceedances + 1) / (n_resamples + 1)),
        "seed": seed,
        "n_resamples": n_resamples,
        "method": "label-permutation one-way F statistic",
        "role": "exploratory omnibus",
    }


def _oneway_f(values: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(str(label) for label in labels))
    if len(unique) < 2:
        raise ValueError("heterogeneity test requires at least two groups")
    overall = float(np.mean(values))
    between = 0.0
    within = 0.0
    for label in unique:
        group = values[labels == label]
        group_mean = float(np.mean(group))
        between += float(group.size) * (group_mean - overall) ** 2
        within += float(np.sum((group - group_mean) ** 2))
    numerator = between / (len(unique) - 1)
    denominator = within / (values.size - len(unique))
    return float(numerator / denominator) if denominator > 0.0 else math.inf


def _validate_pair_token_counts(
    pair_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    observed = {_token_counts(row) for row in rows}
    if len(observed) != 1:
        raise ValueError(f"{pair_id} candidate token counts vary across treatment rows")
    correct, incorrect = observed.pop()
    return {"CORRECT": correct, "INCORRECT": incorrect}


def _token_counts(row: Mapping[str, Any]) -> tuple[int, int]:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("prediction rows must contain candidate score mappings")
    output: list[int] = []
    for candidate in ("CORRECT", "INCORRECT"):
        payload = scores.get(candidate)
        if not isinstance(payload, Mapping):
            raise ValueError(f"scores.{candidate} must be a mapping")
        value = payload.get("token_count")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"scores.{candidate}.token_count must be an integer")
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(f"scores.{candidate}.token_count must be positive")
        output.append(normalized)
    return output[0], output[1]


def _validate_identity_control(
    *,
    pair_id: str,
    treatment: str,
    receiver: str,
    baseline: Mapping[str, Any],
    control: Mapping[str, Any],
) -> None:
    if treatment == "unpatched_rerun":
        if control.get("site") is not None:
            raise ValueError(f"{pair_id} unpatched_rerun must not declare a site")
        if control.get("donor_variant") is not None:
            raise ValueError(f"{pair_id} unpatched_rerun must not declare a donor")
    elif treatment == "self_patch":
        if control.get("site") != EXPECTED_SITE:
            raise ValueError(f"{pair_id} self_patch has unexpected site")
        if str(control.get("donor_variant")) != receiver:
            raise ValueError(f"{pair_id} self_patch donor must equal receiver")
        metadata = control.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("patch_mode") != (
            "simultaneous_bundle"
        ):
            raise ValueError(f"{pair_id} self_patch does not use simultaneous_bundle")
        topk, router_kl, overlap = _manipulation_values(control)
        if abs(topk) > 1e-12 or abs(router_kl) > 1e-12 or abs(overlap - 1.0) > 1e-12:
            raise ValueError(f"{pair_id} self_patch manipulation is not identity")
    else:
        raise ValueError(f"unsupported identity control {treatment}")

    for field in ("boundary_step", "token_position"):
        if control.get(field) != baseline.get(field):
            raise ValueError(f"{pair_id} {treatment} {field} does not match unpatched baseline")
    if control.get("scores") != baseline.get("scores"):
        raise ValueError(f"{pair_id} {treatment} candidate scores do not exactly match baseline")
    for metric in METRICS:
        if _margin(control, metric) != _margin(baseline, metric):
            raise ValueError(
                f"{pair_id} {treatment} {metric} margin does not exactly match baseline"
            )


def _validate_receiver_alignment(
    *,
    pair_id: str,
    treatment: str,
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
) -> None:
    for field in ("boundary_step", "token_position"):
        baseline_value = baseline.get(field)
        intervention_value = intervention.get(field)
        if isinstance(baseline_value, bool) or not isinstance(baseline_value, int):
            raise ValueError(f"{pair_id} unpatched receiver {field} must be an integer")
        if (
            isinstance(intervention_value, bool)
            or not isinstance(intervention_value, int)
            or intervention_value != baseline_value
        ):
            raise ValueError(f"{pair_id} {treatment} {field} does not match receiver baseline")


def _candidate_score_max_abs_delta(
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
) -> float:
    baseline_scores = baseline.get("scores")
    intervention_scores = intervention.get("scores")
    if not isinstance(baseline_scores, Mapping) or not isinstance(
        intervention_scores,
        Mapping,
    ):
        raise ValueError("prediction rows must contain candidate score mappings")
    deltas: list[float] = []
    for candidate in ("CORRECT", "INCORRECT"):
        baseline_candidate = baseline_scores.get(candidate)
        intervention_candidate = intervention_scores.get(candidate)
        if not isinstance(baseline_candidate, Mapping) or not isinstance(
            intervention_candidate,
            Mapping,
        ):
            raise ValueError(f"scores.{candidate} must be a mapping")
        for metric in METRICS:
            deltas.append(
                abs(float(intervention_candidate[metric]) - float(baseline_candidate[metric]))
            )
    return max(deltas, default=0.0)


def _margin(row: Mapping[str, Any], metric: str) -> float:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("prediction rows must contain candidate score mappings")
    values: dict[str, float] = {}
    for candidate in ("CORRECT", "INCORRECT"):
        payload = scores.get(candidate)
        if not isinstance(payload, Mapping):
            raise ValueError(f"scores.{candidate} must be a mapping")
        if metric == "mean_logprob":
            value = payload.get("mean_logprob")
        elif metric == "total_logprob":
            value = payload.get("total_logprob")
        else:
            raise ValueError(f"unknown metric {metric}")
        if value is None:
            raise ValueError(f"scores.{candidate}.{metric} is missing")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"scores.{candidate}.{metric} must be finite")
        values[candidate] = numeric
    margin = values["INCORRECT"] - values["CORRECT"]
    reported_key = f"margin_incorrect_minus_correct_{metric.split('_')[0]}"
    if metric == "mean_logprob":
        reported_key = "margin_incorrect_minus_correct_mean"
    else:
        reported_key = "margin_incorrect_minus_correct_total"
    reported = row.get(reported_key)
    if reported is not None and not math.isclose(
        margin,
        float(reported),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{reported_key} does not match candidate scores")
    return margin


def _manipulation_values(row: Mapping[str, Any]) -> tuple[float, float, float]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("intervention metadata must be a mapping")
    manipulation = metadata.get("manipulation")
    if not isinstance(manipulation, Mapping):
        raise ValueError("intervention metadata has no manipulation diagnostics")
    if manipulation.get("all_registered_hooks_fired") is not True:
        raise ValueError("not all registered router hooks fired")
    if manipulation.get("registered_site_count") != 16:
        raise ValueError("manipulation diagnostics do not report 16 router sites")
    values = (
        float(manipulation["topk_changed_rate"]),
        float(manipulation["router_kl_receiver_to_patched"]),
        float(manipulation["topk_overlap"]),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("manipulation diagnostics must be finite")
    topk_changed_rate, router_kl, topk_overlap = values
    if not 0.0 <= topk_changed_rate <= 1.0:
        raise ValueError("topk_changed_rate must be between 0 and 1")
    if router_kl < 0.0:
        raise ValueError("router_kl_receiver_to_patched must be non-negative")
    if not 0.0 <= topk_overlap <= 1.0:
        raise ValueError("topk_overlap must be between 0 and 1")
    return values


def _error_type(pair_id: str) -> str:
    basename = pair_id.rsplit("/", 1)[-1]
    match = _ERROR_TYPE_PATTERN.match(basename)
    if match is None:
        raise ValueError(f"cannot parse error type from discovery pair_id {pair_id!r}")
    return str(match.group("error_type"))


def _arm_column(direction: str, alpha: float) -> int:
    return DIRECTIONS.index(direction) * len(ALPHAS) + ALPHAS.index(alpha)


def _alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def _quantile_ci(values: np.ndarray, alpha: float) -> list[float]:
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


def _ci_relation(interval: Any) -> str:
    low, high = (float(value) for value in interval)
    if low > 0.0:
        return "positive"
    if high < 0.0:
        return "negative"
    return "includes_zero"


def _interval_excludes_zero(interval: Any) -> bool:
    low, high = (float(value) for value in interval)
    return low > 0.0 or high < 0.0


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**32)


def _verify_source_sha256sums(root: Path) -> dict[str, Any]:
    """Verify every analysis input against the finalized run checksum sidecar.

    Only the four discovery files consumed by this analysis are opened. Other
    paths listed in ``SHA256SUMS`` are parsed but never resolved or read.
    """

    sums_path = root / "SHA256SUMS"
    if sums_path.is_symlink():
        raise ValueError("SHA256SUMS must not be a symlink")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        expected, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative_text in {"", "."}
            or "\\" in relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"unsafe SHA256SUMS path on line {line_number}: {relative_text!r}")
        normalized = relative.as_posix()
        if normalized in entries:
            raise ValueError(f"duplicate SHA256SUMS path: {normalized}")
        entries[normalized] = expected

    verified: dict[str, str] = {}
    for relative_text in _SOURCE_HASHED_INPUTS:
        expected = entries.get(relative_text)
        if expected is None:
            raise ValueError(f"SHA256SUMS does not cover analysis input {relative_text}")
        path = root.joinpath(*PurePosixPath(relative_text).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"hashed analysis input is missing or a symlink: {relative_text}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256SUMS mismatch for {relative_text}: {actual} != {expected}")
        verified[relative_text] = actual
    return {
        "status": "passed",
        "sidecar": "SHA256SUMS",
        "sidecar_sha256": _sha256(sums_path),
        "verified_files": verified,
        "verification_scope": (
            "All and only files consumed by this CPU analysis were verified. "
            "Other sidecar entries were not opened."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(payload)
    return rows


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError("cannot write an empty pair-effect table")
    fieldnames = list(rows[0].keys())
    if any(list(row.keys()) != fieldnames for row in rows):
        raise ValueError("pair-effect rows do not share one deterministic schema")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _format_ci(value: Any) -> str:
    low, high = (float(item) for item in value)
    return f"[{low:+.3f}, {high:+.3f}]"
