"""Post hoc PRMBench category exclusion from released pair-level estimates.

CPU only. Does not change predictions, matches, or the primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DRAWS, SEED = 50_000, 20260908
EXCLUDED_TYPES = {"circular", "redundency"}  # Preserve the dataset spelling.
WINDOWS = ["last_token", "suffix_k", "divergence_k", "full_step"]


def read_inputs(root):
    manifest = json.loads((HERE / "input-manifest.json").read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {entry['path']}")

    types, roles = {}, defaultdict(set)
    with (root / "execution/inputs/discovery-identities.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            pair, error_type = row["pair_id"], row["error_type"]
            if pair in types and types[pair] != error_type:
                raise ValueError(f"Inconsistent error type: {pair}")
            types[pair] = error_type
            roles[pair].add(row["variant"])
    assert len(types) == 290 and all(r == {"clean", "corrupt"} for r in roles.values())
    excluded = {pair for pair, kind in types.items() if kind in EXCLUDED_TYPES}
    assert Counter(types[pair] for pair in excluded) == {"circular": 6, "redundency": 3}

    cells, components = defaultdict(dict), {}
    with (root / "analysis/abc/window_pair_values_v1.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            pair = row["pair_id"]
            assert pair in types
            if pair in components:
                assert components[pair] == row["component"]
            components[pair] = row["component"]
            key = row["model"], row["window"], row["arm"], row["control"]
            assert pair not in cells[key]
            cells[key][pair] = float(row["T"])
    assert len(components) == 290 and len(set(components.values())) == 235
    with (root / "analysis/abc/unified_windows_v1.csv").open(newline="") as stream:
        published = list(csv.DictReader(stream))
    return manifest, types, excluded, cells, components, published


def make_contrasts(cells, excluded):
    checks = []

    def add(model, estimand, window, terms, *, full=False):
        vectors = [(sign, cells[model, w, arm, control]) for sign, w, arm, control in terms]
        # Form each contrast on its own common eligible pairs before exclusion.
        ids = set.intersection(*(set(vector) for _, vector in vectors))
        if not full:
            ids -= excluded
        values = {pair: sum(sign * vector[pair] for sign, vector in vectors) for pair in sorted(ids)}
        assert values
        checks.append((dict(model=model, estimand=estimand, window=window,
                            population="full_discovery" if full else "category_exclusion"), values))

    models = sorted({key[0] for key in cells})
    assert len(models) == 9
    for model in models:
        for window in WINDOWS:
            for arm in ["router_logits", "moe_output"]:
                add(model, "residual_minus_" + arm, window, [
                    (1, window, "post_moe_residual", "paired"),
                    (-1, window, arm, "paired")])
        add(model, "position_gap", "div_minus_suffix", [
            (1, "divergence_k", "post_moe_residual", "paired"),
            (-1, "divergence_k", "all_router", "paired"),
            (-1, "suffix_k", "post_moe_residual", "paired"),
            (1, "suffix_k", "all_router", "paired")])
        add(model, "all_router_minus_normrandom", "full_step", [
            (1, "full_step", "all_router", "paired"),
            (-1, "full_step", "all_router", "norm_random")])
    add("OLMoE", "all_router_minus_normrandom", "full_step", [
        (1, "full_step", "all_router", "paired"),
        (-1, "full_step", "all_router", "norm_random")], full=True)
    assert len(checks) == 91
    return checks


def summarize(checks, components):
    groups = defaultdict(list)
    for metadata, values in checks:
        groups[tuple(sorted(values))].append((metadata, values))
    results, eligibility_sets = [], {}
    for ids, entries in groups.items():
        clusters = sorted({components[pair] for pair in ids})
        index = {cluster: i for i, cluster in enumerate(clusters)}
        matrix = np.asarray([[values[pair] for _, values in entries] for pair in ids])
        assert np.isfinite(matrix).all()
        totals = np.zeros((len(clusters), len(entries)))
        counts = np.zeros(len(clusters), dtype=np.int64)
        for i, pair in enumerate(ids):
            cluster = index[components[pair]]
            totals[cluster] += matrix[i]
            counts[cluster] += 1
        rng = np.random.Generator(np.random.PCG64(SEED))
        draws = np.empty((DRAWS, len(entries)))
        for start in range(0, DRAWS, 256):
            stop = min(start + 256, DRAWS)
            sampled = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
            draws[start:stop] = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)[:, None]
        intervals = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
        eligibility_id = f"eligibility_{len(eligibility_sets) + 1}"
        eligibility_sets[eligibility_id] = {
            "N": len(ids), "K": len(clusters),
            "pair_source_components": {pair: components[pair] for pair in ids}}
        for j, (metadata, _) in enumerate(entries):
            results.append(dict(**metadata, eligibility_set=eligibility_id,
                                N=len(ids), K=len(clusters), mean=float(matrix[:, j].mean()),
                                ci_low=float(intervals[0, j]), ci_high=float(intervals[1, j])))
    return results, eligibility_sets


def validate(results, published):
    full = next(row for row in results if row["population"] == "full_discovery")
    reference = next(row for row in published if row["model"] == "OLMoE"
                     and row["estimand"] == "paired_minus_control"
                     and row["window"] == "full_step" and row["arm"] == "all_router"
                     and row["control"] == "norm_random")
    deltas = {key: full[key] - float(reference[key]) for key in ["mean", "ci_low", "ci_high"]}
    assert full["N"] == int(reference["N"]) and full["K"] == int(reference["K"])
    assert all(abs(delta) < 1e-12 for delta in deltas.values())
    sensitivity = [row for row in results if row["population"] == "category_exclusion"]
    interfaces = [row for row in sensitivity if row["estimand"].startswith("residual_minus_")]
    positions = [row for row in sensitivity if row["estimand"] == "position_gap"]
    return {
        "full_OLMoE_matches_released_primary_result": True,
        "full_OLMoE_difference_from_released_primary_result": deltas,
        "interface_comparisons": len(interfaces),
        "interface_ci_lower_above_zero": sum(row["ci_low"] > 0 for row in interfaces),
        "smallest_interface_ci_lower": min(row["ci_low"] for row in interfaces),
        "position_gap_comparisons": len(positions),
        "position_gap_ci_upper_below_zero": sum(row["ci_high"] < 0 for row in positions),
        "largest_position_gap_ci_upper": max(row["ci_high"] for row in positions),
        "OLMoE_comparison": [full, next(row for row in sensitivity if row["model"] == "OLMoE"
                                       and row["estimand"] == "all_router_minus_normrandom")]}


def check_frozen(package):
    reference_path = HERE / "results.json"
    frozen = json.loads(reference_path.read_text())
    for field in ("input_manifest", "full_discovery_error_type_counts", "excluded_error_type_counts",
                  "excluded_pair_ids", "eligibility_sets", "bootstrap", "contrast_definitions"):
        if package[field] != frozen[field]:
            raise ValueError(f"Frozen reference mismatch: {field}")

    numeric = ("mean", "ci_low", "ci_high")
    max_delta = dict.fromkeys(numeric, 0.0)
    for actual, expected in zip(package["results"], frozen["results"], strict=True):
        attributes = {key: value for key, value in actual.items() if key not in numeric}
        if attributes != {key: value for key, value in expected.items() if key not in numeric}:
            raise ValueError(f"Contrast identity or eligible counts differ: {attributes}")
        for key in numeric:
            diff = abs(actual[key] - expected[key])
            if not np.isfinite(diff) or diff > 1e-12:
                raise ValueError(f"Frozen reference mismatch for {attributes}, {key}: {diff}")
            max_delta[key] = max(max_delta[key], diff)
    return {
        "status": "pass",
        "environment": package["environment"],
        "frozen_results_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "contrasts_compared": len(package["results"]),
        "absolute_tolerance": 1e-12,
        "maximum_absolute_difference": max_delta,
        "summary": package["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=HERE.parents[1],
                        help="Root of the RouteVerify release (inferred from the script by default)")
    args = parser.parse_args()
    manifest, types, excluded, cells, components, published = read_inputs(args.release_root)
    results, eligibility_sets = summarize(make_contrasts(cells, excluded), components)
    package = {
        "analysis": "post_hoc_prmbench_category_exclusion",
        "scope": "Archived T estimates only; fixed original eligibility, predictions, and assignments; no D analysis.",
        "interpretation": "A category exclusion sensitivity analysis, not mathematical validation of retained labels.",
        "contrast_definitions": {
            "residual_minus_router_logits": "paired T(post_moe_residual) - paired T(router_logits), within each window",
            "residual_minus_moe_output": "paired T(post_moe_residual) - paired T(moe_output), within each window",
            "position_gap": "[T(residual)-T(all_router)] at divergence_k minus the same gap at suffix_k; all paired",
            "all_router_minus_normrandom": "paired T(all_router) - norm_random T(all_router), full_step"},
        "bootstrap": {"draws": DRAWS, "seed": SEED, "rng": "numpy.random.Generator(PCG64)",
                      "resampling_unit": "eligible receiver source component",
                      "estimator": "ratio of resampled pair sums to pair counts; equal weight per pair",
                      "interval": "marginal 95% percentile; linear quantiles; no multiplicity correction",
                      "shared_draws": "identical eligible pair sets share draws; generator restarts per set"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "input_manifest": manifest,
        "full_discovery_error_type_counts": dict(sorted(Counter(types.values()).items())),
        "excluded_error_type_counts": dict(sorted(Counter(types[pair] for pair in excluded).items())),
        "excluded_pair_ids": sorted(excluded),
        "eligibility_sets": eligibility_sets,
        "summary": validate(results, published),
        "results": results}
    verification = check_frozen(package)
    output_dir = args.release_root / "build/category_exclusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (("results.json", package), ("verification.json", verification)):
        (output_dir / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
