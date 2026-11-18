"""Recompute the same-label placebo sensitivity checks accompanying Table 17."""

import argparse
import csv
from pathlib import Path

import numpy as np

from reproduce import HERE, MODELS, cluster_means, pair_effects, read_inputs, read_scores

SENSITIVITIES = ("char_count", "word_count", "total_logprob", "category_exclusion", "within_run_centered")
METRICS = ("real_D", "placebo", "real_minus_placebo")
SENSITIVITY_FIELDS = ["sensitivity", "model", "seed", "n_pairs", "n_components", "passed"] + [
    metric + suffix for metric in METRICS for suffix in ("", "_lo", "_hi")
]
DRIFT_FIELDS = ["model", "margin", "pair_count", "trace_count", "mean", "sd", "max_absolute"]


def read_csv(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def reproduce():
    scores = read_scores()
    tfidf = {key: score for key, (score, _) in scores.items()}
    chars, words = {}, {}
    for row in read_csv("boundary_counts.csv"):
        key = row["sample_id"]
        if key in chars or int(row["boundary_step"]) != scores[key][1]:
            raise ValueError(f"Invalid length record for {key}")
        chars[key] = int(row["character_count"])
        words[key] = int(row["word_count"])
    fixed_baselines = {"Nano": {}, "Lightning": {}}
    for row in read_csv("fixed_run_baselines.csv"):
        key = row["pair_id"], row["receiver_role"]
        for model, baseline in fixed_baselines.items():
            if key in baseline:
                raise ValueError(f"Duplicate fixed-run baseline for {model} at {key}")
            baseline[key] = float(row[model])

    results, drift = [], []
    for model, window_key in MODELS.items():
        records, matched = read_inputs(model, window_key, scores)
        for variant in SENSITIVITIES:
            if variant == "within_run_centered" and model not in fixed_baselines:
                continue
            feature = {"char_count": chars, "word_count": words}.get(variant, tfidf)
            centered = fixed_baselines[model] if variant == "within_run_centered" else None
            eligible, real, placebo, same = pair_effects(
                records, matched, feature, "total" if variant == "total_logprob" else "mean", centered
            )
            seed = 0
            if variant == "category_exclusion":
                seed = 20260908
                # The archived source IDs spell the redundancy category "redundency".
                retain = [not any(term in r["pair_id"] for term in ("/circular_prm_", "/redundency_prm_"))
                          for r in records]
                if sum(retain) != 281:
                    raise ValueError("Unexpected size of the category-exclusion subset")
                eligible &= retain
            values = np.column_stack((real, placebo, real - placebo))
            intervals, clusters = cluster_means(records, eligible, values, seed)
            d, p, diff = intervals
            passed = d[1] > 0 and (p[1] <= 0 <= p[2] or p[2] < 0.25 * d[0]) and diff[1] > 0
            results.append([variant, model, seed, int(eligible.sum()), clusters, passed,
                            *[value for group in intervals for value in group]])
            # All uncentered mean-margin variants have identical same-donor drift.
            if model in fixed_baselines and variant in ("char_count", "within_run_centered"):
                label = "within_run_centered" if centered is not None else "raw"
                drift.append([model, label, len(same) // 2, len(same),
                              same.mean(), same.std(ddof=1), np.abs(same).max()])
    return results, drift


def check(rows, fields, ref_name, metadata_fields):
    key = lambda row: tuple(str(value) for value in row[:metadata_fields])
    refs = {tuple(row[k] for k in fields[:metadata_fields]): row for row in read_csv(ref_name)}
    if len(rows) != len(refs) or {key(row) for row in rows} != set(refs):
        raise ValueError(f"Counts, decisions or row identities differ from {ref_name}")
    for row in rows:
        expected = [float(refs[key(row)][f]) for f in fields[metadata_fields:]]
        np.testing.assert_allclose(row[metadata_fields:], expected, rtol=0, atol=1e-12,
                                   err_msg=str(key(row)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/appendix_l"))
    args = parser.parse_args()
    results, drift = reproduce()
    check(results, SENSITIVITY_FIELDS, "sensitivity_reference.csv", 6)
    check(drift, DRIFT_FIELDS, "drift_reference.csv", 4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (("placebo_sensitivities", SENSITIVITY_FIELDS, results),
                               ("identical_donor_drift", DRIFT_FIELDS, drift)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(fields)
            writer.writerows(rows)
    for name in SENSITIVITIES:
        models = [row for row in results if row[0] == name]
        print(f"{name}: {sum(row[5] for row in models)}/{len(models)} models pass "
              f"({models[0][3]} pairs, {models[0][4]} components)")


if __name__ == "__main__":
    main()
