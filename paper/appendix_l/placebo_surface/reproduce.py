"""Recompute Appendix L Tables 17 and 18 from the released numerical observations."""

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "analysis/lib"))
from routeverify.analysis.bootstrap import paired_cluster_bootstrap_indices
from routeverify.analysis.donor_contrast import _source_bootstrap

MODELS = {
    "Qwen3.6": "qwen36_moe",
    "GLM-4.7-Flash": "glm4_moe_lite",
    "Lightning": "nemotron35_lightning",
    "Nano": "nemotron3_nano",
    "Gemma4": "gemma4_moe",
    "OLMoE": "olmoe",
    "GPT-OSS-20B": "gpt_oss_20b",
    "GPT-OSS-120B": "gpt_oss_120b",
}
DRAWS = 50_000
ARM = "post_moe_residual"
ROLES = ("clean", "corrupt")


def read_scores():
    # The five canonical OOF copies share scores and boundary steps.
    path = ROOT / "secondary/probe/text_baseline/oof_predictions.jsonl"
    scores = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["model_id"] == "gemma4":
                key = row["sample_id"]
                if key in scores or row["label"] != key.rsplit("/", 1)[1]:
                    raise ValueError(f"Invalid TF-IDF sample {key}")
                scores[key] = (row["score"], row["boundary_step"])
    return scores


def read_inputs(model, window_key, scores):
    fixed_path = ROOT / "analysis/abc/headline_inputs" / model / "D.json.gz"
    with gzip.open(fixed_path, "rt") as stream:
        fixed = sorted(
            (r for r in json.load(stream) if r["site"].split(".")[-1] == ARM),
            key=lambda r: r["pair_id"],
        )
    ids = {r["pair_id"]: r for r in fixed}
    if len(ids) != len(fixed) or set(scores) != {f"{p}/{v}" for p in ids for v in ROLES}:
        raise ValueError(f"{model}: incompatible pair or OOF identities")
    for record in fixed:
        for role in ROLES:
            sample = record["receiver_identities"][role]
            if scores[sample["sample_id"]][1] != sample["boundary_step"]:
                raise ValueError(f"{model}: boundary step mismatch")

    path = ROOT / "paper/verification_metrics/extracted/windows" / f"{window_key}.csv.gz"
    matched = {}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            if (row["treatment"], row["window"], row["arm"]) != ("matched", "last_token", ARM):
                continue
            key = (row["pair_id"], row["receiver_role"])
            if key in matched or float(row["alpha"]) != 1.0:
                raise ValueError(f"{model}: invalid matched window row {key}")
            matched[key] = row
    if set(matched) != {(p, r) for p in ids for r in ROLES}:
        raise ValueError(f"{model}: inconsistent window coverage")

    for rec in fixed:
        pair = rec["pair_id"]
        window_donor = matched[pair, "clean"]["donor_pair_id"]
        for receiver, opposite in zip(ROLES, reversed(ROLES)):
            w = matched[pair, receiver]
            if (w["donor_role"], w["donor_pair_id"], w["component"], w["donor_component"]) != (
                opposite, window_donor, rec["source_component"], ids[window_donor]["source_component"]
            ):
                raise ValueError(f"{model}: donor or component mismatch at {pair}")
    return fixed, matched


def pair_effects(records, matched, orientation, margin="mean", fixed_baselines=None):
    """Orient the two within-label donor contrasts before averaging receiver roles."""
    eligible, real_d, placebo, identical_donor_drift = [], [], [], []
    for rec in records:
        pair, donor = rec["pair_id"], rec["donor_pair_id"]
        window_donor = matched[pair, "clean"]["donor_pair_id"]
        cells = rec["four_combinations"]
        differences, oriented = [], []
        keep = donor != window_donor
        for receiver, opposite in zip(ROLES, reversed(ROLES)):
            window = matched[pair, receiver]
            a = cells[receiver][opposite][margin]
            b = float(window[f"margin_{margin}"])
            if fixed_baselines is not None:
                a -= fixed_baselines[pair, receiver]
                b -= float(window["baseline_margin_mean"])
            differences.append(cells[receiver]["corrupt"][margin] - cells[receiver]["clean"][margin])
            if donor == window_donor:
                identical_donor_drift.append(b - a)
            else:
                sign = np.sign(orientation[f"{window_donor}/{opposite}"] - orientation[f"{donor}/{opposite}"])
                keep = keep and sign != 0
                oriented.append(sign * (b - a))
        eligible.append(keep)
        real_d.append(np.mean(differences))
        placebo.append(np.mean(oriented) if oriented else np.nan)
    return tuple(np.asarray(a) for a in (eligible, real_d, placebo, identical_donor_drift))


def cluster_means(records, mask, values, seed=0):
    selected = [record for record, keep in zip(records, mask) if keep]
    result = _source_bootstrap(selected, values[mask], seed=seed, n_resamples=DRAWS, alpha=0.05)
    return list(zip(result["point"], result["ci_low"], result["ci_high"])), result["n_clusters"]


def intercept_interval(records, gap, real):
    # Refit intercept and slope from weighted OLS moments in every cluster draw.
    components, memberships = np.unique([r["source_component"] for r in records], return_inverse=True)
    n = len(components)
    draws = paired_cluster_bootstrap_indices(n, seed=0, n_resamples=DRAWS)

    def intercept(weights):
        totals = weights.sum(axis=1)
        mx, my = (weights @ gap) / totals, (weights @ real) / totals
        dx, dy = gap - mx[:, None], real - my[:, None]
        slope = (weights * dx * dy).sum(axis=1) / (weights * dx**2).sum(axis=1)
        return my - slope * mx

    point = intercept(np.ones((1, len(gap))))[0]
    estimates = []
    for start in range(0, DRAWS, 2000):
        batch = draws[start:start + 2000]
        offsets = n * np.arange(len(batch))[:, None]
        counts = np.bincount((batch + offsets).ravel(), minlength=n * len(batch))
        weights = counts.reshape(len(batch), n)[:, memberships].astype(float)
        estimates.extend(intercept(weights))
    return (point, *np.quantile(estimates, (0.025, 0.975)))


def reproduce():
    scores = read_scores()
    tfidf = {sample: value[0] for sample, value in scores.items()}
    results = []
    for model, window_key in MODELS.items():
        records, matched = read_inputs(model, window_key, scores)
        eligible, real, placebo, _ = pair_effects(records, matched, tfidf)
        donors = [r["donor_pair_id"] for r in records]
        gap = np.array([tfidf[f"{d}/corrupt"] - tfidf[f"{d}/clean"] for d in donors])
        p1, k1 = cluster_means(records, eligible, np.column_stack((real, placebo, real - placebo)))
        small = gap <= 0
        p2, k2 = cluster_means(records, small, real[:, None])
        full_components = len({r["source_component"] for r in records})
        sizes = (len(records), full_components, eligible.sum(), k1, small.sum(), k2)
        if sizes != (290, 235, 195, 166, 80, 78):
            raise ValueError(f"{model}: unexpected pair or cluster counts {sizes}")
        intercept = intercept_interval(records, gap, real)
        for estimand, values in zip(("real_D", "placebo", "real_minus_placebo",
                                     "nonpositive_gap_D", "zero_gap_intercept"), [*p1, *p2, intercept]):
            results.append((model, estimand, *values))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write results to a CSV file (default: stdout).")
    args = parser.parse_args()
    results = reproduce()
    with (HERE / "reference.csv").open() as stream:
        reference = list(csv.DictReader(stream))
    if [(r["model"], r["estimand"]) for r in reference] != [row[:2] for row in results]:
        raise ValueError("Reference rows and recomputed estimands differ")
    for row, ref in zip(results, reference):
        expected = [float(ref[k]) for k in ("estimate", "ci_low", "ci_high")]
        np.testing.assert_allclose(row[2:], expected, rtol=0, atol=1e-12, err_msg=str(row[:2]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    stream = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("model", "estimand", "estimate", "ci_low", "ci_high"))
        writer.writerows(results)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
