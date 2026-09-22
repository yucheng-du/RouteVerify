#!/usr/bin/env python3
"""Post hoc secondary wide-D contrasts, from accepted atomic pair archives only.

No GPU/model load, no archive mutation, no NPZ extraction. Python + NumPy only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "analysis/lib"))
import numpy as np  # noqa: E402

from routeverify.analysis.bootstrap import (  # noqa: E402
    paired_cluster_bootstrap_indices,
    paired_cluster_indices_sha256,
)
from routeverify.analysis.window_donor_contrast import _source_bootstrap  # noqa: E402

SEED = 20260908
RESAMPLES = 50000
FIELDS = (
    "d_clean_mean",
    "d_corrupt_mean",
    "d_common_mean",
    "d_clean_total",
    "d_corrupt_total",
    "d_common_total",
)
CANDIDATES = ("CORRECT", "INCORRECT")
ROLES = ("clean", "corrupt")


def digest(b):
    return hashlib.sha256(b).hexdigest()


def canonical(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(p):
    return json.loads(p.read_text())


def write_json(p, d):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_record(p, root=None):
    raw = p.read_bytes()
    return {
        "path": str(p.relative_to(root)) if root else str(p),
        "bytes": len(raw),
        "sha256": digest(raw),
    }


def close(a, b, context):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError((context, a, b))


def project_identity(d):
    result = {k: v for k, v in d.items() if k != "candidate_inputs"}
    result["candidate_inputs"] = {}
    for c, x in d["candidate_inputs"].items():
        result["candidate_inputs"][c] = {k: v for k, v in x.items() if k != "input_ids"}
        result["candidate_inputs"][c]["actual_input_ids_projection_sha256"] = digest(
            canonical(x["input_ids"])
        )
        assert x["input_ids"][x["target_token_start"] : x["target_token_end"]] == x["target_ids"]
    return result


def project_pair(unit, source, summary):
    assert unit["technical_passed"] is True and unit["execution_version"] == "v1"
    assert unit["contract_id"] == summary["contract_id"] and unit["pair_id"] == source["pair_id"]
    assert unit["run_id"] == source["run_id"]
    rows = unit["rows"]
    donor = [r for r in rows if r["treatment"] == "donor_role"]
    assert len(rows) == 10 and len(donor) == 4
    assert {(r["receiver_role"], r["donor_role"]) for r in donor} == {
        (a, b) for a in ROLES for b in ROLES
    }
    assert {r["run_id"] for r in rows} == {unit["run_id"]}
    assert len({r["donor_pair_id"] for r in donor}) == 1
    identity = {}
    cells = {}
    for r in rows:
        assert (
            r["pair_id"] == unit["pair_id"]
            and r["source_component"] == unit["eligibility"]["source_component"]
        )
        for c in CANDIDATES:
            s = r["scores"][c]
            x = r["metadata"]["receiver_identity"]["candidate_inputs"][c]
            assert s["candidate"] == c and s["token_count"] == len(x["target_ids"])
            close(
                s["mean_logprob"],
                s["total_logprob"] / s["token_count"],
                ("score normalization", unit["pair_id"], c),
            )
        for metric in ("mean", "total"):
            close(
                r[f"margin_{metric}"],
                r["scores"]["INCORRECT"][metric + "_logprob"]
                - r["scores"]["CORRECT"][metric + "_logprob"],
                ("margin", unit["pair_id"]),
            )
        if r["treatment"] != "donor_role":
            continue
        for which, role in [("receiver", r["receiver_role"]), ("donor", r["donor_role"])]:
            k = which + "_" + role
            p = project_identity(r["metadata"][which + "_identity"])
            if k in identity:
                assert p == identity[k]
            identity[k] = p
        cells[r["receiver_role"] + "__" + r["donor_role"]] = {
            "margin_mean": r["margin_mean"],
            "margin_total": r["margin_total"],
            "scores": r["scores"],
        }
    rec = {
        "pair_id": unit["pair_id"],
        "source_component": unit["eligibility"]["source_component"],
        "donor_pair_id": donor[0]["donor_pair_id"],
        "donor_source_component": unit["eligibility"]["donor_component"],
        "run_id": unit["run_id"],
        "identity": identity,
        "cells": cells,
        "source": source,
    }
    assert (
        rec["pair_id"] != rec["donor_pair_id"]
        and rec["source_component"] != rec["donor_source_component"]
    )
    for metric in ("mean", "total"):
        for r in ROLES:
            rec[f"d_{r}_{metric}"] = (
                cells[r + "__corrupt"]["margin_" + metric]
                - cells[r + "__clean"]["margin_" + metric]
            )
            for c in CANDIDATES:
                rec[f"delta_{c}_{r}_{metric}"] = (
                    cells[r + "__corrupt"]["scores"][c][metric + "_logprob"]
                    - cells[r + "__clean"]["scores"][c][metric + "_logprob"]
                )
        rec[f"d_common_{metric}"] = (rec[f"d_clean_{metric}"] + rec[f"d_corrupt_{metric}"]) / 2
        for c in CANDIDATES:
            rec[f"delta_{c}_common_{metric}"] = (
                rec[f"delta_{c}_clean_{metric}"] + rec[f"delta_{c}_corrupt_{metric}"]
            ) / 2
    return rec


def load_model(model):
    payload = json.loads(gzip.decompress((HERE / (model + "_pair_projection.json.gz")).read_bytes()))
    pairs = payload["pairs"]
    assert len(pairs) == 290
    # Re-derive all six D values from the four archived cells; check score normalization.
    for pair in pairs:
        for cell in pair["cells"].values():
            for candidate in CANDIDATES:
                score = cell["scores"][candidate]
                close(score["mean_logprob"], score["total_logprob"] / score["token_count"], "candidate normalization")
            for metric in ("mean", "total"):
                close(cell["margin_"+metric], cell["scores"]["INCORRECT"][metric+"_logprob"]-cell["scores"]["CORRECT"][metric+"_logprob"], "margin sign")
        for metric in ("mean", "total"):
            values=[]
            for role in ROLES:
                val=pair["cells"][role+"__corrupt"]["margin_"+metric]-pair["cells"][role+"__clean"]["margin_"+metric]
                close(val,pair[f"d_{role}_{metric}"],"fixed receiver D")
                values.append(val)
            close(sum(values)/2,pair[f"d_common_{metric}"],"D common")
    return pairs, {"input": "released four-cell score and identity projection", "original_archive_member_hash_recheck": False}


def identity_key(rec, scope="source"):
    if scope == "source":
        return {
            "pair_id": rec["pair_id"],
            "source_component": rec["source_component"],
            "donor_pair_id": rec["donor_pair_id"],
            "donor_source_component": rec["donor_source_component"],
            "identities": {
                k: {f: v[f] for f in ("sample_id", "content_hash", "source_component")}
                for k, v in rec["identity"].items()
            },
        }
    return {"pair_id": rec["pair_id"], "identities": rec["identity"]}


def component_replicates(values, groups, indices):
    # Same ratio-of-sums implementation as archived _source_bootstrap, generalized
    # to direct contrast columns. Cluster order and PCG64 draws are unchanged.
    pair_groups = list(groups)
    groups = sorted(set(pair_groups))
    members = [[i for i, r in enumerate(pair_groups) if r == g] for g in groups]
    sums = np.asarray([values[m].sum(axis=0) for m in members])
    sizes = np.asarray([len(m) for m in members])
    out = np.empty((len(indices), values.shape[1]))
    chunk = max(1, 2_000_000 // (len(groups) * values.shape[1]))
    for start in range(0, len(indices), chunk):
        draws = indices[start : start + chunk]
        out[start : start + chunk] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
    return out


def interval(values, draws):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_standard_error": float(np.std(draws, ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "build/wide_d")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    wide = read_json(HERE / "wide_d_nine_model_v1.json")
    models = wide["completed_models"]
    provenance = {"input_kind": "released compact projections of archived scores", "input_files": [], "scope": "Numerical and stored identity checks only; no new GPU forwards or claim of raw archive rehash."}
    data = {}
    validation = {}
    tokens = {}
    rebuild = {}
    for model in models:
        summary_path = HERE / f"{model}_full_v1.json"
        summary = read_json(summary_path)
        pairs, extract_validation = load_model(model)
        data[model] = pairs
        assert len({r["source_component"] for r in pairs}) == 235
        # Raw four-cell rederivation must reproduce every primary/sensitivity point
        # before any new interval is calculated.
        for metric, key in [
            ("mean", "primary_mean_logprob"),
            ("total", "sensitivity_total_logprob"),
        ]:
            for role, label in [
                ("clean", "D_clean"),
                ("corrupt", "D_error"),
                ("common", "D_common"),
            ]:
                close(
                    np.mean([r[f"d_{role}_{metric}"] for r in pairs]),
                    summary[key][label]["mean"],
                    (model, key, label, "point"),
                )
        boot = _source_bootstrap(pairs, seed=SEED, n_resamples=RESAMPLES, alpha=0.05)
        assert (
            boot["indices_sha256"] == summary["bootstrap"]["indices_sha256"]
            and boot["cluster_pair_counts"] == summary["bootstrap"]["cluster_pair_counts"]
        )
        discrepancies = []
        for j, field in enumerate(FIELDS):
            label = {"clean": "D_clean", "corrupt": "D_error", "common": "D_common"}[
                field.split("_")[1]
            ]
            metric = (
                "primary_mean_logprob" if field.endswith("mean") else "sensitivity_total_logprob"
            )
            saved = summary[metric][label]
            for a, b in [
                ("point", "mean"),
                ("ci_low", "ci_low"),
                ("ci_high", "ci_high"),
                ("bootstrap_standard_error", "bootstrap_standard_error"),
            ]:
                close(boot[a][j], saved[b], (model, field, a))
                discrepancies.append(abs(boot[a][j] - saved[b]))
        rebuild[model] = {
            k: summary[k] for k in ("primary_mean_logprob", "sensitivity_total_logprob")
        }
        cnt = Counter()
        for p in pairs:
            for r in p["cells"].values():
                cnt[tuple(r["scores"][c]["token_count"] for c in CANDIDATES)] += 1
        tokens[model] = {
            "candidate_order": CANDIDATES,
            "token_counts_per_four_cell_row": [
                {"counts": list(k), "rows": v} for k, v in sorted(cnt.items())
            ],
            "candidate_target_ids": summary["technical"]["candidate_target_ids"],
        }
        validation[model] = {
            **extract_validation,
            "N": 290,
            "K": 235,
            "unique_fixed_donors": len({r["donor_pair_id"] for r in pairs}),
            "all_six_endpoints_reproduced": True,
            "max_abs_difference_point_CI_SE": max(discrepancies),
        }
        provenance["input_files"].append(file_record(summary_path, HERE))
        print(
            json.dumps(
                {"model": model, "rebuild": "PASS", "max_abs_difference": max(discrepancies)}
            ),
            flush=True,
        )
    reference = data[models[0]]
    source_projections = {m: [identity_key(r) for r in data[m]] for m in models}
    assert all(source_projections[m] == source_projections[models[0]] for m in models)
    gpt_ids_a = [identity_key(r, "full") for r in data["gpt_oss_20b"]]
    gpt_ids_b = [identity_key(r, "full") for r in data["gpt_oss_120b"]]
    cross_identity = {
        "all_nine_models_same_receiver_source_and_donor_assignment": True,
        "n_common_receiver_pairs": 290,
        "n_common_source_components": 235,
        "n_fixed_donor_pairs": 204,
        "source_content_assignment_projection_sha256": digest(
            canonical(source_projections[models[0]])
        ),
        "gpt20_gpt120_all_projected_actual_input_identities_equal": gpt_ids_a == gpt_ids_b,
        "gpt_identity_coverage": {
            "receiver_pair_count": 290,
            "receiver_role_identities": 580,
            "receiver_and_donor_identity_occurrences": 1160,
            "candidate_conditioned_full_input_occurrences": 2320,
            "unique_receiver_sample_ids": len(
                {
                    x["sample_id"]
                    for r in data["gpt_oss_20b"]
                    for k, x in r["identity"].items()
                    if k.startswith("receiver_")
                }
            ),
            "unique_donor_sample_ids": len(
                {
                    x["sample_id"]
                    for r in data["gpt_oss_20b"]
                    for k, x in r["identity"].items()
                    if k.startswith("donor_")
                }
            ),
            "fields_compared": sorted(data["gpt_oss_20b"][0]["identity"]["receiver_clean"]),
            "candidate_input_fields_compared": sorted(
                data["gpt_oss_20b"][0]["identity"]["receiver_clean"]["candidate_inputs"]["CORRECT"]
            ),
            "full_input_method": "For every projected identity and both candidates, compare the full-array SHA256 calculated during the original archive extraction, plus saved input hashes, target ids/positions/shapes and complete step_span. This public rerun checks stored projections only: complete actual input_ids arrays are not included or rehashed.",
            "Harmony_date_boundary": "No date decoding or prompt-string extraction is performed in this public rerun. It compares full-render and full-array hashes saved in the original extraction; complete prompt/token contents, including any date content, are not available for a fresh hash check here.",
        },
        "gpt20_full_identity_projection_sha256": digest(canonical(gpt_ids_a)),
        "gpt120_full_identity_projection_sha256": digest(canonical(gpt_ids_b)),
        "comparison_boundary": "Two fixed checkpoints; not a parameter-count-only causal effect.",
    }
    pair_groups = [p["source_component"] for p in reference]
    idx = paired_cluster_bootstrap_indices(235, seed=SEED, n_resamples=RESAMPLES)
    names = []
    vectors = []
    for m in models:
        names.append(m + "_clean_minus_error")
        vectors.append([r["d_clean_mean"] - r["d_corrupt_mean"] for r in data[m]])
    names.append("gpt20_minus_gpt120_common")
    vectors.append(
        [
            a["d_common_mean"] - b["d_common_mean"]
            for a, b in zip(data["gpt_oss_20b"], data["gpt_oss_120b"], strict=True)
        ]
    )
    # Retain primary/sensitivity Gemma draws and candidate contributions to verify
    # the exact reweighting mechanism on each common component draw.
    for f in (
        "d_common_mean",
        "d_common_total",
        "delta_INCORRECT_common_mean",
        "delta_CORRECT_common_mean",
    ):
        names.append("gemma_" + f)
        vectors.append([r[f] for r in data["gemma4_moe"]])
    values = np.asarray(vectors).T
    draws = component_replicates(values, pair_groups, idx)
    intervals = {name: interval(values[:, j], draws[:, j]) for j, name in enumerate(names)}
    assert tokens["gemma4_moe"]["token_counts_per_four_cell_row"] == [
        {"counts": [2, 3], "rows": 1160}
    ]
    gi = {
        k: names.index("gemma_" + k)
        for k in (
            "d_common_mean",
            "d_common_total",
            "delta_INCORRECT_common_mean",
            "delta_CORRECT_common_mean",
        )
    }
    a = values[:, gi["delta_INCORRECT_common_mean"]]
    b = values[:, gi["delta_CORRECT_common_mean"]]
    np.testing.assert_allclose(values[:, gi["d_common_mean"]], a - b, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        values[:, gi["d_common_total"]], 3 * a - 2 * b, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        draws[:, gi["d_common_total"]],
        3 * draws[:, gi["delta_INCORRECT_common_mean"]]
        - 2 * draws[:, gi["delta_CORRECT_common_mean"]],
        rtol=1e-10,
        atol=1e-10,
    )
    gemma = {
        "candidate_token_counts": {"CORRECT": 2, "INCORRECT": 3},
        "four_cell_rows_verified": 1160,
        "definitions": {
            "A": "mean over receiver pairs and their two roles of change in INCORRECT candidate mean logprob, error donor minus clean donor",
            "B": "same change for CORRECT candidate mean logprob",
            "D_common_mean": "A - B",
            "D_common_total": "3*A - 2*B = 2*D_common_mean + A = 3*D_common_mean + B",
        },
        "A": intervals["gemma_delta_INCORRECT_common_mean"],
        "B": intervals["gemma_delta_CORRECT_common_mean"],
        "primary_D_common": intervals["gemma_d_common_mean"],
        "sensitivity_D_common": intervals["gemma_d_common_total"],
        "fixed_uniform_positive_scaling": False,
        "uniform_scaling_reason": "CORRECT and INCORRECT have unequal token counts, so changing from each-candidate normalization to total gives different relative weights. The endpoint transformation is not one uniform positive scalar; such a scalar would preserve the percentile interval zero crossing.",
        "determinant_A_B_pair_gram": float(
            np.linalg.det(np.column_stack([a, b]).T @ np.column_stack([a, b]))
        ),
        "pair_identity_reweighting_max_abs_residual": float(
            np.max(np.abs(values[:, gi["d_common_total"]] - (3 * a - 2 * b)))
        ),
        "bootstrap_reweighting_max_abs_residual": float(
            np.max(
                np.abs(
                    draws[:, gi["d_common_total"]]
                    - (
                        3 * draws[:, gi["delta_INCORRECT_common_mean"]]
                        - 2 * draws[:, gi["delta_CORRECT_common_mean"]]
                    )
                )
            )
        ),
    }
    result = {
        "schema": "wide_d_posthoc_secondary_contrasts_v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "analysis_status": "posthoc_secondary",
        "primary_endpoint_unchanged": "candidate-wise mean logprob(INCORRECT) minus mean logprob(CORRECT)",
        "bootstrap": {
            "n_pairs": 290,
            "n_clusters": 235,
            "n_resamples": RESAMPLES,
            "seed": SEED,
            "rng": "numpy.random.Generator(PCG64)",
            "indices_sha256": paired_cluster_indices_sha256(idx),
            "unit": "receiver_source_component",
            "weighting": "pair weighted ratio of resampled component sums to pair counts",
            "receiver_roles_jointly_resampled": True,
            "models_jointly_resampled": True,
            "donor_assignment_fixed": True,
            "intervals": "pointwise two-sided 95% percentile; no multiplicity or simultaneous adjustment",
        },
        "role_contrasts": {m: intervals[m + "_clean_minus_error"] for m in models},
        "gpt20_minus_gpt120_D_common": intervals["gpt20_minus_gpt120_common"],
        "cross_model_identity": cross_identity,
        "gemma_endpoint_mechanism": gemma,
        "all_model_candidate_counts": tokens,
        "reconstructed_original_endpoints": rebuild,
        "validation": validation,
        "provenance": provenance,
    }
    write_json(output / "secondary_contrasts.json", result)
    with (output / "direct_contrasts.csv").open("w") as out:
        w = csv.DictWriter(
            out,
            fieldnames=[
                "contrast",
                "mean",
                "ci_low",
                "ci_high",
                "bootstrap_standard_error",
                "analysis_status",
            ],
        )
        w.writeheader()
        for name in names[:10]:
            w.writerow(
                {"contrast": name, **intervals[name], "analysis_status": "posthoc_secondary"}
            )
    with (output / "pair_effects.csv").open("w") as out:
        cols = [
            "model",
            "pair_id",
            "source_component",
            "donor_pair_id",
            "donor_source_component",
            *FIELDS,
            "delta_INCORRECT_common_mean",
            "delta_CORRECT_common_mean",
        ]
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for m in models:
            for r in data[m]:
                w.writerow({"model": m, **{k: r[k] for k in cols if k != "model"}})
    write_json(
        output / "analysis_receipt.json",
        {
            "analysis_status": "posthoc_secondary",
            "command": [sys.executable, str(Path(__file__).resolve()), "--output", str(output)],
            "python": sys.version,
            "numpy": np.__version__,
            "script": file_record(Path(__file__).resolve()),
            "outputs": [
                file_record(output / p)
                for p in ("secondary_contrasts.json", "direct_contrasts.csv", "pair_effects.csv")
            ],
            "projection_files": [
                file_record(HERE / (m + "_pair_projection.json.gz")) for m in models
            ],
            "GPU_forwards": 0,
            "scientific_source_mutations": False,
            "archive_mutations": False,
            "full_archive_hash_recomputed": False,
        },
    )
    print(
        json.dumps(
            {
                "role_contrasts": result["role_contrasts"],
                "gpt_comparison": result["gpt20_minus_gpt120_D_common"],
                "identity": cross_identity,
                "gemma": gemma,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
