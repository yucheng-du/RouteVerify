"""Read-only nine-model window synthesis; no model execution or archive edits.

Rebuild T from executed rows, verify every archived pair estimand, and bootstrap
selected directly paired contrasts under one source-component convention.
Run: OPENBLAS_NUM_THREADS=1 python3 analysis/abc/recompute_windows.py
Portable release adaptation: compressed score projections; original statistics unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import argparse
import gzip
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "build/windows"
SOURCES = {name: HERE / "inputs" / name / "analysis.json.gz" for name in ["Qwen3.6", "Qwen3", "GLM-4.7-Flash", "Lightning", "Nano", "Gemma4", "OLMoE", "GPT-OSS-20B", "GPT-OSS-120B"]}
ANCHORS = dict(zip(SOURCES, [20, 24, 24, 27, 27, 15, 8, 12, 18], strict=False))
WINDOWS = ["last_token", "suffix_k", "divergence_k", "full_step"]
ARMS = ["router_logits", "moe_output", "post_moe_residual", "all_router"]
CONTROLS = ["paired", "matched", "norm_random", "self"]
DRAWS, SEED = 50000, 20260908


def difference(left, right):
    return {p: left[p] - right[p] for p in sorted(left.keys() & right.keys())}


def summarize_group(vectors, components):
    """Pairs retain equal weight. Sample K source clusters, ratio sums/counts.

    Multiple estimands with identical eligibility use shared draws. External
    matched donors and matching assignments are conditioned on, not resampled.
    """
    ids = sorted(vectors[0])
    clusters = sorted({components[p] for p in ids})
    lookup = {c: i for i, c in enumerate(clusters)}
    x = np.asarray([[v[p] for v in vectors] for p in ids], dtype=float)
    assert np.isfinite(x).all()
    sums = np.zeros((len(clusters), len(vectors)))
    counts = np.zeros(len(clusters))
    for i, p in enumerate(ids):
        j = lookup[components[p]]
        sums[j] += x[i]
        counts[j] += 1
    rng = np.random.Generator(np.random.PCG64(SEED))
    means = np.empty((DRAWS, len(vectors)))
    for start in range(0, DRAWS, 256):
        stop = min(start + 256, DRAWS)
        idx = rng.integers(0, len(clusters), (stop - start, len(clusters)))
        means[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
    ci = np.quantile(means, [0.025, 0.975], axis=0)
    result = []
    for j in range(len(vectors)):
        vals = x[:, j]
        trim = np.sort(vals)[int(len(vals) * 0.1) : len(vals) - int(len(vals) * 0.1)]
        keep = np.argsort(np.abs(vals))[:-15]
        result.append(
            dict(
                mean=float(vals.mean()),
                ci_low=float(ci[0, j]),
                ci_high=float(ci[1, j]),
                N=len(ids),
                K=len(clusters),
                median=float(np.median(vals)),
                trimmed_mean_10pct=float(trim.mean()),
                mean_without_15_largest_absolute=float(vals[keep].mean()),
                positive_pair_fraction=float((vals > 0).mean()),
                eligible_pair_ids=ids,
            )
        )
    return result


def load_model(path):
    analysis = json.loads(gzip.decompress(path.read_bytes()))
    pairs = analysis["pair_estimands"]
    cells = defaultdict(dict)
    for r in pairs:
        key = r["window"], r["arm"], r["control"]
        assert r["pair_id"] not in cells[key]
        cells[key][r["pair_id"]] = r["T"]
    components, raw = {}, defaultdict(dict)
    raw_path = path.with_name("rows.jsonl.gz")
    for line in gzip.open(raw_path, "rt"):
        r = json.loads(line)
        p = r["pair_id"]
        if p in components:
            assert components[p] == r["component"]
        components[p] = r["component"]
        if r.get("window") and r["treatment"] in CONTROLS:
            key = r["window"], r["arm"], r["treatment"], p
            role = r["receiver_role"]
            assert role not in raw[key]
            raw[key][role] = (1 if role == "clean" else -1) * (
                r["margin_mean"] - r["baseline_margin_mean"]
            )
    checked = 0
    for key, roles in raw.items():
        if set(roles) != {"clean", "corrupt"}:
            continue
        w, arm, control, p = key
        assert abs(np.mean(list(roles.values())) - cells[w, arm, control][p]) < 1e-12
        checked += 1
    assert checked == len(pairs), (checked, len(pairs))
    for r in analysis["estimates"]:
        if r["estimand"] == "T":
            v = cells[r["window"], r["arm"], r["control"]]
            assert len(v) == r["n"] and abs(np.mean(list(v.values())) - r["mean"]) < 1e-12
    return cells, components, checked


def main():
    global OUT
    parser = argparse.ArgumentParser(description="Recompute all 1,170 archived window panels, 50,000 draws; CPU only.")
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    OUT.mkdir(parents=True, exist_ok=True)
    estimates, evidence, pair_output = [], {}, []
    for model, path in SOURCES.items():
        cells, components, checked = load_model(path)
        panels = []

        # Called synchronously within this model iteration; the helper never escapes it.
        def add(estimand, window, arm, control, vals, left="", right=""):
            panels.append(  # noqa: B023
                (
                    dict(
                        model=model,  # noqa: B023
                        anchor_layer=ANCHORS[model],  # noqa: B023
                        estimand=estimand,
                        window=window,
                        arm=arm,
                        control=control,
                        left=left,
                        right=right,
                    ),
                    vals,
                )
            )

        for w in WINDOWS:
            for arm in ARMS:
                for control in CONTROLS:
                    add("T", w, arm, control, cells[w, arm, control])
                for control in ["matched", "norm_random"]:
                    add(
                        "paired_minus_control",
                        w,
                        arm,
                        control,
                        difference(cells[w, arm, "paired"], cells[w, arm, control]),
                    )
            for arm in ["router_logits", "moe_output", "all_router"]:
                add(
                    "residual_minus_arm",
                    w,
                    arm,
                    "paired",
                    difference(cells[w, "post_moe_residual", "paired"], cells[w, arm, "paired"]),
                )
            add(
                "all_router_minus_single_router",
                w,
                "all_router",
                "paired",
                difference(cells[w, "all_router", "paired"], cells[w, "router_logits", "paired"]),
            )
        for left, right in [
            ("full_step", "last_token"),
            ("divergence_k", "suffix_k"),
            ("full_step", "suffix_k"),
        ]:
            for arm in ARMS:
                add(
                    "window_difference",
                    "",
                    arm,
                    "paired",
                    difference(cells[left, arm, "paired"], cells[right, arm, "paired"]),
                    left,
                    right,
                )
            for arm in ["router_logits", "all_router"]:
                gap_l = difference(
                    cells[left, "post_moe_residual", "paired"], cells[left, arm, "paired"]
                )
                gap_r = difference(
                    cells[right, "post_moe_residual", "paired"], cells[right, arm, "paired"]
                )
                add(
                    "gap_window_difference",
                    "",
                    arm,
                    "paired",
                    difference(gap_l, gap_r),
                    left,
                    right,
                )
        groups = defaultdict(list)
        for meta, vals in panels:
            groups[tuple(sorted(vals))].append((meta, vals))
        for entries in groups.values():
            stats = summarize_group([v for _, v in entries], components)
            # Retain the historical loop binding even though only its metadata is consumed.
            for (meta, vals), stat in zip(entries, stats, strict=False):  # noqa: B007
                record = dict(
                    **meta,
                    **stat,
                    interval_type="source_component_percentile_pair_weighted",
                    draws=DRAWS,
                    seed=SEED,
                    source=str(path),
                    version="unified_windows_v1",
                )
                estimates.append(record)
        for (w, arm, control), vals in cells.items():
            for p, value in vals.items():
                pair_output.append(
                    dict(
                        model=model,
                        window=w,
                        arm=arm,
                        control=control,
                        pair_id=p,
                        component=components[p],
                        T=value,
                    )
                )
        evidence[model] = dict(
            analysis=str(path),
            analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=str(path.with_name("rows.jsonl.gz")),
            verified_pair_estimands=checked,
            input_N=len(components),
            input_K=len(set(components.values())),
            analysis_native_seed=json.loads(gzip.decompress(path.read_bytes()))["seed"],
            analysis_native_draws=json.loads(gzip.decompress(path.read_bytes()))["bootstrap_replicates"],
        )
        print(model, "verified", checked, "pair T; panels", len(panels), flush=True)
    package = dict(
        version="unified_windows_v1",
        draws=DRAWS,
        seed=SEED,
        estimand="mean pair T or directly paired contrast on its actual common eligible pairs",
        resampling="sample eligible receiver source components with replacement; retain all their pairs; ratio of pair sums to counts",
        inference_scope="fixed nine checkpoints, fixed discovery pool and matched donor assignment; pointwise intervals, no multiplicity adjustment or population-of-models test",
        diagnostics="median, 10% trimmed mean and removal of 15 largest absolute pairs are descriptive sensitivity only, not replacement endpoints",
        sources=evidence,
        estimates=estimates,
    )
    (OUT / "unified_windows_v1.json").write_text(json.dumps(package, indent=2) + "\n")
    fields = [k for k in estimates[0] if k != "eligible_pair_ids"]
    with (OUT / "unified_windows_v1.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(estimates)
    with (OUT / "window_pair_values_v1.csv").open("w") as f:
        writer = csv.DictWriter(f, list(pair_output[0]))
        writer.writeheader()
        writer.writerows(pair_output)
    print("Saved", len(estimates), "panels and", len(pair_output), "pair T values")


if __name__ == "__main__":
    main()
