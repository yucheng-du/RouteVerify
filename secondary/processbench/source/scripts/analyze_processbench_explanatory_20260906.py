#!/usr/bin/env python3
"""Retrospective CPU explanations of the frozen 64 ProcessBench natural pairs.

Reads only explicit ProcessBench paths. No model, activation, or other benchmark
prediction data is loaded. All new analyses are exploratory; canonical artifacts
are verified and never overwritten. Requires NumPy; figure output uses Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from routeverify.analysis.bootstrap import paired_cluster_bootstrap_indices  # noqa: E402
from routeverify.analysis.processbench_transfer import (  # noqa: E402
    summarize_processbench_transfer_rows,
)

ARCHIVE = Path(
    "archive://RouteVerify_ProcessBench_3Model_NaturalError_Transfer_20260829"
)
RUNS = {
    "gpt_oss_20b": "gpt_oss_processbench_full_20260829T094519Z",
    "gemma4": "gemma4_processbench_full_20260829T104105Z",
    "nemotron3_nano": "nemotron3_nano_processbench_full_20260829T112713Z",
}
NAMES = {"gpt_oss_20b": "GPT-OSS", "gemma4": "Gemma 4", "nemotron3_nano": "Nano"}
DIRECTIONS = {
    "clean_to_corrupt": ("matched_correct", "first_error"),
    "corrupt_to_clean": ("first_error", "matched_correct"),
}
SITES = ("post_moe_residual", "router_logits")
ROLES = ("matched_correct", "first_error")
N_RESAMPLES = 50_000
SEED = 0
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, kind="json"):
    path = path.resolve(strict=True)
    content = path.read_text()
    if kind == "jsonl":
        result = [json.loads(line) for line in content.splitlines() if line]
    elif kind == "csv":
        result = list(csv.DictReader(content.splitlines()))
    elif kind == "json":
        result = json.loads(content)
    else:
        result = content
    INPUTS[str(path)] = {
        "path": str(path),
        "sha256": sha(path),
        "size_bytes": path.stat().st_size,
        "rows": len(result) if kind in {"csv", "jsonl"} else None,
    }
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def margin(row):
    value = row["scores"]["INCORRECT"]["mean_logprob"] - row["scores"]["CORRECT"]["mean_logprob"]
    assert np.isfinite(value)
    assert abs(value - row["margin_incorrect_minus_correct_mean"]) < 1e-12
    return value


def distribution(values):
    x = np.asarray(values, dtype=float)
    return dict(
        n=len(x),
        mean=float(x.mean()),
        sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        min=float(x.min()),
        q25=float(np.quantile(x, 0.25)),
        median=float(np.median(x)),
        q75=float(np.quantile(x, 0.75)),
        max=float(x.max()),
    )


def interval(values, indices):
    x = np.asarray(values, dtype=float)
    means = x[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"estimate": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi)}, means


def ranks(x):
    x = np.asarray(x)
    return np.array([(np.sum(x < v) + (np.sum(x == v) - 1) / 2) for v in x])


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def figures(output, pairs, subgroup):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.05), gridspec_kw={"width_ratios": [1, 1.12]})
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.80, wspace=0.68)
    colors = {True: "#196C9F", False: "#C77728"}
    gp = [
        r
        for r in pairs
        if r["model_id"] == "gemma4"
        and r["site_family"] == "post_moe_residual"
        and r["direction"] == "clean_to_corrupt"
    ]
    gp.sort(key=lambda r: r["delta"])
    for correct in (True, False):
        axes[0].scatter(
            [i + 1 for i, r in enumerate(gp) if r["receiver_judgment_correct"] == correct],
            [r["delta"] for r in gp if r["receiver_judgment_correct"] == correct],
            s=17,
            c=colors[correct],
            linewidths=0,
            label="Error detected" if correct else "Error missed",
        )
    axes[0].axhline(0, color=".5", lw=0.7)
    axes[0].axhline(np.mean([r["delta"] for r in gp]), color=".3", lw=1, linestyle="--")
    axes[0].set(
        xlabel="Pair rank (all 64)",
        ylabel="Residual effect on margin",
        title="(a) Gemma pair effects",
    )
    axes[0].legend(loc="lower right", fontsize=8, frameon=False, handletextpad=0.4)
    axes[0].set_xticks([1, 16, 32, 48, 64])
    y = 5
    ticks, labels = [], []
    for model in RUNS:
        for correct in (True, False):
            r = next(
                r
                for r in subgroup
                if r["model_id"] == model
                and r["site_family"] == "post_moe_residual"
                and r["direction"] == "clean_to_corrupt"
                and r["receiver_judgment_correct"] == correct
            )
            axes[1].errorbar(
                r["estimate"],
                y,
                xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                color=colors[correct],
                fmt="o",
                markersize=4,
                capsize=2,
                lw=1,
            )
            ticks.append(y)
            labels.append(f"{NAMES[model]} ({r['n_pairs']})")
            y -= 1
        y -= 0.6
    axes[1].axvline(0, color=".5", lw=0.7)
    axes[1].set_yticks(ticks, labels)
    axes[1].set(xlabel="Mean effect (95% CI)", title="(b) Error detection state")
    axes[1].set_ylim(y + 0.1, 5.8)
    axes[1].set_xlim(-0.20, 0.63)
    axes[1].set_xticks([-0.2, 0, 0.3, 0.6])
    # Target-size artwork: 5.5 in wide at scale 1; no bbox cropping or font downscaling.
    fig.suptitle("ProcessBench: correct-to-error residual transfer", y=0.98, fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bbox = text.get_window_extent(renderer)
            if (
                bbox.x0 < -1
                or bbox.y0 < -1
                or bbox.x1 > fig.bbox.x1 + 1
                or bbox.y1 > fig.bbox.y1 + 1
            ):
                outside.append(text.get_text())
    assert not outside, outside
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / "receiver_state_explanation.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(output / "receiver_state_explanation.png", dpi=240)
    plt.close(fig)
    return {
        "width_inches": 5.5,
        "height_inches": 3.05,
        "intended_latex_width": "5.5in = manuscript textwidth at scale 1",
        "smallest_font_points": 8,
        "all_text_within_canvas": True,
        "final_manuscript_embedding_verified": False,
        "matplotlib_version": matplotlib.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument(
        "--output", type=Path, default=REPO / "docs/results/processbench_explanatory_cpu_20260906"
    )
    parser.add_argument(
        "--figures", type=Path, default=REPO / "figures/processbench_explanatory_cpu_20260906"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processed_path = REPO / "data/processed/processbench-natural-transfer-v1.jsonl"
    selection_path = REPO / "data/processed/processbench-natural-transfer-v1.selection.json"
    assert sha(processed_path) == "b1d7f21aff6d5a17d1494b5e8bcc2a4bf43d0e2f7d6689921ccb368d4e735c94"
    assert sha(selection_path) == "0f2d885fc9e7657886f0686a08f1236b62c595703c414d80069cc92b304ca7a2"
    processed = read(processed_path, "jsonl")
    selection = read(selection_path)
    selected = {r["pair_id"]: r for r in selection["pairs"]}
    ids = sorted(selected)
    assert len(ids) == 64 and len(processed) == 128
    traces = {(r["pair_id"], r["metadata"]["transfer_role"]): r for r in processed}
    assert len(traces) == len(processed)
    assert len({r["sample_id"] for r in processed}) == 128
    for pair in selected.values():
        assert pair["same_generator"] and not pair["same_normalized_problem"]
        assert pair["first_error"]["generator"] == pair["matched_correct"]["generator"]
        for role in ROLES:
            r = traces[pair["pair_id"], role]
            assert pair[role]["sample_id"] == r["sample_id"]
            assert pair["boundary_step"] == r["metadata"]["mapped_boundary_step"]
            assert len(r["steps"]) == pair[role]["num_steps"]
            assert r["error_type"] is None
    canon = REPO / "docs/results/unified_cpu_reanalysis_20260829"
    registry = read(canon / "source_files.csv", "csv")
    source_hashes = {r["path"]: r["sha256"] for r in registry}
    canonical = []
    for name in ("directional_effects", "donor_aligned_t", "residual_router_gaps"):
        canonical += [
            r | {"table": name}
            for r in read(canon / f"{name}.csv", "csv")
            if r["benchmark"] == "processbench"
        ]
    read(REPO / "configs/analysis/processbench_natural_transfer.yaml", "text")
    read(REPO / "docs/results/processbench_natural_transfer_20260829/SOURCE_PROVENANCE.md", "text")
    indices = paired_cluster_bootstrap_indices(64, seed=SEED, n_resamples=N_RESAMPLES)
    index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    assert index_hash == "26741af72ce5517a0f8c5f66fdb9b4ffda6cc9283be47c4b9be357c6a2655051"
    # Three repeated normalized problems connect a few otherwise distinct pairs.
    # A component draw carries every linked pair; ratio means retain pair weights.
    parents = {pid: pid for pid in ids}

    def find(pid):
        while parents[pid] != pid:
            pid = parents[pid]
        return pid

    by_problem = {}
    for (pid, _role), trace in traces.items():
        h = trace["metadata"]["normalized_problem_hash"]
        if h in by_problem:
            a, b = sorted((find(pid), find(by_problem[h])))
            parents[b] = a
        by_problem[h] = pid
    components = {}
    for pid in ids:
        components.setdefault(find(pid), []).append(pid)
    component_ids = sorted(components)
    component_index = {pid: i for i, cid in enumerate(component_ids) for pid in components[cid]}
    ci = paired_cluster_bootstrap_indices(len(components), seed=SEED, n_resamples=N_RESAMPLES)
    counts = np.zeros((N_RESAMPLES, len(components)), dtype=np.int64)
    np.add.at(counts, (np.arange(N_RESAMPLES)[:, None], ci), 1)
    component_weights = counts[:, [component_index[pid] for pid in ids]]
    component_sensitivity = []
    construction = []
    for pid in ids:
        p = selected[pid]
        record = {
            "pair_id": pid,
            "problem_component_id": component_ids[component_index[pid]],
            "generator": p["first_error"]["generator"],
            "boundary_step_zero_based": p["boundary_step"],
        }
        for role in ROLES:
            r = traces[pid, role]
            visible = "\n".join(r["steps"][: p["boundary_step"] + 1])
            record.update(
                {
                    f"{role}_sample_id": r["sample_id"],
                    f"{role}_full_trace_steps": len(r["steps"]),
                    f"{role}_visible_steps": p["boundary_step"] + 1,
                    f"{role}_visible_step_chars": len(visible),
                    f"{role}_problem_chars": len(r["problem"]),
                    f"{role}_relative_boundary": (p["boundary_step"] + 1) / len(r["steps"]),
                }
            )
        construction.append(record)
    pair_rows, overall, subgroups, contrasts, states, dispatch, recovery, model_checks = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for model, run in RUNS.items():
        root = args.archive / "runs" / run
        transfer_path, behavior_path = (
            root / "transfer/predictions.jsonl",
            root / "behavior/predictions.jsonl",
        )
        # Match canonical prediction hashes even when the archive is relocated.
        for path in (transfer_path, behavior_path):
            old = str(ARCHIVE / "runs" / run / path.parent.name / path.name)
            assert sha(path) == source_hashes[old]
        transfer = read(transfer_path, "jsonl")
        behavior = read(behavior_path, "jsonl")
        recap = read(root / "recapture/data_manifest.jsonl", "jsonl")
        legacy = summarize_processbench_transfer_rows(transfer, seed=0, n_resamples=1000)
        assert legacy["n_pairs"] == 64 and legacy["n_rows"] == 768
        assert legacy["identity"]["passed"] and legacy["router_native_dispatch"]["passed"]
        assert len(behavior) == len(recap) == 128
        bmap = {(r["pair_id"], r["transfer_role"]): r for r in behavior}
        rmap = {(r["pair_id"], r["transfer_role"]): r for r in recap}
        baseline = {
            (r["pair_id"], r["receiver_role"]): margin(r)
            for r in transfer
            if r["treatment"] == "unpatched"
        }
        tmap = {
            (r["pair_id"], r["treatment"], (r["site"] or "").split(".")[-1]): r
            for r in transfer
            if r["treatment"] in DIRECTIONS
        }
        assert len(baseline) == len(bmap) == len(rmap) == 128 and len(tmap) == 256
        for pid in ids:
            for role in ROLES:
                assert (pid, role) in baseline and (pid, role) in bmap and (pid, role) in rmap
                assert abs(margin(bmap[pid, role]) - baseline[pid, role]) < 1e-12
                assert (
                    bmap[pid, role]["sample_id"]
                    == rmap[pid, role]["sample_id"]
                    == traces[pid, role]["sample_id"]
                )
                assert rmap[pid, role]["boundary_step"] == selected[pid]["boundary_step"]
                assert bmap[pid, role]["prediction"] == (
                    "INCORRECT" if baseline[pid, role] > 0 else "CORRECT"
                )
        vectors = {}
        for direction, (_donor, receiver) in DIRECTIONS.items():
            for site in SITES:
                records = []
                for pid in ids:
                    row = tmap[pid, direction, site]
                    assert row["receiver_role"] == receiver and row["donor_role"] == _donor
                    assert row["boundary_step"] == selected[pid]["boundary_step"]
                    assert row["token_position"] == rmap[pid, receiver]["token_position"]
                    assert row["metadata"]["alpha"] == 1.0
                    b = baseline[pid, receiver]
                    correct = (b > 0) if receiver == "first_error" else (b <= 0)
                    state = ("C" if baseline[pid, "matched_correct"] <= 0 else "I") + (
                        "C" if baseline[pid, "first_error"] <= 0 else "I"
                    )
                    rec = {
                        "model_id": model,
                        "pair_id": pid,
                        "site_family": site,
                        "direction": direction,
                        "receiver_role": receiver,
                        "donor_role": _donor,
                        "receiver_margin": b,
                        "donor_margin": baseline[pid, _donor],
                        "patched_margin": margin(row),
                        "delta": margin(row) - b,
                        "receiver_judgment_correct": correct,
                        "pair_judgment_state_correct_then_error": state,
                        "receiver_boundary_token_position": rmap[pid, receiver]["token_position"],
                        "donor_boundary_token_position": rmap[pid, _donor]["token_position"],
                        "boundary_step_zero_based": selected[pid]["boundary_step"],
                        "topk_changed_rate": None,
                        "topk_overlap": None,
                        "router_kl_receiver_to_patched": None,
                        "native_dispatch_verified": None,
                    }
                    if site == "router_logits":
                        m = row["metadata"]["manipulation"]
                        assert m["all_router_dispatch_verified"] and m["candidate_count"] == 2
                        for key in (
                            "topk_changed_rate",
                            "topk_overlap",
                            "router_kl_receiver_to_patched",
                        ):
                            rec[key] = m[key]
                        rec["native_dispatch_verified"] = True
                    records.append(rec)
                pair_rows.extend(records)
                x = np.array([r["delta"] for r in records])
                vectors[direction, site] = x
                point, _ = interval(x, indices)
                key = {"model_id": model, "site_family": site, "direction": direction}
                trim = int(0.1 * len(x))
                loo = (x.sum() - x) / (len(x) - 1)
                absolute_order = np.argsort(-np.abs(x), kind="stable")
                overall.append(
                    key
                    | point
                    | {
                        "n_pairs": 64,
                        "mean_absolute_delta": float(np.abs(x).mean()),
                        "median": float(np.median(x)),
                        "positive_pairs": int((x > 0).sum()),
                        "negative_pairs": int((x < 0).sum()),
                        "zero_pairs": int((x == 0).sum()),
                        "trim_each_tail_n": trim,
                        "trimmed_mean": float(np.sort(x)[trim:-trim].mean()),
                        "leave_one_out_mean_min": float(loo.min()),
                        "leave_one_out_mean_max": float(loo.max()),
                        "top5_absolute_effect_share": float(
                            np.abs(x)[absolute_order[:5]].sum() / np.abs(x).sum()
                        ),
                        "top5_absolute_effect_pair_ids": ";".join(
                            ids[i] for i in absolute_order[:5]
                        ),
                        "top5_signed_contribution_to_full_mean": float(
                            x[absolute_order[:5]].sum() / len(x)
                        ),
                        "remaining59_contribution_to_full_mean": float(
                            np.delete(x, absolute_order[:5]).sum() / len(x)
                        ),
                    }
                )
                # Fixed observed receiver-state strata, independently resampled;
                # contrast is computed from joint draws, never CI overlap.
                rng = np.random.Generator(np.random.PCG64(SEED))
                draws, points = {}, {}
                for correct in (True, False):
                    sx = np.array(
                        [r["delta"] for r in records if r["receiver_judgment_correct"] == correct]
                    )
                    si = rng.integers(0, len(sx), size=(N_RESAMPLES, len(sx)), dtype=np.int64)
                    summary, means = interval(sx, si)
                    draws[correct], points[correct] = means, summary["estimate"]
                    subgroups.append(
                        key
                        | {"receiver_judgment_correct": correct, "n_pairs": len(sx)}
                        | summary
                        | {
                            "positive_pairs": int((sx > 0).sum()),
                            "negative_pairs": int((sx < 0).sum()),
                        }
                    )
                component_draws = {}
                for correct in (True, False):
                    mask = np.array([r["receiver_judgment_correct"] == correct for r in records])
                    weights = component_weights[:, mask]
                    denom = weights.sum(axis=1)
                    component_draws[correct] = np.divide(
                        weights @ x[mask], denom, out=np.full(N_RESAMPLES, np.nan), where=denom > 0
                    )
                component_statistics = {
                    "all_pairs": component_weights @ x / component_weights.sum(axis=1),
                    "receiver_correctly_judged": component_draws[True],
                    "receiver_misjudged": component_draws[False],
                    "misjudged_minus_correctly_judged": component_draws[False]
                    - component_draws[True],
                }
                for statistic, cdraw in component_statistics.items():
                    valid = np.isfinite(cdraw)
                    lo, hi = np.quantile(cdraw[valid], [0.025, 0.975])
                    estimate = (
                        float(x.mean())
                        if statistic == "all_pairs"
                        else points[True]
                        if statistic == "receiver_correctly_judged"
                        else points[False]
                        if statistic == "receiver_misjudged"
                        else points[False] - points[True]
                    )
                    component_sensitivity.append(
                        key
                        | {
                            "statistic": statistic,
                            "estimate": estimate,
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "n_components": len(components),
                            "valid_replicates": int(valid.sum()),
                            "empty_stratum_replicates": int((~valid).sum()),
                        }
                    )
                difference = draws[False] - draws[True]
                lo, hi = np.quantile(difference, [0.025, 0.975])
                contrasts.append(
                    key
                    | {
                        "contrast": "receiver_misjudged_minus_correctly_judged",
                        "estimate": points[False] - points[True],
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                    }
                )
                for state in ("CI", "CC", "II", "IC"):
                    sx = np.array(
                        [
                            r["delta"]
                            for r in records
                            if r["pair_judgment_state_correct_then_error"] == state
                        ]
                    )
                    states.append(
                        key
                        | {
                            "pair_state": state,
                            "n_pairs": len(sx),
                            "mean": float(sx.mean()),
                            "contribution_to_full_mean": float(sx.sum() / 64),
                        }
                    )
                if site == "router_logits":
                    changed = np.array([r["topk_changed_rate"] > 0 for r in records])
                    replacement = np.array([1 - r["topk_overlap"] for r in records])
                    kl = np.array([r["router_kl_receiver_to_patched"] for r in records])
                    dispatch.append(
                        key
                        | {
                            "n_pairs": 64,
                            "changed_pairs": int(changed.sum()),
                            "unchanged_pairs": int((~changed).sum()),
                            "mean_topk_replaced_fraction": float(replacement.mean()),
                            "mean_absolute_delta": float(np.abs(x).mean()),
                            "changed_mean_absolute_delta": float(np.abs(x[changed]).mean()),
                            "unchanged_mean_absolute_delta": float(np.abs(x[~changed]).mean()),
                            "spearman_replaced_fraction_abs_delta": spearman(
                                replacement, np.abs(x)
                            ),
                            "spearman_router_kl_abs_delta": spearman(kl, np.abs(x)),
                        }
                    )
        # Exact existing full-sample estimands, including T and paired residual-router gap.
        tv = {
            site: 0.5 * (vectors["corrupt_to_clean", site] - vectors["clean_to_corrupt", site])
            for site in SITES
        }
        for site, x in [
            *tv.items(),
            ("residual_minus_router", tv["post_moe_residual"] - tv["router_logits"]),
        ]:
            cdraw = component_weights @ x / component_weights.sum(axis=1)
            lo, hi = np.quantile(cdraw, [0.025, 0.975])
            component_sensitivity.append(
                {
                    "model_id": model,
                    "site_family": site,
                    "direction": "paired_directions",
                    "statistic": "residual_minus_router_gap"
                    if site == "residual_minus_router"
                    else "donor_aligned_T",
                    "estimate": float(x.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "n_components": len(components),
                    "valid_replicates": N_RESAMPLES,
                    "empty_stratum_replicates": 0,
                }
            )
        for r in [r for r in canonical if r["model_id"] == model]:
            if r["table"] == "directional_effects":
                x = vectors[r["direction"], r["site"].split(".")[-1]]
            elif r["table"] == "donor_aligned_t":
                x = tv[r["site"].split(".")[-1]]
            else:
                x = tv["post_moe_residual"] - tv["router_logits"]
            current, _ = interval(x, indices)
            errors = {k: abs(current[k] - float(r[k])) for k in current}
            assert max(errors.values()) < 1e-12, (r["row_id"], errors)
            recovery.append(
                {
                    "canonical_row_id": r["row_id"],
                    "max_absolute_error": max(errors.values()),
                    "pass": True,
                }
            )
        tokens = {role: [rmap[pid, role]["token_position"] for pid in ids] for role in ROLES}
        differences = np.array(tokens["first_error"]) - np.array(tokens["matched_correct"])
        model_checks.append(
            {
                "model_id": model,
                "transfer_rows": len(transfer),
                "behavior_rows": len(behavior),
                "recapture_metadata_rows": len(recap),
                "judgment_accuracy": sum(r["prediction"] == r["gold"] for r in behavior) / 128,
                "pair_state_counts": dict(
                    Counter(
                        ("C" if baseline[pid, "matched_correct"] <= 0 else "I")
                        + ("C" if baseline[pid, "first_error"] <= 0 else "I")
                        for pid in ids
                    )
                ),
                "boundary_token_position": {role: distribution(tokens[role]) for role in ROLES},
                "error_minus_correct_boundary_tokens": distribution(differences),
                "absolute_boundary_token_difference": distribution(np.abs(differences)),
                "native_dispatch": legacy["router_native_dispatch"],
                "identity": legacy["identity"],
            }
        )
    coverage = {
        "n_pairs": 64,
        "n_unique_traces": 128,
        "n_unique_normalized_problems": len(by_problem),
        "n_problem_linked_pair_components": len(components),
        "problem_component_sizes": dict(
            sorted(Counter(len(v) for v in components.values()).items())
        ),
        "n_generators": len({r["generator"] for r in construction}),
        "generators": dict(sorted(Counter(r["generator"] for r in construction).items())),
        "boundary_step_zero_based": dict(
            sorted(Counter(r["boundary_step_zero_based"] for r in construction).items())
        ),
        "first_step_error_pairs": sum(r["boundary_step_zero_based"] == 0 for r in construction),
        "source_population_counts_from_frozen_manifest": selection["counts"]["by_subset"]["gsm8k"],
        "full_trace_steps": {
            role: distribution([r[f"{role}_full_trace_steps"] for r in construction])
            for role in ROLES
        },
        "visible_step_chars": {
            role: distribution([r[f"{role}_visible_step_chars"] for r in construction])
            for role in ROLES
        },
        "absolute_visible_step_char_difference": distribution(
            [
                abs(r["first_error_visible_step_chars"] - r["matched_correct_visible_step_chars"])
                for r in construction
            ]
        ),
        "relative_boundary": {
            role: distribution([r[f"{role}_relative_boundary"] for r in construction])
            for role in ROLES
        },
        "same_total_trace_step_count_pairs": sum(
            r["first_error_full_trace_steps"] == r["matched_correct_full_trace_steps"]
            for r in construction
        ),
        "native_error_type_labels_present": 0,
    }
    output_tables = {
        "problem_component_sensitivity": component_sensitivity,
        "construction_pairs": construction,
        "pair_effects": pair_rows,
        "effect_distribution": overall,
        "receiver_state_effects": subgroups,
        "receiver_state_contrasts": contrasts,
        "pair_state_decomposition": states,
        "dispatch_associations": dispatch,
        "canonical_recovery": recovery,
    }
    for name, rows in output_tables.items():
        write_csv(args.output / f"{name}.csv", rows)
    write_json(args.output / "coverage.json", coverage)
    write_json(args.output / "model_checks.json", model_checks)
    layout = figures(args.figures, pair_rows, subgroups)
    script = Path(__file__).resolve()
    for code in (
        script,
        REPO / "src/routeverify/analysis/bootstrap.py",
        REPO / "src/routeverify/analysis/processbench_transfer.py",
    ):
        read(code, "text")
    provenance = {
        "analysis_id": "processbench_explanatory_cpu_20260906",
        "status": "retrospective_exploratory",
        "release_input": "ec8c04c890356db81ac1c19ff0b677515ab9f441",
        "analysis_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "analysis_git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ).splitlines(),
        "command": [
            sys.executable,
            str(script),
            "--archive",
            str(args.archive),
            "--output",
            str(args.output),
            "--figures",
            str(args.figures),
        ],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "validation_accessed": False,
        "model_execution_performed": False,
        "raw_activation_arrays_loaded": False,
        "source_prediction_rows": 2304,
        "source_behavior_rows": 384,
        "source_recapture_metadata_rows": 384,
        "derived_pair_direction_site_rows": len(pair_rows),
        "canonical_cells_recovered": len(recovery),
        "metric": "INCORRECT mean_logprob minus CORRECT mean_logprob; patched minus receiver unpatched",
        "full_sample_bootstrap": {
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "rng": "PCG64",
            "unit": "matched pair",
            "shared_indices_values_sha256": index_hash,
            "canonical_intervals_recovered": True,
        },
        "subgroup_bootstrap": {
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "rng": "PCG64",
            "unit": "matched pair",
            "method": "conditional fixed-size receiver-state strata; independent sequential draws for correctly judged then misjudged",
            "contrast": "misjudged minus correctly judged from joint independent-stratum draws",
            "interval": "marginal percentile 95%; exploratory; no multiplicity adjustment",
            "selection": "mean-margin sign before intervention; all pairs retained",
        },
        "problem_component_sensitivity": {
            "reason": "128 traces map to 125 normalized problems, linking 64 pairs into 61 components",
            "component_sizes": dict(sorted(Counter(len(v) for v in components.values()).items())),
            "n_resamples": N_RESAMPLES,
            "seed": SEED,
            "rng": "PCG64",
            "method": "sample 61 components with replacement, carrying all pairs; pair-weighted ratio means",
            "state_contrasts": "both subgroup means recomputed within each joint draw; empty-stratum draws counted and omitted",
            "canonical_replaced": False,
            "paired_endpoints": "residual T, router T, and T_residual-T_router calculated directly inside each component draw",
        },
        "influence": {
            "trim_each_tail": "floor(0.1*n)=6 of 64",
            "leave_one_out": "all 64 deletions, descriptive only",
            "top_k_absolute_effect": 5,
            "pair_exclusions_from_full_sample": 0,
        },
        "dispatch": "within model and direction only; tie-aware Spearman descriptive without p-values; no cross-model pooling",
        "construction": "selected frozen traces only; full lengths and relative positions describe hidden suffixes, not model-visible context; token positions are recapture boundary indices, not exact full prompt lengths",
        "limitations": [
            "no new donor control",
            "state association is not causal moderation or mediation",
            "delta shares the baseline used for grouping",
            "same-generator pairs are cross-problem and do not identify semantic correctness independently of content",
            "no error type labels inferred",
            "no population inference beyond frozen matched design without assumptions",
        ],
        "figure_layout": layout,
        "inputs": list(INPUTS.values()),
    }
    write_json(args.output / "provenance.json", provenance)
    outputs = sorted(
        [args.output / f"{name}.csv" for name in output_tables]
        + [args.output / name for name in ("coverage.json", "model_checks.json", "provenance.json")]
        + [args.figures / f"receiver_state_explanation.{suffix}" for suffix in ("pdf", "png")]
    )
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.relative_to(REPO)}\n" for p in outputs)
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "canonical_cells": len(recovery),
                "pair_effect_rows": len(pair_rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
