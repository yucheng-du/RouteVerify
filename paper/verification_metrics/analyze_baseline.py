"""Baseline verification performance of the nine checkpoints on the unpatched receivers.

Units are the unpatched receivers of the archived window assay (one clean and one
error-bearing receiver per pair). The score is the margin M = score(INCORRECT) -
score(CORRECT); the verdict rule is M > 0 -> INCORRECT, M <= 0 -> CORRECT; the positive
class for ranking metrics is the error-bearing ("corrupt") receiver.

Everything is reported twice and never pooled: metric="mean" (mean_logprob margin,
PRIMARY) and metric="total" (total_logprob margin, SENSITIVITY).

All intervals are cluster bootstrap over receiver source components with the shared draw
sequence defined in common.py (50000 draws, PCG64(20260908), batches of 256). Every
endpoint of a model is computed in one pass over those draws.

Scope: baseline on the PRMBench paired assay used in this study; candidate-scoring
verdict. This is NOT an official PRMBench benchmark score.

Run: .venv/bin/python <this file>
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANCHORS,
    DRAWS,
    METRICS,
    MODEL_KEYS,
    MODELS,
    OUT,
    ROLE_TRUTH,
    SCOPE,
    SEED,
    ClusterBootstrap,
    load_window_rows,
    verdict_array,
    write_csv,
    write_json,
)

BASE_DIR = OUT / "baseline"

RATIO_ENDPOINTS = [
    "accuracy",
    "error_recall",
    "clean_false_positive_rate",
    "clean_accuracy",
    "mean_margin_clean",
    "mean_margin_corrupt",
    "mean_gap",
    "fraction_pairs_gap_positive",
    "fraction_pairs_both_correct",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stat_columns(prefix: str, stat: dict) -> dict:
    return {
        prefix: stat["point"],
        f"{prefix}_ci_low": stat["ci_low"],
        f"{prefix}_ci_high": stat["ci_high"],
        f"{prefix}_n_defined_draws": stat["n_defined_draws"],
        f"{prefix}_n_undefined_draws": stat["n_undefined_draws"],
    }


def analyze_one(label: str) -> dict:
    t0 = time.time()
    df = load_window_rows(label)

    unp = (
        df[df["treatment"] == "unpatched"]
        .sort_values(["pair_id", "receiver_role"], kind="stable")
        .reset_index(drop=True)
    )
    rer = df[df["treatment"] == "unpatched_rerun"].set_index(
        ["pair_id", "receiver_role"], drop=False
    )

    pair_ids = unp["pair_id"].to_numpy()
    roles = unp["receiver_role"].to_numpy()
    comps = unp["component"].to_numpy()
    is_corrupt = roles == "corrupt"
    n_units = len(unp)

    clusters = sorted(set(comps.tolist()))
    boot = ClusterBootstrap(clusters)
    cl_index = np.array([boot.index[c] for c in comps], dtype=np.int64)

    # ---- pair bookkeeping (pair-level endpoints ride on the corrupt receiver row)
    by_pair: dict = {}
    for i in range(n_units):
        by_pair.setdefault(pair_ids[i], {})[roles[i]] = i
    complete_pairs = sorted(p for p, d in by_pair.items() if set(d) == {"clean", "corrupt"})
    incomplete_pairs = sorted(p for p, d in by_pair.items() if set(d) != {"clean", "corrupt"})

    margins = {"mean": unp["margin_mean"].to_numpy(float), "total": unp["margin_total"].to_numpy(float)}

    # ---- rerun drift
    rerun_cols = {}
    for metric in METRICS:
        col = f"margin_{metric}"
        vals = np.full(n_units, np.nan)
        for i in range(n_units):
            k = (pair_ids[i], roles[i])
            if k in rer.index:
                vals[i] = float(rer.loc[k, col])
        rerun_cols[metric] = vals
    n_rerun_missing = int(np.isnan(rerun_cols["mean"]).sum())

    # ---- token counts
    token_counts = {
        "CORRECT": sorted({int(v) for v in df["correct_token_count"].dropna().unique()}),
        "INCORRECT": sorted({int(v) for v in df["incorrect_token_count"].dropna().unique()}),
        "CORRECT_unpatched": sorted({int(v) for v in unp["correct_token_count"].unique()}),
        "INCORRECT_unpatched": sorted({int(v) for v in unp["incorrect_token_count"].unique()}),
    }
    token_counts["constant_within_model"] = (
        len(token_counts["CORRECT"]) == 1 and len(token_counts["INCORRECT"]) == 1
    )

    receiver_rows = []
    pair_rows = []
    panel_rows = []
    validation = {
        "model": label,
        "key": MODEL_KEYS[label],
        "n_receivers_unpatched": n_units,
        "n_receivers_rerun": int(len(rer)),
        "n_rerun_missing_partners": n_rerun_missing,
        "n_pairs_total": int(len(by_pair)),
        "n_pairs_complete": len(complete_pairs),
        "pairs_incomplete": incomplete_pairs,
        "n_components_K": len(clusters),
        "token_counts": token_counts,
        "metrics": {},
    }

    per_metric = {}
    for metric in METRICS:
        M = margins[metric]
        v = verdict_array(M)
        truth = np.array([ROLE_TRUTH[r] for r in roles])
        correct = v == truth
        per_metric[metric] = {"M": M, "verdict": v, "correct": correct}

    # ---- receiver table
    for i in range(n_units):
        row = {
            "model": label,
            "scope": SCOPE,
            "pair_id": pair_ids[i],
            "component": comps[i],
            "receiver_role": roles[i],
            "truth": ROLE_TRUTH[roles[i]],
            "margin_mean": margins["mean"][i],
            "margin_total": margins["total"][i],
            "verdict_mean": per_metric["mean"]["verdict"][i],
            "verdict_total": per_metric["total"]["verdict"][i],
            "correct_mean": bool(per_metric["mean"]["correct"][i]),
            "correct_total": bool(per_metric["total"]["correct"][i]),
            "rerun_margin_mean": rerun_cols["mean"][i],
            "rerun_margin_total": rerun_cols["total"][i],
            "rerun_abs_diff_mean": abs(rerun_cols["mean"][i] - margins["mean"][i]),
            "rerun_abs_diff_total": abs(rerun_cols["total"][i] - margins["total"][i]),
            "rerun_verdict_flip_mean": (
                None
                if np.isnan(rerun_cols["mean"][i])
                else bool(
                    ("INCORRECT" if rerun_cols["mean"][i] > 0 else "CORRECT")
                    != per_metric["mean"]["verdict"][i]
                )
            ),
            "rerun_verdict_flip_total": (
                None
                if np.isnan(rerun_cols["total"][i])
                else bool(
                    ("INCORRECT" if rerun_cols["total"][i] > 0 else "CORRECT")
                    != per_metric["total"]["verdict"][i]
                )
            ),
        }
        receiver_rows.append(row)

    # ---- pair table
    for p in complete_pairs:
        ic, ie = by_pair[p]["clean"], by_pair[p]["corrupt"]
        row = {"model": label, "scope": SCOPE, "pair_id": p, "component": comps[ic]}
        for metric in METRICS:
            mC = margins[metric][ic]
            mE = margins[metric][ie]
            row[f"m_C_{metric}"] = mC
            row[f"m_E_{metric}"] = mE
            row[f"g_{metric}"] = mE - mC
            row[f"both_correct_{metric}"] = bool(
                per_metric[metric]["correct"][ic] and per_metric[metric]["correct"][ie]
            )
        pair_rows.append(row)

    # ---- bootstrap panels
    for metric in METRICS:
        M = per_metric[metric]["M"]
        v = per_metric[metric]["verdict"]
        correct = per_metric[metric]["correct"]

        clean_ind = (~is_corrupt).astype(float)
        corrupt_ind = is_corrupt.astype(float)
        pred_incorrect = (v == "INCORRECT").astype(float)

        pair_den = np.zeros(n_units)
        pair_g = np.zeros(n_units)
        pair_g_pos = np.zeros(n_units)
        pair_both = np.zeros(n_units)
        for p in complete_pairs:
            ic, ie = by_pair[p]["clean"], by_pair[p]["corrupt"]
            g = margins[metric][ie] - margins[metric][ic]
            pair_den[ie] = 1.0
            pair_g[ie] = g
            pair_g_pos[ie] = 1.0 if g > 0 else 0.0
            pair_both[ie] = 1.0 if (correct[ic] and correct[ie]) else 0.0

        num = np.column_stack(
            [
                correct.astype(float),                       # accuracy
                corrupt_ind * pred_incorrect,                # error_recall
                clean_ind * pred_incorrect,                  # clean FPR
                clean_ind * (1.0 - pred_incorrect),           # clean accuracy
                clean_ind * M,                               # mean margin clean
                corrupt_ind * M,                             # mean margin corrupt
                pair_g,                                      # mean gap
                pair_g_pos,                                  # fraction pairs g>0
                pair_both,                                   # fraction pairs both correct
            ]
        )
        den = np.column_stack(
            [
                np.ones(n_units),
                corrupt_ind,
                clean_ind,
                clean_ind,
                clean_ind,
                corrupt_ind,
                pair_den,
                pair_den,
                pair_den,
            ]
        )

        res = boot.panel_stats(
            cl_index,
            num,
            den,
            auroc_specs=[{"name": "auroc", "scores": M, "is_positive": is_corrupt}],
            keep_draws=True,
        )
        stats = dict(zip(RATIO_ENDPOINTS, res["ratios"]))
        auroc = res["auroc"][0]

        # balanced accuracy derived from the SAME draws
        rd = res["ratio_draws"]
        recall_col = rd[:, RATIO_ENDPOINTS.index("error_recall")]
        fpr_col = rd[:, RATIO_ENDPOINTS.index("clean_false_positive_rate")]
        bal_draws = (recall_col + 1.0 - fpr_col) / 2.0
        bal_defined = np.isfinite(bal_draws)
        bal_point = (
            None
            if stats["error_recall"]["point"] is None or stats["clean_false_positive_rate"]["point"] is None
            else (stats["error_recall"]["point"] + 1.0 - stats["clean_false_positive_rate"]["point"]) / 2.0
        )
        bal_stat = ClusterBootstrap._summarize(bal_point, bal_draws, bal_defined)

        confusion = {
            "clean_to_CORRECT": int(((~is_corrupt) & (v == "CORRECT")).sum()),
            "clean_to_INCORRECT": int(((~is_corrupt) & (v == "INCORRECT")).sum()),
            "corrupt_to_CORRECT": int((is_corrupt & (v == "CORRECT")).sum()),
            "corrupt_to_INCORRECT": int((is_corrupt & (v == "INCORRECT")).sum()),
        }
        confusion["total"] = sum(confusion.values())

        row = {
            "model": label,
            "key": MODEL_KEYS[label],
            "anchor_layer": ANCHORS[label],
            "metric": metric,
            "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
            "scope": SCOPE,
            "n_receivers": n_units,
            "n_pairs": len(complete_pairs),
            "K": len(clusters),
            "n_clean": int((~is_corrupt).sum()),
            "n_corrupt": int(is_corrupt.sum()),
            "draws": DRAWS,
            "seed": SEED,
            "draws_signature": res["draws_signature"],
        }
        for name in RATIO_ENDPOINTS:
            row.update(_stat_columns(name, stats[name]))
        row.update(_stat_columns("balanced_accuracy", bal_stat))
        row.update(_stat_columns("auroc", auroc))
        row.update({f"confusion_{k}": v_ for k, v_ in confusion.items()})
        row["correct_token_count"] = (
            token_counts["CORRECT"][0] if len(token_counts["CORRECT"]) == 1 else None
        )
        row["incorrect_token_count"] = (
            token_counts["INCORRECT"][0] if len(token_counts["INCORRECT"]) == 1 else None
        )
        row["token_counts_constant"] = token_counts["constant_within_model"]
        panel_rows.append(row)

        undefined = {
            name: stats[name]["n_undefined_draws"] for name in RATIO_ENDPOINTS
        }
        undefined["balanced_accuracy"] = bal_stat["n_undefined_draws"]
        undefined["auroc"] = auroc["n_undefined_draws"]
        validation["metrics"][metric] = {
            "confusion": confusion,
            "confusion_total_equals_n_receivers": confusion["total"] == n_units,
            "undefined_draws": undefined,
            "max_rerun_abs_diff": float(np.nanmax(np.abs(rerun_cols[metric] - margins[metric])))
            if n_units
            else None,
            "n_rerun_verdict_flips": int(
                sum(
                    1
                    for i in range(n_units)
                    if not np.isnan(rerun_cols[metric][i])
                    and ("INCORRECT" if rerun_cols[metric][i] > 0 else "CORRECT") != v[i]
                )
            ),
            "rerun_verdict_flips": [
                {
                    "pair_id": pair_ids[i],
                    "receiver_role": roles[i],
                    "margin": float(margins[metric][i]),
                    "rerun_margin": float(rerun_cols[metric][i]),
                }
                for i in range(n_units)
                if not np.isnan(rerun_cols[metric][i])
                and ("INCORRECT" if rerun_cols[metric][i] > 0 else "CORRECT") != v[i]
            ],
            "draws_signature": res["draws_signature"],
        }

    validation["elapsed_seconds"] = time.time() - t0
    print(
        f"[{label}] N={n_units} pairs={len(complete_pairs)} K={len(clusters)} "
        f"acc={panel_rows[0]['accuracy']:.4f} auroc={panel_rows[0]['auroc']:.4f} "
        f"({validation['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    return {
        "panels": panel_rows,
        "receivers": receiver_rows,
        "pairs": pair_rows,
        "validation": validation,
    }


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = utcnow()
    with mp.Pool(9) as pool:
        results = pool.map(analyze_one, MODELS, chunksize=1)

    order = {m: i for i, m in enumerate(MODELS)}
    panels = pd.DataFrame([r for res in results for r in res["panels"]])
    panels = panels.sort_values(
        by=["model", "metric"], key=lambda s: s.map(order) if s.name == "model" else s, kind="stable"
    ).reset_index(drop=True)
    receivers = pd.DataFrame([r for res in results for r in res["receivers"]])
    pairs = pd.DataFrame([r for res in results for r in res["pairs"]])

    write_csv(panels, BASE_DIR / "baseline_panels_v1.csv")
    write_csv(receivers, BASE_DIR / "baseline_receivers_v1.csv")
    write_csv(pairs, BASE_DIR / "baseline_pairs_v1.csv")

    panel_json = {
        "version": "baseline_panels_v1",
        "scope": SCOPE,
        "notes": {
            "verdict_rule": "M > 0 -> INCORRECT, M <= 0 -> CORRECT (candidate scoring, not free generation)",
            "margin": "M = score(INCORRECT) - score(CORRECT); metric=mean is PRIMARY, metric=total is a separate SENSITIVITY analysis and is never pooled with it",
            "positive_class": "corrupt (error-bearing) receiver",
            "balanced_accuracy": "with the 1:1 clean/corrupt label balance of this assay balanced accuracy equals accuracy exactly, in every bootstrap draw; it is reported for convenience and must NOT be presented as independent evidence",
            "clean_accuracy": "= 1 - clean_false_positive_rate; reported for convenience only",
            "interval": "cluster bootstrap over receiver source components, 50000 draws, PCG64(20260908), two-sided pointwise percentile; no multiplicity adjustment",
            "auroc": "weighted tie-aware AUROC; a receiver's weight in a draw is the multiplicity of its cluster",
            "not_a_benchmark": "these are within-study assay numbers, not official PRMBench benchmark scores",
        },
        "draws": DRAWS,
        "seed": SEED,
        "started_utc": started,
        "finished_utc": utcnow(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_order": MODELS,
        "panels": panels.to_dict(orient="records"),
    }
    write_json(panel_json, BASE_DIR / "baseline_panels_v1.json")

    validation = {
        "version": "validation_baseline_v1",
        "scope": SCOPE,
        "started_utc": started,
        "finished_utc": utcnow(),
        "elapsed_seconds": time.time() - t0,
        "python": sys.version,
        "platform": platform.platform(),
        "expected_receivers_per_model": 580,
        "models": {res["validation"]["model"]: res["validation"] for res in results},
        "model_order": MODELS,
        "undefined_draws_note": "conditional-rate denominators are recomputed inside each draw; any draw with a zero denominator is counted and excluded from the quantile. With every cluster containing at least one clean and one corrupt receiver none are expected.",
    }
    write_json(validation, BASE_DIR / "validation_baseline.json")
    print(f"baseline written; total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
