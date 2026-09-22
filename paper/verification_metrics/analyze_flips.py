"""Decision flip rates under patching, nine-model window panels (stage B1b).

A receiver's decision is the candidate-scoring verdict on the margin
M = score(INCORRECT) - score(CORRECT):  M > 0 -> INCORRECT, M <= 0 -> CORRECT.
Truth is fixed by the receiver role (clean -> CORRECT, error-bearing "corrupt" ->
INCORRECT). A receiver is *correct* when its verdict matches that truth.

For every panel (model x window x arm x control x metric) each receiver with a patched
row contributes one transition between its baseline (unpatched) decision and its patched
decision:

    cc  baseline correct  -> patched correct
    cw  baseline correct  -> patched wrong     (harmful flip)
    wc  baseline wrong    -> patched correct   (beneficial flip)
    ww  baseline wrong    -> patched wrong

plus the raw direction counts n_to_incorrect (verdict CORRECT -> INCORRECT) and
n_to_correct (INCORRECT -> CORRECT), which are label-free.

Two sample sets are reported side by side and never merged:
    common_pairs   PRIMARY   only pairs whose BOTH roles have a patched row in this panel
                             (the eligibility used by the archived pair-level T)
    all_receivers  SECONDARY every receiver with a patched row in this panel

and three receiver groups (all / clean / corrupt) inside each sample set, which share the
same bootstrap draws.

Endpoints (all with cluster-bootstrap CIs, numerator and denominator recomputed inside
every draw):
    flip_rate_all       = (cw + wc) / n
    harmful_rate_all    = cw / n
    beneficial_rate_all = wc / n
    harmful_rate_cond   = cw / (cc + cw)      denominator = baseline-correct receivers
    beneficial_rate_cond= wc / (wc + ww)      denominator = baseline-wrong receivers
    baseline_accuracy   = (cc + cw) / n
    patched_accuracy    = (cc + wc) / n
    delta_accuracy      = (wc - cw) / n       own endpoint; checked against
                                              beneficial_rate_all - harmful_rate_all
    to_incorrect_rate   = n_to_incorrect / n
    to_correct_rate     = n_to_correct / n

metric="mean" (mean_logprob margin) is PRIMARY; metric="total" (total_logprob margin) is a
separate SENSITIVITY analysis. They are never pooled. Baseline margins are joined by
(pair_id, receiver_role) ID from the unpatched rows (patched rows carry no total
baseline), never by line order.

Intervals: cluster bootstrap over receiver source components, 50000 draws, PCG64(20260908),
batches of 256, two-sided pointwise percentile, no multiplicity adjustment. Every endpoint
of a (panel, sample_set) is computed in ONE pass over the shared draws. A draw whose
denominator is zero is UNDEFINED: counted, excluded from the quantile, reported. A point
estimate with a zero denominator is null, never 0.

Read-only on the archives; writes only under
docs/results/verification_metrics_cpu_20260917/flips/.

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
    ARMS,
    CONTROLS,
    DRAWS,
    METRICS,
    MODEL_KEYS,
    MODELS,
    OUT,
    ROLE_TRUTH,
    SEED,
    WINDOWS,
    ClusterBootstrap,
    load_window_rows,
    write_csv,
    write_json,
)

FLIP_DIR = OUT / "flips"

# Scope string for this stage (common.SCOPE describes the baseline stage only).
SCOPE = (
    "decision flips under activation patching on the PRMBench paired assay used in this "
    "study; candidate-scoring verdict; baseline verdicts from the unpatched rows"
)

GROUPS = ["all", "clean", "corrupt"]
SAMPLE_SETS = ["common_pairs", "all_receivers"]

ENDPOINTS = [
    "flip_rate_all",
    "harmful_rate_all",
    "beneficial_rate_all",
    "harmful_rate_cond",
    "beneficial_rate_cond",
    "baseline_accuracy",
    "patched_accuracy",
    "delta_accuracy",
    "to_incorrect_rate",
    "to_correct_rate",
]

# column layout of the (units x E) ratio matrices: group-major, then metric, then endpoint
COLSPEC = [
    (g, m, e) for g in GROUPS for m in METRICS for e in ENDPOINTS
]
COLIDX = {k: i for i, k in enumerate(COLSPEC)}
E_TOTAL = len(COLSPEC)


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

    # ---- baseline verdicts, by ID -------------------------------------------------
    base = pd.read_csv(
        OUT / "baseline" / "baseline_receivers_v1.csv",
        dtype={"model": "string", "pair_id": "string", "receiver_role": "string"},
    )
    base = base[base["model"] == label]
    base_key = list(zip(base["pair_id"].tolist(), base["receiver_role"].tolist()))
    base_margin = {
        "mean": dict(zip(base_key, base["margin_mean"].to_numpy(float))),
        "total": dict(zip(base_key, base["margin_total"].to_numpy(float))),
    }
    base_correct_map = {
        "mean": dict(zip(base_key, base["correct_mean"].to_numpy(bool))),
        "total": dict(zip(base_key, base["correct_total"].to_numpy(bool))),
    }
    base_verdict_map = {
        "mean": dict(zip(base_key, base["verdict_mean"].tolist())),
        "total": dict(zip(base_key, base["verdict_total"].tolist())),
    }

    patched = df[df["treatment"].isin(CONTROLS)].reset_index(drop=True)

    panel_rows: list[dict] = []
    receiver_frames: list[pd.DataFrame] = []
    panel_counts: dict[str, int] = {}
    missing_panels: list[dict] = []
    checks = {
        "max_abs_delta_identity_point": 0.0,
        "max_abs_delta_identity_draws": 0.0,
        "transition_sum_mismatches": [],
        "duplicate_unit_keys": [],
        "baseline_join_misses": 0,
        "total_undefined_draws": 0,
        "n_panel_stats_calls": 0,
        "n_sample_sets_shared": 0,
    }
    self_drift = {
        m: {"max_abs_diff": 0.0, "flips": []} for m in METRICS
    }

    grouped = {k: v for k, v in patched.groupby(["window", "arm", "treatment"], sort=False)}

    for window in WINDOWS:
        for arm in ARMS:
            for control in CONTROLS:
                pkey = (window, arm, control)
                sub = grouped.get(pkey)
                pname = f"{window}|{arm}|{control}"
                if sub is None or len(sub) == 0:
                    missing_panels.append(
                        {
                            "model": label,
                            "window": window,
                            "arm": arm,
                            "control": control,
                            "n_rows": 0,
                        }
                    )
                    panel_counts[pname] = 0
                    continue
                sub = sub.sort_values(["pair_id", "receiver_role"], kind="stable").reset_index(
                    drop=True
                )
                panel_counts[pname] = int(len(sub))

                pair_ids = sub["pair_id"].to_numpy()
                roles = sub["receiver_role"].to_numpy()
                comps = sub["component"].to_numpy()
                n_units = len(sub)

                keys = list(zip(pair_ids.tolist(), roles.tolist()))
                if len(set(keys)) != n_units:
                    seen = set()
                    dups = []
                    for k in keys:
                        if k in seen:
                            dups.append(k)
                        seen.add(k)
                    checks["duplicate_unit_keys"].append(
                        {"model": label, "panel": pname, "keys": sorted(set(dups))}
                    )

                # common-pairs eligibility: both roles of the pair present in this panel
                roles_by_pair: dict = {}
                for i in range(n_units):
                    roles_by_pair.setdefault(pair_ids[i], set()).add(roles[i])
                in_common = np.array(
                    [roles_by_pair[p] == {"clean", "corrupt"} for p in pair_ids], dtype=bool
                )

                is_clean = roles == "clean"
                is_corrupt = roles == "corrupt"

                # ---- per-metric transitions --------------------------------------
                per_metric = {}
                for metric in METRICS:
                    pm = sub[f"margin_{metric}"].to_numpy(float)
                    bm = np.empty(n_units)
                    bcorr = np.empty(n_units, dtype=bool)
                    bverd = np.empty(n_units, dtype=object)
                    miss = 0
                    for i, k in enumerate(keys):
                        if k not in base_margin[metric]:
                            miss += 1
                            bm[i] = np.nan
                            bcorr[i] = False
                            bverd[i] = None
                            continue
                        bm[i] = base_margin[metric][k]
                        bcorr[i] = bool(base_correct_map[metric][k])
                        bverd[i] = base_verdict_map[metric][k]
                    checks["baseline_join_misses"] += miss

                    pverd = np.where(pm > 0, "INCORRECT", "CORRECT")
                    truth = np.array([ROLE_TRUTH[r] for r in roles], dtype=object)
                    pcorr = pverd == truth

                    cc = bcorr & pcorr
                    cw = bcorr & ~pcorr
                    wc = ~bcorr & pcorr
                    ww = ~bcorr & ~pcorr
                    to_inc = (np.array(bverd, dtype=object) == "CORRECT") & (
                        pverd == "INCORRECT"
                    )
                    to_cor = (np.array(bverd, dtype=object) == "INCORRECT") & (
                        pverd == "CORRECT"
                    )

                    tclass = np.where(
                        cc, "cc", np.where(cw, "cw", np.where(wc, "wc", "ww"))
                    )
                    per_metric[metric] = {
                        "patched_margin": pm,
                        "baseline_margin": bm,
                        "baseline_verdict": np.array(bverd, dtype=object),
                        "patched_verdict": pverd,
                        "cc": cc,
                        "cw": cw,
                        "wc": wc,
                        "ww": ww,
                        "to_inc": to_inc,
                        "to_cor": to_cor,
                        "tclass": tclass,
                    }

                    if control == "self":
                        d = np.abs(pm - bm)
                        if np.isfinite(d).any():
                            self_drift[metric]["max_abs_diff"] = max(
                                self_drift[metric]["max_abs_diff"], float(np.nanmax(d))
                            )
                        flip_mask = pverd != np.array(bverd, dtype=object)
                        for i in np.flatnonzero(flip_mask):
                            self_drift[metric]["flips"].append(
                                {
                                    "model": label,
                                    "window": window,
                                    "arm": arm,
                                    "metric": metric,
                                    "pair_id": pair_ids[i],
                                    "receiver_role": roles[i],
                                    "baseline_margin": float(bm[i]),
                                    "patched_margin": float(pm[i]),
                                }
                            )

                # ---- receiver-level long table ------------------------------------
                for metric in METRICS:
                    pmm = per_metric[metric]
                    receiver_frames.append(
                        pd.DataFrame(
                            {
                                "model": label,
                                "window": window,
                                "arm": arm,
                                "control": control,
                                "metric": metric,
                                "pair_id": pair_ids,
                                "component": comps,
                                "receiver_role": roles,
                                "donor_role": sub["donor_role"].to_numpy(),
                                "donor_pair_id": sub["donor_pair_id"].to_numpy(),
                                "baseline_margin": pmm["baseline_margin"],
                                "patched_margin": pmm["patched_margin"],
                                "baseline_verdict": pmm["baseline_verdict"],
                                "patched_verdict": pmm["patched_verdict"],
                                "transition_class": pmm["tclass"],
                                "in_common_pairs": in_common,
                            }
                        )
                    )

                # ---- bootstrap, per sample set -------------------------------------
                sample_masks = {
                    "common_pairs": in_common,
                    "all_receivers": np.ones(n_units, dtype=bool),
                }
                cache: dict[tuple, dict] = {}
                for sample_set in SAMPLE_SETS:
                    mask = sample_masks[sample_set]
                    sig = tuple(np.flatnonzero(mask).tolist())
                    if sig in cache:
                        checks["n_sample_sets_shared"] += 1
                        res_pack = cache[sig]
                    else:
                        res_pack = _bootstrap_units(
                            comps[mask], roles[mask], per_metric, mask, checks
                        )
                        cache[sig] = res_pack
                        checks["n_panel_stats_calls"] += 1

                    stats = res_pack["stats"]
                    for group in GROUPS:
                        gmask = mask.copy()
                        if group == "clean":
                            gmask &= is_clean
                        elif group == "corrupt":
                            gmask &= is_corrupt
                        gn = int(gmask.sum())
                        gpairs = int(len(set(pair_ids[gmask].tolist())))
                        for metric in METRICS:
                            pmm = per_metric[metric]
                            row = {
                                "model": label,
                                "key": MODEL_KEYS[label],
                                "anchor_layer": ANCHORS[label],
                                "window": window,
                                "arm": arm,
                                "control": control,
                                "metric": metric,
                                "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
                                "sample_set": sample_set,
                                "sample_set_role": (
                                    "PRIMARY" if sample_set == "common_pairs" else "SECONDARY"
                                ),
                                "receiver_group": group,
                                "scope": SCOPE,
                                "n_receivers": gn,
                                "n_pairs": gpairs,
                                "K": res_pack["K"],
                                "n_clean": int((gmask & is_clean).sum()),
                                "n_corrupt": int((gmask & is_corrupt).sum()),
                                "n_panel_rows": n_units,
                                "cc": int(pmm["cc"][gmask].sum()),
                                "cw": int(pmm["cw"][gmask].sum()),
                                "wc": int(pmm["wc"][gmask].sum()),
                                "ww": int(pmm["ww"][gmask].sum()),
                                "n_to_incorrect": int(pmm["to_inc"][gmask].sum()),
                                "n_to_correct": int(pmm["to_cor"][gmask].sum()),
                                "draws": DRAWS,
                                "seed": SEED,
                                "draws_signature": res_pack["draws_signature"],
                            }
                            tsum = row["cc"] + row["cw"] + row["wc"] + row["ww"]
                            if tsum != gn:
                                checks["transition_sum_mismatches"].append(
                                    {
                                        "model": label,
                                        "panel": pname,
                                        "sample_set": sample_set,
                                        "group": group,
                                        "metric": metric,
                                        "sum": tsum,
                                        "n": gn,
                                    }
                                )
                            for ep in ENDPOINTS:
                                st = stats[COLIDX[(group, metric, ep)]]
                                row.update(_stat_columns(ep, st))
                                checks["total_undefined_draws"] += st["n_undefined_draws"]
                            # point-estimate identity for delta_accuracy
                            d = row["delta_accuracy"]
                            b_ = row["beneficial_rate_all"]
                            h_ = row["harmful_rate_all"]
                            if d is not None and b_ is not None and h_ is not None:
                                checks["max_abs_delta_identity_point"] = max(
                                    checks["max_abs_delta_identity_point"],
                                    abs(d - (b_ - h_)),
                                )
                            panel_rows.append(row)
                    checks["max_abs_delta_identity_draws"] = max(
                        checks["max_abs_delta_identity_draws"], res_pack["max_delta_draw_diff"]
                    )

    elapsed = time.time() - t0
    receivers = pd.concat(receiver_frames, ignore_index=True) if receiver_frames else pd.DataFrame()
    print(
        f"[{label}] panels={len([c for c in panel_counts.values() if c])} "
        f"rows={len(panel_rows)} receivers={len(receivers)} ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        "panels": panel_rows,
        "receivers": receivers,
        "validation": {
            "model": label,
            "key": MODEL_KEYS[label],
            "panel_row_counts": panel_counts,
            "missing_panels": missing_panels,
            "checks": checks,
            "self_control_drift": self_drift,
            "elapsed_seconds": elapsed,
        },
    }


def _bootstrap_units(comps_sub, roles_sub, per_metric, mask, checks) -> dict:
    """One shared-draw pass over every endpoint of one (panel, sample_set)."""
    clusters = sorted(set(comps_sub.tolist()))
    boot = ClusterBootstrap(clusters)
    cl_index = np.array([boot.index[c] for c in comps_sub], dtype=np.int64)
    n_sub = len(comps_sub)

    is_clean_sub = roles_sub == "clean"
    is_corrupt_sub = roles_sub == "corrupt"

    num = np.zeros((n_sub, E_TOTAL))
    den = np.zeros((n_sub, E_TOTAL))

    for group in GROUPS:
        if group == "all":
            gi = np.ones(n_sub)
        elif group == "clean":
            gi = is_clean_sub.astype(float)
        else:
            gi = is_corrupt_sub.astype(float)
        for metric in METRICS:
            pmm = per_metric[metric]
            cc = pmm["cc"][mask].astype(float)
            cw = pmm["cw"][mask].astype(float)
            wc = pmm["wc"][mask].astype(float)
            ww = pmm["ww"][mask].astype(float)
            ti = pmm["to_inc"][mask].astype(float)
            tc = pmm["to_cor"][mask].astype(float)

            spec = {
                "flip_rate_all": (cw + wc, np.ones(n_sub)),
                "harmful_rate_all": (cw, np.ones(n_sub)),
                "beneficial_rate_all": (wc, np.ones(n_sub)),
                "harmful_rate_cond": (cw, cc + cw),
                "beneficial_rate_cond": (wc, wc + ww),
                "baseline_accuracy": (cc + cw, np.ones(n_sub)),
                "patched_accuracy": (cc + wc, np.ones(n_sub)),
                "delta_accuracy": (wc - cw, np.ones(n_sub)),
                "to_incorrect_rate": (ti, np.ones(n_sub)),
                "to_correct_rate": (tc, np.ones(n_sub)),
            }
            for ep, (nu, de) in spec.items():
                j = COLIDX[(group, metric, ep)]
                num[:, j] = gi * nu
                den[:, j] = gi * de

    res = boot.panel_stats(cl_index, num, den, keep_draws=True)
    rd = res["ratio_draws"]

    max_diff = 0.0
    for group in GROUPS:
        for metric in METRICS:
            jd = COLIDX[(group, metric, "delta_accuracy")]
            jb = COLIDX[(group, metric, "beneficial_rate_all")]
            jh = COLIDX[(group, metric, "harmful_rate_all")]
            diff = np.abs(rd[:, jd] - (rd[:, jb] - rd[:, jh]))
            finite = np.isfinite(diff)
            if finite.any():
                max_diff = max(max_diff, float(diff[finite].max()))
    if max_diff > 1e-12:
        raise AssertionError(f"delta_accuracy draw identity violated: {max_diff}")

    return {
        "stats": res["ratios"],
        "draws_signature": res["draws_signature"],
        "K": res["K"],
        "max_delta_draw_diff": max_diff,
    }


def main() -> None:
    FLIP_DIR.mkdir(parents=True, exist_ok=True)
    started = utcnow()
    t0 = time.time()
    with mp.Pool(9) as pool:
        results = pool.map(analyze_one, MODELS, chunksize=1)

    order = {m: i for i, m in enumerate(MODELS)}
    panels = pd.DataFrame([r for res in results for r in res["panels"]])
    panels["_o"] = panels["model"].map(order)
    panels["_w"] = panels["window"].map({w: i for i, w in enumerate(WINDOWS)})
    panels["_a"] = panels["arm"].map({a: i for i, a in enumerate(ARMS)})
    panels["_c"] = panels["control"].map({c: i for i, c in enumerate(CONTROLS)})
    panels["_s"] = panels["sample_set"].map({s: i for i, s in enumerate(SAMPLE_SETS)})
    panels["_g"] = panels["receiver_group"].map({g: i for i, g in enumerate(GROUPS)})
    panels["_m"] = panels["metric"].map({m: i for i, m in enumerate(METRICS)})
    panels = (
        panels.sort_values(["_o", "_w", "_a", "_c", "_s", "_g", "_m"], kind="stable")
        .drop(columns=["_o", "_w", "_a", "_c", "_s", "_g", "_m"])
        .reset_index(drop=True)
    )

    receivers = pd.concat([res["receivers"] for res in results], ignore_index=True)
    receivers["_o"] = receivers["model"].map(order)
    receivers["_w"] = receivers["window"].map({w: i for i, w in enumerate(WINDOWS)})
    receivers["_a"] = receivers["arm"].map({a: i for i, a in enumerate(ARMS)})
    receivers["_c"] = receivers["control"].map({c: i for i, c in enumerate(CONTROLS)})
    receivers["_m"] = receivers["metric"].map({m: i for i, m in enumerate(METRICS)})
    receivers = (
        receivers.sort_values(
            ["_o", "_w", "_a", "_c", "_m", "pair_id", "receiver_role"], kind="stable"
        )
        .drop(columns=["_o", "_w", "_a", "_c", "_m"])
        .reset_index(drop=True)
    )

    write_csv(panels, FLIP_DIR / "flip_panels_v1.csv")
    write_csv(receivers, FLIP_DIR / "flip_receivers_v1.csv.gz")

    notes = {
        "verdict_rule": "M > 0 -> INCORRECT, M <= 0 -> CORRECT (candidate scoring, not free generation)",
        "margin": "M = score(INCORRECT) - score(CORRECT); metric=mean is PRIMARY, metric=total is a separate SENSITIVITY analysis and is never pooled with it",
        "transition_classes": "cc/cw/wc/ww are baseline->patched correctness transitions; cw is a harmful flip, wc a beneficial flip",
        "direction_counts": "n_to_incorrect / n_to_correct are label-free verdict-direction counts, not correctness transitions",
        "sample_sets": "common_pairs (PRIMARY) keeps only pairs whose both roles have a patched row in the panel, matching the eligibility of the archived pair-level T; all_receivers (SECONDARY) keeps every patched receiver. They are separate samples and must never be merged or averaged.",
        "receiver_groups": "all / clean / corrupt share the same bootstrap draws within a (panel, sample_set)",
        "conditional_rates": "harmful_rate_cond has the baseline-CORRECT receivers as denominator, beneficial_rate_cond the baseline-WRONG ones; the denominators differ and the two rates must never be compared as if they shared one",
        "delta_accuracy": "computed as its own endpoint (wc - cw)/n and checked draw by draw against beneficial_rate_all - harmful_rate_all",
        "baseline_join": "baseline margins and verdicts are joined from the unpatched rows by (pair_id, receiver_role); patched rows carry no total-metric baseline",
        "interval": "cluster bootstrap over receiver source components, 50000 draws, PCG64(20260908), two-sided pointwise percentile; no multiplicity adjustment across the 64 panels per model",
        "null_points": "a point estimate or CI bound of null means the denominator summed to zero; it is never reported as 0",
        "not_a_benchmark": "within-study assay numbers on the paired PRMBench preview subset, not official PRMBench benchmark scores",
    }

    panels_json = {
        "version": "flip_panels_v1",
        "scope": SCOPE,
        "notes": notes,
        "draws": DRAWS,
        "seed": SEED,
        "started_utc": started,
        "finished_utc": utcnow(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_order": MODELS,
        "windows": WINDOWS,
        "arms": ARMS,
        "controls": CONTROLS,
        "sample_sets": SAMPLE_SETS,
        "receiver_groups": GROUPS,
        "endpoints": ENDPOINTS,
        "panels": [
            {k: (None if (v is None or (isinstance(v, float) and not np.isfinite(v)) or v is pd.NA) else v)
             for k, v in rec.items()}
            for rec in panels.to_dict(orient="records")
        ],
    }
    write_json(panels_json, FLIP_DIR / "flip_panels_v1.json")

    # ---- validation ---------------------------------------------------------------
    per_model = {res["validation"]["model"]: res["validation"] for res in results}
    all_self_flips = []
    self_drift_summary = {}
    for label in MODELS:
        v = per_model[label]
        self_drift_summary[label] = {
            m: {
                "max_abs_diff": v["self_control_drift"][m]["max_abs_diff"],
                "n_verdict_flips": len(v["self_control_drift"][m]["flips"]),
            }
            for m in METRICS
        }
        for m in METRICS:
            all_self_flips.extend(v["self_control_drift"][m]["flips"])

    missing = [p for label in MODELS for p in per_model[label]["missing_panels"]]
    mismatches = [
        x for label in MODELS for x in per_model[label]["checks"]["transition_sum_mismatches"]
    ]
    dup = [x for label in MODELS for x in per_model[label]["checks"]["duplicate_unit_keys"]]

    validation = {
        "version": "validation_flips_v1",
        "scope": SCOPE,
        "generated_utc": utcnow(),
        "n_panel_rows": int(len(panels)),
        "n_receiver_rows": int(len(receivers)),
        "transition_sum_equals_n": {
            "all_panels_pass": len(mismatches) == 0,
            "n_checks": int(len(panels)),
            "mismatches": mismatches,
        },
        "delta_accuracy_identity": {
            "max_abs_diff_point": max(
                per_model[l]["checks"]["max_abs_delta_identity_point"] for l in MODELS
            ),
            "max_abs_diff_draws": max(
                per_model[l]["checks"]["max_abs_delta_identity_draws"] for l in MODELS
            ),
            "tolerance": 1e-12,
        },
        "duplicate_unit_keys": dup,
        "baseline_join_misses": {
            l: per_model[l]["checks"]["baseline_join_misses"] for l in MODELS
        },
        "self_control_drift": {
            "per_model": self_drift_summary,
            "n_verdict_flips_total": len(all_self_flips),
            "verdict_flips": sorted(
                all_self_flips,
                key=lambda d: (d["model"], d["window"], d["arm"], d["metric"], d["pair_id"], d["receiver_role"]),
            ),
            "note": "self is the identity control (donor = the receiver itself); any non-zero drift or verdict flip is reported as found, never removed",
        },
        "missing_panels": missing,
        "panel_row_counts": {l: per_model[l]["panel_row_counts"] for l in MODELS},
        "undefined_draws_total": int(
            sum(per_model[l]["checks"]["total_undefined_draws"] for l in MODELS)
        ),
        "undefined_draws_by_model": {
            l: per_model[l]["checks"]["total_undefined_draws"] for l in MODELS
        },
        "n_null_points": {
            ep: int(panels[ep].isna().sum()) for ep in ENDPOINTS
        },
        "bootstrap": {
            "draws": DRAWS,
            "seed": SEED,
            "batch": 256,
            "n_panel_stats_calls": {
                l: per_model[l]["checks"]["n_panel_stats_calls"] for l in MODELS
            },
            "n_sample_sets_sharing_one_pass": {
                l: per_model[l]["checks"]["n_sample_sets_shared"] for l in MODELS
            },
            "draws_signatures": sorted(set(panels["draws_signature"].tolist())),
        },
        "timing_seconds": {
            **{l: per_model[l]["elapsed_seconds"] for l in MODELS},
            "wall_clock_total": time.time() - t0,
        },
        "outputs": {},
    }

    for name in ["flip_panels_v1.csv", "flip_panels_v1.json", "flip_receivers_v1.csv.gz"]:
        p = FLIP_DIR / name
        from common import sha256_file

        validation["outputs"][name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    write_json(validation, FLIP_DIR / "validation_flips.json")
    print(f"total wall clock {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
