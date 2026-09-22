"""Stage B2 -- wide-D supplement: baseline, ID match, verdict flips, donor-role
transitions, D reproduction and identity drift on the archived campaign_20260908_D runs.

Inputs (all read-only):
  OUT/extracted/wide_d/<key>.csv.gz + manifest.json        (stage S2 atomic extraction)
  OUT/baseline/baseline_receivers_v1.csv                   (stage B1a window-run baseline)
  ROOT/docs/results/wide_d_synthesis_20260908/source_reports/reporting/*.json
                                                           (stored D reproduction targets)

Definitions used here (fixed by the task contract; see common.py):
  M = score(INCORRECT) - score(CORRECT); metric "mean" is PRIMARY, "total" is a
  separate SENSITIVITY analysis and the two are never pooled.
  Verdict: M > 0 -> INCORRECT, M <= 0 -> CORRECT. Truth: clean -> CORRECT,
  corrupt (error-bearing) -> INCORRECT.

  D_{i,r} = m(r <- error donor) - m(r <- clean donor)
  D_clean  = mean over clean receivers, D_error = mean over error receivers,
  D_common_i = (D_clean_i + D_error_i) / 2.
  No D/T, no T-D, no normalized D is computed anywhere in this stage.

Two DIFFERENT comparisons are reported and are named explicitly everywhere:
  "baseline_to_patched"        unpatched verdict  ->  patched verdict (per donor_role cell)
  "clean_donor_to_error_donor" verdict under the clean donor -> verdict under the
                               error donor, receiver held fixed (no unpatched row involved)

All intervals are the shared-draw cluster bootstrap over receiver source components
(50000 draws, PCG64(20260908), batches of 256) from common.py. Donor assignments are
fixed and never resampled. Every endpoint of a model x metric is computed in ONE pass
over the same draws, so the panels are comparable draw by draw.

Run: .venv/bin/python <this file>
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
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
    ROOT,
    ROLE_TRUTH,
    SEED,
    ClusterBootstrap,
    sha256_file,
    write_csv,
    write_json,
)

WD_DIR = OUT / "wide_d"
EXTRACT_DIR = OUT / "extracted" / "wide_d"
REPORT_DIR = (
    OUT / "references/wide_d"
)
WINDOW_BASELINE = OUT / "baseline" / "baseline_receivers_v1.csv"

WD_SCOPE = (
    "baseline on the PRMBench paired assay used in this study; wide-D run unpatched "
    "scores; candidate-scoring verdict"
)
FLIP_SCOPE = (
    "wide-D run (campaign_20260908_D, arm=all_router, window=full_step, alpha=1.0) on "
    "the PRMBench paired assay used in this study; candidate-scoring verdict"
)

ID_MATCH_TOL = 1e-9

# baseline endpoint order, identical to OUT/baseline/analyze_baseline.py
BASELINE_ENDPOINTS = [
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

# stage-B1b endpoint vocabulary for the baseline_to_patched comparison.
# (stage B1b had not written its tables when this stage ran; the names and definitions
#  below are implemented locally from the task contract -- see notes_for_downstream.)
FLIP_ENDPOINTS = [
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

TRANSITION_ENDPOINTS = [
    "any_change",
    "to_incorrect",
    "to_correct",
    "accuracy_under_clean_donor",
    "accuracy_under_error_donor",
]

D_ESTIMANDS = ["D_clean", "D_error", "D_common"]
RECEIVER_GROUPS = ["all", "clean", "corrupt"]
DONOR_ROLES = ["clean", "corrupt"]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON records with NaN/NaT rendered as null (strict JSON, no NaN token).

    A null here is an UNDEFINED point estimate (zero denominator), never a zero.
    """
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v is None:
                clean[k] = None
            elif isinstance(v, float) and not np.isfinite(v):
                clean[k] = None
            elif v is pd.NaT or (v is not None and v is pd.NA):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _stat_columns(prefix: str, stat: dict) -> dict:
    return {
        prefix: stat["point"],
        f"{prefix}_ci_low": stat["ci_low"],
        f"{prefix}_ci_high": stat["ci_high"],
        f"{prefix}_n_defined_draws": stat["n_defined_draws"],
        f"{prefix}_n_undefined_draws": stat["n_undefined_draws"],
    }


def _verdict(m: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(m, dtype=float) > 0, "INCORRECT", "CORRECT")


# --------------------------------------------------------------------------- per model


def analyze_one(label: str) -> dict:
    t0 = time.time()
    key = MODEL_KEYS[label]
    df = pd.read_csv(EXTRACT_DIR / f"{key}.csv.gz")

    # ---------------------------------------------------------------- unit index
    unp = (
        df[df["treatment"] == "unpatched"]
        .sort_values(["pair_id", "receiver_role"], kind="stable")
        .reset_index(drop=True)
    )
    n_units = len(unp)
    pair_ids = unp["pair_id"].to_numpy()
    roles = unp["receiver_role"].to_numpy()
    comps = unp["source_component"].to_numpy()
    is_corrupt = roles == "corrupt"
    is_clean = ~is_corrupt
    truth = np.array([ROLE_TRUTH[r] for r in roles])
    unit_of = {(pair_ids[i], roles[i]): i for i in range(n_units)}

    clusters = sorted(set(comps.tolist()))
    boot = ClusterBootstrap(clusters)
    cl_index = np.array([boot.index[c] for c in comps], dtype=np.int64)

    by_pair: dict = {}
    for i in range(n_units):
        by_pair.setdefault(pair_ids[i], {})[roles[i]] = i
    complete_pairs = sorted(p for p, d in by_pair.items() if set(d) == {"clean", "corrupt"})
    incomplete_pairs = sorted(p for p, d in by_pair.items() if set(d) != {"clean", "corrupt"})

    # ---------------------------------------------------------------- margin arrays
    def gather(sub: pd.DataFrame, metric: str) -> np.ndarray:
        out = np.full(n_units, np.nan)
        pid = sub["pair_id"].to_numpy()
        rr = sub["receiver_role"].to_numpy()
        vals = sub[f"margin_{metric}"].to_numpy(float)
        for j in range(len(sub)):
            out[unit_of[(pid[j], rr[j])]] = vals[j]
        return out

    don = df[df["treatment"] == "donor_role"]
    selfp = df[df["treatment"] == "self_patch"]
    rerun = df[df["treatment"] == "unpatched_rerun"]

    M_unp, M_don, M_self, M_rerun = {}, {}, {}, {}
    for metric in METRICS:
        M_unp[metric] = gather(unp, metric)
        M_don[metric] = {
            dr: gather(don[don["donor_role"] == dr], metric) for dr in DONOR_ROLES
        }
        M_self[metric] = gather(selfp, metric)
        M_rerun[metric] = gather(rerun, metric)

    missing = {
        "donor_role_clean": int(np.isnan(M_don["mean"]["clean"]).sum()),
        "donor_role_corrupt": int(np.isnan(M_don["mean"]["corrupt"]).sum()),
        "self_patch": int(np.isnan(M_self["mean"]).sum()),
        "unpatched_rerun": int(np.isnan(M_rerun["mean"]).sum()),
    }
    if any(missing.values()):
        raise RuntimeError(f"[{label}] missing patched/identity partners: {missing}")

    token_counts = {
        "CORRECT": sorted({int(v) for v in df["correct_token_count"].dropna().unique()}),
        "INCORRECT": sorted({int(v) for v in df["incorrect_token_count"].dropna().unique()}),
    }
    token_counts["constant_within_model"] = (
        len(token_counts["CORRECT"]) == 1 and len(token_counts["INCORRECT"]) == 1
    )

    # ---------------------------------------------------------------- ID match (b)
    win = pd.read_csv(WINDOW_BASELINE)
    win = win[win["model"] == label]
    win_idx = {
        (r.pair_id, r.receiver_role): (r.margin_mean, r.margin_total, r.component)
        for r in win.itertuples()
    }
    id_rows = []
    id_summary = {}
    matched = 0
    comp_mismatch = 0
    for i in range(n_units):
        k = (pair_ids[i], roles[i])
        w = win_idx.get(k)
        row = {
            "model": label,
            "key": key,
            "pair_id": pair_ids[i],
            "component": comps[i],
            "receiver_role": roles[i],
            "matched_window_baseline": w is not None,
            "wide_d_margin_mean": M_unp["mean"][i],
            "wide_d_margin_total": M_unp["total"][i],
            "window_margin_mean": None if w is None else float(w[0]),
            "window_margin_total": None if w is None else float(w[1]),
            "component_matches": None if w is None else bool(w[2] == comps[i]),
        }
        if w is not None:
            matched += 1
            if w[2] != comps[i]:
                comp_mismatch += 1
            row["abs_diff_mean"] = abs(M_unp["mean"][i] - float(w[0]))
            row["abs_diff_total"] = abs(M_unp["total"][i] - float(w[1]))
            row["verdict_wide_d_mean"] = _verdict(M_unp["mean"][i : i + 1])[0]
            row["verdict_window_mean"] = _verdict(np.array([w[0]]))[0]
            row["verdict_wide_d_total"] = _verdict(M_unp["total"][i : i + 1])[0]
            row["verdict_window_total"] = _verdict(np.array([w[1]]))[0]
            row["verdict_disagree_mean"] = bool(
                row["verdict_wide_d_mean"] != row["verdict_window_mean"]
            )
            row["verdict_disagree_total"] = bool(
                row["verdict_wide_d_total"] != row["verdict_window_total"]
            )
        else:
            for c in (
                "abs_diff_mean",
                "abs_diff_total",
                "verdict_wide_d_mean",
                "verdict_window_mean",
                "verdict_wide_d_total",
                "verdict_window_total",
                "verdict_disagree_mean",
                "verdict_disagree_total",
            ):
                row[c] = None
        id_rows.append(row)

    id_df = pd.DataFrame(id_rows)
    ok = id_df["matched_window_baseline"].to_numpy(bool)
    for metric in METRICS:
        ad = id_df.loc[ok, f"abs_diff_{metric}"].to_numpy(float)
        sd = (
            id_df.loc[ok, f"wide_d_margin_{metric}"].to_numpy(float)
            - id_df.loc[ok, f"window_margin_{metric}"].to_numpy(float)
        )
        dis = id_df.loc[ok, f"verdict_disagree_{metric}"].to_numpy(bool)
        nz = ad[ad > ID_MATCH_TOL]
        wd_v = _verdict(id_df.loc[ok, f"wide_d_margin_{metric}"].to_numpy(float))
        wn_v = _verdict(id_df.loc[ok, f"window_margin_{metric}"].to_numpy(float))
        truth_ok = np.array([ROLE_TRUTH[r] for r in id_df.loc[ok, "receiver_role"]])
        id_summary[metric] = {
            "n_matched": int(matched),
            "n_unmatched": int(n_units - matched),
            "max_abs_diff": float(ad.max()) if ad.size else None,
            "n_identical_within_1e-9": int((ad <= ID_MATCH_TOL).sum()),
            "n_bit_identical": int((ad == 0.0).sum()),
            "n_abs_diff_gt_1e-9": int((ad > ID_MATCH_TOL).sum()),
            "n_abs_diff_gt_0.01": int((ad > 0.01).sum()),
            "n_abs_diff_gt_0.1": int((ad > 0.1).sum()),
            "signed_mean_diff_wide_d_minus_window": float(sd.mean()) if sd.size else None,
            "nonzero_abs_diff_p50": float(np.quantile(nz, 0.5)) if nz.size else None,
            "nonzero_abs_diff_p90": float(np.quantile(nz, 0.9)) if nz.size else None,
            "nonzero_abs_diff_p99": float(np.quantile(nz, 0.99)) if nz.size else None,
            "window_accuracy": float((wn_v == truth_ok).mean()) if ad.size else None,
            "wide_d_accuracy": float((wd_v == truth_ok).mean()) if ad.size else None,
            "accuracy_delta_wide_d_minus_window": float(
                (wd_v == truth_ok).mean() - (wn_v == truth_ok).mean()
            )
            if ad.size
            else None,
            "n_verdict_disagreements": int(dis.sum()),
            "verdict_disagreements": [
                {
                    "pair_id": id_df.loc[j, "pair_id"],
                    "receiver_role": id_df.loc[j, "receiver_role"],
                    "wide_d_margin": float(id_df.loc[j, f"wide_d_margin_{metric}"]),
                    "window_margin": float(id_df.loc[j, f"window_margin_{metric}"]),
                }
                for j in id_df.index[ok & id_df[f"verdict_disagree_{metric}"].fillna(False).to_numpy(bool)]
            ],
        }
        id_summary[metric]["identical_within_1e-9"] = bool(
            matched == n_units
            and id_summary[metric]["n_identical_within_1e-9"] == matched
            and id_summary[metric]["n_verdict_disagreements"] == 0
        )
    id_verdict = (
        "identical_within_1e-9"
        if id_summary["mean"]["identical_within_1e-9"]
        and id_summary["total"]["identical_within_1e-9"]
        else "wide_d_run_has_its_own_baseline_report_separately"
    )
    id_summary["n_component_mismatches"] = int(comp_mismatch)
    id_summary["conclusion"] = id_verdict

    # ---------------------------------------------------------------- stored D targets
    rep_path = REPORT_DIR / f"{key}_full_v1.json"
    rep = json.loads(rep_path.read_text())
    stored = {
        "mean": rep["primary_mean_logprob"],
        "total": rep["sensitivity_total_logprob"],
    }
    stored_key_path = {
        "mean": f"{rep_path.name}:primary_mean_logprob.<estimand>.{{mean,ci_low,ci_high}}",
        "total": f"{rep_path.name}:sensitivity_total_logprob.<estimand>.{{mean,ci_low,ci_high}}",
    }

    # ---------------------------------------------------------------- panels
    baseline_rows, flip_rows, trans_rows, d_rows = [], [], [], []
    flip_receiver_rows, trans_receiver_rows, d_pair_rows = [], [], []
    validation_metrics = {}

    for metric in METRICS:
        Mb = M_unp[metric]
        vb = _verdict(Mb)
        cb = vb == truth  # baseline correct
        Mc = M_don[metric]["clean"]  # patched with the CLEAN donor
        Me = M_don[metric]["corrupt"]  # patched with the ERROR donor
        vc, ve = _verdict(Mc), _verdict(Me)
        cc_ok, ce_ok = vc == truth, ve == truth

        specs: list[tuple[str, np.ndarray, np.ndarray]] = []

        # ---- (a) wide-D baseline, identical endpoint set to baseline_panels_v1
        clean_ind = is_clean.astype(float)
        corrupt_ind = is_corrupt.astype(float)
        pred_incorrect = (vb == "INCORRECT").astype(float)
        pair_den = np.zeros(n_units)
        pair_g = np.zeros(n_units)
        pair_g_pos = np.zeros(n_units)
        pair_both = np.zeros(n_units)
        for p in complete_pairs:
            ic, ie = by_pair[p]["clean"], by_pair[p]["corrupt"]
            g = Mb[ie] - Mb[ic]
            pair_den[ie] = 1.0
            pair_g[ie] = g
            pair_g_pos[ie] = 1.0 if g > 0 else 0.0
            pair_both[ie] = 1.0 if (cb[ic] and cb[ie]) else 0.0
        ones = np.ones(n_units)
        specs += [
            ("baseline|accuracy", cb.astype(float), ones),
            ("baseline|error_recall", corrupt_ind * pred_incorrect, corrupt_ind),
            ("baseline|clean_false_positive_rate", clean_ind * pred_incorrect, clean_ind),
            ("baseline|clean_accuracy", clean_ind * (1.0 - pred_incorrect), clean_ind),
            ("baseline|mean_margin_clean", clean_ind * Mb, clean_ind),
            ("baseline|mean_margin_corrupt", corrupt_ind * Mb, corrupt_ind),
            ("baseline|mean_gap", pair_g, pair_den),
            ("baseline|fraction_pairs_gap_positive", pair_g_pos, pair_den),
            ("baseline|fraction_pairs_both_correct", pair_both, pair_den),
        ]

        # ---- (c) baseline_to_patched flips, 4 cells + 2 combined panels
        group_mask = {
            "all": ones,
            "clean": clean_ind,
            "corrupt": corrupt_ind,
        }
        flip_cells = {}
        for grp in RECEIVER_GROUPS:
            mask = group_mask[grp]
            for dr in DONOR_ROLES:
                Mp = Mc if dr == "clean" else Me
                vp = vc if dr == "clean" else ve
                cp = cc_ok if dr == "clean" else ce_ok
                cc_i = mask * (cb & cp).astype(float)
                cw_i = mask * (cb & ~cp).astype(float)
                wc_i = mask * (~cb & cp).astype(float)
                ww_i = mask * (~cb & ~cp).astype(float)
                base_ok_i = mask * cb.astype(float)
                base_bad_i = mask * (~cb).astype(float)
                to_inc_i = mask * ((vb == "CORRECT") & (vp == "INCORRECT")).astype(float)
                to_cor_i = mask * ((vb == "INCORRECT") & (vp == "CORRECT")).astype(float)
                pre = f"flip|{grp}|{dr}|"
                specs += [
                    (pre + "flip_rate_all", cw_i + wc_i, mask),
                    (pre + "harmful_rate_all", cw_i, mask),
                    (pre + "beneficial_rate_all", wc_i, mask),
                    (pre + "harmful_rate_cond", cw_i, base_ok_i),
                    (pre + "beneficial_rate_cond", wc_i, base_bad_i),
                    (pre + "baseline_accuracy", base_ok_i, mask),
                    (pre + "patched_accuracy", mask * cp.astype(float), mask),
                    (
                        pre + "delta_accuracy",
                        mask * (cp.astype(float) - cb.astype(float)),
                        mask,
                    ),
                    (pre + "to_incorrect_rate", to_inc_i, mask),
                    (pre + "to_correct_rate", to_cor_i, mask),
                ]
                flip_cells[(grp, dr)] = {
                    "cc": int(cc_i.sum()),
                    "cw": int(cw_i.sum()),
                    "wc": int(wc_i.sum()),
                    "ww": int(ww_i.sum()),
                    "n_units": int(mask.sum()),
                }

        # ---- (d) clean_donor_to_error_donor transitions
        trans_counts = {}
        for grp in RECEIVER_GROUPS:
            mask = group_mask[grp]
            cc_t = mask * ((vc == "CORRECT") & (ve == "CORRECT")).astype(float)
            ci_t = mask * ((vc == "CORRECT") & (ve == "INCORRECT")).astype(float)
            ic_t = mask * ((vc == "INCORRECT") & (ve == "CORRECT")).astype(float)
            ii_t = mask * ((vc == "INCORRECT") & (ve == "INCORRECT")).astype(float)
            pre = f"trans|{grp}|"
            specs += [
                (pre + "any_change", ci_t + ic_t, mask),
                (pre + "to_incorrect", ci_t, mask),
                (pre + "to_correct", ic_t, mask),
                (pre + "accuracy_under_clean_donor", mask * cc_ok.astype(float), mask),
                (pre + "accuracy_under_error_donor", mask * ce_ok.astype(float), mask),
            ]
            trans_counts[grp] = {
                "CORRECT_to_CORRECT": int(cc_t.sum()),
                "CORRECT_to_INCORRECT": int(ci_t.sum()),
                "INCORRECT_to_CORRECT": int(ic_t.sum()),
                "INCORRECT_to_INCORRECT": int(ii_t.sum()),
                "n_units": int(mask.sum()),
            }

        # ---- (e) D reproduction
        D_unit = Me - Mc  # per receiver: error donor minus clean donor
        d_common_carrier = np.zeros(n_units)
        d_common_den = np.zeros(n_units)
        for p in complete_pairs:
            ic, ie = by_pair[p]["clean"], by_pair[p]["corrupt"]
            d_common_carrier[ie] = (D_unit[ic] + D_unit[ie]) / 2.0
            d_common_den[ie] = 1.0
        specs += [
            ("d|D_clean", clean_ind * D_unit, clean_ind),
            ("d|D_error", corrupt_ind * D_unit, corrupt_ind),
            ("d|D_common", d_common_carrier, d_common_den),
        ]

        # ---- one pass over the shared draws
        names = [s[0] for s in specs]
        num = np.column_stack([s[1] for s in specs])
        den = np.column_stack([s[2] for s in specs])
        res = boot.panel_stats(
            cl_index,
            num,
            den,
            auroc_specs=[{"name": "auroc", "scores": Mb, "is_positive": is_corrupt}],
            keep_draws=True,
        )
        stats = dict(zip(names, res["ratios"]))
        auroc = res["auroc"][0]
        sig = res["draws_signature"]

        # balanced accuracy from the SAME draws (equals accuracy under 1:1 balance)
        rd = res["ratio_draws"]
        rec_col = rd[:, names.index("baseline|error_recall")]
        fpr_col = rd[:, names.index("baseline|clean_false_positive_rate")]
        bal_draws = (rec_col + 1.0 - fpr_col) / 2.0
        r_pt = stats["baseline|error_recall"]["point"]
        f_pt = stats["baseline|clean_false_positive_rate"]["point"]
        bal_point = None if (r_pt is None or f_pt is None) else (r_pt + 1.0 - f_pt) / 2.0
        bal_stat = ClusterBootstrap._summarize(bal_point, bal_draws, np.isfinite(bal_draws))

        confusion = {
            "clean_to_CORRECT": int((is_clean & (vb == "CORRECT")).sum()),
            "clean_to_INCORRECT": int((is_clean & (vb == "INCORRECT")).sum()),
            "corrupt_to_CORRECT": int((is_corrupt & (vb == "CORRECT")).sum()),
            "corrupt_to_INCORRECT": int((is_corrupt & (vb == "INCORRECT")).sum()),
        }
        confusion["total"] = sum(confusion.values())

        # ---- (a) baseline row
        brow = {
            "model": label,
            "key": key,
            "anchor_layer": ANCHORS[label],
            "metric": metric,
            "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
            "scope": WD_SCOPE,
            "run": "wide_d",
            "n_receivers": n_units,
            "n_pairs": len(complete_pairs),
            "K": len(clusters),
            "n_clean": int(is_clean.sum()),
            "n_corrupt": int(is_corrupt.sum()),
            "draws": DRAWS,
            "seed": SEED,
            "draws_signature": sig,
        }
        for name in BASELINE_ENDPOINTS:
            brow.update(_stat_columns(name, stats[f"baseline|{name}"]))
        brow.update(_stat_columns("balanced_accuracy", bal_stat))
        brow.update(_stat_columns("auroc", auroc))
        brow.update({f"confusion_{k}": v for k, v in confusion.items()})
        brow["correct_token_count"] = (
            token_counts["CORRECT"][0] if len(token_counts["CORRECT"]) == 1 else None
        )
        brow["incorrect_token_count"] = (
            token_counts["INCORRECT"][0] if len(token_counts["INCORRECT"]) == 1 else None
        )
        brow["token_counts_constant"] = token_counts["constant_within_model"]
        baseline_rows.append(brow)

        # ---- (c) flip rows
        identity_max = 0.0
        for grp in RECEIVER_GROUPS:
            for dr in DONOR_ROLES:
                c = flip_cells[(grp, dr)]
                row = {
                    "model": label,
                    "key": key,
                    "metric": metric,
                    "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
                    "scope": FLIP_SCOPE,
                    "comparison": "baseline_to_patched",
                    "receiver_group": grp,
                    "donor_role": dr,
                    "donor_role_prose": "clean donor" if dr == "clean" else "error donor",
                    "n_units": c["n_units"],
                    "n_pairs": len(complete_pairs),
                    "K": len(clusters),
                    "draws": DRAWS,
                    "seed": SEED,
                    "draws_signature": sig,
                    "cc": c["cc"],
                    "cw": c["cw"],
                    "wc": c["wc"],
                    "ww": c["ww"],
                }
                for name in FLIP_ENDPOINTS:
                    row.update(_stat_columns(name, stats[f"flip|{grp}|{dr}|{name}"]))
                # identity: delta_accuracy == patched_accuracy - baseline_accuracy
                da = row["delta_accuracy"]
                ident = abs(da - (row["patched_accuracy"] - row["baseline_accuracy"]))
                identity_max = max(identity_max, ident)
                row["delta_accuracy_identity_abs_error"] = ident
                row["delta_accuracy_identity_holds"] = bool(ident <= 1e-12)
                assert ident <= 1e-12, f"delta_accuracy identity failed: {label} {metric} {grp} {dr}"
                flip_rows.append(row)

        # ---- (d) transition rows
        for grp in RECEIVER_GROUPS:
            c = trans_counts[grp]
            row = {
                "model": label,
                "key": key,
                "metric": metric,
                "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
                "scope": FLIP_SCOPE,
                "comparison": "clean_donor_to_error_donor",
                "receiver_group": grp,
                "n_units": c["n_units"],
                "n_pairs": len(complete_pairs),
                "K": len(clusters),
                "draws": DRAWS,
                "seed": SEED,
                "draws_signature": sig,
                "t_CORRECT_to_CORRECT": c["CORRECT_to_CORRECT"],
                "t_CORRECT_to_INCORRECT": c["CORRECT_to_INCORRECT"],
                "t_INCORRECT_to_CORRECT": c["INCORRECT_to_CORRECT"],
                "t_INCORRECT_to_INCORRECT": c["INCORRECT_to_INCORRECT"],
            }
            for name in TRANSITION_ENDPOINTS:
                row.update(_stat_columns(name, stats[f"trans|{grp}|{name}"]))
            trans_rows.append(row)

        # ---- (e) D rows
        d_max_point = 0.0
        d_max_ci = 0.0
        for est in D_ESTIMANDS:
            st = stats[f"d|{est}"]
            sv = stored[metric][est]
            row = {
                "model": label,
                "key": key,
                "metric": metric,
                "metric_role": "PRIMARY" if metric == "mean" else "SENSITIVITY",
                "scope": FLIP_SCOPE,
                "estimand": est,
                "definition": "D = m(receiver <- error donor) - m(receiver <- clean donor)",
                "n_units": int(is_clean.sum())
                if est == "D_clean"
                else (int(is_corrupt.sum()) if est == "D_error" else len(complete_pairs)),
                "K": len(clusters),
                "draws": DRAWS,
                "seed": SEED,
                "draws_signature": sig,
                "recomputed_mean": st["point"],
                "recomputed_ci_low": st["ci_low"],
                "recomputed_ci_high": st["ci_high"],
                "n_defined_draws": st["n_defined_draws"],
                "n_undefined_draws": st["n_undefined_draws"],
                "stored_mean": sv["mean"],
                "stored_ci_low": sv["ci_low"],
                "stored_ci_high": sv["ci_high"],
                "stored_key_path": stored_key_path[metric].replace("<estimand>", est),
                "abs_err_point": abs(st["point"] - sv["mean"]),
                "abs_err_ci_low": abs(st["ci_low"] - sv["ci_low"]),
                "abs_err_ci_high": abs(st["ci_high"] - sv["ci_high"]),
            }
            d_max_point = max(d_max_point, row["abs_err_point"])
            d_max_ci = max(d_max_ci, row["abs_err_ci_low"], row["abs_err_ci_high"])
            d_rows.append(row)

        # ---- per-receiver / per-pair detail tables
        for i in range(n_units):
            for dr in DONOR_ROLES:
                Mp = Mc[i] if dr == "clean" else Me[i]
                vp = vc[i] if dr == "clean" else ve[i]
                cp = bool(cc_ok[i] if dr == "clean" else ce_ok[i])
                flip_receiver_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "comparison": "baseline_to_patched",
                        "pair_id": pair_ids[i],
                        "component": comps[i],
                        "receiver_role": roles[i],
                        "donor_role": dr,
                        "truth": truth[i],
                        "baseline_margin": Mb[i],
                        "baseline_verdict": vb[i],
                        "baseline_correct": bool(cb[i]),
                        "patched_margin": Mp,
                        "patched_verdict": vp,
                        "patched_correct": cp,
                        "cell": ("c" if cb[i] else "w") + ("c" if cp else "w"),
                        "verdict_transition": f"{vb[i]}_to_{vp}",
                    }
                )
            trans_receiver_rows.append(
                {
                    "model": label,
                    "metric": metric,
                    "comparison": "clean_donor_to_error_donor",
                    "pair_id": pair_ids[i],
                    "component": comps[i],
                    "receiver_role": roles[i],
                    "truth": truth[i],
                    "margin_clean_donor": Mc[i],
                    "margin_error_donor": Me[i],
                    "verdict_clean_donor": vc[i],
                    "verdict_error_donor": ve[i],
                    "correct_under_clean_donor": bool(cc_ok[i]),
                    "correct_under_error_donor": bool(ce_ok[i]),
                    "transition": f"{vc[i]}_to_{ve[i]}",
                    "D": D_unit[i],
                }
            )
        for p in complete_pairs:
            ic, ie = by_pair[p]["clean"], by_pair[p]["corrupt"]
            d_pair_rows.append(
                {
                    "model": label,
                    "metric": metric,
                    "pair_id": p,
                    "component": comps[ic],
                    "m_clean_receiver_clean_donor": Mc[ic],
                    "m_clean_receiver_error_donor": Me[ic],
                    "D_clean": D_unit[ic],
                    "m_error_receiver_clean_donor": Mc[ie],
                    "m_error_receiver_error_donor": Me[ie],
                    "D_error": D_unit[ie],
                    "D_common": (D_unit[ic] + D_unit[ie]) / 2.0,
                }
            )

        # ---- (f) identity drift
        drift = {}
        for name, arr in (("self_patch", M_self[metric]), ("unpatched_rerun", M_rerun[metric])):
            diff = arr - Mb
            va = _verdict(arr)
            flips = [
                {
                    "pair_id": pair_ids[i],
                    "receiver_role": roles[i],
                    "unpatched_margin": float(Mb[i]),
                    "identity_margin": float(arr[i]),
                }
                for i in range(n_units)
                if va[i] != vb[i]
            ]
            drift[name] = {
                "max_abs_diff": float(np.abs(diff).max()),
                "n_verdict_flips": len(flips),
                "verdict_flips": flips,
            }

        validation_metrics[metric] = {
            "confusion": confusion,
            "confusion_total_equals_n_receivers": confusion["total"] == n_units,
            "undefined_draws": {n: stats[n]["n_undefined_draws"] for n in names}
            | {"auroc": auroc["n_undefined_draws"], "balanced_accuracy": bal_stat["n_undefined_draws"]},
            "max_undefined_draws": max(
                [stats[n]["n_undefined_draws"] for n in names]
                + [auroc["n_undefined_draws"], bal_stat["n_undefined_draws"]]
            ),
            "d_reproduction_max_abs_point_error": d_max_point,
            "d_reproduction_max_abs_ci_error": d_max_ci,
            "delta_accuracy_identity_max_abs_error": identity_max,
            "identity_drift": drift,
            "draws_signature": sig,
        }

    validation = {
        "model": label,
        "key": key,
        "n_rows": int(len(df)),
        "rows_by_treatment": {
            k: int(v) for k, v in sorted(df["treatment"].value_counts().items())
        },
        "n_receivers_unpatched": n_units,
        "n_pairs_total": int(len(by_pair)),
        "n_pairs_complete": len(complete_pairs),
        "pairs_incomplete": incomplete_pairs,
        "n_components_K": len(clusters),
        "n_eligible_rows": int(df["eligible"].astype(bool).sum()),
        "n_ineligible_rows": int((~df["eligible"].astype(bool)).sum()),
        "n_technical_failed_rows": int((~df["technical_passed"].astype(bool)).sum()),
        "unique_donor_pairs": int(
            df.loc[df["treatment"] == "donor_role", "donor_pair_id"].nunique()
        ),
        "missing_patched_partners": missing,
        "token_counts": token_counts,
        "stored_report": {
            "path": str(rep_path),
            "sha256": sha256_file(rep_path),
            "N": rep["N"],
            "K": rep["K"],
            "ineligible_pairs": rep["ineligible_pairs"],
            "unique_fixed_donor_pairs": rep["unique_fixed_donor_pairs"],
            "bootstrap_indices_sha256": rep["bootstrap"].get("indices_sha256"),
            "bootstrap_seed": rep["bootstrap"].get("seed"),
            "bootstrap_n_resamples": rep["bootstrap"].get("n_resamples"),
            "bootstrap_n_clusters": rep["bootstrap"].get("n_clusters"),
            "counts_match_extraction": bool(
                rep["N"] == len(complete_pairs) and rep["K"] == len(clusters)
            ),
        },
        "id_match": id_summary,
        "metrics": validation_metrics,
        "elapsed_seconds": time.time() - t0,
    }

    print(
        f"[{label}] N={len(complete_pairs)} K={len(clusters)} "
        f"acc={baseline_rows[0]['accuracy']:.4f} auroc={baseline_rows[0]['auroc']:.4f} "
        f"idmatch={id_verdict} Dmax={validation_metrics['mean']['d_reproduction_max_abs_point_error']:.2e} "
        f"({validation['elapsed_seconds']:.1f}s)",
        flush=True,
    )

    return {
        "baseline": baseline_rows,
        "flips": flip_rows,
        "transitions": trans_rows,
        "d": d_rows,
        "flip_receivers": flip_receiver_rows,
        "trans_receivers": trans_receiver_rows,
        "d_pairs": d_pair_rows,
        "id_rows": id_rows,
        "validation": validation,
    }


# --------------------------------------------------------------------------- driver


NOTES = {
    "comparisons": {
        "baseline_to_patched": "baseline verdict from the wide-D unpatched row of the receiver vs the verdict under the named donor_role patch; one panel per receiver_group x donor_role.",
        "clean_donor_to_error_donor": "fixed receiver, verdict under the clean donor vs the verdict under the error donor; the unpatched row is NOT involved. This is a DIFFERENT comparison from baseline_to_patched and the two must never be conflated.",
    },
    "flip_endpoints": {
        "cells": "cc/cw/wc/ww = (baseline correct, patched correct) / (baseline correct, patched wrong) / (baseline wrong, patched correct) / (baseline wrong, patched wrong); 'correct' means verdict == truth for that receiver role.",
        "flip_rate_all": "(cw + wc) / n",
        "harmful_rate_all": "cw / n",
        "beneficial_rate_all": "wc / n",
        "harmful_rate_cond": "cw / (cc + cw), conditional on the baseline being correct; the denominator is recomputed inside every bootstrap draw",
        "beneficial_rate_cond": "wc / (wc + ww), conditional on the baseline being wrong; the denominator is recomputed inside every bootstrap draw",
        "baseline_accuracy": "(cc + cw) / n",
        "patched_accuracy": "(cc + wc) / n",
        "delta_accuracy": "patched_accuracy - baseline_accuracy, computed as its own ratio and asserted equal to the difference to 1e-12",
        "to_incorrect_rate": "fraction of receivers whose VERDICT moves CORRECT -> INCORRECT (verdict space, not truth space)",
        "to_correct_rate": "fraction of receivers whose VERDICT moves INCORRECT -> CORRECT",
    },
    "transition_endpoints": {
        "any_change": "(CORRECT->INCORRECT + INCORRECT->CORRECT) / n",
        "to_incorrect": "CORRECT under the clean donor -> INCORRECT under the error donor, over n",
        "to_correct": "INCORRECT under the clean donor -> CORRECT under the error donor, over n",
        "accuracy_under_clean_donor": "fraction of receivers whose verdict under the clean donor equals the truth (context endpoint)",
        "accuracy_under_error_donor": "fraction of receivers whose verdict under the error donor equals the truth (context endpoint)",
    },
    "d": {
        "definition": "D_{i,r} = m(r <- error donor) - m(r <- clean donor); D_clean over clean receivers, D_error over error receivers, D_common_i = (D_clean_i + D_error_i)/2 carried on the error receiver row so it shares draws",
        "not_computed": "no D/T ratio, no T-D difference and no normalized D is computed in this stage",
    },
    "balanced_accuracy": "with the 1:1 clean/corrupt label balance of this assay balanced accuracy equals accuracy exactly in every draw; reported for convenience, NOT independent evidence",
    "clean_accuracy": "= 1 - clean_false_positive_rate; convenience only",
    "metrics": "metric=mean (mean_logprob) is PRIMARY, metric=total (total_logprob) is a separate SENSITIVITY analysis; never pool or average the two",
    "not_a_benchmark": "within-study assay numbers on the paired PRMBench preview subset under a candidate-scoring verdict, not official PRMBench benchmark scores",
    "interval": "cluster bootstrap over receiver source components, 50000 draws, PCG64(20260908), batches of 256, two-sided pointwise percentile; donors fixed, never resampled; no multiplicity adjustment",
    "role_naming": "the archive's role string 'corrupt' is kept verbatim in data files; in prose it is the error receiver / error donor",
    "stage_b1b": "stage B1b had not written its tables when this stage ran, so the baseline_to_patched endpoint names and definitions above were implemented locally from the task contract; if B1b ships a different convention, reconcile before merging tables",
}


def main() -> None:
    WD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = utcnow()
    with mp.Pool(5) as pool:
        results = pool.map(analyze_one, MODELS, chunksize=1)

    order = {m: i for i, m in enumerate(MODELS)}

    def ordered(rows, extra=()):
        df = pd.DataFrame(rows)
        by = ["model"] + list(extra)
        return df.sort_values(
            by=by,
            key=lambda s: s.map(order) if s.name == "model" else s,
            kind="stable",
        ).reset_index(drop=True)

    baseline = ordered([r for res in results for r in res["baseline"]], ["metric"])
    flips = ordered(
        [r for res in results for r in res["flips"]],
        ["metric", "receiver_group", "donor_role"],
    )
    trans = ordered(
        [r for res in results for r in res["transitions"]], ["metric", "receiver_group"]
    )
    dtab = ordered([r for res in results for r in res["d"]], ["metric", "estimand"])
    flip_rx = ordered(
        [r for res in results for r in res["flip_receivers"]],
        ["metric", "pair_id", "receiver_role", "donor_role"],
    )
    trans_rx = ordered(
        [r for res in results for r in res["trans_receivers"]],
        ["metric", "pair_id", "receiver_role"],
    )
    d_pairs = ordered([r for res in results for r in res["d_pairs"]], ["metric", "pair_id"])
    id_rx = ordered([r for res in results for r in res["id_rows"]], ["pair_id", "receiver_role"])

    write_csv(baseline, WD_DIR / "wide_d_baseline_v1.csv")
    write_csv(flips, WD_DIR / "wide_d_flips_baseline_to_patched_v1.csv")
    write_csv(trans, WD_DIR / "wide_d_donor_role_transitions_v1.csv")
    write_csv(dtab, WD_DIR / "wide_d_d_reproduction_v1.csv")
    write_csv(flip_rx, WD_DIR / "wide_d_flips_baseline_to_patched_receivers_v1.csv.gz")
    write_csv(trans_rx, WD_DIR / "wide_d_donor_role_transitions_receivers_v1.csv.gz")
    write_csv(d_pairs, WD_DIR / "wide_d_d_pairs_v1.csv.gz")
    write_csv(id_rx, WD_DIR / "wide_d_baseline_id_match_receivers_v1.csv.gz")

    meta = {
        "started_utc": started,
        "finished_utc": utcnow(),
        "python": sys.version,
        "platform": platform.platform(),
        "draws": DRAWS,
        "seed": SEED,
        "model_order": MODELS,
        "notes": NOTES,
    }

    write_json(
        {"version": "wide_d_baseline_v1", "scope": WD_SCOPE, **meta,
         "panels": records(baseline)},
        WD_DIR / "wide_d_baseline_v1.json",
    )
    write_json(
        {"version": "wide_d_flips_baseline_to_patched_v1", "scope": FLIP_SCOPE,
         "comparison": "baseline_to_patched", **meta,
         "panels": records(flips)},
        WD_DIR / "wide_d_flips_baseline_to_patched_v1.json",
    )
    write_json(
        {"version": "wide_d_donor_role_transitions_v1", "scope": FLIP_SCOPE,
         "comparison": "clean_donor_to_error_donor", **meta,
         "panels": records(trans)},
        WD_DIR / "wide_d_donor_role_transitions_v1.json",
    )
    write_json(
        {"version": "wide_d_d_reproduction_v1", "scope": FLIP_SCOPE, **meta,
         "reproduction_note": "point estimates are expected to match the stored report exactly; the stored run's bootstrap index stream (bootstrap.indices_sha256) is its own and need not equal this stage's draws_signature, so CI endpoints are compared for agreement, not asserted",
         "panels": records(dtab)},
        WD_DIR / "wide_d_d_reproduction_v1.json",
    )
    write_json(
        {
            "version": "wide_d_baseline_id_match_v1",
            "scope": "per-ID comparison of the wide-D unpatched margins with the window-run unpatched margins (join on pair_id + receiver_role)",
            "tolerance": ID_MATCH_TOL,
            "interpretation": (
                "conclusion 'identical_within_1e-9' means the window-run unpatched baseline "
                "may be reused verbatim for this model's wide-D run. Otherwise the wide-D run "
                "re-scored the same receivers to different values on some subset and must be "
                "reported with its own baseline (OUT/wide_d/wide_d_baseline_v1.csv); the token "
                "counts, pair_ids, receiver roles and source components all match, so the "
                "difference is run-to-run numerical, not a different sample or a join error."
            ),
            **meta,
            "models": {res["validation"]["model"]: res["validation"]["id_match"] for res in results},
            "conclusion_by_model": {
                res["validation"]["model"]: res["validation"]["id_match"]["conclusion"]
                for res in results
            },
        },
        WD_DIR / "wide_d_baseline_id_match_v1.json",
    )

    files = {}
    for p in sorted(WD_DIR.iterdir()):
        if p.is_file() and p.name != "validation_wide_d.json":
            files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}

    validation = {
        "version": "validation_wide_d_v1",
        "stage": "B2 wide-D supplement",
        "scope": FLIP_SCOPE,
        **meta,
        "elapsed_seconds": time.time() - t0,
        "inputs": {
            "extracted_dir": str(EXTRACT_DIR),
            "extraction_manifest_sha256": sha256_file(EXTRACT_DIR / "manifest.json"),
            "window_baseline_receivers": str(WINDOW_BASELINE),
            "window_baseline_receivers_sha256": sha256_file(WINDOW_BASELINE),
            "stored_reports_dir": str(REPORT_DIR),
        },
        "check_totals": {
            "models": len(results),
            "expected_rows_per_model": 2900,
            "expected_receivers_per_model": 580,
            "expected_pairs_per_model": 290,
            "all_models_2900_rows": all(r["validation"]["n_rows"] == 2900 for r in results),
            "all_models_580_receivers": all(
                r["validation"]["n_receivers_unpatched"] == 580 for r in results
            ),
            "all_models_290_complete_pairs": all(
                r["validation"]["n_pairs_complete"] == 290 for r in results
            ),
            "all_models_K_235": all(r["validation"]["n_components_K"] == 235 for r in results),
            "max_undefined_draws_any_endpoint": max(
                r["validation"]["metrics"][m]["max_undefined_draws"]
                for r in results
                for m in METRICS
            ),
            "endpoints_with_undefined_draws": sorted(
                {
                    f"{r['validation']['model']}|{m}|{name}"
                    for r in results
                    for m in METRICS
                    for name, c in r["validation"]["metrics"][m]["undefined_draws"].items()
                    if c > 0
                }
            ),
            "undefined_draws_explanation": (
                "beneficial_rate_cond on the clean-receiver panels conditions on the baseline "
                "being WRONG on a clean receiver. Where the clean-receiver false-positive rate "
                "is 0 (OLMoE) that denominator is zero in the data and in every draw, so the "
                "point estimate is null (never 0) and all 50000 draws are undefined and counted. "
                "Every other endpoint has 50000 defined draws."
            ),
            "max_d_point_error_all": max(
                r["validation"]["metrics"][m]["d_reproduction_max_abs_point_error"]
                for r in results
                for m in METRICS
            ),
            "max_d_ci_error_all": max(
                r["validation"]["metrics"][m]["d_reproduction_max_abs_ci_error"]
                for r in results
                for m in METRICS
            ),
            "max_delta_accuracy_identity_error": max(
                r["validation"]["metrics"][m]["delta_accuracy_identity_max_abs_error"]
                for r in results
                for m in METRICS
            ),
            "id_match_conclusions": sorted(
                {r["validation"]["id_match"]["conclusion"] for r in results}
            ),
            "max_identity_drift": max(
                r["validation"]["metrics"][m]["identity_drift"][n]["max_abs_diff"]
                for r in results
                for m in METRICS
                for n in ("self_patch", "unpatched_rerun")
            ),
            "total_identity_verdict_flips": sum(
                r["validation"]["metrics"][m]["identity_drift"][n]["n_verdict_flips"]
                for r in results
                for m in METRICS
                for n in ("self_patch", "unpatched_rerun")
            ),
        },
        "models": {res["validation"]["model"]: res["validation"] for res in results},
        "outputs": files,
    }
    write_json(validation, WD_DIR / "validation_wide_d.json")
    print(f"wide_d written; total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
