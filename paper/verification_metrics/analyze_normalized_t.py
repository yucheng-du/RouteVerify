"""Stage B1c: normalized transfer effect R = mean(T) / mean(pre-transfer gap).

Read-only derived statistics on the archived nine-model window runs. No model
execution, no archive writes, no network, no GPU.

For every model x window x arm x control x metric panel this script rebuilds the
per-pair quantities from the extracted rows and the stage B1a baseline tables:

    delta_clean = m_CpE - m_C
    delta_error = m_EpC - m_E
    T_i         = (delta_clean - delta_error) / 2
    g_i         = m_E - m_C              (pre-transfer gap)
    g_after_i   = m_EpC - m_CpE
    identity      2 * T_i = g_i - g_after_i

and reports, from ONE shared cluster-bootstrap pass per panel:

    mean_T, mean_g, mean_g_after, R = sum(T)/sum(g),
    frac_g_nonpositive, positive_pair_fraction_T

R is a ratio of means over the same eligible pairs and the same metric; never the
mean of per-pair ratios, no epsilon, no absolute values, no sample dropping.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python \
     docs/results/verification_metrics_cpu_20260917/analyze_normalized_t.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

import common  # noqa: E402

OUT = common.OUT / "normalized_t"
BASELINE = common.OUT / "baseline"
ABC = common.OUT / "references/abc"
PAIR_VALUES = ABC / "window_pair_values_v1.csv"
UNIFIED = ABC / "unified_windows_v1.csv"

ENDPOINTS = [
    "mean_T",
    "mean_g",
    "mean_g_after",
    "R",
    "frac_g_nonpositive",
    "positive_pair_fraction_T",
]
STAT_SUFFIXES = ["", "_ci_low", "_ci_high", "_n_defined_draws", "_n_undefined_draws"]

SCOPE = common.SCOPE

INTERPRETATION = (
    "R = sum(T_i) / sum(g_i) over the same eligible pairs and the same metric: a ratio of "
    "means, not the mean of per-pair ratios. R = 0.5 corresponds to a mean post-transfer gap "
    "of zero (mean(g_after) = 0), and R = 1 to an equal-magnitude sign reversal of the mean "
    "gap (mean(g_after) = -mean(g)). R may be negative or exceed 1. R is not a mediation "
    "share, not explained variance and not a probability, and it is not the same quantity as "
    "dividing each sample's effect by its own distance to the decision boundary |M0| (that "
    "per-sample normalization is not computed here). When the denominator mean(g) is close to "
    "zero R is unstable; the stability column records this and blanks the R interval, while "
    "mean_T and its interval are always reported unchanged."
)

# maximum number of panels whose endpoint columns are stacked into one bootstrap pass;
# the draw sequence depends only on K, so chunking leaves every draw identical.
CHUNK_PANELS = 8


def _records(df: pd.DataFrame) -> list[dict]:
    """Full-precision JSON records: native Python objects, NaN/NA -> None."""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def _stat_columns(prefix: str, stat: dict) -> dict:
    return {
        f"{prefix}": stat["point"],
        f"{prefix}_ci_low": stat["ci_low"],
        f"{prefix}_ci_high": stat["ci_high"],
        f"{prefix}_n_defined_draws": stat["n_defined_draws"],
        f"{prefix}_n_undefined_draws": stat["n_undefined_draws"],
    }


STR_COLS = {
    "pair_id": "string",
    "component": "string",
    "receiver_role": "string",
    "treatment": "string",
    "window": "string",
    "arm": "string",
    "donor_role": "string",
    "donor_pair_id": "string",
    "donor_component": "string",
}


def load_rows_exact(model: str) -> pd.DataFrame:
    """common.load_window_rows with float_precision='round_trip'.

    DEVIATION, deliberate and reported: pandas 3.0.3's default CSV float parser
    ('high' / xstrtod) is not correctly rounded and loses up to 1 ulp, so
    common.load_window_rows returns floats that differ from the archived values in
    the last bit. That noise is invisible in the paired contrasts but it turns the
    self control -- whose per-pair T is exactly 0 in the archives -- into +-1e-16
    coin flips, which corrupts sign-based endpoints such as positive_pair_fraction.
    Reading with float_precision='round_trip' recovers the archived float64 values
    bit for bit. Nothing else about the loader changes.
    """
    return pd.read_csv(
        common.window_rows_path(model), dtype=STR_COLS, float_precision="round_trip"
    )


def _load_baseline(model: str):
    recv = pd.read_csv(
        BASELINE / "baseline_receivers_v1.csv",
        usecols=["model", "pair_id", "component", "receiver_role", "margin_mean", "margin_total"],
        dtype={
            "model": "string",
            "pair_id": "string",
            "component": "string",
            "receiver_role": "string",
        },
        float_precision="round_trip",
    )
    recv = recv[recv["model"] == model].drop(columns=["model"]).reset_index(drop=True)
    pairs = pd.read_csv(
        BASELINE / "baseline_pairs_v1.csv",
        usecols=["model", "pair_id", "component", "m_C_mean", "m_E_mean", "m_C_total", "m_E_total"],
        dtype={"model": "string", "pair_id": "string", "component": "string"},
        float_precision="round_trip",
    )
    pairs = pairs[pairs["model"] == model].drop(columns=["model"]).reset_index(drop=True)
    return recv, pairs


def _load_archive_pair_T(model: str) -> pd.DataFrame:
    df = pd.read_csv(
        PAIR_VALUES,
        usecols=["model", "window", "arm", "control", "pair_id", "T"],
        dtype={
            "model": "string",
            "window": "string",
            "arm": "string",
            "control": "string",
            "pair_id": "string",
        },
        float_precision="round_trip",
    )
    df = df[df["model"] == model].drop(columns=["model"]).reset_index(drop=True)
    return df.rename(columns={"T": "archived_T"})


def _load_unified(model: str) -> pd.DataFrame:
    df = pd.read_csv(UNIFIED, float_precision="round_trip")
    df = df[(df["model"] == model) & (df["estimand"] == "T")]
    return df[
        [
            "window",
            "arm",
            "control",
            "mean",
            "ci_low",
            "ci_high",
            "N",
            "K",
            "positive_pair_fraction",
        ]
    ].reset_index(drop=True)


def _analysis_pair_estimands(model: str) -> dict:
    payload = json.loads(common.SOURCES[model].read_text())
    out = {}
    for r in payload["pair_estimands"]:
        out[(r["window"], r["arm"], r["control"], r["pair_id"])] = float(r["T"])
    return out


def build_pair_table(model: str) -> pd.DataFrame:
    """Per pair x panel x metric quantities for one model (long form)."""
    rows = load_rows_exact(model)
    recv, bpairs = _load_baseline(model)

    # unpatched margins, joined by (pair_id, receiver_role) ID -- never by line order
    unp = rows[rows["treatment"] == "unpatched"][
        ["pair_id", "receiver_role", "component", "margin_mean", "margin_total"]
    ].rename(columns={"margin_mean": "base_mean", "margin_total": "base_total"})
    if unp.duplicated(["pair_id", "receiver_role"]).any():
        raise RuntimeError(f"{model}: duplicate unpatched receiver keys")

    # cross-check the stage B1a baseline tables against the same IDs
    chk = unp.merge(
        recv.rename(columns={"margin_mean": "b_mean", "margin_total": "b_total"}),
        on=["pair_id", "receiver_role", "component"],
        how="outer",
        indicator=True,
    )
    baseline_table_set_equal = bool((chk["_merge"] == "both").all())
    baseline_table_max_abs_mean = float(np.abs(chk["base_mean"] - chk["b_mean"]).max())
    baseline_table_max_abs_total = float(np.abs(chk["base_total"] - chk["b_total"]).max())
    gchk = (
        unp.pivot(index="pair_id", columns="receiver_role", values=["base_mean", "base_total"])
        .reset_index()
    )
    g_mine = (gchk[("base_mean", "corrupt")] - gchk[("base_mean", "clean")]).to_numpy(float)
    g_tab = (
        bpairs.set_index("pair_id").loc[gchk["pair_id"].to_numpy(), "g_mean"].to_numpy(float)
        if "g_mean" in bpairs.columns
        else (
            bpairs.set_index("pair_id").loc[gchk["pair_id"].to_numpy(), "m_E_mean"].to_numpy(float)
            - bpairs.set_index("pair_id").loc[gchk["pair_id"].to_numpy(), "m_C_mean"].to_numpy(float)
        )
    )
    baseline_table_max_abs_g = float(np.abs(g_mine - g_tab).max())

    patched = rows[rows["treatment"].isin(common.CONTROLS)][
        [
            "pair_id",
            "component",
            "receiver_role",
            "treatment",
            "window",
            "arm",
            "baseline_margin_mean",
            "margin_mean",
            "margin_total",
        ]
    ].copy()

    merged = patched.merge(unp, on=["pair_id", "receiver_role", "component"], how="left")
    if merged["base_mean"].isna().any():
        raise RuntimeError(f"{model}: patched rows without a baseline partner")
    # the archives carry baseline_margin_mean on patched rows; confirm the ID join agrees
    base_join_max_abs = float(
        np.abs(merged["baseline_margin_mean"].to_numpy() - merged["base_mean"].to_numpy()).max()
    )

    clean = merged[merged["receiver_role"] == "clean"].drop(columns=["receiver_role"])
    corrupt = merged[merged["receiver_role"] == "corrupt"].drop(columns=["receiver_role"])
    keys = ["pair_id", "component", "window", "arm", "treatment"]
    pair = clean.merge(corrupt, on=keys, suffixes=("_c", "_e"), how="inner")

    recs = []
    for metric, suffix in (("mean", "mean"), ("total", "total")):
        m_C = pair[f"base_{suffix}_c"].to_numpy(dtype=float)
        m_E = pair[f"base_{suffix}_e"].to_numpy(dtype=float)
        m_CpE = pair[f"margin_{suffix}_c"].to_numpy(dtype=float)
        m_EpC = pair[f"margin_{suffix}_e"].to_numpy(dtype=float)
        delta_clean = m_CpE - m_C
        delta_error = m_EpC - m_E
        T = (delta_clean - delta_error) / 2.0
        g = m_E - m_C
        g_after = m_EpC - m_CpE
        identity_error = np.abs(2.0 * T - (g - g_after))
        recs.append(
            pd.DataFrame(
                {
                    "model": model,
                    "scope": SCOPE,
                    "window": pair["window"].to_numpy(),
                    "arm": pair["arm"].to_numpy(),
                    "control": pair["treatment"].to_numpy(),
                    "metric": metric,
                    "pair_id": pair["pair_id"].to_numpy(),
                    "component": pair["component"].to_numpy(),
                    "m_C": m_C,
                    "m_E": m_E,
                    "m_CpE": m_CpE,
                    "m_EpC": m_EpC,
                    "delta_clean": delta_clean,
                    "delta_error": delta_error,
                    "T": T,
                    "g": g,
                    "g_after": g_after,
                    "identity_error": identity_error,
                }
            )
        )
    out = pd.concat(recs, ignore_index=True)
    out.attrs["base_join_max_abs"] = base_join_max_abs
    out.attrs["baseline_table_set_equal"] = baseline_table_set_equal
    out.attrs["baseline_table_max_abs_mean"] = baseline_table_max_abs_mean
    out.attrs["baseline_table_max_abs_total"] = baseline_table_max_abs_total
    out.attrs["baseline_table_max_abs_g"] = baseline_table_max_abs_g
    return out


def analyze_model(model: str) -> dict:
    t0 = time.time()
    pair_tab = build_pair_table(model)
    base_join_max_abs = pair_tab.attrs["base_join_max_abs"]
    baseline_table_checks = {
        "baseline_table_ids_equal": pair_tab.attrs["baseline_table_set_equal"],
        "baseline_table_max_abs_diff_margin_mean": pair_tab.attrs["baseline_table_max_abs_mean"],
        "baseline_table_max_abs_diff_margin_total": pair_tab.attrs["baseline_table_max_abs_total"],
        "baseline_table_max_abs_diff_g_mean": pair_tab.attrs["baseline_table_max_abs_g"],
    }

    # ---- archive reproduction (mean metric only; archives store the mean-logprob T)
    arch = _load_archive_pair_T(model)
    mean_tab = pair_tab[pair_tab["metric"] == "mean"]
    joined = mean_tab.merge(
        arch, on=["window", "arm", "control", "pair_id"], how="outer", indicator=True
    )
    only_mine = joined[joined["_merge"] == "left_only"]
    only_archive = joined[joined["_merge"] == "right_only"]
    both = joined[joined["_merge"] == "both"]
    arch_abs_err = np.abs(both["T"].to_numpy(dtype=float) - both["archived_T"].to_numpy(dtype=float))
    archive_max_abs_error = float(arch_abs_err.max()) if arch_abs_err.size else 0.0

    arch_map = dict(
        zip(
            zip(
                arch["window"].astype(str),
                arch["arm"].astype(str),
                arch["control"].astype(str),
                arch["pair_id"].astype(str),
                strict=True,
            ),
            arch["archived_T"].to_numpy(dtype=float),
            strict=True,
        )
    )
    analysis_map = _analysis_pair_estimands(model)
    analysis_only = sorted(set(analysis_map) - set(arch_map))
    csv_only = sorted(set(arch_map) - set(analysis_map))
    analysis_vs_csv_max = 0.0
    for k in set(analysis_map) & set(arch_map):
        analysis_vs_csv_max = max(analysis_vs_csv_max, abs(analysis_map[k] - arch_map[k]))

    # per-pair T vs analysis.json directly
    mine_keys = set(
        zip(
            mean_tab["window"].astype(str),
            mean_tab["arm"].astype(str),
            mean_tab["control"].astype(str),
            mean_tab["pair_id"].astype(str),
            strict=True,
        )
    )
    analysis_set_equal = mine_keys == set(analysis_map)
    mine_T = dict(
        zip(
            zip(
                mean_tab["window"].astype(str),
                mean_tab["arm"].astype(str),
                mean_tab["control"].astype(str),
                mean_tab["pair_id"].astype(str),
                strict=True,
            ),
            mean_tab["T"].to_numpy(dtype=float),
            strict=True,
        )
    )
    analysis_max_abs_error = 0.0
    for k in mine_keys & set(analysis_map):
        analysis_max_abs_error = max(analysis_max_abs_error, abs(mine_T[k] - analysis_map[k]))

    unified = _load_unified(model)
    unified_idx = unified.set_index(["window", "arm", "control"])
    arch_groups = {
        k: dict(zip(v["pair_id"].astype(str), v["archived_T"].astype(float), strict=True))
        for k, v in arch.groupby(["window", "arm", "control"], sort=True)
    }

    # ---- panels: group by identical eligible pair set so shared draws are reused
    panel_keys = []
    panel_data = {}
    for (window, arm, control, metric), sub in pair_tab.groupby(
        ["window", "arm", "control", "metric"], sort=True
    ):
        sub = sub.sort_values("pair_id", kind="mergesort")
        key = (window, arm, control, metric)
        panel_keys.append(key)
        panel_data[key] = sub

    groups = {}
    for key in panel_keys:
        sub = panel_data[key]
        eligible = tuple(sub["pair_id"].tolist())
        groups.setdefault(eligible, []).append(key)

    panel_records = []
    for eligible, keys in sorted(groups.items(), key=lambda kv: (len(kv[0]), kv[1][0])):
        ref = panel_data[keys[0]]
        components = ref["component"].to_numpy()
        clusters = sorted(set(components.tolist()))
        boot = common.ClusterBootstrap(clusters)
        cluster_index = np.array([boot.index[c] for c in components], dtype=np.int64)
        n_units = len(eligible)
        ones = np.ones(n_units)

        for chunk_start in range(0, len(keys), CHUNK_PANELS):
            chunk = keys[chunk_start : chunk_start + CHUNK_PANELS]
            num_cols, den_cols = [], []
            for key in chunk:
                sub = panel_data[key]
                if tuple(sub["pair_id"].tolist()) != eligible:
                    raise RuntimeError(f"{model}: eligibility mismatch inside group for {key}")
                T = sub["T"].to_numpy(dtype=float)
                g = sub["g"].to_numpy(dtype=float)
                g_after = sub["g_after"].to_numpy(dtype=float)
                num_cols += [T, g, g_after, T, (g <= 0).astype(float), (T > 0).astype(float)]
                den_cols += [ones, ones, ones, g, ones, ones]
            res = boot.panel_stats(
                cluster_index,
                ratio_num=np.column_stack(num_cols),
                ratio_den=np.column_stack(den_cols),
                keep_draws=True,
            )
            ratio_draws = res["ratio_draws"]

            for j, key in enumerate(chunk):
                window, arm, control, metric = key
                sub = panel_data[key]
                base = j * len(ENDPOINTS)
                stats = {
                    name: res["ratios"][base + i] for i, name in enumerate(ENDPOINTS)
                }
                # denominator diagnostics: the mean_g draw column shares its sign with sum(g)
                mean_g_draws = ratio_draws[:, base + 1]
                n_den_nonpositive = int(np.sum(mean_g_draws <= 0.0))
                n_den_zero = int(np.sum(mean_g_draws == 0.0))

                g = sub["g"].to_numpy(dtype=float)
                T = sub["T"].to_numpy(dtype=float)
                sum_g = float(g.sum())
                mean_g_stat = stats["mean_g"]
                ci_excludes_zero = (
                    mean_g_stat["ci_low"] is not None
                    and mean_g_stat["ci_high"] is not None
                    and (mean_g_stat["ci_low"] > 0.0 or mean_g_stat["ci_high"] < 0.0)
                )
                reasons = []
                if not ci_excludes_zero:
                    reasons.append("mean_g CI includes zero")
                if n_den_nonpositive > 0:
                    reasons.append(f"{n_den_nonpositive} draws with sum(g) <= 0")
                stability = "stable" if not reasons else "unstable"

                r_stat = dict(stats["R"])
                if stability == "unstable":
                    r_stat["ci_low"] = None
                    r_stat["ci_high"] = None
                if sum_g == 0.0:
                    r_stat["point"] = None

                rec = {
                    "model": model,
                    "key": common.MODEL_KEYS[model],
                    "anchor_layer": common.ANCHORS[model],
                    "scope": SCOPE,
                    "window": window,
                    "arm": arm,
                    "control": control,
                    "metric": metric,
                    "metric_role": "primary" if metric == "mean" else "sensitivity",
                    "N": int(len(eligible)),
                    "K": int(len(clusters)),
                    "n_pairs_model": 290,
                    "n_pairs_structural_na": int(290 - len(eligible)),
                    "panel_missing": False,
                    "draws": common.DRAWS,
                    "seed": common.SEED,
                    "draws_signature": res["draws_signature"],
                }
                for name in ENDPOINTS:
                    rec.update(_stat_columns(name, r_stat if name == "R" else stats[name]))
                rec.update(
                    {
                        "sum_T": float(T.sum()),
                        "sum_g": sum_g,
                        "n_pairs_g_nonpositive": int(np.sum(g <= 0)),
                        "n_draws_den_nonpositive": n_den_nonpositive,
                        "n_draws_den_zero": n_den_zero,
                        "stability": stability,
                        "stability_reason": "; ".join(reasons),
                        "identity_max_abs_error": float(sub["identity_error"].max()),
                    }
                )

                # archive reproduction, mean metric only
                if metric == "mean":
                    ak = (window, arm, control)
                    a_map = arch_groups.get(ak, {})
                    a_ids = set(a_map)
                    my_ids = set(eligible)
                    errs = [abs(float(t) - a_map[p]) for p, t in zip(eligible, T, strict=True) if p in a_map]
                    rec["archive_n_pairs"] = int(len(a_ids))
                    rec["archive_set_equal"] = bool(a_ids == my_ids)
                    rec["archive_missing_from_mine"] = int(len(a_ids - my_ids))
                    rec["archive_extra_in_mine"] = int(len(my_ids - a_ids))
                    rec["archive_pair_max_abs_error"] = float(max(errs)) if errs else None
                    if ak in unified_idx.index:
                        u = unified_idx.loc[ak]
                        rec["unified_mean_abs_error"] = float(
                            abs(float(u["mean"]) - stats["mean_T"]["point"])
                        )
                        rec["unified_ci_low_abs_error"] = float(
                            abs(float(u["ci_low"]) - stats["mean_T"]["ci_low"])
                        )
                        rec["unified_ci_high_abs_error"] = float(
                            abs(float(u["ci_high"]) - stats["mean_T"]["ci_high"])
                        )
                        rec["unified_N_match"] = bool(int(u["N"]) == len(eligible))
                        rec["unified_K_match"] = bool(int(u["K"]) == len(clusters))
                        rec["unified_positive_pair_fraction_abs_error"] = float(
                            abs(
                                float(u["positive_pair_fraction"])
                                - stats["positive_pair_fraction_T"]["point"]
                            )
                        )
                    else:
                        rec["unified_mean_abs_error"] = None
                        rec["unified_ci_low_abs_error"] = None
                        rec["unified_ci_high_abs_error"] = None
                        rec["unified_N_match"] = None
                        rec["unified_K_match"] = None
                        rec["unified_positive_pair_fraction_abs_error"] = None
                else:
                    rec["archive_n_pairs"] = None
                    rec["archive_set_equal"] = None
                    rec["archive_missing_from_mine"] = None
                    rec["archive_extra_in_mine"] = None
                    rec["archive_pair_max_abs_error"] = None
                    rec["unified_mean_abs_error"] = None
                    rec["unified_ci_low_abs_error"] = None
                    rec["unified_ci_high_abs_error"] = None
                    rec["unified_N_match"] = None
                    rec["unified_K_match"] = None
                    rec["unified_positive_pair_fraction_abs_error"] = None
                panel_records.append(rec)

    panels = pd.DataFrame(panel_records)
    panels = panels.sort_values(["window", "arm", "control", "metric"], kind="mergesort").reset_index(
        drop=True
    )

    # attach archived_T / archive_abs_error to the pair table (mean metric only)
    pair_out = pair_tab.merge(
        arch, on=["window", "arm", "control", "pair_id"], how="left"
    )
    pair_out.loc[pair_out["metric"] != "mean", "archived_T"] = np.nan
    pair_out["archive_abs_error"] = (pair_out["T"] - pair_out["archived_T"]).abs()
    pair_out = pair_out.sort_values(
        ["metric", "window", "arm", "control", "pair_id"], kind="mergesort"
    ).reset_index(drop=True)

    validation = {
        "model": model,
        "key": common.MODEL_KEYS[model],
        "analysis_json": str(common.SOURCES[model]),
        "baseline_join_max_abs_diff_vs_baseline_margin_mean": base_join_max_abs,
        **baseline_table_checks,
        "archive_csv_set_equal_both_directions": bool(
            len(only_mine) == 0 and len(only_archive) == 0
        ),
        "archive_csv_only_mine": int(len(only_mine)),
        "archive_csv_only_archive": int(len(only_archive)),
        "archive_csv_max_abs_error": archive_max_abs_error,
        "analysis_json_set_equal": bool(analysis_set_equal),
        "analysis_json_max_abs_error": float(analysis_max_abs_error),
        "analysis_json_vs_pair_values_csv_max_abs_error": float(analysis_vs_csv_max),
        "analysis_json_only": len(analysis_only),
        "pair_values_csv_only": len(csv_only),
        "self_control_max_abs_T_mean": float(
            pair_tab.loc[
                (pair_tab["metric"] == "mean") & (pair_tab["control"] == "self"), "T"
            ].abs().max()
        ),
        "self_control_max_abs_T_total": float(
            pair_tab.loc[
                (pair_tab["metric"] == "total") & (pair_tab["control"] == "self"), "T"
            ].abs().max()
        ),
        "identity_max_abs_error_mean": float(
            pair_tab.loc[pair_tab["metric"] == "mean", "identity_error"].max()
        ),
        "identity_max_abs_error_total": float(
            pair_tab.loc[pair_tab["metric"] == "total", "identity_error"].max()
        ),
        "unified_mean_max_abs_error": float(
            panels["unified_mean_abs_error"].max(skipna=True)
        ),
        "unified_ci_low_max_abs_error": float(
            panels["unified_ci_low_abs_error"].max(skipna=True)
        ),
        "unified_ci_high_max_abs_error": float(
            panels["unified_ci_high_abs_error"].max(skipna=True)
        ),
        "unified_N_all_match": bool(panels["unified_N_match"].dropna().all()),
        "unified_K_all_match": bool(panels["unified_K_match"].dropna().all()),
        "unified_positive_pair_fraction_max_abs_error": float(
            panels["unified_positive_pair_fraction_abs_error"].max(skipna=True)
        ),
        "archive_panels_all_set_equal": bool(panels["archive_set_equal"].dropna().all()),
        "n_panels": int(len(panels)),
        "n_unstable_panels": int((panels["stability"] == "unstable").sum()),
        "seconds": time.time() - t0,
    }
    return {"panels": panels, "pairs": pair_out, "validation": validation}


def _worker(model: str):
    return analyze_model(model)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(9) as pool:
        results = pool.map(_worker, common.MODELS, chunksize=1)

    order = {m: i for i, m in enumerate(common.MODELS)}
    panels = pd.concat([r["panels"] for r in results], ignore_index=True)
    panels["_o"] = panels["model"].map(order)
    panels = panels.sort_values(
        ["_o", "window", "arm", "control", "metric"], kind="mergesort"
    ).drop(columns=["_o"]).reset_index(drop=True)

    pairs = pd.concat([r["pairs"] for r in results], ignore_index=True)
    pairs["_o"] = pairs["model"].map(order)
    pairs = pairs.sort_values(
        ["_o", "metric", "window", "arm", "control", "pair_id"], kind="mergesort"
    ).drop(columns=["_o"]).reset_index(drop=True)

    common.write_csv(panels, OUT / "normalized_t_panels_v1.csv")
    common.write_csv(pairs, OUT / "normalized_t_pairs_v1.csv.gz")

    unstable = panels[panels["stability"] == "unstable"][
        ["model", "window", "arm", "control", "metric", "N", "K", "mean_g", "mean_g_ci_low",
         "mean_g_ci_high", "R", "n_draws_den_nonpositive", "n_draws_den_zero", "stability_reason"]
    ]

    payload = {
        "version": "normalized_t_v1",
        "scope": SCOPE,
        "interpretation": INTERPRETATION,
        "definitions": {
            "T": "(delta_clean - delta_error) / 2 with delta_clean = m_CpE - m_C and delta_error = m_EpC - m_E",
            "g": "m_E - m_C (pre-transfer gap)",
            "g_after": "m_EpC - m_CpE (post-transfer gap)",
            "identity": "2 * T_i = g_i - g_after_i",
            "R": "sum(T_i) / sum(g_i) over the same eligible pairs and the same metric",
            "metric_mean": "margin from mean_logprob (PRIMARY)",
            "metric_total": "margin from total_logprob (SENSITIVITY; never pooled with mean)",
            "eligibility": "pairs where BOTH receiver roles have a patched row in that panel",
            "stability": "stable iff the mean_g CI excludes zero and no draw has sum(g) <= 0; otherwise unstable (R point kept, R interval blanked)",
        },
        "bootstrap": {
            "draws": common.DRAWS,
            "seed": common.SEED,
            "batch": common.BATCH,
            "clusters": "receiver source components among the eligible pairs",
            "interval_type": "source_component_percentile_pair_weighted, two-sided pointwise 2.5/97.5",
        },
        "numerics": (
            "Floats are read with pandas float_precision='round_trip'. pandas 3.0.3's default "
            "CSV float parser loses up to 1 ulp, which is harmless for the paired contrasts but "
            "turns the exactly-zero self-control T into +-1e-16 sign noise; round_trip recovers "
            "the archived float64 values bit for bit. Unpatched margins are taken from the "
            "extracted unpatched rows joined by (pair_id, receiver_role); the stage B1a baseline "
            "tables are cross-checked against the same IDs and agree to <= 1 ulp "
            "(validation_normalized_t.json, baseline_table_* fields)."
        ),
        "models": common.MODELS,
        "n_panels": int(len(panels)),
        "n_pair_rows": int(len(pairs)),
        "panels": _records(panels),
    }
    common.write_json(payload, OUT / "normalized_t_panels_v1.json")

    validation = {
        "version": "normalized_t_v1",
        "scope": SCOPE,
        "per_model": {r["validation"]["model"]: r["validation"] for r in results},
        "overall": {
            "identity_max_abs_error_mean": max(
                r["validation"]["identity_max_abs_error_mean"] for r in results
            ),
            "identity_max_abs_error_total": max(
                r["validation"]["identity_max_abs_error_total"] for r in results
            ),
            "archive_pair_T_max_abs_error": max(
                r["validation"]["archive_csv_max_abs_error"] for r in results
            ),
            "analysis_json_pair_T_max_abs_error": max(
                r["validation"]["analysis_json_max_abs_error"] for r in results
            ),
            "archive_eligible_sets_all_equal": all(
                r["validation"]["archive_csv_set_equal_both_directions"] for r in results
            )
            and all(r["validation"]["analysis_json_set_equal"] for r in results),
            "unified_mean_max_abs_error": max(
                r["validation"]["unified_mean_max_abs_error"] for r in results
            ),
            "unified_ci_low_max_abs_error": max(
                r["validation"]["unified_ci_low_max_abs_error"] for r in results
            ),
            "unified_ci_high_max_abs_error": max(
                r["validation"]["unified_ci_high_max_abs_error"] for r in results
            ),
            "unified_positive_pair_fraction_max_abs_error": max(
                r["validation"]["unified_positive_pair_fraction_max_abs_error"] for r in results
            ),
            "unified_N_K_all_match": all(
                r["validation"]["unified_N_all_match"] and r["validation"]["unified_K_all_match"]
                for r in results
            ),
            "baseline_table_max_abs_diff_margin_mean": max(
                r["validation"]["baseline_table_max_abs_diff_margin_mean"] for r in results
            ),
            "baseline_table_max_abs_diff_margin_total": max(
                r["validation"]["baseline_table_max_abs_diff_margin_total"] for r in results
            ),
            "baseline_table_max_abs_diff_g_mean": max(
                r["validation"]["baseline_table_max_abs_diff_g_mean"] for r in results
            ),
            "baseline_table_ids_all_equal": all(
                r["validation"]["baseline_table_ids_equal"] for r in results
            ),
            "n_panels": int(len(panels)),
            "n_unstable_panels": int(len(unstable)),
            "n_panels_missing": 0,
            "self_control_max_abs_T_mean": max(
                r["validation"]["self_control_max_abs_T_mean"] for r in results
            ),
            "self_control_max_abs_T_total": max(
                r["validation"]["self_control_max_abs_T_total"] for r in results
            ),
            "n_undefined_draws_total": int(
                sum(panels[f"{e}_n_undefined_draws"].sum() for e in ENDPOINTS)
            ),
            "wall_seconds": time.time() - t0,
        },
        "unstable_panels": _records(unstable),
    }
    common.write_json(validation, OUT / "validation_normalized_t.json")

    for name in (
        "normalized_t_panels_v1.csv",
        "normalized_t_panels_v1.json",
        "normalized_t_pairs_v1.csv.gz",
        "validation_normalized_t.json",
    ):
        print(name, common.sha256_file(OUT / name), flush=True)
    print("panels", len(panels), "pair rows", len(pairs), "wall", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
