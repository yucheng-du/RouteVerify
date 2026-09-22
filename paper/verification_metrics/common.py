"""Shared constants, verdict rules and cluster bootstrap for the CPU verification-metrics stage.

Read-only derived statistics on archived nine-model window runs. No model execution,
no archive writes, no network.

DEFINITIONS (fixed by the task contract)
----------------------------------------
M (margin) = score(INCORRECT) - score(CORRECT).
  metric "mean"  -> mean_logprob version  (PRIMARY)
  metric "total" -> total_logprob version (SENSITIVITY; never pooled with "mean")

Verdict rule (candidate scoring, not free generation):
  M > 0  -> "INCORRECT"
  M <= 0 -> "CORRECT"
Truth: clean receiver -> CORRECT, corrupt (error-bearing) receiver -> INCORRECT.
A receiver is baseline-correct iff verdict(unpatched M) == truth.
The zero threshold is never moved and no sample is ever dropped.

Per pair i, with unpatched margins m_C (clean receiver) and m_E (corrupt receiver),
and patched margins m_CpE (clean receiver patched with the error donor) and
m_EpC (corrupt receiver patched with the clean donor):

  delta_clean = m_CpE - m_C
  delta_error = m_EpC - m_E
  T_i         = (delta_clean - delta_error) / 2
  g_i         = m_E - m_C                      (pre-transfer gap)
  g_after_i   = m_EpC - m_CpE
  identity:     2 * T_i = g_i - g_after_i

For the controls (self / matched / norm_random) the same formula is applied to that
control's patched rows.

Normalized effect:
  R = mean(T_i) / mean(g_i) = sum(T_i) / sum(g_i)
over the SAME eligible pairs and the SAME metric. Ratio of means, never the mean of
per-pair ratios; no epsilon, no absolute values, no sample dropping.

BOOTSTRAP CONVENTION (replicates summarize_group in
docs/results/abc_synthesis_20260908/unified_windows.py exactly)
-----------------------------------------------------------------
Clusters are the sorted distinct receiver source components among the eligible units.
    rng = np.random.Generator(np.random.PCG64(20260908))
    for start in range(0, 50000, 256):
        idx = rng.integers(0, K, (min(256, 50000 - start), K))
A draw's statistic is (sum over sampled clusters of per-cluster numerator sums) divided
by (sum over sampled clusters of per-cluster denominator sums); a cluster sampled m
times counts m times, and pairs/receivers keep equal weight inside a cluster. The
interval is the two-sided pointwise percentile CI np.quantile(draws, [0.025, 0.975]).

Identical eligibility (the same sorted cluster list) implies an identical draw sequence,
so every endpoint of a panel is computed in ONE pass over shared draws
(ClusterBootstrap.panel_stats). Donor assignments are fixed and never resampled.
Conditional-rate denominators are recomputed inside each draw; a draw whose denominator
is zero is UNDEFINED: it is counted, excluded from the quantile, and reported. A point
estimate with a zero denominator is null, never 0.

Weighted, tie-aware AUROC per draw: the weight of a receiver is the multiplicity of its
cluster in that draw, and
  AUROC = sum_{i in corrupt, j in clean} w_i w_j [1(M_i > M_j) + 0.5 * 1(M_i == M_j)]
          / ((sum_i w_i) (sum_j w_j))
The n_corrupt x n_clean comparison matrix is precomputed once and applied to batches of
draws by matrix product.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT


MODELS = [
    "Qwen3.6",
    "Qwen3",
    "GLM-4.7-Flash",
    "Lightning",
    "Nano",
    "Gemma4",
    "OLMoE",
    "GPT-OSS-20B",
    "GPT-OSS-120B",
]

MODEL_KEYS = {
    "Qwen3.6": "qwen36_moe",
    "Qwen3": "qwen3_moe",
    "GLM-4.7-Flash": "glm4_moe_lite",
    "Lightning": "nemotron35_lightning",
    "Nano": "nemotron3_nano",
    "Gemma4": "gemma4_moe",
    "OLMoE": "olmoe",
    "GPT-OSS-20B": "gpt_oss_20b",
    "GPT-OSS-120B": "gpt_oss_120b",
}

SOURCES = {label: OUT / 'references/windows' / (key + '.json') for label, key in MODEL_KEYS.items()}

ANCHORS = {
    "Qwen3.6": 20,
    "Qwen3": 24,
    "GLM-4.7-Flash": 24,
    "Lightning": 27,
    "Nano": 27,
    "Gemma4": 15,
    "OLMoE": 8,
    "GPT-OSS-20B": 12,
    "GPT-OSS-120B": 18,
}

WINDOWS = ["last_token", "suffix_k", "divergence_k", "full_step"]
ARMS = ["router_logits", "moe_output", "post_moe_residual", "all_router"]
CONTROLS = ["paired", "self", "matched", "norm_random"]
METRICS = ["mean", "total"]

DRAWS = 50000
SEED = 20260908
BATCH = 256

ROLE_TRUTH = {"clean": "CORRECT", "corrupt": "INCORRECT"}

SCOPE = (
    "baseline on the PRMBench paired assay used in this study; candidate-scoring verdict"
)

WINDOW_ROW_COLUMNS = [
    "pair_id",
    "component",
    "receiver_role",
    "treatment",
    "window",
    "arm",
    "alpha",
    "donor_role",
    "donor_pair_id",
    "donor_component",
    "baseline_margin_mean",
    "margin_mean",
    "margin_total",
    "correct_mean_logprob",
    "correct_total_logprob",
    "correct_token_count",
    "incorrect_mean_logprob",
    "incorrect_total_logprob",
    "incorrect_token_count",
    "row_index",
]


# --------------------------------------------------------------------------- verdicts


def verdict(m: float) -> str:
    """Candidate-scoring verdict: M > 0 -> INCORRECT, M <= 0 -> CORRECT."""
    return "INCORRECT" if m > 0 else "CORRECT"


def verdict_array(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return np.where(m > 0, "INCORRECT", "CORRECT")


def is_correct(m: float, role: str) -> bool:
    """True iff the verdict on margin m matches the truth implied by the receiver role."""
    return verdict(m) == ROLE_TRUTH[role]


# --------------------------------------------------------------------------- loading


def window_rows_path(label: str) -> Path:
    return OUT / "extracted" / "windows" / f"{MODEL_KEYS[label]}.csv.gz"


def load_window_rows(label: str) -> pd.DataFrame:
    """Load the slim extracted rows for one model (written by extract_windows.py)."""
    path = window_rows_path(label)
    df = pd.read_csv(
        path,
        dtype={
            "pair_id": "string",
            "component": "string",
            "receiver_role": "string",
            "treatment": "string",
            "window": "string",
            "arm": "string",
            "donor_role": "string",
            "donor_pair_id": "string",
            "donor_component": "string",
        },
    )
    return df


# --------------------------------------------------------------------------- bootstrap


class ClusterBootstrap:
    """Cluster (receiver source component) bootstrap with a fixed shared draw sequence.

    All endpoints requested in one panel_stats call consume the SAME index batches, so
    panels with identical eligibility are directly comparable draw by draw.
    """

    def __init__(
        self,
        cluster_ids_sorted: list[str],
        draws: int = DRAWS,
        seed: int = SEED,
        batch: int = BATCH,
    ) -> None:
        self.cluster_ids = list(cluster_ids_sorted)
        if self.cluster_ids != sorted(self.cluster_ids):
            raise ValueError("cluster_ids must be sorted")
        if len(set(self.cluster_ids)) != len(self.cluster_ids):
            raise ValueError("cluster_ids must be unique")
        self.K = len(self.cluster_ids)
        self.draws = int(draws)
        self.seed = int(seed)
        self.batch = int(batch)
        self.index = {c: i for i, c in enumerate(self.cluster_ids)}

    # -- internals ---------------------------------------------------------

    def _batches(self):
        """Yield (start, stop, idx) exactly as unified_windows.py generates them."""
        rng = np.random.Generator(np.random.PCG64(self.seed))
        for start in range(0, self.draws, self.batch):
            stop = min(start + self.batch, self.draws)
            idx = rng.integers(0, self.K, (stop - start, self.K))
            yield start, stop, idx

    @staticmethod
    def _summarize(point, draw_values, defined_mask):
        n_def = int(defined_mask.sum())
        n_undef = int(defined_mask.size - n_def)
        if n_def == 0:
            lo = hi = None
        else:
            q = np.quantile(draw_values[defined_mask], [0.025, 0.975])
            lo, hi = float(q[0]), float(q[1])
        return {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_defined_draws": n_def,
            "n_undefined_draws": n_undef,
        }

    # -- public ------------------------------------------------------------

    def draws_signature(self) -> str:
        """sha256 over the concatenated draw index arrays (provenance of the draw sequence)."""
        h = hashlib.sha256()
        for _, _, idx in self._batches():
            h.update(np.ascontiguousarray(idx).tobytes())
        return h.hexdigest()

    def panel_stats(
        self,
        unit_cluster_index: np.ndarray,
        ratio_num: np.ndarray | None = None,
        ratio_den: np.ndarray | None = None,
        auroc_specs: list[dict] | None = None,
        keep_draws: bool = False,
    ) -> dict:
        """One pass over the shared draws for every endpoint of a panel.

        ratio_num / ratio_den: arrays of shape (units, E).
        auroc_specs: list of {"name", "scores", "is_positive"}.
        Returns {"ratios": [stat, ...], "auroc": [stat, ...], "draws_signature": hex,
                 optionally "ratio_draws" / "auroc_draws"}.
        """
        unit_cluster_index = np.asarray(unit_cluster_index, dtype=np.int64)
        n_units = unit_cluster_index.shape[0]
        if n_units and (unit_cluster_index.min() < 0 or unit_cluster_index.max() >= self.K):
            raise ValueError("unit_cluster_index out of range")

        have_ratio = ratio_num is not None
        if have_ratio:
            ratio_num = np.asarray(ratio_num, dtype=float)
            ratio_den = np.asarray(ratio_den, dtype=float)
            if ratio_num.ndim == 1:
                ratio_num = ratio_num[:, None]
                ratio_den = ratio_den[:, None]
            if ratio_num.shape != ratio_den.shape or ratio_num.shape[0] != n_units:
                raise ValueError("ratio arrays must be (units, E) and aligned")
            if not np.isfinite(ratio_num).all() or not np.isfinite(ratio_den).all():
                raise ValueError("ratio arrays must be finite")
            E = ratio_num.shape[1]
            cnum = np.zeros((self.K, E))
            cden = np.zeros((self.K, E))
            np.add.at(cnum, unit_cluster_index, ratio_num)
            np.add.at(cden, unit_cluster_index, ratio_den)
            ratio_draws = np.empty((self.draws, E))
        else:
            E = 0
            ratio_draws = None

        specs = list(auroc_specs or [])
        auroc_prepared = []
        for spec in specs:
            scores = np.asarray(spec["scores"], dtype=float)
            pos_mask = np.asarray(spec["is_positive"], dtype=bool)
            if scores.shape[0] != n_units or pos_mask.shape[0] != n_units:
                raise ValueError("auroc arrays must align with units")
            pos_idx = np.flatnonzero(pos_mask)
            neg_idx = np.flatnonzero(~pos_mask)
            sp = scores[pos_idx][:, None]
            sn = scores[neg_idx][None, :]
            cmp_matrix = (sp > sn).astype(float) + 0.5 * (sp == sn).astype(float)
            auroc_prepared.append(
                {
                    "name": spec.get("name", ""),
                    "cmp": cmp_matrix,
                    "pos_clusters": unit_cluster_index[pos_idx],
                    "neg_clusters": unit_cluster_index[neg_idx],
                    "n_pos": int(pos_idx.size),
                    "n_neg": int(neg_idx.size),
                }
            )
        auroc_draws = (
            np.empty((self.draws, len(auroc_prepared))) if auroc_prepared else None
        )

        need_multiplicity = bool(auroc_prepared)
        h = hashlib.sha256()
        for start, stop, idx in self._batches():
            h.update(np.ascontiguousarray(idx).tobytes())
            b = stop - start
            if have_ratio:
                num = cnum[idx].sum(axis=1)  # (b, E)
                den = cden[idx].sum(axis=1)  # (b, E)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio_draws[start:stop] = np.where(den != 0.0, num / np.where(den == 0.0, 1.0, den), np.nan)
            if need_multiplicity:
                flat = idx + (np.arange(b, dtype=np.int64) * self.K)[:, None]
                mult = np.bincount(flat.ravel(), minlength=b * self.K).reshape(b, self.K)
                mult = mult.astype(float)
                for j, prep in enumerate(auroc_prepared):
                    wp = mult[:, prep["pos_clusters"]]  # (b, n_pos)
                    wn = mult[:, prep["neg_clusters"]]  # (b, n_neg)
                    num = ((wp @ prep["cmp"]) * wn).sum(axis=1)
                    den = wp.sum(axis=1) * wn.sum(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        auroc_draws[start:stop, j] = np.where(
                            den != 0.0, num / np.where(den == 0.0, 1.0, den), np.nan
                        )
        signature = h.hexdigest()

        ratio_stats = []
        for j in range(E):
            den_total = float(cden[:, j].sum())
            num_total = float(cnum[:, j].sum())
            point = None if den_total == 0.0 else num_total / den_total
            col = ratio_draws[:, j]
            ratio_stats.append(self._summarize(point, col, np.isfinite(col)))

        auroc_stats = []
        for j, prep in enumerate(auroc_prepared):
            denom = prep["n_pos"] * prep["n_neg"]
            point = None if denom == 0 else float(prep["cmp"].sum() / denom)
            col = auroc_draws[:, j]
            stat = self._summarize(point, col, np.isfinite(col))
            stat["name"] = prep["name"]
            stat["n_positive"] = prep["n_pos"]
            stat["n_negative"] = prep["n_neg"]
            auroc_stats.append(stat)

        result = {
            "ratios": ratio_stats,
            "auroc": auroc_stats,
            "draws_signature": signature,
            "n_draws": self.draws,
            "K": self.K,
        }
        if keep_draws:
            result["ratio_draws"] = ratio_draws
            result["auroc_draws"] = auroc_draws
        return result

    def ratio_stats(
        self,
        unit_cluster_index: np.ndarray,
        numerators: np.ndarray,
        denominators: np.ndarray,
    ) -> list[dict]:
        return self.panel_stats(unit_cluster_index, numerators, denominators)["ratios"]

    def weighted_auroc(
        self,
        unit_cluster_index: np.ndarray,
        scores: np.ndarray,
        is_positive: np.ndarray,
    ) -> dict:
        out = self.panel_stats(
            unit_cluster_index,
            auroc_specs=[{"name": "auroc", "scores": scores, "is_positive": is_positive}],
        )["auroc"][0]
        return {
            k: out[k]
            for k in ("point", "ci_low", "ci_high", "n_defined_draws", "n_undefined_draws")
        }


# --------------------------------------------------------------------------- io


def write_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if path.name.endswith(".gz"):
        kwargs["compression"] = {"method": "gzip", "mtime": 0}
    df.to_csv(path, index=False, float_format="%.17g", **kwargs)
    return path


def write_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
    path.write_text(text + "\n")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)!r}")


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
