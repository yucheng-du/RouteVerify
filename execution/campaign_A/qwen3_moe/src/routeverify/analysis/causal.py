"""Causal effect summaries."""

from __future__ import annotations

import numpy as np

from routeverify.analysis.bootstrap import bootstrap_mean_ci


def paired_ate(
    treated: list[float] | np.ndarray,
    control: list[float] | np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    treated_array = np.asarray(treated, dtype=float)
    control_array = np.asarray(control, dtype=float)
    if treated_array.shape != control_array.shape:
        raise ValueError("treated/control shapes differ")
    effects = treated_array - control_array
    summary = bootstrap_mean_ci(effects, seed=seed)
    std = float(np.std(effects, ddof=1)) if effects.size > 1 else 0.0
    summary["standardized_effect"] = float(summary["mean"] / std) if std else 0.0
    summary["sign_consistency"] = float(np.mean(np.sign(effects) == np.sign(summary["mean"])))
    return summary
