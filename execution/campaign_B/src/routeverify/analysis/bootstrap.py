"""Bootstrap utilities."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

PAIRED_CLUSTER_RNG = "numpy.random.Generator(PCG64)"
PAIRED_CLUSTER_DRAW_METHOD = "paired_cluster_shared_indices"
_BOOTSTRAP_TARGET_ELEMENTS_PER_CHUNK = 2_000_000


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.choice(array, size=array.size, replace=True)
        means[index] = sample.mean()
    lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(array.mean()), "ci_low": float(lower), "ci_high": float(upper)}


def paired_cluster_bootstrap_indices(
    n_clusters: int,
    *,
    seed: int,
    n_resamples: int,
) -> np.ndarray:
    """Return deterministic shared cluster-resampling indices.

    Each row is one bootstrap replicate and contains ``n_clusters`` draws with
    replacement from ``[0, n_clusters)``. Reusing this matrix for every
    estimand preserves the observed cross-estimand covariance within each
    cluster. The explicit PCG64 bit generator makes the RNG contract auditable.
    """

    validated_clusters = _strict_positive_int(n_clusters, name="n_clusters")
    validated_resamples = _strict_resample_count(n_resamples)
    validated_seed = _strict_seed(seed)
    rng = np.random.Generator(np.random.PCG64(validated_seed))
    indices = rng.integers(
        0,
        validated_clusters,
        size=(validated_resamples, validated_clusters),
        dtype=np.int64,
    )
    indices.flags.writeable = False
    return indices


def paired_cluster_bootstrap_mean_ci(
    values_by_pair: list[list[float]] | np.ndarray,
    *,
    indices: np.ndarray,
    alpha: float = 0.05,
    simultaneous: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Summarize matrix means using shared paired-cluster bootstrap draws.

    ``values_by_pair`` must have shape ``[n_pairs, n_estimands]``. Marginal
    intervals are percentile intervals. When ``simultaneous`` is true, the
    result also includes a two-sided max-|t| interval across all estimands,
    using the bootstrap standard error of each estimand as its scale.

    Passing ``seed`` is optional for externally supplied index matrices. When
    supplied, the function regenerates the expected PCG64 indices and refuses
    a mismatch, so the recorded seed and RNG provenance cannot silently diverge
    from the actual draws.
    """

    values = _finite_matrix(values_by_pair)
    validated_indices = _paired_indices(indices, n_clusters=values.shape[0])
    validated_alpha = _strict_alpha(alpha)
    if not isinstance(simultaneous, (bool, np.bool_)):
        raise TypeError("simultaneous must be a boolean")
    simultaneous = bool(simultaneous)

    recorded_seed: int | None = None
    rng_name = "external"
    draw_method = "external_shared_indices"
    if seed is not None:
        recorded_seed = _strict_seed(seed)
        expected = paired_cluster_bootstrap_indices(
            values.shape[0],
            seed=recorded_seed,
            n_resamples=validated_indices.shape[0],
        )
        if not np.array_equal(validated_indices, expected):
            raise ValueError("indices do not match the recorded seed and PCG64 contract")
        rng_name = PAIRED_CLUSTER_RNG
        draw_method = PAIRED_CLUSTER_DRAW_METHOD

    resampled_means = _cluster_resampled_means(values, validated_indices)
    point = values.mean(axis=0)
    lower, upper = np.quantile(
        resampled_means,
        [validated_alpha / 2.0, 1.0 - validated_alpha / 2.0],
        axis=0,
    )
    bootstrap_standard_error = np.std(resampled_means, axis=0, ddof=1)
    centered = resampled_means - point
    covariance_centered = resampled_means - np.mean(
        resampled_means,
        axis=0,
        keepdims=True,
    )
    covariance = covariance_centered.T @ covariance_centered / (validated_indices.shape[0] - 1)

    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "paired_cluster_percentile",
        "draw_method": draw_method,
        "rng": rng_name,
        "seed": recorded_seed,
        "n_pairs": int(values.shape[0]),
        "n_estimands": int(values.shape[1]),
        "n_resamples": int(validated_indices.shape[0]),
        "alpha": validated_alpha,
        "confidence_level": 1.0 - validated_alpha,
        "indices_sha256": _indices_sha256(validated_indices),
        "point": [float(value) for value in point],
        "ci_low": [float(value) for value in lower],
        "ci_high": [float(value) for value in upper],
        "bootstrap_standard_error": [float(value) for value in bootstrap_standard_error],
        "bootstrap_covariance": [
            [float(value) for value in row] for row in np.atleast_2d(covariance)
        ],
    }
    if simultaneous:
        result["simultaneous_ci"] = _max_t_interval(
            point=point,
            centered_bootstrap_means=centered,
            bootstrap_standard_error=bootstrap_standard_error,
            alpha=validated_alpha,
        )
    return result


def paired_cluster_indices_sha256(indices: np.ndarray) -> str:
    """Return the canonical SHA-256 of an auditable shared-draw matrix."""

    array = np.asarray(indices)
    if array.ndim != 2:
        raise ValueError("indices must have shape [n_resamples, n_pairs]")
    n_clusters = _strict_positive_int(array.shape[1], name="n_pairs")
    validated = _paired_indices(array, n_clusters=n_clusters)
    return _indices_sha256(validated)


def _cluster_resampled_means(
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute bootstrap means without materializing all draws at once."""

    denominator = max(1, values.shape[0] * values.shape[1])
    chunk_size = max(1, _BOOTSTRAP_TARGET_ELEMENTS_PER_CHUNK // denominator)
    output = np.empty((indices.shape[0], values.shape[1]), dtype=float)
    for start in range(0, indices.shape[0], chunk_size):
        stop = min(start + chunk_size, indices.shape[0])
        output[start:stop] = values[indices[start:stop]].mean(axis=1)
    return output


def _max_t_interval(
    *,
    point: np.ndarray,
    centered_bootstrap_means: np.ndarray,
    bootstrap_standard_error: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    nonzero = bootstrap_standard_error > 0.0
    standardized = np.zeros_like(centered_bootstrap_means)
    if np.any(nonzero):
        standardized[:, nonzero] = (
            centered_bootstrap_means[:, nonzero] / bootstrap_standard_error[nonzero]
        )
    if np.any(~nonzero):
        zero_scale_deviations = centered_bootstrap_means[:, ~nonzero]
        tolerance = np.finfo(float).eps * np.maximum(
            1.0,
            np.abs(point[~nonzero]),
        )
        if np.any(np.abs(zero_scale_deviations) > tolerance):
            raise RuntimeError("zero bootstrap standard error with nonzero bootstrap deviation")
    max_abs_t = np.max(np.abs(standardized), axis=1)
    critical_value = float(np.quantile(max_abs_t, 1.0 - alpha))
    half_width = critical_value * bootstrap_standard_error
    return {
        "method": "paired_cluster_max_t",
        "studentization": "bootstrap_standard_error",
        "familywise_confidence_level": 1.0 - alpha,
        "critical_value": critical_value,
        "ci_low": [float(value) for value in point - half_width],
        "ci_high": [float(value) for value in point + half_width],
    }


def _finite_matrix(values: list[list[float]] | np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values_by_pair must be a numeric 2D matrix") from exc
    if array.ndim != 2:
        raise ValueError("values_by_pair must have shape [n_pairs, n_estimands]")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("values_by_pair must have non-empty pair and estimand axes")
    if not np.all(np.isfinite(array)):
        raise ValueError("values_by_pair must contain only finite values")
    return array


def _paired_indices(indices: np.ndarray, *, n_clusters: int) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 2:
        raise ValueError("indices must have shape [n_resamples, n_pairs]")
    if array.shape[0] < 2:
        raise ValueError("indices must contain at least two bootstrap resamples")
    if array.shape[1] != n_clusters:
        raise ValueError(
            f"indices second axis must equal n_pairs={n_clusters}, got {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError("indices must use an integer dtype")
    if np.any(array < 0) or np.any(array >= n_clusters):
        raise ValueError(f"indices must be in [0, {n_clusters})")
    return np.ascontiguousarray(array, dtype=np.int64)


def _strict_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _strict_resample_count(value: int) -> int:
    normalized = _strict_positive_int(value, name="n_resamples")
    if normalized < 2:
        raise ValueError("n_resamples must be at least 2")
    return normalized


def _strict_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a non-negative integer")
    normalized = int(seed)
    if normalized < 0:
        raise ValueError("seed must be a non-negative integer")
    return normalized


def _strict_alpha(alpha: float) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError("alpha must be a finite number between 0 and 1")
    normalized = float(alpha)
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError("alpha must be a finite number between 0 and 1")
    return normalized


def _indices_sha256(indices: np.ndarray) -> str:
    canonical = np.ascontiguousarray(indices, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(f"{canonical.shape[0]}x{canonical.shape[1]}\n".encode())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()
