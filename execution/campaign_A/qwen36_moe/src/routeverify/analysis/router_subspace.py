"""CPU row-space/null-space SVD analysis for exported router matrices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def row_null_space_projection(
    matrix: object,
    direction: object,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> dict[str, Any]:
    weights = np.asarray(matrix, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if weights.ndim != 2:
        raise ValueError("router weight must be a 2D matrix")
    if vector.shape[0] != weights.shape[1]:
        raise ValueError(
            f"direction dimension {vector.shape[0]} does not match hidden size {weights.shape[1]}"
        )
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(vector)):
        raise ValueError("matrix and direction must be finite")
    _u, singular_values, vt = np.linalg.svd(weights, full_matrices=False)
    default_rtol = np.finfo(np.float64).eps * max(weights.shape)
    resolved_rtol = default_rtol if rtol is None else float(rtol)
    threshold = max(
        float(atol), resolved_rtol * float(singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.sum(singular_values > threshold))
    basis = vt[:rank, :]
    row_projection = basis.T @ (basis @ vector) if rank else np.zeros_like(vector)
    null_projection = vector - row_projection
    direction_norm = float(np.linalg.norm(vector))
    row_norm = float(np.linalg.norm(row_projection))
    null_norm = float(np.linalg.norm(null_projection))
    denominator = direction_norm**2
    return {
        "shape": list(weights.shape),
        "rank": rank,
        "rank_threshold": threshold,
        "rtol": resolved_rtol,
        "atol": float(atol),
        "singular_values": singular_values.tolist(),
        "direction_norm": direction_norm,
        "row_projection_norm": row_norm,
        "null_projection_norm": null_norm,
        "row_energy_fraction": None if denominator == 0.0 else row_norm**2 / denominator,
        "null_energy_fraction": None if denominator == 0.0 else null_norm**2 / denominator,
        "reconstruction_error": float(np.linalg.norm(vector - row_projection - null_projection)),
        "orthogonality_error": float(abs(np.dot(row_projection, null_projection))),
    }


def analyze_router_subspaces(
    matrices: Mapping[int, np.ndarray],
    directions: Mapping[int, np.ndarray],
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> dict[str, Any]:
    missing = sorted(set(matrices) - set(directions))
    if missing:
        raise ValueError(f"probe direction artifact is missing layers: {missing}")
    layers = {
        f"layer_{layer}": row_null_space_projection(
            matrix,
            directions[layer],
            rtol=rtol,
            atol=atol,
        )
        for layer, matrix in sorted(matrices.items())
    }
    return {"schema_version": 1, "layers": layers}
