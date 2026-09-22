"""Expert-ID-aware feature construction for routing probes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

EXPERT_ID_MEMBERSHIP_COUNT = "expert_id_membership_count"
EXPERT_ID_ALIGNED_SPARSE_WEIGHTS = "expert_id_aligned_sparse_weights"
RANK_ONLY_SORTED_WEIGHTS_LEGACY = "rank_only_sorted_weights_legacy"

ROUTING_PROBE_FEATURE_NAMES = (
    EXPERT_ID_MEMBERSHIP_COUNT,
    EXPERT_ID_ALIGNED_SPARSE_WEIGHTS,
    RANK_ONLY_SORTED_WEIGHTS_LEGACY,
)


def routing_feature_vectors(
    topk_ids: Any,
    topk_weights: Any,
    *,
    num_experts: int,
) -> dict[str, np.ndarray]:
    """Construct aligned routing features for one captured token.

    Membership values are counts so the helper also supports aggregating a
    flattened multi-position capture; for a normal top-k token they are binary.
    """

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    ids_array = _to_numpy(topk_ids).reshape(-1)
    weights = np.asarray(_to_numpy(topk_weights), dtype=np.float32).reshape(-1)
    if ids_array.shape != weights.shape:
        raise ValueError("top-k ids and weights must have the same shape")
    if ids_array.size == 0:
        raise ValueError("top-k ids and weights must be non-empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("top-k weights must be finite")
    if not np.issubdtype(ids_array.dtype, np.integer):
        numeric_ids = np.asarray(ids_array, dtype=np.float64)
        if not np.all(np.equal(numeric_ids, np.floor(numeric_ids))):
            raise ValueError("top-k expert IDs must be integers")
    ids = np.asarray(ids_array, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= num_experts):
        raise ValueError("top-k expert ID is outside [0, num_experts)")

    membership = np.zeros(num_experts, dtype=np.float32)
    aligned_weights = np.zeros(num_experts, dtype=np.float32)
    np.add.at(membership, ids, 1.0)
    np.add.at(aligned_weights, ids, weights)
    rank_only_weights = np.sort(weights)[::-1].copy()
    return {
        EXPERT_ID_MEMBERSHIP_COUNT: membership,
        EXPERT_ID_ALIGNED_SPARSE_WEIGHTS: aligned_weights,
        RANK_ONLY_SORTED_WEIGHTS_LEGACY: rank_only_weights,
    }


def routing_features_from_manifest_arrays(
    row: Mapping[str, Any],
    arrays: Mapping[str, Any],
    *,
    site_prefix: str,
    num_experts: int | None = None,
) -> dict[str, np.ndarray]:
    """Construct routing features from one capture manifest row and tensor mapping."""

    ids_key = f"{site_prefix}.router_topk_ids"
    weights_key = f"{site_prefix}.router_topk_weights"
    logits_key = f"{site_prefix}.router_logits"
    if ids_key not in arrays:
        raise KeyError(f"missing routing array: {ids_key}")
    if weights_key not in arrays:
        raise KeyError(f"missing routing array: {weights_key}")

    inferred_num_experts = num_experts
    if logits_key in arrays:
        logits = _manifest_token_vector(arrays[logits_key], row)
        if inferred_num_experts is None:
            inferred_num_experts = int(logits.size)
        elif logits.size != inferred_num_experts:
            raise ValueError(
                f"configured num_experts={inferred_num_experts} but {logits_key} has "
                f"dimension {logits.size}"
            )
    if inferred_num_experts is None:
        raise ValueError("num_experts is required when router logits are unavailable")

    return routing_feature_vectors(
        _manifest_token_vector(arrays[ids_key], row),
        _manifest_token_vector(arrays[weights_key], row),
        num_experts=inferred_num_experts,
    )


def _manifest_token_vector(value: Any, row: Mapping[str, Any]) -> np.ndarray:
    array = _to_numpy(value)
    token_index = int(row.get("token_index", 0))
    num_positions = int(row.get("num_positions_in_file", 1))
    if token_index < 0:
        raise ValueError("token_index must be non-negative")

    # Capture tensors use [batch=1, positions, ...]. Remove the batch axis,
    # then select the manifest row's exact captured position.
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim >= 2 and array.shape[0] == num_positions:
        if token_index >= array.shape[0]:
            raise ValueError("token_index is outside the captured position axis")
        array = array[token_index]
    elif array.ndim >= 2 and array.shape[0] == 1 and token_index == 0:
        array = array[0]
    elif token_index != 0:
        raise ValueError("cannot locate token_index in routing array shape")
    return np.asarray(array).reshape(-1)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return cast(np.ndarray, tensor.numpy())
    return np.asarray(value)
