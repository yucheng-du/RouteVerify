"""Patch helpers with strict shape checks."""

from __future__ import annotations

from typing import cast

import numpy as np

from routeverify.instrumentation.sites import PatchBundle, site_key


def replace_slice(
    receiver: np.ndarray,
    donor: np.ndarray,
    *,
    token_position: int,
    strict_shape: bool = True,
) -> np.ndarray:
    """Return a copy of receiver with one sequence-position slice replaced."""

    if receiver.ndim < 2:
        raise ValueError("receiver must include batch and sequence dimensions")
    if token_position < 0 or token_position >= receiver.shape[1]:
        raise IndexError("token_position out of range")
    target_slice = receiver[:, token_position, ...]
    if strict_shape and target_slice.shape != donor.shape:
        raise ValueError(
            f"donor shape {donor.shape} does not match target slice {target_slice.shape}"
        )
    patched = receiver.copy()
    patched[:, token_position, ...] = donor
    return patched


def max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def apply_numpy_patch_bundle(
    receivers_by_site: dict[str, np.ndarray],
    bundle: PatchBundle,
) -> dict[str, np.ndarray]:
    """Pure-NumPy reference composition used by CPU-only bundle tests."""

    patched = {key: value.copy() for key, value in receivers_by_site.items()}
    for patch in bundle.patches:
        key = site_key(patch.site)
        if key not in patched:
            raise ValueError(f"missing receiver array for patch site {key}")
        patched[key] = replace_slice(
            patched[key],
            np.asarray(patch.replacement),
            token_position=patch.token_position,
            strict_shape=patch.strict_shape,
        )
    return patched


def interpolate_replacement(receiver: object, donor: object, *, alpha: float) -> np.ndarray:
    """Return receiver→donor interpolation/extrapolation at ``alpha``."""

    receiver_array = np.asarray(receiver)
    donor_array = np.asarray(donor)
    if receiver_array.shape != donor_array.shape:
        raise ValueError(
            f"receiver shape {receiver_array.shape} does not match donor shape {donor_array.shape}"
        )
    if not np.issubdtype(receiver_array.dtype, np.number) or not np.issubdtype(
        donor_array.dtype, np.number
    ):
        raise TypeError("receiver and donor must be numeric arrays")
    return np.asarray(
        receiver_array.astype(np.float64)
        + float(alpha) * (donor_array.astype(np.float64) - receiver_array.astype(np.float64)),
        dtype=np.result_type(receiver_array.dtype, donor_array.dtype, np.float32),
    )


def softmax_probabilities(logits: object) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("router logits must have at least two experts")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return cast(
        np.ndarray, probabilities / probabilities.sum(axis=-1, keepdims=True)
    )


def router_manipulation_metrics(
    receiver_logits: object,
    patched_logits: object,
    *,
    receiver_topk_ids: object,
    patched_topk_ids: object,
) -> dict[str, float]:
    """Measure router change using the gate's actual old/new ``torch.topk`` IDs."""

    receiver = np.asarray(receiver_logits, dtype=np.float64)
    patched = np.asarray(patched_logits, dtype=np.float64)
    if receiver.shape != patched.shape:
        raise ValueError(f"router-logit shape mismatch: {receiver.shape} vs {patched.shape}")
    receiver_ids = np.asarray(receiver_topk_ids)
    patched_ids = np.asarray(patched_topk_ids)
    if receiver_ids.shape != patched_ids.shape:
        raise ValueError(
            f"router-id shape mismatch: {receiver_ids.shape} vs {patched_ids.shape}"
        )
    if receiver_ids.ndim < 1 or receiver_ids.shape[-1] <= 0:
        raise ValueError("router IDs must include a non-empty top-k dimension")
    if receiver_ids.shape[:-1] != receiver.shape[:-1]:
        raise ValueError(
            "router IDs and logits must have identical leading dimensions: "
            f"{receiver_ids.shape[:-1]} vs {receiver.shape[:-1]}"
        )
    top_k = receiver_ids.shape[-1]
    receiver_sets = receiver_ids.reshape(-1, top_k)
    patched_sets = patched_ids.reshape(-1, top_k)
    overlaps = np.asarray(
        [
            len(set(left.tolist()) & set(right.tolist())) / top_k
            for left, right in zip(receiver_sets, patched_sets, strict=True)
        ],
        dtype=np.float64,
    )
    receiver_probs = softmax_probabilities(receiver)
    patched_probs = softmax_probabilities(patched)
    epsilon = np.finfo(np.float64).tiny
    kl = np.sum(
        receiver_probs
        * (
            np.log(np.maximum(receiver_probs, epsilon)) - np.log(np.maximum(patched_probs, epsilon))
        ),
        axis=-1,
    )
    return {
        "topk_changed_rate": float(np.mean(overlaps < 1.0)),
        "topk_overlap": float(np.mean(overlaps)),
        "router_kl_receiver_to_patched": float(np.mean(kl)),
    }


def forced_reroute_logits(receiver_logits: object, *, top_k: int) -> np.ndarray:
    """Pure-NumPy reference for the smoke-only forced-reroute construction.

    The real model hook uses the gate-returned IDs and constructs the replacement in
    the router's target dtype. This helper mirrors the construction for CPU tests.
    """

    receiver = np.asarray(receiver_logits)
    if receiver.ndim < 1 or top_k <= 0 or top_k >= receiver.shape[-1]:
        raise ValueError("forced reroute requires 0 < top_k < num_experts")
    dtype = receiver.dtype if np.issubdtype(receiver.dtype, np.floating) else np.dtype("float32")
    flat = receiver.astype(dtype, copy=True).reshape(-1, receiver.shape[-1])
    forced = flat.copy()
    low = np.finfo(dtype).min
    high = np.finfo(dtype).max
    for index, row in enumerate(flat):
        # Stable sorting is only used to build the pure reference. The model path
        # never infers IDs with NumPy: it receives the gate's exact `_router_ids`.
        selected = np.argsort(row, kind="stable")[-top_k:]
        selected_mask = np.zeros(row.shape, dtype=bool)
        selected_mask[selected] = True
        forced[index, selected_mask] = low
        forced[index, ~selected_mask] = high
    return forced.reshape(receiver.shape).astype(dtype, copy=False)
