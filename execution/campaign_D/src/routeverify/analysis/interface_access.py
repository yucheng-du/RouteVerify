"""CPU-only interface-access analysis from archived recapture tensors.

The helpers in this module intentionally operate on recapture manifests and
``safetensors`` files only.  They never instantiate a model or load model
weights.  Supervised comparisons use one saved source-grouped fold manifest
and the same L2-logistic estimator for every representation family.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np

from routeverify.analysis.fair_comparison import (
    evaluate_dense_logistic_grouped_cv,
    paired_group_bootstrap_auroc_difference,
)
from routeverify.analysis.routing_probes import routing_feature_vectors
from routeverify.data.schema import stable_hash

POST_MOE_RESIDUAL = "post_moe_residual"
ROUTER_INPUT = "router_input"
ROUTER_LOGITS = "router_logits"
EXPERT_ID_MEMBERSHIP = "expert_id_membership_count"
EXPERT_ID_ALIGNED_WEIGHTS = "expert_id_aligned_sparse_weights"
RANK_ONLY_WEIGHTS_LEGACY = "rank_only_sorted_weights_legacy"

INTERFACE_FEATURE_FAMILIES = (
    POST_MOE_RESIDUAL,
    ROUTER_INPUT,
    ROUTER_LOGITS,
    EXPERT_ID_MEMBERSHIP,
    EXPERT_ID_ALIGNED_WEIGHTS,
    RANK_ONLY_WEIGHTS_LEGACY,
)

PRIMARY_INTERFACE_CHAIN = (
    POST_MOE_RESIDUAL,
    ROUTER_INPUT,
    ROUTER_LOGITS,
    EXPERT_ID_ALIGNED_WEIGHTS,
    EXPERT_ID_MEMBERSHIP,
)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest and reject malformed or duplicate sample rows."""

    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object: {path}")
            row = cast(dict[str, Any], value)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"manifest row {line_number} lacks sample_id: {path}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
            seen_sample_ids.add(sample_id)
            variant = str(row.get("variant", ""))
            if variant not in {"clean", "corrupt"}:
                raise ValueError(f"unsupported variant for {sample_id}: {variant}")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def load_selection_source_ids(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Map pair IDs to archived source IDs without reading source text."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError(f"invalid recapture selection manifest: {path}")
    mapping: dict[str, str] = {}
    for value in payload["pairs"]:
        if not isinstance(value, Mapping):
            raise ValueError("selection-manifest pair must be an object")
        pair_id = str(value.get("pair_id", ""))
        source_id = str(value.get("source_id", ""))
        if not pair_id or not source_id:
            raise ValueError("selection-manifest pair lacks pair_id/source_id")
        if pair_id in mapping:
            raise ValueError(f"duplicate pair in selection manifest: {pair_id}")
        mapping[pair_id] = source_id
    expected = int(payload.get("selected_pair_count", len(mapping)))
    if len(mapping) != expected:
        raise ValueError(
            f"selection pair count mismatch: manifest={expected}, observed={len(mapping)}"
        )
    return mapping, cast(dict[str, Any], payload)


def load_canonical_source_problem_groups(
    path: Path,
    *,
    allowed_pair_group_ids: Mapping[str, str],
    expected_definition: str = "connected_components_of_source_suffix_or_normalized_problem",
) -> tuple[dict[str, str], dict[str, Any]]:
    """Join an allowlisted pair set to a pre-existing canonical group manifest.

    Only pair/group identifiers are consumed.  The canonical manifest may cover
    a broader historical universe, but rows outside ``allowed_pair_group_ids``
    do not enter the returned mapping or analysis folds.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pairs"), list):
        raise ValueError(f"invalid canonical source-problem manifest: {path}")
    if payload.get("status") != "ok":
        raise ValueError(f"canonical source-problem manifest is not status=ok: {path}")
    if str(payload.get("definition", "")) != expected_definition:
        raise ValueError(
            "canonical grouping definition mismatch: "
            f"{payload.get('definition')!r} != {expected_definition!r}"
        )

    allowed_pairs = set(allowed_pair_group_ids)
    mapping: dict[str, str] = {}
    all_pair_ids: set[str] = set()
    for value in payload["pairs"]:
        if not isinstance(value, Mapping):
            raise ValueError("canonical source-problem pair must be an object")
        pair_id = str(value.get("pair_id", ""))
        if not pair_id:
            raise ValueError("canonical source-problem pair lacks pair_id")
        if pair_id in all_pair_ids:
            raise ValueError(f"duplicate pair in canonical source-problem manifest: {pair_id}")
        all_pair_ids.add(pair_id)
        if pair_id not in allowed_pairs:
            continue
        pair_group_id = str(value.get("pair_group_id", ""))
        source_problem_id = str(value.get("source_problem_id", ""))
        if not pair_group_id or not source_problem_id:
            raise ValueError(f"canonical row lacks group identifiers: {pair_id}")
        if pair_group_id != allowed_pair_group_ids[pair_id]:
            raise ValueError(
                f"canonical pair_group_id differs from selection for {pair_id}: "
                f"{pair_group_id!r} != {allowed_pair_group_ids[pair_id]!r}"
            )
        mapping[pair_id] = source_problem_id

    missing = sorted(allowed_pairs - set(mapping))
    extra = sorted(set(mapping) - allowed_pairs)
    if missing or extra:
        raise ValueError(
            "canonical source-problem join does not exactly cover the allowlist: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return mapping, cast(dict[str, Any], payload)


def source_suffix_group_id(source_id: str) -> str:
    """Return the source-suffix group available inside recapture provenance.

    PRMBench source IDs have the form ``<error_type>_prm_<source_suffix>``.
    Removing the synthetic error-type prefix keeps variants of the same source
    component together without consulting source text or held-out data.
    """

    marker = "_prm_"
    if marker not in source_id:
        raise ValueError(f"cannot parse PRMBench source ID: {source_id}")
    _error_type, suffix = source_id.split(marker, 1)
    if not suffix:
        raise ValueError(f"empty PRMBench source suffix: {source_id}")
    return f"prmbench:preview:source_suffix:{suffix}"


def recapture_source_components(selection_manifest: Mapping[str, Any]) -> dict[str, str]:
    """Build source components using only authorized selection provenance.

    Pairs are connected when they share the source suffix or the exact archived
    clean-example content hash.  The latter safely captures repeated clean
    source traces without reading source text.  It is deliberately narrower
    than a normalized-problem-text relation, which cannot be reconstructed
    from recapture artifacts alone.
    """

    values = selection_manifest.get("pairs")
    if not isinstance(values, list) or not values:
        raise ValueError("selection manifest lacks pair provenance")
    pair_rows: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("selection-manifest pair must be an object")
        pair_id = str(value.get("pair_id", ""))
        source_id = str(value.get("source_id", ""))
        clean_hash = str(value.get("clean_content_hash", ""))
        if not pair_id or not source_id or not clean_hash:
            raise ValueError("selection pair lacks pair_id/source_id/clean_content_hash")
        if pair_id in pair_rows:
            raise ValueError(f"duplicate pair in selection manifest: {pair_id}")
        pair_rows[pair_id] = value

    parent = {pair_id: pair_id for pair_id in pair_rows}

    def find(pair_id: str) -> str:
        while parent[pair_id] != pair_id:
            parent[pair_id] = parent[parent[pair_id]]
            pair_id = parent[pair_id]
        return pair_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        parent[merge] = keep

    by_relation: dict[tuple[str, str], str] = {}
    for pair_id, row in sorted(pair_rows.items()):
        relations = (
            ("source_suffix", source_suffix_group_id(str(row["source_id"]))),
            ("clean_content_hash", str(row["clean_content_hash"])),
        )
        for relation in relations:
            prior = by_relation.get(relation)
            if prior is None:
                by_relation[relation] = pair_id
            else:
                union(pair_id, prior)

    members_by_root: dict[str, list[str]] = {}
    for pair_id in pair_rows:
        members_by_root.setdefault(find(pair_id), []).append(pair_id)
    component_by_pair: dict[str, str] = {}
    for members in members_by_root.values():
        ordered = sorted(members)
        component_id = (
            "prmbench:preview:recapture_source_component:"
            + stable_hash(ordered, length=32)
        )
        for pair_id in ordered:
            component_by_pair[pair_id] = component_id
    if set(component_by_pair) != set(pair_rows):
        raise AssertionError("source-component mapping does not cover all selected pairs")
    return component_by_pair


def canonicalize_common_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    pair_source_ids: Mapping[str, str],
    pair_analysis_group_ids: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select and order exactly the reference pair set with source-safe groups."""

    allowed_pairs = set(pair_source_ids)
    selected = [dict(row) for row in rows if str(row.get("pair_id", "")) in allowed_pairs]
    by_sample_id = {str(row["sample_id"]): row for row in selected}
    expected_sample_ids = {
        f"{pair_id}/{variant}"
        for pair_id in allowed_pairs
        for variant in ("clean", "corrupt")
    }
    observed_sample_ids = set(by_sample_id)
    missing = sorted(expected_sample_ids - observed_sample_ids)
    extra = sorted(observed_sample_ids - expected_sample_ids)
    if missing or extra:
        raise ValueError(
            "recapture rows do not exactly cover the common pair set: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    ordered = [by_sample_id[sample_id] for sample_id in sorted(expected_sample_ids)]
    groups: list[str] = []
    for row in ordered:
        pair_id = str(row["pair_id"])
        sample_id = str(row["sample_id"])
        variant = str(row["variant"])
        if not sample_id.endswith(f"/{variant}"):
            raise ValueError(f"sample/variant mismatch: {sample_id} vs {variant}")
        groups.append(
            source_suffix_group_id(pair_source_ids[pair_id])
            if pair_analysis_group_ids is None
            else str(pair_analysis_group_ids[pair_id])
        )
    return ordered, groups


def assert_row_identity_matches(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require exact semantic row identity across model recaptures."""

    if len(reference_rows) != len(candidate_rows):
        raise ValueError("recapture row counts differ")
    fields = ("sample_id", "pair_id", "group_id", "variant", "boundary_step", "error_type")
    for index, (reference, candidate) in enumerate(
        zip(reference_rows, candidate_rows, strict=True)
    ):
        for field in fields:
            if reference.get(field) != candidate.get(field):
                raise ValueError(
                    f"recapture semantic identity differs at row {index}, field {field}: "
                    f"{reference.get(field)!r} != {candidate.get(field)!r}"
                )


def resolve_archived_activation(recapture_root: Path, manifest_value: Any) -> Path:
    """Resolve stale remote activation paths by their immutable basename."""

    basename = Path(str(manifest_value)).name
    if not basename.endswith(".safetensors") or basename in {"", ".safetensors"}:
        raise ValueError(f"invalid activation filename in manifest: {manifest_value!r}")
    resolved = recapture_root / "activations" / basename
    if not resolved.is_file():
        raise FileNotFoundError(f"archived activation is missing: {resolved}")
    return resolved


def extract_layer_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    recapture_root: Path,
    layer: int,
    num_experts: int,
) -> dict[str, np.ndarray]:
    """Extract every comparable interface representation for one routed layer."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Install analysis dependencies with `uv sync --extra analysis`.") from exc

    prefix = f"layer_{layer}"
    required_keys = (
        f"{prefix}.post_moe_residual",
        f"{prefix}.router_input",
        f"{prefix}.router_logits",
        f"{prefix}.router_topk_ids",
        f"{prefix}.router_topk_weights",
    )
    values: dict[str, list[np.ndarray]] = {
        family: [] for family in INTERFACE_FEATURE_FAMILIES
    }
    for row in rows:
        arrays = row.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ValueError(f"recapture row lacks array metadata: {row.get('sample_id')}")
        missing_metadata = [key for key in required_keys if key not in arrays]
        if missing_metadata:
            raise KeyError(
                f"recapture metadata lacks layer {layer} arrays for {row.get('sample_id')}: "
                f"{missing_metadata}"
            )
        activation_path = resolve_archived_activation(recapture_root, row.get("activation_file"))
        with safe_open(activation_path, framework="np") as handle:
            available = set(handle.keys())
            missing = [key for key in required_keys if key not in available]
            if missing:
                raise KeyError(f"activation {activation_path} lacks arrays: {missing}")
            residual = _manifest_token_vector(handle.get_tensor(required_keys[0]), row)
            router_input = _manifest_token_vector(handle.get_tensor(required_keys[1]), row)
            router_logits = _manifest_token_vector(handle.get_tensor(required_keys[2]), row)
            topk_ids = _manifest_token_vector(handle.get_tensor(required_keys[3]), row)
            topk_weights = _manifest_token_vector(handle.get_tensor(required_keys[4]), row)

        if router_logits.size != num_experts:
            raise ValueError(
                f"router-logit dimension {router_logits.size} != num_experts {num_experts}"
            )
        routing = routing_feature_vectors(
            topk_ids,
            topk_weights,
            num_experts=num_experts,
        )
        values[POST_MOE_RESIDUAL].append(residual.astype(np.float32, copy=False))
        values[ROUTER_INPUT].append(router_input.astype(np.float32, copy=False))
        values[ROUTER_LOGITS].append(router_logits.astype(np.float32, copy=False))
        values[EXPERT_ID_MEMBERSHIP].append(routing[EXPERT_ID_MEMBERSHIP])
        values[EXPERT_ID_ALIGNED_WEIGHTS].append(routing[EXPERT_ID_ALIGNED_WEIGHTS])
        values[RANK_ONLY_WEIGHTS_LEGACY].append(routing[RANK_ONLY_WEIGHTS_LEGACY])

    matrices = {name: np.stack(vectors) for name, vectors in values.items()}
    row_count = len(rows)
    for name, matrix in matrices.items():
        if matrix.ndim != 2 or matrix.shape[0] != row_count:
            raise AssertionError(f"invalid extracted feature matrix for {name}: {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"non-finite values in extracted feature matrix: {name}")
    return matrices


def evaluate_feature_families(
    features_by_name: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_group_ids: Sequence[str],
    split_manifest: dict[str, Any],
    analysis_group_definition: str,
    seed: int,
    n_splits: int,
    bootstrap_resamples: int,
    logistic_c: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run matched L2-logistic probes and adjacent-interface comparisons."""

    labels = np.asarray([str(row["variant"]) == "corrupt" for row in rows], dtype=bool)
    sample_ids = [str(row["sample_id"]) for row in rows]
    pair_group_ids = [str(row["group_id"]) for row in rows]
    results: dict[str, Any] = {}
    for name in INTERFACE_FEATURE_FAMILIES:
        if name not in features_by_name:
            raise KeyError(f"missing interface feature family: {name}")
        results[name] = evaluate_dense_logistic_grouped_cv(
            features_by_name[name],
            labels,
            sample_ids=sample_ids,
            pair_group_ids=pair_group_ids,
            analysis_group_ids=analysis_group_ids,
            split_manifest=split_manifest,
            analysis_group_definition=analysis_group_definition,
            seed=seed,
            n_splits=n_splits,
            bootstrap_resamples=bootstrap_resamples,
            logistic_c=logistic_c,
        )

    comparisons: list[dict[str, Any]] = []
    for left_name, right_name in pairwise(PRIMARY_INTERFACE_CHAIN):
        left_scores = _oof_scores(results[left_name], expected_sample_ids=sample_ids)
        right_scores = _oof_scores(results[right_name], expected_sample_ids=sample_ids)
        difference = paired_group_bootstrap_auroc_difference(
            labels,
            left_scores,
            right_scores,
            analysis_group_ids,
            left_name=left_name,
            right_name=right_name,
            seed=seed,
            n_resamples=bootstrap_resamples,
        )
        comparisons.append(difference)
    return results, comparisons


def compact_probe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep claim-relevant metrics while storing shared folds and OOF rows once."""

    compact_folds = []
    for fold in cast(list[Mapping[str, Any]], result["folds"]):
        compact_folds.append(
            {
                key: value
                for key, value in fold.items()
                if key
                not in {
                    "train_group_ids",
                    "eval_group_ids",
                }
            }
        )
    return {
        key: value
        for key, value in result.items()
        if key not in {"folds", "oof_predictions"}
    } | {"folds": compact_folds}


def _oof_scores(
    result: Mapping[str, Any],
    *,
    expected_sample_ids: Sequence[str],
) -> np.ndarray:
    predictions = result.get("oof_predictions")
    if not isinstance(predictions, list):
        raise ValueError("probe result lacks OOF predictions")
    score_by_sample = {
        str(row["sample_id"]): float(row["score"])
        for row in predictions
        if isinstance(row, Mapping)
    }
    if set(score_by_sample) != set(expected_sample_ids):
        raise ValueError("OOF predictions do not exactly cover expected sample IDs")
    return np.asarray([score_by_sample[sample_id] for sample_id in expected_sample_ids])


def _manifest_token_vector(value: Any, row: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(value)
    token_index = int(row.get("token_index", 0))
    num_positions = int(row.get("num_positions_in_file", 1))
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim >= 2 and array.shape[0] == num_positions:
        if token_index < 0 or token_index >= array.shape[0]:
            raise ValueError("token_index is outside the captured position axis")
        array = array[token_index]
    elif array.ndim >= 2 and array.shape[0] == 1 and token_index == 0:
        array = array[0]
    elif token_index != 0:
        raise ValueError("cannot locate token_index in activation array")
    return np.asarray(array).reshape(-1)
