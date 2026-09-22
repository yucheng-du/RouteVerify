"""PRMBench paired subset normalization and auditing."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routeverify.data.schema import TraceExample, normalize_text, stable_hash

PRMBENCH_ID = "hitsmy/PRMBench_Preview"
PRMBENCH_REVISION = "5cc7683d0ae5797f84d7aeac0607966f277c39e1"
PRMBENCH_RESOLVED_DATA_REVISION = "59abf58d51100c6befccf45fe8114d995fcc3b8e"
PRMBENCH_RAW_SHA256 = "3efe79ff688751369da6161635c247318611032071ee46cde6a20a95125bc966"
PRMBENCH_SOURCE_FILENAME = "prmbench_preview.jsonl"
_INT_RE = re.compile(r"-?\d+")
_HF_RESOLVED_REVISION_RE = re.compile(r"@([0-9a-f]{40})(?:/|$)")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PairEligibility:
    eligible: bool
    reason: str
    index_base: str | None = None


@dataclass(frozen=True)
class PRMBenchPair:
    clean: TraceExample
    corrupt: TraceExample
    eligibility: PairEligibility


@dataclass(frozen=True)
class PRMBenchLoadResult:
    pairs: list[PRMBenchPair]
    source_provenance: dict[str, Any]
    pair_audit: dict[str, Any]
    selection_manifest: dict[str, Any]


def _split_steps(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if len(lines) <= 1:
            pieces = re.split(r"(?:^|\n|\s)(?:Step\s*)?\d+[\).\:]\s+", value)
            lines = [piece.strip() for piece in pieces if piece.strip()]
    elif isinstance(value, Iterable):
        lines = [str(item).strip() for item in value if str(item).strip()]
    else:
        lines = []
    return tuple(lines)


def _parse_error_steps(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "[]", "null", "nan"}:
            return ()
        return tuple(int(match.group(0)) for match in _INT_RE.finditer(value))
    if isinstance(value, Iterable):
        parsed: list[int] = []
        for item in value:
            if isinstance(item, int):
                parsed.append(item)
            elif isinstance(item, str) and item.strip():
                parsed.extend(int(match.group(0)) for match in _INT_RE.finditer(item))
        return tuple(parsed)
    return ()


def normalize_error_steps(
    raw_error_steps: Any,
    *,
    n_steps: int,
    preferred_base: int | None = None,
) -> tuple[tuple[int, ...], str]:
    """Normalize PRMBench error indices to zero-based, preserving base metadata."""

    raw = _parse_error_steps(raw_error_steps)
    if not raw:
        return (), "empty"
    if any(index < 0 for index in raw):
        raise ValueError(f"negative PRMBench error index: {raw}")

    if preferred_base == 0:
        normalized = raw
        base = "zero"
    elif preferred_base == 1:
        normalized = tuple(index - 1 for index in raw)
        base = "one"
    elif 0 in raw:
        normalized = raw
        base = "zero"
    elif any(index == n_steps for index in raw):
        normalized = tuple(index - 1 for index in raw)
        base = "one"
    else:
        # PRMBench examples commonly use human-readable one-based step numbers.
        normalized = tuple(index - 1 for index in raw)
        base = "one_ambiguous"

    if any(index < 0 or index >= n_steps for index in normalized):
        raise ValueError(
            f"normalized error indices {normalized} out of bounds for {n_steps} steps; raw={raw}"
        )
    return tuple(sorted(set(normalized))), base


def eligible_pair_from_prmbench_record(
    record: dict[str, Any],
    *,
    subset: str = "preview",
    preferred_index_base: int | None = None,
) -> PRMBenchPair | None:
    """Return a clean/corrupt pair when a raw PRMBench row passes MVP filters."""

    original_question = str(record.get("original_question") or record.get("question") or "").strip()
    modified_question = str(record.get("modified_question") or record.get("question") or "").strip()
    if not original_question or not modified_question:
        return None
    if normalize_text(original_question) != normalize_text(modified_question):
        return None

    clean_steps = _split_steps(record.get("original_process"))
    corrupt_steps = _split_steps(record.get("modified_process") or record.get("modified_steps"))
    if not clean_steps or not corrupt_steps or len(clean_steps) != len(corrupt_steps):
        return None

    try:
        error_steps, index_base = normalize_error_steps(
            record.get("error_steps"),
            n_steps=len(corrupt_steps),
            preferred_base=preferred_index_base,
        )
    except ValueError:
        return None
    if not error_steps:
        return None

    first_error = min(error_steps)
    if clean_steps[:first_error] != corrupt_steps[:first_error]:
        return None
    if clean_steps[first_error] == corrupt_steps[first_error]:
        return None

    source_idx = str(record.get("idx") or stable_hash(record))
    pair_id = f"prmbench/{subset}/{source_idx}"
    common_metadata = {
        "source_idx": source_idx,
        "classification": record.get("classification"),
        "reason": record.get("reason"),
        "index_base": index_base,
        "raw_error_steps": record.get("error_steps"),
    }
    clean = TraceExample(
        sample_id=f"{pair_id}/clean",
        dataset="prmbench",
        subset=subset,
        problem=original_question,
        steps=clean_steps,
        first_error=None,
        error_steps=(),
        is_all_correct=True,
        pair_id=pair_id,
        variant="clean",
        error_type=None if record.get("classification") is None else str(record.get("classification")),
        metadata=common_metadata,
    )
    corrupt = TraceExample(
        sample_id=f"{pair_id}/corrupt",
        dataset="prmbench",
        subset=subset,
        problem=modified_question,
        steps=corrupt_steps,
        first_error=first_error,
        error_steps=error_steps,
        is_all_correct=False,
        pair_id=pair_id,
        variant="corrupt",
        error_type=None if record.get("classification") is None else str(record.get("classification")),
        metadata=common_metadata,
    )
    return PRMBenchPair(
        clean=clean,
        corrupt=corrupt,
        eligibility=PairEligibility(True, "eligible", index_base=index_base),
    )


def load_prmbench_pairs(
    *,
    hf_id: str = PRMBENCH_ID,
    split: str = "train",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    expected_resolved_data_revision: str | None = None,
    expected_raw_sha256: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    include_ambiguous_index: bool = False,
) -> list[PRMBenchPair]:
    """Load a deterministic pair selection from Hugging Face.

    ``limit`` is always interpreted in clean/corrupt pairs, never individual
    variants. Use :func:`load_prmbench_pairs_with_provenance` when the source
    and selection manifest also need to be persisted.
    """

    return load_prmbench_pairs_with_provenance(
        hf_id=hf_id,
        split=split,
        cache_dir=cache_dir,
        revision=revision,
        expected_resolved_data_revision=expected_resolved_data_revision,
        expected_raw_sha256=expected_raw_sha256,
        limit=limit,
        seed=seed,
        include_ambiguous_index=include_ambiguous_index,
    ).pairs


def load_prmbench_pairs_with_provenance(
    *,
    hf_id: str = PRMBENCH_ID,
    split: str = "train",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    expected_resolved_data_revision: str | None = None,
    expected_raw_sha256: str | None = None,
    limit: int | None = None,
    seed: int = 0,
    include_ambiguous_index: bool = False,
) -> PRMBenchLoadResult:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Install data dependencies with `uv sync --extra data`.") from exc

    if limit is not None and limit < 0:
        raise ValueError("PRMBench pair limit must be non-negative")
    if expected_resolved_data_revision is not None and revision is None:
        raise ValueError("PRMBench revision must be pinned when a resolved revision is expected")

    dataset = load_dataset(
        hf_id,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
    )
    source_provenance = _dataset_source_provenance(
        dataset,
        hf_id=hf_id,
        split=split,
        cache_dir=cache_dir,
        requested_revision=revision,
        expected_resolved_data_revision=expected_resolved_data_revision,
        expected_raw_sha256=expected_raw_sha256,
    )
    candidate_pairs: list[PRMBenchPair] = []
    audit_counts: Counter[str] = Counter()
    for record in dataset:
        pair = eligible_pair_from_prmbench_record(dict(record))
        if pair is None:
            audit_counts["ineligible_record"] += 1
            continue
        if not include_ambiguous_index and pair.eligibility.index_base == "one_ambiguous":
            audit_counts["ambiguous_index_base"] += 1
            continue
        candidate_pairs.append(pair)
        audit_counts["eligible_pair"] += 1

    complete_pairs, pair_audit = audit_prmbench_example_pairs(
        example
        for pair in candidate_pairs
        for example in (pair.clean, pair.corrupt)
    )
    pair_audit["record_counts"] = dict(sorted(audit_counts.items()))
    pair_audit["source_record_count"] = int(sum(audit_counts.values()))
    selected = select_prmbench_pairs(complete_pairs, limit=limit, seed=seed)
    selection_manifest = prmbench_selection_manifest(
        selected,
        available_pair_count=len(complete_pairs),
        limit=limit,
        seed=seed,
        source=source_provenance,
    )
    return PRMBenchLoadResult(
        pairs=selected,
        source_provenance=source_provenance,
        pair_audit=pair_audit,
        selection_manifest=selection_manifest,
    )


def audit_prmbench_example_pairs(
    examples: Iterable[TraceExample],
) -> tuple[list[PRMBenchPair], dict[str, Any]]:
    """Build only complete, unique clean/corrupt pairs and audit exclusions."""

    materialized = list(examples)
    grouped: dict[str, list[TraceExample]] = defaultdict(list)
    unpaired_rows: list[TraceExample] = []
    for example in materialized:
        if example.pair_id is None:
            unpaired_rows.append(example)
            continue
        grouped[example.pair_id].append(example)

    complete: list[PRMBenchPair] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for pair_id, rows in sorted(grouped.items()):
        variant_counts = Counter(row.variant for row in rows)
        reasons: list[str] = []
        if variant_counts.get("clean", 0) == 0 or variant_counts.get("corrupt", 0) == 0:
            reasons.append("missing_variant")
        if variant_counts.get("clean", 0) > 1 or variant_counts.get("corrupt", 0) > 1:
            reasons.append("duplicate_variant")
        unexpected = sorted(set(variant_counts) - {"clean", "corrupt"})
        if unexpected:
            reasons.append("unexpected_variant")
        if len(rows) != 2:
            reasons.append("row_count_not_two")
        if len({row.group_id for row in rows}) != 1:
            reasons.append("group_id_mismatch")
        if reasons:
            unique_reasons = sorted(set(reasons))
            reason_counts.update(unique_reasons)
            excluded.append(
                {
                    "pair_id": pair_id,
                    "reasons": unique_reasons,
                    "variant_counts": dict(sorted(variant_counts.items())),
                    "sample_ids": sorted(row.sample_id for row in rows),
                }
            )
            continue

        clean = next(row for row in rows if row.variant == "clean")
        corrupt = next(row for row in rows if row.variant == "corrupt")
        complete.append(
            PRMBenchPair(
                clean=clean,
                corrupt=corrupt,
                eligibility=PairEligibility(
                    True,
                    "complete_clean_corrupt_pair",
                    index_base=(
                        None
                        if corrupt.metadata.get("index_base") is None
                        else str(corrupt.metadata.get("index_base"))
                    ),
                ),
            )
        )

    if unpaired_rows:
        reason_counts["missing_pair_id"] += len(unpaired_rows)
        excluded.extend(
            {
                "pair_id": None,
                "reasons": ["missing_pair_id"],
                "variant_counts": {row.variant: 1},
                "sample_ids": [row.sample_id],
            }
            for row in sorted(unpaired_rows, key=lambda item: item.sample_id)
        )

    return complete, {
        "input_row_count": len(materialized),
        "candidate_pair_count": len(grouped),
        "complete_pair_count": len(complete),
        "excluded_pair_count": len(excluded),
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "excluded_pairs": excluded,
    }


def select_prmbench_pairs(
    pairs: Iterable[PRMBenchPair],
    *,
    limit: int | None,
    seed: int,
) -> list[PRMBenchPair]:
    """Select pairs by a deterministic seeded hash of pair id."""

    if limit is not None and limit < 0:
        raise ValueError("PRMBench pair limit must be non-negative")
    ranked = sorted(
        pairs,
        key=lambda pair: stable_hash(
            {"seed": seed, "pair_id": pair.clean.pair_id or pair.clean.sample_id},
            length=64,
        ),
    )
    return ranked if limit is None else ranked[:limit]


def prmbench_source_problem_id(pair: PRMBenchPair) -> str:
    """Return the source-problem cluster shared across PRMBench error variants.

    PRMBench preview IDs are ``<error_type>_prm_<source_suffix>``.  Several
    error variants can therefore originate from the same underlying source
    problem even though each variant has a distinct clean/corrupt ``pair_id``.
    Fail closed if the pinned dataset no longer follows that contract instead
    of silently falling back to pair-level grouping.
    """

    clean_source = pair.clean.metadata.get("source_idx")
    corrupt_source = pair.corrupt.metadata.get("source_idx")
    if clean_source is None or corrupt_source is None:
        raise ValueError("PRMBench source-problem grouping requires metadata.source_idx")
    if str(clean_source) != str(corrupt_source):
        raise ValueError("PRMBench clean/corrupt source_idx values differ")
    if pair.clean.subset != pair.corrupt.subset:
        raise ValueError("PRMBench clean/corrupt subsets differ")

    clean_error_type = pair.clean.error_type
    corrupt_error_type = pair.corrupt.error_type
    if clean_error_type is None or corrupt_error_type is None:
        raise ValueError("PRMBench source-problem grouping requires error_type")
    if clean_error_type != corrupt_error_type:
        raise ValueError("PRMBench clean/corrupt error types differ")

    prefix = f"{clean_error_type}_prm_"
    source_idx = str(clean_source)
    if not source_idx.startswith(prefix):
        raise ValueError(
            "PRMBench source_idx does not match '<error_type>_prm_<source_suffix>': "
            f"{source_idx!r} for error_type={clean_error_type!r}"
        )
    source_suffix = source_idx[len(prefix) :]
    if not source_suffix:
        raise ValueError("PRMBench source_idx has an empty source-problem suffix")
    return f"prmbench:{pair.clean.subset}:source_problem:{source_suffix}"


def prmbench_source_problem_components(
    pairs: Iterable[PRMBenchPair],
) -> dict[str, str]:
    """Cluster pairs by source suffix or identical normalized problem text.

    PRMBench occasionally repeats the same problem under different source
    suffixes.  The transitive closure of both relations keeps those duplicates,
    as well as the ordinary cross-error variants, in one analysis group.
    """

    materialized = list(pairs)
    pair_by_group: dict[str, PRMBenchPair] = {}
    for pair in materialized:
        group_id = pair.clean.group_id
        if group_id != pair.corrupt.group_id:
            raise ValueError("PRMBench clean/corrupt pair group IDs differ")
        if group_id in pair_by_group:
            raise ValueError(f"duplicate PRMBench pair group: {group_id}")
        if pair.clean.dataset != pair.corrupt.dataset:
            raise ValueError("PRMBench clean/corrupt datasets differ")
        if normalize_text(pair.clean.problem) != normalize_text(pair.corrupt.problem):
            raise ValueError("PRMBench clean/corrupt normalized problems differ")
        pair_by_group[group_id] = pair

    parents = {group_id: group_id for group_id in pair_by_group}

    def find(group_id: str) -> str:
        while parents[group_id] != group_id:
            parents[group_id] = parents[parents[group_id]]
            group_id = parents[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        parents[upper] = lower

    suffix_groups: dict[str, list[str]] = defaultdict(list)
    problem_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id, pair in pair_by_group.items():
        suffix_groups[prmbench_source_problem_id(pair)].append(group_id)
        problem_key = (
            pair.clean.dataset,
            pair.clean.subset,
            normalize_text(pair.clean.problem),
        )
        problem_groups[problem_key].append(group_id)

    for grouped_ids in (*suffix_groups.values(), *problem_groups.values()):
        first = grouped_ids[0]
        for group_id in grouped_ids[1:]:
            union(first, group_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(pair_by_group):
        members_by_root[find(group_id)].append(group_id)

    result: dict[str, str] = {}
    for members in members_by_root.values():
        first_pair = pair_by_group[members[0]]
        component_hash = stable_hash(sorted(members), length=64)
        component_id = (
            f"{first_pair.clean.dataset}:{first_pair.clean.subset}:"
            f"source_problem_component:{component_hash}"
        )
        for group_id in members:
            result[group_id] = component_id
    return result


def prmbench_selection_manifest(
    pairs: Iterable[PRMBenchPair],
    *,
    available_pair_count: int,
    limit: int | None,
    seed: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a canonical, self-hashing manifest for a selected pair set."""

    rows = [_pair_manifest_row(pair) for pair in pairs]
    processed_file = source.get("processed_file")
    split_manifest = source.get("split_manifest")
    prepared_selection_manifest = source.get("prepared_selection_manifest")
    source_manifest = {
        "source_type": source.get("source_type"),
        "dataset": source.get("dataset"),
        "hf_id": source.get("hf_id") or source.get("configured_hf_id"),
        "split": source.get("split"),
        "requested_revision": source.get("requested_revision") or source.get("revision"),
        "resolved_data_revision": source.get("resolved_data_revision")
        or source.get("expected_resolved_data_revision"),
        "raw_sha256": source.get("raw_sha256") or source.get("expected_raw_sha256"),
        "processed_file_sha256": (
            processed_file.get("sha256") if isinstance(processed_file, Mapping) else None
        ),
        "split_manifest_sha256": (
            split_manifest.get("sha256") if isinstance(split_manifest, Mapping) else None
        ),
        "prepared_selection_manifest_hash": (
            prepared_selection_manifest.get("manifest_hash")
            if isinstance(prepared_selection_manifest, Mapping)
            else None
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "prmbench",
        "selection_unit": "clean_corrupt_pair",
        "selection_rule": {
            "name": "ascending_sha256_seed_pair_id",
            "hash_payload": ["seed", "pair_id"],
            "limit_semantics": "pair_count",
        },
        "seed": seed,
        "requested_pair_limit": limit,
        "available_pair_count": available_pair_count,
        "selected_pair_count": len(rows),
        "selected_example_count": len(rows) * 2,
        "source": source_manifest,
        "pairs": rows,
        "pairs_hash": stable_hash(rows, length=64),
    }
    payload["manifest_hash"] = stable_hash(payload, length=64)
    return payload


def _pair_manifest_row(pair: PRMBenchPair) -> dict[str, Any]:
    pair_id = pair.clean.pair_id or pair.corrupt.pair_id or pair.clean.sample_id
    source_idx = pair.clean.metadata.get("source_idx") or pair.corrupt.metadata.get("source_idx")
    return {
        "pair_id": pair_id,
        "group_id": pair.clean.group_id,
        "source_id": None if source_idx is None else str(source_idx),
        "clean_sample_id": pair.clean.sample_id,
        "corrupt_sample_id": pair.corrupt.sample_id,
        "clean_content_hash": pair.clean.content_hash,
        "corrupt_content_hash": pair.corrupt.content_hash,
    }


def _dataset_source_provenance(
    dataset: Any,
    *,
    hf_id: str,
    split: str,
    cache_dir: str | Path | None,
    requested_revision: str | None,
    expected_resolved_data_revision: str | None,
    expected_raw_sha256: str | None,
) -> dict[str, Any]:
    info = getattr(dataset, "info", None)
    download_checksums = getattr(info, "download_checksums", None) or {}
    source_files: list[dict[str, Any]] = []
    resolved_revisions: set[str] = set()
    observed_raw_sha256: str | None = None
    if not download_checksums and expected_resolved_data_revision is not None:
        if requested_revision is None:
            raise ValueError(
                "PRMBench revision must be pinned when a resolved revision is expected"
            )
        source_file = _verified_prmbench_hf_source_file(
            hf_id=hf_id,
            filename=PRMBENCH_SOURCE_FILENAME,
            requested_revision=requested_revision,
            registered_revision=expected_resolved_data_revision,
            expected_raw_sha256=expected_raw_sha256,
        )
        source_files.append(source_file)
        resolved_revisions.add(str(source_file["resolved_data_revision"]))
        observed_raw_sha256 = str(source_file["cached_sha256"])
    for uri, metadata in sorted(download_checksums.items()):
        match = _HF_RESOLVED_REVISION_RE.search(str(uri))
        resolved_revision = None if match is None else match.group(1)
        if resolved_revision is not None:
            resolved_revisions.add(resolved_revision)
        filename = str(uri).rsplit("/", 1)[-1]
        checksum = metadata.get("checksum") if isinstance(metadata, Mapping) else None
        cached_sha256 = _cached_hf_source_sha256(
            hf_id=hf_id,
            filename=filename,
            revision=requested_revision,
        )
        if cached_sha256 is None and isinstance(checksum, str) and len(checksum) == 64:
            cached_sha256 = checksum
        if cached_sha256 is not None:
            observed_raw_sha256 = cached_sha256
        source_files.append(
            {
                "uri": str(uri),
                "filename": filename,
                "resolved_data_revision": resolved_revision,
                "num_bytes": (
                    metadata.get("num_bytes") if isinstance(metadata, Mapping) else None
                ),
                "reported_checksum": checksum,
                "cached_sha256": cached_sha256,
            }
        )

    if expected_resolved_data_revision is not None:
        if resolved_revisions != {expected_resolved_data_revision}:
            raise ValueError(
                "PRMBench resolved data revision mismatch: "
                f"expected {expected_resolved_data_revision}, observed {sorted(resolved_revisions)}"
            )
    if (
        expected_raw_sha256 is not None
        and observed_raw_sha256 is not None
        and observed_raw_sha256 != expected_raw_sha256
    ):
        raise ValueError(
            "PRMBench raw source SHA256 mismatch: "
            f"expected {expected_raw_sha256}, observed {observed_raw_sha256}"
        )
    if expected_raw_sha256 is not None and observed_raw_sha256 is None:
        raise ValueError(
            "PRMBench raw source SHA256 could not be verified from the local cache; "
            f"expected {expected_raw_sha256}"
        )

    return {
        "source_type": "hf_loader",
        "dataset": "prmbench",
        "hf_id": hf_id,
        "split": split,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "requested_revision": requested_revision,
        "resolved_data_revision": (
            next(iter(resolved_revisions)) if len(resolved_revisions) == 1 else None
        ),
        "expected_resolved_data_revision": expected_resolved_data_revision,
        "raw_sha256": observed_raw_sha256 or expected_raw_sha256,
        "expected_raw_sha256": expected_raw_sha256,
        "raw_sha256_verified": observed_raw_sha256 is not None,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "source_files": source_files,
    }


def _verified_prmbench_hf_source_file(
    *,
    hf_id: str,
    filename: str,
    requested_revision: str,
    registered_revision: str,
    expected_raw_sha256: str | None,
) -> dict[str, Any]:
    """Verify datasets-5 source provenance without trusting absent builder checksums."""

    loaded = _hf_source_snapshot_metadata(
        hf_id=hf_id,
        filename=filename,
        revision=requested_revision,
    )
    registered = _hf_source_snapshot_metadata(
        hf_id=hf_id,
        filename=filename,
        revision=registered_revision,
    )
    loaded_revision = str(loaded["resolved_data_revision"])
    observed_registered_revision = str(registered["resolved_data_revision"])
    if loaded_revision != requested_revision:
        raise ValueError(
            "PRMBench requested dataset revision mismatch: "
            f"expected {requested_revision}, observed {loaded_revision}"
        )
    if observed_registered_revision != registered_revision:
        raise ValueError(
            "PRMBench resolved data revision mismatch: "
            f"expected {registered_revision}, observed {[observed_registered_revision]}"
        )

    loaded_sha256 = str(loaded["reported_checksum"])
    registered_sha256 = str(registered["reported_checksum"])
    if loaded_sha256 != registered_sha256:
        raise ValueError(
            "PRMBench immutable source snapshots differ: "
            f"requested {requested_revision} has {loaded_sha256}, "
            f"registered {registered_revision} has {registered_sha256}"
        )
    if expected_raw_sha256 is not None and loaded_sha256 != expected_raw_sha256:
        raise ValueError(
            "PRMBench raw source SHA256 mismatch: "
            f"expected {expected_raw_sha256}, observed {loaded_sha256}"
        )

    cached_sha256 = _cached_hf_source_sha256(
        hf_id=hf_id,
        filename=filename,
        revision=requested_revision,
    )
    if cached_sha256 is None:
        raise ValueError(
            "PRMBench raw source SHA256 could not be verified from the local cache; "
            f"expected {expected_raw_sha256 or loaded_sha256}"
        )
    if cached_sha256 != loaded_sha256:
        raise ValueError(
            "PRMBench raw source SHA256 mismatch: "
            f"expected {loaded_sha256}, observed {cached_sha256}"
        )

    return {
        "uri": registered["uri"],
        "loaded_uri": loaded["uri"],
        "filename": filename,
        "resolved_data_revision": observed_registered_revision,
        "loaded_revision": loaded_revision,
        "num_bytes": registered["num_bytes"],
        "reported_checksum": registered_sha256,
        "loaded_reported_checksum": loaded_sha256,
        "cached_sha256": cached_sha256,
        "verification_strategy": "immutable_snapshot_content_equivalence",
    }


def _hf_source_snapshot_metadata(
    *,
    hf_id: str,
    filename: str,
    revision: str,
) -> dict[str, Any]:
    """Resolve one immutable Hub snapshot and return its content identity."""

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        metadata = get_hf_file_metadata(
            hf_hub_url(
                repo_id=hf_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
        )
    except Exception as exc:  # pragma: no cover - depends on the external Hub
        raise ValueError(
            f"PRMBench source metadata could not be resolved at revision {revision}: {filename}"
        ) from exc

    resolved_revision = metadata.commit_hash
    checksum = metadata.etag
    if (
        not isinstance(resolved_revision, str)
        or _COMMIT_SHA_RE.fullmatch(resolved_revision) is None
    ):
        raise ValueError(
            "PRMBench source metadata did not resolve to an immutable revision: "
            f"{resolved_revision!r}"
        )
    if not isinstance(checksum, str) or _SHA256_RE.fullmatch(checksum) is None:
        raise ValueError(f"PRMBench source metadata did not expose a raw SHA256: {checksum!r}")
    return {
        "uri": f"hf://datasets/{hf_id}@{resolved_revision}/{filename}",
        "filename": filename,
        "resolved_data_revision": resolved_revision,
        "num_bytes": metadata.size,
        "reported_checksum": checksum,
    }


def _cached_hf_source_sha256(
    *,
    hf_id: str,
    filename: str,
    revision: str | None,
) -> str | None:
    """Hash an already-cached Hub source file without permitting a download."""

    try:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=hf_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception:  # pragma: no cover - depends on the external HF cache layout
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
