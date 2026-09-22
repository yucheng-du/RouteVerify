"""Read explicitly pinned normalized discovery pairs without a remote fallback."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from routeverify.config import DataConfig
from routeverify.data.prmbench import (
    PRMBenchLoadResult,
    audit_prmbench_example_pairs,
    prmbench_selection_manifest,
    select_prmbench_pairs,
)
from routeverify.data.schema import stable_hash, trace_example_from_json


def load_frozen_prmbench(
    config: DataConfig, *, limit: int | None, seed: int
) -> PRMBenchLoadResult:
    """Verify the entire pinned file and discovery membership before subsetting."""
    extras = config.__pydantic_extra__ or {}
    path = Path(extras["local_pair_file"])
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != config.expected_processed_sha256:
        raise ValueError("frozen PRMBench normalized file SHA256 mismatch")
    split_path = Path(extras["local_pair_split_manifest"])
    split = json.loads(split_path.read_text())
    unhashed = {key: value for key, value in split.items() if key != "manifest_hash"}
    if (
        split.get("manifest_hash") != extras["local_pair_split_manifest_hash"]
        or split["manifest_hash"] != stable_hash(unhashed, length=64)
    ):
        raise ValueError("frozen PRMBench split identity mismatch")
    examples = [trace_example_from_json(json.loads(line)) for line in raw.splitlines() if line]
    pairs, audit = audit_prmbench_example_pairs(examples)
    discovery = set(split["discovery_group_ids"])
    if (
        len(examples) != 2 * len(discovery)
        or len(pairs) != len(discovery)
        or {pair.clean.group_id for pair in pairs} != discovery
        or discovery.intersection(split["validation_group_ids"])
    ):
        raise ValueError("frozen PRMBench file must contain all and only discovery pairs")
    provenance_path = Path(extras["local_pair_provenance"])
    provenance_raw = provenance_path.read_bytes()
    if hashlib.sha256(provenance_raw).hexdigest() != extras["local_pair_provenance_sha256"]:
        raise ValueError("frozen PRMBench normalization provenance SHA256 mismatch")
    provenance = json.loads(provenance_raw)
    if not (
        provenance.get("output_sha256") == actual_sha
        and provenance.get("source_revision") == config.revision
        and config.revision == config.expected_resolved_data_revision
        and provenance.get("source_sha256") == config.expected_raw_sha256
        and provenance.get("all_580_content_hashes_match_frozen_selected_pairs") is True
        and config.validation_accessed is False
    ):
        raise ValueError("frozen PRMBench source or normalization provenance mismatch")
    selected = select_prmbench_pairs(pairs, limit=limit, seed=seed)
    source = {
        "source_type": "pinned_normalized_discovery_jsonl",
        "dataset": "prmbench",
        "hf_id": config.hf_id,
        "split": config.split,
        "requested_revision": config.revision,
        "resolved_data_revision": config.expected_resolved_data_revision,
        "raw_sha256": config.expected_raw_sha256,
        "processed_file": {"path": str(path), "sha256": actual_sha, "row_count": len(examples)},
        "normalization_provenance": {
            "path": str(provenance_path),
            "sha256": extras["local_pair_provenance_sha256"],
        },
        "validation_accessed": False,
    }
    return PRMBenchLoadResult(
        pairs=selected,
        source_provenance=source,
        pair_audit=audit,
        selection_manifest=prmbench_selection_manifest(
            selected, available_pair_count=len(pairs), limit=limit, seed=seed, source=source
        ),
    )
