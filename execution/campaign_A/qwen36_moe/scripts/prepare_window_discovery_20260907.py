#!/usr/bin/env python3
"""Rebuild the frozen 290 PRMBench discovery pairs from a local pinned raw JSONL.

No source is downloaded. Only source IDs in the frozen discovery partition are
normalized, and every sample ID/content hash must match the frozen selection.
Output is a new versioned JSONL with two rows per pair, plus provenance. Existing
different files and the canonical pilot file are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from routeverify.analysis.protocols import validate_discovery_validation_manifest  # noqa: E402
from routeverify.data.prmbench import (  # noqa: E402
    PRMBENCH_RAW_SHA256,
    PRMBENCH_RESOLVED_DATA_REVISION,
    PRMBENCH_REVISION,
    PRMBenchPair,
    eligible_pair_from_prmbench_record,
)
from routeverify.data.schema import stable_hash, trace_example_from_json  # noqa: E402

DEFAULT_SPLIT = REPO_ROOT / "docs/results/first_batch_zero_gpu_20260730/discovery_validation_split.json"
DEFAULT_SELECTED = REPO_ROOT / "docs/results/first_batch_zero_gpu_20260730/selected_pairs.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_discovery(
    *,
    source: Path,
    output: Path,
    split_manifest: Path = DEFAULT_SPLIT,
    selected_manifest: Path = DEFAULT_SELECTED,
    check: bool = False,
) -> dict[str, Any]:
    """Validate every required input before writing the deterministic new dataset."""

    source = source.absolute()
    output = output.absolute()
    split_manifest = split_manifest.absolute()
    selected_manifest = selected_manifest.absolute()
    if output.suffix != ".jsonl":
        raise ValueError("--output must be a new versioned .jsonl path")
    protected = {
        source.resolve(),
        split_manifest.resolve(),
        selected_manifest.resolve(),
        (REPO_ROOT / "data/processed/prmbench-pairs.jsonl").resolve(),
    }
    if output.resolve() in protected:
        raise ValueError("refusing to overwrite source, manifests, or the canonical pilot")
    source_hash = _sha256(source)
    if source_hash != PRMBENCH_RAW_SHA256:
        raise ValueError("source SHA256 differs from the pinned PRMBench raw data")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    validate_discovery_validation_manifest(split)
    discovery = set(split["discovery_group_ids"])
    if len(discovery) != 290:
        raise ValueError("this execution contract requires exactly 290 frozen discovery pairs")
    selected_payload = json.loads(selected_manifest.read_text(encoding="utf-8"))
    expected_selected_hash = stable_hash(
        {key: value for key, value in selected_payload.items() if key != "manifest_hash"},
        length=64,
    )
    if selected_payload.get("manifest_hash") != expected_selected_hash:
        raise ValueError("selected-pairs manifest self-hash mismatch")
    selected_rows = [
        row for row in selected_payload["pairs"] if row["group_id"] in discovery
    ]
    selected = {row["source_id"]: row for row in selected_rows}
    if len(selected_rows) != 290 or len(selected) != 290:
        raise ValueError("frozen selection does not contain 290 unique discovery source IDs")
    if {row["group_id"] for row in selected_rows} != discovery:
        raise ValueError("frozen selection and discovery group IDs differ")
    pairs: dict[str, PRMBenchPair] = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line)
            source_id = str(raw.get("idx"))
            if source_id not in selected:
                continue
            expected = selected[source_id]
            pair = eligible_pair_from_prmbench_record(raw)
            if pair is None:
                raise ValueError(f"frozen discovery source no longer normalizes: {source_id}")
            if pair.clean.group_id != expected["group_id"]:
                raise ValueError(f"group ID differs from frozen selection: {source_id}")
            if pair.clean.group_id in pairs:
                raise ValueError(f"duplicate discovery source in raw JSONL: {source_id}")
            for role in ("clean", "corrupt"):
                example = getattr(pair, role)
                if example.content_hash != expected[f"{role}_content_hash"]:
                    raise ValueError(f"{role} content hash differs from frozen selection: {source_id}")
                if example.sample_id != expected[f"{role}_sample_id"]:
                    raise ValueError(f"{role} sample ID differs from frozen selection: {source_id}")
            pairs[pair.clean.group_id] = pair
    if set(pairs) != discovery:
        raise ValueError(f"raw source misses {len(discovery - set(pairs))} discovery pairs")
    rows = [
        getattr(pairs[group_id], role).to_json()
        for group_id in sorted(pairs)
        for role in ("clean", "corrupt")
    ]
    for row in rows:
        if trace_example_from_json(row).content_hash != row["content_hash"]:
            raise ValueError("normalized row fails round-trip content hash check")
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    provenance: dict[str, Any] = {
        "output": str(output),
        "output_sha256": hashlib.sha256(content).hexdigest(),
        "pair_count": 290,
        "example_count": 580,
        "all_580_content_hashes_match_frozen_selected_pairs": True,
        "source": str(source),
        "source_revision": PRMBENCH_REVISION,
        "source_resolved_data_revision": PRMBENCH_RESOLVED_DATA_REVISION,
        "source_sha256": source_hash,
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "selected_pairs_manifest": str(selected_manifest),
        "selected_pairs_manifest_sha256": _sha256(selected_manifest),
        "validation_accessed": False,
        "scope": "Only frozen discovery source IDs normalized. No protected validation experiment read.",
        "serialization": "sorted group_id; clean then corrupt; UTF-8; ensure_ascii=False; sort_keys=True; one newline per row",
        "builder": "scripts/prepare_window_discovery_20260907.py",
    }
    if check:
        if not output.is_file() or output.read_bytes() != content:
            raise ValueError("existing output does not match the exact frozen discovery reconstruction")
        return {**provenance, "check": "passed", "files_written": False}
    if output.exists() and output.read_bytes() != content:
        raise FileExistsError("output already exists with different content; choose a new versioned path")
    provenance_path = output.with_suffix(".provenance.json")
    provenance_content = (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old.get("output_sha256") != provenance["output_sha256"]:
            raise FileExistsError("provenance already exists for different content; choose a new versioned path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(content)
    provenance_path.write_bytes(provenance_content)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Local pinned prmbench_preview.jsonl; never downloaded.")
    parser.add_argument("--output", required=True, type=Path, help="New versioned normalized JSONL path.")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--selected-manifest", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--check", action="store_true", help="Check byte-for-byte reconstruction without writing.")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_discovery(
                source=args.source,
                output=args.output,
                split_manifest=args.split_manifest,
                selected_manifest=args.selected_manifest,
                check=args.check,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
