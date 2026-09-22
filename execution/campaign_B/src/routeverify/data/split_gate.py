"""Gate-0 guards for split-first capture and donor matching."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def load_pair_partitions(path: str | Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the frozen discovery/validation group IDs without modifying the artifact."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    discovery = [str(value) for value in payload.get("discovery_group_ids", [])]
    validation = [str(value) for value in payload.get("validation_group_ids", [])]
    if not discovery or not validation:
        raise ValueError("split manifest must contain non-empty discovery and validation IDs")
    overlap = sorted(set(discovery) & set(validation))
    if overlap:
        raise ValueError(f"split manifest is not disjoint: {overlap[:5]}")
    partitions = {group_id: "discovery" for group_id in discovery}
    partitions.update({group_id: "validation" for group_id in validation})
    return partitions, payload


def assert_split_local_matching_edges(
    edges: Iterable[tuple[str, str]],
    partition_by_group: Mapping[str, str],
) -> None:
    """Reject any matching edge whose endpoints cross frozen partitions."""

    cross_split: list[tuple[str, str, str, str]] = []
    for receiver, donor in edges:
        if receiver not in partition_by_group:
            raise ValueError(f"receiver group is absent from frozen split manifest: {receiver}")
        if donor not in partition_by_group:
            raise ValueError(f"donor group is absent from frozen split manifest: {donor}")
        receiver_split = partition_by_group[receiver]
        donor_split = partition_by_group[donor]
        if receiver_split != donor_split:
            cross_split.append((receiver, donor, receiver_split, donor_split))
    if cross_split:
        raise ValueError(f"Gate 0 cross-split matching edge(s): {cross_split[:5]}")


def select_partition_group_ids(
    path: str | Path,
    *,
    partition: str,
) -> tuple[tuple[str, ...], dict[str, str], dict[str, Any]]:
    if partition not in {"discovery", "validation"}:
        raise ValueError("split_partition must be discovery or validation")
    partitions, payload = load_pair_partitions(path)
    selected = tuple(
        group_id for group_id, group_partition in partitions.items() if group_partition == partition
    )
    return selected, partitions, payload
