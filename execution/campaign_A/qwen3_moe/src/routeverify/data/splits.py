"""Deterministic grouped splits."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from routeverify.data.schema import TraceExample, stable_hash

SplitName = str


def deterministic_grouped_split(
    examples: Iterable[TraceExample],
    *,
    seed: int = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
) -> dict[SplitName, list[TraceExample]]:
    """Split examples by group_id so pairs/problems never leak across splits."""

    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    groups: dict[str, list[TraceExample]] = defaultdict(list)
    for example in examples:
        groups[example.group_id].append(example)

    assignments: dict[SplitName, list[TraceExample]] = {"train": [], "val": [], "test": []}
    for group_id, group_examples in groups.items():
        bucket = int(stable_hash({"group": group_id, "seed": seed}, length=12), 16) / float(16**12)
        if bucket < train:
            split = "train"
        elif bucket < train + val:
            split = "val"
        else:
            split = "test"
        assignments[split].extend(sorted(group_examples, key=lambda item: item.sample_id))
    return assignments


def split_manifest(splits: dict[SplitName, list[TraceExample]]) -> dict[str, list[str]]:
    return {
        split: [example.sample_id for example in sorted(examples, key=lambda item: item.sample_id)]
        for split, examples in splits.items()
    }
