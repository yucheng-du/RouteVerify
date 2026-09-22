"""BIG-Bench Mistake optional normalization."""

from __future__ import annotations

from typing import Any

from routeverify.data.schema import TraceExample, stable_hash


def trace_from_bigbench_mistake_record(record: dict[str, Any], *, subset: str) -> TraceExample:
    """Convert a BIG-Bench Mistake row to TraceExample.

    This optional dataset is not part of the first pilot, but the normalizer keeps
    the shared schema contract ready for later OOD/correction experiments.
    """

    steps_value = record.get("steps", ())
    if isinstance(steps_value, str):
        steps = tuple(line.strip() for line in steps_value.splitlines() if line.strip())
    else:
        steps = tuple(str(step).strip() for step in steps_value if str(step).strip())
    if not steps:
        raise ValueError("BIG-Bench Mistake record has no steps")

    raw_index = record.get("mistake_index")
    first_error = None if raw_index is None or int(raw_index) == -1 else int(raw_index)
    error_steps = () if first_error is None else (first_error,)
    sample_id = str(record.get("id") or stable_hash(record))
    return TraceExample(
        sample_id=f"bigbench_mistake/{subset}/{sample_id}",
        dataset="bigbench_mistake",
        subset=subset,
        problem=str(record.get("input") or "").strip(),
        steps=steps,
        first_error=first_error,
        error_steps=error_steps,
        is_all_correct=first_error is None,
        variant="natural",
        metadata={"answer": record.get("answer"), "target": record.get("target")},
    )
