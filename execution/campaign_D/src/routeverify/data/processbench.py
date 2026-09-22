"""ProcessBench normalization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from routeverify.data.schema import TraceExample, normalize_error_label, stable_hash

PROCESSBENCH_ID = "Qwen/ProcessBench"


def _coerce_steps(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        steps = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, Iterable):
        steps = [str(step).strip() for step in value if str(step).strip()]
    else:
        raise TypeError("steps must be a string or iterable")
    if not steps:
        raise ValueError("ProcessBench record has no steps")
    return tuple(steps)


def trace_from_processbench_record(record: dict[str, Any], *, subset: str) -> TraceExample:
    """Convert a raw ProcessBench row to TraceExample."""

    steps = _coerce_steps(record["steps"])
    raw_label = record.get("label")
    first_error, error_steps, is_all_correct = normalize_error_label(
        None if raw_label is None else int(raw_label),
        len(steps),
    )
    source_id = str(record.get("id") or stable_hash({"problem": record["problem"], "steps": steps}))
    return TraceExample(
        sample_id=f"processbench/{subset}/{source_id}",
        dataset="processbench",
        subset=subset,
        problem=str(record["problem"]).strip(),
        steps=steps,
        first_error=first_error,
        error_steps=error_steps,
        is_all_correct=is_all_correct,
        generator=None if record.get("generator") is None else str(record.get("generator")),
        variant="natural",
        metadata={
            "source_id": source_id,
            "final_answer_correct": record.get("final_answer_correct"),
            "raw_label": raw_label,
        },
    )


def load_processbench(
    *,
    subset: str,
    split: str = "train",
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[TraceExample]:
    """Load ProcessBench through datasets, if the optional dependency is installed."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised by CLI users
        raise RuntimeError("Install data dependencies with `uv sync --extra data`.") from exc

    cache = str(cache_dir) if cache_dir else None
    try:
        dataset = load_dataset(
            PROCESSBENCH_ID,
            subset,
            split=split,
            cache_dir=cache,
            revision=revision,
        )
    except ValueError as exc:
        if "BuilderConfig" not in str(exc):
            raise
        # ProcessBench is currently published as one default config with
        # benchmark subsets exposed as HF splits: gsm8k/math/olympiadbench/omnimath.
        dataset = load_dataset(PROCESSBENCH_ID, split=subset, cache_dir=cache, revision=revision)
    examples: list[TraceExample] = []
    for record in dataset:
        examples.append(trace_from_processbench_record(dict(record), subset=subset))
    if limit is not None:
        examples = sorted(
            examples,
            key=lambda example: stable_hash({"seed": seed, "sample_id": example.sample_id}),
        )[:limit]
    return examples


def balanced_processbench_sample(
    examples: Iterable[TraceExample],
    *,
    limit: int,
    seed: int = 0,
) -> list[TraceExample]:
    """Return a deterministic error/all-correct balanced ProcessBench sample."""

    if limit <= 0:
        return []
    materialized = list(examples)
    all_correct = [example for example in materialized if example.is_all_correct]
    error = [example for example in materialized if not example.is_all_correct]
    if not all_correct or not error:
        return _stable_sample(materialized, limit=limit, seed=seed)

    first_target = limit // 2
    second_target = limit - first_target
    error_target = min(second_target, len(error))
    all_correct_target = min(first_target, len(all_correct))
    remainder = limit - error_target - all_correct_target
    if remainder > 0:
        error_target = min(len(error), error_target + remainder)
        remainder = limit - error_target - all_correct_target
    if remainder > 0:
        all_correct_target = min(len(all_correct), all_correct_target + remainder)

    selected_errors = _stable_sample(error, limit=error_target, seed=seed)
    selected_all_correct = _sample_by_generator_targets(
        targets=selected_errors,
        candidates=all_correct,
        limit=all_correct_target,
        seed=seed + 1,
    )
    selected = [*selected_errors, *selected_all_correct]
    return sorted(
        selected,
        key=lambda example: stable_hash({"seed": seed, "sample_id": example.sample_id}),
    )


def _sample_by_generator_targets(
    *,
    targets: Iterable[TraceExample],
    candidates: Iterable[TraceExample],
    limit: int,
    seed: int,
) -> list[TraceExample]:
    if limit <= 0:
        return []
    target_list = list(targets)
    candidate_list = _stable_sample(candidates, limit=10**12, seed=seed)
    selected: list[TraceExample] = []
    used_sample_ids: set[str] = set()
    for target in target_list:
        if len(selected) >= limit:
            break
        ranked = sorted(
            (
                candidate
                for candidate in candidate_list
                if candidate.sample_id not in used_sample_ids
            ),
            key=lambda candidate: (
                candidate.generator != target.generator,
                stable_hash(
                    {
                        "seed": seed,
                        "target": target.sample_id,
                        "candidate": candidate.sample_id,
                    }
                ),
            ),
        )
        if not ranked:
            break
        selected.append(ranked[0])
        used_sample_ids.add(ranked[0].sample_id)

    if len(selected) < limit:
        for candidate in candidate_list:
            if candidate.sample_id in used_sample_ids:
                continue
            selected.append(candidate)
            used_sample_ids.add(candidate.sample_id)
            if len(selected) >= limit:
                break
    return selected


def _stable_sample(
    examples: Iterable[TraceExample],
    *,
    limit: int,
    seed: int,
) -> list[TraceExample]:
    return sorted(
        examples,
        key=lambda example: stable_hash({"seed": seed, "sample_id": example.sample_id}),
    )[:limit]


def write_examples_jsonl(examples: Iterable[TraceExample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(output)
