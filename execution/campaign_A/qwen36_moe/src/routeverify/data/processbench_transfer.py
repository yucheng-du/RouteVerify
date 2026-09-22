"""Frozen ProcessBench natural-error transfer data contract.

The transfer unit is a pair of independently generated solutions from the
same public ProcessBench generator, not necessarily the same problem.  One
solution is annotated all-correct and the other has a public first-error
label.  Both are rendered through the same current-step ordinal; the
all-correct side's ordinal is a *mapped comparison boundary*, never an
invented natural error.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from routeverify.data.prompts import PromptRender, step_binary
from routeverify.data.schema import (
    TraceExample,
    normalize_text,
    stable_hash,
    trace_example_from_json,
)

PROCESSBENCH_ID = "Qwen/ProcessBench"
PROCESSBENCH_REVISION = "3bdcd5371ed567559a78f559c01c13a6deee7604"
PROCESSBENCH_HF_CONFIG = "default"
PROCESSBENCH_LICENSE = "apache-2.0"
PROCESSBENCH_SUBSETS: tuple[str, ...] = ("gsm8k",)
PROCESSBENCH_SOURCE_FILES: dict[str, dict[str, int | str]] = {
    "gsm8k": {
        "filename": "gsm8k.json",
        "sha256": "fb1c59dfd1e83e1c5a48dbe598db27e4bc0b8bc524a09445f2d4b69178f61349",
        "num_bytes": 574343,
        "num_rows": 400,
    },
    "math": {
        "filename": "math.json",
        "sha256": "60cb0d7a69cbec98d43cf50855b9b7e7294599c3c9a24bdd031abd18c3b6b136",
        "num_bytes": 1938778,
        "num_rows": 1000,
    },
    "olympiadbench": {
        "filename": "olympiadbench.json",
        "sha256": "e9d801c9f3c03f949adbfa08e90cb21ad85c0b9224bc5e98326f8215c218aa55",
        "num_bytes": 2808955,
        "num_rows": 1000,
    },
    "omnimath": {
        "filename": "omnimath.json",
        "sha256": "e8d7cf4512c82168dae2fe48b1718069a3002e7910af1d180a27ee0d9948ba47",
        "num_bytes": 2842872,
        "num_rows": 1000,
    },
}
PROCESSBENCH_SOURCE_AUDIT: dict[str, dict[str, int]] = {
    "gsm8k": {
        "source_rows": 400,
        "first_error_rows": 207,
        "all_correct_rows": 193,
        "eligible_same_generator_first_error_rows": 205,
    },
}

TRANSFER_PROTOCOL_ID = "processbench_same_generator_natural_transfer_v1"
PAIRING_RULE_ID = "same_generator_exact_ordinal_unique_matching_v1"
PROMPT_PROJECTION_ID = "step_binary_prefix_through_manifest_boundary_v1"
TRANSFER_ROLES: tuple[str, str] = ("matched_correct", "first_error")
_NUMBERED_STEP_RE = re.compile(r"(?m)^\s*(?:Step\s*)?(\d+)[.):]\s+")
_INTEGER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class NaturalTransferPair:
    """Two same-generator natural traces aligned at one public error ordinal."""

    pair_id: str
    matched_correct: TraceExample
    first_error: TraceExample
    boundary_step: int

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise ValueError("natural transfer pair_id must be non-empty")
        if self.matched_correct.pair_id != self.pair_id:
            raise ValueError("matched-correct pair_id mismatch")
        if self.first_error.pair_id != self.pair_id:
            raise ValueError("first-error pair_id mismatch")
        if self.matched_correct.variant != "clean":
            raise ValueError("matched-correct compatibility variant must be clean")
        if self.first_error.variant != "corrupt":
            raise ValueError("first-error compatibility variant must be corrupt")
        if not self.matched_correct.is_all_correct:
            raise ValueError("matched-correct trace must have public label -1")
        if self.matched_correct.error_steps:
            raise ValueError("matched-correct trace must not contain error steps")
        if self.first_error.first_error != self.boundary_step:
            raise ValueError("first-error trace does not match the paired boundary")
        if self.first_error.error_steps != (self.boundary_step,):
            raise ValueError("first-error trace must contain only its public first-error label")
        if self.boundary_step < 0:
            raise ValueError("paired boundary must be non-negative")
        if self.boundary_step >= len(self.matched_correct.steps):
            raise ValueError("mapped correct boundary is outside its trace")
        if self.boundary_step >= len(self.first_error.steps):
            raise ValueError("first-error boundary is outside its trace")
        if self.matched_correct.dataset != "processbench":
            raise ValueError("matched-correct trace must come from ProcessBench")
        if self.first_error.dataset != "processbench":
            raise ValueError("first-error trace must come from ProcessBench")
        if self.matched_correct.subset != self.first_error.subset:
            raise ValueError("natural transfer pair members must share a subset")
        if self.matched_correct.subset not in PROCESSBENCH_SUBSETS:
            raise ValueError("natural transfer pair has an unsupported subset")
        if not self.matched_correct.generator or not self.first_error.generator:
            raise ValueError("natural transfer pair members must record a generator")
        if self.matched_correct.generator != self.first_error.generator:
            raise ValueError("natural transfer requires the same generator")
        if self.matched_correct.metadata.get("transfer_role") != "matched_correct":
            raise ValueError("matched-correct role metadata is missing")
        if self.first_error.metadata.get("transfer_role") != "first_error":
            raise ValueError("first-error role metadata is missing")
        expected_metadata_fields = {
            "source_id",
            "raw_label",
            "final_answer_correct",
            "source_schema",
            "transfer_protocol_id",
            "pairing_rule_id",
            "mapped_boundary_step",
            "same_generator",
            "same_normalized_problem",
            "label_visible_to_model",
            "post_boundary_steps_visible",
            "transfer_role",
            "boundary_source",
            "normalized_problem_hash",
            "source_variant",
        }
        for example in (self.matched_correct, self.first_error):
            if set(example.metadata) != expected_metadata_fields:
                raise ValueError("paired trace metadata fields do not match the frozen schema")
            if example.metadata.get("mapped_boundary_step") != self.boundary_step:
                raise ValueError("mapped boundary metadata mismatch")
            if example.metadata.get("transfer_protocol_id") != TRANSFER_PROTOCOL_ID:
                raise ValueError("transfer protocol metadata mismatch")
            if example.metadata.get("pairing_rule_id") != PAIRING_RULE_ID:
                raise ValueError("pairing rule metadata mismatch")
            if example.metadata.get("same_generator") is not True:
                raise ValueError("same-generator metadata must be true")
            if example.metadata.get("label_visible_to_model") is not False:
                raise ValueError("labels must be hidden from model-facing prompts")
            if example.metadata.get("post_boundary_steps_visible") is not False:
                raise ValueError("post-boundary steps must be hidden from model-facing prompts")
            if example.metadata.get("source_variant") != "natural":
                raise ValueError("paired traces must preserve their natural source provenance")
            if example.metadata.get("source_schema") != "processbench_public_json_v1":
                raise ValueError("paired trace source schema metadata mismatch")
            source_id = example.metadata.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("paired trace source_id metadata must be non-empty")
            if example.sample_id != f"processbench/{example.subset}/{source_id}":
                raise ValueError("paired trace sample_id does not match its source_id")
            if type(example.metadata.get("final_answer_correct")) is not bool:
                raise ValueError("paired trace final-answer metadata must be boolean")
            if example.error_type is not None:
                raise ValueError("natural transfer traces must not have a synthetic error_type")

        correct_label = self.matched_correct.metadata.get("raw_label")
        error_label = self.first_error.metadata.get("raw_label")
        if type(correct_label) is not int or correct_label != -1:
            raise ValueError("matched-correct raw label must be integer -1")
        if type(error_label) is not int or error_label != self.boundary_step:
            raise ValueError("first-error raw label must equal the mapped boundary")
        if self.matched_correct.metadata.get("boundary_source") != "paired_first_error_ordinal":
            raise ValueError("matched-correct boundary source metadata mismatch")
        if (
            self.first_error.metadata.get("boundary_source")
            != "processbench_public_first_error_label"
        ):
            raise ValueError("first-error boundary source metadata mismatch")

        correct_problem_hash = stable_hash(
            {
                "subset": self.matched_correct.subset,
                "problem": normalize_text(self.matched_correct.problem),
            },
            length=32,
        )
        error_problem_hash = stable_hash(
            {
                "subset": self.first_error.subset,
                "problem": normalize_text(self.first_error.problem),
            },
            length=32,
        )
        same_normalized_problem = correct_problem_hash == error_problem_hash
        for example, expected_hash in (
            (self.matched_correct, correct_problem_hash),
            (self.first_error, error_problem_hash),
        ):
            if example.metadata.get("normalized_problem_hash") != expected_hash:
                raise ValueError("normalized problem hash metadata mismatch")
            if example.metadata.get("same_normalized_problem") is not same_normalized_problem:
                raise ValueError("same-normalized-problem audit metadata mismatch")


@dataclass(frozen=True)
class ProcessBenchTransferBundle:
    """Validated selected rows plus their self-hashing manifest."""

    pairs: tuple[NaturalTransferPair, ...]
    manifest: dict[str, Any]
    processed_jsonl_bytes: bytes

    @property
    def examples(self) -> tuple[TraceExample, ...]:
        return tuple(
            example
            for pair in self.pairs
            for example in (pair.matched_correct, pair.first_error)
        )


def parse_processbench_steps(value: Any) -> tuple[str, ...]:
    """Parse a step field without silently deleting or re-indexing steps.

    The pinned public snapshot uses ``list[str]``.  Explicitly numbered strings
    are supported for fixtures and future schema-compatible exports; ambiguous
    unnumbered multiline strings fail closed because splitting them can turn
    bullet lines inside one mathematical step into new labeled steps.
    """

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise TypeError(f"ProcessBench step {index} must be a string")
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"ProcessBench step {index} is empty")
            parsed.append(stripped)
        if not parsed:
            raise ValueError("ProcessBench record has no steps")
        return tuple(parsed)

    if not isinstance(value, str):
        raise TypeError("ProcessBench steps must be list[str] or an explicitly numbered string")
    text = value.strip()
    if not text:
        raise ValueError("ProcessBench record has no steps")
    matches = list(_NUMBERED_STEP_RE.finditer(text))
    if not matches:
        if "\n" in text:
            raise ValueError("ambiguous unnumbered multiline ProcessBench steps")
        return (text,)
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"numbered ProcessBench steps are not contiguous from 1: {numbers}")
    if text[: matches[0].start()].strip():
        raise ValueError("unexpected text before the first numbered ProcessBench step")
    parsed = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        step = text[match.end() : end].strip()
        if not step:
            raise ValueError(f"ProcessBench step {index} is empty")
        parsed.append(step)
    return tuple(parsed)


def parse_processbench_label(value: Any, *, n_steps: int) -> int | None:
    """Return a strict zero-based first error, or ``None`` for public label -1."""

    if type(n_steps) is not int or n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if isinstance(value, bool):
        raise TypeError("boolean ProcessBench labels are invalid")
    if isinstance(value, int):
        label = value
    elif isinstance(value, str) and _INTEGER_RE.fullmatch(value.strip()):
        label = int(value.strip())
    else:
        raise TypeError("ProcessBench label must be an integer or exact integer string")
    if label == -1:
        return None
    if label < 0 or label >= n_steps:
        raise ValueError(f"ProcessBench label {label} is out of bounds for {n_steps} steps")
    return label


def strict_trace_from_processbench_record(
    record: Mapping[str, Any],
    *,
    subset: str,
) -> TraceExample:
    """Normalize one pinned-source row under the strict transfer schema."""

    expected_fields = {"id", "generator", "problem", "steps", "label", "final_answer_correct"}
    missing = sorted(expected_fields - set(record))
    if missing:
        raise ValueError(f"ProcessBench row is missing required fields: {missing}")
    unexpected = sorted(set(record) - expected_fields)
    if unexpected:
        raise ValueError(f"ProcessBench row has unexpected fields: {unexpected}")
    source_id = record["id"]
    generator = record["generator"]
    problem = record["problem"]
    final_answer_correct = record["final_answer_correct"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise TypeError("ProcessBench id must be a non-empty string")
    if not isinstance(generator, str) or not generator.strip():
        raise TypeError("ProcessBench generator must be a non-empty string")
    if not isinstance(problem, str) or not problem.strip():
        raise TypeError("ProcessBench problem must be a non-empty string")
    if not isinstance(final_answer_correct, bool):
        raise TypeError("ProcessBench final_answer_correct must be boolean")
    steps = parse_processbench_steps(record["steps"])
    first_error = parse_processbench_label(record["label"], n_steps=len(steps))
    return TraceExample(
        sample_id=f"processbench/{subset}/{source_id.strip()}",
        dataset="processbench",
        subset=subset,
        problem=problem.strip(),
        steps=steps,
        first_error=first_error,
        error_steps=() if first_error is None else (first_error,),
        is_all_correct=first_error is None,
        generator=generator.strip(),
        variant="natural",
        metadata={
            "source_id": source_id.strip(),
            "raw_label": int(record["label"]),
            "final_answer_correct": final_answer_correct,
            "source_schema": "processbench_public_json_v1",
        },
    )


def _source_contract() -> dict[str, Any]:
    return {
        "hf_id": PROCESSBENCH_ID,
        "revision": PROCESSBENCH_REVISION,
        "hf_config": PROCESSBENCH_HF_CONFIG,
        "license": PROCESSBENCH_LICENSE,
        "files": [
            {
                "hf_config": PROCESSBENCH_HF_CONFIG,
                "hf_split": subset,
                **PROCESSBENCH_SOURCE_FILES[subset],
            }
            for subset in PROCESSBENCH_SUBSETS
        ],
    }


def _selection_contract(*, seed: int, pairs_per_subset: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "subsets": list(PROCESSBENCH_SUBSETS),
        "pairs_per_subset": pairs_per_subset,
        "pairing_rule_id": PAIRING_RULE_ID,
        "selection_unit": "same_generator_matched_correct_first_error_pair",
        "same_generator_required": True,
        "same_normalized_problem_required": False,
        "matched_correct_reuse_allowed": False,
        "first_error_reuse_allowed": False,
        "generator_balancing": "deterministic round-robin over first-error generator",
        "edge_ranking": [
            "absolute_step_count_difference",
            "absolute_prefix_character_count_difference",
            "sha256_seed_pair_ids",
        ],
    }


def _eligibility_contract() -> dict[str, Any]:
    return {
        "included_first_error": "public label is an in-bounds zero-based index",
        "included_matched_correct": "public label is -1 and trace contains mapped ordinal",
        "natural_error_boundary": "first_error public label, zero-based",
        "matched_correct_boundary": "paired first_error ordinal, not a natural error label",
        "post_error_steps": "excluded from every model-facing step-binary prompt",
        "first_step_errors": "included; mapped ordinal zero is valid",
        "unpaired_all_correct": "not selected and never assigned a boundary",
    }


def _boundary_contract() -> dict[str, Any]:
    return {
        "mapping": "matched_correct_step = first_error_step",
        "model_shared": True,
        "tokenizer_used_for_selection": False,
        "roles": list(TRANSFER_ROLES),
    }


def _prompt_contract() -> dict[str, Any]:
    return {
        "projection_id": PROMPT_PROJECTION_ID,
        "renderer": "routeverify.data.prompts.step_binary",
        "candidate_labels": ["CORRECT", "INCORRECT"],
        "visible_fields": ["problem", "steps_through_mapped_boundary", "fixed_instruction"],
        "forbidden_fields": [
            "id",
            "generator",
            "raw_label",
            "first_error",
            "is_all_correct",
            "final_answer_correct",
            "pair_id",
            "transfer_role",
        ],
        "label_visible_to_model": False,
    }


def build_processbench_transfer_bundle(
    source_root: str | Path,
    *,
    seed: int = 20260828,
    pairs_per_subset: int = 64,
    smoke_pairs_per_subset: int = 8,
) -> ProcessBenchTransferBundle:
    """Build the deterministic same-generator natural transfer selection."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(pairs_per_subset) is not int or pairs_per_subset <= 0:
        raise ValueError("pairs_per_subset must be positive")
    if (
        type(smoke_pairs_per_subset) is not int
        or smoke_pairs_per_subset <= 0
        or smoke_pairs_per_subset > pairs_per_subset
    ):
        raise ValueError("smoke_pairs_per_subset must be in [1, pairs_per_subset]")
    root = Path(source_root)
    source_rows: dict[str, list[TraceExample]] = {}
    source_audit: dict[str, Any] = {}
    selected_pairs: list[NaturalTransferPair] = []
    for subset in PROCESSBENCH_SUBSETS:
        examples = _load_verified_source_subset(root, subset=subset)
        source_rows[subset] = examples
        selected, matching_audit = _select_generator_matched_pairs(
            examples,
            limit=pairs_per_subset,
            seed=seed,
            subset=subset,
        )
        if len(selected) != pairs_per_subset:
            raise ValueError(
                f"subset {subset} has only {len(selected)} eligible same-generator pairs; "
                f"requested {pairs_per_subset}"
            )
        selected_pairs.extend(selected)
        error_count = sum(not example.is_all_correct for example in examples)
        correct_count = len(examples) - error_count
        source_audit[subset] = {
            "source_rows": len(examples),
            "first_error_rows": error_count,
            "all_correct_rows": correct_count,
            **matching_audit,
            "selected_pairs": len(selected),
            "excluded_from_transfer": {
                "first_error_not_selected_or_no_available_generator_matched_counterpart": (
                    error_count - len(selected)
                ),
                "all_correct_not_selected_or_not_used_by_unique_matching": (
                    correct_count - len(selected)
                ),
            },
        }

    pair_rows = [_manifest_pair_row(pair) for pair in selected_pairs]
    examples = [
        example
        for pair in selected_pairs
        for example in (pair.matched_correct, pair.first_error)
    ]
    jsonl_bytes = _examples_jsonl_bytes(examples)
    processed_sha256 = hashlib.sha256(jsonl_bytes).hexdigest()
    smoke_pair_ids = [
        pair.pair_id
        for subset in PROCESSBENCH_SUBSETS
        for pair in [candidate for candidate in selected_pairs if candidate.first_error.subset == subset][
            :smoke_pairs_per_subset
        ]
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": TRANSFER_PROTOCOL_ID,
        "analysis_role": "exploratory_natural_error_transfer",
        "validation_accessed": False,
        "source": _source_contract(),
        "selection": _selection_contract(seed=seed, pairs_per_subset=pairs_per_subset),
        "eligibility": _eligibility_contract(),
        "boundary_contract": _boundary_contract(),
        "prompt_contract": _prompt_contract(),
        "counts": {
            "source_rows": sum(len(rows) for rows in source_rows.values()),
            "selected_pairs": len(selected_pairs),
            "selected_traces": len(examples),
            "first_error_traces": len(selected_pairs),
            "all_correct_traces": len(selected_pairs),
            "natural_error_boundaries": len(selected_pairs),
            "mapped_correct_boundaries": len(selected_pairs),
            "transfer_boundaries": len(selected_pairs) * 2,
            "by_subset": source_audit,
        },
        "run_roles": {
            "smoke": {
                "pair_ids": smoke_pair_ids,
                "pair_count": len(smoke_pair_ids),
                "pairs_per_subset": smoke_pairs_per_subset,
            },
            "full": {
                "pair_ids": [pair.pair_id for pair in selected_pairs],
                "pair_count": len(selected_pairs),
                "pairs_per_subset": pairs_per_subset,
            },
        },
        "processed_jsonl": {
            "sha256": processed_sha256,
            "num_bytes": len(jsonl_bytes),
            "row_count": len(examples),
            "rows_hash": stable_hash([example.to_json() for example in examples], length=64),
        },
        "pairs": pair_rows,
        "pairs_hash": stable_hash(pair_rows, length=64),
    }
    manifest["manifest_hash"] = stable_hash(manifest, length=64)
    return ProcessBenchTransferBundle(
        pairs=tuple(selected_pairs),
        manifest=manifest,
        processed_jsonl_bytes=jsonl_bytes,
    )


def write_processbench_transfer_bundle(
    bundle: ProcessBenchTransferBundle,
    *,
    data_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Atomically write the selected public rows and selection manifest."""

    data_output = Path(data_path)
    manifest_output = Path(manifest_path)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    data_tmp = data_output.with_suffix(data_output.suffix + ".tmp")
    manifest_tmp = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    data_tmp.write_bytes(bundle.processed_jsonl_bytes)
    manifest_tmp.write_text(
        json.dumps(bundle.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_tmp.replace(data_output)
    manifest_tmp.replace(manifest_output)


def _same_json_payload(left: Any, right: Any) -> bool:
    return stable_hash(left, length=64) == stable_hash(right, length=64)


def _validate_frozen_manifest_contract(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "analysis_role",
        "validation_accessed",
        "source",
        "selection",
        "eligibility",
        "boundary_contract",
        "prompt_contract",
        "counts",
        "run_roles",
        "processed_jsonl",
        "pairs",
        "pairs_hash",
        "manifest_hash",
    }
    if set(manifest) != expected_keys:
        raise ValueError("ProcessBench transfer manifest has unexpected top-level fields")
    if manifest.get("schema_version") != 1 or type(manifest.get("schema_version")) is not int:
        raise ValueError("unexpected ProcessBench transfer schema_version")
    if manifest.get("protocol_id") != TRANSFER_PROTOCOL_ID:
        raise ValueError("unexpected ProcessBench transfer protocol_id")
    if manifest.get("analysis_role") != "exploratory_natural_error_transfer":
        raise ValueError("unexpected ProcessBench transfer analysis_role")
    if manifest.get("validation_accessed") is not False:
        raise ValueError("ProcessBench transfer manifest must record validation_accessed=false")
    if not _same_json_payload(manifest.get("source"), _source_contract()):
        raise ValueError("ProcessBench transfer source contract mismatch")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("ProcessBench transfer selection contract must be an object")
    seed = selection.get("seed")
    pairs_per_subset = selection.get("pairs_per_subset")
    if type(seed) is not int:
        raise ValueError("ProcessBench transfer selection seed must be an integer")
    if type(pairs_per_subset) is not int or pairs_per_subset <= 0:
        raise ValueError("ProcessBench transfer pairs_per_subset must be positive")
    expected_selection = _selection_contract(
        seed=seed,
        pairs_per_subset=pairs_per_subset,
    )
    if not _same_json_payload(selection, expected_selection):
        raise ValueError("ProcessBench transfer selection contract mismatch")
    for field, expected in (
        ("eligibility", _eligibility_contract()),
        ("boundary_contract", _boundary_contract()),
        ("prompt_contract", _prompt_contract()),
    ):
        if not _same_json_payload(manifest.get(field), expected):
            raise ValueError(f"ProcessBench transfer {field} mismatch")


def load_processbench_transfer_bundle(
    *,
    data_path: str | Path,
    manifest_path: str | Path,
    expected_data_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> ProcessBenchTransferBundle:
    """Load and fully validate a committed transfer bundle without HF access."""

    data_file = Path(data_path)
    manifest_file = Path(manifest_path)
    data_bytes = data_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_data_sha256 is not None and data_sha256 != expected_data_sha256:
        raise ValueError(
            f"processed ProcessBench SHA256 mismatch: expected={expected_data_sha256} "
            f"actual={data_sha256}"
        )
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"ProcessBench manifest SHA256 mismatch: expected={expected_manifest_sha256} "
            f"actual={manifest_sha256}"
        )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("ProcessBench transfer manifest must be a JSON object")
    recorded_manifest_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    computed_manifest_hash = stable_hash(unhashed, length=64)
    if recorded_manifest_hash != computed_manifest_hash:
        raise ValueError(
            "ProcessBench transfer semantic manifest hash mismatch: "
            f"recorded={recorded_manifest_hash} computed={computed_manifest_hash}"
        )
    _validate_frozen_manifest_contract(manifest)
    processed = manifest.get("processed_jsonl")
    if not isinstance(processed, Mapping):
        raise ValueError("ProcessBench transfer manifest lacks processed_jsonl provenance")
    if set(processed) != {"sha256", "num_bytes", "row_count", "rows_hash"}:
        raise ValueError("ProcessBench processed_jsonl provenance has unexpected fields")
    if processed.get("sha256") != data_sha256:
        raise ValueError("processed ProcessBench bytes do not match the manifest")
    if type(processed.get("num_bytes")) is not int or processed.get("num_bytes") != len(
        data_bytes
    ):
        raise ValueError("processed ProcessBench byte count does not match the manifest")
    if type(processed.get("row_count")) is not int:
        raise ValueError("processed ProcessBench row count must be an integer")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank row in processed ProcessBench JSONL at line {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"processed ProcessBench row {line_number} is not an object")
        rows.append(row)
    if len(rows) != processed.get("row_count"):
        raise ValueError("processed ProcessBench row count does not match the manifest")
    examples = [trace_example_from_json(row) for row in rows]
    if _examples_jsonl_bytes(examples) != data_bytes:
        raise ValueError(
            "processed ProcessBench rows are not in the canonical normalized JSONL schema"
        )
    if stable_hash([example.to_json() for example in examples], length=64) != processed.get(
        "rows_hash"
    ):
        raise ValueError("processed ProcessBench semantic rows hash mismatch")

    grouped: dict[str, list[TraceExample]] = defaultdict(list)
    for example in examples:
        if example.pair_id is None:
            raise ValueError("every ProcessBench transfer row must have pair_id")
        grouped[example.pair_id].append(example)
    manifest_pairs = manifest.get("pairs")
    if not isinstance(manifest_pairs, list):
        raise ValueError("ProcessBench transfer manifest pairs must be a list")
    pairs: list[NaturalTransferPair] = []
    seen_samples: set[str] = set()
    for pair_row in manifest_pairs:
        if not isinstance(pair_row, Mapping):
            raise ValueError("ProcessBench transfer pair row must be an object")
        pair_id_value = pair_row.get("pair_id")
        if not isinstance(pair_id_value, str) or not pair_id_value:
            raise ValueError("ProcessBench transfer pair_id must be a non-empty string")
        pair_id = pair_id_value
        members = grouped.pop(pair_id, [])
        if len(members) != 2:
            raise ValueError(f"pair {pair_id} must have exactly two processed rows")
        by_role = {str(member.metadata.get("transfer_role")): member for member in members}
        if set(by_role) != set(TRANSFER_ROLES):
            raise ValueError(f"pair {pair_id} has invalid transfer roles: {sorted(by_role)}")
        matched = by_role["matched_correct"]
        error = by_role["first_error"]
        boundary_value = pair_row.get("boundary_step")
        if type(boundary_value) is not int:
            raise ValueError(f"pair {pair_id} boundary_step must be an integer")
        boundary = boundary_value
        pair = NaturalTransferPair(
            pair_id=pair_id,
            matched_correct=matched,
            first_error=error,
            boundary_step=boundary,
        )
        expected_row = _manifest_pair_row(pair)
        if not _same_json_payload(dict(pair_row), expected_row):
            raise ValueError(f"pair {pair_id} does not match its manifest row")
        for member in members:
            if member.sample_id in seen_samples:
                raise ValueError(f"processed sample reused across pairs: {member.sample_id}")
            seen_samples.add(member.sample_id)
        pairs.append(pair)
    if grouped:
        raise ValueError(f"processed rows contain unmanifested pairs: {sorted(grouped)}")
    if stable_hash([_manifest_pair_row(pair) for pair in pairs], length=64) != manifest.get(
        "pairs_hash"
    ):
        raise ValueError("ProcessBench transfer pairs hash mismatch")
    manifest_ordered_examples = [
        example
        for pair in pairs
        for example in (pair.matched_correct, pair.first_error)
    ]
    if examples != manifest_ordered_examples:
        raise ValueError("processed ProcessBench row order must match the manifest pair order")
    _validate_counts(manifest, pairs)
    _validate_run_roles(manifest, pairs)
    return ProcessBenchTransferBundle(
        pairs=tuple(pairs),
        manifest=cast(dict[str, Any], manifest),
        processed_jsonl_bytes=data_bytes,
    )


def select_transfer_pairs(
    bundle: ProcessBenchTransferBundle,
    *,
    run_role: Literal["smoke", "full"],
) -> tuple[NaturalTransferPair, ...]:
    """Select the frozen smoke or full pair IDs in manifest order."""

    role_payload = bundle.manifest["run_roles"][run_role]
    pair_ids = [str(value) for value in role_payload["pair_ids"]]
    by_id = {pair.pair_id: pair for pair in bundle.pairs}
    return tuple(by_id[pair_id] for pair_id in pair_ids)


def render_processbench_transfer_prompt(
    pair: NaturalTransferPair,
    *,
    role: Literal["matched_correct", "first_error"],
) -> PromptRender:
    """Render only the blind model-facing projection for one paired boundary."""

    if role not in TRANSFER_ROLES:
        raise ValueError(f"unsupported ProcessBench transfer role: {role}")
    example = pair.matched_correct if role == "matched_correct" else pair.first_error
    prefix_steps = example.steps[: pair.boundary_step + 1]
    blind_projection = TraceExample(
        sample_id="processbench/model-facing-projection",
        dataset="processbench",
        subset=example.subset,
        problem=example.problem,
        steps=prefix_steps,
        first_error=None,
        error_steps=(),
        is_all_correct=True,
    )
    return step_binary(blind_projection, step_index=len(prefix_steps) - 1)


def selected_transfer_jsonl_bytes(pairs: Iterable[NaturalTransferPair]) -> bytes:
    """Serialize a frozen pair subset in matched-correct/first-error order."""

    return _examples_jsonl_bytes(
        [
            example
            for pair in pairs
            for example in (pair.matched_correct, pair.first_error)
        ]
    )


def processbench_transfer_selection_manifest(
    bundle: ProcessBenchTransferBundle,
    *,
    pairs: Iterable[NaturalTransferPair],
    requested_pair_limit: int | None,
) -> dict[str, Any]:
    """Return a self-hashing execution selection linked to the full manifest."""

    selected = list(pairs)
    if requested_pair_limit is None:
        expected_count = len(bundle.pairs)
    elif type(requested_pair_limit) is int and requested_pair_limit >= 0:
        expected_count = min(requested_pair_limit, len(bundle.pairs))
    else:
        raise ValueError("requested_pair_limit must be a non-negative integer or None")
    expected_pairs = list(bundle.pairs[:expected_count])
    if len(selected) != expected_count or any(
        not _same_json_payload(_manifest_pair_row(actual), _manifest_pair_row(expected))
        for actual, expected in zip(selected, expected_pairs, strict=True)
    ):
        raise ValueError("execution selection must be the requested frozen manifest prefix")
    pair_rows = [_manifest_pair_row(pair) for pair in selected]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": TRANSFER_PROTOCOL_ID,
        "dataset": "processbench_transfer",
        "selection_unit": "same_generator_matched_correct_first_error_pair",
        "selection_rule": {
            "name": "first_n_frozen_manifest_pairs",
            "limit_semantics": "pair_count",
            "parent_manifest_hash": bundle.manifest["manifest_hash"],
        },
        "seed": bundle.manifest["selection"]["seed"],
        "requested_pair_limit": requested_pair_limit,
        "available_pair_count": len(bundle.pairs),
        "selected_pair_count": len(selected),
        "selected_example_count": len(selected) * 2,
        "source": bundle.manifest["source"],
        "parent_manifest_hash": bundle.manifest["manifest_hash"],
        "pairs": pair_rows,
        "pairs_hash": stable_hash(pair_rows, length=64),
        "validation_accessed": False,
    }
    payload["manifest_hash"] = stable_hash(payload, length=64)
    return payload


def _load_verified_source_subset(root: Path, *, subset: str) -> list[TraceExample]:
    spec = PROCESSBENCH_SOURCE_FILES[subset]
    path = root / str(spec["filename"])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(
            f"ProcessBench {subset} source SHA256 mismatch: "
            f"expected={spec['sha256']} actual={digest}"
        )
    if len(raw) != spec["num_bytes"]:
        raise ValueError(f"ProcessBench {subset} source byte count mismatch")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"ProcessBench {subset} source must be a JSON array")
    if len(payload) != spec["num_rows"]:
        raise ValueError(f"ProcessBench {subset} source row count mismatch")
    examples: list[TraceExample] = []
    for row_index, row in enumerate(payload):
        if not isinstance(row, Mapping):
            raise ValueError(f"ProcessBench {subset} source row {row_index} must be an object")
        examples.append(strict_trace_from_processbench_record(row, subset=subset))
    sample_ids = [example.sample_id for example in examples]
    duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"ProcessBench {subset} contains duplicate IDs: {duplicates}")
    return examples


def _select_generator_matched_pairs(
    examples: list[TraceExample],
    *,
    limit: int,
    seed: int,
    subset: str,
) -> tuple[list[NaturalTransferPair], dict[str, Any]]:
    errors = [example for example in examples if example.first_error is not None]
    correct = [example for example in examples if example.is_all_correct]
    grouped_errors: dict[str, list[TraceExample]] = defaultdict(list)
    for error in errors:
        grouped_errors[error.generator or "unknown"].append(error)
    for generator, members in grouped_errors.items():
        grouped_errors[generator] = sorted(
            members,
            key=lambda example: stable_hash(
                {"seed": seed, "subset": subset, "error": example.sample_id}, length=64
            ),
        )
    generators = sorted(
        grouped_errors,
        key=lambda generator: stable_hash(
            {"seed": seed, "subset": subset, "generator": generator}, length=64
        ),
    )
    ordered_errors: list[TraceExample] = []
    offset = 0
    while True:
        progressed = False
        for generator in generators:
            members = grouped_errors[generator]
            if offset >= len(members):
                continue
            ordered_errors.append(members[offset])
            progressed = True
        if not progressed:
            break
        offset += 1

    eligible_errors = [
        error
        for error in errors
        if any(
            donor.generator == error.generator
            and error.first_error is not None
            and error.first_error < len(donor.steps)
            for donor in correct
        )
    ]
    selected: list[NaturalTransferPair] = []
    used_correct: set[str] = set()
    for error in ordered_errors:
        assert error.first_error is not None
        donors = [
            donor
            for donor in correct
            if donor.sample_id not in used_correct
            and donor.generator == error.generator
            and error.first_error < len(donor.steps)
        ]
        if not donors:
            continue
        donor = min(
            donors,
            key=lambda candidate: _pair_edge_rank(
                error,
                candidate,
                seed=seed,
                subset=subset,
            ),
        )
        boundary = error.first_error
        pair_hash = stable_hash(
            {
                "subset": subset,
                "first_error": error.sample_id,
                "matched_correct": donor.sample_id,
            },
            length=32,
        )
        pair_id = f"processbench-transfer/{subset}/{pair_hash}"
        error_problem_hash = stable_hash(
            {"subset": subset, "problem": normalize_text(error.problem)}, length=32
        )
        correct_problem_hash = stable_hash(
            {"subset": subset, "problem": normalize_text(donor.problem)}, length=32
        )
        shared = {
            "transfer_protocol_id": TRANSFER_PROTOCOL_ID,
            "pairing_rule_id": PAIRING_RULE_ID,
            "mapped_boundary_step": boundary,
            "same_generator": True,
            "same_normalized_problem": error_problem_hash == correct_problem_hash,
            "label_visible_to_model": False,
            "post_boundary_steps_visible": False,
        }
        matched_correct = replace(
            donor,
            pair_id=pair_id,
            variant="clean",
            metadata={
                **donor.metadata,
                **shared,
                "transfer_role": "matched_correct",
                "boundary_source": "paired_first_error_ordinal",
                "normalized_problem_hash": correct_problem_hash,
                "source_variant": "natural",
            },
        )
        first_error = replace(
            error,
            pair_id=pair_id,
            variant="corrupt",
            metadata={
                **error.metadata,
                **shared,
                "transfer_role": "first_error",
                "boundary_source": "processbench_public_first_error_label",
                "normalized_problem_hash": error_problem_hash,
                "source_variant": "natural",
            },
        )
        selected.append(
            NaturalTransferPair(
                pair_id=pair_id,
                matched_correct=matched_correct,
                first_error=first_error,
                boundary_step=boundary,
            )
        )
        used_correct.add(donor.sample_id)
        if len(selected) >= limit:
            break
    return selected, {
        "eligible_same_generator_first_error_rows": len(eligible_errors),
        "available_all_correct_rows": len(correct),
        "selected_error_generators": dict(
            sorted(Counter(pair.first_error.generator for pair in selected).items())
        ),
    }


def _pair_edge_rank(
    error: TraceExample,
    donor: TraceExample,
    *,
    seed: int,
    subset: str,
) -> tuple[int, int, str]:
    if error.first_error is None:
        raise ValueError("pair edge requires a first-error trace")
    boundary = error.first_error
    return (
        abs(len(error.steps) - len(donor.steps)),
        abs(_prefix_character_count(error, boundary) - _prefix_character_count(donor, boundary)),
        stable_hash(
            {
                "seed": seed,
                "subset": subset,
                "error": error.sample_id,
                "matched_correct": donor.sample_id,
            },
            length=64,
        ),
    )


def _prefix_character_count(example: TraceExample, boundary: int) -> int:
    return len(normalize_text(example.problem)) + sum(
        len(normalize_text(step)) for step in example.steps[: boundary + 1]
    )


def _manifest_pair_row(pair: NaturalTransferPair) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "subset": pair.first_error.subset,
        "boundary_step": pair.boundary_step,
        "same_generator": True,
        "same_normalized_problem": pair.first_error.metadata["same_normalized_problem"],
        "matched_correct": _manifest_member_row(pair.matched_correct),
        "first_error": _manifest_member_row(pair.first_error),
    }


def _manifest_member_row(example: TraceExample) -> dict[str, Any]:
    return {
        "sample_id": example.sample_id,
        "source_id": example.metadata["source_id"],
        "generator": example.generator,
        "transfer_role": example.metadata["transfer_role"],
        "raw_label": example.metadata["raw_label"],
        "first_error": example.first_error,
        "is_all_correct": example.is_all_correct,
        "mapped_boundary_step": example.metadata["mapped_boundary_step"],
        "boundary_source": example.metadata["boundary_source"],
        "num_steps": len(example.steps),
        "content_hash": example.content_hash,
    }


def _examples_jsonl_bytes(examples: Iterable[TraceExample]) -> bytes:
    return "".join(
        json.dumps(example.to_json(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    ).encode("utf-8")


def _validate_counts(
    manifest: Mapping[str, Any],
    pairs: list[NaturalTransferPair],
) -> None:
    counts = manifest.get("counts")
    expected_keys = {
        "source_rows",
        "selected_pairs",
        "selected_traces",
        "first_error_traces",
        "all_correct_traces",
        "natural_error_boundaries",
        "mapped_correct_boundaries",
        "transfer_boundaries",
        "by_subset",
    }
    if not isinstance(counts, Mapping) or set(counts) != expected_keys:
        raise ValueError("ProcessBench transfer counts contract mismatch")
    selected_count = len(pairs)
    source_count = sum(int(PROCESSBENCH_SOURCE_FILES[subset]["num_rows"]) for subset in PROCESSBENCH_SUBSETS)
    expected_totals = {
        "source_rows": source_count,
        "selected_pairs": selected_count,
        "selected_traces": selected_count * 2,
        "first_error_traces": selected_count,
        "all_correct_traces": selected_count,
        "natural_error_boundaries": selected_count,
        "mapped_correct_boundaries": selected_count,
        "transfer_boundaries": selected_count * 2,
    }
    for field, expected in expected_totals.items():
        if type(counts.get(field)) is not int or counts.get(field) != expected:
            raise ValueError(f"ProcessBench transfer count mismatch for {field}")

    by_subset = counts.get("by_subset")
    if not isinstance(by_subset, Mapping) or set(by_subset) != set(PROCESSBENCH_SUBSETS):
        raise ValueError("ProcessBench transfer by-subset counts contract mismatch")
    expected_subset_keys = {
        "source_rows",
        "first_error_rows",
        "all_correct_rows",
        "eligible_same_generator_first_error_rows",
        "available_all_correct_rows",
        "selected_error_generators",
        "selected_pairs",
        "excluded_from_transfer",
    }
    for subset in PROCESSBENCH_SUBSETS:
        subset_counts = by_subset.get(subset)
        if not isinstance(subset_counts, Mapping) or set(subset_counts) != expected_subset_keys:
            raise ValueError(f"ProcessBench transfer {subset} count fields mismatch")
        source_audit = PROCESSBENCH_SOURCE_AUDIT[subset]
        for field, expected in source_audit.items():
            if type(subset_counts.get(field)) is not int or subset_counts.get(field) != expected:
                raise ValueError(f"ProcessBench transfer {subset} count mismatch for {field}")
        if subset_counts.get("available_all_correct_rows") != source_audit["all_correct_rows"]:
            raise ValueError(f"ProcessBench transfer {subset} available-correct count mismatch")

        subset_pairs = [pair for pair in pairs if pair.first_error.subset == subset]
        if subset_counts.get("selected_pairs") != len(subset_pairs):
            raise ValueError(f"ProcessBench transfer {subset} selected-pair count mismatch")
        expected_generators = dict(
            sorted(Counter(pair.first_error.generator for pair in subset_pairs).items())
        )
        if not _same_json_payload(
            subset_counts.get("selected_error_generators"), expected_generators
        ):
            raise ValueError(f"ProcessBench transfer {subset} generator counts mismatch")
        expected_excluded = {
            "first_error_not_selected_or_no_available_generator_matched_counterpart": (
                source_audit["first_error_rows"] - len(subset_pairs)
            ),
            "all_correct_not_selected_or_not_used_by_unique_matching": (
                source_audit["all_correct_rows"] - len(subset_pairs)
            ),
        }
        if not _same_json_payload(
            subset_counts.get("excluded_from_transfer"), expected_excluded
        ):
            raise ValueError(f"ProcessBench transfer {subset} exclusion counts mismatch")


def _validate_run_roles(
    manifest: Mapping[str, Any],
    pairs: list[NaturalTransferPair],
) -> None:
    run_roles = manifest.get("run_roles")
    if not isinstance(run_roles, Mapping) or set(run_roles) != {"smoke", "full"}:
        raise ValueError("ProcessBench transfer manifest requires smoke and full run roles")
    role_payloads: dict[str, Mapping[str, Any]] = {}
    for role in ("smoke", "full"):
        payload = run_roles.get(role)
        if not isinstance(payload, Mapping) or set(payload) != {
            "pair_ids",
            "pair_count",
            "pairs_per_subset",
        }:
            raise ValueError(f"ProcessBench transfer {role} run-role contract mismatch")
        pair_ids = payload.get("pair_ids")
        if not isinstance(pair_ids, list) or any(
            not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids
        ):
            raise ValueError(f"ProcessBench transfer {role} pair_ids must be strings")
        if type(payload.get("pair_count")) is not int:
            raise ValueError(f"ProcessBench transfer {role} pair_count must be an integer")
        if type(payload.get("pairs_per_subset")) is not int:
            raise ValueError(
                f"ProcessBench transfer {role} pairs_per_subset must be an integer"
            )
        if payload.get("pair_count") != len(pair_ids):
            raise ValueError(f"ProcessBench transfer {role} pair_count mismatch")
        role_payloads[role] = payload

    all_ids = [pair.pair_id for pair in pairs]
    full_ids = list(role_payloads["full"]["pair_ids"])
    smoke_ids = list(role_payloads["smoke"]["pair_ids"])
    if full_ids != all_ids:
        raise ValueError("full run-role pair order must equal the manifest pair order")
    selection = manifest["selection"]
    full_per_subset = role_payloads["full"]["pairs_per_subset"]
    if full_per_subset != selection["pairs_per_subset"]:
        raise ValueError("full run-role pairs_per_subset must match the selection contract")
    full_counts = Counter(pair.first_error.subset for pair in pairs)
    if full_counts != Counter({subset: full_per_subset for subset in PROCESSBENCH_SUBSETS}):
        raise ValueError("full run-role pairs must be balanced across ProcessBench subsets")

    smoke_per_subset = role_payloads["smoke"]["pairs_per_subset"]
    if smoke_per_subset <= 0 or smoke_per_subset > full_per_subset:
        raise ValueError("smoke pairs_per_subset must be in [1, full pairs_per_subset]")
    expected_smoke_ids = [
        pair.pair_id
        for subset in PROCESSBENCH_SUBSETS
        for pair in [candidate for candidate in pairs if candidate.first_error.subset == subset][
            :smoke_per_subset
        ]
    ]
    if smoke_ids != expected_smoke_ids:
        raise ValueError("smoke run-role IDs must be the frozen first pairs of each subset")
