"""Typer CLI for RouteVerify pilot workflows."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from routeverify.analysis.discovery_effects import (
    analyze_discovery_run,
    write_discovery_effect_artifacts,
)
from routeverify.analysis.patch_effects import patch_effect_markdown, summarize_patch_rows
from routeverify.analysis.plots import build_diagnostic_plots
from routeverify.analysis.probe_directions import (
    load_probe_direction_artifact,
    produce_probe_direction_artifact,
    validate_router_direction_compatibility,
)
from routeverify.analysis.probes import (
    activation_norm_probe_summary,
    grouped_holdout_probe_summary,
    label_permutation_probe_summary,
    matched_negative_indices,
    mean_difference_probe_summary,
    metadata_probe_summary,
    probe_metrics_markdown,
    repeated_grouped_holdout_probe_summary,
    top_pc_probe_summary,
)
from routeverify.analysis.processbench_transfer import (
    summarize_processbench_transfer_rows,
)
from routeverify.analysis.router_subspace import analyze_router_subspaces
from routeverify.analysis.router_visibility import (
    router_visibility_markdown,
    router_visibility_metrics,
)
from routeverify.analysis.routing import (
    aligned_gate_weight_js_divergence,
    expert_set_turnover,
    fraction_changed,
    routing_metrics_markdown,
    topk_jaccard,
)
from routeverify.config import (
    DataConfig,
    ensure_external_cache,
    environment_snapshot,
    git_state,
    load_data_config,
    load_experiment_config,
    load_model_config,
    load_yaml,
)
from routeverify.data.prmbench import (
    PairEligibility,
    PRMBenchLoadResult,
    PRMBenchPair,
    audit_prmbench_example_pairs,
    load_prmbench_pairs_with_provenance,
    prmbench_selection_manifest,
    select_prmbench_pairs,
)
from routeverify.data.processbench import (
    PROCESSBENCH_ID,
    balanced_processbench_sample,
    load_processbench,
    write_examples_jsonl,
)
from routeverify.data.processbench_transfer import (
    load_processbench_transfer_bundle,
    processbench_transfer_selection_manifest,
)
from routeverify.data.prompts import (
    STEP_BINARY_CANDIDATES,
    PromptRender,
    continuation_pair,
    plain_trace,
    step_binary,
    tokenize_render,
)
from routeverify.data.prompts import (
    first_error as first_error_prompt,
)
from routeverify.data.schema import (
    TokenizedTrace,
    TraceExample,
    stable_hash,
    trace_example_from_json,
)
from routeverify.data.split_gate import (
    assert_split_local_matching_edges,
    select_partition_group_ids,
)
from routeverify.data.splits import deterministic_grouped_split, split_manifest
from routeverify.eval.continuation import continuation_margin, continuation_metrics
from routeverify.eval.processbench import (
    FirstErrorPrediction,
    first_error_candidate_labels,
    first_error_metrics,
    first_error_prediction_from_candidate,
)
from routeverify.eval.verification import binary_verification_metrics, verification_margin
from routeverify.instrumentation.patching import (
    interpolate_replacement,
)
from routeverify.instrumentation.router_weights import (
    export_router_weight_matrices,
    load_router_weight_export,
)
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchSpec,
)
from routeverify.instrumentation.storage import (
    atomic_write_json,
    atomic_write_jsonl,
    write_safetensors,
)
from routeverify.models.base import PreparedBatch
from routeverify.models.discovery import discover_from_model_id
from routeverify.models.moe import (
    DEFAULT_MOE_CONFIG_PATHS,
    MoEBlockLayout,
    MoEConfigPaths,
    RouterSemantics,
    canonical_site_manifest,
)
from routeverify.models.registry import REGISTRY, create_adapter
from routeverify.reporting.build_report import summary_markdown
from routeverify.reporting.common_core_smoke import (
    build_common_core_smoke_contract,
    common_core_smoke_review,
)
from routeverify.reporting.qwen_gate2 import qwen_gate2_review
from routeverify.reporting.qwen_smoke import qwen_smoke_review
from routeverify.reporting.second_batch import (
    build_olmoe_review_contract,
    olmoe_discovery_prerequisite_review,
    second_batch_frozen_discovery_review,
    second_batch_smoke_review,
)

console = Console()
app = typer.Typer(help="Causal audits of process-error signals in MoE reasoning models.")
data_app = typer.Typer(help="Download, normalize, and split datasets.")
model_app = typer.Typer(help="Inspect model module structure.")
eval_app = typer.Typer(help="Run behavior baselines.")
analyze_app = typer.Typer(help="Analyze saved activations and routes.")
intervene_app = typer.Typer(help="Run causal patching experiments.")

app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")
app.add_typer(analyze_app, name="analyze")
app.add_typer(intervene_app, name="intervene")


ConfigPath = Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)]
LimitOpt = Annotated[int | None, typer.Option("--limit", help="Maximum examples to process.")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Deterministic seed.")]
ExperimentSeedOpt = Annotated[
    int | None,
    typer.Option("--seed", help="Override the seed declared by the experiment config."),
]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Resolve and print work without running.")
]


def _print_config(title: str, config: object) -> None:
    console.rule(title)
    if hasattr(config, "model_dump"):
        console.print_json(data=config.model_dump(mode="json"))
    else:
        console.print(config)


@dataclass(frozen=True)
class _LoadedExamples:
    examples: list[TraceExample]
    provenance: dict[str, Any]


def _load_prmbench_selection_from_hf(
    data_config: DataConfig,
    *,
    limit: int | None,
    seed: int,
) -> PRMBenchLoadResult:
    _validate_prmbench_data_config(data_config)
    if (data_config.__pydantic_extra__ or {}).get("local_pair_file") is not None:
        from routeverify.data.frozen_prmbench import load_frozen_prmbench

        return load_frozen_prmbench(data_config, limit=limit, seed=seed)
    return load_prmbench_pairs_with_provenance(
        hf_id=str(data_config.hf_id),
        split=data_config.split,
        cache_dir=data_config.cache_dir,
        revision=data_config.revision,
        expected_resolved_data_revision=data_config.expected_resolved_data_revision,
        expected_raw_sha256=data_config.expected_raw_sha256,
        limit=limit,
        seed=seed,
    )


def _load_processbench_transfer_selection(
    data_config: DataConfig,
    *,
    limit: int | None,
) -> PRMBenchLoadResult:
    """Load frozen natural pairs through the generic paired patching container."""

    if limit is not None and limit < 0:
        raise typer.BadParameter("ProcessBench transfer pair limit must be non-negative")
    required = {
        "output": data_config.output,
        "selection_manifest": data_config.selection_manifest,
        "expected_processed_sha256": data_config.expected_processed_sha256,
        "expected_selection_manifest_sha256": data_config.expected_selection_manifest_sha256,
        "revision": data_config.revision,
        "expected_raw_sha256": data_config.expected_raw_sha256,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise typer.BadParameter(
            "ProcessBench transfer data config must pin: " + ", ".join(missing)
        )
    if data_config.validation_accessed is not False:
        raise typer.BadParameter(
            "ProcessBench transfer data config must record validation_accessed=false"
        )
    bundle = load_processbench_transfer_bundle(
        data_path=cast(Path, data_config.output),
        manifest_path=cast(Path, data_config.selection_manifest),
        expected_data_sha256=data_config.expected_processed_sha256,
        expected_manifest_sha256=data_config.expected_selection_manifest_sha256,
    )
    source = bundle.manifest["source"]
    if source.get("revision") != data_config.revision:
        raise typer.BadParameter("ProcessBench transfer revision does not match its manifest")
    source_files = source.get("files")
    observed_raw_hashes = {
        str(row.get("sha256"))
        for row in source_files
        if isinstance(row, Mapping)
    } if isinstance(source_files, list) else set()
    if data_config.expected_raw_sha256 not in observed_raw_hashes:
        raise typer.BadParameter("ProcessBench raw source SHA256 does not match its manifest")
    selected_natural = list(bundle.pairs if limit is None else bundle.pairs[:limit])
    selected_manifest = processbench_transfer_selection_manifest(
        bundle,
        pairs=selected_natural,
        requested_pair_limit=limit,
    )
    pairs = [
        PRMBenchPair(
            clean=pair.matched_correct,
            corrupt=pair.first_error,
            eligibility=PairEligibility(
                True,
                "frozen_processbench_natural_pair",
                index_base="zero",
            ),
        )
        for pair in selected_natural
    ]
    return PRMBenchLoadResult(
        pairs=pairs,
        source_provenance={
            "source_type": "frozen_processbench_transfer_bundle",
            "dataset": "processbench_transfer",
            "hf_id": source.get("hf_id"),
            "hf_config": source.get("hf_config"),
            "split": data_config.split,
            "requested_revision": data_config.revision,
            "resolved_data_revision": source.get("revision"),
            "raw_sha256": data_config.expected_raw_sha256,
            "source_files": source_files,
            "processed_file": {
                "path": str(data_config.output),
                "sha256": data_config.expected_processed_sha256,
                "row_count": len(bundle.examples),
            },
            "prepared_selection_manifest": {
                "path": str(data_config.selection_manifest),
                "sha256": data_config.expected_selection_manifest_sha256,
                "manifest_hash": bundle.manifest["manifest_hash"],
            },
            "selection_manifest": selected_manifest,
            "validation_accessed": False,
        },
        pair_audit={
            "input_row_count": len(bundle.examples),
            "candidate_pair_count": len(bundle.pairs),
            "complete_pair_count": len(pairs),
            "excluded_pair_count": len(bundle.pairs) - len(pairs),
            "roles": ["matched_correct", "first_error"],
            "compatibility_variants": ["clean", "corrupt"],
        },
        selection_manifest=selected_manifest,
    )


def _examples_and_provenance_from_processbench_transfer_result(
    result: PRMBenchLoadResult,
    *,
    data_config: DataConfig,
    limit: int | None,
    seed: int,
) -> _LoadedExamples:
    examples = [
        example
        for pair in result.pairs
        for example in (pair.clean, pair.corrupt)
    ]
    provenance = _with_selection_provenance(
        {
            **result.source_provenance,
            "selection_manifest": result.selection_manifest,
            "pair_audit": result.pair_audit,
        },
        examples,
        data_config=data_config,
        limit=limit,
        seed=seed,
    )
    provenance["selection"].update(
        {
            "unit": "matched_correct_first_error_pair",
            "pair_count": len(result.pairs),
            "rule": result.selection_manifest["selection_rule"],
            "manifest_hash": result.selection_manifest["manifest_hash"],
            "parent_manifest_hash": result.selection_manifest["parent_manifest_hash"],
        }
    )
    return _LoadedExamples(examples=examples, provenance=provenance)


def _validate_prmbench_data_config(data_config: DataConfig) -> None:
    missing = [
        name
        for name, value in (
            ("hf_id", data_config.hf_id),
            ("revision", data_config.revision),
            ("expected_resolved_data_revision", data_config.expected_resolved_data_revision),
            ("expected_raw_sha256", data_config.expected_raw_sha256),
        )
        if value is None
    ]
    if missing:
        raise typer.BadParameter(
            "PRMBench data config must pin source provenance fields: " + ", ".join(missing)
        )


def _examples_and_provenance_from_prmbench_result(
    result: PRMBenchLoadResult,
    *,
    data_config: DataConfig,
    limit: int | None,
    seed: int,
    fallback_from_processed_file: dict[str, Any] | None = None,
) -> _LoadedExamples:
    examples = [
        example
        for pair in result.pairs
        for example in (
            pair.clean,
            pair.corrupt,
        )
    ]
    base = {
        **result.source_provenance,
        "configured_hf_id": data_config.hf_id,
        "subset": data_config.subset or "preview",
        "revision": data_config.revision,
        "pair_audit": result.pair_audit,
        "selection_manifest": result.selection_manifest,
    }
    provenance = _with_selection_provenance(
        base,
        examples,
        data_config=data_config,
        limit=limit,
        seed=seed,
    )
    provenance["selection"].update(
        {
            "unit": "clean_corrupt_pair",
            "pair_count": len(result.pairs),
            "rule": result.selection_manifest["selection_rule"],
            "manifest_hash": result.selection_manifest["manifest_hash"],
        }
    )
    if fallback_from_processed_file is not None:
        provenance["fallback_from_processed_file"] = fallback_from_processed_file
    return _LoadedExamples(examples=examples, provenance=provenance)


def _load_examples(
    data_config: DataConfig,
    *,
    limit: int | None,
    seed: int = 0,
) -> list[TraceExample]:
    return _load_examples_with_provenance(data_config, limit=limit, seed=seed).examples


def _load_examples_with_provenance(
    data_config: DataConfig,
    *,
    limit: int | None,
    seed: int = 0,
) -> _LoadedExamples:
    if data_config.dataset == "processbench_transfer":
        result = _load_processbench_transfer_selection(data_config, limit=limit)
        return _examples_and_provenance_from_processbench_transfer_result(
            result,
            data_config=data_config,
            limit=limit,
            seed=seed,
        )
    if data_config.dataset == "prmbench":
        _validate_prmbench_data_config(data_config)
    processed_candidate: dict[str, Any] | None = None
    if data_config.output is not None and data_config.output.exists():
        processed_examples = _read_examples_file(data_config.output)
        examples = _filter_examples_by_split_manifest(
            processed_examples,
            data_config.output.with_suffix(".splits.json"),
            split=data_config.split,
        )
        processed_candidate = _processed_file_provenance(
            data_config,
            path=data_config.output,
            source_row_count=len(processed_examples),
            available_examples=examples,
        )
        if data_config.dataset == "prmbench":
            manifest_issue = _prmbench_processed_manifest_issue(
                processed_candidate,
                data_config=data_config,
                path=data_config.output,
            )
            if manifest_issue is not None:
                processed_candidate["fallback_reason"] = manifest_issue
            else:
                complete_pairs, pair_audit = audit_prmbench_example_pairs(examples)
                processed_candidate["pair_audit"] = pair_audit
                insufficient_cache = limit is not None and len(complete_pairs) < limit
                if not insufficient_cache:
                    selected_pairs = select_prmbench_pairs(
                        complete_pairs,
                        limit=limit,
                        seed=seed,
                    )
                    selection_manifest = prmbench_selection_manifest(
                        selected_pairs,
                        available_pair_count=len(complete_pairs),
                        limit=limit,
                        seed=seed,
                        source=processed_candidate,
                    )
                    selected = [
                        example for pair in selected_pairs for example in (pair.clean, pair.corrupt)
                    ]
                    provenance = _with_selection_provenance(
                        {
                            **processed_candidate,
                            "selection_manifest": selection_manifest,
                        },
                        selected,
                        data_config=data_config,
                        limit=limit,
                        seed=seed,
                    )
                    provenance["selection"].update(
                        {
                            "unit": "clean_corrupt_pair",
                            "pair_count": len(selected_pairs),
                            "rule": selection_manifest["selection_rule"],
                            "manifest_hash": selection_manifest["manifest_hash"],
                        }
                    )
                    return _LoadedExamples(examples=selected, provenance=provenance)
                processed_candidate["fallback_reason"] = {
                    "code": "insufficient_complete_pairs",
                    "requested_pair_limit": limit,
                    "available_complete_pairs": len(complete_pairs),
                }
        else:
            insufficient_processbench_cache = (
                data_config.dataset == "processbench"
                and limit is not None
                and len(examples) < limit
            )
            insufficient_balanced_cache = (
                data_config.dataset == "processbench"
                and data_config.balanced
                and limit is not None
                and (
                    len(examples) < limit
                    or not any(example.is_all_correct for example in examples)
                    or not any(not example.is_all_correct for example in examples)
                )
            )
            insufficient_cache = insufficient_processbench_cache or insufficient_balanced_cache
            if (
                data_config.dataset == "processbench"
                and data_config.balanced
                and limit is not None
                and not insufficient_cache
            ):
                selected = balanced_processbench_sample(examples, limit=limit, seed=seed)
                return _LoadedExamples(
                    examples=selected,
                    provenance=_with_selection_provenance(
                        processed_candidate,
                        selected,
                        data_config=data_config,
                        limit=limit,
                        seed=seed,
                    ),
                )
            if not insufficient_cache:
                selected = _stable_limit_examples(examples, limit=limit, seed=seed)
                return _LoadedExamples(
                    examples=selected,
                    provenance=_with_selection_provenance(
                        processed_candidate,
                        selected,
                        data_config=data_config,
                        limit=limit,
                        seed=seed,
                    ),
                )

    if data_config.dataset == "processbench":
        subset = data_config.subset or "gsm8k"
        load_limit = None if data_config.balanced and limit is not None else limit
        examples = load_processbench(
            subset=subset,
            split=data_config.split,
            cache_dir=data_config.cache_dir,
            revision=data_config.revision,
            limit=load_limit,
            seed=seed,
        )
        if data_config.balanced and limit is not None:
            examples = balanced_processbench_sample(examples, limit=limit, seed=seed)
        provenance = _raw_loader_provenance(
            data_config,
            hf_id=PROCESSBENCH_ID,
            subset=subset,
            selected_examples=examples,
            limit=limit,
            seed=seed,
        )
        if processed_candidate is not None:
            provenance["fallback_from_processed_file"] = processed_candidate
        return _LoadedExamples(examples=examples, provenance=provenance)
    if data_config.dataset == "prmbench":
        result = _load_prmbench_selection_from_hf(
            data_config,
            limit=limit,
            seed=seed,
        )
        return _examples_and_provenance_from_prmbench_result(
            result,
            data_config=data_config,
            limit=limit,
            seed=seed,
            fallback_from_processed_file=processed_candidate,
        )
    raise typer.BadParameter(f"unsupported dataset: {data_config.dataset}")


def _processed_file_provenance(
    data_config: DataConfig,
    *,
    path: Path,
    source_row_count: int,
    available_examples: list[TraceExample],
) -> dict[str, Any]:
    return {
        "source_type": "processed_jsonl",
        "dataset": data_config.dataset,
        "subset": data_config.subset,
        "split": data_config.split,
        "configured_hf_id": data_config.hf_id,
        "requested_revision": data_config.revision,
        "expected_resolved_data_revision": data_config.expected_resolved_data_revision,
        "expected_raw_sha256": data_config.expected_raw_sha256,
        "processed_file": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "row_count": source_row_count,
        },
        "split_manifest": _split_manifest_provenance(
            path.with_suffix(".splits.json"),
            split=data_config.split,
        ),
        "prepared_selection_manifest": _selection_manifest_file_provenance(
            path.with_suffix(".selection.json")
        ),
        "available_row_count": len(available_examples),
    }


def _prmbench_processed_manifest_issue(
    provenance: Mapping[str, Any],
    *,
    data_config: DataConfig,
    path: Path,
) -> dict[str, Any] | None:
    prepared = provenance.get("prepared_selection_manifest")
    manifest_path = path.with_suffix(".selection.json")
    if not isinstance(prepared, Mapping):
        return {
            "code": "missing_selection_manifest",
            "path": str(manifest_path),
        }

    source = prepared.get("source")
    expected = {
        "requested_revision": data_config.revision,
        "resolved_data_revision": data_config.expected_resolved_data_revision,
        "raw_sha256": data_config.expected_raw_sha256,
    }
    mismatches = {
        field: {
            "expected": value,
            "actual": source.get(field) if isinstance(source, Mapping) else None,
        }
        for field, value in expected.items()
        if not isinstance(source, Mapping) or source.get(field) != value
    }
    if mismatches:
        return {
            "code": "selection_manifest_source_mismatch",
            "path": str(manifest_path),
            "fields": mismatches,
        }
    return None


def _raw_loader_provenance(
    data_config: DataConfig,
    *,
    hf_id: str,
    subset: str,
    selected_examples: list[TraceExample],
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    return _with_selection_provenance(
        {
            "source_type": "hf_loader",
            "dataset": data_config.dataset,
            "hf_id": hf_id,
            "configured_hf_id": data_config.hf_id,
            "subset": subset,
            "split": data_config.split,
            "revision": data_config.revision,
            "cache_dir": str(data_config.cache_dir),
        },
        selected_examples,
        data_config=data_config,
        limit=limit,
        seed=seed,
    )


def _with_selection_provenance(
    provenance: dict[str, Any] | None,
    selected_examples: list[TraceExample],
    *,
    data_config: DataConfig,
    limit: int | None,
    seed: int,
) -> dict[str, Any]:
    if provenance is None:
        provenance = {
            "source_type": "unknown",
            "dataset": data_config.dataset,
            "subset": data_config.subset,
            "split": data_config.split,
        }
    selected_rows = _selected_row_provenance(selected_examples)
    return {
        **provenance,
        "selection": {
            "limit": limit,
            "seed": seed,
            "balanced": data_config.balanced,
            "row_count": len(selected_rows),
            "rows_hash": stable_hash(selected_rows, length=64),
            "rows": selected_rows,
        },
    }


def _selected_row_provenance(examples: list[TraceExample]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": example.sample_id,
            "dataset": example.dataset,
            "subset": example.subset,
            "source_id": example.metadata.get("source_id"),
            "source_idx": example.metadata.get("source_idx"),
            "content_hash": example.content_hash,
            "group_id": example.group_id,
            "first_error": example.first_error,
            "is_all_correct": example.is_all_correct,
            "generator": example.generator,
            "variant": example.variant,
        }
        for example in examples
    ]


def _split_manifest_provenance(path: Path, *, split: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        row_count = 0
        total_row_count = 0
    else:
        split_rows = manifest.get(split, [])
        row_count = len(split_rows) if isinstance(split_rows, list) else 0
        total_row_count = sum(len(value) for value in manifest.values() if isinstance(value, list))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "selected_split": split,
        "row_count": row_count,
        "total_row_count": total_row_count,
    }


def _selection_manifest_file_provenance(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"selection manifest must be a JSON object: {path}")
    recorded_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    computed_hash = stable_hash(unhashed, length=64)
    if recorded_hash != computed_hash:
        raise ValueError(
            f"selection manifest hash mismatch for {path}: "
            f"recorded={recorded_hash}, computed={computed_hash}"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "manifest_hash": recorded_hash,
        "selected_pair_count": manifest.get("selected_pair_count"),
        "source": manifest.get("source"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_sites(
    adapter: Any,
    site_kinds: list[str],
    *,
    layers: list[int] | None = None,
) -> tuple[LayerSite, ...]:
    available = adapter.discover_layers()
    requested = set(site_kinds)
    requested_layers = None if layers is None else set(layers)
    return tuple(
        site
        for site in available
        if site.kind in requested
        and (requested_layers is None or site.layer_index in requested_layers)
    )


def _discovered_router_block_indices(adapter: Any) -> tuple[int, ...]:
    """Return canonical absolute decoder indices with router-logit sites."""

    return tuple(
        sorted(
            {
                site.layer_index
                for site in adapter.discover_layers()
                if site.kind == "router_logits"
            }
        )
    )


def _adapter_moe_config_paths(alias: str) -> MoEConfigPaths:
    """Use registered adapter locators without blocking generic model inspection."""

    if alias not in REGISTRY:
        return DEFAULT_MOE_CONFIG_PATHS
    adapter = create_adapter(alias)
    config_paths = getattr(adapter, "moe_config_paths", DEFAULT_MOE_CONFIG_PATHS)
    if not isinstance(config_paths, MoEConfigPaths):
        raise TypeError("adapter.moe_config_paths must be MoEConfigPaths")
    return config_paths


def _capture_site_contract(adapter: Any, sites: list[LayerSite] | tuple[LayerSite, ...]) -> dict[str, Any]:
    """Serialize selected site identity and optional common MoE topology once per run."""

    layout: MoEBlockLayout | None = None
    layout_method = getattr(adapter, "moe_block_layout", None)
    if callable(layout_method):
        layout = layout_method()
        if not isinstance(layout, MoEBlockLayout):
            raise TypeError("adapter.moe_block_layout() must return MoEBlockLayout")
    semantics = getattr(adapter, "router_semantics", None)
    semantics_id = semantics.identifier if isinstance(semantics, RouterSemantics) else None
    return {
        "schema_version": 1,
        "adapter": {
            "alias": getattr(adapter, "alias", None),
            "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        },
        "moe_layout": None if layout is None else layout.to_metadata(),
        "router_semantics_id": semantics_id,
        "site_schema_version": 1,
        "sites": canonical_site_manifest(sites, layout=layout),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _append_working_jsonl(
    path: Path, rows: list[dict[str, Any]], *, initialize: bool = False
) -> None:
    """Opt-in, single-lineage working journal; canonical output remains unchanged."""
    if os.environ.get("ROUTEVERIFY_COMMON_CORE_WORKING_JOURNAL") != "1":
        return
    if not initialize and not rows:
        return
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if initialize:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if initialize else "a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if initialize:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _read_examples_file(path: Path) -> list[TraceExample]:
    return [trace_example_from_json(row) for row in _read_jsonl(path)]


def _filter_examples_by_split_manifest(
    examples: list[TraceExample],
    manifest_path: Path,
    *,
    split: str,
) -> list[TraceExample]:
    if not manifest_path.exists():
        return examples
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_ids = set(manifest.get(split, []))
    if not selected_ids:
        return []
    return [example for example in examples if example.sample_id in selected_ids]


def _stable_limit_examples(
    examples: list[TraceExample],
    *,
    limit: int | None,
    seed: int,
) -> list[TraceExample]:
    if limit is None:
        return sorted(examples, key=lambda example: example.sample_id)
    return sorted(
        examples,
        key=lambda example: stable_hash({"seed": seed, "sample_id": example.sample_id}),
    )[:limit]


def _resolve_artifact_path(run: Path, value: str | Path) -> Path:
    path = Path(value)
    candidates = (
        [path]
        if path.is_absolute()
        else [path, run / path, run / path.name, run / "activations" / path.name]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"artifact path not found: {value}")


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - exercised by model users
        raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc
    return cast(dict[str, Any], load_file(path))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            tensor = tensor.float()
        return cast(np.ndarray, tensor.numpy())
    return np.asarray(value)


def _manifest_error_label(row: dict[str, Any]) -> bool | None:
    if "is_error_boundary" in row:
        return bool(row["is_error_boundary"])
    if "label" in row:
        return bool(row["label"])
    if "is_all_correct" in row:
        return not bool(row["is_all_correct"])
    if "first_error" in row:
        return row["first_error"] is not None
    return None


def _manifest_array(value: Any, row: dict[str, Any]) -> np.ndarray:
    array = _tensor_to_numpy(value)
    token_index = int(row.get("token_index", 0))
    if array.ndim >= 2 and array.shape[0] == 1 and token_index < array.shape[1]:
        return array[:, token_index : token_index + 1, ...]
    return array


def _activation_vector(value: Any, row: dict[str, Any]) -> np.ndarray | None:
    array = _manifest_array(value, row)
    if not np.issubdtype(array.dtype, np.floating):
        return None
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _router_margin(logits: Any, selected_ids: Any) -> float | None:
    logits_array = np.asarray(_tensor_to_numpy(logits), dtype=np.float32).reshape(-1)
    ids = np.asarray(_tensor_to_numpy(selected_ids), dtype=np.int64).reshape(-1)
    if ids.size == 0 or logits_array.size == 0:
        return None
    valid_ids = ids[(ids >= 0) & (ids < logits_array.size)]
    if valid_ids.size == 0 or valid_ids.size >= logits_array.size:
        return None
    unselected = np.ones(logits_array.size, dtype=bool)
    unselected[valid_ids] = False
    return float(np.min(logits_array[valid_ids]) - np.max(logits_array[unselected]))


def _max_abs_tensor_delta(left: Any, right: Any) -> float:
    left_array = np.asarray(_tensor_to_numpy(left), dtype=np.float32)
    right_array = np.asarray(_tensor_to_numpy(right), dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ValueError(f"logit shapes differ: {left_array.shape} vs {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array)))


def _layer_index_from_site_key(key: str) -> int:
    layer_name = key.split(".", 1)[0]
    if not layer_name.startswith("layer_"):
        raise ValueError(f"site key does not start with layer_: {key}")
    return int(layer_name.removeprefix("layer_"))


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else float(sum(present) / len(present))


def _write_optional_parquet(path: Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    try:
        pandas = cast(Any, __import__("pandas"))
        path.parent.mkdir(parents=True, exist_ok=True)
        pandas.DataFrame(rows).to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional parquet stack
        return f"skipped: {exc}"
    return str(path)


@dataclass(frozen=True)
class _PatchPrompt:
    raw_render: PromptRender
    rendered_input: PromptRender
    step_end_tokens: tuple[int, ...]
    boundary_token: int
    provenance: dict[str, Any]


def _partition_validation_accessed(split_partition: str | None) -> bool | None:
    if split_partition == "discovery":
        return False
    if split_partition == "validation":
        return True
    return None


def _data_validation_accessed(
    data_config: DataConfig,
    *,
    split_partition: str | None,
) -> bool | None:
    if data_config.dataset == "processbench_transfer":
        if data_config.validation_accessed is not False:
            raise ValueError(
                "ProcessBench transfer runs must record validation_accessed=false"
            )
        return False
    return _partition_validation_accessed(split_partition)


def _annotate_processbench_transfer_row(row: dict[str, Any]) -> None:
    """Add public-data roles without changing the legacy patching compatibility keys."""

    role_by_variant = {
        "clean": "matched_correct",
        "corrupt": "first_error",
    }
    treatment_by_compatibility_name = {
        "unpatched": "unpatched",
        "unpatched_rerun": "unpatched_rerun",
        "self_patch": "self_patch",
        "clean_to_corrupt": "matched_correct_to_first_error",
        "corrupt_to_clean": "first_error_to_matched_correct",
    }
    receiver = row.get("receiver_variant")
    donor = row.get("donor_variant")
    treatment = row.get("treatment")
    row["receiver_role"] = role_by_variant.get(str(receiver))
    row["donor_role"] = (
        None if donor is None else role_by_variant.get(str(donor), str(donor))
    )
    row["treatment_semantic"] = treatment_by_compatibility_name.get(
        str(treatment), str(treatment)
    )


def _render_and_boundary_token(
    adapter: Any,
    example: TraceExample,
    *,
    boundary_step: int,
    prompt_mode: str,
    input_format: str,
) -> _PatchPrompt:
    if prompt_mode != "step_binary":
        raise ValueError("paired patching currently supports prompt_mode=step_binary only")
    raw_render = step_binary(example, step_index=boundary_step)
    formatter = getattr(adapter, "chat_completion_render", None)
    if input_format == "chat":
        if not callable(formatter):
            raise ValueError("input_format=chat requires adapter.chat_completion_render")
        position_render = formatter(raw_render)
        render_method = (
            f"{type(adapter).__module__}.{type(adapter).__qualname__}.chat_completion_render"
        )
    elif input_format == "raw":
        if callable(formatter):
            raise ValueError(
                "input_format=raw is incompatible with an adapter that implicitly chat-wraps "
                "candidate scoring"
            )
        position_render = raw_render
        render_method = "raw_prompt"
    else:
        raise ValueError(f"unsupported patch input_format: {input_format}")
    if not isinstance(position_render, PromptRender):
        raise TypeError("patch input renderer must return PromptRender")
    tokenized = tokenize_render(
        position_render, sample_id=example.sample_id, tokenizer=adapter._require_tokenizer()
    )
    provenance = {
        "prompt_mode": prompt_mode,
        "input_format": input_format,
        "render_method": render_method,
        "raw_prompt_hash": stable_hash({"prompt": raw_render.prompt}, length=64),
        "rendered_prompt_hash": stable_hash({"prompt": position_render.prompt}, length=64),
        "input_ids_hash": stable_hash({"input_ids": list(tokenized.input_ids)}, length=64),
        "tokenizer_class": (
            f"{type(adapter._require_tokenizer()).__module__}."
            f"{type(adapter._require_tokenizer()).__qualname__}"
        ),
        "add_special_tokens": True,
    }
    render_provenance = getattr(adapter, "harmony_render_provenance", None)
    if callable(render_provenance):
        adapter_provenance = render_provenance(raw_render.prompt)
        if adapter_provenance.get("rendered_prompt_hash") != provenance["rendered_prompt_hash"]:
            raise RuntimeError(
                "adapter render provenance does not match the boundary-alignment render"
            )
        provenance["adapter_render"] = adapter_provenance
    return _PatchPrompt(
        raw_render=raw_render,
        rendered_input=position_render,
        step_end_tokens=tokenized.step_end_tokens,
        boundary_token=tokenized.step_end_tokens[boundary_step],
        provenance=provenance,
    )


def _validate_patch_protocol(
    experiment_config: Any,
    *,
    adapter: Any | None = None,
) -> dict[str, Any]:
    if experiment_config.prompt_mode != "step_binary":
        raise typer.BadParameter("paired patching requires prompt_mode=step_binary")
    if experiment_config.input_format not in {"raw", "chat"}:
        raise typer.BadParameter("paired patching input_format must be raw or chat")
    if experiment_config.boundary_strategy != "paired_first_error":
        raise typer.BadParameter("paired patching requires boundary_strategy=paired_first_error")
    if adapter is not None:
        formatter = getattr(adapter, "chat_completion_render", None)
        if experiment_config.input_format == "chat" and not callable(formatter):
            raise typer.BadParameter(
                "paired patch input_format=chat requires adapter.chat_completion_render"
            )
        if experiment_config.input_format == "raw" and callable(formatter):
            raise typer.BadParameter(
                "adapter implicitly chat-wraps patch candidate scoring; configure input_format=chat"
            )
    if experiment_config.patch_mode not in {"single_site_sweep", "simultaneous_bundle"}:
        raise typer.BadParameter("patch_mode must be single_site_sweep or simultaneous_bundle")
    if not experiment_config.alphas:
        raise typer.BadParameter("alphas must be non-empty")
    if any(not np.isfinite(alpha) for alpha in experiment_config.alphas):
        raise typer.BadParameter("alphas must be finite")
    if experiment_config.patch_mode == "simultaneous_bundle":
        if experiment_config.sites != ["router_logits"]:
            raise typer.BadParameter(
                "simultaneous_bundle currently requires sites: [router_logits]"
            )
        if experiment_config.expected_router_hooks is None:
            raise typer.BadParameter("simultaneous_bundle requires expected_router_hooks")
        required_controls = {
            "unpatched",
            "unpatched_rerun",
            "self_patch",
            "alpha_zero_identity",
        }
        if experiment_config.forced_reroute_control:
            required_controls.add("forced_reroute_if_needed")
        elif "forced_reroute_if_needed" in experiment_config.controls:
            raise typer.BadParameter(
                "forced_reroute_if_needed control requires forced_reroute_control=true"
            )
        missing_controls = sorted(required_controls - set(experiment_config.controls))
        if missing_controls:
            raise typer.BadParameter(
                f"simultaneous_bundle config is missing required controls: {missing_controls}"
            )
    # For single-site mode, unpatched/rerun/self-patch rows are intrinsic to
    # the implementation and predate explicit ``controls`` fields in older
    # configs. Explicit-control configs list them for provenance; compatibility
    # configs remain valid because those rows are always emitted.
    if (experiment_config.split_manifest is None) != (experiment_config.split_partition is None):
        raise typer.BadParameter("split_manifest and split_partition must be configured together")
    protocol = {
        "prompt_mode": experiment_config.prompt_mode,
        "input_format": experiment_config.input_format,
        "boundary_strategy": experiment_config.boundary_strategy,
        "candidates": list(STEP_BINARY_CANDIDATES),
        "patch_mode": experiment_config.patch_mode,
        "alphas": [float(alpha) for alpha in experiment_config.alphas],
        "forced_reroute_control": bool(experiment_config.forced_reroute_control),
        "controls": list(experiment_config.controls),
        "split_manifest": (
            None
            if experiment_config.split_manifest is None
            else str(experiment_config.split_manifest)
        ),
        "split_partition": experiment_config.split_partition,
        "expected_router_hooks": experiment_config.expected_router_hooks,
    }
    if not experiment_config.include_equivalence_analysis:
        protocol["include_equivalence_analysis"] = False
    if (
        experiment_config.patch_mode == "single_site_sweep"
        and experiment_config.single_site_controls_explicit
    ):
        protocol["single_site_controls_explicit"] = True
    if experiment_config.patch_mode == "simultaneous_bundle":
        protocol["sites"] = list(experiment_config.sites)
        protocol["layers"] = (
            None
            if experiment_config.layers is None
            else list(experiment_config.layers)
        )
    return protocol


def _score_row(
    *,
    pair_id: str,
    receiver_variant: str,
    donor_variant: str | None,
    treatment: str,
    site: LayerSite | str | None,
    scores: Any,
    boundary_step: int,
    token_position: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "receiver_variant": receiver_variant,
        "donor_variant": donor_variant,
        "treatment": treatment,
        "site": (
            None
            if site is None
            else site
            if isinstance(site, str)
            else f"layer_{site.layer_index}.{site.kind}"
        ),
        "boundary_step": boundary_step,
        "token_position": token_position,
        "margin_incorrect_minus_correct": verification_margin(scores),
        "margin_incorrect_minus_correct_mean": verification_margin(
            scores, normalized=True
        ),
        "margin_incorrect_minus_correct_total": verification_margin(
            scores, normalized=False
        ),
        "primary_score": "mean_logprob",
        "scores": {key: value.__dict__ for key, value in scores.items()},
        "metadata": metadata or {},
    }


def _pair_key(pair: PRMBenchPair) -> str:
    return pair.clean.pair_id or pair.corrupt.pair_id or pair.clean.sample_id


def _site_key(site: LayerSite) -> str:
    return f"layer_{site.layer_index}.{site.kind}"


def _matched_control_pair(
    pairs: list[PRMBenchPair],
    current_index: int,
    *,
    seed: int,
) -> PRMBenchPair | None:
    """Select a deterministic matched donor from another PRMBench pair."""

    if len(pairs) < 2:
        return None
    current = pairs[current_index]
    current_error = current.corrupt.first_error
    current_type = current.corrupt.error_type

    def rank(candidate: PRMBenchPair) -> tuple[int, int, str]:
        candidate_error = candidate.corrupt.first_error
        distance = 10**9
        if current_error is not None and candidate_error is not None:
            distance = abs(current_error - candidate_error)
        type_mismatch = int(candidate.corrupt.error_type != current_type)
        tie_break = stable_hash(
            {
                "seed": seed,
                "current": _pair_key(current),
                "candidate": _pair_key(candidate),
            }
        )
        return type_mismatch, distance, tie_break

    candidates = [pair for index, pair in enumerate(pairs) if index != current_index]
    return min(candidates, key=rank)


def _norm_matched_random_array(reference: Any, *, seed: int, salt: str) -> np.ndarray:
    """Return a deterministic Gaussian tensor with the same shape and norm as reference."""

    if hasattr(reference, "detach"):
        reference_array = reference.detach().float().cpu().numpy()
    else:
        reference_array = np.asarray(reference, dtype=np.float32)
    rng_seed = int(stable_hash({"seed": seed, "salt": salt}, length=16), 16) % (2**32)
    rng = np.random.default_rng(rng_seed)
    random_array = rng.standard_normal(reference_array.shape).astype(np.float32)
    reference_norm = float(np.linalg.norm(reference_array.astype(np.float32)))
    random_norm = float(np.linalg.norm(random_array))
    if random_norm == 0.0:
        return np.zeros_like(random_array)
    return cast(np.ndarray, random_array * (reference_norm / random_norm))


def _norm_matched_random_delta_replacement(
    receiver: Any,
    donor: Any,
    *,
    seed: int,
    salt: str,
) -> np.ndarray:
    """Return receiver plus a random delta matched to the donor-receiver delta norm."""

    receiver_array = _tensor_to_numpy(receiver).astype(np.float32)
    donor_array = _tensor_to_numpy(donor).astype(np.float32)
    if receiver_array.shape != donor_array.shape:
        raise ValueError(
            f"receiver/donor shapes differ: {receiver_array.shape} vs {donor_array.shape}"
        )
    delta = donor_array - receiver_array
    random_delta = _norm_matched_random_array(delta, seed=seed, salt=salt)
    return cast(np.ndarray, receiver_array + random_delta)


def _candidate_site_replacements(
    captures_by_candidate: dict[str, Any],
    candidates: list[str],
    key: str,
) -> dict[str, Any]:
    return {candidate: captures_by_candidate[candidate].arrays[key] for candidate in candidates}


def _candidate_patch_bundles(
    *,
    receiver_captures: Mapping[str, Any],
    donor_captures: Mapping[str, Any],
    candidates: list[str],
    sites: tuple[LayerSite, ...],
    token_position: int,
    alpha: float,
    forced: bool = False,
) -> dict[str, PatchBundle]:
    bundles: dict[str, PatchBundle] = {}
    for candidate in candidates:
        patches: list[PatchSpec] = []
        for site in sites:
            key = _site_key(site)
            receiver = _tensor_to_numpy(receiver_captures[candidate].arrays[key])
            donor = _tensor_to_numpy(donor_captures[candidate].arrays[key])
            replacement = (
                receiver
                if forced
                else interpolate_replacement(receiver, donor, alpha=alpha)
            )
            patches.append(
                PatchSpec(
                    site=site,
                    token_position=token_position,
                    replacement=replacement,
                    metadata={"alpha": alpha, "forced_reroute": forced},
                )
            )
        bundles[candidate] = PatchBundle(
            patches=tuple(patches),
            metadata={"alpha": alpha, "forced_reroute": forced},
        )
    return bundles


def _summarize_candidate_patch_diagnostics(
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    expected_router_sites: set[str] | None = None,
) -> dict[str, Any]:
    site_metrics: list[Mapping[str, Any]] = []
    hook_counts: dict[str, int] = {}
    registered_counts: list[int] = []
    all_fired = True
    inferred_router_sites = {
        str(site)
        for candidate_diagnostics in diagnostics.values()
        for site in candidate_diagnostics.get("registered_sites", [])
        if str(site).endswith(".router_logits")
    }
    resolved_router_sites = (
        inferred_router_sites if expected_router_sites is None else expected_router_sites
    )
    router_dispatch_present: list[bool] = []
    router_dispatch_exact_coverage: list[bool] = []
    router_dispatch_verified: list[bool] = []
    for candidate_diagnostics in diagnostics.values():
        registered_counts.append(int(candidate_diagnostics["registered_site_count"]))
        all_fired = all_fired and bool(candidate_diagnostics["all_registered_hooks_fired"])
        for key, count in candidate_diagnostics["hook_fire_counts"].items():
            hook_counts[key] = hook_counts.get(key, 0) + int(count)
        site_metrics.extend(candidate_diagnostics["manipulation_by_site"].values())
        registered_router_sites = {
            str(site)
            for site in candidate_diagnostics.get("registered_sites", [])
            if str(site).endswith(".router_logits")
        }
        dispatch_by_site = candidate_diagnostics.get("dispatch_by_site")
        if resolved_router_sites:
            dispatch_mapping = (
                dispatch_by_site if isinstance(dispatch_by_site, Mapping) else {}
            )
            dispatch_sites = set(dispatch_mapping)
            exact_coverage = (
                registered_router_sites == resolved_router_sites
                and dispatch_sites == resolved_router_sites
            )
            router_dispatch_present.append(bool(dispatch_mapping))
            router_dispatch_exact_coverage.append(exact_coverage)
            router_dispatch_verified.append(
                exact_coverage
                and all(
                    isinstance(value, Mapping)
                    and value.get("downstream_dispatch_verified") is True
                    for value in dispatch_mapping.values()
                )
                and candidate_diagnostics.get("all_router_dispatch_verified") is True
            )
    dispatch_expected = bool(resolved_router_sites)
    complete_candidate_coverage = bool(diagnostics) and (
        len(router_dispatch_present) == len(diagnostics)
    )
    return {
        "schema_version": 1,
        "candidate_count": len(diagnostics),
        "candidate_diagnostics_present": bool(diagnostics),
        "registered_site_count": min(registered_counts) if registered_counts else 0,
        "all_registered_hooks_fired": all_fired,
        "router_dispatch_diagnostics_expected": dispatch_expected,
        "router_dispatch_diagnostics_present": (
            complete_candidate_coverage and all(router_dispatch_present)
            if dispatch_expected
            else False
        ),
        "router_dispatch_exact_site_coverage": (
            complete_candidate_coverage and all(router_dispatch_exact_coverage)
            if dispatch_expected
            else None
        ),
        "all_router_dispatch_verified": (
            complete_candidate_coverage and all(router_dispatch_verified)
            if dispatch_expected
            else None
        ),
        "hook_fire_counts": hook_counts,
        "topk_changed_rate": _mean_or_none(
            [
                float(metric["topk_changed_rate"])
                for metric in site_metrics
                if metric.get("topk_changed_rate") is not None
            ]
        ),
        "topk_overlap": _mean_or_none(
            [
                float(metric["topk_overlap"])
                for metric in site_metrics
                if metric.get("topk_overlap") is not None
            ]
        ),
        "router_kl_receiver_to_patched": _mean_or_none(
            [
                float(metric["router_kl_receiver_to_patched"])
                for metric in site_metrics
                if metric.get("router_kl_receiver_to_patched") is not None
            ]
        ),
        "candidate_diagnostics": diagnostics,
    }


def _score_candidate_single_site_patches(
    adapter: Any,
    prompt: str,
    candidates: list[str],
    *,
    site: LayerSite,
    token_position: int,
    replacements_by_candidate: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score a single-site intervention while retaining native dispatch diagnostics."""

    diagnostic_scorer = getattr(
        adapter, "score_candidates_with_candidate_patches_diagnostics", None
    )
    if not callable(diagnostic_scorer):
        scores = adapter.score_candidates_with_candidate_patches(
            prompt,
            candidates,
            site=site,
            token_position=token_position,
            replacements_by_candidate=replacements_by_candidate,
        )
        router_expected = site.kind == "router_logits"
        return scores, {
            "schema_version": None,
            "candidate_count": len(candidates),
            "candidate_diagnostics_present": False,
            "registered_site_count": None,
            "all_registered_hooks_fired": None,
            "router_dispatch_diagnostics_expected": router_expected,
            "router_dispatch_diagnostics_present": False,
            "router_dispatch_exact_site_coverage": False if router_expected else None,
            "all_router_dispatch_verified": False if router_expected else None,
            "hook_fire_counts": {},
            "candidate_diagnostics": {},
        }
    scores, diagnostics = diagnostic_scorer(
        prompt,
        candidates,
        site=site,
        token_position=token_position,
        replacements_by_candidate=replacements_by_candidate,
    )
    expected_router_sites = {_site_key(site)} if site.kind == "router_logits" else set()
    return scores, _summarize_candidate_patch_diagnostics(
        diagnostics,
        expected_router_sites=expected_router_sites,
    )


def _apply_split_first_selection(
    result: PRMBenchLoadResult,
    *,
    split_manifest_path: Path,
    split_partition: str,
    limit: int | None,
    seed: int,
) -> tuple[PRMBenchLoadResult, dict[str, str], dict[str, Any]]:
    selected_groups, partitions, split_payload = select_partition_group_ids(
        split_manifest_path,
        partition=split_partition,
    )
    by_group = {pair.clean.group_id: pair for pair in result.pairs}
    missing = [group_id for group_id in selected_groups if group_id not in by_group]
    if missing:
        raise ValueError(
            f"frozen split IDs are absent from the pinned dataset selection: {missing[:5]}"
        )
    pairs = [by_group[group_id] for group_id in selected_groups]
    if limit is not None:
        pairs = pairs[:limit]
    examples = [example for pair in pairs for example in (pair.clean, pair.corrupt)]
    _checked_pairs, pair_audit = audit_prmbench_example_pairs(examples)
    source = {
        **result.source_provenance,
        "split_first": True,
        "split_partition": split_partition,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_hash": split_payload.get("manifest_hash"),
    }
    selection = prmbench_selection_manifest(
        pairs,
        available_pair_count=len(selected_groups),
        limit=limit,
        seed=seed,
        source=source,
    )
    selection["selection_rule"] = {
        "name": "frozen_split_manifest_order",
        "split_partition": split_partition,
        "limit_semantics": "first_n_frozen_group_ids",
        "split_manifest_hash": split_payload.get("manifest_hash"),
        "no_cross_split_matching": True,
    }
    selection["frozen_split"] = {
        "path": str(split_manifest_path),
        "manifest_hash": split_payload.get("manifest_hash"),
        "partition": split_partition,
    }
    selection.pop("manifest_hash", None)
    selection["manifest_hash"] = stable_hash(selection, length=64)
    return (
        PRMBenchLoadResult(
            pairs=pairs,
            source_provenance=source,
            pair_audit=pair_audit,
            selection_manifest=selection,
        ),
        partitions,
        split_payload,
    )


def _boundary_steps_for_example(
    example: TraceExample,
    *,
    strategy: str = "prefix_until_error",
) -> tuple[int, ...]:
    if strategy == "prefix_until_error":
        if example.first_error is None:
            return tuple(range(len(example.steps)))
        return tuple(range(example.first_error + 1))
    if strategy == "first_error_or_final":
        return (example.first_error if example.first_error is not None else len(example.steps) - 1,)
    if strategy == "first_error_with_previous":
        if example.first_error is None:
            return (len(example.steps) - 1,)
        if example.first_error == 0:
            return (0,)
        return (example.first_error - 1, example.first_error)
    if strategy == "first_error_or_matched_all_correct":
        return (example.first_error if example.first_error is not None else len(example.steps) - 1,)
    raise ValueError(f"unsupported boundary_strategy: {strategy}")


def _capture_plain_trace_boundary_separately(strategy: str) -> bool:
    return strategy != "prefix_until_error"


def _boundary_label(example: TraceExample, boundary_step: int) -> bool:
    return example.first_error == boundary_step


def _cached_plain_trace(
    example: TraceExample,
    *,
    adapter: Any,
    tokenizer: Any,
    cache: dict[str, tuple[PromptRender, TokenizedTrace]],
    input_format: str = "raw",
) -> tuple[PromptRender, TokenizedTrace]:
    cached = cache.get(f"{input_format}:plain_trace:{example.sample_id}")
    if cached is not None:
        return cached
    render = _prepare_capture_render(adapter, plain_trace(example), input_format=input_format)
    tokenized = tokenize_render(render, sample_id=example.sample_id, tokenizer=tokenizer)
    cache[f"{input_format}:plain_trace:{example.sample_id}"] = (render, tokenized)
    return render, tokenized


def _prepare_capture_render(
    adapter: Any,
    render: PromptRender,
    *,
    input_format: str,
) -> PromptRender:
    if input_format == "raw":
        return render
    if input_format == "chat":
        formatter = getattr(adapter, "chat_completion_render", None)
        if not callable(formatter):
            raise ValueError("input_format=chat requires adapter.chat_completion_render")
        prepared = formatter(render)
        if not isinstance(prepared, PromptRender):
            raise TypeError("chat_completion_render must return PromptRender")
        return prepared
    raise ValueError(f"unsupported input_format: {input_format}")


def _boundary_render_and_position(
    example: TraceExample,
    *,
    adapter: Any,
    tokenizer: Any,
    prompt_mode: str,
    boundary_step: int,
    input_format: str,
    plain_trace_cache: dict[str, tuple[PromptRender, TokenizedTrace]],
) -> tuple[PromptRender, TokenizedTrace, int]:
    if prompt_mode == "plain_trace":
        render, tokenized = _cached_plain_trace(
            example,
            adapter=adapter,
            tokenizer=tokenizer,
            cache=plain_trace_cache,
            input_format=input_format,
        )
    elif prompt_mode == "step_binary":
        raw_render = step_binary(example, step_index=boundary_step)
        render = _prepare_capture_render(adapter, raw_render, input_format=input_format)
        tokenized = tokenize_render(
            render,
            sample_id=f"{example.sample_id}/{prompt_mode}/{input_format}/{boundary_step}",
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"unsupported prompt_mode: {prompt_mode}")
    return render, tokenized, tokenized.step_end_tokens[boundary_step]


def _matched_all_correct_boundary_plan(
    examples: list[TraceExample],
    *,
    adapter: Any,
    tokenizer: Any,
    cache: dict[str, tuple[PromptRender, TokenizedTrace]],
    prompt_mode: str = "plain_trace",
    input_format: str = "raw",
    seed: int,
) -> tuple[dict[str, tuple[int, ...]], dict[tuple[str, int], dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    all_correct: list[TraceExample] = []
    for example in examples:
        if example.first_error is not None:
            target_step = example.first_error
            _, _, token_position = _boundary_render_and_position(
                example,
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_mode=prompt_mode,
                boundary_step=target_step,
                input_format=input_format,
                plain_trace_cache=cache,
            )
            targets.append(
                {
                    "sample_id": example.sample_id,
                    "generator": example.generator,
                    "boundary_step": target_step,
                    "token_position": token_position,
                }
            )
        else:
            all_correct.append(example)

    boundary_steps: dict[str, tuple[int, ...]] = {}
    boundary_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for target in targets:
        step = int(target["boundary_step"])
        boundary_steps[str(target["sample_id"])] = (step,)
        boundary_metadata[(str(target["sample_id"]), step)] = {
            "boundary_match_role": "positive_target"
        }

    if not targets:
        for example in all_correct:
            step = len(example.steps) - 1
            boundary_steps[example.sample_id] = (step,)
            boundary_metadata[(example.sample_id, step)] = {
                "boundary_match_role": "negative_unmatched"
            }
        return boundary_steps, boundary_metadata

    # The boundary token position depends only on (candidate example, boundary_step) --
    # never on the target -- so memoize it once per (sample_id, boundary_step) and reuse it
    # across the target loop (and the unmatched-fallback ranking below). Without this,
    # step_binary/chat plans re-tokenize each candidate len(targets)x redundantly.
    candidate_token_positions: dict[tuple[str, int], int] = {}

    def _candidate_token_position(candidate: TraceExample, boundary_step: int) -> int:
        key = (candidate.sample_id, boundary_step)
        cached = candidate_token_positions.get(key)
        if cached is None:
            _, _, cached = _boundary_render_and_position(
                candidate,
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_mode=prompt_mode,
                boundary_step=boundary_step,
                input_format=input_format,
                plain_trace_cache=cache,
            )
            candidate_token_positions[key] = cached
        return cached

    candidate_pairs: list[tuple[float, str, int, int, int, int]] = []
    for example_index, example in enumerate(all_correct):
        for target_index, target in enumerate(targets):
            for boundary_step in range(len(example.steps)):
                token_position = _candidate_token_position(example, boundary_step)
                distance = _boundary_target_distance(
                    target=target,
                    candidate=example,
                    boundary_step=boundary_step,
                    token_position=token_position,
                )
                tie_break = stable_hash(
                    {
                        "seed": seed,
                        "target": target["sample_id"],
                        "candidate": example.sample_id,
                        "boundary_step": boundary_step,
                    }
                )
                candidate_pairs.append(
                    (
                        distance,
                        tie_break,
                        example_index,
                        target_index,
                        boundary_step,
                        token_position,
                    )
                )

    used_examples: set[int] = set()
    used_targets: set[int] = set()
    for distance, _, example_index, target_index, boundary_step, token_position in sorted(
        candidate_pairs
    ):
        if example_index in used_examples or target_index in used_targets:
            continue
        example = all_correct[example_index]
        target = targets[target_index]
        boundary_steps[example.sample_id] = (boundary_step,)
        boundary_metadata[(example.sample_id, boundary_step)] = _boundary_match_metadata(
            target=target,
            distance=distance,
            token_position=token_position,
        )
        used_examples.add(example_index)
        used_targets.add(target_index)

    for example_index, example in enumerate(all_correct):
        if example_index in used_examples:
            continue
        ranked = sorted(
            (
                (
                    _boundary_target_distance(
                        target=target,
                        candidate=example,
                        boundary_step=boundary_step,
                        token_position=token_position,
                    ),
                    stable_hash(
                        {
                            "seed": seed,
                            "target": target["sample_id"],
                            "candidate": example.sample_id,
                            "boundary_step": boundary_step,
                        }
                    ),
                    target,
                    boundary_step,
                    token_position,
                )
                for target in targets
                for boundary_step in range(len(example.steps))
                for token_position in (_candidate_token_position(example, boundary_step),)
            ),
            key=lambda item: (item[0], item[1]),
        )
        distance, _, target, boundary_step, token_position = ranked[0]
        boundary_steps[example.sample_id] = (boundary_step,)
        boundary_metadata[(example.sample_id, boundary_step)] = _boundary_match_metadata(
            target=target,
            distance=distance,
            token_position=token_position,
        )
    return boundary_steps, boundary_metadata


def _paired_first_error_boundary_plan(
    examples: list[TraceExample],
) -> tuple[dict[str, tuple[int, ...]], dict[tuple[str, int], dict[str, Any]]]:
    """Assign both members of each frozen pair to the corrupt member's first error."""

    grouped: dict[str, list[TraceExample]] = {}
    for example in examples:
        if example.pair_id is None:
            raise ValueError("paired_first_error requires pair_id on every example")
        grouped.setdefault(example.group_id, []).append(example)
    boundary_steps: dict[str, tuple[int, ...]] = {}
    metadata: dict[tuple[str, int], dict[str, Any]] = {}
    for group_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(
                f"paired_first_error requires exactly two rows per group: {group_id}"
            )
        by_variant = {member.variant: member for member in members}
        if set(by_variant) != {"clean", "corrupt"}:
            raise ValueError(
                f"paired_first_error requires clean/corrupt variants: {group_id}"
            )
        corrupt = by_variant["corrupt"]
        if corrupt.first_error is None:
            raise ValueError(f"corrupt pair member has no first_error: {group_id}")
        boundary = corrupt.first_error
        for member in members:
            if boundary >= len(member.steps):
                raise ValueError(
                    f"paired boundary {boundary} exceeds trace length for {member.sample_id}"
                )
            boundary_steps[member.sample_id] = (boundary,)
            metadata[(member.sample_id, boundary)] = {
                "boundary_match_role": "paired_first_error",
                "paired_group_id": group_id,
                "paired_corrupt_sample_id": corrupt.sample_id,
            }
    return boundary_steps, metadata


def _boundary_target_distance(
    *,
    target: dict[str, Any],
    candidate: TraceExample,
    boundary_step: int,
    token_position: int,
) -> float:
    distance = abs(boundary_step - int(target["boundary_step"])) / 4.0
    distance += abs(token_position - int(target["token_position"])) / 256.0
    distance += 5.0 * float(candidate.generator != target.get("generator"))
    return distance


def _boundary_match_metadata(
    *,
    target: dict[str, Any],
    distance: float,
    token_position: int,
) -> dict[str, Any]:
    return {
        "boundary_match_role": "negative_matched_all_correct",
        "matched_target_sample_id": target["sample_id"],
        "matched_target_generator": target.get("generator"),
        "matched_target_step": target["boundary_step"],
        "matched_target_token_position": target["token_position"],
        "matched_token_delta": token_position - int(target["token_position"]),
        "matched_step_delta": None,
        "boundary_match_distance": distance,
    }


def _capture_manifest_row(
    *,
    example: TraceExample,
    prompt_mode: str,
    input_format: str,
    boundary_strategy: str,
    activation_path: Path,
    boundary_step: int,
    token_position: int,
    token_index: int,
    num_positions_in_file: int,
    capture_arrays: dict[str, Any],
    render: PromptRender,
    tokenized: TokenizedTrace,
    boundary_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row_boundary_metadata = dict(boundary_metadata or {})
    if row_boundary_metadata.get("matched_target_step") is not None:
        row_boundary_metadata["matched_step_delta"] = boundary_step - int(
            row_boundary_metadata["matched_target_step"]
        )
    return {
        "sample_id": example.sample_id,
        "dataset": example.dataset,
        "subset": example.subset,
        "generator": example.generator,
        "pair_id": example.pair_id,
        "variant": example.variant,
        "transfer_role": example.metadata.get("transfer_role"),
        "boundary_source": example.metadata.get("boundary_source"),
        "mapped_boundary_step": example.metadata.get("mapped_boundary_step"),
        "error_type": example.error_type,
        "first_error": example.first_error,
        "error_steps": list(example.error_steps),
        "is_all_correct": example.is_all_correct,
        "is_error_boundary": _boundary_label(example, boundary_step),
        "group_id": example.group_id,
        "content_hash": example.content_hash,
        "prompt_mode": prompt_mode,
        "input_format": input_format,
        "prompt_hash": stable_hash({"prompt": render.prompt}, length=32),
        "input_ids_hash": stable_hash({"input_ids": list(tokenized.input_ids)}, length=32),
        "boundary_strategy": boundary_strategy,
        "activation_file": str(activation_path),
        "boundary_step": boundary_step,
        "token_position": token_position,
        "token_index": token_index,
        "num_positions_in_file": num_positions_in_file,
        "arrays": {key: tuple(value.shape) for key, value in capture_arrays.items()},
        **row_boundary_metadata,
    }


@app.callback()
def root() -> None:
    """RouteVerify command group."""


@data_app.command("prepare")
def data_prepare(
    config: ConfigPath,
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
) -> None:
    """Download and normalize a supported dataset."""

    paths = ensure_external_cache()
    data_config = load_data_config(config)
    _print_config("data config", data_config)
    console.print(f"HF dataset cache: {paths.dataset_cache}")
    if dry_run:
        return

    examples = []
    selection_manifest: dict[str, Any] | None = None
    if data_config.dataset == "processbench":
        subset = data_config.subset or "gsm8k"
        load_limit = None if data_config.balanced and limit is not None else limit
        examples = load_processbench(
            subset=subset,
            split=data_config.split,
            cache_dir=data_config.cache_dir,
            revision=data_config.revision,
            limit=load_limit,
            seed=seed,
        )
        if data_config.balanced and limit is not None:
            examples = balanced_processbench_sample(examples, limit=limit, seed=seed)
        output = data_config.output or Path("data/processed") / f"processbench-{subset}.jsonl"
        write_examples_jsonl(examples, output)
    elif data_config.dataset == "prmbench":
        result = _load_prmbench_selection_from_hf(data_config, limit=limit, seed=seed)
        for pair in result.pairs:
            examples.extend([pair.clean, pair.corrupt])
        selection_manifest = result.selection_manifest
        output = data_config.output or Path("data/processed/prmbench_pairs.jsonl")
        write_examples_jsonl(examples, output)
    else:
        raise typer.BadParameter(f"unsupported dataset: {data_config.dataset}")

    splits = deterministic_grouped_split(examples, seed=seed)
    manifest_path = output.with_suffix(".splits.json")
    atomic_write_json(manifest_path, split_manifest(splits))
    console.print(f"Wrote {len(examples)} examples to {output}")
    console.print(f"Wrote split manifest to {manifest_path}")
    if selection_manifest is not None:
        selection_path = output.with_suffix(".selection.json")
        atomic_write_json(selection_path, selection_manifest)
        console.print(f"Wrote pair selection manifest to {selection_path}")


@model_app.command("inspect")
def model_inspect(
    config: ConfigPath,
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
    load_weights: Annotated[
        bool,
        typer.Option("--load-weights", help="Load full checkpoint weights for discovery."),
    ] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Inspect a model and write a module-discovery report."""

    paths = ensure_external_cache()
    model_config = load_model_config(config)
    if load_weights:
        model_config = model_config.model_copy(update={"load_weights": True})
    _print_config("model config", model_config)
    console.print(f"HF model cache: {paths.model_cache}")
    if dry_run:
        return
    report = discover_from_model_id(
        model_config,
        limit_modules=limit,
        config_paths=_adapter_moe_config_paths(model_config.alias),
    )
    report.metadata["seed"] = seed
    output_path = output or Path("artifacts") / model_config.alias / "model_discovery.json"
    atomic_write_json(output_path, report.to_json())
    console.print(f"Wrote discovery report to {output_path}")


@model_app.command("check-capture")
def model_check_capture(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    data: Annotated[Path, typer.Option("--data", exists=True, readable=True)],
    limit: LimitOpt = 1,
    seed: SeedOpt = 0,
    tolerance: Annotated[
        float,
        typer.Option("--tolerance", help="Maximum allowed absolute logit delta."),
    ] = 1e-6,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Verify that capture hooks do not change model logits on a small example."""

    ensure_external_cache()
    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    data_config = load_data_config(data)
    _print_config("model config", model_config)
    if dry_run:
        return
    loaded_examples = _load_examples_with_provenance(data_config, limit=limit, seed=seed)
    examples = loaded_examples.examples
    if not examples:
        raise typer.BadParameter("no examples loaded")
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    example = examples[0]
    raw_render = plain_trace(example)
    render = _prepare_capture_render(
        adapter,
        raw_render,
        input_format=(
            "chat"
            if callable(getattr(adapter, "chat_completion_render", None))
            else "raw"
        ),
    )
    tokenizer = adapter._require_tokenizer()
    tokenized = tokenize_render(render, sample_id=example.sample_id, tokenizer=tokenizer)
    token_position = tokenized.step_end_tokens[0]
    router_blocks = _discovered_router_block_indices(adapter)
    if not router_blocks:
        raise typer.BadParameter("no router capture sites found")
    capture_block = router_blocks[0]
    sites = _selected_sites(
        adapter,
        [
            "pre_moe_residual",
            "router_input",
            "router_logits",
            "router_topk_ids",
            "router_topk_weights",
            "moe_output",
            "post_moe_residual",
        ],
        layers=[capture_block],
    )
    if not sites:
        raise typer.BadParameter(f"no capture sites found at routed block {capture_block}")
    batch = PreparedBatch(prompts=(render.prompt,), tokenized=(tokenized,))
    baseline_logits = adapter.forward_logits(batch)
    capture, captured_logits = adapter.capture_with_logits(
        batch,
        CaptureSpec(sites=sites, token_positions=(token_position,), dtype="float32"),
    )
    rerun_logits = adapter.forward_logits(batch)
    max_capture_delta = _max_abs_tensor_delta(baseline_logits, captured_logits)
    max_rerun_delta = _max_abs_tensor_delta(baseline_logits, rerun_logits)
    payload = {
        "model": model_config.model_dump(mode="json"),
        "dataset": data_config.dataset,
        "data_provenance": loaded_examples.provenance,
        "sample_id": example.sample_id,
        "input_format": "chat" if render.prompt != raw_render.prompt else "raw",
        "rendered_prompt_hash": stable_hash({"prompt": render.prompt}, length=64),
        "seed": seed,
        "block_index": capture_block,
        "token_position": token_position,
        "sites": sorted(capture.arrays),
        "max_abs_delta_baseline_vs_capture": max_capture_delta,
        "max_abs_delta_baseline_vs_rerun": max_rerun_delta,
        "tolerance": tolerance,
        "passed": max(max_capture_delta, max_rerun_delta) <= tolerance,
    }
    output_path = output or Path("artifacts") / model_config.alias / "capture_check.json"
    atomic_write_json(output_path, payload)
    console.print_json(data=payload)
    if not payload["passed"]:
        raise typer.Exit(code=1)


@model_app.command("runtime-preflight")
def model_runtime_preflight(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    dry_run: DryRunOpt = False,
) -> None:
    """Load a pinned adapter and record its actual routed-block contract."""

    ensure_external_cache()
    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    _print_config("model config", model_config)
    if dry_run:
        return
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    validator = getattr(adapter, "_validate_moe_layout", None)
    if callable(validator):
        validator()
    inspector = getattr(adapter, "runtime_inspect", None)
    if not callable(inspector):
        raise typer.BadParameter(
            f"adapter {model_config.alias} does not expose runtime_inspect"
        )
    runtime = inspector()
    sites = adapter.discover_layers()
    capture_contract = _capture_site_contract(adapter, sites)
    passed = bool(
        runtime.get("layout_consistent") is True
        and runtime.get("contract_matches", True) is True
    )
    payload = {
        "schema_version": 1,
        "passed": passed,
        "adapter_alias": model_config.alias,
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "model_id": model_config.model_id,
        "revision": model_config.revision,
        "tokenizer_revision": model_config.tokenizer_revision,
        "router_semantics_id": capture_contract.get("router_semantics_id"),
        "moe_layout": capture_contract.get("moe_layout"),
        "runtime_inspect": runtime,
        "environment": environment_snapshot(model=model_config),
        "validation_accessed": False,
    }
    atomic_write_json(output, payload)
    console.print_json(data=payload)
    console.print(f"Wrote runtime preflight to {output}")
    if not passed:
        raise typer.Exit(code=1)


@model_app.command("export-router-weights")
def model_export_router_weights(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    dry_run: DryRunOpt = False,
) -> None:
    """Export all router matrices during a pinned model load."""

    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    _print_config("model config", model_config)
    console.print(f"router matrix output: {output}")
    if dry_run:
        return
    ensure_external_cache()
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    manifest = export_router_weight_matrices(
        adapter=adapter,
        model_config=model_config,
        output=output,
    )
    console.print_json(data=manifest)
    console.print(f"Wrote router matrices to {output}")


@eval_app.command("behavior")
def eval_behavior(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    data: Annotated[Path, typer.Option("--data", exists=True, readable=True)],
    mode: Annotated[
        str,
        typer.Option("--mode", help="Behavior endpoint: step-binary or first-error."),
    ] = "step-binary",
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override behavior output directory."),
    ] = None,
    split_manifest: Annotated[
        Path | None,
        typer.Option(
            "--split-manifest",
            exists=True,
            readable=True,
            help="Frozen PRMBench split manifest; filtering occurs before --limit.",
        ),
    ] = None,
    split_partition: Annotated[
        str | None,
        typer.Option("--split-partition", help="Frozen partition (discovery or validation)."),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Run a behavior baseline on a small dataset slice."""

    ensure_external_cache()
    mode = mode.replace("_", "-")
    if mode not in {"step-binary", "first-error"}:
        raise typer.BadParameter("mode must be step-binary or first-error")
    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    data_config = load_data_config(data)
    if (split_manifest is None) != (split_partition is None):
        raise typer.BadParameter(
            "--split-manifest and --split-partition must be provided together"
        )
    if split_manifest is not None and data_config.dataset != "prmbench":
        raise typer.BadParameter("frozen split behavior selection currently requires PRMBench")
    table = Table("field", "value")
    table.add_row("model", model_config.model_id)
    table.add_row("dataset", data_config.dataset)
    table.add_row("mode", mode)
    table.add_row("limit", str(limit))
    table.add_row("seed", str(seed))
    table.add_row("split_manifest", str(split_manifest))
    table.add_row("split_partition", str(split_partition))
    console.print(table)
    if dry_run:
        return
    if mode == "first-error" and data_config.dataset != "processbench":
        raise typer.BadParameter("first-error behavior currently requires ProcessBench data")

    if split_manifest is not None and split_partition is not None:
        prmbench_result = _load_prmbench_selection_from_hf(
            data_config,
            limit=None,
            seed=seed,
        )
        prmbench_result, _partitions, _split_payload = _apply_split_first_selection(
            prmbench_result,
            split_manifest_path=split_manifest,
            split_partition=split_partition,
            limit=limit,
            seed=seed,
        )
        loaded_examples = _examples_and_provenance_from_prmbench_result(
            prmbench_result,
            data_config=data_config,
            limit=limit,
            seed=seed,
        )
    else:
        loaded_examples = _load_examples_with_provenance(data_config, limit=limit, seed=seed)
    examples = loaded_examples.examples
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)

    resolved_output_dir = (
        output_dir
        or Path("artifacts") / model_config.alias / "behavior" / f"{data_config.dataset}_{mode}"
    )
    working_rows = resolved_output_dir / "predictions.working.jsonl"
    if mode == "step-binary":
        _append_working_jsonl(working_rows, [], initialize=True)

    rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    if mode == "step-binary":
        rows = []
        for example in examples:
            mapped_boundary = example.metadata.get("mapped_boundary_step")
            if data_config.dataset == "processbench_transfer":
                if not isinstance(mapped_boundary, int):
                    raise ValueError(
                        f"ProcessBench transfer row lacks mapped boundary: {example.sample_id}"
                    )
                step_index = mapped_boundary
            else:
                step_index = (
                    example.first_error
                    if example.first_error is not None
                    else len(example.steps) - 1
                )
            gold = "INCORRECT" if example.first_error is not None else "CORRECT"
            prompt = step_binary(example, step_index=step_index).prompt
            scores = adapter.score_candidates(prompt, list(STEP_BINARY_CANDIDATES))
            render_provenance = getattr(adapter, "harmony_render_provenance", None)
            input_provenance = (
                render_provenance(prompt, include_text=True)
                if callable(render_provenance)
                else None
            )
            mean_margin = verification_margin(scores, normalized=True)
            total_margin = verification_margin(scores, normalized=False)
            pred = "INCORRECT" if mean_margin > 0 else "CORRECT"
            row: dict[str, Any] = {
                "sample_id": example.sample_id,
                "dataset": example.dataset,
                "subset": example.subset,
                "generator": example.generator,
                "pair_id": example.pair_id,
                "group_id": example.group_id,
                "variant": example.variant,
                "transfer_role": example.metadata.get("transfer_role"),
                "error_type": example.error_type,
                "mode": mode,
                "step_index": step_index,
                "gold": gold,
                "prediction": pred,
                "margin_incorrect_minus_correct": mean_margin,
                "margin_incorrect_minus_correct_mean": mean_margin,
                "margin_incorrect_minus_correct_total": total_margin,
                "primary_score": "mean_logprob",
                "scores": {key: value.__dict__ for key, value in scores.items()},
            }
            if input_provenance is not None:
                row["input_provenance"] = input_provenance
            rows.append(row)
            if len(rows) % 2 == 0:
                _append_working_jsonl(working_rows, rows[-2:])
        if len(rows) % 2:
            _append_working_jsonl(working_rows, rows[-1:])
        metrics = {"mode": mode, **binary_verification_metrics(rows)}
    else:
        rows = []
        predictions: list[FirstErrorPrediction] = []
        for example in examples:
            prompt = first_error_prompt(example).prompt
            candidates = first_error_candidate_labels(example)
            scores = adapter.score_candidates(prompt, candidates)
            prediction_label = max(candidates, key=lambda candidate: scores[candidate].mean_logprob)
            prediction = first_error_prediction_from_candidate(prediction_label)
            predictions.append(prediction)
            rows.append(
                {
                    "sample_id": example.sample_id,
                    "dataset": example.dataset,
                    "subset": example.subset,
                    "mode": mode,
                    "gold_first_error": example.first_error,
                    "prediction_first_error": prediction,
                    "prediction_label": prediction_label,
                    "prediction_source": "candidate_mean_logprob",
                    "scores": {key: value.__dict__ for key, value in scores.items()},
                }
            )
        metrics = {"mode": mode, **first_error_metrics(examples, predictions)}

    atomic_write_jsonl(resolved_output_dir / "predictions.jsonl", rows)
    atomic_write_json(resolved_output_dir / "metrics.json", metrics)
    behavior_environment = environment_snapshot(
        model=model_config,
        data=data_config,
        data_provenance=loaded_examples.provenance,
        seed=seed,
    )
    behavior_environment["split_partition"] = split_partition
    behavior_environment["validation_accessed"] = _data_validation_accessed(
        data_config,
        split_partition=split_partition,
    )
    atomic_write_json(resolved_output_dir / "environment.json", behavior_environment)
    if isinstance(loaded_examples.provenance.get("selection_manifest"), dict):
        selection_manifest_payload = loaded_examples.provenance["selection_manifest"]
        atomic_write_json(
            resolved_output_dir / "selection_manifest.json",
            selection_manifest_payload,
        )
        selected_rows = loaded_examples.provenance["selection"]["rows"]
        atomic_write_json(
            resolved_output_dir / "data_manifest.json",
            {
                "schema_version": 1,
                "pair_ids": sorted(
                    {
                        str(row["pair_id"])
                        for row in rows
                        if row.get("pair_id") is not None
                    }
                ),
                "group_ids": sorted(
                    {
                        str(row["group_id"])
                        for row in rows
                        if row.get("group_id") is not None
                    }
                ),
                "row_count": len(rows),
                "selection_manifest_hash": selection_manifest_payload.get("manifest_hash"),
                "selection_rows_hash": loaded_examples.provenance["selection"].get(
                    "rows_hash"
                ),
                "selected_rows_hash": stable_hash(selected_rows, length=64),
                "frozen_split": selection_manifest_payload.get("frozen_split"),
                "split_partition": split_partition,
                "validation_accessed": _data_validation_accessed(
                    data_config,
                    split_partition=split_partition,
                ),
            },
        )
    console.print_json(data=metrics)
    console.print(f"Wrote predictions to {resolved_output_dir / 'predictions.jsonl'}")


@eval_app.command("continuation")
def eval_continuation(
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    data: Annotated[Path, typer.Option("--data", exists=True, readable=True)],
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
) -> None:
    """Score correct versus corrupt PRMBench next-step continuations."""

    ensure_external_cache()
    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    data_config = load_data_config(data)
    if data_config.dataset != "prmbench":
        raise typer.BadParameter("continuation evaluation currently requires PRMBench data")
    table = Table("field", "value")
    table.add_row("model", model_config.model_id)
    table.add_row("dataset", data_config.dataset)
    table.add_row("limit", str(limit))
    table.add_row("seed", str(seed))
    console.print(table)
    if dry_run:
        return

    prmbench_result = _load_prmbench_selection_from_hf(
        data_config,
        limit=limit,
        seed=seed,
    )
    pairs = prmbench_result.pairs
    loaded_prmbench = _examples_and_provenance_from_prmbench_result(
        prmbench_result,
        data_config=data_config,
        limit=limit,
        seed=seed,
    )
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)

    rows = []
    for pair in pairs:
        prompt, correct_next, corrupt_next = continuation_pair(pair.clean, pair.corrupt)
        scores = adapter.score_candidates(prompt, [correct_next, corrupt_next])
        mean_margin = continuation_margin(
            scores,
            correct_candidate=correct_next,
            corrupt_candidate=corrupt_next,
            normalized=True,
        )
        total_margin = continuation_margin(
            scores,
            correct_candidate=correct_next,
            corrupt_candidate=corrupt_next,
            normalized=False,
        )
        rows.append(
            {
                "pair_id": _pair_key(pair),
                "boundary_step": pair.corrupt.first_error,
                "error_type": pair.corrupt.error_type,
                "correct_next_step": correct_next,
                "corrupt_next_step": corrupt_next,
                "margin_correct_minus_corrupt_mean": mean_margin,
                "margin_correct_minus_corrupt_total": total_margin,
                "scores": {key: value.__dict__ for key, value in scores.items()},
            }
        )

    metrics = continuation_metrics(rows)
    output_dir = Path("artifacts") / model_config.alias / "continuation" / data_config.dataset
    atomic_write_jsonl(output_dir / "predictions.jsonl", rows)
    atomic_write_json(output_dir / "metrics.json", metrics)
    atomic_write_json(
        output_dir / "environment.json",
        environment_snapshot(
            model=model_config,
            data=data_config,
            data_provenance=loaded_prmbench.provenance,
            seed=seed,
        ),
    )
    atomic_write_json(output_dir / "selection_manifest.json", prmbench_result.selection_manifest)
    console.print_json(data=metrics)
    console.print(f"Wrote continuation predictions to {output_dir / 'predictions.jsonl'}")


@app.command("extract")
def extract(
    experiment: Annotated[Path, typer.Option("--experiment", "-e", exists=True, readable=True)],
    limit: LimitOpt = None,
    seed: ExperimentSeedOpt = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Override experiment output directory."),
    ] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Capture step-boundary activations according to an experiment config."""

    ensure_external_cache()
    experiment_config = load_experiment_config(experiment)
    if limit is not None:
        experiment_config.limit = limit
    if output_dir is not None:
        experiment_config.output_dir = output_dir
    if seed is not None:
        experiment_config.seed = seed
    effective_seed = experiment_config.seed
    if experiment_config.prompt_mode not in {"plain_trace", "step_binary"}:
        raise typer.BadParameter("experiment prompt_mode must be plain_trace or step_binary")
    if experiment_config.input_format not in {"raw", "chat"}:
        raise typer.BadParameter("experiment input_format must be raw or chat")
    _print_config("experiment config", experiment_config)
    if dry_run:
        return

    model_config = load_model_config(experiment_config.model).model_copy(
        update={"load_weights": True}
    )
    data_config = load_data_config(experiment_config.data)
    extract_partition_by_group: dict[str, str] | None = None
    if experiment_config.split_manifest is not None:
        if data_config.dataset != "prmbench" or experiment_config.split_partition is None:
            raise typer.BadParameter(
                "split-first extraction requires PRMBench and split_partition"
            )
        extract_result = _load_prmbench_selection_from_hf(
            data_config,
            limit=None,
            seed=effective_seed,
        )
        extract_result, extract_partition_by_group, _split_payload = (
            _apply_split_first_selection(
                extract_result,
                split_manifest_path=experiment_config.split_manifest,
                split_partition=experiment_config.split_partition,
                limit=experiment_config.limit,
                seed=effective_seed,
            )
        )
        loaded_examples = _examples_and_provenance_from_prmbench_result(
            extract_result,
            data_config=data_config,
            limit=experiment_config.limit,
            seed=effective_seed,
        )
    else:
        loaded_examples = _load_examples_with_provenance(
            data_config,
            limit=experiment_config.limit,
            seed=effective_seed,
        )
    examples = loaded_examples.examples
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    sites = _selected_sites(adapter, experiment_config.sites, layers=experiment_config.layers)
    if not sites:
        raise typer.BadParameter("no capture sites matched the experiment config")

    output_dir = experiment_config.output_dir
    if (
        experiment_config.split_manifest is not None
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise typer.BadParameter(
            f"split-first capture refuses non-empty output_dir: {output_dir}"
        )
    activation_dir = output_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_snapshot(
        model=model_config,
        data=data_config,
        data_provenance=loaded_examples.provenance,
        seed=effective_seed,
    )
    environment["split_partition"] = experiment_config.split_partition
    environment["validation_accessed"] = _data_validation_accessed(
        data_config,
        split_partition=experiment_config.split_partition,
    )
    environment["capture_site_contract"] = _capture_site_contract(adapter, sites)
    atomic_write_json(output_dir / "environment.json", environment)
    if isinstance(loaded_examples.provenance.get("selection_manifest"), dict):
        atomic_write_json(
            output_dir / "selection_manifest.json",
            loaded_examples.provenance["selection_manifest"],
        )
    atomic_write_json(output_dir / "model_discovery.json", adapter.discovery_report.to_json())
    _atomic_write_text(
        output_dir / "config.resolved.yaml",
        yaml.safe_dump(experiment_config.model_dump(mode="json"), sort_keys=True),
    )

    rows = []
    working_manifest = output_dir / "data_manifest.working.jsonl"
    if experiment_config.prompt_mode == "step_binary":
        _append_working_jsonl(working_manifest, [], initialize=True)
    tokenizer = adapter._require_tokenizer()
    plain_trace_cache: dict[str, tuple[PromptRender, TokenizedTrace]] = {}
    boundary_step_overrides: dict[str, tuple[int, ...]] = {}
    boundary_metadata: dict[tuple[str, int], dict[str, Any]] = {}
    if experiment_config.boundary_strategy == "first_error_or_matched_all_correct":
        boundary_step_overrides, boundary_metadata = _matched_all_correct_boundary_plan(
            examples,
            adapter=adapter,
            tokenizer=tokenizer,
            cache=plain_trace_cache,
            prompt_mode=experiment_config.prompt_mode,
            input_format=experiment_config.input_format,
            seed=effective_seed,
        )
        if extract_partition_by_group is not None:
            sample_groups = {example.sample_id: example.group_id for example in examples}
            edges = [
                (
                    sample_groups[sample_id],
                    sample_groups[str(row["matched_target_sample_id"])],
                )
                for (sample_id, _boundary), row in boundary_metadata.items()
                if row.get("matched_target_sample_id") is not None
            ]
            assert_split_local_matching_edges(edges, extract_partition_by_group)
    elif experiment_config.boundary_strategy == "paired_first_error":
        boundary_step_overrides, boundary_metadata = _paired_first_error_boundary_plan(
            examples
        )
        if extract_partition_by_group is not None:
            assert_split_local_matching_edges(
                [
                    (members[0].group_id, members[1].group_id)
                    for members in (
                        [example for example in examples if example.group_id == group_id]
                        for group_id in sorted({example.group_id for example in examples})
                    )
                ],
                extract_partition_by_group,
            )
    for example in examples:
        if example.sample_id in boundary_step_overrides:
            boundary_steps = boundary_step_overrides[example.sample_id]
        else:
            boundary_steps = _boundary_steps_for_example(
                example,
                strategy=experiment_config.boundary_strategy,
            )
        if experiment_config.prompt_mode == "plain_trace":
            render, tokenized = _cached_plain_trace(
                example,
                adapter=adapter,
                tokenizer=tokenizer,
                cache=plain_trace_cache,
                input_format=experiment_config.input_format,
            )
            if _capture_plain_trace_boundary_separately(experiment_config.boundary_strategy):
                for boundary_step in boundary_steps:
                    token_position = tokenized.step_end_tokens[boundary_step]
                    row_boundary_metadata = boundary_metadata.get(
                        (example.sample_id, boundary_step)
                    )
                    capture = adapter.capture(
                        PreparedBatch(prompts=(render.prompt,), tokenized=(tokenized,)),
                        CaptureSpec(sites=sites, token_positions=(token_position,)),
                    )
                    activation_path = (
                        activation_dir
                        / f"{stable_hash(f'{example.sample_id}:{experiment_config.prompt_mode}:{experiment_config.boundary_strategy}:{boundary_step}')}.safetensors"
                    )
                    write_safetensors(activation_path, capture.arrays)
                    rows.append(
                        _capture_manifest_row(
                            example=example,
                            prompt_mode=experiment_config.prompt_mode,
                            input_format=experiment_config.input_format,
                            boundary_strategy=experiment_config.boundary_strategy,
                            activation_path=activation_path,
                            boundary_step=boundary_step,
                            token_position=token_position,
                            token_index=0,
                            num_positions_in_file=1,
                            capture_arrays=capture.arrays,
                            render=render,
                            tokenized=tokenized,
                            boundary_metadata=row_boundary_metadata,
                        )
                    )
                continue

            token_positions = tuple(tokenized.step_end_tokens[step] for step in boundary_steps)
            capture = adapter.capture(
                PreparedBatch(prompts=(render.prompt,), tokenized=(tokenized,)),
                CaptureSpec(sites=sites, token_positions=token_positions),
            )
            activation_path = activation_dir / f"{stable_hash(example.sample_id)}.safetensors"
            write_safetensors(activation_path, capture.arrays)
            for token_index, (boundary_step, token_position) in enumerate(
                zip(boundary_steps, token_positions, strict=True)
            ):
                row_boundary_metadata = boundary_metadata.get((example.sample_id, boundary_step))
                rows.append(
                    _capture_manifest_row(
                        example=example,
                        prompt_mode=experiment_config.prompt_mode,
                        input_format=experiment_config.input_format,
                        boundary_strategy=experiment_config.boundary_strategy,
                        activation_path=activation_path,
                        boundary_step=boundary_step,
                        token_position=token_position,
                        token_index=token_index,
                        num_positions_in_file=len(token_positions),
                        capture_arrays=capture.arrays,
                        render=render,
                        tokenized=tokenized,
                        boundary_metadata=row_boundary_metadata,
                    )
                )
        else:
            for boundary_step in boundary_steps:
                row_boundary_metadata = boundary_metadata.get((example.sample_id, boundary_step))
                render, tokenized, token_position = _boundary_render_and_position(
                    example,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt_mode=experiment_config.prompt_mode,
                    boundary_step=boundary_step,
                    input_format=experiment_config.input_format,
                    plain_trace_cache=plain_trace_cache,
                )
                capture = adapter.capture(
                    PreparedBatch(prompts=(render.prompt,), tokenized=(tokenized,)),
                    CaptureSpec(sites=sites, token_positions=(token_position,)),
                )
                activation_path = (
                    activation_dir
                    / f"{stable_hash(f'{example.sample_id}:{experiment_config.prompt_mode}:{boundary_step}')}.safetensors"
                )
                write_safetensors(activation_path, capture.arrays)
                rows.append(
                    _capture_manifest_row(
                        example=example,
                        prompt_mode=experiment_config.prompt_mode,
                        input_format=experiment_config.input_format,
                        boundary_strategy=experiment_config.boundary_strategy,
                        activation_path=activation_path,
                        boundary_step=boundary_step,
                        token_position=token_position,
                        token_index=0,
                        num_positions_in_file=1,
                        capture_arrays=capture.arrays,
                        render=render,
                        tokenized=tokenized,
                        boundary_metadata=row_boundary_metadata,
                    )
                )
                if experiment_config.prompt_mode == "step_binary" and len(rows) % 2 == 0:
                    _append_working_jsonl(working_manifest, rows[-2:])

    if experiment_config.prompt_mode == "step_binary" and len(rows) % 2:
        _append_working_jsonl(working_manifest, rows[-1:])
    atomic_write_jsonl(output_dir / "data_manifest.jsonl", rows)
    atomic_write_json(
        output_dir / "metrics.json",
        {
            "n": len(examples),
            "num_boundaries": len(rows),
            "num_sites": len(sites),
            "prompt_mode": experiment_config.prompt_mode,
            "input_format": experiment_config.input_format,
            "boundary_strategy": experiment_config.boundary_strategy,
        },
    )
    console.print(f"Wrote {len(rows)} boundary rows to {activation_dir}")


@analyze_app.command("probe")
def analyze_probe(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    repeated_holdouts: Annotated[
        int,
        typer.Option("--repeated-holdouts", min=0, help="Number of grouped holdout repeats."),
    ] = 5,
    label_permutations: Annotated[
        int,
        typer.Option("--label-permutations", min=0, help="Within-group label permutation nulls."),
    ] = 20,
    matched_negatives: Annotated[
        int,
        typer.Option("--matched-negatives", min=0, help="Metadata-nearest negatives per positive."),
    ] = 1,
    dry_run: DryRunOpt = False,
) -> None:
    """Fit low-capacity probe analyses from a run directory."""

    manifest_path = run / "data_manifest.jsonl"
    if not manifest_path.exists():
        raise typer.BadParameter(f"missing capture manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path, limit=limit)
    features_by_site: dict[str, list[np.ndarray]] = {}
    labels_by_site: dict[str, list[bool]] = {}
    groups_by_site: dict[str, list[str]] = {}
    rows_by_site: dict[str, list[dict[str, Any]]] = {}
    skipped_rows = 0

    for row in rows:
        label = _manifest_error_label(row)
        if label is None:
            skipped_rows += 1
            continue
        activation_path = _resolve_artifact_path(run, str(row["activation_file"]))
        arrays = _load_safetensors(activation_path)
        for key, value in arrays.items():
            vector = _activation_vector(value, row)
            if vector is None:
                continue
            existing = features_by_site.get(key)
            if existing is not None and existing and existing[0].shape != vector.shape:
                continue
            features_by_site.setdefault(key, []).append(vector)
            labels_by_site.setdefault(key, []).append(label)
            groups_by_site.setdefault(key, []).append(str(row.get("group_id") or row["sample_id"]))
            rows_by_site.setdefault(key, []).append(row)

    site_metrics: dict[str, Any] = {}
    for key, feature_rows in sorted(features_by_site.items()):
        features = np.stack(feature_rows)
        labels = np.asarray(labels_by_site[key], dtype=bool)
        summary = mean_difference_probe_summary(features, labels)
        summary["holdout"] = grouped_holdout_probe_summary(
            features,
            labels,
            groups_by_site[key],
            seed=seed,
        )
        summary["repeated_holdout"] = repeated_grouped_holdout_probe_summary(
            features,
            labels,
            groups_by_site[key],
            seed=seed,
            repeats=repeated_holdouts,
        )
        summary["metadata_baseline"] = metadata_probe_summary(rows_by_site[key], labels)
        summary["activation_norm_baseline"] = activation_norm_probe_summary(features, labels)
        summary["top_pc_baseline"] = top_pc_probe_summary(features, labels)
        summary["label_permutation_null"] = label_permutation_probe_summary(
            features,
            labels,
            groups_by_site[key],
            seed=seed,
            permutations=label_permutations,
        )
        matched_indices = matched_negative_indices(
            rows_by_site[key],
            labels,
            max_negatives_per_positive=matched_negatives,
            seed=seed,
        )
        if matched_indices and len(matched_indices) < labels.shape[0]:
            matched_features = features[matched_indices]
            matched_labels = labels[matched_indices]
            matched_groups = [groups_by_site[key][index] for index in matched_indices]
            matched_rows = [rows_by_site[key][index] for index in matched_indices]
            matched_summary = mean_difference_probe_summary(matched_features, matched_labels)
            matched_summary["num_selected_rows"] = len(matched_indices)
            matched_summary["metadata_baseline"] = metadata_probe_summary(
                matched_rows,
                matched_labels,
            )
            matched_summary["holdout"] = grouped_holdout_probe_summary(
                matched_features,
                matched_labels,
                matched_groups,
                seed=seed,
            )
            summary["matched_negative_subset"] = matched_summary
        else:
            summary["matched_negative_subset"] = {
                "status": "not_applied",
                "num_selected_rows": len(matched_indices),
            }
        site_metrics[key] = summary

    holdout_ok_sites = sum(
        1
        for value in site_metrics.values()
        if isinstance(value.get("holdout"), dict) and value["holdout"].get("status") == "ok"
    )
    payload = {
        "run": str(run),
        "limit": limit,
        "seed": seed,
        "dry_run": dry_run,
        "repeated_holdouts": repeated_holdouts,
        "label_permutations": label_permutations,
        "matched_negatives_per_positive": matched_negatives,
        "num_manifest_rows": len(rows),
        "skipped_rows_missing_label": skipped_rows,
        "num_holdout_ok_sites": holdout_ok_sites,
        "sites": site_metrics,
    }
    console.print_json(data=payload)
    if not dry_run:
        atomic_write_json(run / "analysis" / "probe_metrics.json", payload)
        console.print(f"Wrote {run / 'analysis' / 'probe_metrics.json'}")


@analyze_app.command("routing")
def analyze_routing(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
) -> None:
    """Compute route statistics from a run directory."""

    manifest_path = run / "data_manifest.jsonl"
    if not manifest_path.exists():
        raise typer.BadParameter(f"missing capture manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path, limit=limit)
    route_rows: list[dict[str, Any]] = []
    pair_inputs: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}

    for row in rows:
        activation_path = _resolve_artifact_path(run, str(row["activation_file"]))
        arrays = _load_safetensors(activation_path)
        for key, value in arrays.items():
            if not key.endswith(".router_topk_ids"):
                continue
            prefix = key.removesuffix(".router_topk_ids")
            ids = np.asarray(_manifest_array(value, row), dtype=np.int64).reshape(-1)
            weights_key = f"{prefix}.router_topk_weights"
            logits_key = f"{prefix}.router_logits"
            weights = None
            if weights_key in arrays:
                weights = np.asarray(
                    _manifest_array(arrays[weights_key], row), dtype=np.float32
                ).reshape(-1)
            margin_value = (
                _router_margin(_manifest_array(arrays[logits_key], row), ids)
                if logits_key in arrays
                else None
            )
            route_row = {
                "sample_id": row["sample_id"],
                "pair_id": row.get("pair_id"),
                "variant": row.get("variant"),
                "boundary_step": row.get("boundary_step"),
                "is_error_boundary": row.get("is_error_boundary"),
                "site": prefix,
                "topk_ids": ids.astype(int).tolist(),
                "topk_weights": None if weights is None else weights.astype(float).tolist(),
                "routing_margin": margin_value,
            }
            route_rows.append(route_row)
            pair_id = row.get("pair_id")
            variant = row.get("variant")
            if pair_id is not None and variant in {"clean", "corrupt"}:
                pair_inputs.setdefault((str(pair_id), prefix, int(row["boundary_step"])), {})[
                    str(variant)
                ] = route_row

    pair_rows: list[dict[str, Any]] = []
    for (pair_id, site, boundary_step), variants in sorted(pair_inputs.items()):
        clean = variants.get("clean")
        corrupt = variants.get("corrupt")
        if clean is None or corrupt is None:
            continue
        clean_weights = clean.get("topk_weights")
        corrupt_weights = corrupt.get("topk_weights")
        js_value = None
        if clean_weights is not None and corrupt_weights is not None:
            js_value = aligned_gate_weight_js_divergence(
                clean["topk_ids"],
                clean_weights,
                corrupt["topk_ids"],
                corrupt_weights,
            )
        clean_margin = cast(float | None, clean.get("routing_margin"))
        corrupt_margin = cast(float | None, corrupt.get("routing_margin"))
        pair_rows.append(
            {
                "pair_id": pair_id,
                "site": site,
                "boundary_step": boundary_step,
                "is_error_boundary": bool(corrupt.get("is_error_boundary")),
                "topk_jaccard": topk_jaccard(clean["topk_ids"], corrupt["topk_ids"]),
                "fraction_changed": fraction_changed(clean["topk_ids"], corrupt["topk_ids"]),
                "expert_set_turnover": expert_set_turnover(
                    clean["topk_ids"],
                    corrupt["topk_ids"],
                ),
                "gate_weight_js_divergence": js_value,
                "gate_weight_js_divergence_semantics": "expert_id_aligned_sparse_union",
                "clean_routing_margin": clean_margin,
                "corrupt_routing_margin": corrupt_margin,
                "routing_margin_delta_corrupt_minus_clean": (
                    None
                    if clean_margin is None or corrupt_margin is None
                    else corrupt_margin - clean_margin
                ),
            }
        )

    by_site: dict[str, dict[str, Any]] = {}
    for site in sorted({row["site"] for row in route_rows}):
        site_pairs = [row for row in pair_rows if row["site"] == site]
        error_site_pairs = [row for row in site_pairs if row["is_error_boundary"]]
        primary_pairs = error_site_pairs or site_pairs
        by_site[site] = {
            "n_routes": sum(1 for row in route_rows if row["site"] == site),
            "n_pairs": len({row["pair_id"] for row in site_pairs}),
            "n_pair_boundaries": len(site_pairs),
            "n_error_boundaries": len(error_site_pairs),
            "mean_topk_jaccard": _mean_or_none(
                [cast(float, row["topk_jaccard"]) for row in primary_pairs]
            ),
            "mean_fraction_changed": _mean_or_none(
                [cast(float, row["fraction_changed"]) for row in primary_pairs]
            ),
            "mean_expert_set_turnover": _mean_or_none(
                [cast(float, row["expert_set_turnover"]) for row in primary_pairs]
            ),
            "mean_gate_weight_js_divergence": _mean_or_none(
                [cast(float | None, row["gate_weight_js_divergence"]) for row in primary_pairs]
            ),
            "mean_routing_margin_delta_corrupt_minus_clean": _mean_or_none(
                [
                    cast(float | None, row["routing_margin_delta_corrupt_minus_clean"])
                    for row in primary_pairs
                ]
            ),
            "mean_topk_jaccard_all_boundaries": _mean_or_none(
                [cast(float, row["topk_jaccard"]) for row in site_pairs]
            ),
        }

    parquet_status = (
        None if dry_run else _write_optional_parquet(run / "routes.parquet", route_rows)
    )
    payload = {
        "run": str(run),
        "limit": limit,
        "seed": seed,
        "dry_run": dry_run,
        "num_manifest_rows": len(rows),
        "num_route_rows": len(route_rows),
        "num_pair_route_rows": len(pair_rows),
        "routes_parquet": parquet_status,
        "sites": by_site,
    }
    console.print_json(data=payload)
    if not dry_run:
        atomic_write_json(run / "analysis" / "routing_metrics.json", payload)
        atomic_write_jsonl(run / "analysis" / "routes.jsonl", route_rows)
        atomic_write_jsonl(run / "analysis" / "route_pair_stats.jsonl", pair_rows)
        console.print(f"Wrote {run / 'analysis' / 'routing_metrics.json'}")


@analyze_app.command("router-visibility")
def analyze_router_visibility(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    model: Annotated[Path, typer.Option("--model", exists=True, readable=True)],
    limit: LimitOpt = None,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", help="Centered finite-difference step in router-input units."),
    ] = 0.1,
    random_directions: Annotated[
        int,
        typer.Option("--random-directions", min=0, help="Norm-matched random directions per site."),
    ] = 8,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
) -> None:
    """Measure whether fitted error directions locally affect actual router logits."""

    ensure_external_cache()
    manifest_path = run / "data_manifest.jsonl"
    if not manifest_path.exists():
        raise typer.BadParameter(f"missing capture manifest: {manifest_path}")
    rows = _read_jsonl(manifest_path, limit=limit)
    features_by_site: dict[str, list[np.ndarray]] = {}
    labels_by_site: dict[str, list[bool]] = {}
    skipped_rows = 0
    for row in rows:
        label = _manifest_error_label(row)
        if label is None:
            skipped_rows += 1
            continue
        activation_path = _resolve_artifact_path(run, str(row["activation_file"]))
        arrays = _load_safetensors(activation_path)
        for key, value in arrays.items():
            if not key.endswith(".router_input"):
                continue
            vector = _activation_vector(value, row)
            if vector is None:
                continue
            existing = features_by_site.get(key)
            if existing is not None and existing and existing[0].shape != vector.shape:
                continue
            features_by_site.setdefault(key, []).append(vector)
            labels_by_site.setdefault(key, []).append(label)

    model_config = load_model_config(model).model_copy(update={"load_weights": True})
    adapter = create_adapter(model_config.alias)
    if features_by_site:
        adapter.load(model_config)

    site_metrics: dict[str, Any] = {}
    for key, feature_rows in sorted(features_by_site.items()):
        layer_index = _layer_index_from_site_key(key)
        gate = adapter._site_module(LayerSite(layer_index, "router_logits", ""))
        site_metrics[key] = router_visibility_metrics(
            gate,
            np.stack(feature_rows),
            np.asarray(labels_by_site[key], dtype=bool),
            epsilon=epsilon,
            random_directions=random_directions,
            seed=seed,
        )

    payload = {
        "run": str(run),
        "model": model_config.model_dump(mode="json"),
        "limit": limit,
        "seed": seed,
        "dry_run": dry_run,
        "epsilon": epsilon,
        "random_directions": random_directions,
        "num_manifest_rows": len(rows),
        "skipped_rows_missing_label": skipped_rows,
        "sites": site_metrics,
    }
    console.print_json(data=payload)
    if not dry_run:
        atomic_write_json(run / "analysis" / "router_visibility_metrics.json", payload)
        console.print(f"Wrote {run / 'analysis' / 'router_visibility_metrics.json'}")


@analyze_app.command("plots")
def analyze_plots(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    dry_run: DryRunOpt = False,
) -> None:
    """Build lightweight SVG plots from saved analysis metrics."""

    analysis_payload: dict[str, Any] = {}
    for name in ("probe_metrics", "routing_metrics", "router_visibility_metrics"):
        analysis_path = run / "analysis" / f"{name}.json"
        if analysis_path.exists():
            analysis_payload[name] = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not analysis_payload:
        raise typer.BadParameter(f"no analysis metrics found under {run / 'analysis'}")
    output_dir = run / "report" / "plots"
    plots = {} if dry_run else build_diagnostic_plots(analysis_payload, output_dir)
    payload = {"run": str(run), "dry_run": dry_run, "plots": plots}
    console.print_json(data=payload)


@analyze_app.command("discovery-effects")
def analyze_discovery_effects(
    run_root: Annotated[
        Path,
        typer.Option(
            "--run-root",
            exists=True,
            file_okay=False,
            help="Finalized frozen-discovery run root; validation is never opened.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="New independent directory for derived CPU-analysis artifacts.",
        ),
    ],
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260731,
    bootstrap_resamples: Annotated[
        int,
        typer.Option("--bootstrap-resamples", min=2),
    ] = 50_000,
    permutation_resamples: Annotated[
        int,
        typer.Option("--permutation-resamples", min=1),
    ] = 50_000,
    ci_alpha: Annotated[
        float,
        typer.Option("--ci-alpha", min=0.000001, max=0.999999),
    ] = 0.05,
) -> None:
    """Run pair-clustered CPU analysis on a finalized discovery package."""

    artifacts = analyze_discovery_run(
        run_root,
        seed=seed,
        n_resamples=bootstrap_resamples,
        ci_alpha=ci_alpha,
        permutation_resamples=permutation_resamples,
    )
    payload = write_discovery_effect_artifacts(artifacts, output_dir)
    console.print_json(data=payload)
    console.print(f"Wrote frozen-discovery CPU analysis to {output_dir}")


@analyze_app.command("router-subspace")
def analyze_router_subspace(
    weights: Annotated[Path, typer.Option("--weights", exists=True, readable=True)],
    directions: Annotated[Path, typer.Option("--directions", exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    rtol: Annotated[float | None, typer.Option("--rtol")] = None,
    atol: Annotated[float, typer.Option("--atol")] = 0.0,
    dry_run: DryRunOpt = False,
) -> None:
    """Project probe directions into exported router row/null spaces on CPU."""

    payload = {
        "weights": str(weights),
        "weights_sha256": _sha256_file(weights),
        "weights_manifest": str(weights.with_suffix(weights.suffix + ".manifest.json")),
        "directions": str(directions),
        "directions_sha256": _sha256_file(directions),
        "rtol": rtol,
        "atol": atol,
    }
    console.print_json(data=payload)
    if dry_run:
        return
    matrices, matrix_manifest = load_router_weight_export(weights)
    layer_directions, direction_manifest = load_probe_direction_artifact(directions)
    validate_router_direction_compatibility(
        matrix_manifest=matrix_manifest,
        direction_manifest=direction_manifest,
        matrix_layers=set(matrices),
        direction_layers=set(layer_directions),
    )
    result = analyze_router_subspaces(
        matrices,
        layer_directions,
        rtol=rtol,
        atol=atol,
    )
    result["provenance"] = {
        **payload,
        "router_weight_manifest": matrix_manifest,
        "probe_direction_manifest": direction_manifest,
        "environment": environment_snapshot(),
    }
    atomic_write_json(output, result)
    console.print_json(data=result)
    console.print(f"Wrote router subspace analysis to {output}")


@analyze_app.command("probe-direction")
def analyze_probe_direction(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    site_kind: Annotated[str, typer.Option("--site-kind")] = "router_input",
    dry_run: DryRunOpt = False,
) -> None:
    """Produce a paired corrupt-minus-clean direction from a frozen recapture."""

    if site_kind != "router_input":
        raise typer.BadParameter("--site-kind must be router_input")
    payload = {
        "run": str(run),
        "output": str(output),
        "site_kind": site_kind,
        "direction_definition": (
            "unit_normalized_mean_paired_corrupt_minus_clean_router_input"
        ),
    }
    console.print_json(data=payload)
    if dry_run:
        return
    manifest = produce_probe_direction_artifact(
        run=run,
        output=output,
        site_kind=site_kind,
    )
    console.print_json(data=manifest)
    console.print(f"Wrote probe directions to {output}")


@intervene_app.command("patch")
def intervene_patch(
    experiment: Annotated[Path, typer.Option("--experiment", "-e", exists=True, readable=True)],
    limit: LimitOpt = None,
    seed: ExperimentSeedOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Run paired clean/corrupt patching experiments."""

    ensure_external_cache()
    experiment_config = load_experiment_config(experiment)
    if limit is not None:
        experiment_config.limit = limit
    if seed is not None:
        experiment_config.seed = seed
    effective_seed = experiment_config.seed
    protocol = _validate_patch_protocol(experiment_config)
    _print_config("experiment config", experiment_config)
    if dry_run:
        return

    model_config = load_model_config(experiment_config.model).model_copy(
        update={"load_weights": True}
    )
    data_config = load_data_config(experiment_config.data)
    legacy_single_site_controls = (
        data_config.dataset != "processbench_transfer"
        and not experiment_config.single_site_controls_explicit
    )
    if data_config.dataset == "prmbench":
        prmbench_result = _load_prmbench_selection_from_hf(
            data_config,
            limit=(
                None
                if experiment_config.split_manifest is not None
                else experiment_config.limit
            ),
            seed=effective_seed,
        )
    elif data_config.dataset == "processbench_transfer":
        if experiment_config.split_manifest is not None:
            raise typer.BadParameter(
                "ProcessBench transfer uses its frozen selection manifest, not a split manifest"
            )
        if experiment_config.include_equivalence_analysis:
            raise typer.BadParameter(
                "ProcessBench transfer requires include_equivalence_analysis=false"
            )
        prmbench_result = _load_processbench_transfer_selection(
            data_config,
            limit=experiment_config.limit,
        )
    else:
        raise typer.BadParameter(
            "paired patching requires PRMBench or frozen ProcessBench transfer data"
        )
    partition_by_group: dict[str, str] | None = None
    frozen_split_payload: dict[str, Any] | None = None
    if (
        experiment_config.split_manifest is not None
        and experiment_config.split_partition is not None
    ):
        prmbench_result, partition_by_group, frozen_split_payload = _apply_split_first_selection(
            prmbench_result,
            split_manifest_path=experiment_config.split_manifest,
            split_partition=experiment_config.split_partition,
            limit=experiment_config.limit,
            seed=effective_seed,
        )
    pairs = prmbench_result.pairs
    if data_config.dataset == "processbench_transfer":
        loaded_prmbench = _examples_and_provenance_from_processbench_transfer_result(
            prmbench_result,
            data_config=data_config,
            limit=experiment_config.limit,
            seed=effective_seed,
        )
    else:
        loaded_prmbench = _examples_and_provenance_from_prmbench_result(
            prmbench_result,
            data_config=data_config,
            limit=experiment_config.limit,
            seed=effective_seed,
        )
    adapter = create_adapter(model_config.alias)
    adapter.load(model_config)
    protocol = _validate_patch_protocol(experiment_config, adapter=adapter)
    sites = tuple(
        site
        for site in _selected_sites(
            adapter, experiment_config.sites, layers=experiment_config.layers
        )
        if site.kind in {"moe_output", "post_moe_residual", "router_logits"}
    )
    if not sites:
        raise typer.BadParameter("no supported patch sites matched the experiment config")

    output_dir = experiment_config.output_dir
    if (
        experiment_config.patch_mode == "simultaneous_bundle"
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise typer.BadParameter(
            f"simultaneous bundle refuses non-empty output_dir: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    router_weight_manifest = None
    if experiment_config.export_router_weights is not None:
        router_weight_output = experiment_config.export_router_weights
        if not router_weight_output.is_absolute():
            router_weight_output = output_dir / router_weight_output
        router_weight_manifest = export_router_weight_matrices(
            adapter=adapter,
            model_config=model_config,
            output=router_weight_output,
            schema_version=experiment_config.router_weight_schema_version,
        )
    command = {"argv": list(sys.argv), "shell": shlex.join(sys.argv)}
    config_paths = {
        "experiment": Path(experiment),
        "model": Path(experiment_config.model),
        "data": Path(experiment_config.data),
    }
    config_sources = {
        name: {
            "path": str(path),
            "sha256": _sha256_file(path) if path.exists() else None,
        }
        for name, path in config_paths.items()
    }
    resolved_config = {
        **experiment_config.model_dump(mode="json"),
        "resolved_model": model_config.model_dump(mode="json"),
        "resolved_data": data_config.model_dump(mode="json"),
        "actual_protocol": protocol,
        "command": command,
        "config_sources": config_sources,
    }
    environment = environment_snapshot(
        model=model_config,
        data=data_config,
        data_provenance=loaded_prmbench.provenance,
        seed=effective_seed,
    )
    environment["command"] = command
    environment["config_sources"] = config_sources
    environment["resolved_experiment"] = resolved_config
    environment["split_partition"] = experiment_config.split_partition
    environment["validation_accessed"] = _data_validation_accessed(
        data_config,
        split_partition=experiment_config.split_partition,
    )
    environment["capture_site_contract"] = _capture_site_contract(adapter, sites)
    atomic_write_json(
        output_dir / "environment.json",
        environment,
    )
    atomic_write_json(output_dir / "selection_manifest.json", prmbench_result.selection_manifest)
    atomic_write_json(output_dir / "model_discovery.json", adapter.discovery_report.to_json())
    _atomic_write_text(
        output_dir / "config.resolved.yaml",
        yaml.safe_dump(resolved_config, sort_keys=True),
    )

    rows: list[dict[str, Any]] = []
    working_rows = output_dir / "predictions.working.jsonl"
    _append_working_jsonl(working_rows, [], initialize=True)
    candidates = list(STEP_BINARY_CANDIDATES)
    pair_prompt_inputs: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        pair_row_start = len(rows)
        clean = pair.clean
        corrupt = pair.corrupt
        if partition_by_group is not None:
            assert_split_local_matching_edges(
                [(clean.group_id, corrupt.group_id)],
                partition_by_group,
            )
        if corrupt.first_error is None:
            continue
        boundary_step = corrupt.first_error
        clean_prompt = _render_and_boundary_token(
            adapter,
            clean,
            boundary_step=boundary_step,
            prompt_mode=experiment_config.prompt_mode,
            input_format=experiment_config.input_format,
        )
        corrupt_prompt = _render_and_boundary_token(
            adapter,
            corrupt,
            boundary_step=boundary_step,
            prompt_mode=experiment_config.prompt_mode,
            input_format=experiment_config.input_format,
        )
        clean_render = clean_prompt.raw_render
        corrupt_render = corrupt_prompt.raw_render
        clean_step_tokens = clean_prompt.step_end_tokens
        corrupt_step_tokens = corrupt_prompt.step_end_tokens
        clean_token = clean_prompt.boundary_token
        corrupt_token = corrupt_prompt.boundary_token
        pair_id = _pair_key(pair)
        clean_prompt_record = dict(clean_prompt.provenance)
        corrupt_prompt_record = dict(corrupt_prompt.provenance)
        if bool(getattr(adapter, "persist_rendered_text", False)):
            clean_prompt_record["rendered_prompt_text"] = clean_prompt.rendered_input.prompt
            corrupt_prompt_record["rendered_prompt_text"] = corrupt_prompt.rendered_input.prompt
        pair_prompt_inputs.append(
            {
                "pair_id": pair_id,
                **(
                    {
                        "matched_correct": clean_prompt_record,
                        "first_error": corrupt_prompt_record,
                    }
                    if data_config.dataset == "processbench_transfer"
                    else {
                        "clean": clean_prompt_record,
                        "corrupt": corrupt_prompt_record,
                    }
                ),
            }
        )

        clean_baseline = adapter.score_candidates(clean_render.prompt, candidates)
        clean_rerun = adapter.score_candidates(clean_render.prompt, candidates)
        corrupt_baseline = adapter.score_candidates(corrupt_render.prompt, candidates)
        corrupt_rerun = adapter.score_candidates(corrupt_render.prompt, candidates)
        rows.append(
            _score_row(
                pair_id=pair_id,
                receiver_variant="clean",
                donor_variant=None,
                treatment="unpatched",
                site=None,
                scores=clean_baseline,
                boundary_step=boundary_step,
                token_position=clean_token,
            )
        )
        rows.append(
            _score_row(
                pair_id=pair_id,
                receiver_variant="clean",
                donor_variant=None,
                treatment="unpatched_rerun",
                site=None,
                scores=clean_rerun,
                boundary_step=boundary_step,
                token_position=clean_token,
            )
        )
        rows.append(
            _score_row(
                pair_id=pair_id,
                receiver_variant="corrupt",
                donor_variant=None,
                treatment="unpatched",
                site=None,
                scores=corrupt_baseline,
                boundary_step=boundary_step,
                token_position=corrupt_token,
            )
        )
        rows.append(
            _score_row(
                pair_id=pair_id,
                receiver_variant="corrupt",
                donor_variant=None,
                treatment="unpatched_rerun",
                site=None,
                scores=corrupt_rerun,
                boundary_step=boundary_step,
                token_position=corrupt_token,
            )
        )

        clean_capture = adapter.capture_candidate_activations(
            clean_render.prompt,
            candidates,
            CaptureSpec(sites=sites, token_positions=(clean_token,), dtype="float32"),
        )
        corrupt_capture = adapter.capture_candidate_activations(
            corrupt_render.prompt,
            candidates,
            CaptureSpec(sites=sites, token_positions=(corrupt_token,), dtype="float32"),
        )
        if experiment_config.patch_mode == "simultaneous_bundle":
            expected_hooks = int(experiment_config.expected_router_hooks or 0)
            if len(sites) != expected_hooks:
                raise RuntimeError(
                    f"simultaneous bundle expected {expected_hooks} router sites, "
                    f"resolved {len(sites)}"
                )
            bundle_label = f"bundle.router_logits[{len(sites)}]"
            expected_bundle_router_sites = {_site_key(site) for site in sites}
            clean_self_scores, clean_self_diagnostics = (
                adapter.score_candidates_with_candidate_patch_bundles(
                    clean_render.prompt,
                    candidates,
                    bundles_by_candidate=_candidate_patch_bundles(
                        receiver_captures=clean_capture,
                        donor_captures=clean_capture,
                        candidates=candidates,
                        sites=sites,
                        token_position=clean_token,
                        alpha=1.0,
                    ),
                )
            )
            corrupt_self_scores, corrupt_self_diagnostics = (
                adapter.score_candidates_with_candidate_patch_bundles(
                    corrupt_render.prompt,
                    candidates,
                    bundles_by_candidate=_candidate_patch_bundles(
                        receiver_captures=corrupt_capture,
                        donor_captures=corrupt_capture,
                        candidates=candidates,
                        sites=sites,
                        token_position=corrupt_token,
                        alpha=1.0,
                    ),
                )
            )
            rows.extend(
                [
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="clean",
                        donor_variant="clean",
                        treatment="self_patch",
                        site=bundle_label,
                        scores=clean_self_scores,
                        boundary_step=boundary_step,
                        token_position=clean_token,
                        metadata={
                            "patch_mode": "simultaneous_bundle",
                            "manipulation": _summarize_candidate_patch_diagnostics(
                                clean_self_diagnostics,
                                expected_router_sites=expected_bundle_router_sites,
                            ),
                        },
                    ),
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="corrupt",
                        donor_variant="corrupt",
                        treatment="self_patch",
                        site=bundle_label,
                        scores=corrupt_self_scores,
                        boundary_step=boundary_step,
                        token_position=corrupt_token,
                        metadata={
                            "patch_mode": "simultaneous_bundle",
                            "manipulation": _summarize_candidate_patch_diagnostics(
                                corrupt_self_diagnostics,
                                expected_router_sites=expected_bundle_router_sites,
                            ),
                        },
                    ),
                ]
            )
            direction_changed: dict[str, float] = {
                "clean_to_corrupt": 0.0,
                "corrupt_to_clean": 0.0,
            }
            for alpha in experiment_config.alphas:
                treatments = (
                    (
                        "clean_to_corrupt",
                        corrupt_render.prompt,
                        corrupt_capture,
                        clean_capture,
                        "corrupt",
                        "clean",
                        corrupt_token,
                    ),
                    (
                        "corrupt_to_clean",
                        clean_render.prompt,
                        clean_capture,
                        corrupt_capture,
                        "clean",
                        "corrupt",
                        clean_token,
                    ),
                )
                for (
                    treatment,
                    receiver_prompt,
                    receiver_capture,
                    donor_capture,
                    receiver_variant,
                    donor_variant,
                    receiver_token,
                ) in treatments:
                    scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
                        receiver_prompt,
                        candidates,
                        bundles_by_candidate=_candidate_patch_bundles(
                            receiver_captures=receiver_capture,
                            donor_captures=donor_capture,
                            candidates=candidates,
                            sites=sites,
                            token_position=receiver_token,
                            alpha=float(alpha),
                        ),
                    )
                    manipulation = _summarize_candidate_patch_diagnostics(
                        diagnostics,
                        expected_router_sites=expected_bundle_router_sites,
                    )
                    changed = float(manipulation["topk_changed_rate"] or 0.0)
                    direction_changed[treatment] = max(direction_changed[treatment], changed)
                    rows.append(
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant=receiver_variant,
                            donor_variant=donor_variant,
                            treatment=treatment,
                            site=bundle_label,
                            scores=scores,
                            boundary_step=boundary_step,
                            token_position=receiver_token,
                            metadata={
                                "patch_mode": "simultaneous_bundle",
                                "alpha": float(alpha),
                                "control": "dose_response",
                                "manipulation": manipulation,
                            },
                        )
                    )
            if experiment_config.forced_reroute_control:
                forced_treatments = (
                    (
                        "forced_reroute_clean_receiver",
                        clean_render.prompt,
                        clean_capture,
                        corrupt_capture,
                        "clean",
                        clean_token,
                        "corrupt_to_clean",
                    ),
                    (
                        "forced_reroute_corrupt_receiver",
                        corrupt_render.prompt,
                        corrupt_capture,
                        clean_capture,
                        "corrupt",
                        corrupt_token,
                        "clean_to_corrupt",
                    ),
                )
                for (
                    treatment,
                    receiver_prompt,
                    receiver_capture,
                    donor_capture,
                    receiver_variant,
                    receiver_token,
                    dose_direction,
                ) in forced_treatments:
                    if direction_changed[dose_direction] > 0.01:
                        continue
                    scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
                        receiver_prompt,
                        candidates,
                        bundles_by_candidate=_candidate_patch_bundles(
                            receiver_captures=receiver_capture,
                            donor_captures=donor_capture,
                            candidates=candidates,
                            sites=sites,
                            token_position=receiver_token,
                            alpha=0.0,
                            forced=True,
                        ),
                    )
                    rows.append(
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant=receiver_variant,
                            donor_variant="forced",
                            treatment=treatment,
                            site=bundle_label,
                            scores=scores,
                            boundary_step=boundary_step,
                            token_position=receiver_token,
                            metadata={
                                "patch_mode": "simultaneous_bundle",
                                "control": "forced_reroute_smoke_only",
                                "activation_reason": (
                                    f"max alpha top-k change for {dose_direction} "
                                    f"was {direction_changed[dose_direction]:.6g}"
                                ),
                                "manipulation": _summarize_candidate_patch_diagnostics(
                                    diagnostics,
                                    expected_router_sites=expected_bundle_router_sites,
                                ),
                            },
                        )
                    )
            for row in rows[pair_row_start:]:
                receiver = str(row["receiver_variant"])
                row["input_provenance"] = (
                    clean_prompt.provenance if receiver == "clean" else corrupt_prompt.provenance
                )
                if data_config.dataset == "processbench_transfer":
                    _annotate_processbench_transfer_row(row)
            _append_working_jsonl(working_rows, rows[pair_row_start:])
            continue
        clean_pre_capture = None
        corrupt_pre_capture = None
        clean_pre_token = None
        corrupt_pre_token = None
        if (
            (
                legacy_single_site_controls
                or "pre_error_when_available" in experiment_config.controls
            )
            and boundary_step > 0
        ):
            clean_pre_token = clean_step_tokens[boundary_step - 1]
            corrupt_pre_token = corrupt_step_tokens[boundary_step - 1]
            clean_pre_capture = adapter.capture_candidate_activations(
                clean_render.prompt,
                candidates,
                CaptureSpec(sites=sites, token_positions=(clean_pre_token,), dtype="float32"),
            )
            corrupt_pre_capture = adapter.capture_candidate_activations(
                corrupt_render.prompt,
                candidates,
                CaptureSpec(sites=sites, token_positions=(corrupt_pre_token,), dtype="float32"),
            )

        control_pair = (
            _matched_control_pair(pairs, pair_index, seed=effective_seed)
            if legacy_single_site_controls
            or "random_matched_donor" in experiment_config.controls
            else None
        )
        control_clean_capture = None
        control_corrupt_capture = None
        control_pair_id = None
        if control_pair is not None and control_pair.corrupt.first_error is not None:
            control_pair_id = _pair_key(control_pair)
            if partition_by_group is not None:
                assert_split_local_matching_edges(
                    [(pair.clean.group_id, control_pair.clean.group_id)],
                    partition_by_group,
                )
            control_boundary = control_pair.corrupt.first_error
            control_clean_prompt = _render_and_boundary_token(
                adapter,
                control_pair.clean,
                boundary_step=control_boundary,
                prompt_mode=experiment_config.prompt_mode,
                input_format=experiment_config.input_format,
            )
            control_corrupt_prompt = _render_and_boundary_token(
                adapter,
                control_pair.corrupt,
                boundary_step=control_boundary,
                prompt_mode=experiment_config.prompt_mode,
                input_format=experiment_config.input_format,
            )
            control_clean_capture = adapter.capture_candidate_activations(
                control_clean_prompt.raw_render.prompt,
                candidates,
                CaptureSpec(
                    sites=sites,
                    token_positions=(control_clean_prompt.boundary_token,),
                    dtype="float32",
                ),
            )
            control_corrupt_capture = adapter.capture_candidate_activations(
                control_corrupt_prompt.raw_render.prompt,
                candidates,
                CaptureSpec(
                    sites=sites,
                    token_positions=(control_corrupt_prompt.boundary_token,),
                    dtype="float32",
                ),
            )

        for site in sites:
            key = _site_key(site)
            clean_self, clean_self_manipulation = _score_candidate_single_site_patches(
                adapter,
                clean_render.prompt,
                candidates,
                site=site,
                token_position=clean_token,
                replacements_by_candidate=_candidate_site_replacements(
                    clean_capture, candidates, key
                ),
            )
            corrupt_self, corrupt_self_manipulation = _score_candidate_single_site_patches(
                adapter,
                corrupt_render.prompt,
                candidates,
                site=site,
                token_position=corrupt_token,
                replacements_by_candidate=_candidate_site_replacements(
                    corrupt_capture, candidates, key
                ),
            )
            clean_to_corrupt, clean_to_corrupt_manipulation = (
                _score_candidate_single_site_patches(
                    adapter,
                    corrupt_render.prompt,
                    candidates,
                    site=site,
                    token_position=corrupt_token,
                    replacements_by_candidate=_candidate_site_replacements(
                        clean_capture, candidates, key
                    ),
                )
            )
            corrupt_to_clean, corrupt_to_clean_manipulation = (
                _score_candidate_single_site_patches(
                    adapter,
                    clean_render.prompt,
                    candidates,
                    site=site,
                    token_position=clean_token,
                    replacements_by_candidate=_candidate_site_replacements(
                        corrupt_capture, candidates, key
                    ),
                )
            )
            rows.extend(
                [
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="clean",
                        donor_variant="clean",
                        treatment="self_patch",
                        site=site,
                        scores=clean_self,
                        boundary_step=boundary_step,
                        token_position=clean_token,
                        metadata={
                            "patch_mode": "single_site_sweep",
                            "alpha": 1.0,
                            "manipulation": clean_self_manipulation,
                        },
                    ),
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="corrupt",
                        donor_variant="corrupt",
                        treatment="self_patch",
                        site=site,
                        scores=corrupt_self,
                        boundary_step=boundary_step,
                        token_position=corrupt_token,
                        metadata={
                            "patch_mode": "single_site_sweep",
                            "alpha": 1.0,
                            "manipulation": corrupt_self_manipulation,
                        },
                    ),
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="corrupt",
                        donor_variant="clean",
                        treatment="clean_to_corrupt",
                        site=site,
                        scores=clean_to_corrupt,
                        boundary_step=boundary_step,
                        token_position=corrupt_token,
                        metadata={
                            "patch_mode": "single_site_sweep",
                            "alpha": 1.0,
                            "manipulation": clean_to_corrupt_manipulation,
                        },
                    ),
                    _score_row(
                        pair_id=pair_id,
                        receiver_variant="clean",
                        donor_variant="corrupt",
                        treatment="corrupt_to_clean",
                        site=site,
                        scores=corrupt_to_clean,
                        boundary_step=boundary_step,
                        token_position=clean_token,
                        metadata={
                            "patch_mode": "single_site_sweep",
                            "alpha": 1.0,
                            "manipulation": corrupt_to_clean_manipulation,
                        },
                    ),
                ]
            )
            if (
                legacy_single_site_controls
                or "norm_random_delta" in experiment_config.controls
            ):
                clean_random_delta, clean_random_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        clean_render.prompt,
                        candidates,
                        site=site,
                        token_position=clean_token,
                        replacements_by_candidate={
                            candidate: _norm_matched_random_delta_replacement(
                                clean_capture[candidate].arrays[key],
                                corrupt_capture[candidate].arrays[key],
                                seed=effective_seed,
                                salt=(
                                    f"{pair_id}:clean:{key}:"
                                    f"norm_random_delta:{candidate}"
                                ),
                            )
                            for candidate in candidates
                        },
                    )
                )
                corrupt_random_delta, corrupt_random_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        corrupt_render.prompt,
                        candidates,
                        site=site,
                        token_position=corrupt_token,
                        replacements_by_candidate={
                            candidate: _norm_matched_random_delta_replacement(
                                corrupt_capture[candidate].arrays[key],
                                clean_capture[candidate].arrays[key],
                                seed=effective_seed,
                                salt=(
                                    f"{pair_id}:corrupt:{key}:"
                                    f"norm_random_delta:{candidate}"
                                ),
                            )
                            for candidate in candidates
                        },
                    )
                )
                rows.extend(
                    [
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="clean",
                            donor_variant="random",
                            treatment="norm_random_delta",
                            site=site,
                            scores=clean_random_delta,
                            boundary_step=boundary_step,
                            token_position=clean_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "manipulation": clean_random_manipulation,
                            },
                        ),
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="corrupt",
                            donor_variant="random",
                            treatment="norm_random_delta",
                            site=site,
                            scores=corrupt_random_delta,
                            boundary_step=boundary_step,
                            token_position=corrupt_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "manipulation": corrupt_random_manipulation,
                            },
                        ),
                    ]
                )

            if (
                clean_pre_capture is not None
                and corrupt_pre_capture is not None
                and clean_pre_token is not None
                and corrupt_pre_token is not None
            ):
                pre_clean_to_corrupt, pre_clean_to_corrupt_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        corrupt_render.prompt,
                        candidates,
                        site=site,
                        token_position=corrupt_pre_token,
                        replacements_by_candidate=_candidate_site_replacements(
                            clean_pre_capture, candidates, key
                        ),
                    )
                )
                pre_corrupt_to_clean, pre_corrupt_to_clean_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        clean_render.prompt,
                        candidates,
                        site=site,
                        token_position=clean_pre_token,
                        replacements_by_candidate=_candidate_site_replacements(
                            corrupt_pre_capture, candidates, key
                        ),
                    )
                )
                rows.extend(
                    [
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="corrupt",
                            donor_variant="clean",
                            treatment="pre_error_clean_to_corrupt",
                            site=site,
                            scores=pre_clean_to_corrupt,
                            boundary_step=boundary_step,
                            token_position=corrupt_pre_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "manipulation": pre_clean_to_corrupt_manipulation,
                            },
                        ),
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="clean",
                            donor_variant="corrupt",
                            treatment="pre_error_corrupt_to_clean",
                            site=site,
                            scores=pre_corrupt_to_clean,
                            boundary_step=boundary_step,
                            token_position=clean_pre_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "manipulation": pre_corrupt_to_clean_manipulation,
                            },
                        ),
                    ]
                )

            if (
                control_clean_capture is not None
                and control_corrupt_capture is not None
                and control_pair_id is not None
            ):
                random_clean_to_corrupt, random_clean_to_corrupt_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        corrupt_render.prompt,
                        candidates,
                        site=site,
                        token_position=corrupt_token,
                        replacements_by_candidate=_candidate_site_replacements(
                            control_clean_capture,
                            candidates,
                            key,
                        ),
                    )
                )
                random_corrupt_to_clean, random_corrupt_to_clean_manipulation = (
                    _score_candidate_single_site_patches(
                        adapter,
                        clean_render.prompt,
                        candidates,
                        site=site,
                        token_position=clean_token,
                        replacements_by_candidate=_candidate_site_replacements(
                            control_corrupt_capture,
                            candidates,
                            key,
                        ),
                    )
                )
                rows.extend(
                    [
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="corrupt",
                            donor_variant="matched_clean",
                            treatment="random_matched_clean_to_corrupt",
                            site=site,
                            scores=random_clean_to_corrupt,
                            boundary_step=boundary_step,
                            token_position=corrupt_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "control_pair_id": control_pair_id,
                                "manipulation": random_clean_to_corrupt_manipulation,
                            },
                        ),
                        _score_row(
                            pair_id=pair_id,
                            receiver_variant="clean",
                            donor_variant="matched_corrupt",
                            treatment="random_matched_corrupt_to_clean",
                            site=site,
                            scores=random_corrupt_to_clean,
                            boundary_step=boundary_step,
                            token_position=clean_token,
                            metadata={
                                "patch_mode": "single_site_sweep",
                                "control_pair_id": control_pair_id,
                                "manipulation": random_corrupt_to_clean_manipulation,
                            },
                        ),
                    ]
                )

        for row in rows[pair_row_start:]:
            receiver = str(row["receiver_variant"])
            row["input_provenance"] = (
                clean_prompt.provenance if receiver == "clean" else corrupt_prompt.provenance
            )
            if data_config.dataset == "processbench_transfer":
                _annotate_processbench_transfer_row(row)
        _append_working_jsonl(working_rows, rows[pair_row_start:])

    atomic_write_jsonl(output_dir / "predictions.jsonl", rows)
    primary_rows = [
        row
        for row in rows
        if row.get("metadata", {}).get("control") != "forced_reroute_smoke_only"
    ]
    if data_config.dataset == "processbench_transfer":
        metrics = summarize_processbench_transfer_rows(
            primary_rows,
            seed=effective_seed,
            identity_tolerance=experiment_config.identity_tolerance,
        )
    else:
        metrics = summarize_patch_rows(primary_rows, seed=effective_seed)
    metrics["excluded_smoke_control_rows"] = len(rows) - len(primary_rows)
    atomic_write_json(
        output_dir / "data_manifest.json",
        {
            "pair_ids": [pair.clean.pair_id or pair.clean.sample_id for pair in pairs],
            "group_ids": [pair.clean.group_id for pair in pairs],
            "num_pairs": len(pairs),
            "selection_manifest_file": "selection_manifest.json",
            "selection_manifest_hash": prmbench_result.selection_manifest["manifest_hash"],
            "dataset_source": prmbench_result.source_provenance,
            "pair_audit": prmbench_result.pair_audit,
            "protocol": protocol,
            "frozen_split": frozen_split_payload,
            "split_partition": experiment_config.split_partition,
            "validation_accessed": _data_validation_accessed(
                data_config,
                split_partition=experiment_config.split_partition,
            ),
            "prompt_inputs": pair_prompt_inputs,
            "prompt_inputs_hash": stable_hash(pair_prompt_inputs, length=64),
            "code": environment["git"],
            "command": command,
            "resolved_config": resolved_config,
            "sites": [f"layer_{site.layer_index}.{site.kind}" for site in sites],
            "controls": (
                list(experiment_config.controls)
                if data_config.dataset == "processbench_transfer"
                or (
                    experiment_config.patch_mode == "single_site_sweep"
                    and experiment_config.single_site_controls_explicit
                )
                else [
                    "unpatched_rerun",
                    "self_patch",
                    "norm_random_delta",
                    "pre_error_patch_when_available",
                    "random_matched_donor_when_available",
                    "alpha_zero_identity_for_simultaneous_bundle",
                ]
                + (
                    ["forced_reroute_when_dose_response_does_not_change_topk"]
                    if experiment_config.forced_reroute_control
                    else []
                )
            ),
            "router_weight_export": router_weight_manifest,
        },
    )
    atomic_write_json(output_dir / "metrics.json", metrics)
    console.print_json(data=metrics)
    console.print(f"Wrote patch predictions to {output_dir / 'predictions.jsonl'}")


@app.command("qwen-smoke-check")
def qwen_smoke_check(
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Artifact root containing qwen3_moe outputs."),
    ] = Path("artifacts"),
    run: Annotated[
        Path,
        typer.Option("--run", help="Qwen3 smoke extraction run directory."),
    ] = Path("artifacts/qwen3_diagnostic_pilot"),
    min_examples: Annotated[
        int,
        typer.Option("--min-examples", min=1, help="Minimum smoke example count."),
    ] = 8,
    max_examples: Annotated[
        int,
        typer.Option(
            "--max-examples",
            min=0,
            help="Maximum smoke example count; set 0 to disable.",
        ),
    ] = 16,
    require_activation_files: Annotated[
        bool,
        typer.Option(
            "--require-activation-files",
            help="Also require referenced .safetensors files to be copied back.",
        ),
    ] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Review RunPod Qwen3 Gate 1 smoke artifacts without loading model weights."""

    payload = qwen_smoke_review(
        artifact_root=artifact_root,
        run_dir=run,
        min_examples=min_examples,
        max_examples=None if max_examples == 0 else max_examples,
        require_activation_files=require_activation_files,
    )
    console.print_json(data=payload)
    if output is not None and not dry_run:
        atomic_write_json(output, payload)
        console.print(f"Wrote Qwen3 smoke review to {output}")
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("second-batch-check")
def second_batch_check(
    patch_run: Annotated[Path, typer.Option("--patch-run", exists=True, file_okay=False)],
    behavior_run: Annotated[Path, typer.Option("--behavior-run", exists=True, file_okay=False)],
    model_profile: Annotated[str, typer.Option("--model-profile")],
    gate_config: Annotated[
        Path | None, typer.Option("--gate-config", exists=True, readable=True)
    ] = None,
    recapture_run: Annotated[
        Path | None, typer.Option("--recapture-run", file_okay=False)
    ] = None,
    run_role: Annotated[
        str,
        typer.Option("--run-role", help="Artifact contract: smoke or frozen_discovery."),
    ] = "smoke",
    experiment_config: Annotated[
        Path | None,
        typer.Option("--experiment-config", exists=True, readable=True),
    ] = None,
    expected_pairs: Annotated[int, typer.Option("--expected-pairs", min=1)] = 8,
    expected_behavior_pairs: Annotated[
        int | None, typer.Option("--expected-behavior-pairs", min=1)
    ] = None,
    expected_hooks: Annotated[int, typer.Option("--expected-hooks", min=1)] = 48,
    tolerance: Annotated[float, typer.Option("--tolerance", min=0.0)] = 1e-5,
    min_behavior_auroc: Annotated[
        float, typer.Option("--min-behavior-auroc", min=0.0, max=1.0)
    ] = 0.55,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Hard-gate second-batch smoke/discovery artifacts without model weights."""

    if model_profile not in {"olmoe", "qwen"}:
        raise typer.BadParameter("model-profile must be olmoe or qwen")
    if run_role not in {"smoke", "frozen_discovery"}:
        raise typer.BadParameter("run-role must be smoke or frozen_discovery")
    if run_role == "frozen_discovery" and model_profile != "olmoe":
        raise typer.BadParameter(
            "frozen_discovery contract is currently registered only for OLMoE"
        )
    if run_role == "frozen_discovery" and experiment_config is None:
        raise typer.BadParameter(
            "--experiment-config is required for frozen_discovery"
        )
    gate: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    if gate_config is not None:
        gate = load_yaml(gate_config)
        configured_hooks = (
            gate.get("profiles", {})
            .get(model_profile, {})
            .get("expected_router_hooks")
        )
        if configured_hooks is None or int(configured_hooks) != expected_hooks:
            raise typer.BadParameter(
                f"expected-hooks {expected_hooks} does not match gate config "
                f"value {configured_hooks}"
            )
        tolerance = float(
            gate.get("identity", {}).get(
                "candidate_score_max_abs_delta", tolerance
            )
        )
        min_behavior_auroc = float(
            gate.get("behavior", {}).get("minimum", min_behavior_auroc)
        )
    if experiment_config is not None and model_profile == "olmoe":
        if gate is None or gate_config is None:
            raise typer.BadParameter(
                "--gate-config is required when checking an OLMoE experiment contract"
            )
        experiment_payload = load_yaml(experiment_config)
        if experiment_payload.get("split_partition") != "discovery":
            raise typer.BadParameter(
                "second-batch OLMoE checks require the frozen discovery partition"
            )
        split_path = Path(str(experiment_payload.get("split_manifest", "")))
        if not split_path.is_file():
            raise typer.BadParameter(
                f"experiment split manifest is unavailable: {split_path}"
            )
        split_payload = cast(
            dict[str, Any],
            json.loads(split_path.read_text(encoding="utf-8")),
        )
        try:
            contract = build_olmoe_review_contract(
                gate_config=gate,
                experiment_config=experiment_payload,
                split_manifest=split_payload,
                run_role=run_role,
                gate_config_sha256=_sha256_file(gate_config),
                experiment_config_sha256=_sha256_file(experiment_config),
                experiment_config_path=str(experiment_config),
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if expected_pairs != contract["expected_pairs"]:
            raise typer.BadParameter(
                f"expected-pairs {expected_pairs} does not match {run_role} "
                f"contract {contract['expected_pairs']}"
            )
        if expected_behavior_pairs is None:
            expected_behavior_pairs = int(contract["expected_behavior_pairs"])
        elif expected_behavior_pairs != contract["expected_behavior_pairs"]:
            raise typer.BadParameter(
                f"expected-behavior-pairs {expected_behavior_pairs} does not match "
                f"{run_role} contract {contract['expected_behavior_pairs']}"
            )
    if run_role == "frozen_discovery":
        assert contract is not None
        payload = second_batch_frozen_discovery_review(
            patch_run=patch_run,
            behavior_run=behavior_run,
            contract=contract,
            tolerance=tolerance,
            min_behavior_auroc=min_behavior_auroc,
        )
    else:
        payload = second_batch_smoke_review(
            patch_run=patch_run,
            behavior_run=behavior_run,
            model_profile=model_profile,
            expected_pairs=expected_pairs,
            expected_hooks=expected_hooks,
            recapture_run=recapture_run,
            tolerance=tolerance,
            min_behavior_auroc=min_behavior_auroc,
            expected_behavior_pairs=expected_behavior_pairs,
            contract=contract,
        )
        payload["run_role"] = run_role
        payload["expected_behavior_pairs"] = expected_behavior_pairs
        if contract is not None:
            payload["contract"] = {
                key: value
                for key, value in contract.items()
                if not key.startswith("_")
            }
    payload["gate_config"] = (
        None
        if gate_config is None
        else {"path": str(gate_config), "sha256": _sha256_file(gate_config)}
    )
    console.print_json(data=payload)
    if output is not None and not dry_run:
        atomic_write_json(output, payload)
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("common-core-smoke-check")
def common_core_smoke_check(
    profile_config: Annotated[
        Path,
        typer.Option("--profile-config", exists=True, readable=True),
    ],
    model_profile: Annotated[str, typer.Option("--model-profile")],
    runtime_report: Annotated[
        Path,
        typer.Option("--runtime-report", exists=True, readable=True),
    ],
    behavior_run: Annotated[
        Path,
        typer.Option("--behavior-run", exists=True, file_okay=False),
    ],
    recapture_run: Annotated[
        Path,
        typer.Option("--recapture-run", exists=True, file_okay=False),
    ],
    router_run: Annotated[
        Path,
        typer.Option("--router-run", exists=True, file_okay=False),
    ],
    endpoint_run: Annotated[
        Path,
        typer.Option("--endpoint-run", exists=True, file_okay=False),
    ],
    prerequisite_smoke_review: Annotated[
        Path | None,
        typer.Option(
            "--prerequisite-smoke-review",
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Hard-gate a model-4/5 common-core smoke without loading model weights."""

    contract = build_common_core_smoke_contract(
        gate_config=load_yaml(profile_config),
        model_profile=model_profile,
        prerequisite_smoke_review=prerequisite_smoke_review,
    )
    payload = common_core_smoke_review(
        contract=contract,
        runtime_report=runtime_report,
        behavior_run=behavior_run,
        recapture_run=recapture_run,
        router_run=router_run,
        endpoint_run=endpoint_run,
    )
    console.print_json(data=payload)
    if output is not None and not dry_run:
        atomic_write_json(output, payload)
        console.print(f"Wrote common-core {contract['run_role']} review to {output}")
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("olmoe-discovery-prerequisite-check")
def olmoe_discovery_prerequisite_check(
    smoke_review: Annotated[
        Path,
        typer.Option("--smoke-review", exists=True, readable=True, dir_okay=False),
    ],
    next_run_id: Annotated[str, typer.Option("--next-run-id")],
    next_run_root: Annotated[Path, typer.Option("--next-run-root")],
    next_package: Annotated[Path, typer.Option("--next-package")],
    gate_config: Annotated[
        Path,
        typer.Option("--gate-config", exists=True, readable=True, dir_okay=False),
    ] = Path("configs/analysis/second_batch_gpu_gate.yaml"),
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Require an intact same-Pod OLMoE smoke PASS before discovery."""

    state = git_state()
    current_environment = {
        key: os.environ.get(key)
        for key in (
            "RUNPOD_POD_ID",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
        )
    }
    payload = olmoe_discovery_prerequisite_review(
        smoke_review_path=smoke_review,
        gate_config_path=gate_config,
        next_run_id=next_run_id,
        next_run_root=next_run_root,
        next_package_path=next_package,
        current_commit=(
            None if state.get("commit") is None else str(state["commit"])
        ),
        current_dirty=cast(bool | None, state.get("dirty")),
        current_environment=current_environment,
    )
    console.print_json(data=payload)
    if output is not None and not dry_run:
        atomic_write_json(output, payload)
        console.print(f"Wrote OLMoE discovery prerequisite review to {output}")
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("qwen-gate2-check")
def qwen_gate2_check(
    run_root: Annotated[
        Path,
        typer.Option("--run-root", exists=True, file_okay=False, help="Gate 2 run root."),
    ],
    expected_examples: Annotated[
        int,
        typer.Option("--expected-examples", min=1, help="Expected bounded example count."),
    ] = 16,
    expected_seed: Annotated[int, typer.Option("--expected-seed", help="Expected seed.")] = 0,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Review bounded Qwen3 Gate 2 repeat artifacts without loading model weights."""

    payload = qwen_gate2_review(
        run_root=run_root,
        expected_examples=expected_examples,
        expected_seed=expected_seed,
    )
    console.print_json(data=payload)
    if output is not None and not dry_run:
        atomic_write_json(output, payload)
        console.print(f"Wrote Qwen3 Gate 2 review to {output}")
    if not payload["passed"]:
        raise typer.Exit(code=1)


@app.command("report")
def report(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False)],
    limit: LimitOpt = None,
    seed: SeedOpt = 0,
    dry_run: DryRunOpt = False,
) -> None:
    """Produce a minimal machine-readable report summary."""

    snapshot = environment_snapshot(cwd=Path.cwd(), seed=seed)
    payload: dict[str, Any] = {
        "run": str(run),
        "limit": limit,
        "dry_run": dry_run,
        "environment": snapshot,
    }
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = run / "predictions.jsonl"
    if predictions.exists():
        rows = _read_jsonl(predictions, limit=limit)
        payload["num_prediction_rows"] = len(rows)
        if rows and {"treatment", "receiver_variant", "margin_incorrect_minus_correct"}.issubset(
            rows[0]
        ):
            patch_metrics = summarize_patch_rows(rows, seed=seed)
            payload["patch_metrics"] = patch_metrics
    analysis_payload: dict[str, Any] = {}
    for name in ("probe_metrics", "routing_metrics", "router_visibility_metrics"):
        analysis_path = run / "analysis" / f"{name}.json"
        if analysis_path.exists():
            analysis_payload[name] = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis_payload:
        payload["analysis"] = analysis_payload
    plot_dir = run / "report" / "plots"
    if plot_dir.exists():
        payload["plots"] = {path.stem: str(path) for path in sorted(plot_dir.glob("*.svg"))}
    console.print_json(data=payload)
    if not dry_run:
        output = run / "report" / "summary.json"
        atomic_write_json(output, payload)
        _atomic_write_text(run / "report" / "summary.md", summary_markdown(payload))
        if "patch_metrics" in payload:
            atomic_write_json(run / "report" / "patch_metrics.json", payload["patch_metrics"])
            _atomic_write_text(
                run / "report" / "patch_effects.md",
                patch_effect_markdown(payload["patch_metrics"]),
            )
        analysis = payload.get("analysis", {})
        if isinstance(analysis, dict):
            if "probe_metrics" in analysis:
                _atomic_write_text(
                    run / "report" / "probe_metrics.md",
                    probe_metrics_markdown(analysis["probe_metrics"]),
                )
            if "routing_metrics" in analysis:
                _atomic_write_text(
                    run / "report" / "routing_metrics.md",
                    routing_metrics_markdown(analysis["routing_metrics"]),
                )
            if "router_visibility_metrics" in analysis:
                _atomic_write_text(
                    run / "report" / "router_visibility_metrics.md",
                    router_visibility_markdown(analysis["router_visibility_metrics"]),
                )
        console.print(f"Wrote {output}")


def main() -> None:
    app()
