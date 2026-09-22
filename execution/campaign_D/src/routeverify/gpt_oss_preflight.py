"""Runtime-only GPT-OSS inspect and hook-fidelity stage.

This module is intentionally independent of the OLMoE residual runners.  It loads
one GPT-OSS checkpoint instance, inspects the post-quantization module layout, then
verifies capture identity, self-patch identity, and a forced router dispatch change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from routeverify.config import (
    environment_snapshot,
    load_data_config,
    load_model_config,
)
from routeverify.data.prompts import STEP_BINARY_CANDIDATES
from routeverify.data.schema import stable_hash
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchSpec,
    SiteKind,
    site_key,
)
from routeverify.instrumentation.storage import atomic_write_json
from routeverify.models.gpt_oss import GptOssAdapter
from routeverify.models.registry import create_adapter


def _max_abs_tensor_delta(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(
        left.detach().float().cpu().numpy() if hasattr(left, "detach") else left,
        dtype=np.float32,
    )
    right_array = np.asarray(
        right.detach().float().cpu().numpy() if hasattr(right, "detach") else right,
        dtype=np.float32,
    )
    if left_array.shape != right_array.shape:
        raise ValueError(f"tensor shapes differ: {left_array.shape} vs {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0


def _capture_sites(
    adapter: GptOssAdapter, sentinel_layers: tuple[int, ...]
) -> tuple[LayerSite, ...]:
    layer_count = len(adapter._layers())
    if not sentinel_layers:
        raise ValueError("sentinel_layers must be non-empty")
    if any(index < 0 or index >= layer_count for index in sentinel_layers):
        raise ValueError(
            f"sentinel layer outside runtime range 0..{layer_count - 1}: {sentinel_layers}"
        )
    requested: dict[str, LayerSite] = {}
    for layer_index in range(layer_count):
        site = LayerSite(
            layer_index,
            "router_logits",
            adapter._module_name(layer_index, "router_logits"),
        )
        requested[site_key(site)] = site
    for layer_index in sentinel_layers:
        for kind in (
            "pre_moe_residual",
            "router_input",
            "router_topk_ids",
            "router_topk_weights",
            "moe_output",
            "post_moe_residual",
        ):
            site = LayerSite(layer_index, kind, adapter._module_name(layer_index, kind))
            requested[site_key(site)] = site
    return tuple(requested[key] for key in sorted(requested))


def run_runtime_hook_check(
    *,
    model_path: Path,
    data_path: Path,
    split_manifest_path: Path,
    split_partition: str,
    pair_limit: int,
    seed: int,
    sentinel_layers: tuple[int, ...],
    output_dir: Path,
    capture_tolerance: float,
    identity_tolerance: float,
) -> dict[str, Any]:
    """Execute the single-instance runtime inspect and hook-fidelity stage."""

    if split_partition != "discovery":
        raise ValueError("GPT-OSS preflight is discovery-only; validation is sealed")
    if pair_limit != 8:
        raise ValueError("GPT-OSS preflight contract requires exactly eight discovery pairs")

    import torch

    # Imported lazily to keep the independent runtime entry point free of CLI cycles.
    from routeverify.cli import (
        _apply_split_first_selection,
        _examples_and_provenance_from_prmbench_result,
        _load_prmbench_selection_from_hf,
        _render_and_boundary_token,
    )

    model_config = load_model_config(model_path).model_copy(update={"load_weights": True})
    data_config = load_data_config(data_path)
    if model_config.alias != "gpt_oss_20b":
        raise ValueError("runtime hook check requires alias=gpt_oss_20b")
    if data_config.dataset != "prmbench":
        raise ValueError("runtime hook check requires PRMBench")

    selection = _load_prmbench_selection_from_hf(data_config, limit=None, seed=seed)
    selection, _partition_by_group, frozen_split = _apply_split_first_selection(
        selection,
        split_manifest_path=split_manifest_path,
        split_partition=split_partition,
        limit=pair_limit,
        seed=seed,
    )
    loaded = _examples_and_provenance_from_prmbench_result(
        selection,
        data_config=data_config,
        limit=pair_limit,
        seed=seed,
    )
    pair = next((item for item in selection.pairs if item.corrupt.first_error is not None), None)
    if pair is None or pair.corrupt.first_error is None:
        raise RuntimeError("the eight-pair discovery slice has no corrupt boundary")

    adapter_any = create_adapter(model_config.alias)
    if not isinstance(adapter_any, GptOssAdapter):
        raise TypeError("registry did not return GptOssAdapter")
    adapter = adapter_any
    adapter.load(model_config)
    runtime_inspect = adapter.runtime_inspect()

    boundary_step = pair.corrupt.first_error
    prepared = _render_and_boundary_token(
        adapter,
        pair.corrupt,
        boundary_step=boundary_step,
        prompt_mode="step_binary",
        input_format="chat",
    )
    candidate = STEP_BINARY_CANDIDATES[0]
    inputs, _target_start, _target_end, _target_ids = adapter._encode_prompt_candidate(
        prepared.raw_render.prompt,
        candidate,
    )
    original_linear = torch.nn.functional.linear
    with torch.inference_mode():
        baseline_outputs = adapter._require_model()(
            **inputs,
            use_cache=False,
            output_router_logits=False,
        )
    baseline_logits = baseline_outputs.logits.detach().cpu()

    sites = _capture_sites(adapter, sentinel_layers)
    capture, captured_logits = adapter._capture_inputs(
        inputs,
        CaptureSpec(
            sites=sites,
            token_positions=(prepared.boundary_token,),
            dtype="float32",
        ),
    )
    capture_delta = _max_abs_tensor_delta(baseline_logits, captured_logits)
    linear_restored_after_capture = torch.nn.functional.linear is original_linear

    identity_kinds: tuple[SiteKind, ...] = (
        "post_moe_residual",
        "moe_output",
        "router_logits",
    )
    identity_sites = tuple(
        LayerSite(
            sentinel_layers[len(sentinel_layers) // 2],
            kind,
            adapter._module_name(sentinel_layers[len(sentinel_layers) // 2], kind),
        )
        for kind in identity_kinds
    )
    identity_results: dict[str, Any] = {}
    for site in identity_sites:
        key = site_key(site)
        patch = PatchSpec(
            site=site,
            token_position=prepared.boundary_token,
            replacement=capture.arrays[key],
        )
        patched_outputs, diagnostics = adapter._run_with_patch_inputs_diagnostics(inputs, patch)
        identity_results[key] = {
            "max_abs_logit_delta": _max_abs_tensor_delta(
                baseline_logits, patched_outputs.logits.detach().cpu()
            ),
            "diagnostics": diagnostics,
        }

    identity_hooks_ok = all(
        result["diagnostics"].get("all_registered_hooks_fired") is True
        for result in identity_results.values()
    )
    identity_interceptors_restored = all(
        result["diagnostics"].get("mxfp4_routing_interceptor", {}).get("restored") is True
        for result in identity_results.values()
    )

    router_site = next(site for site in identity_sites if site.kind == "router_logits")
    forced_patch = PatchSpec(
        site=router_site,
        token_position=prepared.boundary_token,
        replacement=capture.arrays[site_key(router_site)],
        metadata={"forced_reroute": True, "control": "hook_fidelity"},
    )
    forced_outputs, forced_diagnostics = adapter._run_with_patch_inputs_diagnostics(
        inputs, forced_patch
    )
    forced_key = site_key(router_site)
    identity_router_dispatch_verified = (
        identity_results[forced_key]["diagnostics"].get("all_router_dispatch_verified") is True
    )
    forced_manipulation = forced_diagnostics["manipulation_by_site"][forced_key]
    forced_dispatch = forced_diagnostics["dispatch_by_site"][forced_key]
    forced_output_delta = _max_abs_tensor_delta(
        baseline_logits, forced_outputs.logits.detach().cpu()
    )

    with torch.inference_mode():
        rerun_outputs = adapter._require_model()(
            **inputs,
            use_cache=False,
            output_router_logits=False,
        )
    rerun_delta = _max_abs_tensor_delta(baseline_logits, rerun_outputs.logits.detach().cpu())
    linear_restored_final = torch.nn.functional.linear is original_linear
    identity_max = max(float(result["max_abs_logit_delta"]) for result in identity_results.values())
    passed = bool(
        runtime_inspect["contract_matches"]
        and runtime_inspect["native_mxfp4"]
        and capture.metadata["all_registered_hooks_fired"]
        and capture_delta <= capture_tolerance
        and identity_max <= identity_tolerance
        and identity_hooks_ok
        and identity_interceptors_restored
        and identity_router_dispatch_verified
        and float(forced_manipulation["topk_changed_rate"]) > 0.0
        and forced_dispatch["downstream_dispatch_verified"]
        and forced_diagnostics["all_router_dispatch_verified"]
        and forced_diagnostics["mxfp4_routing_interceptor"].get("restored") is True
        and forced_output_delta > 0.0
        and rerun_delta <= capture_tolerance
        and linear_restored_after_capture
        and linear_restored_final
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "runtime_inspect.json", runtime_inspect)
    hook_payload = {
        "schema_version": 1,
        "passed": passed,
        "sample": {
            "pair_id": pair.clean.pair_id or pair.clean.sample_id,
            "sample_id": pair.corrupt.sample_id,
            "variant": "corrupt",
            "candidate": candidate,
            "boundary_step": boundary_step,
            "boundary_token": prepared.boundary_token,
        },
        "render": adapter.harmony_render_provenance(
            prepared.raw_render.prompt,
            include_text=True,
        ),
        "capture": {
            "site_count": len(sites),
            "sites": [site_key(site) for site in sites],
            "array_shapes": {
                key: [int(value) for value in tensor.shape]
                for key, tensor in capture.arrays.items()
            },
            "metadata": capture.metadata,
            "max_abs_logit_delta": capture_delta,
            "tolerance": capture_tolerance,
        },
        "self_patch_identity": {
            "sites": identity_results,
            "max_abs_logit_delta": identity_max,
            "tolerance": identity_tolerance,
            "all_registered_hooks_fired": identity_hooks_ok,
            "all_interceptors_restored": identity_interceptors_restored,
            "router_dispatch_verified": identity_router_dispatch_verified,
        },
        "forced_router": {
            "site": forced_key,
            "manipulation": forced_manipulation,
            "dispatch": forced_dispatch,
            "max_abs_final_logit_delta": forced_output_delta,
            "diagnostics": forced_diagnostics,
        },
        "restoration": {
            "functional_linear_restored_after_capture": linear_restored_after_capture,
            "functional_linear_restored_final": linear_restored_final,
            "post_patch_rerun_max_abs_logit_delta": rerun_delta,
        },
        "validation_accessed": False,
    }
    atomic_write_json(output_dir / "hook_fidelity.json", hook_payload)
    atomic_write_json(output_dir / "selection_manifest.json", selection.selection_manifest)
    prompt_record = dict(prepared.provenance)
    prompt_record["rendered_prompt_text"] = prepared.rendered_input.prompt
    atomic_write_json(
        output_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "pair_ids": [item.clean.pair_id or item.clean.sample_id for item in selection.pairs],
            "num_pairs": len(selection.pairs),
            "runtime_pair_id": pair.clean.pair_id or pair.clean.sample_id,
            "selection_manifest_hash": selection.selection_manifest["manifest_hash"],
            "frozen_split": frozen_split,
            "prompt": prompt_record,
            "prompt_hash": stable_hash(prompt_record, length=64),
            "validation_accessed": False,
        },
    )
    if adapter.discovery_report is None:
        raise RuntimeError("loaded GPT-OSS adapter did not produce a discovery report")
    atomic_write_json(output_dir / "model_discovery.json", adapter.discovery_report.to_json())
    environment = environment_snapshot(
        model=model_config,
        data=data_config,
        data_provenance=loaded.provenance,
        seed=seed,
    )
    environment["validation_accessed"] = False
    environment["frozen_split"] = frozen_split
    atomic_write_json(output_dir / "environment.json", environment)
    (output_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "model": model_config.model_dump(mode="json"),
                "data": data_config.model_dump(mode="json"),
                "split_manifest": str(split_manifest_path),
                "split_partition": split_partition,
                "pair_limit": pair_limit,
                "seed": seed,
                "sentinel_layers": list(sentinel_layers),
                "capture_tolerance": capture_tolerance,
                "identity_tolerance": identity_tolerance,
                "validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return hook_payload


def _parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sentinel layers must be comma-separated integers"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("sentinel layers must be non-empty")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runtime = subparsers.add_parser("runtime-check")
    runtime.add_argument("--model", type=Path, required=True)
    runtime.add_argument("--data", type=Path, required=True)
    runtime.add_argument("--split-manifest", type=Path, required=True)
    runtime.add_argument("--split-partition", default="discovery")
    runtime.add_argument("--pair-limit", type=int, default=8)
    runtime.add_argument("--seed", type=int, default=20260809)
    runtime.add_argument("--sentinel-layers", type=_parse_layers, default=(0, 12, 23))
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--capture-tolerance", type=float, default=1e-5)
    runtime.add_argument("--identity-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.command == "runtime-check":
        payload = run_runtime_hook_check(
            model_path=args.model,
            data_path=args.data,
            split_manifest_path=args.split_manifest,
            split_partition=args.split_partition,
            pair_limit=args.pair_limit,
            seed=args.seed,
            sentinel_layers=args.sentinel_layers,
            output_dir=args.output_dir,
            capture_tolerance=args.capture_tolerance,
            identity_tolerance=args.identity_tolerance,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not payload["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
