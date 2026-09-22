#!/usr/bin/env python3
"""GLM common core: pinned local data, one checkpoint load, existing CLI and reviewer."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from routeverify import cli
from routeverify.config import load_data_config
from routeverify.data.schema import stable_hash
from routeverify.instrumentation.storage import atomic_write_json
from routeverify.reporting.common_core_smoke import (
    build_common_core_smoke_contract,
    common_core_smoke_review,
    resolve_common_core_stage_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = Path("configs/experiments/campaign_20260907_A")


def source_identity(root: Path = ROOT) -> dict[str, Any]:
    paths = [root / "pyproject.toml", root / "uv.lock",
             root / "scripts/glm_common_core_campaign_20260907.py",
             root / "scripts/run_interface_access_analysis.py",
             root / "configs/models/glm4_moe_lite.yaml"]
    paths.extend((root / "src").rglob("*.py"))
    paths.extend((root / CONFIGS).glob("glm_common_core*.yaml"))
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }
    return {"files": files, "sha256": stable_hash(files, length=64)}


def expected_work(pairs: int, behavior_pairs: int) -> dict[str, Any]:
    # One full candidate input per CORRECT/INCORRECT score or patch capture.
    # Recapture is the original prompt-only representation capture: 1 forward/row.
    return {
        "behavior": {"scoring_rows": 2 * behavior_pairs, "forwards": 4 * behavior_pairs},
        "recapture": {"capture_rows": 2 * pairs, "arrays_per_row": 230, "forwards": 2 * pairs},
        "router": {"scoring_rows": 14 * pairs, "forwards": 32 * pairs},
        "endpoint": {"scoring_rows": 52 * pairs, "forwards": 112 * pairs},
        "total_forwards": 146 * pairs + 4 * behavior_pairs,
        "load_count": 1,
        "runtime_preflight_forwards": 0,
    }


class MeasuredAdapterFactory:
    """Reuse one native adapter; count actual attempted/completed model forwards."""

    def __init__(self, factory: Any) -> None:
        self.factory = factory
        self.adapter: Any = None
        self.loaded_config: Any = None
        self.stage = "runtime"
        self.phase = "runtime"
        self.measurements: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def record(self) -> dict[str, Any]:
        return self.measurements.setdefault(self.stage, {}).setdefault(
            self.phase, {"seconds": 0.0, "attempted_forwards": 0, "completed_forwards": 0}
        )

    def wrap(self, operation: Any, phase: str) -> Any:
        @functools.wraps(operation)
        def measured(*args: Any, **kwargs: Any) -> Any:
            previous = self.phase
            if previous == phase:
                return operation(*args, **kwargs)
            self.phase = phase
            started = time.monotonic()
            try:
                return operation(*args, **kwargs)
            finally:
                self.record()["seconds"] += time.monotonic() - started
                self.phase = previous
        return measured

    def __call__(self, alias: str) -> Any:
        if self.adapter is not None:
            if self.adapter.alias != alias:
                raise ValueError("GLM common-core cannot mix model aliases")
            return self.adapter
        self.adapter = self.factory(alias)
        original_load = self.adapter.load

        def load_once(config: Any) -> None:
            payload = config.model_dump(mode="json")
            if self.loaded_config is not None:
                if payload != self.loaded_config:
                    raise ValueError("model config changed between common-core stages")
                return
            self.wrap(original_load, "load")(config)
            self.loaded_config = payload
            model = self.adapter._require_model()

            def attempted(*_args: Any) -> None:
                self.record()["attempted_forwards"] += 1

            def completed(*_args: Any) -> None:
                self.record()["completed_forwards"] += 1

            self.handles.extend([
                model.register_forward_pre_hook(attempted), model.register_forward_hook(completed)
            ])

        self.adapter.load = load_once
        for name, phase in (
            ("capture", "capture"),
            ("capture_candidate_activations", "capture"),
            ("score_candidates", "score"),
            ("score_candidates_with_candidate_patches", "score"),
            ("score_candidates_with_candidate_patches_diagnostics", "score"),
            ("score_candidates_with_candidate_patch_bundles", "score"),
        ):
            if hasattr(self.adapter, name):
                setattr(self.adapter, name, self.wrap(getattr(self.adapter, name), phase))
        return self.adapter


def build_plan(mode: str, smoke_review: Path | None, *, planning_only: bool) -> dict[str, Any]:
    gate_path = CONFIGS / f"glm_common_core_{mode}.yaml"
    gate = yaml.safe_load(gate_path.read_text())
    source = source_identity()
    gate["execution_source_sha256"] = source["sha256"]
    if smoke_review is not None:
        smoke = json.loads(smoke_review.read_text())
        profile = gate["profiles"]["glm4_moe_lite"]
        profile["prerequisite_smoke_review_sha256"] = hashlib.sha256(smoke_review.read_bytes()).hexdigest()
        profile["prerequisite_smoke_commit"] = smoke.get("provenance", {}).get("git", {}).get("commit")
    contract = build_common_core_smoke_contract(
        gate_config=gate, model_profile="glm4_moe_lite", root=ROOT,
        prerequisite_smoke_review=smoke_review, planning_only=planning_only,
    )
    data = load_data_config(contract["data_config"])
    result = cli._load_prmbench_selection_from_hf(data, limit=None, seed=contract["seed"])
    selected, _, _ = cli._apply_split_first_selection(
        result, split_manifest_path=Path(contract["split_manifest"]),
        split_partition="discovery", limit=contract["expected_pairs"], seed=contract["seed"],
    )
    if [pair.clean.pair_id for pair in selected.pairs] != contract["expected_pair_ids"]:
        raise ValueError("local file selection differs from frozen common-core order")
    return {
        "schema_version": 1, "mode": mode, "planning_only": planning_only,
        "source_identity": source, "gate_config": gate, "contract": contract,
        "expected_work": expected_work(contract["expected_pairs"], contract["expected_behavior_pairs"]),
        "selection_manifest": selected.selection_manifest,
        "prerequisite_status": "bound" if smoke_review is not None else "pending_real_checkpoint_smoke",
        "validation_accessed": False,
    }


def run(plan: dict[str, Any], output: Path) -> None:
    import torch

    contract = plan["contract"]
    if os.environ.get("ALLOW_GPU_RUN") != "1" or os.environ.get(contract["authorization_env"]) != "1":
        raise ValueError(f"runtime requires ALLOW_GPU_RUN=1 and {contract['authorization_env']}=1")
    if not torch.cuda.is_available():
        raise ValueError("real checkpoint common-core requires CUDA")
    if plan["planning_only"]:
        raise ValueError("a dry-run plan cannot authorize execution")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "run_plan.json", plan)
    resolved_dir = output / "resolved_configs"
    resolved_dir.mkdir()
    stage_paths: dict[str, Path] = {}
    for stage, source in contract["stage_paths"].items():
        stage_paths[stage] = resolved_dir / f"{stage}.yaml"
        payload = resolve_common_core_stage_payload(
            source=Path(source), output_dir=output / stage, seed=contract["seed"],
            limit=contract["expected_pairs"], split_manifest=Path(contract["resolved_split_manifest"]),
            split_partition="discovery",
        )
        stage_paths[stage].write_text(yaml.safe_dump(payload, sort_keys=False))
    runtime_path = output / "runtime" / "runtime_preflight.json"
    source_sha = plan["source_identity"]["sha256"]
    original_factory = cli.create_adapter
    measured = MeasuredAdapterFactory(original_factory)
    cli.create_adapter = measured
    timings: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "schema_version": 1, "complete": False, "passed": False,
        "execution_source_sha256": source_sha, "mode": plan["mode"],
        "expected_work": plan["expected_work"], "stages": timings,
        "validation_accessed": False,
    }
    operations = {
        "runtime": lambda: cli.model_runtime_preflight(Path(contract["model_config"]), runtime_path),
        "behavior": lambda: cli.eval_behavior(
            Path(contract["model_config"]), Path(contract["data_config"]), mode="step-binary",
            limit=contract["expected_behavior_pairs"], seed=contract["seed"],
            output_dir=output / "behavior", split_manifest=Path(contract["resolved_split_manifest"]),
            split_partition="discovery",
        ),
        "recapture": lambda: cli.extract(stage_paths["recapture"]),
        "router": lambda: cli.intervene_patch(stage_paths["router"]),
        "endpoint": lambda: cli.intervene_patch(stage_paths["endpoint"]),
    }
    try:
        for stage, operation in operations.items():
            measured.stage = measured.phase = stage
            summary["active_stage"] = stage
            if source_identity()["sha256"] != source_sha:
                raise ValueError("execution sources changed after plan freeze")
            torch.cuda.init()  # Initialize allocator before peak-stat reset.
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            try:
                operation()
            finally:
                torch.cuda.synchronize()
                timing = {
                    "wall_seconds": time.monotonic() - started,
                    "operations": measured.measurements.get(stage, {}),
                    "cuda_allocator_peaks": [
                        {"device": device, "allocated_bytes": torch.cuda.max_memory_allocated(device),
                         "reserved_bytes": torch.cuda.max_memory_reserved(device)}
                        for device in range(torch.cuda.device_count())
                    ],
                }
                timings[stage] = timing
                atomic_write_json(output / "execution_summary.json", summary)
            if source_identity()["sha256"] != source_sha:
                raise ValueError("execution sources changed during stage")
            environment_path = runtime_path if stage == "runtime" else output / stage / "environment.json"
            environment = json.loads(environment_path.read_text())
            target = environment["environment"] if stage == "runtime" else environment
            target["execution_source_sha256"] = source_sha
            atomic_write_json(environment_path, environment)
            count = sum(item["completed_forwards"] for item in timing["operations"].values())
            expected = 0 if stage == "runtime" else plan["expected_work"][stage]["forwards"]
            if count != expected:
                raise ValueError(f"{stage} completed {count} forwards; expected {expected}")
        review = common_core_smoke_review(
            contract=contract, runtime_report=runtime_path, behavior_run=output / "behavior",
            recapture_run=output / "recapture", router_run=output / "router", endpoint_run=output / "endpoint",
        )
        atomic_write_json(output / f"common_core_{plan['mode']}_review.json", review)
        summary.update(technical_review_passed=review["passed"])
        if not review["passed"]:
            raise ValueError("technical artifact review failed; inspect saved checks")
        if plan["mode"] == "full":
            summary["active_stage"] = "analysis"
            native_visibility(measured.adapter, output / "recapture", seed=contract["seed"])
            cli.analyze_routing(output / "recapture", seed=contract["seed"])
            cli.analyze_probe(output / "recapture", seed=contract["seed"])
            direction = output / "recapture" / "analysis" / "router_input_direction.npz"
            try:
                cli.analyze_probe_direction(output / "recapture", direction)
            except ValueError as exc:
                # An undefined empirical direction is a scientific NA, not a
                # reason to reject correctly collected full experiment rows.
                if "paired mean direction has zero/nonfinite norm" not in str(exc):
                    raise
                atomic_write_json(output / "recapture" / "analysis" / "router_subspace.json", {
                    "status": "undefined_empirical_direction", "reason": str(exc),
                    "validation_accessed": False,
                })
            else:
                cli.analyze_router_subspace(output / "router" / "router_weights.npz", direction,
                                            output / "recapture" / "analysis" / "router_subspace.json")
            config = yaml.safe_load((CONFIGS / "glm_common_core_interface_access.yaml").read_text())
            config["canonical_source_problem_manifest"] = str(
                ROOT / config["canonical_source_problem_manifest"])
            config["canonical_grouping_provenance"] = str(ROOT / config["canonical_grouping_provenance"])
            analysis_config = resolved_dir / "interface_access.yaml"
            analysis_config.write_text(yaml.safe_dump(config, sort_keys=False))
            subprocess.run([
                sys.executable, "scripts/run_interface_access_analysis.py", "--config", str(analysis_config),
                "--archive-root", str(output), "--output", str(output / "analysis" / "interface_access"),
            ], check=True)
        summary.update(complete=True, passed=True, active_stage="completed")
    except BaseException as exc:
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        cli.create_adapter = original_factory
        for handle in measured.handles:
            handle.remove()
        atomic_write_json(output / "execution_summary.json", summary)


def native_visibility(adapter: Any, recapture: Path, *, seed: int) -> None:
    """Reuse the existing finite-difference analysis with native GLM top-k IDs."""
    import numpy as np
    import torch

    from routeverify.analysis.router_visibility import router_visibility_metrics

    features: dict[int, list[Any]] = {}
    labels: list[bool] = []
    for row in cli._read_jsonl(recapture / "data_manifest.jsonl"):
        labels.append(bool(row["is_error_boundary"]))
        arrays = cli._load_safetensors(cli._resolve_artifact_path(recapture, row["activation_file"]))
        for key, value in arrays.items():
            if key.endswith(".router_input"):
                features.setdefault(cli._layer_index_from_site_key(key), []).append(
                    cli._activation_vector(value, row))

    class NativeGate(torch.nn.Module):
        def __init__(self, gate: Any) -> None:
            super().__init__()
            self.gate = gate
            self.calls = 0

        def forward(self, hidden: Any) -> Any:
            self.calls += 1
            routed = adapter._route_from_logits(self.gate, self.gate(hidden))
            return routed.logits, routed.topk_weights, routed.topk_ids

    sites = {}
    calls = 0
    for layer, rows in sorted(features.items()):
        view = NativeGate(adapter._layer(layer).mlp.gate)
        matrix = np.stack(rows)
        labels_array = np.asarray(labels)
        delta = matrix[labels_array].mean(axis=0) - matrix[~labels_array].mean(axis=0)
        if np.linalg.norm(delta) == 0:
            sites[f"layer_{layer}.router_input"] = {"status": "zero_empirical_direction"}
        else:
            sites[f"layer_{layer}.router_input"] = router_visibility_metrics(
                view, matrix, labels_array, seed=seed)
        calls += view.calls
    atomic_write_json(recapture / "analysis" / "router_visibility_metrics.json", {
        "sites": sites, "seed": seed, "gate_only_evaluations": calls,
        "full_model_forwards": 0, "validation_accessed": False,
        "router_ids": "native_sigmoid_bias_group_topk",
        "logit_sensitivity": "raw_fp32_pre_sigmoid_pre_bias",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke-review", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = build_plan(args.mode, args.smoke_review, planning_only=args.dry_run)
    if args.dry_run:
        if args.output.exists():
            raise ValueError(f"refusing existing dry-run output: {args.output}")
        atomic_write_json(args.output, plan)
        print(json.dumps({"output": str(args.output), "expected_work": plan["expected_work"]}, indent=2))
        return
    run(plan, args.output.resolve())


if __name__ == "__main__":
    main()
