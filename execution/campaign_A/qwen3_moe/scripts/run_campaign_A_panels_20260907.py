#!/usr/bin/env python3
"""Run one A model's D/bundle and window panels in a fresh namespace.

This is an offline, single-model launcher. It neither creates a Pod nor installs
dependencies/downloads weights. GLM original common-core has a separate runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ("qwen36_moe", "qwen3_moe", "glm4_moe_lite")
INPUTS = Path("data/campaign_20260907_A")


def verify_inputs() -> int:
    manifest = json.loads(
        (ROOT / "docs/results/campaign_20260907_A/input_asset_manifest.json").read_text()
    )
    for name, expected in manifest["files"].items():
        payload = (ROOT / name).read_bytes()
        if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    return len(manifest["files"])


def commands(alias: str, output: Path, n_resamples: int) -> list[tuple[str, list[str]]]:
    pairs = str(INPUTS / "discovery/window_discovery_pairs.jsonl")
    donor = [sys.executable, "-m", "routeverify.donor_contrast", "--config",
             f"configs/experiments/donor_bundle_20260907/{alias}.yaml",
             "--pairs-jsonl", pairs, "--mode", "both"]
    window = [sys.executable, "-m", "routeverify.window_interventions", "--config",
              f"configs/windows/{alias}_core_20260907.yaml", "--pairs", pairs,
              "--tokenizer-path", str(INPUTS / "tokenizers" / alias)]
    return [
        ("donor_smoke", [*donor, "--limit", "2", "--output-dir", str(output / "donor_smoke")]),
        ("donor_full", [*donor, "--output-dir", str(output / "donor_full")]),
        ("window_smoke", [*window, "--mode", "smoke", "--output", str(output / "window_smoke")]),
        ("window_full", [*window, "--mode", "full", "--smoke-report",
                                 str(output / "window_smoke/execution_summary.json"),
                                 "--output", str(output / "window_full")]),
        ("donor_analysis", [sys.executable, "-m", "routeverify.donor_contrast", "--analyze",
                            str(output / "donor_full"), "--n-resamples", str(n_resamples)]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALIASES)
    parser.add_argument("--output", required=True, type=Path, help="new per-model attempt root")
    parser.add_argument("--dry-run", action="store_true", help="check local inputs and print commands only")
    parser.add_argument("--n-resamples", type=int, default=50000)
    args = parser.parse_args()
    if args.n_resamples < 2:
        parser.error("n-resamples must be at least two")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("refusing nonempty attempt root; preserve it and restart only the failed unit")
    checked = verify_inputs()
    steps = commands(args.model, output, args.n_resamples)
    print(json.dumps({"model": args.model, "verified_input_files": checked,
                      "dry_run": args.dry_run, "commands": [
                          {"stage": stage, "argv": argv} for stage, argv in steps]}, indent=2), flush=True)
    if args.dry_run:
        return
    if sys.platform != "linux" or not os.environ.get("HF_HUB_CACHE"):
        parser.error("execution needs the supplied Linux Pod and explicit HF_HUB_CACHE")
    import torch

    if not torch.cuda.is_available():
        parser.error("checkpoint execution requires CUDA")
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment["PYTHONPATH"] = str(ROOT / "src")
    output.mkdir(parents=True, exist_ok=True)
    record = {"model": args.model, "status": "running", "completed_stages": [], "active_stage": None}
    record_path = output / "launcher_state.json"
    try:
        for stage, argv in steps:
            record["active_stage"] = stage
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"[{stage}] {shlex.join(argv)}", flush=True)
            subprocess.run(argv, cwd=ROOT, env=environment, check=True)
            record["completed_stages"].append(stage)
        record.update(status="complete", active_stage=None)
    except BaseException as exc:
        record.update(status="failed", failure_type=type(exc).__name__)
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
