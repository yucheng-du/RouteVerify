#!/usr/bin/env python3
"""Generate or verify the frozen ProcessBench natural-transfer bundle offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from routeverify.data.processbench_transfer import (
    PROCESSBENCH_REVISION,
    build_processbench_transfer_bundle,
    load_processbench_transfer_bundle,
    write_processbench_transfer_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Local HF snapshot directory containing gsm8k.json; never fetched.",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=Path("data/processed/processbench-natural-transfer-v1.jsonl"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/processbench-natural-transfer-v1.selection.json"),
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--smoke-pairs", type=int, default=8)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify committed files and exact regeneration; do not write.",
    )
    return parser


def _default_source_root() -> Path:
    candidates = (
        Path(".cache/huggingface/hub")
        / "datasets--Qwen--ProcessBench"
        / "snapshots"
        / PROCESSBENCH_REVISION,
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--Qwen--ProcessBench"
        / "snapshots"
        / PROCESSBENCH_REVISION,
    )
    for candidate in candidates:
        if (candidate / "gsm8k.json").is_file():
            return candidate
    raise FileNotFoundError(
        "pinned ProcessBench snapshot is not present in an approved local cache; "
        "pass --source-root after placing the public data locally"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = args.source_root or _default_source_root()
    data_output = args.data_output if args.data_output.is_absolute() else repo_root / args.data_output
    manifest_output = (
        args.manifest_output
        if args.manifest_output.is_absolute()
        else repo_root / args.manifest_output
    )
    rebuilt = build_processbench_transfer_bundle(
        source_root,
        seed=args.seed,
        pairs_per_subset=args.pairs,
        smoke_pairs_per_subset=args.smoke_pairs,
    )
    if args.check:
        committed = load_processbench_transfer_bundle(
            data_path=data_output,
            manifest_path=manifest_output,
        )
        if committed.processed_jsonl_bytes != rebuilt.processed_jsonl_bytes:
            raise SystemExit("committed ProcessBench transfer JSONL differs from regeneration")
        if committed.manifest != rebuilt.manifest:
            raise SystemExit("committed ProcessBench transfer manifest differs from regeneration")
    else:
        write_processbench_transfer_bundle(
            rebuilt,
            data_path=data_output,
            manifest_path=manifest_output,
        )

    payload = {
        "mode": "check" if args.check else "generate",
        "source_root": str(source_root),
        "source_revision": PROCESSBENCH_REVISION,
        "data_path": str(data_output),
        "data_sha256": _sha256(data_output),
        "manifest_path": str(manifest_output),
        "manifest_sha256": _sha256(manifest_output),
        "semantic_manifest_hash": rebuilt.manifest["manifest_hash"],
        "counts": rebuilt.manifest["counts"],
        "run_roles": {
            role: {"pair_count": value["pair_count"]}
            for role, value in rebuilt.manifest["run_roles"].items()
        },
        "validation_accessed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
