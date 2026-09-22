#!/usr/bin/env python3
"""Run the artifact-only ProcessBench natural-transfer review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routeverify.reporting.processbench_transfer import (
    review_processbench_transfer_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--run-role", choices=("smoke", "full"), required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--behavior-run", type=Path, required=True)
    parser.add_argument("--recapture-run", type=Path, required=True)
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    review = review_processbench_transfer_artifacts(
        repo_root=repo_root,
        profile_config=_resolve(repo_root, args.profile_config),
        model_profile=args.model_profile,
        run_role=args.run_role,
        runtime_report=_resolve(repo_root, args.runtime_report),
        behavior_run=_resolve(repo_root, args.behavior_run),
        recapture_run=_resolve(repo_root, args.recapture_run),
        transfer_run=_resolve(repo_root, args.transfer_run),
    )
    output = _resolve(repo_root, args.output)
    _write_json(output, review)
    print(json.dumps(review, indent=2, sort_keys=True))
    if not review["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
