"""Atomic second-batch run packaging with unambiguous terminal markers."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path
from typing import Literal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_run_package(
    run_root: Path,
    package_path: Path,
    *,
    status: Literal["success", "failure"] = "success",
) -> None:
    """Hash and atomically package a run, then write its terminal marker last."""

    run_root = run_root.resolve()
    package_path = package_path.resolve()
    if status not in {"success", "failure"}:
        raise ValueError(f"unsupported terminal status: {status}")
    checksum_path = Path(f"{package_path}.sha256")
    partial_package = Path(f"{package_path}.partial")
    partial_checksum = Path(f"{checksum_path}.partial")
    sums_path = run_root / "SHA256SUMS"
    sums_tmp = run_root / "SHA256SUMS.tmp"
    success = run_root / "SUCCESS"
    failed = run_root / "FAILED"
    terminal_marker = success if status == "success" else failed
    if not run_root.is_dir():
        raise ValueError(f"run root is not a directory: {run_root}")
    if any(
        path.exists()
        for path in (package_path, checksum_path, partial_package, partial_checksum)
    ):
        raise FileExistsError("refusing to overwrite package/checksum/partial output")
    success.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    try:
        entries = []
        excluded = {"SHA256SUMS", "SUCCESS", "FAILED"}
        for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
            relative = path.relative_to(run_root)
            if relative.as_posix() in excluded:
                continue
            entries.append(f"{_sha256(path)}  {relative.as_posix()}\n")
        sums_tmp.write_text("".join(entries), encoding="utf-8")
        sums_tmp.replace(sums_path)

        package_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(partial_package, "w:gz") as archive:
            archive.add(run_root, arcname=run_root.name)
        partial_checksum.write_text(
            f"{_sha256(partial_package)}  {package_path.name}\n",
            encoding="utf-8",
        )
        partial_package.replace(package_path)
        partial_checksum.replace(checksum_path)
        terminal_marker.touch()
    except BaseException:
        success.unlink(missing_ok=True)
        sums_path.unlink(missing_ok=True)
        sums_tmp.unlink(missing_ok=True)
        for path in (partial_package, partial_checksum, package_path, checksum_path):
            path.unlink(missing_ok=True)
        failed.touch()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--status",
        choices=("success", "failure"),
        default="success",
        help="Terminal marker written only after the package and checksum are durable.",
    )
    args = parser.parse_args()
    finalize_run_package(args.run_root, args.package, status=args.status)


if __name__ == "__main__":
    main()
