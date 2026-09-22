"""Atomic storage helpers for artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(output)


def write_safetensors(path: str | Path, tensors: dict[str, Any]) -> None:
    """Write tensors atomically using safetensors."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - exercised by model users
        raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    prepared = {
        key: value.contiguous() if hasattr(value, "contiguous") else value
        for key, value in tensors.items()
    }
    save_file(prepared, tmp)
    tmp.replace(output)
