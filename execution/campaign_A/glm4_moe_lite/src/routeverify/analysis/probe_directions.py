"""Produce and verify paired probe directions from frozen recapture artifacts."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from routeverify.data.schema import stable_hash
from routeverify.instrumentation.storage import atomic_write_json

ArrayLoader = Callable[[Path], Mapping[str, Any]]


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _default_array_loader(path: Path) -> Mapping[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional analysis dependency
        raise RuntimeError("Install analysis/model dependencies to read safetensors.") from exc
    return cast(Mapping[str, Any], load_file(path, device="cpu"))


def _resolve_activation(run: Path, value: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [
        path,
        run / path,
        run / path.name,
        run / "activations" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"recapture activation is missing: {value}")


def _vector(value: Any, *, token_index: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        if token_index >= array.shape[1]:
            raise IndexError(f"token_index {token_index} outside activation shape {array.shape}")
        array = array[:, token_index : token_index + 1, ...]
    vector = np.asarray(array, dtype=np.float32).reshape(-1)
    if not vector.size or not np.all(np.isfinite(vector)):
        raise ValueError("probe-direction activation must be non-empty and finite")
    return vector


def produce_probe_direction_artifact(
    *,
    run: str | Path,
    output: str | Path,
    site_kind: str = "router_input",
    array_loader: ArrayLoader | None = None,
) -> dict[str, Any]:
    """Write unit-normalized mean paired corrupt-minus-clean directions."""

    if site_kind != "router_input":
        raise ValueError("formal probe directions must use site_kind=router_input")
    run_path = Path(run)
    output_path = Path(output)
    rows = [
        json.loads(line)
        for line in (run_path / "data_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection = json.loads((run_path / "selection_manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((run_path / "environment.json").read_text(encoding="utf-8"))
    frozen = selection.get("frozen_split")
    if not isinstance(frozen, dict) or not frozen.get("manifest_hash") or not frozen.get(
        "partition"
    ):
        raise ValueError("probe directions require a frozen split selection manifest")
    if any(row.get("boundary_strategy") != "paired_first_error" for row in rows):
        raise ValueError("probe directions require paired_first_error recapture rows")
    loader = array_loader or _default_array_loader
    cache: dict[Path, Mapping[str, Any]] = {}
    vectors: dict[tuple[str, str, int], np.ndarray] = {}
    group_ids: dict[str, str] = {}
    activation_hashes: dict[str, str] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        variant = str(row.get("variant") or "")
        group_id = str(row.get("group_id") or "")
        if not pair_id or variant not in {"clean", "corrupt"} or not group_id:
            raise ValueError("recapture rows must bind pair_id, group_id, and clean/corrupt variant")
        previous_group = group_ids.setdefault(pair_id, group_id)
        if previous_group != group_id:
            raise ValueError(f"pair {pair_id} spans multiple groups")
        activation_path = _resolve_activation(run_path, str(row["activation_file"]))
        arrays = cache.setdefault(activation_path, loader(activation_path))
        activation_hashes[str(activation_path)] = hashlib.sha256(
            activation_path.read_bytes()
        ).hexdigest()
        keys = sorted(
            key for key in arrays if key.endswith(f".{site_kind}") and key.startswith("layer_")
        )
        if not keys:
            raise ValueError(f"no {site_kind} arrays found in {activation_path}")
        for key in keys:
            layer = int(key.split(".", 1)[0].removeprefix("layer_"))
            vector_key = (pair_id, variant, layer)
            if vector_key in vectors:
                raise ValueError(f"duplicate recapture vector: {vector_key}")
            vectors[vector_key] = _vector(arrays[key], token_index=int(row.get("token_index", 0)))

    pair_ids = sorted(group_ids)
    layers = sorted({layer for _pair, _variant, layer in vectors})
    directions: dict[str, np.ndarray] = {}
    layer_rows: list[dict[str, Any]] = []
    for layer in layers:
        deltas = []
        for pair_id in pair_ids:
            clean_key = (pair_id, "clean", layer)
            corrupt_key = (pair_id, "corrupt", layer)
            if clean_key not in vectors or corrupt_key not in vectors:
                raise ValueError(f"layer {layer} is missing a clean/corrupt vector for {pair_id}")
            if vectors[clean_key].shape != vectors[corrupt_key].shape:
                raise ValueError(f"layer {layer} clean/corrupt dimensions differ for {pair_id}")
            deltas.append(vectors[corrupt_key] - vectors[clean_key])
        direction = np.mean(np.stack(deltas), axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"layer {layer} paired mean direction has zero/nonfinite norm")
        direction = np.asarray(direction / norm, dtype=np.float32)
        key = f"layer_{layer}"
        directions[key] = direction
        layer_rows.append(
            {
                "layer_index": layer,
                "site": f"layer_{layer}.{site_kind}",
                "array_key": key,
                "shape": list(direction.shape),
                "dtype": str(direction.dtype),
                "sha256": _array_sha256(direction),
                "pair_count": len(pair_ids),
            }
        )

    buffer = io.BytesIO()
    np.savez(buffer, **cast(dict[str, Any], directions))
    payload = buffer.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)
    model = environment.get("model", {})
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": "numpy_npz_float32",
        "direction_definition": "unit_normalized_mean_paired_corrupt_minus_clean_router_input",
        "site_kind": site_kind,
        "model_id": model.get("model_id"),
        "model_revision": model.get("revision"),
        "direction_file": output_path.name,
        "direction_file_sha256": hashlib.sha256(payload).hexdigest(),
        "num_layers": len(layer_rows),
        "layers": layer_rows,
        "pair_ids": pair_ids,
        "group_ids": sorted(group_ids.values()),
        "pairs_hash": stable_hash(pair_ids, length=64),
        "groups_hash": stable_hash(sorted(group_ids.values()), length=64),
        "selection_manifest_hash": selection.get("manifest_hash"),
        "selection_pairs_hash": selection.get("pairs_hash"),
        "frozen_split": frozen,
        "activation_file_hashes": activation_hashes,
    }
    manifest["manifest_hash"] = stable_hash(manifest, length=64)
    atomic_write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def load_probe_direction_artifact(
    path: str | Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    direction_path = Path(path)
    manifest_path = direction_path.with_suffix(direction_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_manifest_hash = manifest.get("manifest_hash")
    unhashed = dict(manifest)
    unhashed.pop("manifest_hash", None)
    if recorded_manifest_hash != stable_hash(unhashed, length=64):
        raise ValueError("probe direction manifest hash mismatch")
    if hashlib.sha256(direction_path.read_bytes()).hexdigest() != manifest.get(
        "direction_file_sha256"
    ):
        raise ValueError("probe direction file hash does not match manifest")
    directions: dict[int, np.ndarray] = {}
    with np.load(direction_path, allow_pickle=False) as archive:
        expected_keys = {str(row["array_key"]) for row in manifest.get("layers", [])}
        if set(archive.files) != expected_keys:
            raise ValueError("probe direction archive keys do not match manifest")
        for row in manifest["layers"]:
            array = np.asarray(archive[row["array_key"]])
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                raise ValueError(f"probe direction metadata mismatch: {row['array_key']}")
            if _array_sha256(array) != row["sha256"]:
                raise ValueError(f"probe direction hash mismatch: {row['array_key']}")
            directions[int(row["layer_index"])] = array
    return directions, manifest


def validate_router_direction_compatibility(
    *,
    matrix_manifest: Mapping[str, Any],
    direction_manifest: Mapping[str, Any],
    matrix_layers: set[int],
    direction_layers: set[int],
) -> None:
    if not direction_manifest.get("model_id") or not direction_manifest.get("model_revision"):
        raise ValueError("probe direction artifact lacks pinned model provenance")
    for field in ("model_id", "model_revision"):
        if matrix_manifest.get(field) != direction_manifest.get(field):
            raise ValueError(f"router/direction {field} mismatch")
    if direction_manifest.get("site_kind") != "router_input":
        raise ValueError("router subspace analysis requires router_input directions")
    if (
        direction_manifest.get("direction_definition")
        != "unit_normalized_mean_paired_corrupt_minus_clean_router_input"
    ):
        raise ValueError("unsupported probe direction definition")
    frozen = direction_manifest.get("frozen_split")
    if not isinstance(frozen, Mapping) or not frozen.get("manifest_hash") or not frozen.get(
        "partition"
    ):
        raise ValueError("probe direction artifact lacks frozen split provenance")
    if not direction_manifest.get("selection_manifest_hash"):
        raise ValueError("probe direction artifact lacks selection provenance")
    if matrix_layers != direction_layers:
        raise ValueError(
            f"router/direction layer mismatch: {sorted(matrix_layers)} vs "
            f"{sorted(direction_layers)}"
        )
