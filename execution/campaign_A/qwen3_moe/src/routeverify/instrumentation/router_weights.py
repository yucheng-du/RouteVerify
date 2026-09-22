"""Export auditable router matrices without serializing the full model."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from routeverify.config import ModelConfig
from routeverify.instrumentation.sites import site_key
from routeverify.instrumentation.storage import atomic_write_json
from routeverify.models.moe import MoEBlockLayout, RouterSemantics


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _aggregate_router_rows(rows: list[dict[str, Any]], *, schema_version: int) -> str:
    if schema_version == 1:
        serialized = "\n".join(f"{row['array_key']}:{row['sha256']}" for row in rows)
    else:
        serialized = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def export_router_weight_matrices(
    *,
    adapter: Any,
    model_config: ModelConfig,
    output: str | Path,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Export layer→router weights as float32 NPZ plus a hashed JSON manifest."""

    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported router matrix schema version: {schema_version}")

    output_path = Path(output)
    sites = sorted(
        (site for site in adapter.discover_layers() if site.kind == "router_logits"),
        key=lambda site: (site.layer_index, site.module_name),
    )
    if not sites:
        raise RuntimeError("no router-logit sites were discovered")
    indices = [site.layer_index for site in sites]
    duplicates = sorted({index for index in indices if indices.count(index) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate router matrix layers discovered: {duplicates}")

    layout_method = getattr(adapter, "moe_block_layout", None)
    layout = layout_method() if callable(layout_method) else None
    if layout is not None and not isinstance(layout, MoEBlockLayout):
        raise TypeError("adapter.moe_block_layout() must return MoEBlockLayout")
    if layout is None and model_config.expected_total_block_count is not None:
        layout = MoEBlockLayout(
            total_block_count=model_config.expected_total_block_count,
            router_block_indices=tuple(indices),
        )
    if schema_version == 2 and layout is None:
        raise ValueError("router matrix schema v2 requires an explicit MoE block layout")
    if layout is not None and tuple(indices) != layout.router_block_indices:
        raise RuntimeError(
            "router export sites do not match adapter layout: "
            f"sites={indices} layout={list(layout.router_block_indices)}"
        )
    expected_total = model_config.expected_total_block_count
    if (
        layout is not None
        and expected_total is not None
        and layout.total_block_count != expected_total
    ):
        raise RuntimeError(
            "router export total block count does not match expected value: "
            f"layout={layout.total_block_count} expected={expected_total}"
        )
    arrays: dict[str, np.ndarray] = {}
    layers: list[dict[str, Any]] = []
    for site in sites:
        module = adapter._site_module(site)
        weight = getattr(module, "weight", None)
        if weight is None:
            linear = getattr(module, "linear", None)
            weight = getattr(linear, "weight", None)
        if weight is None:
            raise RuntimeError(f"router module has no weight matrix: {site.module_name}")
        source_dtype = str(getattr(weight, "dtype", "unknown"))
        if hasattr(weight, "detach"):
            array = weight.detach().float().cpu().numpy()
        else:
            array = np.asarray(weight, dtype=np.float32)
        array = np.ascontiguousarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"router weight must be a matrix at layer {site.layer_index}: {array.shape}"
            )
        key = f"layer_{site.layer_index}"
        arrays[key] = array
        row: dict[str, Any] = {
            "layer_index": site.layer_index,
            "site_key": site_key(site),
            "module_name": site.module_name,
            "array_key": key,
            "shape": list(array.shape),
            "source_dtype": source_dtype,
            "storage_dtype": str(array.dtype),
            "sha256": _array_sha256(array),
        }
        if layout is not None:
            row.update(layout.block_metadata(site.layer_index))
        layers.append(row)
    expected_routers = model_config.expected_router_block_count
    if expected_routers is not None and len(layers) != expected_routers:
        raise RuntimeError(
            f"expected {expected_routers} router matrices, discovered {len(layers)}"
        )
    expected_indices = model_config.expected_router_block_indices
    if expected_indices is not None and indices != expected_indices:
        raise RuntimeError(
            "router matrix indices do not match expected value: "
            f"discovered={indices} expected={expected_indices}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez(buffer, **cast(dict[str, Any], arrays))
    payload = buffer.getvalue()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)
    aggregate = _aggregate_router_rows(layers, schema_version=schema_version)
    semantics = getattr(adapter, "router_semantics", None)
    semantics_id = semantics.identifier if isinstance(semantics, RouterSemantics) else None
    total_blocks = (
        layout.total_block_count
        if layout is not None
        else model_config.expected_total_block_count
    )
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "format": "numpy_npz_float32",
        "adapter_alias": model_config.alias,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "tokenizer_revision": model_config.tokenizer_revision,
        "matrix_file": output_path.name,
        "matrix_file_sha256": hashlib.sha256(payload).hexdigest(),
        # Historical Qwen reviewers consume ``num_layers`` as the exported
        # router-matrix count. Keep it while adding unambiguous topology fields.
        "num_layers": len(layers),
        "total_blocks": total_blocks,
        "num_router_blocks": len(layers),
        "router_block_indices": indices,
        "router_semantics_id": semantics_id,
        "aggregate_matrix_sha256": aggregate,
        "layers": layers,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_router_weight_export(path: str | Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    matrix_path = Path(path)
    manifest_path = matrix_path.with_suffix(matrix_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version", 1)
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported router matrix schema version: {schema_version}")
    file_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    if file_hash != manifest.get("matrix_file_sha256"):
        raise ValueError("router matrix file hash does not match manifest")
    if manifest.get("matrix_file") != matrix_path.name:
        raise ValueError("router matrix filename does not match manifest")
    layer_rows = manifest.get("layers")
    if not isinstance(layer_rows, list) or manifest.get("num_layers") != len(layer_rows):
        raise ValueError("router matrix layer count does not match manifest")
    row_indices = [int(row["layer_index"]) for row in layer_rows]
    if schema_version >= 2 and row_indices != sorted(set(row_indices)):
        raise ValueError("router matrix layers must be strictly increasing and unique")
    if schema_version >= 2:
        if manifest.get("num_router_blocks") != len(layer_rows):
            raise ValueError("router matrix router-block count does not match manifest")
        if manifest.get("router_block_indices") != row_indices:
            raise ValueError("router matrix indices do not match manifest")
        total_blocks = manifest.get("total_blocks")
        if not isinstance(total_blocks, int) or total_blocks <= 0:
            raise ValueError("router matrix total block count is missing or invalid")
        for router_ordinal, row in enumerate(layer_rows):
            layer_index = int(row["layer_index"])
            expected_site_key = f"layer_{layer_index}.router_logits"
            if layer_index < 0 or layer_index >= total_blocks:
                raise ValueError(f"router matrix layer is outside total blocks: {layer_index}")
            if int(row.get("block_index", -1)) != layer_index:
                raise ValueError(f"router matrix block index mismatch: {expected_site_key}")
            if row.get("site_key") != expected_site_key:
                raise ValueError(f"router matrix site key mismatch: {expected_site_key}")
            if row.get("array_key") != f"layer_{layer_index}":
                raise ValueError(f"router matrix array key mismatch: {expected_site_key}")
            if row.get("router_ordinal") != router_ordinal:
                raise ValueError(f"router matrix ordinal mismatch: {expected_site_key}")
            denominator = max(total_blocks - 1, 1)
            if float(row.get("normalized_depth", -1.0)) != layer_index / denominator:
                raise ValueError(f"router matrix normalized depth mismatch: {expected_site_key}")
            if not str(row.get("module_name", "")):
                raise ValueError(f"router matrix module name is missing: {expected_site_key}")
    else:
        if manifest.get("num_router_blocks", len(layer_rows)) != len(layer_rows):
            raise ValueError("router matrix router-block count does not match manifest")
        if manifest.get("router_block_indices", row_indices) != row_indices:
            raise ValueError("router matrix indices do not match manifest")
    aggregate = _aggregate_router_rows(layer_rows, schema_version=int(schema_version))
    if aggregate != manifest.get("aggregate_matrix_sha256"):
        raise ValueError("router matrix aggregate hash does not match manifest")
    matrices: dict[int, np.ndarray] = {}
    with np.load(matrix_path, allow_pickle=False) as archive:
        expected_keys = {str(row["array_key"]) for row in layer_rows}
        if set(archive.files) != expected_keys:
            raise ValueError("router matrix archive keys do not match manifest")
        for row in layer_rows:
            array = np.asarray(archive[row["array_key"]])
            if list(array.shape) != row["shape"] or str(array.dtype) != row["storage_dtype"]:
                raise ValueError(f"router matrix metadata mismatch: {row['array_key']}")
            if _array_sha256(array) != row["sha256"]:
                raise ValueError(f"router matrix hash mismatch: {row['array_key']}")
            layer_index = int(row["layer_index"])
            if layer_index in matrices:
                raise ValueError(f"duplicate router matrix layer: {layer_index}")
            matrices[layer_index] = array
    return matrices, manifest
