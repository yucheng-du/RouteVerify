"""Configuration loading and run defaults."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EXTERNAL_ROOT = Path(".cache")
DEFAULT_HF_HOME = DEFAULT_EXTERNAL_ROOT / "huggingface"


class PathConfig(BaseModel):
    """Filesystem defaults for local runs."""

    model_cache: Path = DEFAULT_HF_HOME / "hub"
    dataset_cache: Path = DEFAULT_HF_HOME / "datasets"
    artifact_root: Path = Path("artifacts")
    data_root: Path = Path("data")


class RuntimeConfig(BaseModel):
    """Common runtime knobs used by CLI commands."""

    seed: int = 0
    limit: int | None = None
    dry_run: bool = False
    device: str = "auto"
    dtype: str = "bfloat16"


class ModelConfig(BaseModel):
    """Model loading and inspection settings."""

    model_config = ConfigDict(extra="allow")

    alias: str
    model_id: str
    revision: str | None = None
    tokenizer_revision: str | None = None
    cache_dir: Path = DEFAULT_HF_HOME / "hub"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    device_map: str | dict[str, Any] = "auto"
    dtype: str = "bfloat16"
    load_weights: bool = False
    # ``expected_layers`` is the legacy all-routed contract. New hybrid/sparse
    # adapters should set the two block counts independently and, when known,
    # bind the absolute decoder indices that contain routers.
    expected_layers: int | None = None
    expected_total_blocks: int | None = Field(default=None, gt=0)
    expected_router_blocks: int | None = Field(default=None, gt=0)
    expected_router_block_indices: list[int] | None = None
    expected_experts: int | None = None
    expected_top_k: int | None = None
    expected_model_type: str | None = None
    expected_architectures: list[str] | None = None
    expected_chat_template_sha256: str | None = None
    expected_transformers_version: str | None = None
    expected_kernels_version: str | None = None
    expected_triton_version: str | None = None
    expected_mxfp4_kernel_repo: str | None = None
    expected_mxfp4_kernel_revision: str | None = None
    require_native_mxfp4: bool = False
    harmony_reasoning_effort: str = "medium"

    @model_validator(mode="after")
    def _validate_block_contract(self) -> Self:
        if "expected_moe_layers" in (self.__pydantic_extra__ or {}):
            raise ValueError(
                "expected_moe_layers is ambiguous; use expected_router_blocks and "
                "expected_router_block_indices"
            )
        if (
            self.expected_layers is not None
            and self.expected_total_blocks is not None
            and self.expected_layers != self.expected_total_blocks
        ):
            raise ValueError("expected_layers and expected_total_blocks must agree")
        indices = self.expected_router_block_indices
        if indices is not None:
            if not indices:
                raise ValueError("expected_router_block_indices must be non-empty")
            if any(index < 0 for index in indices):
                raise ValueError("expected_router_block_indices must be non-negative")
            if indices != sorted(set(indices)):
                raise ValueError(
                    "expected_router_block_indices must be strictly increasing and unique"
                )
            if (
                self.expected_router_blocks is not None
                and len(indices) != self.expected_router_blocks
            ):
                raise ValueError(
                    "expected_router_blocks must equal the number of "
                    "expected_router_block_indices"
                )
            total = self.expected_total_block_count
            if total is not None and any(index >= total for index in indices):
                raise ValueError("expected router block index is outside expected total blocks")
        total = self.expected_total_block_count
        routers = self.expected_router_block_count
        if total is not None and routers is not None and routers > total:
            raise ValueError("expected_router_blocks cannot exceed expected total blocks")
        return self

    @property
    def expected_total_block_count(self) -> int | None:
        """Return the explicit total-block contract or its legacy equivalent."""

        return (
            self.expected_total_blocks
            if self.expected_total_blocks is not None
            else self.expected_layers
        )

    @property
    def expected_router_block_count(self) -> int | None:
        """Return the router-block contract without conflating an explicit total count."""

        if self.expected_router_blocks is not None:
            return self.expected_router_blocks
        if self.expected_router_block_indices is not None:
            return len(self.expected_router_block_indices)
        # Backwards compatibility for existing OLMoE/Qwen configs, whose
        # ``expected_layers`` historically meant both totals because every block routes.
        return self.expected_layers


class DataConfig(BaseModel):
    """Dataset source and normalized output settings."""

    model_config = ConfigDict(extra="allow")

    dataset: str
    hf_id: str | None = None
    revision: str | None = None
    expected_resolved_data_revision: str | None = None
    expected_raw_sha256: str | None = None
    subset: str | None = None
    split: str = "train"
    cache_dir: Path = DEFAULT_HF_HOME / "datasets"
    output: Path | None = None
    pair_output: Path | None = None
    selection_manifest: Path | None = None
    expected_processed_sha256: str | None = None
    expected_selection_manifest_sha256: str | None = None
    validation_accessed: bool | None = None
    balanced: bool = False


class ExperimentConfig(BaseModel):
    """Minimal experiment config shared by pilot commands."""

    model: str | Path
    data: str | Path
    output_dir: Path = Path("artifacts")
    seed: int = 0
    limit: int | None = None
    sites: list[str] = Field(default_factory=list)
    layers: list[int] | None = None
    prompt_mode: str = "plain_trace"
    input_format: str = "raw"
    boundary_strategy: str = "prefix_until_error"
    patch_mode: str = "single_site_sweep"
    alphas: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    forced_reroute_control: bool = False
    controls: list[str] = Field(default_factory=list)
    split_manifest: Path | None = None
    split_partition: str | None = None
    expected_router_hooks: int | None = None
    export_router_weights: Path | None = None
    router_weight_schema_version: int = Field(default=1, ge=1, le=2)
    include_equivalence_analysis: bool = True
    identity_tolerance: float = Field(default=1.0e-5, ge=0.0)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a plain mapping."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        msg = f"Config must be a mapping: {config_path}"
        raise ValueError(msg)
    return loaded


def load_model_config(path: str | Path) -> ModelConfig:
    config = ModelConfig.model_validate(load_yaml(path))
    cache_dir = _model_cache_from_env()
    if cache_dir is not None:
        config = config.model_copy(update={"cache_dir": cache_dir})
    return config


def load_data_config(path: str | Path) -> DataConfig:
    config = DataConfig.model_validate(load_yaml(path))
    cache_dir = _dataset_cache_from_env()
    if cache_dir is not None:
        config = config.model_copy(update={"cache_dir": cache_dir})
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def ensure_external_cache(paths: PathConfig | None = None) -> PathConfig:
    """Create and export Hugging Face cache paths, preferring the external drive."""

    resolved = paths or _path_config_from_env()
    resolved.model_cache.mkdir(parents=True, exist_ok=True)
    resolved.dataset_cache.mkdir(parents=True, exist_ok=True)
    hf_home = resolved.model_cache.parent
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(resolved.model_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(resolved.dataset_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(resolved.model_cache))
    return resolved


def _path_config_from_env() -> PathConfig:
    model_cache = _model_cache_from_env()
    dataset_cache = _dataset_cache_from_env()
    if model_cache is None and dataset_cache is None:
        return PathConfig()
    hf_home = _hf_home_from_env()
    return PathConfig(
        model_cache=model_cache or hf_home / "hub",
        dataset_cache=dataset_cache or hf_home / "datasets",
    )


def _hf_home_from_env() -> Path:
    value = os.environ.get("HF_HOME")
    return Path(value) if value else DEFAULT_HF_HOME


def _model_cache_from_env() -> Path | None:
    value = os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE")
    if value:
        return Path(value)
    hf_home = os.environ.get("HF_HOME")
    return None if hf_home is None else Path(hf_home) / "hub"


def _dataset_cache_from_env() -> Path | None:
    value = os.environ.get("HF_DATASETS_CACHE")
    if value:
        return Path(value)
    hf_home = os.environ.get("HF_HOME")
    return None if hf_home is None else Path(hf_home) / "datasets"


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    """Return git metadata, tolerating a non-git checkout."""

    root = Path(cwd)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"available": False, "commit": None, "dirty": None}


def environment_snapshot(
    *,
    cwd: str | Path = ".",
    model: ModelConfig | None = None,
    data: DataConfig | None = None,
    data_provenance: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a JSON-serializable environment summary for run artifacts."""

    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_state(cwd),
        "seed": seed,
    }
    if model is not None:
        snapshot["model"] = model.model_dump(mode="json")
    if data is not None:
        snapshot["data"] = data.model_dump(mode="json")
    if data_provenance is not None:
        snapshot["data_provenance"] = data_provenance
    for package in ("torch", "transformers", "accelerate", "datasets"):
        try:
            module = __import__(package)
            snapshot[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            snapshot[package] = None
    snapshot["cuda"] = _cuda_snapshot()
    snapshot["files"] = _tracked_file_hashes(Path(cwd))
    snapshot["runtime_env"] = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "HOSTNAME",
            "RUNPOD_POD_ID",
        )
        if os.environ.get(key) is not None
    }
    return snapshot


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}

    payload: dict[str, Any] = {
        "torch_available": True,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    payload["devices"] = devices
    return payload


def _tracked_file_hashes(root: Path) -> dict[str, str | None]:
    return {
        "pyproject.toml": _sha256_if_present(root / "pyproject.toml"),
        "uv.lock": _sha256_if_present(root / "uv.lock"),
    }


def _sha256_if_present(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_environment(path: str | Path, snapshot: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
