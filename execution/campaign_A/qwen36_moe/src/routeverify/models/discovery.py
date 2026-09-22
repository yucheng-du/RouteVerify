"""Heuristic model-module discovery for MoE adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.models.moe import (
    DEFAULT_MOE_CONFIG_PATHS,
    MoEConfigPaths,
    resolve_moe_config,
)


@dataclass(frozen=True)
class ModuleCandidate:
    name: str
    class_name: str
    role: str

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "class_name": self.class_name, "role": self.role}


@dataclass
class DiscoveryReport:
    alias: str
    model_id: str
    candidates: list[ModuleCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "summary": self.summary(),
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.candidates:
            counts[candidate.role] = counts.get(candidate.role, 0) + 1
        return counts


def classify_module(name: str, module: Any) -> str | None:
    """Classify likely MoE-related modules by name/class heuristics."""

    lowered_name = name.lower()
    class_name = module.__class__.__name__.lower()
    name_parts = set(lowered_name.replace("_", ".").split("."))
    terminal_name = lowered_name.rsplit(".", maxsplit=1)[-1]
    if terminal_name in {"router", "gate"} or "router" in class_name:
        return "router"
    if (
        lowered_name.endswith(".experts")
        or lowered_name.endswith(".expert")
        or "expert" in class_name
    ):
        return "experts"
    if (
        "sparsemoe" in class_name
        or "moeblock" in class_name
        or "moe_block" in lowered_name
        or "moe" in name_parts
    ):
        return "moe_block"
    return None


def discover_from_named_modules(
    named_modules: list[tuple[str, Any]] | Any,
    *,
    alias: str,
    model_id: str,
    limit_modules: int | None = None,
) -> DiscoveryReport:
    """Build a discovery report from a model's named_modules iterable."""

    report = DiscoveryReport(alias=alias, model_id=model_id)
    iterator = named_modules if isinstance(named_modules, list) else list(named_modules)
    for index, (name, module) in enumerate(iterator):
        if limit_modules is not None and index >= limit_modules:
            break
        role = classify_module(name, module)
        if role is None:
            continue
        report.candidates.append(
            ModuleCandidate(name=name, class_name=module.__class__.__name__, role=role)
        )
    if not report.candidates:
        msg = "No MoE/router/expert modules were discovered; add an explicit adapter mapping."
        raise RuntimeError(msg)
    return report


def discover_from_model_id(
    config: ModelConfig,
    *,
    limit_modules: int | None = None,
    config_paths: MoEConfigPaths = DEFAULT_MOE_CONFIG_PATHS,
) -> DiscoveryReport:
    """Load a Hugging Face model and run module discovery."""

    ensure_external_cache()
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - exercised by model users
        raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

    hf_config = AutoConfig.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    if config.load_weights:
        model_factory = cast(Any, AutoModelForCausalLM)
        model = model_factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
            torch_dtype="auto",
            attn_implementation=config.attn_implementation,
        )
    else:
        model_factory = cast(Any, AutoModelForCausalLM)
        with init_empty_weights():
            model = model_factory.from_config(
                hf_config,
                trust_remote_code=config.trust_remote_code,
                attn_implementation=config.attn_implementation,
            )
    report = discover_from_named_modules(
        model.named_modules(),
        alias=config.alias,
        model_id=config.model_id,
        limit_modules=limit_modules,
    )
    resolved_config = resolve_moe_config(hf_config, config_paths)
    report.metadata.update(
        {
            "architectures": (
                None
                if resolved_config.architectures is None
                else list(resolved_config.architectures)
            ),
            "model_type": resolved_config.model_type,
            "num_hidden_layers": resolved_config.total_blocks,
            "num_experts": resolved_config.experts,
            "num_local_experts": resolved_config.experts,
            "num_experts_per_tok": resolved_config.top_k,
            "config_field_sources": resolved_config.sources,
            "config_resolution_errors": list(resolved_config.errors),
            "load_weights": config.load_weights,
        }
    )
    validate_expected_metadata(report, config)
    return report


def validate_expected_metadata(report: DiscoveryReport, config: ModelConfig) -> None:
    """Fail loudly when explicit architecture expectations are contradicted."""

    resolution_errors = report.metadata.get("config_resolution_errors") or []
    if resolution_errors:
        raise RuntimeError(
            "could not resolve a consistent model config contract: "
            + "; ".join(str(error) for error in resolution_errors)
        )
    if (
        config.expected_model_type is not None
        and report.metadata.get("model_type") != config.expected_model_type
    ):
        raise RuntimeError(
            "discovered model type does not match expected value: "
            f"model_type={report.metadata.get('model_type')} "
            f"expected={config.expected_model_type}"
        )
    if config.expected_architectures is not None and set(
        report.metadata.get("architectures") or []
    ) != set(config.expected_architectures):
        raise RuntimeError(
            "discovered architectures do not match expected value: "
            f"architectures={report.metadata.get('architectures')} "
            f"expected={config.expected_architectures}"
        )
    _validate_expected_value(
        label="layer count",
        metadata_key="num_hidden_layers",
        discovered=report.metadata.get("num_hidden_layers"),
        expected=config.expected_total_block_count,
    )
    _validate_expected_value(
        label="expert count",
        metadata_key="num_experts",
        discovered=report.metadata.get("num_experts"),
        expected=config.expected_experts,
    )
    _validate_expected_value(
        label="top-k expert count",
        metadata_key="num_experts_per_tok",
        discovered=report.metadata.get("num_experts_per_tok"),
        expected=config.expected_top_k,
    )


def _validate_expected_value(
    *,
    label: str,
    metadata_key: str,
    discovered: Any,
    expected: int | None,
) -> None:
    if expected is None:
        return
    if discovered is None:
        raise RuntimeError(f"could not verify expected {label}: missing {metadata_key}")
    if int(discovered) != expected:
        raise RuntimeError(
            f"discovered {label} does not match expected value: "
            f"{metadata_key}={discovered} expected={expected}"
        )
