"""Model-agnostic MoE layout, router, and no-weight preflight contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from routeverify.config import ModelConfig
from routeverify.instrumentation.sites import LayerSite, SiteKind, site_key

MOE_SITE_KINDS: tuple[SiteKind, ...] = (
    "pre_moe_residual",
    "router_input",
    "router_logits",
    "router_topk_ids",
    "router_topk_weights",
    "moe_output",
    "post_moe_residual",
)

SITE_SEMANTIC_BOUNDARIES: dict[SiteKind, str] = {
    "pre_moe_residual": "moe_branch_input",
    "router_input": "router_input",
    "router_logits": "pre_native_router_selection",
    "router_topk_ids": "native_selected_expert_ids",
    "router_topk_weights": "native_selected_expert_weights",
    "moe_output": "moe_branch_output",
    "post_moe_residual": "post_moe_block_output",
}


@dataclass(frozen=True)
class RouterOutput:
    """Normalized artifacts produced by an adapter's native router semantics."""

    logits: Any
    topk_weights: Any
    topk_ids: Any


@dataclass(frozen=True)
class RouterSemantics:
    """Named adapter-native mapping from router logits to dispatch artifacts.

    Shared patching code delegates to this callable and deliberately performs no
    softmax, sigmoid, normalization, grouping, or bias correction itself.
    """

    identifier: str
    route_from_logits: Callable[[Any, Any], RouterOutput]

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("router semantics identifier must be non-empty")

    def route(self, router_module: Any, router_logits: Any) -> RouterOutput:
        return self.route_from_logits(router_module, router_logits)


@dataclass(frozen=True)
class MoEBlockLayout:
    """Absolute decoder-block topology for a possibly sparse/hybrid MoE model."""

    total_block_count: int
    router_block_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.total_block_count <= 0:
            raise ValueError("total_block_count must be positive")
        if not self.router_block_indices:
            raise ValueError("router_block_indices must be non-empty")
        if tuple(sorted(set(self.router_block_indices))) != self.router_block_indices:
            raise ValueError("router_block_indices must be strictly increasing and unique")
        invalid = [
            index
            for index in self.router_block_indices
            if index < 0 or index >= self.total_block_count
        ]
        if invalid:
            raise ValueError(
                "router block indices are outside the decoder layout: "
                f"total={self.total_block_count} invalid={invalid}"
            )

    @property
    def router_block_count(self) -> int:
        return len(self.router_block_indices)

    def block_metadata(self, block_index: int) -> dict[str, int | float]:
        """Serialize router ordinal and ``index / (total_blocks - 1)`` depth."""

        try:
            router_ordinal = self.router_block_indices.index(block_index)
        except ValueError as exc:
            raise ValueError(f"block {block_index} is not a routed block") from exc
        denominator = max(self.total_block_count - 1, 1)
        return {
            "block_index": block_index,
            "router_ordinal": router_ordinal,
            "normalized_depth": block_index / denominator,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "total_blocks": self.total_block_count,
            "router_blocks": self.router_block_count,
            "router_block_indices": list(self.router_block_indices),
            "router_block_metadata": [
                self.block_metadata(index) for index in self.router_block_indices
            ],
        }


@dataclass(frozen=True)
class MoEConfigPaths:
    """Ordered model-specific paths for projecting an HF config into common fields."""

    model_type: tuple[str, ...] = ("model_type",)
    architectures: tuple[str, ...] = ("architectures",)
    total_blocks: tuple[str, ...] = ("num_hidden_layers",)
    experts: tuple[str, ...] = ("num_experts", "num_local_experts")
    top_k: tuple[str, ...] = ("num_experts_per_tok",)

    def __post_init__(self) -> None:
        for label, paths in self._items():
            if not paths or any(not path.strip() for path in paths):
                raise ValueError(f"{label} config paths must be non-empty")
            if len(set(paths)) != len(paths):
                raise ValueError(f"{label} config paths must be unique")

    def _items(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return (
            ("model_type", self.model_type),
            ("architectures", self.architectures),
            ("total_blocks", self.total_blocks),
            ("experts", self.experts),
            ("top_k", self.top_k),
        )


DEFAULT_MOE_CONFIG_PATHS = MoEConfigPaths()


@dataclass(frozen=True)
class ResolvedMoEConfig:
    """Canonical static config values plus their model-specific source paths."""

    model_type: str | None
    architectures: tuple[str, ...] | None
    total_blocks: int | None
    experts: int | None
    top_k: int | None
    sources: dict[str, str | None]
    errors: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": (
                None if self.architectures is None else list(self.architectures)
            ),
            "total_blocks": self.total_blocks,
            "experts": self.experts,
            "top_k": self.top_k,
            "source_paths": dict(self.sources),
        }


def resolve_moe_config(
    hf_config: Any,
    paths: MoEConfigPaths = DEFAULT_MOE_CONFIG_PATHS,
) -> ResolvedMoEConfig:
    """Resolve common config fields from explicit attribute/mapping paths."""

    errors: list[str] = []
    model_type, model_type_source = _resolve_config_field(
        hf_config,
        label="model_type",
        paths=paths.model_type,
        normalize=_normalize_string,
        errors=errors,
    )
    architectures, architectures_source = _resolve_config_field(
        hf_config,
        label="architectures",
        paths=paths.architectures,
        normalize=_normalize_architectures,
        errors=errors,
    )
    total_blocks, total_blocks_source = _resolve_config_field(
        hf_config,
        label="total_blocks",
        paths=paths.total_blocks,
        normalize=_normalize_positive_int,
        errors=errors,
    )
    experts, experts_source = _resolve_config_field(
        hf_config,
        label="experts",
        paths=paths.experts,
        normalize=_normalize_positive_int,
        errors=errors,
    )
    top_k, top_k_source = _resolve_config_field(
        hf_config,
        label="top_k",
        paths=paths.top_k,
        normalize=_normalize_positive_int,
        errors=errors,
    )
    return ResolvedMoEConfig(
        model_type=model_type,
        architectures=architectures,
        total_blocks=total_blocks,
        experts=experts,
        top_k=top_k,
        sources={
            "model_type": model_type_source,
            "architectures": architectures_source,
            "total_blocks": total_blocks_source,
            "experts": experts_source,
            "top_k": top_k_source,
        },
        errors=tuple(errors),
    )


def build_moe_sites(
    layout: MoEBlockLayout,
    module_name: Callable[[int, SiteKind], str],
    *,
    kinds: Sequence[SiteKind] = MOE_SITE_KINDS,
) -> list[LayerSite]:
    """Build canonical sites at the layout's absolute routed-block indices."""

    return [
        LayerSite(
            layer_index=block_index,
            kind=kind,
            module_name=module_name(block_index, kind),
        )
        for block_index in layout.router_block_indices
        for kind in kinds
    ]


def canonical_site_manifest(
    sites: Iterable[LayerSite],
    *,
    layout: MoEBlockLayout | None = None,
) -> list[dict[str, Any]]:
    """Return stable, duplicate-free site provenance sorted by absolute block index."""

    order = {kind: index for index, kind in enumerate(MOE_SITE_KINDS)}
    sorted_sites = sorted(
        sites,
        key=lambda site: (
            site.layer_index,
            order.get(site.kind, len(order)),
            site.kind,
            site.module_name,
        ),
    )
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for site in sorted_sites:
        key = site_key(site)
        if key in seen:
            raise ValueError(f"duplicate site key: {key}")
        seen.add(key)
        row: dict[str, Any] = {
            "site_key": key,
            "block_index": site.layer_index,
            "kind": site.kind,
            "module_name": site.module_name,
            "semantic_boundary": SITE_SEMANTIC_BOUNDARIES[site.kind],
        }
        if layout is not None and site.layer_index in layout.router_block_indices:
            row.update(layout.block_metadata(site.layer_index))
        rows.append(row)
    return rows


def preflight_moe_layout(
    *,
    model_config: ModelConfig,
    hf_config: Any,
    layout: MoEBlockLayout,
    sites: Iterable[LayerSite],
    router_semantics_id: str,
    config_paths: MoEConfigPaths = DEFAULT_MOE_CONFIG_PATHS,
    adapter_class: str | None = None,
    adapter_alias: str | None = None,
    adapter_model_id: str | None = None,
) -> dict[str, Any]:
    """Validate and serialize a static/synthetic layout without loading weights or doing I/O."""

    resolved = resolve_moe_config(hf_config, config_paths)
    errors = list(resolved.errors)
    try:
        site_rows = canonical_site_manifest(sites, layout=layout)
    except ValueError as exc:
        errors.append(str(exc))
        site_rows = []

    if not router_semantics_id.strip():
        errors.append("router semantics identifier is missing")
    if adapter_alias is not None and adapter_alias != model_config.alias:
        errors.append(
            "adapter alias does not match model config: "
            f"adapter={adapter_alias} config={model_config.alias}"
        )
    expected = {
        "total_blocks": model_config.expected_total_block_count,
        "router_blocks": model_config.expected_router_block_count,
        "router_block_indices": (
            None
            if model_config.expected_router_block_indices is None
            else list(model_config.expected_router_block_indices)
        ),
        "experts": model_config.expected_experts,
        "top_k": model_config.expected_top_k,
        "model_type": model_config.expected_model_type,
        "architectures": model_config.expected_architectures,
    }

    _compare_expected(
        errors,
        label="total block count",
        expected=expected["total_blocks"],
        actual=layout.total_block_count,
    )
    _compare_expected(
        errors,
        label="router block count",
        expected=expected["router_blocks"],
        actual=layout.router_block_count,
    )
    _compare_expected(
        errors,
        label="router block indices",
        expected=expected["router_block_indices"],
        actual=list(layout.router_block_indices),
    )
    _compare_expected(
        errors,
        label="expert count",
        expected=expected["experts"],
        actual=resolved.experts,
    )
    _compare_expected(
        errors,
        label="top-k expert count",
        expected=expected["top_k"],
        actual=resolved.top_k,
    )
    _compare_expected(
        errors,
        label="model type",
        expected=expected["model_type"],
        actual=resolved.model_type,
    )
    expected_architectures = model_config.expected_architectures
    if expected_architectures is not None:
        _compare_expected(
            errors,
            label="architectures",
            expected=sorted(expected_architectures),
            actual=(
                None if resolved.architectures is None else sorted(resolved.architectures)
            ),
        )
    if resolved.total_blocks is None:
        if expected["total_blocks"] is not None:
            errors.append("could not resolve total block count from static config")
    elif resolved.total_blocks != layout.total_block_count:
        errors.append(
            "static config total block count does not match adapter layout: "
            f"config={resolved.total_blocks} layout={layout.total_block_count}"
        )

    if site_rows:
        _validate_site_rows(errors, layout=layout, rows=site_rows)
    else:
        errors.append("adapter did not expose any common MoE sites")

    return {
        "schema_version": 1,
        "mode": "static_config_synthetic_modules",
        "requires_weights": False,
        "adapter": {
            "alias": adapter_alias or model_config.alias,
            "config_alias": model_config.alias,
            "class": adapter_class,
        },
        "model": {
            "model_id": model_config.model_id,
            "adapter_model_id": adapter_model_id,
            "revision": model_config.revision,
            "tokenizer_revision": model_config.tokenizer_revision,
        },
        "expected_contract": expected,
        "config_contract": resolved.to_metadata(),
        "layout": layout.to_metadata(),
        "router_semantics_id": router_semantics_id,
        "site_schema_version": 1,
        "sites": site_rows,
        "errors": errors,
        "passed": not errors,
    }


def preflight_moe_adapter(
    *,
    adapter: Any,
    model_config: ModelConfig,
    hf_config: Any,
) -> dict[str, Any]:
    """Run pure preflight against an adapter backed by empty or synthetic modules.

    This function never calls ``adapter.load``, Hugging Face loaders, cache setup,
    or filesystem/network APIs. The caller supplies the already available static
    config and an adapter whose lightweight module layout can be inspected.
    """

    semantics = adapter.router_semantics
    if not isinstance(semantics, RouterSemantics):
        raise TypeError("adapter.router_semantics must be RouterSemantics")
    return preflight_moe_layout(
        model_config=model_config,
        hf_config=hf_config,
        layout=adapter.moe_block_layout(),
        sites=adapter.discover_layers(),
        router_semantics_id=semantics.identifier,
        config_paths=adapter.moe_config_paths,
        adapter_class=f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        adapter_alias=getattr(adapter, "alias", None),
        adapter_model_id=getattr(adapter, "model_id", None),
    )


def _validate_site_rows(
    errors: list[str],
    *,
    layout: MoEBlockLayout,
    rows: list[dict[str, Any]],
) -> None:
    expected_kinds = set(MOE_SITE_KINDS)
    rows_by_block: dict[int, set[str]] = {}
    for row in rows:
        rows_by_block.setdefault(int(row["block_index"]), set()).add(str(row["kind"]))
    actual_indices = tuple(sorted(rows_by_block))
    if actual_indices != layout.router_block_indices:
        errors.append(
            "site router block indices do not match adapter layout: "
            f"sites={list(actual_indices)} layout={list(layout.router_block_indices)}"
        )
    for block_index in layout.router_block_indices:
        kinds = rows_by_block.get(block_index, set())
        if kinds != expected_kinds:
            errors.append(
                f"router block {block_index} does not expose the common site contract: "
                f"missing={sorted(expected_kinds - kinds)} extra={sorted(kinds - expected_kinds)}"
            )


def _compare_expected(
    errors: list[str],
    *,
    label: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected is None:
        return
    if actual is None:
        errors.append(f"could not verify expected {label}: value is missing")
    elif actual != expected:
        errors.append(f"{label} mismatch: actual={actual} expected={expected}")


def _resolve_config_field(
    root: Any,
    *,
    label: str,
    paths: tuple[str, ...],
    normalize: Callable[[Any], Any],
    errors: list[str],
) -> tuple[Any, str | None]:
    found: list[tuple[str, Any]] = []
    for path in paths:
        present, raw = _read_config_path(root, path)
        if not present or raw is None:
            continue
        try:
            found.append((path, normalize(raw)))
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid {label} at {path}: {exc}")
    if not found:
        return None, None
    first_path, first_value = found[0]
    conflicts = [(path, value) for path, value in found[1:] if value != first_value]
    if conflicts:
        rendered = ", ".join(
            [f"{first_path}={first_value!r}", *[f"{path}={value!r}" for path, value in conflicts]]
        )
        errors.append(f"conflicting {label} config values: {rendered}")
    return first_value, first_path


def _read_config_path(root: Any, path: str) -> tuple[bool, Any]:
    current = root
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return False, None
            current = current[component]
        elif hasattr(current, component):
            current = getattr(current, component)
        else:
            return False, None
    return True, current


def _normalize_string(value: Any) -> str:
    text = str(value)
    if not text:
        raise ValueError("must be non-empty")
    return text


def _normalize_architectures(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("must be a sequence of names")
    names = tuple(str(item) for item in value)
    if not names or any(not name for name in names):
        raise ValueError("must contain non-empty names")
    return names


def _normalize_positive_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("must be an integer, not bool")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError("must be positive")
    return normalized
