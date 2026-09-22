"""Names and specs for capture/patch sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SiteKind = Literal[
    "pre_moe_residual",
    "router_input",
    "router_logits",
    "router_topk_ids",
    "router_topk_weights",
    "moe_output",
    "post_moe_residual",
]


@dataclass(frozen=True)
class LayerSite:
    layer_index: int
    kind: SiteKind
    module_name: str


@dataclass(frozen=True)
class CaptureSpec:
    sites: tuple[LayerSite, ...]
    token_positions: tuple[int, ...]
    dtype: str = "float16"

    def __post_init__(self) -> None:
        validate_token_positions(self.token_positions, label="capture token_positions")
        keys = [site_key(site) for site in self.sites]
        if not keys:
            raise ValueError("capture sites must be non-empty")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate capture sites are not allowed")


@dataclass(frozen=True)
class PatchSpec:
    site: LayerSite
    token_position: int
    replacement: object
    strict_shape: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replacement is None:
            raise ValueError(f"patch replacement is required for {site_key(self.site)}")
        if self.token_position < 0:
            raise ValueError("patch token_position must be non-negative")


@dataclass(frozen=True)
class MultiPositionPatchSpec:
    """Replace a complete window atomically at one site in one forward.

    Position order determines the replacement's token axis. Unlike ``PatchSpec``,
    even a singleton window keeps that axis: ``[batch, positions, ...]``. Repeated
    donor positions are permitted in the mapping that builds ``replacement``;
    receiver positions in this spec must be unique.
    """

    site: LayerSite
    token_positions: tuple[int, ...]
    replacement: object
    strict_shape: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.replacement is None:
            raise ValueError(f"patch replacement is required for {site_key(self.site)}")
        validate_token_positions(self.token_positions, label="patch token_positions")


PatchSiteSpec = PatchSpec | MultiPositionPatchSpec


@dataclass(frozen=True)
class PatchBundle:
    """A set of sites patched atomically during one model forward.

    ``PatchSpec`` remains the single-site public API. Adapters accept either type,
    while this bundle makes simultaneous multi-layer interventions explicit rather
    than conflating them with a sequential layer sweep.
    """

    patches: tuple[PatchSiteSpec, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("patch bundle must contain at least one patch")
        keys = [site_key(patch.site) for patch in self.patches]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate patch sites are not allowed: {duplicates}")


PatchRequest = PatchSiteSpec | PatchBundle


def site_key(site: LayerSite) -> str:
    return f"layer_{site.layer_index}.{site.kind}"


def patch_specs(request: PatchRequest) -> tuple[PatchSiteSpec, ...]:
    """Normalize the backwards-compatible single-site and bundle APIs."""

    return request.patches if isinstance(request, PatchBundle) else (request,)


def patch_positions(patch: PatchSiteSpec) -> tuple[int, ...]:
    """Return every receiver target without changing the legacy patch API."""

    if isinstance(patch, MultiPositionPatchSpec):
        return patch.token_positions
    return (patch.token_position,)


def patch_index(patch: PatchSiteSpec) -> int | list[int]:
    """Preserve the token axis for windows, and the old slice shape for one token."""

    if isinstance(patch, MultiPositionPatchSpec):
        return list(patch.token_positions)
    return patch.token_position


def validate_token_positions(positions: tuple[int, ...], *, label: str) -> None:
    if not positions:
        raise ValueError(f"{label} must be non-empty")
    if any(not isinstance(position, int) or isinstance(position, bool) for position in positions):
        raise TypeError(f"{label} must contain integers")
    if any(position < 0 for position in positions):
        raise ValueError(f"{label} must be non-negative")
    if len(set(positions)) != len(positions):
        raise ValueError(f"duplicate {label} are not allowed")
