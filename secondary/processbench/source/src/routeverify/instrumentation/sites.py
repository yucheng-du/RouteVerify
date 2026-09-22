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
class PatchBundle:
    """A set of sites patched atomically during one model forward.

    ``PatchSpec`` remains the single-site public API. Adapters accept either type,
    while this bundle makes simultaneous multi-layer interventions explicit rather
    than conflating them with a sequential layer sweep.
    """

    patches: tuple[PatchSpec, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.patches:
            raise ValueError("patch bundle must contain at least one patch")
        keys = [site_key(patch.site) for patch in self.patches]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate patch sites are not allowed: {duplicates}")


PatchRequest = PatchSpec | PatchBundle


def site_key(site: LayerSite) -> str:
    return f"layer_{site.layer_index}.{site.kind}"


def patch_specs(request: PatchRequest) -> tuple[PatchSpec, ...]:
    """Normalize the backwards-compatible single-site and bundle APIs."""

    return request.patches if isinstance(request, PatchBundle) else (request,)
