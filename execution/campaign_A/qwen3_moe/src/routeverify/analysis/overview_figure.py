"""Project overview figure for the RouteVerify manuscript.

The figure is intentionally conceptual: it explains the paired first-error setting,
the three intervention topologies, and the evidence/claim boundary without
replotting the quantitative result figures.  Scientific labels are derived from the
frozen unified-results schema and checked before rendering.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase

PAPER_WIDTH_IN = 6.75
OVERVIEW_HEIGHT_IN = 2.52
MINIMUM_FONT_SIZE_PT = 6.5

RESIDUAL = "#0072B2"
MOE_OUTPUT = "#CC79A7"
SAME_ROUTER = "#D55E00"
ALL_ROUTER = "#009E73"
INK = "#222222"
MUTED = "#5F6368"
PANEL_EDGE = "#CBD3D8"
PANEL_FILL = "#F8FAFB"
ACCESS_FILL = "#E8F3F8"
MODERN_FILL = "#E6F2F8"
EXTENDED_FILL = "#FFF3DA"
LEGACY_FILL = "#EFEFEF"
BOUNDARY_FILL = "#F2ECF7"
ERROR = "#B44A3A"
CORRECT = "#2B7A4B"

MODEL_ORDER = (
    "gemma4",
    "nemotron3_nano",
    "nemotron35_lightning",
    "qwen36_moe",
    "gpt_oss_20b",
    "qwen3",
    "olmoe",
)

MOE_OUTPUT_MODEL_IDS = (
    "gemma4",
    "nemotron3_nano",
    "nemotron35_lightning",
    "qwen36_moe",
    "gpt_oss_20b",
    "qwen3",
)


@dataclass(frozen=True)
class Region:
    """Axis-relative rectangle used as a layout ownership boundary."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class OwnedText:
    artist: Text
    owner: Region


@dataclass(frozen=True)
class OwnedArtist:
    artist: Artist
    owner: Region
    allow_touching_owner: bool = False


@dataclass(frozen=True)
class Connector:
    artist: FancyArrowPatch
    source: Region
    target: Region


@dataclass(frozen=True)
class OverviewSpec:
    """Scientific facts that the conceptual figure is allowed to state."""

    analysis_id: str
    protocol_coverage: tuple[tuple[str, str, bool, bool, bool], ...]
    qwen36_alpha1_t: float
    qwen36_alpha2_t: float
    qwen36_alpha2_ci_low: float
    qwen36_row_ids: tuple[str, str]
    processbench_models: tuple[str, ...]
    processbench_n_pairs: int
    access_model_count: int
    access_group_count: int
    moe_output_model_ids: tuple[str, ...]


def configure_overview_style() -> None:
    """Apply the same vector-safe visual language as the result figures."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "routeverify-overview-v1",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def build_overview_spec(payload: Mapping[str, Any]) -> OverviewSpec:
    """Validate frozen headline structure and derive overview-only facts."""

    headline_raw = payload.get("headline_results")
    if not isinstance(headline_raw, Mapping):
        raise ValueError("unified results lack headline_results")
    headline = cast(Mapping[str, Any], headline_raw)

    coverage: list[tuple[str, str, bool, bool, bool]] = []
    for model_id in MODEL_ORDER:
        key = f"prmbench|{model_id}"
        row = headline.get(key)
        if not isinstance(row, Mapping):
            raise KeyError(f"missing headline row: {key}")
        residual = isinstance(row.get("residual"), Mapping)
        same = isinstance(row.get("same_layer_router"), Mapping)
        all_router = row.get("all_router")
        bundle = isinstance(all_router, Mapping) and any(
            isinstance(value, Mapping) for value in all_router.values()
        )
        grade = str(row.get("protocol_grade", ""))
        coverage.append((model_id, grade, residual, same, bundle))
        if not residual:
            raise ValueError(f"overview requires a residual row for {model_id}")

    expected_coverage = {
        "gemma4": ("modern", True, True, True),
        "nemotron3_nano": ("modern", True, True, True),
        "nemotron35_lightning": ("modern", True, True, True),
        "qwen36_moe": ("modern", True, True, True),
        "gpt_oss_20b": ("extended", True, True, True),
        "qwen3": ("extended", True, True, False),
        "olmoe": ("legacy-compatible", True, False, True),
    }
    for model_id, grade, residual, same, bundle in coverage:
        if (grade, residual, same, bundle) != expected_coverage[model_id]:
            raise ValueError(
                "protocol topology changed for "
                f"{model_id}: {(grade, residual, same, bundle)}"
            )

    qwen = cast(Mapping[str, Any], headline["prmbench|qwen36_moe"])
    qwen_all = cast(Mapping[str, Any], qwen["all_router"])
    alpha1 = _dose_cell(qwen_all, 1.0)
    alpha2 = _dose_cell(qwen_all, 2.0)
    alpha1_t = cast(Mapping[str, Any], alpha1["donor_aligned_t"])
    alpha2_t = cast(Mapping[str, Any], alpha2["donor_aligned_t"])
    alpha1_estimate = float(alpha1_t["estimate"])
    alpha2_estimate = float(alpha2_t["estimate"])
    alpha2_low = float(alpha2_t["ci_low"])
    if not (alpha2_estimate > alpha1_estimate and alpha2_low > 0.0):
        raise ValueError("Qwen3.6 no longer supports the stated dose boundary")

    processbench_models = tuple(
        model_id
        for model_id in ("gpt_oss_20b", "gemma4", "nemotron3_nano")
        if isinstance(headline.get(f"processbench|{model_id}"), Mapping)
    )
    if len(processbench_models) != 3:
        raise ValueError("overview expects the frozen three-model ProcessBench extension")
    process_ns: set[int] = set()
    for model_id in processbench_models:
        process = cast(Mapping[str, Any], headline[f"processbench|{model_id}"])
        process_residual = cast(Mapping[str, Any], process["residual"])
        t = cast(Mapping[str, Any], process_residual["donor_aligned_t"])
        process_ns.add(int(t["n_pairs"]))
    if process_ns != {64}:
        raise ValueError(f"unexpected ProcessBench pair counts: {sorted(process_ns)}")

    return OverviewSpec(
        analysis_id=str(payload.get("analysis_id", "unknown")),
        protocol_coverage=tuple(coverage),
        qwen36_alpha1_t=alpha1_estimate,
        qwen36_alpha2_t=alpha2_estimate,
        qwen36_alpha2_ci_low=alpha2_low,
        qwen36_row_ids=(str(alpha1_t["row_id"]), str(alpha2_t["row_id"])),
        processbench_models=processbench_models,
        processbench_n_pairs=process_ns.pop(),
        access_model_count=6,
        access_group_count=235,
        moe_output_model_ids=MOE_OUTPUT_MODEL_IDS,
    )


def _dose_cell(all_router: Mapping[str, Any], alpha: float) -> Mapping[str, Any]:
    value = all_router.get(str(alpha)) or all_router.get(str(int(alpha)))
    if not isinstance(value, Mapping):
        raise KeyError(f"missing all-router alpha={alpha:g}")
    return cast(Mapping[str, Any], value)


def make_overview_figure(spec: OverviewSpec) -> mpl.figure.Figure:
    """Render the project-specific early overview figure."""

    configure_overview_style()
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, OVERVIEW_HEIGHT_IN), facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    texts: list[OwnedText] = []
    artists: list[OwnedArtist] = []
    connectors: list[Connector] = []

    panel_a = Region("panel-a", 0.012, 0.045, 0.244, 0.958)
    panel_b = Region("panel-b", 0.272, 0.045, 0.653, 0.958)
    panel_c = Region("panel-c", 0.681, 0.045, 0.988, 0.958)
    for panel in (panel_a, panel_b, panel_c):
        _owned_round_box(ax, artists, panel, panel, facecolor=PANEL_FILL, edgecolor=PANEL_EDGE)

    _panel_a(ax, texts, artists, panel_a)
    _panel_b(ax, texts, artists, panel_b)
    _panel_c(ax, texts, artists, panel_c, spec)

    _flow_connector(ax, connectors, panel_a, panel_b)
    _flow_connector(ax, connectors, panel_b, panel_c)

    validate_overview_layout(
        fig,
        ax=ax,
        texts=texts,
        artists=artists,
        connectors=connectors,
        expected_width=PAPER_WIDTH_IN,
        minimum_font_size=MINIMUM_FONT_SIZE_PT,
    )
    fig.__dict__["_routeverify_overview_registry"] = {
        "texts": texts,
        "artists": artists,
        "connectors": connectors,
    }
    return fig


def _panel_a(
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    artists: list[OwnedArtist],
    panel: Region,
) -> None:
    title = Region("a-title", 0.030, 0.850, 0.238, 0.930)
    _owned_text(ax, texts, title, 0.032, 0.905, "a  First-error pairs", size=7.8, weight="bold")

    clean_label = Region("clean-label", 0.029, 0.692, 0.091, 0.768)
    error_label = Region("error-label", 0.029, 0.502, 0.091, 0.578)
    _owned_text(ax, texts, clean_label, 0.032, 0.735, "correct", size=6.7, color=CORRECT, weight="bold")
    _owned_text(ax, texts, error_label, 0.032, 0.545, "error", size=6.7, color=ERROR, weight="bold")

    for index, x in enumerate((0.112, 0.155, 0.198), start=1):
        region = Region(f"clean-step-{index}", x - 0.017, 0.665, x + 0.017, 0.775)
        _token(ax, artists, texts, region, f"s{index}", CORRECT)
    for index, x in enumerate((0.112, 0.155, 0.198), start=1):
        region = Region(f"error-step-{index}", x - 0.017, 0.475, x + 0.017, 0.585)
        _token(ax, artists, texts, region, f"s{index}", CORRECT if index < 3 else ERROR)

    first_error = Region("first-error", 0.161, 0.390, 0.228, 0.462)
    _owned_text(
        ax,
        texts,
        first_error,
        0.194,
        0.430,
        "first error",
        size=6.5,
        color=ERROR,
        weight="bold",
        ha="center",
    )
    arrow_owner = Region("paired-directions", 0.040, 0.295, 0.232, 0.390)
    _double_arrow(ax, artists, arrow_owner, (0.096, 0.350), (0.202, 0.350))
    _owned_text(
        ax,
        texts,
        arrow_owner,
        0.149,
        0.317,
        "donor  <->  receiver",
        size=6.5,
        ha="center",
        va="center",
    )

    endpoint = Region("endpoint-question", 0.032, 0.088, 0.226, 0.260)
    _owned_round_box(ax, artists, endpoint, panel, facecolor="white", edgecolor=PANEL_EDGE)
    _owned_text(
        ax,
        texts,
        Region("endpoint-copy", 0.044, 0.105, 0.214, 0.240),
        0.129,
        0.173,
        "Same verifier margin\nWhich internal site\nhas leverage?",
        size=6.7,
        ha="center",
        va="center",
        linespacing=1.10,
        weight="bold",
    )


def _panel_b(
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    artists: list[OwnedArtist],
    panel: Region,
) -> None:
    _owned_text(
        ax,
        texts,
        Region("b-title", 0.291, 0.850, 0.638, 0.930),
        0.293,
        0.905,
        "b  Patch native MoE interfaces",
        size=8.1,
        weight="bold",
    )

    cards = (
        (
            Region("residual-card", 0.286, 0.462, 0.369, 0.808),
            "Post-MoE\nresidual",
            RESIDUAL,
            "residual",
        ),
        (
            Region("moe-output-card", 0.374, 0.462, 0.457, 0.808),
            "MoE/MLP\noutput",
            MOE_OUTPUT,
            "output",
        ),
        (
            Region("same-router-card", 0.462, 0.462, 0.545, 0.808),
            "Same-layer\nrouter",
            SAME_ROUTER,
            "same",
        ),
        (
            Region("all-router-card", 0.550, 0.462, 0.633, 0.808),
            "All-router\nbundle",
            ALL_ROUTER,
            "all",
        ),
    )
    for region, label, color, mode in cards:
        _owned_round_box(ax, artists, region, panel, facecolor="white", edgecolor=PANEL_EDGE)
        _moe_icon(ax, artists, region, mode=mode, color=color)
        _owned_text(
            ax,
            texts,
            Region(f"{region.name}-label", region.x0 + 0.002, 0.700, region.x1 - 0.002, 0.795),
            (region.x0 + region.x1) / 2,
            0.758,
            label,
            size=6.6,
            color=color,
            weight="bold",
            ha="center",
            va="center",
            linespacing=1.05,
        )
        explanation = {
            "residual": "combined\nstate",
            "output": "output\nboundary",
            "same": "one native\ngate",
            "all": "all routed\ngates",
        }[mode]
        _owned_text(
            ax,
            texts,
            Region(f"{region.name}-copy", region.x0 + 0.002, 0.468, region.x1 - 0.002, 0.562),
            (region.x0 + region.x1) / 2,
            0.514,
            explanation,
            size=6.5,
            color=MUTED,
            ha="center",
            va="center",
            linespacing=1.05,
        )

    coverage = Region("protocol-coverage", 0.290, 0.085, 0.635, 0.398)
    _owned_round_box(ax, artists, coverage, panel, facecolor="white", edgecolor=PANEL_EDGE)
    _owned_text(
        ax,
        texts,
        Region("coverage-title", 0.304, 0.330, 0.619, 0.383),
        0.306,
        0.357,
        "Protocol topology is explicit",
        size=6.8,
        weight="bold",
    )
    _coverage_row(
        ax,
        texts,
        artists,
        0.294,
        0.255,
        MODERN_FILL,
        "Modern (4)",
        "residual + output + routers",
    )
    _coverage_row(
        ax,
        texts,
        artists,
        0.294,
        0.177,
        EXTENDED_FILL,
        "Extended",
        "GPT-OSS full | Qwen3 no bundle",
    )
    _coverage_row(
        ax,
        texts,
        artists,
        0.294,
        0.099,
        LEGACY_FILL,
        "Legacy",
        "OLMoE: residual + joint bundle",
    )


def _panel_c(
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    artists: list[OwnedArtist],
    panel: Region,
    spec: OverviewSpec,
) -> None:
    _owned_text(
        ax,
        texts,
        Region("c-title", 0.699, 0.850, 0.979, 0.930),
        0.701,
        0.905,
        "c  Evidence map and boundaries",
        size=7.2,
        weight="bold",
    )

    access = Region("access-card", 0.700, 0.685, 0.970, 0.838)
    _owned_round_box(ax, artists, access, panel, facecolor=ACCESS_FILL, edgecolor="#9DC5D8")
    _owned_text(
        ax,
        texts,
        Region("access-copy", 0.708, 0.700, 0.962, 0.830),
        0.711,
        0.765,
        f"Access | {spec.access_model_count} recapture models\n"
        "strong: residual/router input\n"
        "weaker: router logits\n"
        "and expert-aligned weights",
        size=6.5,
        weight="bold",
        va="center",
        linespacing=0.92,
    )

    prm = Region("prm-card", 0.700, 0.440, 0.833, 0.680)
    _owned_round_box(ax, artists, prm, panel, facecolor="white", edgecolor=PANEL_EDGE)
    _owned_text(
        ax,
        texts,
        Region("prm-copy", 0.705, 0.486, 0.828, 0.670),
        0.766,
        0.582,
        "PRMBench\n7 fixed cells\nresidual >\nrouter: 7/7\noutput: 6/6",
        size=6.5,
        color=INK,
        weight="bold",
        ha="center",
        va="center",
        linespacing=0.96,
    )
    _effect_glyph(ax, artists, prm)

    qwen = Region("qwen-card", 0.843, 0.440, 0.970, 0.680)
    _owned_round_box(ax, artists, qwen, panel, facecolor="#EAF7F2", edgecolor="#8FC8B5")
    _owned_text(
        ax,
        texts,
        Region("qwen-copy", 0.848, 0.486, 0.965, 0.675),
        0.906,
        0.577,
        "Qwen3.6\nrouter boundary\nall-router grows\nalpha=1 -> 2\nalpha=2 CI > 0",
        size=6.5,
        color=ALL_ROUTER,
        weight="bold",
        ha="center",
        va="center",
        linespacing=0.96,
    )
    _dose_glyph(ax, artists, qwen)

    process = Region("process-card", 0.700, 0.270, 0.970, 0.425)
    _owned_round_box(
        ax,
        artists,
        process,
        panel,
        facecolor="white",
        edgecolor="#7C8790",
        linestyle=(0, (3, 2)),
    )
    _owned_text(
        ax,
        texts,
        Region("process-copy", 0.706, 0.280, 0.964, 0.415),
        0.709,
        0.348,
        "Independent ProcessBench\n"
        f"{len(spec.processbench_models)} models x {spec.processbench_n_pairs} "
        "natural-error pairs\n"
        "not a seven-model gate",
        size=6.5,
        weight="bold",
        va="center",
        linespacing=1.08,
    )

    boundary = Region("boundary-card", 0.700, 0.070, 0.970, 0.235)
    _owned_round_box(ax, artists, boundary, panel, facecolor=BOUNDARY_FILL, edgecolor="#B7A7C9")
    _owned_text(
        ax,
        texts,
        Region("boundary-copy", 0.706, 0.078, 0.964, 0.225),
        0.835,
        0.153,
        "Supports topology-stratified\n"
        "access/effect-scale divergence\n"
        "No mediation or causal-use proof\n"
        "No universal/prospective claim",
        size=6.5,
        weight="bold",
        ha="center",
        va="center",
        linespacing=1.00,
    )


def _token(
    ax: mpl.axes.Axes,
    artists: list[OwnedArtist],
    texts: list[OwnedText],
    region: Region,
    label: str,
    color: str,
) -> None:
    patch_region = Region(
        f"{region.name}-patch",
        region.x0,
        region.y0 + 0.024,
        region.x1,
        region.y1,
    )
    _owned_round_box(ax, artists, patch_region, region, facecolor=color, edgecolor=color, radius=0.008)
    _owned_text(
        ax,
        texts,
        Region(
            f"{region.name}-text",
            region.x0 + 0.002,
            region.y0 + 0.035,
            region.x1 - 0.002,
            region.y1 - 0.008,
        ),
        (region.x0 + region.x1) / 2,
        (region.y0 + region.y1) / 2 + 0.010,
        label,
        size=6.5,
        color="white",
        weight="bold",
        ha="center",
        va="center",
    )


def _moe_icon(
    ax: mpl.axes.Axes,
    artists: list[OwnedArtist],
    region: Region,
    *,
    mode: str,
    color: str,
) -> None:
    xmid = (region.x0 + region.x1) / 2
    router_y = 0.638
    router = Region(f"{region.name}-router", xmid - 0.014, router_y - 0.020, xmid + 0.014, router_y + 0.020)
    router_patch = Rectangle(
        (router.x0, router.y0),
        router.width,
        router.height,
        transform=ax.transAxes,
        facecolor=color if mode in {"same", "all"} else "#D9DEE2",
        edgecolor=INK,
        linewidth=0.55,
        zorder=4,
    )
    ax.add_patch(router_patch)
    artists.append(OwnedArtist(router_patch, region))
    for offset in (-0.028, 0.0, 0.028):
        circle = Circle(
            (xmid + offset, 0.595),
            0.010,
            transform=ax.transAxes,
            facecolor=color if mode == "output" else "#FFFFFF",
            edgecolor=INK,
            linewidth=0.55,
            zorder=4,
        )
        ax.add_patch(circle)
        artists.append(OwnedArtist(circle, region))
    residual = FancyArrowPatch(
        (xmid - 0.034, 0.570),
        (xmid + 0.034, 0.570),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=6.0,
        linewidth=2.1 if mode == "residual" else 0.8,
        color=color if mode == "residual" else "#AEB7BD",
        zorder=4,
    )
    ax.add_patch(residual)
    artists.append(OwnedArtist(residual, region))
    if mode == "all":
        for y in (0.605, 0.635, 0.665):
            marker = Rectangle(
                (region.x0 + 0.012, y - 0.008),
                0.014,
                0.016,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0.4,
                zorder=4,
            )
            ax.add_patch(marker)
            artists.append(OwnedArtist(marker, region))


def _coverage_row(
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    artists: list[OwnedArtist],
    x: float,
    y: float,
    fill: str,
    label: str,
    detail: str,
) -> None:
    row = Region(f"coverage-{label}", x, y, 0.631, y + 0.067)
    _owned_round_box(ax, artists, row, row, facecolor=fill, edgecolor=fill, radius=0.006)
    _owned_text(
        ax,
        texts,
        Region(f"{row.name}-label", x + 0.009, y + 0.009, x + 0.098, y + 0.058),
        x + 0.011,
        y + 0.034,
        label,
        size=6.5,
        weight="bold",
        va="center",
    )
    _owned_text(
        ax,
        texts,
        Region(f"{row.name}-detail", x + 0.101, y + 0.009, 0.620, y + 0.058),
        x + 0.105,
        y + 0.034,
        detail,
        size=6.5,
        va="center",
    )


def _effect_glyph(
    ax: mpl.axes.Axes, artists: list[OwnedArtist], owner: Region
) -> None:
    y = owner.y0 + 0.020
    blue = Rectangle(
        (owner.x0 + 0.015, y),
        0.074,
        0.010,
        transform=ax.transAxes,
        facecolor=RESIDUAL,
        edgecolor=RESIDUAL,
        zorder=5,
    )
    orange = Rectangle(
        (owner.x0 + 0.015, y - 0.014),
        0.024,
        0.010,
        transform=ax.transAxes,
        facecolor=SAME_ROUTER,
        edgecolor=SAME_ROUTER,
        zorder=5,
    )
    for patch in (blue, orange):
        ax.add_patch(patch)
        artists.append(OwnedArtist(patch, owner))


def _dose_glyph(ax: mpl.axes.Axes, artists: list[OwnedArtist], owner: Region) -> None:
    line = FancyArrowPatch(
        (owner.x0 + 0.026, owner.y0 + 0.022),
        (owner.x1 - 0.022, owner.y0 + 0.050),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=5.5,
        linewidth=1.2,
        color=ALL_ROUTER,
        zorder=5,
    )
    ax.add_patch(line)
    artists.append(OwnedArtist(line, owner))
    for x, y in (
        (owner.x0 + 0.030, owner.y0 + 0.023),
        (owner.x1 - 0.027, owner.y0 + 0.050),
    ):
        dot = Circle(
            (x, y),
            0.006,
            transform=ax.transAxes,
            facecolor=ALL_ROUTER,
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
        ax.add_patch(dot)
        artists.append(OwnedArtist(dot, owner))


def _owned_text(
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    owner: Region,
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    color: str = INK,
    weight: str = "normal",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.15,
) -> Text:
    artist = ax.text(
        x,
        y,
        value,
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=10,
    )
    texts.append(OwnedText(artist=artist, owner=owner))
    return artist


def _owned_round_box(
    ax: mpl.axes.Axes,
    artists: list[OwnedArtist],
    region: Region,
    owner: Region,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.65,
    radius: float = 0.012,
    linestyle: str | tuple[int, tuple[int, ...]] = "solid",
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (region.x0, region.y0),
        region.width,
        region.height,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=1,
    )
    ax.add_patch(patch)
    artists.append(OwnedArtist(patch, owner, allow_touching_owner=region == owner))
    return patch


def _double_arrow(
    ax: mpl.axes.Axes,
    artists: list[OwnedArtist],
    owner: Region,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="<|-|>",
        mutation_scale=7,
        linewidth=0.9,
        color=MUTED,
        zorder=5,
    )
    ax.add_patch(arrow)
    artists.append(OwnedArtist(arrow, owner))


def _flow_connector(
    ax: mpl.axes.Axes,
    connectors: list[Connector],
    source: Region,
    target: Region,
) -> None:
    arrow = FancyArrowPatch(
        (source.x1 + 0.005, 0.505),
        (target.x0 - 0.005, 0.505),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=1.0,
        color="#83919A",
        zorder=8,
    )
    ax.add_patch(arrow)
    connectors.append(Connector(arrow, source, target))


def validate_overview_layout(
    fig: mpl.figure.Figure,
    *,
    ax: mpl.axes.Axes,
    texts: list[OwnedText],
    artists: list[OwnedArtist],
    connectors: list[Connector],
    expected_width: float,
    minimum_font_size: float,
    tolerance_px: float = 1.5,
) -> None:
    """Geometry guard for canvas, local ownership, overlap, and connectors."""

    width, _height = fig.get_size_inches()
    if not math.isclose(float(width), expected_width, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"overview width {width:.6f}in != {expected_width:.6f}in")

    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    canvas = fig.bbox
    errors: list[str] = []

    visible_texts = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    registered = {record.artist for record in texts}
    unowned = [artist.get_text()[:40] for artist in visible_texts if artist not in registered]
    if unowned:
        errors.append(f"visible text lacks owner: {unowned}")

    text_boxes: dict[Text, Bbox] = {}
    for text_record in texts:
        bbox = text_record.artist.get_window_extent(renderer=renderer)
        text_boxes[text_record.artist] = bbox
        owner_bbox = _display_bbox(ax, text_record.owner)
        if float(text_record.artist.get_fontsize()) < minimum_font_size:
            errors.append(
                f"text {text_record.artist.get_text()[:45]!r} is "
                f"{text_record.artist.get_fontsize():.2f}pt"
            )
        if not _bbox_contains(canvas, bbox, tolerance_px):
            errors.append(
                f"text escapes canvas: {text_record.artist.get_text()[:45]!r}"
            )
        if not _bbox_contains(owner_bbox, bbox, tolerance_px):
            errors.append(
                f"text escapes owner {text_record.owner.name}: "
                f"{text_record.artist.get_text()[:45]!r}"
            )

    for first, second in combinations(texts, 2):
        if _positive_bbox_overlap(text_boxes[first.artist], text_boxes[second.artist], tolerance_px):
            errors.append(
                "text overlap: "
                f"{first.artist.get_text()[:28]!r} with {second.artist.get_text()[:28]!r}"
            )

    for artist_record in artists:
        bbox = artist_record.artist.get_window_extent(renderer=renderer)
        owner_bbox = _display_bbox(ax, artist_record.owner)
        local_tolerance = 3.0 if artist_record.allow_touching_owner else tolerance_px
        if not _bbox_contains(canvas, bbox, local_tolerance):
            errors.append(f"artist escapes canvas for owner {artist_record.owner.name}")
        if not _bbox_contains(owner_bbox, bbox, local_tolerance):
            errors.append(f"artist escapes owner {artist_record.owner.name}")

    for connector in connectors:
        bbox = connector.artist.get_window_extent(renderer=renderer)
        if not _bbox_contains(canvas, bbox, tolerance_px):
            errors.append("flow connector escapes canvas")
        for text, text_bbox in text_boxes.items():
            if _positive_bbox_overlap(bbox, text_bbox, 0.5):
                errors.append(f"flow connector intersects text: {text.get_text()[:38]!r}")

    if errors:
        raise ValueError("Overview layout QA failed:\n- " + "\n- ".join(errors))


def _display_bbox(ax: mpl.axes.Axes, region: Region) -> Bbox:
    (x0, y0), (x1, y1) = ax.transAxes.transform(
        ((region.x0, region.y0), (region.x1, region.y1))
    )
    return Bbox.from_extents(x0, y0, x1, y1)


def _bbox_contains(outer: BboxBase, inner: BboxBase, tolerance: float) -> bool:
    return (
        inner.x0 >= outer.x0 - tolerance
        and inner.y0 >= outer.y0 - tolerance
        and inner.x1 <= outer.x1 + tolerance
        and inner.y1 <= outer.y1 + tolerance
    )


def _positive_bbox_overlap(first: BboxBase, second: BboxBase, tolerance: float) -> bool:
    overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
    overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
    return overlap_x > tolerance and overlap_y > tolerance
