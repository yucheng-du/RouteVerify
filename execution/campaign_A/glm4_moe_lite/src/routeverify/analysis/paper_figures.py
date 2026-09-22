"""Paper-figure data extraction and rendering for the RouteVerify main results.

The module consumes the unified CPU reanalysis artifact rather than re-estimating
effects.  It preserves canonical row IDs in every derived source-data row so
that plotted marks remain traceable to paired-bootstrap source-of-truth rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox

PAPER_WIDTH_IN = 6.75

MODEL_ORDER = (
    "gemma4",
    "nemotron3_nano",
    "nemotron35_lightning",
    "qwen36_moe",
    "gpt_oss_20b",
    "qwen3",
    "olmoe",
)

MODEL_LABELS = {
    "gemma4": "Gemma 4",
    "nemotron3_nano": "Nemotron-3 Nano",
    "nemotron35_lightning": "Nemotron-3.5 Lightning",
    "qwen36_moe": "Qwen3.6",
    "gpt_oss_20b": "GPT-OSS-20B",
    "qwen3": "Qwen3†",
    "olmoe": "OLMoE‡",
}

SITE_LABELS = {
    "post_moe_residual": "Residual state",
    "same_layer_router": "Same-layer router",
    "all_router": "All-router bundle",
}

SITE_COLORS = {
    "post_moe_residual": "#0072B2",
    "same_layer_router": "#D55E00",
    "all_router": "#009E73",
}

SITE_MARKERS = {
    "post_moe_residual": "o",
    "same_layer_router": "s",
    "all_router": "D",
}

SITE_Y_OFFSETS = {
    "post_moe_residual": 0.12,
    "same_layer_router": 0.0,
    "all_router": -0.12,
}

GRADE_BY_MODEL = {
    "gemma4": "modern",
    "nemotron3_nano": "modern",
    "nemotron35_lightning": "modern",
    "qwen36_moe": "modern",
    "gpt_oss_20b": "extended",
    "qwen3": "extended",
    "olmoe": "legacy-compatible",
}

MAIN_Y = {
    "gemma4": 7.0,
    "nemotron3_nano": 6.0,
    "nemotron35_lightning": 5.0,
    "qwen36_moe": 4.0,
    "gpt_oss_20b": 2.55,
    "qwen3": 1.55,
    "olmoe": 0.15,
}

DIRECTION_COLORS = {
    "clean_to_corrupt": "#0072B2",
    "corrupt_to_clean": "#D55E00",
    "donor_aligned_t": "#222222",
}

DIRECTION_MARKERS = {
    "clean_to_corrupt": "<",
    "corrupt_to_clean": ">",
    "donor_aligned_t": "D",
}


def configure_paper_style() -> None:
    """Apply a compact vector-safe style suitable for an ICLR figure*."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 0.0,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def extract_figure_source_rows(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Extract all plotted rows from the canonical unified-results payload."""

    _validate_payload(payload)
    return {
        "main": _extract_main_rows(payload),
        "directional_dose": _extract_directional_dose_rows(payload),
        "processbench_transfer": _extract_processbench_rows(payload),
        "router_strength": _extract_router_strength_rows(payload),
    }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    required = (
        "directional_effects",
        "donor_aligned_t",
        "native_manipulation",
        "headline_results",
    )
    missing = [key for key in required if not isinstance(payload.get(key), Mapping)]
    if missing:
        raise ValueError(f"unified-results payload lacks keyed tables: {missing}")


def _headline(payload: Mapping[str, Any], benchmark: str, model_id: str) -> Mapping[str, Any]:
    table = cast(Mapping[str, Any], payload["headline_results"])
    key = f"{benchmark}|{model_id}"
    row = table.get(key)
    if not isinstance(row, Mapping):
        raise KeyError(f"missing headline result: {key}")
    return row


def _referenced_row(
    payload: Mapping[str, Any],
    table_name: str,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    row_id = str(reference.get("row_id", ""))
    table = cast(Mapping[str, Any], payload[table_name])
    row = table.get(row_id)
    if not isinstance(row, Mapping):
        raise KeyError(f"missing {table_name} row: {row_id}")
    result = dict(row)
    if result.get("estimate") is None:
        raise ValueError(f"plotted row has no estimate: {row_id}")
    for field in ("estimate", "ci_low", "ci_high"):
        value = float(result[field])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field} in {row_id}")
    return result


def _source_row(
    row: Mapping[str, Any],
    *,
    plot_role: str,
    availability_status: str = "available",
) -> dict[str, Any]:
    fields = (
        "row_id",
        "model_id",
        "display_name",
        "benchmark",
        "protocol_grade",
        "site",
        "site_family",
        "topology",
        "anchor_layer",
        "common_anchor",
        "alpha",
        "alpha_status",
        "direction",
        "direction_semantic",
        "effect_kind",
        "estimate",
        "ci_low",
        "ci_high",
        "n_pairs",
        "pair_set_sha256",
        "indices_sha256",
        "source_prediction_path",
    )
    return {
        "plot_role": plot_role,
        "availability_status": availability_status,
        **{field: row.get(field) for field in fields},
    }


def _missing_source_row(
    *,
    benchmark: str,
    model_id: str,
    plot_role: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "plot_role": plot_role,
        "availability_status": reason,
        "row_id": None,
        "model_id": model_id,
        "display_name": MODEL_LABELS[model_id],
        "benchmark": benchmark,
        "protocol_grade": GRADE_BY_MODEL[model_id],
        "site": None,
        "site_family": plot_role,
        "topology": None,
        "anchor_layer": None,
        "common_anchor": None,
        "alpha": None,
        "alpha_status": "not_available",
        "direction": None,
        "direction_semantic": None,
        "effect_kind": None,
        "estimate": None,
        "ci_low": None,
        "ci_high": None,
        "n_pairs": None,
        "pair_set_sha256": None,
        "indices_sha256": None,
        "source_prediction_path": None,
    }


def _extract_main_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        headline = _headline(payload, "prmbench", model_id)
        residual = cast(Mapping[str, Any], headline["residual"])
        rows.append(
            _source_row(
                _referenced_row(
                    payload,
                    "donor_aligned_t",
                    cast(Mapping[str, Any], residual["donor_aligned_t"]),
                ),
                plot_role="post_moe_residual",
            )
        )

        same_layer = headline.get("same_layer_router")
        if isinstance(same_layer, Mapping):
            rows.append(
                _source_row(
                    _referenced_row(
                        payload,
                        "donor_aligned_t",
                        cast(Mapping[str, Any], same_layer["donor_aligned_t"]),
                    ),
                    plot_role="same_layer_router",
                )
            )
        else:
            rows.append(
                _missing_source_row(
                    benchmark="prmbench",
                    model_id=model_id,
                    plot_role="same_layer_router",
                    reason="legacy_topology_not_available",
                )
            )

        all_router = headline.get("all_router")
        all_router_cell = None
        if isinstance(all_router, Mapping):
            all_router_cell = all_router.get("1.0") or all_router.get("1")
        if isinstance(all_router_cell, Mapping):
            rows.append(
                _source_row(
                    _referenced_row(
                        payload,
                        "donor_aligned_t",
                        cast(Mapping[str, Any], all_router_cell["donor_aligned_t"]),
                    ),
                    plot_role="all_router",
                )
            )
        else:
            rows.append(
                _missing_source_row(
                    benchmark="prmbench",
                    model_id=model_id,
                    plot_role="all_router",
                    reason="historical_topology_not_available",
                )
            )
    return rows


def _extract_directional_dose_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in (
        "gemma4",
        "nemotron3_nano",
        "nemotron35_lightning",
        "qwen36_moe",
    ):
        headline = _headline(payload, "prmbench", model_id)
        all_router = cast(Mapping[str, Any], headline["all_router"])
        dose = 1.0
        cell_value = all_router.get(str(dose)) or all_router.get(str(int(dose)))
        if not isinstance(cell_value, Mapping):
            raise KeyError(f"missing all-router dose {dose} for {model_id}")
        directional = cast(Mapping[str, Any], cell_value["directional"])
        for direction in ("clean_to_corrupt", "corrupt_to_clean"):
            row = _referenced_row(
                payload,
                "directional_effects",
                cast(Mapping[str, Any], directional[direction]),
            )
            source_row = _source_row(row, plot_role=direction)
            if direction == "clean_to_corrupt":
                source_row |= {
                    "plot_estimate": -float(row["estimate"]),
                    "plot_ci_low": -float(row["ci_high"]),
                    "plot_ci_high": -float(row["ci_low"]),
                    "plot_sign_alignment": "negative_raw_clean_to_corrupt_effect",
                }
            else:
                source_row |= {
                    "plot_estimate": float(row["estimate"]),
                    "plot_ci_low": float(row["ci_low"]),
                    "plot_ci_high": float(row["ci_high"]),
                    "plot_sign_alignment": "raw_corrupt_to_clean_effect",
                }
            rows.append(source_row)
        t_row = _referenced_row(
            payload,
            "donor_aligned_t",
            cast(Mapping[str, Any], cell_value["donor_aligned_t"]),
        )
        t_source = _source_row(t_row, plot_role="donor_aligned_t")
        t_source |= {
            "plot_estimate": float(t_row["estimate"]),
            "plot_ci_low": float(t_row["ci_low"]),
            "plot_ci_high": float(t_row["ci_high"]),
            "plot_sign_alignment": "donor_aligned_average",
        }
        rows.append(t_source)
    return rows


def _extract_processbench_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in ("gpt_oss_20b", "gemma4", "nemotron3_nano"):
        headline = _headline(payload, "processbench", model_id)
        for plot_role, cell_name in (
            ("post_moe_residual", "residual"),
            ("same_layer_router", "same_layer_router"),
        ):
            cell = headline.get(cell_name)
            if not isinstance(cell, Mapping):
                raise KeyError(f"missing ProcessBench {cell_name} for {model_id}")
            directional = cast(Mapping[str, Any], cell["directional"])
            for direction in ("clean_to_corrupt", "corrupt_to_clean"):
                row = _referenced_row(
                    payload,
                    "directional_effects",
                    cast(Mapping[str, Any], directional[direction]),
                )
                rows.append(_source_row(row, plot_role=plot_role))
    return rows


def _extract_router_strength_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    table = cast(Mapping[str, Any], payload["native_manipulation"])
    rows: list[dict[str, Any]] = []
    for model_id in ("nemotron35_lightning", "qwen36_moe"):
        headline = _headline(payload, "prmbench", model_id)
        all_router = cast(Mapping[str, Any], headline["all_router"])
        for dose in (1.0, 2.0):
            matches = [
                dict(row)
                for row in table.values()
                if isinstance(row, Mapping)
                and row.get("benchmark") == "prmbench"
                and row.get("model_id") == model_id
                and row.get("site_family") == "all_router"
                and row.get("effect_kind") == "direct"
                and row.get("direction") in {"clean_to_corrupt", "corrupt_to_clean"}
                and row.get("alpha") is not None
                and math.isclose(float(row["alpha"]), dose, abs_tol=1.0e-12)
            ]
            if len(matches) != 2:
                raise ValueError(
                    f"expected two native direction rows for {model_id}/alpha={dose}, "
                    f"observed {len(matches)}"
                )
            weights = [int(row.get("n_native_metadata_present") or 0) for row in matches]
            if not all(weight > 0 for weight in weights):
                raise ValueError(f"native manipulation metadata unavailable for {model_id}")

            def weighted(
                field: str,
                *,
                matched_rows: tuple[Mapping[str, Any], ...] = tuple(matches),
                row_weights: tuple[int, ...] = tuple(weights),
                current_model: str = model_id,
            ) -> float:
                values = [row.get(field) for row in matched_rows]
                if any(value is None for value in values):
                    raise ValueError(
                        f"native field {field} unavailable for {current_model}"
                    )
                numeric_values = [float(cast(Any, value)) for value in values]
                return float(
                    np.average(
                        numeric_values,
                        weights=np.asarray(row_weights, dtype=float),
                    )
                )

            dispatch_rates = [
                row.get("all_router_dispatch_verified_rate") for row in matches
            ]
            headline_cell = all_router.get(str(dose)) or all_router.get(str(int(dose)))
            if not isinstance(headline_cell, Mapping):
                raise KeyError(f"missing all-router headline dose {dose} for {model_id}")
            t_row = _referenced_row(
                payload,
                "donor_aligned_t",
                cast(Mapping[str, Any], headline_cell["donor_aligned_t"]),
            )
            rows.append(
                {
                    "plot_role": "all_router_native_strength",
                    "availability_status": "available_native",
                    "row_id": ";".join(sorted(str(row["row_id"]) for row in matches)),
                    "model_id": model_id,
                    "display_name": matches[0].get("display_name"),
                    "benchmark": "prmbench",
                    "protocol_grade": matches[0].get("protocol_grade"),
                    "site": matches[0].get("site"),
                    "site_family": "all_router",
                    "topology": "all_router",
                    "alpha": dose,
                    "alpha_status": matches[0].get("alpha_status"),
                    "n_pair_direction_rows": sum(weights),
                    "topk_changed_rate_mean": weighted("topk_changed_rate_mean"),
                    "topk_overlap_mean": weighted("topk_overlap_mean"),
                    "router_kl_receiver_to_patched_mean": weighted(
                        "router_kl_receiver_to_patched_mean"
                    ),
                    "t_row_id": t_row["row_id"],
                    "t_estimate": t_row["estimate"],
                    "t_ci_low": t_row["ci_low"],
                    "t_ci_high": t_row["ci_high"],
                    "t_n_pairs": t_row["n_pairs"],
                    "t_pair_set_sha256": t_row.get("pair_set_sha256"),
                    "t_indices_sha256": t_row.get("indices_sha256"),
                    "all_router_dispatch_verified_rate": (
                        None
                        if any(value is None for value in dispatch_rates)
                        else float(
                            np.average(
                                [float(cast(Any, value)) for value in dispatch_rates],
                                weights=weights,
                            )
                        )
                    ),
                    "source_prediction_path": matches[0].get("source_prediction_path"),
                }
            )
    return rows


def make_main_effect_figure(rows: Sequence[Mapping[str, Any]]) -> mpl.figure.Figure:
    """Render the topology-aware seven-model effect-scale figure."""

    configure_paper_style()
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_id"]), []).append(row)
    if set(by_model) != set(MODEL_ORDER):
        raise ValueError("main figure rows do not cover exactly seven expected models")

    available = [row for row in rows if row.get("estimate") is not None]
    full_low = min(float(row["ci_low"]) for row in available)
    full_high = max(float(row["ci_high"]) for row in available)
    full_xlim = (min(-0.055, full_low - 0.025), max(1.08, full_high + 0.045))
    router_rows = [
        row
        for row in available
        if row["plot_role"] in {"same_layer_router", "all_router"}
    ]
    router_bound = max(
        0.025,
        max(abs(float(row["ci_low"])) for row in router_rows),
        max(abs(float(row["ci_high"])) for row in router_rows),
    )
    zoom_xlim = (-1.18 * router_bound, 1.18 * router_bound)

    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 3.75), dpi=180)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(3.65, 1.55),
        left=0.225,
        right=0.970,
        bottom=0.17,
        top=0.80,
        wspace=0.12,
    )
    ax_full = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1], sharey=ax_full)
    axes = (ax_full, ax_zoom)

    bands = (
        (3.52, 7.47, "#EAF3F8"),
        (1.08, 3.02, "#FFF4DE"),
        (-0.31, 0.62, "#F0F0F0"),
    )
    for ax in axes:
        for low, high, color in bands:
            ax.axhspan(low, high, color=color, zorder=0)
        ax.axvline(0.0, color="#666666", linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
        ax.grid(axis="x", color="#FFFFFF", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(-0.42, 7.60)

    for model_id in MODEL_ORDER:
        model_rows = [row for row in by_model[model_id] if row.get("estimate") is not None]
        ordered_rows = sorted(
            model_rows,
            key=lambda row: (
                "post_moe_residual",
                "same_layer_router",
                "all_router",
            ).index(str(row["plot_role"])),
        )
        row_by_role = {str(row["plot_role"]): row for row in ordered_rows}
        connector_style = (
            "--" if model_id == "qwen3" else ":" if model_id == "olmoe" else "-"
        )
        for left_role, right_role in (
            ("post_moe_residual", "same_layer_router"),
            ("same_layer_router", "all_router"),
        ):
            if left_role not in row_by_role or right_role not in row_by_role:
                continue
            ax_full.plot(
                [
                    float(row_by_role[left_role]["estimate"]),
                    float(row_by_role[right_role]["estimate"]),
                ],
                [
                    MAIN_Y[model_id] + SITE_Y_OFFSETS[left_role],
                    MAIN_Y[model_id] + SITE_Y_OFFSETS[right_role],
                ],
                color="#777777",
                linewidth=0.75,
                linestyle=connector_style,
                zorder=2,
            )
        for row in ordered_rows:
            role = str(row["plot_role"])
            estimate = float(row["estimate"])
            y_value = MAIN_Y[model_id] + SITE_Y_OFFSETS[role]
            error = np.asarray(
                [[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]]
            )
            for ax in (ax_full, ax_zoom):
                if ax is ax_zoom and role == "post_moe_residual":
                    continue
                ax.errorbar(
                    estimate,
                    y_value,
                    xerr=error,
                    fmt=SITE_MARKERS[role],
                    markersize=4.3,
                    markerfacecolor=(
                        "white" if model_id in {"qwen3", "olmoe"} else SITE_COLORS[role]
                    ),
                    markeredgecolor=SITE_COLORS[role],
                    markeredgewidth=0.8,
                    ecolor=SITE_COLORS[role],
                    elinewidth=0.9,
                    capsize=1.8,
                    capthick=0.75,
                    zorder=4,
                    clip_on=True,
                )

    y_ticks = [MAIN_Y[model_id] for model_id in MODEL_ORDER]
    ax_full.set_yticks(y_ticks, [MODEL_LABELS[model_id] for model_id in MODEL_ORDER])
    ax_zoom.tick_params(axis="y", labelleft=False)
    ax_full.set_xlim(*full_xlim)
    ax_zoom.set_xlim(*zoom_xlim)
    ax_full.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax_zoom.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax_full.axvspan(
        zoom_xlim[0],
        zoom_xlim[1],
        facecolor="#FFFFFF",
        edgecolor="#777777",
        linewidth=0.55,
        linestyle=(0, (2, 2)),
        alpha=0.42,
        zorder=1,
    )
    ax_full.set_title("a   Full effect scale", loc="left", fontweight="bold", pad=7)
    ax_zoom.set_title("b   Router-scale zoom", loc="left", fontweight="bold", pad=7)
    ax_full.set_xlabel(r"Donor-aligned transfer $T$ (mean-logprob margin)", labelpad=5)
    ax_zoom.set_xlabel(r"$T$ near zero", labelpad=5)
    ax_zoom.text(
        zoom_xlim[1] * 0.96,
        MAIN_Y["qwen3"] + SITE_Y_OFFSETS["all_router"],
        "all-router NA",
        ha="right",
        va="center",
        fontsize=6.5,
        color=SITE_COLORS["all_router"],
    )
    ax_zoom.text(
        zoom_xlim[1] * 0.96,
        MAIN_Y["olmoe"] + SITE_Y_OFFSETS["same_layer_router"],
        "same-layer NA",
        ha="right",
        va="center",
        fontsize=6.5,
        color=SITE_COLORS["same_layer_router"],
    )

    group_transform = ax_full.get_yaxis_transform()
    ax_full.text(
        -0.265,
        7.54,
        "MODERN · PAIRED FIRST-ERROR",
        transform=group_transform,
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color="#315D73",
        clip_on=False,
    )
    ax_full.text(
        -0.265,
        3.08,
        "EXTENDED",
        transform=group_transform,
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color="#8A5A00",
        clip_on=False,
    )
    ax_full.text(
        -0.265,
        0.66,
        "LEGACY-COMPATIBLE",
        transform=group_transform,
        ha="left",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        color="#555555",
        clip_on=False,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker=SITE_MARKERS[role],
            color="none",
            markerfacecolor=SITE_COLORS[role],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=5.2,
            label=SITE_LABELS[role],
        )
        for role in ("post_moe_residual", "same_layer_router", "all_router")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.94),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
    )
    fig.text(
        0.970,
        0.025,
        "† retrospective historical protocol   ‡ no same-layer router endpoint",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    validate_paper_figure(fig, expected_width=PAPER_WIDTH_IN, minimum_font_size=6.5)
    return fig


def make_secondary_figure(
    directional_rows: Sequence[Mapping[str, Any]],
    processbench_rows: Sequence[Mapping[str, Any]],
    router_strength_rows: Sequence[Mapping[str, Any]],
) -> mpl.figure.Figure:
    """Render direction, transfer, and manipulation-strength secondary panels."""

    configure_paper_style()
    fig = plt.figure(figsize=(PAPER_WIDTH_IN, 3.60), dpi=180)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.50, 2.18, 1.12),
        left=0.075,
        right=0.970,
        bottom=0.17,
        top=0.64,
        wspace=0.42,
    )
    ax_direction = fig.add_subplot(grid[0, 0])
    transfer_grid = grid[0, 1].subgridspec(1, 2, wspace=0.10)
    ax_transfer_forward = fig.add_subplot(transfer_grid[0, 0])
    ax_transfer_reverse = fig.add_subplot(
        transfer_grid[0, 1], sharex=ax_transfer_forward, sharey=ax_transfer_forward
    )
    ax_strength = fig.add_subplot(grid[0, 2])

    _plot_directional_dose(ax_direction, directional_rows)
    _plot_processbench_transfer(
        (ax_transfer_forward, ax_transfer_reverse), processbench_rows
    )
    _plot_router_strength(ax_strength, router_strength_rows)

    direction_box = ax_direction.get_position()
    transfer_left = ax_transfer_forward.get_position().x0
    transfer_right = ax_transfer_reverse.get_position().x1
    strength_box = ax_strength.get_position()
    fig.text(
        direction_box.x0,
        0.94,
        "a   Direction decomposition",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(
        transfer_left,
        0.94,
        "b   ProcessBench natural transfer",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(
        strength_box.x0,
        0.94,
        "c   Native routing",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    direction_handles = [
        Line2D(
            [0],
            [0],
            marker=DIRECTION_MARKERS[component],
            color="none",
            markerfacecolor=DIRECTION_COLORS[component],
            markeredgecolor="white" if component != "donor_aligned_t" else "#222222",
            markersize=4.8,
            label=label,
        )
        for component, label in (
            ("clean_to_corrupt", "clean"),
            ("corrupt_to_clean", "error"),
            ("donor_aligned_t", r"$T$"),
        )
    ]
    fig.legend(
        handles=direction_handles,
        loc="upper center",
        bbox_to_anchor=((direction_box.x0 + direction_box.x1) / 2.0, 0.86),
        ncol=3,
        frameon=False,
        handletextpad=0.20,
        columnspacing=0.50,
    )
    site_handles = [
        Line2D(
            [0],
            [0],
            marker=SITE_MARKERS[role],
            color="none",
            markerfacecolor=SITE_COLORS[role],
            markeredgecolor="white",
            markersize=4.8,
            label=label,
        )
        for role, label in (
            ("post_moe_residual", "residual"),
            ("same_layer_router", "router"),
        )
    ]
    fig.legend(
        handles=site_handles,
        loc="upper center",
        bbox_to_anchor=((transfer_left + transfer_right) / 2.0, 0.86),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.70,
    )
    strength_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=cast(Any, linestyle),
            markersize=4.5,
            label=label,
        )
        for _model, (label, color, marker, linestyle) in {
            "qwen36_moe": ("Qwen3.6", "#CC79A7", "o", "-"),
            "nemotron35_lightning": ("Lightning", "#009E73", "s", "--"),
        }.items()
    ]
    fig.legend(
        handles=strength_handles,
        loc="upper center",
        bbox_to_anchor=((strength_box.x0 + strength_box.x1) / 2.0, 0.86),
        ncol=1,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    validate_paper_figure(fig, expected_width=PAPER_WIDTH_IN, minimum_font_size=6.5)
    return fig


def _plot_directional_dose(
    ax: mpl.axes.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row_positions = {
        ("gemma4", 1.0): 3.0,
        ("nemotron3_nano", 1.0): 2.0,
        ("nemotron35_lightning", 1.0): 1.0,
        ("qwen36_moe", 1.0): 0.0,
    }
    labels = {
        ("gemma4", 1.0): "Gemma 4",
        ("nemotron3_nano", 1.0): "Nano",
        ("nemotron35_lightning", 1.0): "Lightning",
        ("qwen36_moe", 1.0): "Qwen3.6",
    }
    values = []
    for row in rows:
        component = str(row["plot_role"])
        dose = float(row["alpha"])
        key = (str(row["model_id"]), dose)
        if key not in row_positions:
            raise ValueError(f"unexpected directional-dose row: {key}")
        y_value = row_positions[key]
        estimate = float(row.get("plot_estimate", row["estimate"]))
        ci_low = float(row.get("plot_ci_low", row["ci_low"]))
        ci_high = float(row.get("plot_ci_high", row["ci_high"]))
        values.extend([ci_low, ci_high])
        error = np.asarray(
            [[estimate - ci_low], [ci_high - estimate]]
        )
        ax.errorbar(
            estimate,
            y_value,
            xerr=error,
            fmt=DIRECTION_MARKERS[component],
            markersize=4.1 if component != "donor_aligned_t" else 4.4,
            markerfacecolor=DIRECTION_COLORS[component],
            markeredgecolor="white" if component != "donor_aligned_t" else "#222222",
            markeredgewidth=0.45,
            ecolor=DIRECTION_COLORS[component],
            elinewidth=0.8,
            capsize=1.5,
            zorder=3 if component == "donor_aligned_t" else 2,
        )
    bound = max(0.05, max(abs(value) for value in values) * 1.10)
    ax.set_xlim(-bound, bound)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.set_ylim(-0.55, 3.55)
    ordered = sorted(row_positions, key=lambda key: row_positions[key])
    ax.set_yticks([row_positions[key] for key in ordered], [labels[key] for key in ordered])
    ax.axvline(0.0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Donor-aligned arm effect", labelpad=4)


def _plot_processbench_transfer(
    axes: tuple[mpl.axes.Axes, mpl.axes.Axes],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    y_by_model = {"gemma4": 2.0, "nemotron3_nano": 1.0, "gpt_oss_20b": 0.0}
    labels = {
        "gemma4": "Gemma 4",
        "nemotron3_nano": "Nano",
        "gpt_oss_20b": "GPT-OSS",
    }
    offsets = {"post_moe_residual": 0.10, "same_layer_router": -0.10}
    bounds = [float(row[field]) for row in rows for field in ("ci_low", "ci_high")]
    low = min(-0.035, min(bounds) - 0.015)
    high = max(0.18, max(bounds) + 0.02)
    for ax, direction in zip(
        axes, ("clean_to_corrupt", "corrupt_to_clean"), strict=True
    ):
        for row in rows:
            if row.get("direction") != direction:
                continue
            model_id = str(row["model_id"])
            role = str(row["plot_role"])
            estimate = float(row["estimate"])
            error = np.asarray(
                [[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]]
            )
            ax.errorbar(
                estimate,
                y_by_model[model_id] + offsets[role],
                xerr=error,
                fmt=SITE_MARKERS[role],
                markersize=4.0,
                markerfacecolor=SITE_COLORS[role],
                markeredgecolor="white",
                markeredgewidth=0.5,
                ecolor=SITE_COLORS[role],
                elinewidth=0.8,
                capsize=1.5,
                zorder=3,
            )
        ax.set_xlim(low, high)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.set_ylim(-0.5, 2.5)
        ax.axvline(0.0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.65)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Effect", labelpad=4)
    ordered = sorted(y_by_model, key=lambda key: y_by_model[key])
    axes[0].set_yticks(
        [y_by_model[key] for key in ordered], [labels[key] for key in ordered]
    )
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].set_title("clean donor →\nfirst-error receiver", pad=4, fontsize=6.8)
    axes[1].set_title(
        "first-error donor →\nmatched-clean receiver", pad=4, fontsize=6.8
    )


def _plot_router_strength(
    ax: mpl.axes.Axes,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    styles = {
        "qwen36_moe": ("Qwen3.6", "#CC79A7", "o", "-"),
        "nemotron35_lightning": ("Lightning", "#009E73", "s", "--"),
    }
    for model_id, (label, color, marker, linestyle) in styles.items():
        selected = sorted(
            [row for row in rows if row["model_id"] == model_id],
            key=lambda row: float(row["alpha"]),
        )
        if len(selected) != 2:
            raise ValueError(f"router-strength dose coverage incomplete for {model_id}")
        x_values = [float(row["topk_changed_rate_mean"]) for row in selected]
        y_values = [float(row["t_estimate"]) for row in selected]
        ax.plot(
            x_values,
            y_values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            zorder=2,
        )
        for row, x_value, y_value in zip(selected, x_values, y_values, strict=True):
            dose = float(row["alpha"])
            y_error = np.asarray(
                [
                    [y_value - float(row["t_ci_low"])],
                    [float(row["t_ci_high"]) - y_value],
                ]
            )
            ax.errorbar(
                x_value,
                y_value,
                yerr=y_error,
                fmt=marker,
                markersize=4.8,
                markerfacecolor=color if dose == 1.0 else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.6,
                zorder=3,
            )
            ax.annotate(
                rf"$\alpha={int(dose)}$",
                (x_value, y_value),
                xytext={
                    ("qwen36_moe", 1.0): (-8, 9),
                    ("qwen36_moe", 2.0): (4, 4),
                    ("nemotron35_lightning", 1.0): (4, -10),
                    ("nemotron35_lightning", 2.0): (4, -11),
                }[(model_id, dose)],
                textcoords="offset points",
                fontsize=6.5,
                color=color,
            )
    all_x = [float(row["topk_changed_rate_mean"]) for row in rows]
    all_y_bounds = [
        float(row[field]) for row in rows for field in ("t_ci_low", "t_ci_high")
    ]
    ax.set_xlim(min(all_x) - 0.035, min(1.01, max(all_x) + 0.035))
    ax.set_ylim(min(-0.012, min(all_y_bounds) - 0.004), max(0.042, max(all_y_bounds) + 0.004))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.grid(color="#E6E6E6", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_xlabel("Top-k assignments changed", labelpad=4)
    ax.set_ylabel(r"All-router $T$", labelpad=3)


def validate_paper_figure(
    fig: mpl.figure.Figure,
    *,
    expected_width: float,
    minimum_font_size: float,
    canvas_tolerance_px: float = 1.5,
) -> None:
    """Adapt the paper-layout guard to chart figures with tick/axes checks."""

    width, _height = fig.get_size_inches()
    if not math.isclose(float(width), expected_width, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"figure width {width:.6f}in != {expected_width:.6f}in")
    fig.canvas.draw()
    renderer = cast(Any, fig.canvas).get_renderer()
    canvas = fig.bbox
    errors: list[str] = []
    texts = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    for artist in texts:
        bbox = artist.get_window_extent(renderer=renderer)
        font_size = float(artist.get_fontsize())
        if font_size < minimum_font_size:
            errors.append(
                f"text {artist.get_text()[:50]!r} is {font_size:.2f}pt"
            )
        if (
            bbox.x0 < canvas.x0 - canvas_tolerance_px
            or bbox.y0 < canvas.y0 - canvas_tolerance_px
            or bbox.x1 > canvas.x1 + canvas_tolerance_px
            or bbox.y1 > canvas.y1 + canvas_tolerance_px
        ):
            errors.append(f"text escapes canvas: {artist.get_text()[:50]!r}")

    axes = [ax for ax in fig.axes if ax.get_visible()]
    for first_axis, second_axis in combinations(axes, 2):
        if _positive_bbox_overlap(
            first_axis.get_position(), second_axis.get_position(), 1.0e-6
        ):
            errors.append("axes plotting regions overlap")
    for ax in axes:
        for labels, label_kind in ((ax.get_xticklabels(), "x"), (ax.get_yticklabels(), "y")):
            visible = [label for label in labels if label.get_visible() and label.get_text()]
            for first_label, second_label in combinations(visible, 2):
                first_bbox = first_label.get_window_extent(renderer=renderer)
                second_bbox = second_label.get_window_extent(renderer=renderer)
                if _positive_bbox_overlap(first_bbox, second_bbox, canvas_tolerance_px):
                    errors.append(f"overlapping {label_kind}-tick labels")
                    break
        free_text = [artist for artist in ax.texts if artist.get_visible() and artist.get_text()]
        for first_text, second_text in combinations(free_text, 2):
            first_bbox = first_text.get_window_extent(renderer=renderer)
            second_bbox = second_text.get_window_extent(renderer=renderer)
            if _positive_bbox_overlap(first_bbox, second_bbox, canvas_tolerance_px):
                errors.append(
                    "same-axes annotation overlap: "
                    f"{first_text.get_text()[:30]!r} with {second_text.get_text()[:30]!r}"
                )
    for first_text, second_text in combinations(texts, 2):
        if first_text.axes is second_text.axes and first_text.axes is not None:
            continue
        first_bbox = first_text.get_window_extent(renderer=renderer)
        second_bbox = second_text.get_window_extent(renderer=renderer)
        if _positive_bbox_overlap(first_bbox, second_bbox, canvas_tolerance_px):
            errors.append(
                "cross-owner text overlap: "
                f"{first_text.get_text()[:30]!r} with {second_text.get_text()[:30]!r}"
            )

    if errors:
        raise ValueError("Paper figure layout QA failed:\n- " + "\n- ".join(errors))


def _positive_bbox_overlap(first: Bbox, second: Bbox, tolerance: float) -> bool:
    overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
    overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
    return overlap_x > tolerance and overlap_y > tolerance
