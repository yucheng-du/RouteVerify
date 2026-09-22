"""Paper-scale deterministic figures for discovery-effect secondary analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

from routeverify.analysis.discovery_effects import (
    ALPHAS,
    DIRECTIONS,
    NONZERO_ALPHAS,
    ORIENTATION,
)

mpl.use("Agg")
mpl.rcParams["svg.hashsalt"] = "routeverify-olmoe-discovery-effects-v1"
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 8.0
mpl.rcParams["axes.titlesize"] = 9.0
mpl.rcParams["axes.labelsize"] = 8.0
mpl.rcParams["xtick.labelsize"] = 7.0
mpl.rcParams["ytick.labelsize"] = 7.0
mpl.rcParams["legend.fontsize"] = 7.0

from matplotlib import pyplot as plt  # noqa: E402

_DIRECTION_COLORS = {
    "clean_to_corrupt": "#2563eb",
    "corrupt_to_clean": "#dc2626",
}
_ALPHA_COLORS = {
    0.5: "#0f766e",
    1.0: "#7c3aed",
    2.0: "#ea580c",
}
_DIRECTION_LABELS = {
    "clean_to_corrupt": "Clean donor → corrupt receiver",
    "corrupt_to_clean": "Corrupt donor → clean receiver",
}


def build_discovery_effect_plots(
    analysis: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, dict[str, str]]:
    """Write deterministic SVG and PNG figures and return their paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = {
        "dose_response_mean": _dose_response_figure(analysis, pair_rows),
        "manipulation_dose_response": _manipulation_dose_figure(analysis),
        "manipulation_outcome_association": _association_figure(
            analysis,
            pair_rows,
        ),
        "error_type_heterogeneity": _heterogeneity_figure(analysis),
        "mean_total_sensitivity": _mean_total_figure(analysis, pair_rows),
    }
    paths: dict[str, dict[str, str]] = {}
    for name, figure in figures.items():
        paths[name] = _save_figure(figure, output / name)
        plt.close(figure)
    return paths


def _dose_response_figure(
    analysis: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Figure:
    confidence_label = _confidence_label(analysis)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        dpi=150,
        sharey=True,
        constrained_layout=True,
    )
    cell_effects = _mapping(_mapping(analysis, "cell_effects"), "mean_logprob")
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        by_pair = _pair_trajectories(
            pair_rows,
            direction=direction,
            value_key="effect_mean_logprob",
        )
        for values in by_pair.values():
            axis.plot(
                ALPHAS,
                values,
                color="#9ca3af",
                linewidth=0.35,
                alpha=0.09,
                zorder=1,
            )
        direction_payload = _mapping(cell_effects, direction)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for alpha in ALPHAS:
            entry = _mapping(direction_payload, f"{alpha:g}")
            means.append(float(entry["mean"]))
            interval = entry["pointwise_ci"]
            lows.append(float(interval[0]))
            highs.append(float(interval[1]))
        mean_array = np.asarray(means)
        axis.errorbar(
            ALPHAS,
            mean_array,
            yerr=np.vstack((mean_array - lows, np.asarray(highs) - mean_array)),
            color=_DIRECTION_COLORS[direction],
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            capsize=2.8,
            zorder=3,
            label=f"Pair mean ± {confidence_label}",
        )
        axis.axhline(0.0, color="#111827", linewidth=0.8, linestyle="--", zorder=0)
        expected = "expected < 0" if direction == "clean_to_corrupt" else "expected > 0"
        axis.set_title(f"{_DIRECTION_LABELS[direction]}\n({expected})")
        axis.set_xlabel("Interpolation alpha")
        axis.set_xticks(ALPHAS)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.6)
    axes[0].set_ylabel("Patched - unpatched margin (nats / target token)")
    axes[0].legend(loc="upper left", frameon=False)
    figure.suptitle(
        "OLMoE all-router joint intervention: discovery dose response",
        fontsize=10.0,
        fontweight="bold",
    )
    _validate_layout(figure)
    return figure


def _manipulation_dose_figure(analysis: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
        dpi=150,
        constrained_layout=True,
    )
    manipulation = _mapping(analysis, "manipulation")
    cells = _mapping(manipulation, "cell_summaries")
    specifications = (
        ("topk_changed_rate", "Mean top-k changed rate"),
        ("router_kl_receiver_to_patched", "Mean router KL"),
    )
    for axis, (measure, label) in zip(axes, specifications, strict=True):
        measure_payload = _mapping(cells, measure)
        displayed_highs: list[float] = []
        for direction in DIRECTIONS:
            direction_payload = _mapping(measure_payload, direction)
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for alpha in ALPHAS:
                entry = _mapping(direction_payload, f"{alpha:g}")
                means.append(float(entry["mean"]))
                interval = entry["pointwise_ci"]
                lows.append(float(interval[0]))
                highs.append(float(interval[1]))
            displayed_highs.extend(highs)
            mean_array = np.asarray(means)
            axis.errorbar(
                ALPHAS,
                mean_array,
                yerr=np.vstack((mean_array - lows, np.asarray(highs) - mean_array)),
                color=_DIRECTION_COLORS[direction],
                marker="o",
                linewidth=1.6,
                markersize=4.0,
                capsize=2.6,
                label=_DIRECTION_LABELS[direction],
            )
        axis.set_xlabel("Interpolation alpha")
        axis.set_ylabel(label)
        axis.set_xticks(ALPHAS)
        axis.grid(color="#e5e7eb", linewidth=0.6)
        if measure == "router_kl_receiver_to_patched":
            _set_padded_nonnegative_scale(
                axis,
                data_max=max(displayed_highs),
            )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axes[1].legend(loc="upper left", frameon=False)
    figure.suptitle(
        "Routing manipulation increases with dose",
        fontsize=10.0,
        fontweight="bold",
    )
    _validate_layout(figure)
    return figure


def _association_figure(
    analysis: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Figure:
    confidence_label = _confidence_label(analysis)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.6),
        dpi=150,
        constrained_layout=False,
        sharex="col",
        sharey="row",
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.105,
        top=0.86,
        wspace=0.18,
        hspace=0.10,
    )
    associations = _mapping(
        _mapping(analysis, "manipulation"),
        "outcome_associations",
    )
    primary = _mapping(associations, "mean_logprob")
    measures = (
        ("topk_changed_rate", "Top-k change z-score"),
        ("router_kl_receiver_to_patched", "Router KL rank z-score"),
    )
    x_bounds = np.zeros(len(measures), dtype=float)
    y_bounds = np.zeros(len(DIRECTIONS), dtype=float)
    for row_index, direction in enumerate(DIRECTIONS):
        direction_payload = _mapping(primary, direction)
        for column_index, (measure, x_label) in enumerate(measures):
            axis = axes[row_index, column_index]
            x, y, alphas = _association_points(
                pair_rows,
                direction=direction,
                measure=measure,
            )
            x_bounds[column_index] = max(
                x_bounds[column_index],
                float(np.max(np.abs(x))),
            )
            y_bounds[row_index] = max(
                y_bounds[row_index],
                float(np.max(np.abs(y))),
            )
            for alpha in NONZERO_ALPHAS:
                mask = alphas == alpha
                axis.scatter(
                    x[mask],
                    y[mask],
                    s=8,
                    alpha=0.28,
                    color=_ALPHA_COLORS[alpha],
                    edgecolors="none",
                    label=f"alpha={alpha:g}",
                )
            slope = float(np.sum(x * y) / np.sum(x**2))
            line_x = np.asarray([float(np.min(x)), float(np.max(x))])
            axis.plot(
                line_x,
                slope * line_x,
                color="#111827",
                linewidth=1.3,
            )
            entry = _mapping(direction_payload, measure)
            interval = entry["ci"]
            axis.text(
                0.03,
                0.96,
                (
                    f"β={float(entry['estimate']):+.3f}\n"
                    f"{confidence_label} "
                    f"[{float(interval[0]):+.3f}, {float(interval[1]):+.3f}]"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "#d1d5db", "pad": 2.0},
            )
            axis.axhline(0.0, color="#9ca3af", linewidth=0.6)
            axis.axvline(0.0, color="#9ca3af", linewidth=0.6)
            axis.grid(color="#f3f4f6", linewidth=0.5)
            if row_index == 1:
                axis.set_xlabel(x_label)
            if column_index == 0:
                axis.set_ylabel(f"{_DIRECTION_LABELS[direction]}\nOriented effect z-score")
    for column_index, data_bound in enumerate(x_bounds):
        _set_padded_symmetric_scale(
            axes[-1, column_index],
            dimension="x",
            data_bound=float(data_bound),
        )
    for row_index, data_bound in enumerate(y_bounds):
        _set_padded_symmetric_scale(
            axes[row_index, 0],
            dimension="y",
            data_bound=float(data_bound),
        )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles[:3],
        labels[:3],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        ncols=3,
    )
    figure.suptitle(
        "Within-dose manipulation-outcome association (diagnostic, not mediation)",
        y=0.99,
        fontsize=10.0,
        fontweight="bold",
    )
    _validate_layout(figure)
    return figure


def _heterogeneity_figure(analysis: Mapping[str, Any]) -> Figure:
    figure, axis = plt.subplots(
        figsize=(7.2, 4.1),
        dpi=150,
        constrained_layout=True,
    )
    heterogeneity = _mapping(analysis, "error_type_heterogeneity")
    groups = _mapping(heterogeneity, "groups")
    ordered = sorted(
        groups.items(),
        key=lambda item: (-int(_mapping(groups, str(item[0]))["n"]), str(item[0])),
    )
    y_positions = np.arange(len(ordered))[::-1]
    for y, (name, _raw_entry) in zip(y_positions, ordered, strict=True):
        entry = _mapping(groups, str(name))
        mean = float(entry["mean"])
        interval = entry["pointwise_ci"]
        low, high = float(interval[0]), float(interval[1])
        small = entry["inference_status"] == "descriptive_only_small_n"
        color = "#9ca3af" if small else "#2563eb"
        axis.errorbar(
            mean,
            y,
            xerr=np.asarray([[mean - low], [high - mean]]),
            marker="o",
            markersize=4.5,
            capsize=2.8,
            color=color,
            linewidth=1.2,
        )
        axis.text(
            high,
            y + 0.20,
            f" n={int(entry['n'])}" + (" · descriptive" if small else ""),
            fontsize=7.0,
            color=color,
            ha="left",
            va="center",
        )
    axis.set_yticks(y_positions)
    axis.set_yticklabels([str(name) for name, _entry in ordered])
    axis.axvline(0.0, color="#111827", linewidth=0.8, linestyle="--")
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.6)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="both"))
    axis.set_xlabel("Symmetric expected-direction effect at alpha=1 (nats / target token)")
    omnibus = _mapping(heterogeneity, "omnibus_permutation")
    axis.set_title(
        "Discovery error-type heterogeneity\n"
        f"exploratory omnibus permutation p={float(omnibus['p_value']):.3g}"
    )
    _validate_layout(figure)
    return figure


def _mean_total_figure(
    analysis: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> Figure:
    figure, axis = plt.subplots(
        figsize=(5.4, 4.7),
        dpi=150,
        constrained_layout=True,
    )
    for direction in DIRECTIONS:
        for alpha in NONZERO_ALPHAS:
            selected = [
                row
                for row in pair_rows
                if row["direction"] == direction and float(row["alpha"]) == alpha
            ]
            mean_values = np.asarray([float(row["effect_mean_logprob"]) for row in selected])
            total_values = np.asarray([float(row["effect_total_logprob"]) for row in selected])
            x = _standardize(mean_values)
            y = _standardize(total_values)
            axis.scatter(
                x,
                y,
                s=8,
                alpha=0.25,
                color=_DIRECTION_COLORS[direction],
                marker={0.5: "o", 1.0: "s", 2.0: "^"}[alpha],
                edgecolors="none",
                label=f"{_DIRECTION_LABELS[direction]}, alpha={alpha:g}",
            )
    limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    bound = max(abs(limits[0]), abs(limits[1]), abs(y_limits[0]), abs(y_limits[1]))
    _set_padded_symmetric_scale(axis, dimension="x", data_bound=bound)
    _set_padded_symmetric_scale(axis, dimension="y", data_bound=bound)
    padded_bound = float(axis.get_xlim()[1])
    axis.plot(
        [-padded_bound, padded_bound],
        [-padded_bound, padded_bound],
        color="#111827",
        linewidth=1.0,
    )
    axis.axhline(0.0, color="#d1d5db", linewidth=0.6)
    axis.axvline(0.0, color="#d1d5db", linewidth=0.6)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean-logprob pair effect (within-cell z-score)")
    axis.set_ylabel("Total-logprob pair effect (within-cell z-score)")
    sensitivity = _mapping(analysis, "mean_total_sensitivity")
    axis.text(
        0.03,
        0.97,
        (
            f"Spearman rho={float(sensitivity['standardized_pair_effect_spearman']):.3f}\n"
            f"Sign agreement={float(sensitivity['pair_cell_sign_agreement']):.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "pad": 2.0},
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=2,
        frameon=False,
    )
    axis.set_title("Mean versus total log-probability sensitivity")
    axis.grid(color="#f3f4f6", linewidth=0.5)
    _validate_layout(figure)
    return figure


def _pair_trajectories(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    value_key: str,
) -> dict[str, list[float]]:
    output: dict[str, dict[float, float]] = {}
    for row in rows:
        if row["direction"] != direction:
            continue
        output.setdefault(str(row["pair_id"]), {})[float(row["alpha"])] = float(row[value_key])
    normalized: dict[str, list[float]] = {}
    for pair_id, values in output.items():
        if set(values) != set(ALPHAS):
            raise ValueError(f"incomplete trajectory for {pair_id}")
        normalized[pair_id] = [values[alpha] for alpha in ALPHAS]
    return normalized


def _association_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    measure: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    alpha_values: list[np.ndarray] = []
    for alpha in NONZERO_ALPHAS:
        selected = [
            row for row in rows if row["direction"] == direction and float(row["alpha"]) == alpha
        ]
        x = np.asarray([float(row[measure]) for row in selected])
        if measure == "router_kl_receiver_to_patched":
            x = _midranks(x)
        y = np.asarray(
            [ORIENTATION[direction] * float(row["effect_mean_logprob"]) for row in selected]
        )
        x_values.append(_standardize(x))
        y_values.append(_standardize(y))
        alpha_values.append(np.full(x.size, alpha))
    return (
        np.concatenate(x_values),
        np.concatenate(y_values),
        np.concatenate(alpha_values),
    )


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _standardize(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0.0:
        raise ValueError("cannot standardize a zero-variance vector")
    return (values - float(np.mean(values))) / standard_deviation


def _confidence_label(analysis: Mapping[str, Any]) -> str:
    bootstrap = _mapping(analysis, "bootstrap")
    ci_alpha = float(bootstrap["ci_alpha"])
    if not 0.0 < ci_alpha < 1.0:
        raise ValueError("analysis bootstrap ci_alpha must be between 0 and 1")
    return f"{100.0 * (1.0 - ci_alpha):g}% CI"


def _set_padded_symmetric_scale(
    axis: Any,
    *,
    dimension: str,
    data_bound: float,
) -> None:
    """Keep shared-axis outer tick labels safely inside the fixed figure canvas."""

    tick_bound = max(1.0, float(np.ceil(data_bound * 2.0) / 2.0))
    limit = tick_bound + max(0.25, 0.12 * tick_bound)
    ticks = np.linspace(-tick_bound, tick_bound, 5)
    if dimension == "x":
        axis.set_xlim(-limit, limit)
        axis.set_xticks(ticks)
        return
    if dimension == "y":
        axis.set_ylim(-limit, limit)
        axis.set_yticks(ticks)
        return
    raise ValueError(f"unsupported dimension: {dimension}")


def _set_padded_nonnegative_scale(axis: Any, *, data_max: float) -> None:
    """Place nonnegative ticks inside padded limits instead of outside the canvas."""

    if not np.isfinite(data_max) or data_max <= 0.0:
        raise ValueError("nonnegative scale requires a finite positive maximum")
    locator = MaxNLocator(nbins=5, steps=[1.0, 2.0, 2.5, 5.0, 10.0])
    tick_values = np.asarray(locator.tick_values(0.0, data_max), dtype=float)
    tick_values = tick_values[tick_values >= 0.0]
    tick_max = float(np.max(tick_values))
    axis.set_ylim(-0.03 * tick_max, 1.08 * tick_max)
    axis.set_yticks(tick_values)


def _save_figure(figure: Figure, base: Path) -> dict[str, str]:
    _validate_layout(figure)
    paths: dict[str, str] = {}
    for extension, options in (
        (
            "svg",
            {
                "format": "svg",
                "metadata": {
                    "Date": None,
                    "Creator": "RouteVerify discovery CPU analysis",
                },
            },
        ),
        (
            "png",
            {
                "format": "png",
                "dpi": 240,
                "metadata": {"Software": "RouteVerify discovery CPU analysis"},
            },
        ),
    ):
        path = base.with_suffix(f".{extension}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        figure.savefig(
            temporary,
            facecolor="white",
            **options,
        )
        if extension == "svg":
            svg_text = temporary.read_text(encoding="utf-8")
            normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
            temporary.write_text(normalized_svg, encoding="utf-8")
        temporary.replace(path)
        paths[extension] = str(path)
    return paths


def _validate_layout(
    figure: Figure,
    *,
    minimum_font_size: float = 6.5,
    canvas_tolerance_px: float = 2.0,
) -> None:
    """Fail when text escapes the fixed canvas or paper-scale fonts are too small."""

    figure.canvas.draw()
    renderer = cast(Any, figure.canvas).get_renderer()
    canvas = figure.bbox
    errors: list[str] = []
    for artist in figure.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        font_size = float(artist.get_fontsize())
        if font_size < minimum_font_size:
            errors.append(f"text {artist.get_text()[:50]!r} is {font_size:.2f} pt")
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - canvas_tolerance_px
            or bounds.y0 < canvas.y0 - canvas_tolerance_px
            or bounds.x1 > canvas.x1 + canvas_tolerance_px
            or bounds.y1 > canvas.y1 + canvas_tolerance_px
        ):
            errors.append(f"text {artist.get_text()[:50]!r} escapes the figure canvas")
    figure_title = getattr(figure, "_suptitle", None)
    if figure_title is not None and figure_title.get_visible():
        title_bounds = figure_title.get_window_extent(renderer=renderer)
        for legend in figure.legends:
            if _bounds_overlap(
                title_bounds,
                legend.get_window_extent(renderer=renderer),
                tolerance_px=canvas_tolerance_px,
            ):
                errors.append("figure title overlaps a figure-level legend")
    x_labels = [
        axis.xaxis.label
        for axis in figure.axes
        if axis.xaxis.label.get_visible() and axis.xaxis.label.get_text().strip()
    ]
    for first_index, first_label in enumerate(x_labels):
        first_bounds = first_label.get_window_extent(renderer=renderer)
        for second_label in x_labels[first_index + 1 :]:
            if _bounds_overlap(
                first_bounds,
                second_label.get_window_extent(renderer=renderer),
                tolerance_px=canvas_tolerance_px,
            ):
                errors.append("x-axis labels overlap")
    positions = [axis.get_position() for axis in figure.axes]
    for first_index, first in enumerate(positions):
        for second in positions[first_index + 1 :]:
            overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                errors.append("axes rectangles overlap")
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"discovery figure layout QA failed:\n{formatted}")


def _bounds_overlap(first: Any, second: Any, *, tolerance_px: float) -> bool:
    overlap_x = min(float(first.x1), float(second.x1)) - max(
        float(first.x0),
        float(second.x0),
    )
    overlap_y = min(float(first.y1), float(second.y1)) - max(
        float(first.y0),
        float(second.y0),
    )
    return overlap_x > tolerance_px and overlap_y > tolerance_px


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value
