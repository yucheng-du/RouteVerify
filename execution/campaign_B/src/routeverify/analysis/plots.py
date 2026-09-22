"""Lightweight SVG plots for diagnostic reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LAYER_RE = re.compile(r"layer_(\d+)(?:\.([A-Za-z0-9_]+))?$")
_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563")


def build_diagnostic_plots(analysis: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write available diagnostic plots and return plot-name to path mapping."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}

    probe = analysis.get("probe_metrics")
    if isinstance(probe, Mapping):
        series = _probe_series(probe)
        if series:
            path = output / "probe_auroc_by_layer.svg"
            _write_text(path, line_plot_svg("Probe AUROC by layer", series, y_min=0.0, y_max=1.0))
            plots["probe_auroc_by_layer"] = str(path)

    routing = analysis.get("routing_metrics")
    if isinstance(routing, Mapping):
        series = _routing_series(routing)
        if series:
            path = output / "routing_by_layer.svg"
            _write_text(path, line_plot_svg("Routing statistics by layer", series, y_min=0.0))
            plots["routing_by_layer"] = str(path)

    visibility = analysis.get("router_visibility_metrics")
    if isinstance(visibility, Mapping):
        series = _router_visibility_series(visibility)
        if series:
            path = output / "router_visibility_by_layer.svg"
            _write_text(path, line_plot_svg("Router visibility by layer", series, y_min=0.0))
            plots["router_visibility_by_layer"] = str(path)

    return plots


def line_plot_svg(
    title: str,
    series: Mapping[str, Mapping[int, float]],
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 720,
    height: int = 420,
) -> str:
    """Render a small multi-series line plot as standalone SVG."""

    clean_series = {
        name: {layer: value for layer, value in values.items() if value is not None}
        for name, values in series.items()
    }
    clean_series = {name: values for name, values in clean_series.items() if values}
    if not clean_series:
        raise ValueError("series must contain at least one point")

    layers = sorted({layer for values in clean_series.values() for layer in values})
    values = [value for series_values in clean_series.values() for value in series_values.values()]
    xmin, xmax = min(layers), max(layers)
    ymin = min(values) if y_min is None else y_min
    ymax = max(values) if y_max is None else y_max
    if ymin == ymax:
        ymax = ymin + 1.0

    margin_left, margin_right, margin_top, margin_bottom = 58, 160, 42, 54
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(layer: int) -> float:
        if xmin == xmax:
            return margin_left + plot_width / 2
        return margin_left + (layer - xmin) / (xmax - xmin) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (ymax - value) / (ymax - ymin) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="24" font-family="sans-serif" font-size="18" font-weight="600">{_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        _tick_label(margin_left - 8, y_pos(ymin), _format_tick(ymin), anchor="end"),
        _tick_label(margin_left - 8, y_pos(ymax), _format_tick(ymax), anchor="end"),
        _tick_label(x_pos(xmin), margin_top + plot_height + 24, str(xmin), anchor="middle"),
        _tick_label(x_pos(xmax), margin_top + plot_height + 24, str(xmax), anchor="middle"),
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 12}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#374151">layer</text>',
    ]
    for index, (name, values_by_layer) in enumerate(sorted(clean_series.items())):
        color = _COLORS[index % len(_COLORS)]
        points = " ".join(
            f"{x_pos(layer):.1f},{y_pos(value):.1f}"
            for layer, value in sorted(values_by_layer.items())
        )
        if len(values_by_layer) == 1:
            layer, value = next(iter(values_by_layer.items()))
            lines.append(
                f'<circle cx="{x_pos(layer):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{color}"/>'
            )
        else:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'
            )
        legend_y = margin_top + 22 + index * 22
        legend_x = margin_left + plot_width + 24
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{_escape(name)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _probe_series(metrics: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping):
        return output
    for key, value in sites.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        parsed = _parse_layer_site(key)
        if parsed is None or value.get("auroc") is None:
            continue
        layer, kind = parsed
        output.setdefault(kind or "unknown", {})[layer] = float(value["auroc"])
    return output


def _routing_series(metrics: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {
        "topk_jaccard": {},
        "fraction_changed": {},
        "gate_weight_js": {},
    }
    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping):
        return {}
    for key, value in sites.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        parsed = _parse_layer_site(key)
        if parsed is None:
            continue
        layer, _kind = parsed
        _set_if_present(output["topk_jaccard"], layer, value.get("mean_topk_jaccard"))
        _set_if_present(output["fraction_changed"], layer, value.get("mean_fraction_changed"))
        _set_if_present(output["gate_weight_js"], layer, value.get("mean_gate_weight_js_divergence"))
    return {name: values for name, values in output.items() if values}


def _router_visibility_series(metrics: Mapping[str, Any]) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {
        "visibility_ratio": {},
        "normalized_jvp": {},
        "topk_change": {},
    }
    sites = metrics.get("sites", {})
    if not isinstance(sites, Mapping):
        return {}
    for key, value in sites.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            continue
        parsed = _parse_layer_site(key)
        if parsed is None:
            continue
        layer, _kind = parsed
        _set_if_present(output["visibility_ratio"], layer, value.get("visibility_ratio_vs_random"))
        _set_if_present(output["normalized_jvp"], layer, value.get("error_direction_mean_normalized_jvp"))
        _set_if_present(output["topk_change"], layer, value.get("error_direction_topk_change_fraction"))
    return {name: values for name, values in output.items() if values}


def _parse_layer_site(key: str) -> tuple[int, str | None] | None:
    match = _LAYER_RE.fullmatch(key)
    if match is None:
        return None
    layer = int(match.group(1))
    kind = match.group(2)
    return layer, kind


def _set_if_present(target: dict[int, float], layer: int, value: Any) -> None:
    if value is not None:
        target[layer] = float(value)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _tick_label(x: float, y: float, text: str, *, anchor: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="11" text-anchor="{anchor}" fill="#374151">{_escape(text)}</text>'


def _format_tick(value: float) -> str:
    return f"{value:.3g}"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
