"""Build compact report summaries from run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from routeverify.instrumentation.storage import atomic_write_json


def build_summary(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(run_dir) / "report" / "summary.json"
    atomic_write_json(output, payload)
    return output


def summary_markdown(payload: dict[str, Any]) -> str:
    """Render a compact, human-readable run summary."""

    lines = ["# RouteVerify Run Summary", ""]
    lines.append(f"- run: `{payload.get('run', '')}`")
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        lines.extend(["", "## Metrics", ""])
        lines.extend(_metrics_markdown(metrics))

    patch_metrics = payload.get("patch_metrics")
    if isinstance(patch_metrics, dict):
        lines.extend(["", "## Patch Controls", ""])
        lines.extend(_patch_controls_markdown(patch_metrics))
        lines.extend(["", "## Top Patch Effects", ""])
        lines.extend(_top_patch_effects_markdown(patch_metrics))

    analysis = payload.get("analysis")
    if isinstance(analysis, dict):
        lines.extend(["", "## Analysis Artifacts", ""])
        for key in ("probe_metrics", "routing_metrics", "router_visibility_metrics"):
            if key in analysis:
                lines.append(f"- `{key}`: available")
    plots = payload.get("plots")
    if isinstance(plots, dict) and plots:
        lines.extend(["", "## Plots", ""])
        for name, path in sorted(plots.items()):
            lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines).rstrip() + "\n"


def _metrics_markdown(metrics: dict[str, Any]) -> list[str]:
    mode = metrics.get("mode")
    if mode == "step-binary":
        return _table(
            ["n", "accuracy", "AUROC", "positives", "negatives", "mean margin"],
            [
                [
                    metrics.get("n"),
                    _fmt(metrics.get("accuracy")),
                    _fmt(metrics.get("auroc")),
                    metrics.get("positives"),
                    metrics.get("negatives"),
                    _fmt(metrics.get("mean_margin_incorrect_minus_correct")),
                ]
            ],
        )
    if mode == "first-error":
        presence = metrics.get("error_presence", {})
        localization = metrics.get("first_error_localization", {})
        return _table(
            [
                "n",
                "accuracy",
                "presence F1",
                "localization F1",
                "mean index distance",
                "all-correct FP rate",
            ],
            [
                [
                    metrics.get("n"),
                    _fmt(metrics.get("accuracy")),
                    _fmt(presence.get("f1") if isinstance(presence, dict) else None),
                    _fmt(localization.get("f1") if isinstance(localization, dict) else None),
                    _fmt(metrics.get("mean_abs_error_index_distance")),
                    _fmt(metrics.get("all_correct_false_positive_rate")),
                ]
            ],
        )
    if "mean_margin_correct_minus_corrupt_mean" in metrics:
        return _table(
            ["n", "mean margin", "total margin", "mean preference", "total preference"],
            [
                [
                    metrics.get("n"),
                    _fmt(metrics.get("mean_margin_correct_minus_corrupt_mean")),
                    _fmt(metrics.get("mean_margin_correct_minus_corrupt_total")),
                    _fmt(metrics.get("preference_rate_correct_over_corrupt_mean")),
                    _fmt(metrics.get("preference_rate_correct_over_corrupt_total")),
                ]
            ],
        )
    if {"num_boundaries", "num_sites"}.issubset(metrics):
        headers = ["examples", "boundaries", "sites"]
        row = [metrics.get("n"), metrics.get("num_boundaries"), metrics.get("num_sites")]
        if "prompt_mode" in metrics:
            headers.append("prompt mode")
            row.append(metrics.get("prompt_mode"))
        if "boundary_strategy" in metrics:
            headers.append("boundary strategy")
            row.append(metrics.get("boundary_strategy"))
        return _table(headers, [row])
    if {"num_pairs", "num_rows"}.issubset(metrics):
        return _table(["pairs", "rows"], [[metrics.get("num_pairs"), metrics.get("num_rows")]])
    return ["No recognized high-level metrics found."]


def _patch_controls_markdown(metrics: dict[str, Any]) -> list[str]:
    self_deltas = metrics.get("max_abs_self_patch_delta_by_site", {})
    rerun_deltas = metrics.get("max_abs_unpatched_rerun_delta_by_receiver", {})
    rows: list[list[Any]] = []
    if isinstance(self_deltas, dict):
        for site, value in sorted(self_deltas.items()):
            rows.append([f"self patch {site}", _fmt(value)])
    if isinstance(rerun_deltas, dict):
        for receiver, value in sorted(rerun_deltas.items()):
            rows.append([f"unpatched rerun {receiver}", _fmt(value)])
    return _table(["control", "max abs delta"], rows) if rows else ["No patch controls found."]


def _top_patch_effects_markdown(metrics: dict[str, Any], *, max_rows: int = 10) -> list[str]:
    effect_key = "control_adjusted_ate_margin"
    effects = metrics.get(effect_key, {})
    if not isinstance(effects, dict) or not effects:
        effect_key = "ate_margin_vs_unpatched"
        effects = metrics.get(effect_key, {})
    if not isinstance(effects, dict) or not effects:
        return ["No patch effects found."]
    rows = [f"source: `{effect_key}`"]
    effect_rows = []
    for key, value in sorted(
        effects.items(),
        key=lambda item: abs(float(item[1].get("mean", 0.0))) if isinstance(item[1], dict) else 0.0,
        reverse=True,
    )[:max_rows]:
        if not isinstance(value, dict):
            continue
        effect_rows.append(
            [
                key,
                value.get("n"),
                _fmt(value.get("mean")),
                _fmt(value.get("ci_low")),
                _fmt(value.get("ci_high")),
                _fmt(value.get("standardized_effect")),
                _fmt(value.get("bh_q_family")),
            ]
        )
    rows.extend(
        _table(
            ["treatment/site", "n", "mean", "CI low", "CI high", "std effect", "q"],
            effect_rows,
        )
    )
    return rows


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    align = ["---"] + ["---:" for _ in headers[1:]]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
