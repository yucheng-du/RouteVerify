"""Unified CPU-only secondary analysis for frozen RouteVerify patch rows.

The implementation deliberately consumes prediction rows and their immutable
sidecars only.  It never opens a discovery/validation split manifest, loads a
model, imports torch, or mutates an experiment archive.  All intervals are
ordinary marginal percentile intervals over paired row clusters; the module
does not compute population p-values, max-T intervals, TOST, or SESOI tests.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from routeverify.analysis.bootstrap import (
    paired_cluster_bootstrap_indices,
    paired_cluster_bootstrap_mean_ci,
)
from routeverify.analysis.probes import binary_auroc
from routeverify.instrumentation.storage import atomic_write_json

DIRECTIONS = ("clean_to_corrupt", "corrupt_to_clean")
EXPECTED_RECEIVER = {"clean_to_corrupt": "corrupt", "corrupt_to_clean": "clean"}
EXPECTED_DONOR = {"clean_to_corrupt": "clean", "corrupt_to_clean": "corrupt"}
CONTROL_TREATMENTS = {
    "norm_random_delta": "norm_random",
    "random_matched_clean_to_corrupt": "random_matched",
    "random_matched_corrupt_to_clean": "random_matched",
    "pre_error_clean_to_corrupt": "pre_error",
    "pre_error_corrupt_to_clean": "pre_error",
}
ALLOWED_TREATMENTS = {
    "unpatched",
    "unpatched_rerun",
    "self_patch",
    *DIRECTIONS,
    *CONTROL_TREATMENTS,
}
SITE_PATTERN = re.compile(
    r"^layer_(?P<layer>\d+)\.(?P<family>router_logits|moe_output|post_moe_residual)$"
)
BUNDLE_PATTERN = re.compile(r"^bundle\.router_logits\[(?P<count>\d+)]$")
PRIMARY_METRIC = "mean_logprob_margin_incorrect_minus_correct"
PRIMARY_UNITS = "log-probability per candidate token"
IDENTITY_TOLERANCE = 1.0e-5

PairMap = dict[str, float]
CellKey = tuple[str, float | None]
DirectKey = tuple[str, float | None, str]
ControlKey = tuple[str, str, str]


@dataclass
class JsonlLoad:
    rows: list[dict[str, Any]]
    raw_row_count: int
    selected_row_count: int
    sha256: str
    size_bytes: int


@dataclass
class PatchRows:
    path: Path
    raw_row_count: int
    selected_row_count: int
    pair_ids: tuple[str, ...]
    baselines: dict[tuple[str, str], float]
    reruns: dict[tuple[str, str], float]
    self_patches: dict[tuple[str, str, str], float]
    direct_margins: dict[DirectKey, PairMap]
    controls_margins: dict[ControlKey, PairMap]
    manipulation_rows: dict[DirectKey, list[dict[str, Any]]]
    control_manipulation_rows: dict[ControlKey, list[dict[str, Any]]]
    identity: dict[str, Any]
    source_record: dict[str, Any]

    def direct_effects(self) -> dict[DirectKey, PairMap]:
        output: dict[DirectKey, PairMap] = {}
        for key, margins in self.direct_margins.items():
            direction = key[2]
            receiver = EXPECTED_RECEIVER[direction]
            output[key] = {
                pair_id: margin - self.baselines[(pair_id, receiver)]
                for pair_id, margin in margins.items()
            }
        return output

    def control_effects(self) -> dict[ControlKey, PairMap]:
        output: dict[ControlKey, PairMap] = {}
        for key, margins in self.controls_margins.items():
            direction = key[2]
            receiver = EXPECTED_RECEIVER[direction]
            output[key] = {
                pair_id: margin - self.baselines[(pair_id, receiver)]
                for pair_id, margin in margins.items()
            }
        return output


@dataclass
class VectorRequest:
    table: str
    row_id: str
    group: str
    pair_ids: tuple[str, ...]
    values: np.ndarray


@dataclass
class ResultState:
    tables: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {
            "directional_effects": {},
            "control_effects": {},
            "donor_aligned_t": {},
            "residual_router_gaps": {},
            "native_manipulation": {},
            "behavior_metrics": {},
            "identity_checks": {},
            "review_status": {},
            "availability": {},
        }
    )
    requests: list[VectorRequest] = field(default_factory=list)
    pair_estimands: list[dict[str, Any]] = field(default_factory=list)
    input_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    headline_refs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_vector(
        self,
        *,
        table: str,
        row: dict[str, Any],
        values_by_pair: Mapping[str, float],
        group: str,
    ) -> str:
        row_id = str(row["row_id"])
        if row_id in self.tables[table]:
            raise ValueError(f"duplicate {table} row_id: {row_id}")
        pair_ids = tuple(sorted(values_by_pair))
        if not pair_ids:
            raise ValueError(f"empty pair vector for {row_id}")
        values = np.asarray([values_by_pair[pair_id] for pair_id in pair_ids], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite pair vector for {row_id}")
        stored = dict(row)
        stored.update(
            {
                "estimate": None,
                "ci_low": None,
                "ci_high": None,
                "bootstrap_standard_error": None,
                "n_pairs": len(pair_ids),
                "pair_set_sha256": pair_ids_sha256(pair_ids),
            }
        )
        self.tables[table][row_id] = stored
        self.requests.append(
            VectorRequest(
                table=table,
                row_id=row_id,
                group=group,
                pair_ids=pair_ids,
                values=values,
            )
        )
        for pair_id, value in zip(pair_ids, values, strict=True):
            self.pair_estimands.append(
                {
                    "summary_table": table,
                    "summary_row_id": row_id,
                    "pair_id": pair_id,
                    "value": float(value),
                }
            )
        return row_id

    def add_na(self, *, table: str, row: dict[str, Any], reason: str) -> str:
        row_id = str(row["row_id"])
        if row_id in self.tables[table]:
            raise ValueError(f"duplicate {table} row_id: {row_id}")
        self.tables[table][row_id] = {
            **row,
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "bootstrap_standard_error": None,
            "n_pairs": None,
            "pair_set_sha256": None,
            "bootstrap_indices_sha256": None,
            "bootstrap_indices_values_sha256": None,
            "availability_status": reason,
        }
        return row_id

    def register_source(self, record: Mapping[str, Any], *, role: str) -> None:
        path = str(record["path"])
        existing = self.input_sources.get(path)
        if existing is None:
            self.input_sources[path] = {**record, "roles": [role]}
            return
        if existing["sha256"] != record["sha256"]:
            raise ValueError(f"source hash changed while analyzing: {path}")
        roles = list(existing["roles"])
        if role not in roles:
            roles.append(role)
        existing["roles"] = sorted(roles)
        selected_counts = {
            int(value)
            for value in existing.get("selected_row_counts", [])
            if value is not None
        }
        for value in (
            existing.get("selected_row_count"),
            record.get("selected_row_count"),
        ):
            if value is not None:
                selected_counts.add(int(value))
        if len(selected_counts) > 1:
            existing["selected_row_count"] = None
            existing["selected_row_counts"] = sorted(selected_counts)
            existing["selection_status"] = "multiple_analysis_scopes"


def pair_ids_sha256(pair_ids: Iterable[str]) -> str:
    """Hash sorted pair identifiers using the archive's LF-delimited contract."""

    ordered = sorted(set(pair_ids))
    payload = "".join(f"{pair_id}\n" for pair_id in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unified analysis manifest must use schema_version=1")
    return payload


def _read_jsonl(
    path: str | Path,
    *,
    selected_pair_ids: set[str] | None = None,
) -> JsonlLoad:
    source = Path(path).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    raw_row_count = 0
    size_bytes = 0
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            size_bytes += len(raw)
            if not raw.strip():
                continue
            raw_row_count += 1
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL row must be an object at {source}:{line_number}")
            pair_id = str(parsed.get("pair_id", "")).strip()
            if selected_pair_ids is None or pair_id in selected_pair_ids:
                rows.append(parsed)
    return JsonlLoad(
        rows=rows,
        raw_row_count=raw_row_count,
        selected_row_count=len(rows),
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def _file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "path": str(source),
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "row_count": None,
        "selected_row_count": None,
    }


def _jsonl_record(path: Path, loaded: JsonlLoad) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": loaded.sha256,
        "size_bytes": loaded.size_bytes,
        "row_count": loaded.raw_row_count,
        "selected_row_count": loaded.selected_row_count,
    }


def _mean_margin(row: Mapping[str, Any], *, label: str) -> float:
    scores = row.get("scores")
    if isinstance(scores, Mapping):
        try:
            correct_payload = scores["CORRECT"]
            incorrect_payload = scores["INCORRECT"]
            if not isinstance(correct_payload, Mapping) or not isinstance(
                incorrect_payload, Mapping
            ):
                raise TypeError
            correct = float(correct_payload["mean_logprob"])
            incorrect = float(incorrect_payload["mean_logprob"])
            value = incorrect - correct
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} has incomplete mean-logprob scores") from exc
    else:
        raw = row.get("margin_incorrect_minus_correct_mean")
        if raw is None:
            raise ValueError(f"{label} has no mean-logprob margin")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} has no mean-logprob margin") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} mean-logprob margin must be finite")
    stored = row.get("margin_incorrect_minus_correct_mean")
    if stored is not None and not math.isclose(
        value,
        float(stored),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{label} stored mean margin disagrees with scores")
    return value


def _direction_from_control(treatment: str, receiver: str) -> str:
    if treatment == "norm_random_delta":
        if receiver == "corrupt":
            return "clean_to_corrupt"
        if receiver == "clean":
            return "corrupt_to_clean"
        raise ValueError(f"invalid norm-random receiver: {receiver}")
    for direction in DIRECTIONS:
        if treatment.endswith(direction):
            return direction
    raise ValueError(f"cannot derive direction from control treatment: {treatment}")


def _site_fields(site: str) -> dict[str, Any]:
    match = SITE_PATTERN.fullmatch(site)
    if match is not None:
        raw_family = match.group("family")
        family = "same_layer_router" if raw_family == "router_logits" else raw_family
        return {
            "site": site,
            "site_family": family,
            "topology": "single_site",
            "anchor_layer": int(match.group("layer")),
            "registered_site_count": 1,
        }
    bundle = BUNDLE_PATTERN.fullmatch(site)
    if bundle is not None:
        return {
            "site": site,
            "site_family": "all_router",
            "topology": "all_router",
            "anchor_layer": None,
            "registered_site_count": int(bundle.group("count")),
        }
    raise ValueError(f"unsupported patch site: {site}")


def _insert_unique(target: dict[Any, Any], key: Any, value: Any, *, label: str) -> None:
    if key in target:
        raise ValueError(f"duplicate {label} row for {key}")
    target[key] = value


def _alpha(row: Mapping[str, Any], *, label: str) -> float | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("alpha") is None:
        return None
    try:
        value = float(metadata["alpha"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} alpha must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} alpha must be finite")
    return value


def _compact_manipulation(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return {"present": False}
    payload = metadata.get("manipulation")
    if not isinstance(payload, Mapping):
        return {"present": False, "patch_mode": metadata.get("patch_mode")}
    return {
        "present": True,
        "patch_mode": metadata.get("patch_mode"),
        "router_kl_receiver_to_patched": payload.get("router_kl_receiver_to_patched"),
        "topk_changed_rate": payload.get("topk_changed_rate"),
        "topk_overlap": payload.get("topk_overlap"),
        "all_registered_hooks_fired": payload.get("all_registered_hooks_fired"),
        "all_router_dispatch_verified": payload.get("all_router_dispatch_verified"),
        "registered_site_count": payload.get("registered_site_count"),
        "router_dispatch_diagnostics_expected": payload.get(
            "router_dispatch_diagnostics_expected"
        ),
        "router_dispatch_diagnostics_present": payload.get(
            "router_dispatch_diagnostics_present"
        ),
    }


def load_patch_rows(
    path: str | Path,
    *,
    selected_pair_ids: set[str],
    raw_expected_rows: int,
    selected_expected_rows: int,
    expected_sites: set[str],
    alpha_status: str,
    expected_controls: set[str],
) -> PatchRows:
    """Load and strictly normalize one frozen endpoint or all-router file."""

    source = Path(path).resolve(strict=True)
    loaded = _read_jsonl(source, selected_pair_ids=selected_pair_ids)
    if loaded.raw_row_count != raw_expected_rows:
        raise ValueError(
            f"{source} raw rows={loaded.raw_row_count}, expected={raw_expected_rows}"
        )
    if loaded.selected_row_count != selected_expected_rows:
        raise ValueError(
            f"{source} selected rows={loaded.selected_row_count}, "
            f"expected={selected_expected_rows}"
        )

    baselines: dict[tuple[str, str], float] = {}
    reruns: dict[tuple[str, str], float] = {}
    self_patches: dict[tuple[str, str, str], float] = {}
    direct_margins: dict[DirectKey, PairMap] = defaultdict(dict)
    controls_margins: dict[ControlKey, PairMap] = defaultdict(dict)
    manipulation_rows: dict[DirectKey, list[dict[str, Any]]] = defaultdict(list)
    control_manipulation_rows: dict[ControlKey, list[dict[str, Any]]] = defaultdict(list)
    observed_sites: set[str] = set()

    for row_index, row in enumerate(loaded.rows):
        label = f"{source}:{row_index + 1}"
        treatment = str(row.get("treatment", ""))
        if treatment not in ALLOWED_TREATMENTS:
            raise ValueError(f"{label} unsupported treatment {treatment!r}")
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id not in selected_pair_ids:
            raise ValueError(f"{label} invalid selected pair_id")
        receiver = str(row.get("receiver_variant", "")).strip()
        if receiver not in {"clean", "corrupt"}:
            raise ValueError(f"{label} invalid receiver_variant {receiver!r}")
        margin = _mean_margin(row, label=label)

        if treatment in {"unpatched", "unpatched_rerun"}:
            if row.get("site") is not None or row.get("donor_variant") is not None:
                raise ValueError(f"{label} baseline/rerun row names a site or donor")
            target = baselines if treatment == "unpatched" else reruns
            _insert_unique(target, (pair_id, receiver), margin, label=treatment)
            continue

        site = str(row.get("site", "")).strip()
        if site not in expected_sites:
            raise ValueError(f"{label} unexpected site {site!r}")
        observed_sites.add(site)
        donor = str(row.get("donor_variant", "")).strip()

        if treatment == "self_patch":
            if donor != receiver:
                raise ValueError(f"{label} self-patch donor != receiver")
            _insert_unique(
                self_patches,
                (pair_id, receiver, site),
                margin,
                label="self_patch",
            )
            continue

        if treatment in DIRECTIONS:
            direction = treatment
            if receiver != EXPECTED_RECEIVER[direction] or donor != EXPECTED_DONOR[direction]:
                raise ValueError(f"{label} donor/receiver disagree with {direction}")
            dose = _alpha(row, label=label)
            if alpha_status == "explicit_full_strength_1":
                if dose is None or not math.isclose(dose, 1.0, abs_tol=1.0e-12):
                    raise ValueError(f"{label} requires explicit alpha=1")
            elif alpha_status == "fixed_full_strength_legacy_unrecorded":
                if dose is not None:
                    raise ValueError(f"{label} legacy endpoint unexpectedly records alpha")
            elif alpha_status == "explicit_dose":
                if dose not in {0.0, 0.5, 1.0, 2.0}:
                    raise ValueError(f"{label} unsupported all-router alpha {dose}")
            else:
                raise ValueError(f"unsupported alpha_status: {alpha_status}")
            direct_key = (site, dose, direction)
            _insert_unique(
                direct_margins[direct_key],
                pair_id,
                margin,
                label=f"{direction}/{site}/{dose}",
            )
            manipulation_rows[direct_key].append(_compact_manipulation(row))
            continue

        control_name = CONTROL_TREATMENTS[treatment]
        if control_name not in expected_controls:
            raise ValueError(f"{label} unexpected {control_name} control")
        direction = _direction_from_control(treatment, receiver)
        if receiver != EXPECTED_RECEIVER[direction]:
            raise ValueError(f"{label} receiver disagrees with control direction")
        control_key = (site, control_name, direction)
        _insert_unique(
            controls_margins[control_key],
            pair_id,
            margin,
            label=f"{control_name}/{direction}/{site}",
        )
        control_manipulation_rows[control_key].append(_compact_manipulation(row))

    if observed_sites != expected_sites:
        raise ValueError(
            f"{source} observed sites differ: {sorted(observed_sites)} != {sorted(expected_sites)}"
        )
    ordered_pairs = tuple(sorted(selected_pair_ids))
    expected_baselines = {(pair_id, variant) for pair_id in ordered_pairs for variant in ("clean", "corrupt")}
    if set(baselines) != expected_baselines or set(reruns) != expected_baselines:
        raise ValueError(f"{source} does not have exact baseline/rerun coverage")
    expected_self = {
        (pair_id, variant, site)
        for pair_id in ordered_pairs
        for variant in ("clean", "corrupt")
        for site in expected_sites
    }
    if set(self_patches) != expected_self:
        raise ValueError(f"{source} does not have exact self-patch coverage")

    cells = {(site, dose) for site, dose, _direction in direct_margins}
    if not cells:
        raise ValueError(f"{source} has no direct rows")
    for site, dose in sorted(cells, key=lambda item: (item[0], str(item[1]))):
        for direction in DIRECTIONS:
            direct_coverage_key = (site, dose, direction)
            if set(direct_margins.get(direct_coverage_key, {})) != selected_pair_ids:
                raise ValueError(
                    f"{source} incomplete direct coverage for {direct_coverage_key}"
                )

    observed_controls = {key[1] for key in controls_margins}
    if observed_controls != expected_controls:
        raise ValueError(
            f"{source} controls differ: {sorted(observed_controls)} != {sorted(expected_controls)}"
        )
    pre_error_pair_set: set[str] | None = None
    for site in sorted(expected_sites):
        for control_name in sorted(expected_controls):
            for direction in DIRECTIONS:
                control_coverage_key = (site, control_name, direction)
                pair_set = set(controls_margins.get(control_coverage_key, {}))
                if control_name == "pre_error":
                    if len(pair_set) != 285:
                        raise ValueError(
                            f"{source} pre-error n != 285 for {control_coverage_key}"
                        )
                    if pre_error_pair_set is None:
                        pre_error_pair_set = pair_set
                    elif pair_set != pre_error_pair_set:
                        raise ValueError(f"{source} inconsistent pre-error pair set")
                elif pair_set != selected_pair_ids:
                    raise ValueError(
                        f"{source} incomplete {control_name} coverage for {control_coverage_key}"
                    )

    rerun_deltas = [
        abs(reruns[key] - baseline) for key, baseline in baselines.items()
    ]
    self_deltas = [
        abs(margin - baselines[(pair_id, variant)])
        for (pair_id, variant, _site), margin in self_patches.items()
    ]
    rerun_max = max(rerun_deltas, default=0.0)
    self_max = max(self_deltas, default=0.0)
    identity = {
        "identity_tolerance": IDENTITY_TOLERANCE,
        "unpatched_rerun_n": len(rerun_deltas),
        "unpatched_rerun_max_abs_delta": rerun_max,
        "self_patch_n": len(self_deltas),
        "self_patch_max_abs_delta": self_max,
        "passed": rerun_max <= IDENTITY_TOLERANCE and self_max <= IDENTITY_TOLERANCE,
    }
    if not identity["passed"]:
        raise ValueError(f"{source} identity check failed: {identity}")

    return PatchRows(
        path=source,
        raw_row_count=loaded.raw_row_count,
        selected_row_count=loaded.selected_row_count,
        pair_ids=ordered_pairs,
        baselines=baselines,
        reruns=reruns,
        self_patches=self_patches,
        direct_margins=dict(direct_margins),
        controls_margins=dict(controls_margins),
        manipulation_rows=dict(manipulation_rows),
        control_manipulation_rows=dict(control_manipulation_rows),
        identity=identity,
        source_record=_jsonl_record(source, loaded),
    )


def _alpha_label(alpha: float | None, alpha_status: str) -> str:
    if alpha is None:
        return "na-legacy" if "legacy" in alpha_status else "na"
    if math.isclose(alpha, round(alpha), rel_tol=0.0, abs_tol=1.0e-12):
        return str(round(alpha))
    return str(alpha).replace(".", "p")


def _site_slug(site: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", site.lower()).strip("-")


def _row_id(
    prefix: str,
    *,
    benchmark: str,
    model_id: str,
    site: str,
    alpha: float | None,
    alpha_status: str,
    effect_kind: str,
    direction: str | None = None,
) -> str:
    parts = [
        prefix,
        benchmark,
        model_id,
        _site_slug(site),
        f"alpha-{_alpha_label(alpha, alpha_status)}",
        effect_kind,
    ]
    if direction is not None:
        parts.append(direction)
    return "/".join(parts)


def _base_row(
    model: Mapping[str, Any],
    *,
    benchmark: str,
    site: str,
    alpha: float | None,
    alpha_status: str,
    source_path: str,
) -> dict[str, Any]:
    return {
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "model_hf_id": model.get("model_hf_id"),
        "model_revision": model.get("model_revision"),
        "benchmark": benchmark,
        "protocol_grade": model["protocol_grade"],
        **_site_fields(site),
        "common_anchor": (
            _site_fields(site)["anchor_layer"] in set(model.get("common_anchors", []))
            if _site_fields(site)["anchor_layer"] is not None
            else None
        ),
        "alpha": alpha,
        "alpha_status": alpha_status,
        "metric": PRIMARY_METRIC,
        "units": PRIMARY_UNITS,
        "source_prediction_path": source_path,
    }


def _direction_semantics(benchmark: str, direction: str) -> dict[str, str]:
    if benchmark == "processbench":
        if direction == "clean_to_corrupt":
            return {
                "direction_semantic": "matched_correct_to_first_error",
                "donor_role": "matched_correct",
                "receiver_role": "first_error",
            }
        return {
            "direction_semantic": "first_error_to_matched_correct",
            "donor_role": "first_error",
            "receiver_role": "matched_correct",
        }
    return {
        "direction_semantic": direction,
        "donor_role": EXPECTED_DONOR[direction],
        "receiver_role": EXPECTED_RECEIVER[direction],
    }


def donor_aligned_t(forward: Mapping[str, float], reverse: Mapping[str, float]) -> PairMap:
    pair_ids = set(forward) & set(reverse)
    if pair_ids != set(forward) or pair_ids != set(reverse):
        raise ValueError("T directions must use identical pair sets")
    return {
        pair_id: 0.5 * (reverse[pair_id] - forward[pair_id])
        for pair_id in pair_ids
    }


def paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> PairMap:
    pair_ids = set(left) & set(right)
    if pair_ids != set(left) or pair_ids != set(right):
        raise ValueError("paired gap vectors must use identical pair sets")
    return {pair_id: left[pair_id] - right[pair_id] for pair_id in pair_ids}


def _summary_numbers(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None, None
    return float(np.mean(finite)), min(finite), max(finite)


def _bool_rate(values: Sequence[Any]) -> float | None:
    observed = [value for value in values if isinstance(value, bool)]
    if not observed:
        return None
    return float(np.mean(observed))


def _summarize_manipulation(
    rows: Sequence[Mapping[str, Any]],
    *,
    site_family: str,
) -> dict[str, Any]:
    present = [row for row in rows if row.get("present") is True]
    dispatch_rate = _bool_rate(
        [row.get("all_router_dispatch_verified") for row in present]
    )
    if site_family not in {"same_layer_router", "all_router"}:
        status = "not_applicable_non_router_site"
    elif not present:
        status = "legacy_not_recorded"
    elif dispatch_rate is None:
        status = "available_native_strength_legacy_dispatch_status_unrecorded"
    else:
        status = "available_native"
    kl_mean, kl_min, kl_max = _summary_numbers(
        [float(row["router_kl_receiver_to_patched"]) for row in present if row.get("router_kl_receiver_to_patched") is not None]
    )
    changed_mean, changed_min, changed_max = _summary_numbers(
        [float(row["topk_changed_rate"]) for row in present if row.get("topk_changed_rate") is not None]
    )
    overlap_mean, overlap_min, overlap_max = _summary_numbers(
        [float(row["topk_overlap"]) for row in present if row.get("topk_overlap") is not None]
    )
    registered = [
        int(row["registered_site_count"])
        for row in present
        if row.get("registered_site_count") is not None
    ]
    return {
        "availability_status": status,
        "n_rows": len(rows),
        "n_native_metadata_present": len(present),
        "router_kl_receiver_to_patched_mean": kl_mean,
        "router_kl_receiver_to_patched_min": kl_min,
        "router_kl_receiver_to_patched_max": kl_max,
        "topk_changed_rate_mean": changed_mean,
        "topk_changed_rate_min": changed_min,
        "topk_changed_rate_max": changed_max,
        "topk_overlap_mean": overlap_mean,
        "topk_overlap_min": overlap_min,
        "topk_overlap_max": overlap_max,
        "all_registered_hooks_fired_rate": _bool_rate(
            [row.get("all_registered_hooks_fired") for row in present]
        ),
        "all_router_dispatch_verified_rate": dispatch_rate,
        "router_dispatch_diagnostics_expected_rate": _bool_rate(
            [row.get("router_dispatch_diagnostics_expected") for row in present]
        ),
        "router_dispatch_diagnostics_present_rate": _bool_rate(
            [row.get("router_dispatch_diagnostics_present") for row in present]
        ),
        "registered_site_count_min": min(registered) if registered else None,
        "registered_site_count_max": max(registered) if registered else None,
        "patch_modes": sorted(
            {str(row["patch_mode"]) for row in present if row.get("patch_mode") is not None}
        ),
    }


def _add_patch_summaries(
    state: ResultState,
    *,
    model: Mapping[str, Any],
    benchmark: str,
    patch: PatchRows,
    alpha_status: str,
    expected_controls: set[str],
    source_arm: str,
) -> tuple[dict[CellKey, PairMap], dict[tuple[str, str], PairMap]]:
    """Add direct/control/T rows and return direct T and direct-effect maps."""

    model_id = str(model["model_id"])
    group = f"{benchmark}|{model_id}"
    source_path = str(patch.path)
    direct_effects = patch.direct_effects()
    control_effects = patch.control_effects()
    direct_t: dict[CellKey, PairMap] = {}
    direct_by_site_direction: dict[tuple[str, str], PairMap] = {}

    cells = sorted(
        {(site, dose) for site, dose, _direction in direct_effects},
        key=lambda value: (_site_fields(value[0])["topology"], value[0], str(value[1])),
    )
    for site, dose in cells:
        base = _base_row(
            model,
            benchmark=benchmark,
            site=site,
            alpha=dose,
            alpha_status=alpha_status,
            source_path=source_path,
        )
        for direction in DIRECTIONS:
            values = direct_effects[(site, dose, direction)]
            direct_by_site_direction[(site, direction)] = values
            row_id = _row_id(
                "directional",
                benchmark=benchmark,
                model_id=model_id,
                site=site,
                alpha=dose,
                alpha_status=alpha_status,
                effect_kind="direct",
                direction=direction,
            )
            state.add_vector(
                table="directional_effects",
                row={
                    "row_id": row_id,
                    **base,
                    "effect_kind": "direct",
                    "direction": direction,
                    **_direction_semantics(benchmark, direction),
                    "estimand_definition": "patched_minus_same_pair_receiver_unpatched",
                },
                values_by_pair=values,
                group=group,
            )

            manipulation_id = _row_id(
                "manipulation",
                benchmark=benchmark,
                model_id=model_id,
                site=site,
                alpha=dose,
                alpha_status=alpha_status,
                effect_kind="direct",
                direction=direction,
            )
            manipulation = _summarize_manipulation(
                patch.manipulation_rows[(site, dose, direction)],
                site_family=str(base["site_family"]),
            )
            state.tables["native_manipulation"][manipulation_id] = {
                "row_id": manipulation_id,
                **base,
                "effect_kind": "direct",
                "direction": direction,
                **_direction_semantics(benchmark, direction),
                **manipulation,
            }

        t_values = donor_aligned_t(
            direct_effects[(site, dose, "clean_to_corrupt")],
            direct_effects[(site, dose, "corrupt_to_clean")],
        )
        direct_t[(site, dose)] = t_values
        t_id = _row_id(
            "t",
            benchmark=benchmark,
            model_id=model_id,
            site=site,
            alpha=dose,
            alpha_status=alpha_status,
            effect_kind="direct",
        )
        state.add_vector(
            table="donor_aligned_t",
            row={
                "row_id": t_id,
                **base,
                "effect_kind": "direct",
                "forward_direction": "clean_to_corrupt",
                "reverse_direction": "corrupt_to_clean",
                "estimand_definition": "0.5*(corrupt_to_clean-clean_to_corrupt)",
                "positive_interpretation": "donor_aligned_restoration",
            },
            values_by_pair=t_values,
            group=group,
        )

        for control_name in sorted(expected_controls):
            control_forward = control_effects[(site, control_name, "clean_to_corrupt")]
            control_reverse = control_effects[(site, control_name, "corrupt_to_clean")]
            direct_forward = direct_effects[(site, dose, "clean_to_corrupt")]
            direct_reverse = direct_effects[(site, dose, "corrupt_to_clean")]
            common_pairs = (
                set(control_forward)
                & set(control_reverse)
                & set(direct_forward)
                & set(direct_reverse)
            )
            if set(control_forward) != common_pairs or set(control_reverse) != common_pairs:
                raise ValueError(f"control directions do not share exact pairs: {site}/{control_name}")
            direct_forward_common = {pair_id: direct_forward[pair_id] for pair_id in common_pairs}
            direct_reverse_common = {pair_id: direct_reverse[pair_id] for pair_id in common_pairs}
            for direction, control_values, direct_values in (
                ("clean_to_corrupt", control_forward, direct_forward_common),
                ("corrupt_to_clean", control_reverse, direct_reverse_common),
            ):
                control_manipulation_id = _row_id(
                    "manipulation",
                    benchmark=benchmark,
                    model_id=model_id,
                    site=site,
                    alpha=None,
                    alpha_status="control_dose_not_parameterized",
                    effect_kind=f"control_raw_{control_name}",
                    direction=direction,
                )
                state.tables["native_manipulation"][control_manipulation_id] = {
                    "row_id": control_manipulation_id,
                    **base,
                    "alpha": None,
                    "alpha_status": "control_dose_not_parameterized",
                    "effect_kind": f"control_raw_{control_name}",
                    "control_name": control_name,
                    "direction": direction,
                    **_direction_semantics(benchmark, direction),
                    **_summarize_manipulation(
                        patch.control_manipulation_rows[(site, control_name, direction)],
                        site_family=str(base["site_family"]),
                    ),
                }
                for contrast_kind, values in (
                    ("control_raw", control_values),
                    (
                        "direct_minus_control",
                        paired_difference(direct_values, control_values),
                    ),
                ):
                    effect_kind = f"{contrast_kind}_{control_name}"
                    control_id = _row_id(
                        "control",
                        benchmark=benchmark,
                        model_id=model_id,
                        site=site,
                        alpha=dose,
                        alpha_status=alpha_status,
                        effect_kind=effect_kind,
                        direction=direction,
                    )
                    state.add_vector(
                        table="control_effects",
                        row={
                            "row_id": control_id,
                            **base,
                            "effect_kind": effect_kind,
                            "control_name": control_name,
                            "contrast_kind": contrast_kind,
                            "direction": direction,
                            **_direction_semantics(benchmark, direction),
                            "estimand_definition": (
                                "control_patched_minus_same_pair_receiver_unpatched"
                                if contrast_kind == "control_raw"
                                else "direct_effect_minus_receiver_aligned_control_effect"
                            ),
                        },
                        values_by_pair=values,
                        group=group,
                    )

            control_t = donor_aligned_t(control_forward, control_reverse)
            direct_t_common = donor_aligned_t(direct_forward_common, direct_reverse_common)
            for contrast_kind, values in (
                ("control_raw", control_t),
                ("direct_minus_control", paired_difference(direct_t_common, control_t)),
            ):
                effect_kind = f"{contrast_kind}_{control_name}"
                control_t_id = _row_id(
                    "t",
                    benchmark=benchmark,
                    model_id=model_id,
                    site=site,
                    alpha=dose,
                    alpha_status=alpha_status,
                    effect_kind=effect_kind,
                )
                state.add_vector(
                    table="donor_aligned_t",
                    row={
                        "row_id": control_t_id,
                        **base,
                        "effect_kind": effect_kind,
                        "control_name": control_name,
                        "contrast_kind": contrast_kind,
                        "forward_direction": "clean_to_corrupt",
                        "reverse_direction": "corrupt_to_clean",
                        "estimand_definition": (
                            "0.5*(control_corrupt_to_clean-control_clean_to_corrupt)"
                            if contrast_kind == "control_raw"
                            else "direct_T_minus_receiver_aligned_control_T"
                        ),
                        "positive_interpretation": "donor_aligned_restoration",
                    },
                    values_by_pair=values,
                    group=group,
                )

    identity_id = f"identity/{benchmark}/{model_id}/{source_arm}"
    state.tables["identity_checks"][identity_id] = {
        "row_id": identity_id,
        "benchmark": benchmark,
        "model_id": model_id,
        "display_name": model["display_name"],
        "protocol_grade": model["protocol_grade"],
        "source_prediction_path": source_path,
        **patch.identity,
    }
    return direct_t, direct_by_site_direction


def _gap_row_id(
    *,
    benchmark: str,
    model_id: str,
    comparison_family: str,
    anchor_layer: int,
    router_alpha: float | None,
    router_alpha_status: str,
) -> str:
    return "/".join(
        (
            "gap",
            benchmark,
            model_id,
            comparison_family,
            f"layer-{anchor_layer}",
            f"router-alpha-{_alpha_label(router_alpha, router_alpha_status)}",
            "direct",
        )
    )


def _add_gap(
    state: ResultState,
    *,
    model: Mapping[str, Any],
    benchmark: str,
    comparison_family: str,
    anchor_layer: int,
    residual_site: str,
    residual_alpha: float | None,
    residual_alpha_status: str,
    residual_t: Mapping[str, float],
    residual_source_path: str,
    router_site: str,
    router_alpha: float | None,
    router_alpha_status: str,
    router_t: Mapping[str, float],
    router_source_path: str,
) -> str:
    row_id = _gap_row_id(
        benchmark=benchmark,
        model_id=str(model["model_id"]),
        comparison_family=comparison_family,
        anchor_layer=anchor_layer,
        router_alpha=router_alpha,
        router_alpha_status=router_alpha_status,
    )
    return state.add_vector(
        table="residual_router_gaps",
        row={
            "row_id": row_id,
            "model_id": model["model_id"],
            "display_name": model["display_name"],
            "model_hf_id": model.get("model_hf_id"),
            "model_revision": model.get("model_revision"),
            "benchmark": benchmark,
            "protocol_grade": model["protocol_grade"],
            "comparison_family": comparison_family,
            "site": residual_site,
            "site_family": "residual_minus_router",
            "topology": (
                "single_site_residual_vs_same_layer_router"
                if comparison_family == "same_layer_router"
                else "single_site_residual_vs_simultaneous_all_router"
            ),
            "anchor_layer": anchor_layer,
            "common_anchor": anchor_layer in set(model.get("common_anchors", [])),
            "alpha": router_alpha,
            "alpha_status": router_alpha_status,
            "residual_site": residual_site,
            "residual_alpha": residual_alpha,
            "residual_alpha_status": residual_alpha_status,
            "router_site": router_site,
            "router_alpha": router_alpha,
            "router_alpha_status": router_alpha_status,
            "effect_kind": "direct",
            "metric": PRIMARY_METRIC,
            "units": PRIMARY_UNITS,
            "estimand_definition": "donor_aligned_T_residual_minus_donor_aligned_T_router",
            "residual_source_prediction_path": residual_source_path,
            "router_source_prediction_path": router_source_path,
        },
        values_by_pair=paired_difference(residual_t, router_t),
        group=f"{benchmark}|{model['model_id']}",
    )


def _behavior_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pair_ids: set[str],
    benchmark: str,
) -> dict[str, Any]:
    expected_keys = {
        (pair_id, variant)
        for pair_id in expected_pair_ids
        for variant in ("clean", "corrupt")
    }
    observed: dict[tuple[str, str], tuple[bool, float]] = {}
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id", ""))
        variant = str(row.get("variant", ""))
        key = (pair_id, variant)
        if key not in expected_keys:
            raise ValueError(f"unexpected behavior key: {key}")
        if key in observed:
            raise ValueError(f"duplicate behavior key: {key}")
        margin = _mean_margin(row, label=f"behavior row {index}")
        gold = str(row.get("gold", ""))
        expected_gold = "CORRECT" if variant == "clean" else "INCORRECT"
        if gold != expected_gold:
            raise ValueError(f"behavior {key} gold={gold}, expected={expected_gold}")
        predicted = "INCORRECT" if margin > 0.0 else "CORRECT"
        if row.get("prediction") is not None and str(row["prediction"]) != predicted:
            raise ValueError(f"behavior {key} prediction disagrees with mean margin")
        observed[key] = (gold == "INCORRECT", margin)
    if set(observed) != expected_keys:
        raise ValueError("behavior rows do not have exact pair/variant coverage")
    ordered = [observed[key] for key in sorted(observed)]
    labels = np.asarray([item[0] for item in ordered], dtype=bool)
    margins = np.asarray([item[1] for item in ordered], dtype=float)
    predictions = margins > 0.0
    return {
        "n_rows": len(ordered),
        "n_pairs": len(expected_pair_ids),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
        "accuracy": float(np.mean(labels == predictions)),
        "auroc": float(binary_auroc(labels, margins)),
        "mean_margin_incorrect_minus_correct": float(np.mean(margins)),
        "metric": PRIMARY_METRIC,
        "pair_set_sha256": pair_ids_sha256(expected_pair_ids),
        "role_semantics": (
            {
                "clean": "matched_correct",
                "corrupt": "first_error",
                "counterfactual_pair_claimed": False,
            }
            if benchmark == "processbench"
            else {"clean": "clean", "corrupt": "corrupt", "counterfactual_pair_claimed": True}
        ),
    }


def _load_behavior(
    state: ResultState,
    *,
    model: Mapping[str, Any],
    benchmark: str,
    behavior: Mapping[str, Any],
    selected_pair_ids: set[str],
) -> None:
    path = Path(str(behavior["predictions"])).resolve(strict=True)
    loaded = _read_jsonl(path, selected_pair_ids=selected_pair_ids)
    raw_expected = int(behavior["raw_expected_rows"])
    selected_expected = int(behavior["selected_expected_rows"])
    if loaded.raw_row_count != raw_expected or loaded.selected_row_count != selected_expected:
        raise ValueError(
            f"behavior counts for {model['model_id']}: "
            f"raw={loaded.raw_row_count}/{raw_expected}, "
            f"selected={loaded.selected_row_count}/{selected_expected}"
        )
    state.register_source(
        _jsonl_record(path, loaded),
        role=f"{benchmark}:{model['model_id']}:behavior_predictions",
    )
    metrics_path = Path(str(behavior["metrics"])).resolve(strict=True)
    state.register_source(
        _file_record(metrics_path),
        role=f"{benchmark}:{model['model_id']}:behavior_metrics",
    )
    summary = _behavior_summary(
        loaded.rows,
        expected_pair_ids=selected_pair_ids,
        benchmark=benchmark,
    )
    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    def validate_archived_metrics(payload: Mapping[str, Any]) -> None:
        field_map = {
            "n": "n_rows",
            "accuracy": "accuracy",
            "auroc": "auroc",
            "mean_margin_incorrect_minus_correct": (
                "mean_margin_incorrect_minus_correct"
            ),
        }
        for archived_field, observed_field in field_map.items():
            expected_value = source_metrics.get(archived_field)
            observed_value = payload[observed_field]
            if expected_value is None or not math.isclose(
                float(observed_value),
                float(expected_value),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"behavior metrics mismatch for {model['model_id']} "
                    f"field {archived_field}"
                )

    comparison_status = "not_comparable_selected_common_subset"
    if loaded.raw_row_count == loaded.selected_row_count:
        validate_archived_metrics(summary)
        comparison_status = "matched_archived_metrics"
    row_id = f"behavior/{benchmark}/{model['model_id']}"
    state.tables["behavior_metrics"][row_id] = {
        "row_id": row_id,
        "benchmark": benchmark,
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "protocol_grade": model["protocol_grade"],
        "source_prediction_path": str(path),
        "source_metrics_path": str(metrics_path),
        "source_metrics_comparison_status": comparison_status,
        "selection_scope": (
            "native_archive_and_common_pair"
            if loaded.raw_row_count == loaded.selected_row_count
            else "common_pair_subset"
        ),
        "raw_source_row_count": loaded.raw_row_count,
        "selected_row_count": loaded.selected_row_count,
        **summary,
    }
    if loaded.raw_row_count != loaded.selected_row_count:
        native = _read_jsonl(path)
        state.register_source(
            _jsonl_record(path, native),
            role=f"{benchmark}:{model['model_id']}:behavior_predictions_native_archive",
        )
        native_pair_ids = {
            str(row.get("pair_id", "")).strip()
            for row in native.rows
            if str(row.get("pair_id", "")).strip()
        }
        native_summary = _behavior_summary(
            native.rows,
            expected_pair_ids=native_pair_ids,
            benchmark=benchmark,
        )
        validate_archived_metrics(native_summary)
        native_row_id = f"behavior/{benchmark}/{model['model_id']}/native_archive"
        state.tables["behavior_metrics"][native_row_id] = {
            "row_id": native_row_id,
            "benchmark": benchmark,
            "model_id": model["model_id"],
            "display_name": model["display_name"],
            "protocol_grade": model["protocol_grade"],
            "source_prediction_path": str(path),
            "source_metrics_path": str(metrics_path),
            "source_metrics_comparison_status": "matched_archived_metrics",
            "selection_scope": "native_archive",
            "raw_source_row_count": native.raw_row_count,
            "selected_row_count": native.selected_row_count,
            **native_summary,
        }


def _review_passed(payload: Mapping[str, Any]) -> bool:
    passed = payload.get("passed")
    decision = payload.get("decision", payload.get("scientific_decision"))
    return passed is True and (decision is None or str(decision).upper() == "PASS")


def _load_review(
    state: ResultState,
    *,
    model: Mapping[str, Any],
    benchmark: str,
    path: str | Path,
    expected_check_count: int | None,
    review_arm: str,
) -> None:
    review_path = Path(path).resolve(strict=True)
    record = _file_record(review_path)
    state.register_source(
        record,
        role=f"{benchmark}:{model['model_id']}:review:{review_arm}",
    )
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not _review_passed(payload):
        raise ValueError(f"review is not PASS: {review_path}")
    checks = payload.get("checks")
    check_count = len(checks) if isinstance(checks, list) else None
    if check_count != expected_check_count:
        raise ValueError(
            f"review check count mismatch for {review_path}: {check_count} != {expected_check_count}"
        )
    if isinstance(checks, list) and any(
        not isinstance(check, Mapping) or check.get("passed") is not True
        for check in checks
    ):
        raise ValueError(f"review contains a non-passing check: {review_path}")
    validation_accessed = payload.get("validation_accessed")
    if validation_accessed is True:
        raise ValueError(f"review reports validation accessed: {review_path}")
    row_id = f"review/{benchmark}/{model['model_id']}/{review_arm}"
    state.tables["review_status"][row_id] = {
        "row_id": row_id,
        "benchmark": benchmark,
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "review_arm": review_arm,
        "review_path": str(review_path),
        "review_sha256": record["sha256"],
        "review_status": "pass",
        "passed": True,
        "checks_passed": check_count,
        "checks_total": check_count,
        "checks_status": (
            "legacy_schema_without_checks_array"
            if check_count is None
            else "all_checks_passed"
        ),
        "validation_accessed": validation_accessed,
        "validation_status": (
            "not_accessed"
            if validation_accessed is False
            else "legacy_not_recorded"
        ),
    }


def _derive_pair_ids(path: str | Path, *, unpatched_only: bool) -> set[str]:
    source = Path(path).resolve(strict=True)
    pair_ids: set[str] = set()
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row must be an object at {source}:{line_number}")
            if unpatched_only and row.get("treatment") != "unpatched":
                continue
            pair_id = str(row.get("pair_id", "")).strip()
            if pair_id:
                pair_ids.add(pair_id)
    if not pair_ids:
        raise ValueError(f"no pair ids derived from {source}")
    return pair_ids


def _expanded_prm_model(model: Mapping[str, Any]) -> dict[str, Any]:
    expanded = dict(model)
    if model.get("run_root") is None:
        return expanded
    root = Path(str(model["run_root"]))
    endpoint_stage = str(model["endpoint_stage"])
    endpoint_root = root / endpoint_stage
    expanded["endpoint"] = {
        "predictions": str(endpoint_root / "predictions.jsonl"),
        "config": str(endpoint_root / "config.resolved.yaml"),
        "metrics": str(endpoint_root / "metrics.json"),
        "raw_expected_rows": int(model["endpoint_expected_rows"]),
        "selected_expected_rows": int(model["endpoint_expected_rows"]),
        "pair_selection": "prmbench_common_290",
        "alpha_status": model["endpoint_alpha_status"],
    }
    router_root = root / "router"
    expanded["all_router"] = {
        "predictions": str(router_root / "predictions.jsonl"),
        "config": str(router_root / "config.resolved.yaml"),
        "metrics": str(router_root / "metrics.json"),
        "raw_expected_rows": 4060,
        "selected_expected_rows": 4060,
        "pair_selection": "prmbench_common_290",
        "registered_site_count": int(model["all_router_registered_site_count"]),
    }
    behavior_root = root / "behavior"
    expanded["behavior"] = {
        "predictions": str(behavior_root / "predictions.jsonl"),
        "metrics": str(behavior_root / "metrics.json"),
        "raw_expected_rows": int(model["behavior_expected_rows"]),
        "selected_expected_rows": int(model["behavior_expected_rows"]),
        "pair_selection": "prmbench_behavior_common_64",
    }
    expanded["reviews"] = [str(root / str(model["review_file"]))]
    expanded["review_expected_check_counts"] = [
        int(model["review_expected_check_count"])
    ]
    return expanded


def _register_arm_sidecars(
    state: ResultState,
    *,
    benchmark: str,
    model_id: str,
    arm: str,
    arm_spec: Mapping[str, Any],
) -> dict[str, Any]:
    code_record: dict[str, Any] = {
        "arm": arm,
        "commit": None,
        "dirty": None,
        "status": "environment_not_emitted",
    }
    for field_name in ("config", "metrics"):
        value = arm_spec.get(field_name)
        if value is None:
            continue
        state.register_source(
            _file_record(str(value)),
            role=f"{benchmark}:{model_id}:{arm}_{field_name}",
        )
    predictions = Path(str(arm_spec["predictions"])).resolve(strict=True)
    environment = predictions.parent / "environment.json"
    if environment.is_file():
        record = _file_record(environment)
        state.register_source(
            record,
            role=f"{benchmark}:{model_id}:{arm}_environment",
        )
        payload = json.loads(environment.read_text(encoding="utf-8"))
        git = payload.get("git") if isinstance(payload, Mapping) else None
        if isinstance(git, Mapping):
            code_record = {
                "arm": arm,
                "commit": git.get("commit"),
                "dirty": git.get("dirty"),
                "status": "recorded",
                "environment_path": str(environment),
                "environment_sha256": record["sha256"],
            }
        validation_accessed = payload.get("validation_accessed")
        if validation_accessed is True:
            raise ValueError(f"source environment reports validation accessed: {environment}")
        code_record["validation_accessed"] = validation_accessed
    return code_record


def _endpoint_sites(model: Mapping[str, Any]) -> set[str]:
    anchors = [int(value) for value in model["anchors"]]
    if str(model["model_id"]) == "olmoe":
        return {f"layer_{anchors[0]}.post_moe_residual"}
    return {
        f"layer_{layer}.{suffix}"
        for layer in anchors
        for suffix in ("router_logits", "moe_output", "post_moe_residual")
    }


def _record_availability(
    state: ResultState,
    *,
    model: Mapping[str, Any],
    benchmark: str,
    component: str,
    status: str,
    reason: str | None = None,
) -> None:
    row_id = f"availability/{benchmark}/{model['model_id']}/{component}"
    state.tables["availability"][row_id] = {
        "row_id": row_id,
        "benchmark": benchmark,
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "protocol_grade": model["protocol_grade"],
        "component": component,
        "availability_status": status,
        "reason": reason,
    }


def _cell_reference(
    *,
    benchmark: str,
    model_id: str,
    site: str,
    alpha: float | None,
    alpha_status: str,
) -> dict[str, Any]:
    return {
        "site": site,
        "alpha": alpha,
        "alpha_status": alpha_status,
        "directional": {
            direction: _row_id(
                "directional",
                benchmark=benchmark,
                model_id=model_id,
                site=site,
                alpha=alpha,
                alpha_status=alpha_status,
                effect_kind="direct",
                direction=direction,
            )
            for direction in DIRECTIONS
        },
        "donor_aligned_t": _row_id(
            "t",
            benchmark=benchmark,
            model_id=model_id,
            site=site,
            alpha=alpha,
            alpha_status=alpha_status,
            effect_kind="direct",
        ),
    }


def _analyze_prm_model(
    state: ResultState,
    *,
    raw_model: Mapping[str, Any],
    common_pairs: set[str],
    behavior_pairs: set[str],
) -> None:
    model = _expanded_prm_model(raw_model)
    model_id = str(model["model_id"])
    endpoint_spec = model["endpoint"]
    if not isinstance(endpoint_spec, Mapping):
        raise ValueError(f"{model_id} endpoint spec must be a mapping")
    endpoint_sites = _endpoint_sites(model)
    endpoint_controls = (
        {"norm_random", "random_matched"}
        if model_id == "olmoe"
        else {"norm_random", "random_matched", "pre_error"}
    )
    endpoint_alpha_status = str(endpoint_spec["alpha_status"])
    endpoint = load_patch_rows(
        str(endpoint_spec["predictions"]),
        selected_pair_ids=common_pairs,
        raw_expected_rows=int(endpoint_spec["raw_expected_rows"]),
        selected_expected_rows=int(endpoint_spec["selected_expected_rows"]),
        expected_sites=endpoint_sites,
        alpha_status=endpoint_alpha_status,
        expected_controls=endpoint_controls,
    )
    state.register_source(
        endpoint.source_record,
        role=f"prmbench:{model_id}:endpoint_predictions",
    )
    code_provenance = [
        _register_arm_sidecars(
            state,
            benchmark="prmbench",
            model_id=model_id,
            arm="endpoint",
            arm_spec=endpoint_spec,
        )
    ]
    endpoint_t, _endpoint_directions = _add_patch_summaries(
        state,
        model=model,
        benchmark="prmbench",
        patch=endpoint,
        alpha_status=endpoint_alpha_status,
        expected_controls=endpoint_controls,
        source_arm="endpoint",
    )
    for control_name in ("norm_random", "random_matched", "pre_error"):
        _record_availability(
            state,
            model=model,
            benchmark="prmbench",
            component=f"control_{control_name}",
            status=(
                "available" if control_name in endpoint_controls else "legacy_not_available"
            ),
            reason=(
                None
                if control_name in endpoint_controls
                else "control was not present in the legacy OLMoE residual protocol"
            ),
        )
    endpoint_dose = 1.0 if endpoint_alpha_status == "explicit_full_strength_1" else None
    headline_anchor = int(model["headline_anchor"])
    residual_site = f"layer_{headline_anchor}.post_moe_residual"
    residual_t = endpoint_t[(residual_site, endpoint_dose)]
    headline: dict[str, Any] = {
        "benchmark": "prmbench",
        "model_id": model_id,
        "display_name": model["display_name"],
        "protocol_grade": model["protocol_grade"],
        "anchor_layer": headline_anchor,
        "residual": _cell_reference(
            benchmark="prmbench",
            model_id=model_id,
            site=residual_site,
            alpha=endpoint_dose,
            alpha_status=endpoint_alpha_status,
        ),
        "same_layer_router": None,
        "same_layer_gap": None,
        "all_router": {},
        "all_router_gaps": {},
    }

    if model_id != "olmoe":
        for anchor in [int(value) for value in model["anchors"]]:
            anchor_residual_site = f"layer_{anchor}.post_moe_residual"
            router_site = f"layer_{anchor}.router_logits"
            _add_gap(
                state,
                model=model,
                benchmark="prmbench",
                comparison_family="same_layer_router",
                anchor_layer=anchor,
                residual_site=anchor_residual_site,
                residual_alpha=endpoint_dose,
                residual_alpha_status=endpoint_alpha_status,
                residual_t=endpoint_t[(anchor_residual_site, endpoint_dose)],
                residual_source_path=str(endpoint.path),
                router_site=router_site,
                router_alpha=endpoint_dose,
                router_alpha_status=endpoint_alpha_status,
                router_t=endpoint_t[(router_site, endpoint_dose)],
                router_source_path=str(endpoint.path),
            )
        headline_router_site = f"layer_{headline_anchor}.router_logits"
        headline["same_layer_router"] = _cell_reference(
            benchmark="prmbench",
            model_id=model_id,
            site=headline_router_site,
            alpha=endpoint_dose,
            alpha_status=endpoint_alpha_status,
        )
        headline["same_layer_gap"] = _gap_row_id(
            benchmark="prmbench",
            model_id=model_id,
            comparison_family="same_layer_router",
            anchor_layer=headline_anchor,
            router_alpha=endpoint_dose,
            router_alpha_status=endpoint_alpha_status,
        )
        _record_availability(
            state,
            model=model,
            benchmark="prmbench",
            component="same_layer_router",
            status="available",
        )
    else:
        missing_gap_id = _gap_row_id(
            benchmark="prmbench",
            model_id=model_id,
            comparison_family="same_layer_router",
            anchor_layer=headline_anchor,
            router_alpha=None,
            router_alpha_status="legacy_not_available",
        )
        state.add_na(
            table="residual_router_gaps",
            row={
                "row_id": missing_gap_id,
                "benchmark": "prmbench",
                "model_id": model_id,
                "display_name": model["display_name"],
                "protocol_grade": model["protocol_grade"],
                "comparison_family": "same_layer_router",
                "site_family": "residual_minus_router",
                "topology": "single_site_residual_vs_same_layer_router",
                "anchor_layer": headline_anchor,
                "alpha": None,
                "alpha_status": "legacy_not_available",
                "effect_kind": "direct",
            },
            reason="legacy_not_available",
        )
        headline["same_layer_gap"] = missing_gap_id
        _record_availability(
            state,
            model=model,
            benchmark="prmbench",
            component="same_layer_router",
            status="legacy_not_available",
            reason="OLMoE residual archive has no same-layer router endpoint rows",
        )

    all_router_spec = model.get("all_router")
    if isinstance(all_router_spec, Mapping):
        bundle_site = f"bundle.router_logits[{int(all_router_spec['registered_site_count'])}]"
        all_router = load_patch_rows(
            str(all_router_spec["predictions"]),
            selected_pair_ids=common_pairs,
            raw_expected_rows=int(all_router_spec["raw_expected_rows"]),
            selected_expected_rows=int(all_router_spec["selected_expected_rows"]),
            expected_sites={bundle_site},
            alpha_status="explicit_dose",
            expected_controls=set(),
        )
        state.register_source(
            all_router.source_record,
            role=f"prmbench:{model_id}:all_router_predictions",
        )
        code_provenance.append(
            _register_arm_sidecars(
                state,
                benchmark="prmbench",
                model_id=model_id,
                arm="all_router",
                arm_spec=all_router_spec,
            )
        )
        all_router_t, _all_router_directions = _add_patch_summaries(
            state,
            model=model,
            benchmark="prmbench",
            patch=all_router,
            alpha_status="explicit_dose",
            expected_controls=set(),
            source_arm="all_router",
        )
        for dose in (0.5, 1.0, 2.0):
            headline["all_router"][str(dose)] = _cell_reference(
                benchmark="prmbench",
                model_id=model_id,
                site=bundle_site,
                alpha=dose,
                alpha_status="explicit_dose",
            )
            headline["all_router_gaps"][str(dose)] = _add_gap(
                state,
                model=model,
                benchmark="prmbench",
                comparison_family="all_router",
                anchor_layer=headline_anchor,
                residual_site=residual_site,
                residual_alpha=endpoint_dose,
                residual_alpha_status=endpoint_alpha_status,
                residual_t=residual_t,
                residual_source_path=str(endpoint.path),
                router_site=bundle_site,
                router_alpha=dose,
                router_alpha_status="explicit_dose",
                router_t=all_router_t[(bundle_site, dose)],
                router_source_path=str(all_router.path),
            )
        _record_availability(
            state,
            model=model,
            benchmark="prmbench",
            component="all_router_alpha_0.5_1_2",
            status="available",
        )
    else:
        for dose in (0.5, 1.0, 2.0):
            missing_id = _gap_row_id(
                benchmark="prmbench",
                model_id=model_id,
                comparison_family="all_router",
                anchor_layer=headline_anchor,
                router_alpha=dose,
                router_alpha_status="legacy_not_available",
            )
            state.add_na(
                table="residual_router_gaps",
                row={
                    "row_id": missing_id,
                    "benchmark": "prmbench",
                    "model_id": model_id,
                    "display_name": model["display_name"],
                    "protocol_grade": model["protocol_grade"],
                    "comparison_family": "all_router",
                    "site_family": "residual_minus_router",
                    "topology": "single_site_residual_vs_simultaneous_all_router",
                    "anchor_layer": headline_anchor,
                    "alpha": None,
                    "alpha_status": "legacy_not_available",
                    "requested_alpha": dose,
                    "residual_site": residual_site,
                    "residual_alpha": endpoint_dose,
                    "residual_alpha_status": endpoint_alpha_status,
                    "router_site": None,
                    "router_alpha": None,
                    "router_alpha_status": "legacy_not_available",
                    "requested_router_alpha": dose,
                    "effect_kind": "direct",
                },
                reason="legacy_not_available",
            )
            headline["all_router_gaps"][str(dose)] = missing_id
        _record_availability(
            state,
            model=model,
            benchmark="prmbench",
            component="all_router_alpha_0.5_1_2",
            status="legacy_not_available",
            reason="Qwen3 corrected endpoint archive has no simultaneous bundle-router arm",
        )

    behavior_spec = model["behavior"]
    if not isinstance(behavior_spec, Mapping):
        raise ValueError(f"{model_id} behavior spec must be a mapping")
    _load_behavior(
        state,
        model=model,
        benchmark="prmbench",
        behavior=behavior_spec,
        selected_pair_ids=behavior_pairs,
    )
    code_provenance.append(
        _register_arm_sidecars(
            state,
            benchmark="prmbench",
            model_id=model_id,
            arm="behavior",
            arm_spec=behavior_spec,
        )
    )

    reviews = model.get("reviews", [])
    expected_counts = model.get("review_expected_check_counts", [])
    if not isinstance(reviews, list) or not isinstance(expected_counts, list):
        raise ValueError(f"{model_id} reviews must be lists")
    if len(reviews) != len(expected_counts):
        raise ValueError(f"{model_id} review paths/count contracts differ")
    if reviews:
        review_arms = ["residual", "all_router"] if model_id == "olmoe" else ["full"]
        for review_path, expected_count, review_arm in zip(
            reviews, expected_counts, review_arms, strict=True
        ):
            _load_review(
                state,
                model=model,
                benchmark="prmbench",
                path=str(review_path),
                expected_check_count=(
                    None if expected_count is None else int(expected_count)
                ),
                review_arm=review_arm,
            )
    else:
        review_id = f"review/prmbench/{model_id}/corrected_endpoint"
        state.tables["review_status"][review_id] = {
            "row_id": review_id,
            "benchmark": "prmbench",
            "model_id": model_id,
            "display_name": model["display_name"],
            "review_arm": "corrected_endpoint",
            "review_path": None,
            "review_sha256": None,
            "review_status": "legacy_no_dedicated_review_for_corrected_rows",
            "passed": None,
            "checks_passed": None,
            "checks_total": None,
            "checks_status": "legacy_not_available",
            "validation_accessed": None,
            "validation_status": "legacy_not_recorded",
        }

    run_root = model.get("run_root")
    validation_status = "legacy_not_recorded"
    provenance_manifest_path: str | None = None
    if run_root is not None:
        provenance_candidate = Path(str(run_root)) / "provenance_manifest.json"
        if provenance_candidate.is_file():
            record = _file_record(provenance_candidate)
            state.register_source(
                record,
                role=f"prmbench:{model_id}:provenance_manifest",
            )
            payload = json.loads(provenance_candidate.read_text(encoding="utf-8"))
            if payload.get("validation_accessed") is not False:
                raise ValueError(
                    f"expected validation_accessed=false: {provenance_candidate}"
                )
            validation_status = "not_accessed"
            provenance_manifest_path = str(provenance_candidate)
            review_id = f"review/prmbench/{model_id}/full"
            if review_id in state.tables["review_status"]:
                state.tables["review_status"][review_id].update(
                    {
                        "validation_accessed": False,
                        "validation_status": "not_accessed_via_provenance_manifest",
                        "validation_provenance_path": str(provenance_candidate),
                    }
                )
    validation_values = {item.get("validation_accessed") for item in code_provenance}
    if provenance_manifest_path is None:
        if False in validation_values and None in validation_values:
            validation_status = "mixed_not_accessed_and_legacy_not_recorded"
        elif False in validation_values:
            validation_status = "not_accessed"

    full_grid = model_id != "olmoe"
    state.model_registry[f"prmbench|{model_id}"] = {
        "registry_id": f"prmbench|{model_id}",
        "benchmark": "prmbench",
        "model_id": model_id,
        "display_name": model["display_name"],
        "model_hf_id": model.get("model_hf_id"),
        "model_revision": model.get("model_revision"),
        "data_revision": (
            None
            if model_id == "qwen3"
            else "5cc7683d0ae5797f84d7aeac0607966f277c39e1"
        ),
        "data_revision_status": (
            "legacy_not_recorded" if model_id == "qwen3" else "recorded"
        ),
        "protocol_grade": model["protocol_grade"],
        "anchors": [int(value) for value in model["anchors"]],
        "common_anchors": [int(value) for value in model.get("common_anchors", [])],
        "extended_only_anchors": sorted(
            set(int(value) for value in model["anchors"])
            - set(int(value) for value in model.get("common_anchors", []))
        ),
        "headline_anchor": headline_anchor,
        "analysis_pair_count": len(common_pairs),
        "pair_set_sha256": pair_ids_sha256(common_pairs),
        "endpoint_raw_row_count": endpoint.raw_row_count,
        "endpoint_selected_row_count": endpoint.selected_row_count,
        "endpoint_alpha_status": endpoint_alpha_status,
        "full_endpoint_grid_eligible": full_grid,
        "full_endpoint_raw_row_estimand_available": full_grid,
        "cross_model_raw_row_pooling_performed": False,
        "pooling_scope": (
            "registered_residual_vs_bundle_router_only"
            if model_id == "olmoe"
            else "model_stratified_common_pair_endpoint_grid"
        ),
        "same_layer_router_status": "available" if full_grid else "legacy_not_available",
        "all_router_status": (
            "available" if isinstance(all_router_spec, Mapping) else "legacy_not_available"
        ),
        "controls_available": sorted(endpoint_controls),
        "review_status": model["review_status"],
        "validation_status": validation_status,
        "provenance_manifest_path": provenance_manifest_path,
        "source_code_provenance": code_provenance,
    }
    state.headline_refs[f"prmbench|{model_id}"] = headline


def _analyze_process_model(
    state: ResultState,
    *,
    raw_model: Mapping[str, Any],
    common_pairs: set[str],
) -> None:
    model = {
        **raw_model,
        "anchors": [int(raw_model["headline_anchor"])],
        "common_anchors": [int(raw_model["headline_anchor"])],
    }
    model_id = str(model["model_id"])
    anchor = int(model["headline_anchor"])
    root = Path(str(model["run_root"]))
    transfer_root = root / "transfer"
    transfer_spec = {
        "predictions": str(transfer_root / "predictions.jsonl"),
        "config": str(transfer_root / "config.resolved.yaml"),
        "metrics": str(transfer_root / "metrics.json"),
    }
    sites = {
        f"layer_{anchor}.post_moe_residual",
        f"layer_{anchor}.router_logits",
    }
    transfer = load_patch_rows(
        transfer_spec["predictions"],
        selected_pair_ids=common_pairs,
        raw_expected_rows=768,
        selected_expected_rows=768,
        expected_sites=sites,
        alpha_status="explicit_full_strength_1",
        expected_controls=set(),
    )
    state.register_source(
        transfer.source_record,
        role=f"processbench:{model_id}:transfer_predictions",
    )
    code_provenance = [
        _register_arm_sidecars(
            state,
            benchmark="processbench",
            model_id=model_id,
            arm="transfer",
            arm_spec=transfer_spec,
        )
    ]
    transfer_t, _directions = _add_patch_summaries(
        state,
        model=model,
        benchmark="processbench",
        patch=transfer,
        alpha_status="explicit_full_strength_1",
        expected_controls=set(),
        source_arm="transfer",
    )
    residual_site = f"layer_{anchor}.post_moe_residual"
    router_site = f"layer_{anchor}.router_logits"
    gap_id = _add_gap(
        state,
        model=model,
        benchmark="processbench",
        comparison_family="same_layer_router",
        anchor_layer=anchor,
        residual_site=residual_site,
        residual_alpha=1.0,
        residual_alpha_status="explicit_full_strength_1",
        residual_t=transfer_t[(residual_site, 1.0)],
        residual_source_path=str(transfer.path),
        router_site=router_site,
        router_alpha=1.0,
        router_alpha_status="explicit_full_strength_1",
        router_t=transfer_t[(router_site, 1.0)],
        router_source_path=str(transfer.path),
    )

    behavior_root = root / "behavior"
    behavior_spec = {
        "predictions": str(behavior_root / "predictions.jsonl"),
        "metrics": str(behavior_root / "metrics.json"),
        "raw_expected_rows": 128,
        "selected_expected_rows": 128,
        "pair_selection": "processbench_common_64",
    }
    _load_behavior(
        state,
        model=model,
        benchmark="processbench",
        behavior=behavior_spec,
        selected_pair_ids=common_pairs,
    )
    code_provenance.append(
        _register_arm_sidecars(
            state,
            benchmark="processbench",
            model_id=model_id,
            arm="behavior",
            arm_spec=behavior_spec,
        )
    )
    review_path = root / "processbench_transfer_full_review.json"
    _load_review(
        state,
        model=model,
        benchmark="processbench",
        path=review_path,
        expected_check_count=int(model["review_expected_check_count"]),
        review_arm="full",
    )
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    if review_payload.get("validation_accessed") is not False:
        raise ValueError(f"ProcessBench review must record validation_accessed=false: {review_path}")

    _record_availability(
        state,
        model=model,
        benchmark="processbench",
        component="same_layer_router",
        status="available",
    )
    _record_availability(
        state,
        model=model,
        benchmark="processbench",
        component="all_router_alpha_0.5_1_2",
        status="not_in_processbench_transfer_protocol",
    )
    _record_availability(
        state,
        model=model,
        benchmark="processbench",
        component="norm_random_random_matched_pre_error_controls",
        status="not_in_processbench_transfer_protocol",
    )

    state.model_registry[f"processbench|{model_id}"] = {
        "registry_id": f"processbench|{model_id}",
        "benchmark": "processbench",
        "model_id": model_id,
        "display_name": model["display_name"],
        "model_hf_id": model.get("model_hf_id"),
        "model_revision": model.get("model_revision"),
        "data_revision": "3bdcd5371ed567559a78f559c01c13a6deee7604",
        "data_revision_status": "recorded",
        "protocol_grade": model["protocol_grade"],
        "comparability_scope": "processbench_three_model_natural_error_transfer",
        "anchors": [anchor],
        "common_anchors": [anchor],
        "extended_only_anchors": [],
        "headline_anchor": anchor,
        "analysis_pair_count": len(common_pairs),
        "pair_set_sha256": pair_ids_sha256(common_pairs),
        "transfer_raw_row_count": transfer.raw_row_count,
        "transfer_selected_row_count": transfer.selected_row_count,
        "endpoint_alpha_status": "explicit_full_strength_1",
        "full_endpoint_grid_eligible": None,
        "full_endpoint_raw_row_estimand_available": None,
        "cross_model_raw_row_pooling_performed": False,
        "same_layer_router_status": "available",
        "all_router_status": "not_in_protocol",
        "controls_available": [],
        "review_status": "pass",
        "validation_status": "not_accessed",
        "source_code_provenance": code_provenance,
        "counterfactual_pair_claimed": False,
    }
    state.headline_refs[f"processbench|{model_id}"] = {
        "benchmark": "processbench",
        "model_id": model_id,
        "display_name": model["display_name"],
        "protocol_grade": model["protocol_grade"],
        "anchor_layer": anchor,
        "residual": _cell_reference(
            benchmark="processbench",
            model_id=model_id,
            site=residual_site,
            alpha=1.0,
            alpha_status="explicit_full_strength_1",
        ),
        "same_layer_router": _cell_reference(
            benchmark="processbench",
            model_id=model_id,
            site=router_site,
            alpha=1.0,
            alpha_status="explicit_full_strength_1",
        ),
        "same_layer_gap": gap_id,
        "all_router": {},
        "all_router_gaps": {},
    }


def _bootstrap_requests(
    state: ResultState,
    *,
    seed: int,
    n_resamples: int,
    alpha: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, tuple[str, ...]], list[VectorRequest]] = defaultdict(list)
    for request in state.requests:
        grouped[(request.group, request.pair_ids)].append(request)
    registry: dict[str, dict[str, Any]] = {}
    indices_by_n: dict[int, np.ndarray] = {}
    for (group, pair_ids), requests in sorted(
        grouped.items(), key=lambda item: (item[0][0], pair_ids_sha256(item[0][1]))
    ):
        n_pairs = len(pair_ids)
        indices = indices_by_n.get(n_pairs)
        if indices is None:
            indices = paired_cluster_bootstrap_indices(
                n_pairs,
                seed=seed,
                n_resamples=n_resamples,
            )
            indices_by_n[n_pairs] = indices
        matrix = np.column_stack([request.values for request in requests])
        summary = paired_cluster_bootstrap_mean_ci(
            matrix,
            indices=indices,
            alpha=alpha,
            simultaneous=False,
            seed=seed,
        )
        if "simultaneous_ci" in summary:
            raise RuntimeError("unified analysis must not emit simultaneous/max-T intervals")
        indices_values_sha256 = hashlib.sha256(
            np.ascontiguousarray(indices, dtype="<i8").tobytes(order="C")
        ).hexdigest()
        for index, request in enumerate(requests):
            row = state.tables[request.table][request.row_id]
            row.update(
                {
                    "estimate": summary["point"][index],
                    "ci_low": summary["ci_low"][index],
                    "ci_high": summary["ci_high"][index],
                    "bootstrap_standard_error": summary["bootstrap_standard_error"][index],
                    "bootstrap_method": summary["method"],
                    "bootstrap_seed": seed,
                    "bootstrap_n_resamples": n_resamples,
                    "confidence_level": 1.0 - alpha,
                    "bootstrap_indices_sha256": summary["indices_sha256"],
                    "bootstrap_indices_values_sha256": indices_values_sha256,
                    "multiplicity_adjustment": None,
                }
            )
        registry_id = f"{group}|pairs-{pair_ids_sha256(pair_ids)}"
        registry[registry_id] = {
            "registry_id": registry_id,
            "analysis_group": group,
            "n_pairs": n_pairs,
            "pair_set_sha256": pair_ids_sha256(pair_ids),
            "n_estimands": len(requests),
            "method": summary["method"],
            "draw_method": summary["draw_method"],
            "rng": summary["rng"],
            "seed": seed,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "indices_sha256": summary["indices_sha256"],
            "indices_values_sha256": indices_values_sha256,
            "indices_sha256_contract": "sha256('<n_resamples>x<n_pairs>\\n' + little_endian_int64_bytes)",
            "indices_values_sha256_contract": "sha256(little_endian_int64_bytes)",
            "simultaneous": False,
            "multiplicity_adjustment": None,
        }
    return registry


def _materialize_headlines(state: ResultState) -> dict[str, dict[str, Any]]:
    def lookup(table: str, row_id: str) -> dict[str, Any]:
        row = state.tables[table][row_id]
        return {
            "row_id": row_id,
            "estimate": row["estimate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "n_pairs": row["n_pairs"],
        }

    output: dict[str, dict[str, Any]] = {}
    for key, refs in sorted(state.headline_refs.items()):
        row = {
            name: value
            for name, value in refs.items()
            if name
            not in {
                "residual",
                "same_layer_router",
                "same_layer_gap",
                "all_router",
                "all_router_gaps",
            }
        }
        residual = refs["residual"]
        row["residual"] = {
            "site": residual["site"],
            "alpha": residual["alpha"],
            "alpha_status": residual["alpha_status"],
            "directional": {
                direction: lookup("directional_effects", row_id)
                for direction, row_id in residual["directional"].items()
            },
            "donor_aligned_t": lookup("donor_aligned_t", residual["donor_aligned_t"]),
        }
        same_router = refs["same_layer_router"]
        if same_router is None:
            row["same_layer_router"] = None
        else:
            row["same_layer_router"] = {
                "site": same_router["site"],
                "alpha": same_router["alpha"],
                "alpha_status": same_router["alpha_status"],
                "directional": {
                    direction: lookup("directional_effects", row_id)
                    for direction, row_id in same_router["directional"].items()
                },
                "donor_aligned_t": lookup(
                    "donor_aligned_t", same_router["donor_aligned_t"]
                ),
            }
        row["same_layer_gap"] = lookup(
            "residual_router_gaps", refs["same_layer_gap"]
        )
        row["all_router"] = {}
        for dose, cell in refs["all_router"].items():
            row["all_router"][dose] = {
                "site": cell["site"],
                "alpha": cell["alpha"],
                "alpha_status": cell["alpha_status"],
                "directional": {
                    direction: lookup("directional_effects", row_id)
                    for direction, row_id in cell["directional"].items()
                },
                "donor_aligned_t": lookup("donor_aligned_t", cell["donor_aligned_t"]),
            }
        row["all_router_gaps"] = {
            dose: lookup("residual_router_gaps", row_id)
            for dose, row_id in refs["all_router_gaps"].items()
        }
        output[key] = row
    return output


def analyze_unified_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Analyze every frozen source named by a validated unified manifest."""

    sources = manifest.get("common_pair_sources")
    expected_hashes = manifest.get("common_pair_sha256")
    if not isinstance(sources, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ValueError("manifest lacks common pair source/hash contracts")
    common_pairs = _derive_pair_ids(str(sources["prmbench"]), unpatched_only=True)
    behavior_pairs = _derive_pair_ids(
        str(sources["prmbench_behavior"]), unpatched_only=False
    )
    process_pairs = _derive_pair_ids(str(sources["processbench"]), unpatched_only=False)
    pair_contracts = {
        "prmbench": common_pairs,
        "prmbench_behavior": behavior_pairs,
        "processbench": process_pairs,
    }
    expected_counts = {"prmbench": 290, "prmbench_behavior": 64, "processbench": 64}
    for name, pair_ids in pair_contracts.items():
        observed_hash = pair_ids_sha256(pair_ids)
        if len(pair_ids) != expected_counts[name]:
            raise ValueError(f"{name} pair count={len(pair_ids)}, expected={expected_counts[name]}")
        if observed_hash != expected_hashes[name]:
            raise ValueError(
                f"{name} pair hash={observed_hash}, expected={expected_hashes[name]}"
            )

    state = ResultState()
    prm_models = manifest.get("prmbench_models")
    process_models = manifest.get("processbench_models")
    if not isinstance(prm_models, list) or not isinstance(process_models, list):
        raise ValueError("manifest model registries must be lists")
    for model in prm_models:
        if not isinstance(model, Mapping):
            raise ValueError("every PRMBench model spec must be a mapping")
        _analyze_prm_model(
            state,
            raw_model=model,
            common_pairs=common_pairs,
            behavior_pairs=behavior_pairs,
        )
    for model in process_models:
        if not isinstance(model, Mapping):
            raise ValueError("every ProcessBench model spec must be a mapping")
        _analyze_process_model(state, raw_model=model, common_pairs=process_pairs)

    bootstrap = manifest.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("manifest lacks bootstrap settings")
    seed = int(bootstrap["seed"])
    n_resamples = int(bootstrap["n_resamples"])
    alpha = float(bootstrap["alpha"])
    bootstrap_registry = _bootstrap_requests(
        state,
        seed=seed,
        n_resamples=n_resamples,
        alpha=alpha,
    )
    headlines = _materialize_headlines(state)
    tables = {
        name: dict(sorted(rows.items())) for name, rows in sorted(state.tables.items())
    }
    results = {
        "schema_version": 1,
        "analysis_id": manifest["analysis_id"],
        "generated_at": manifest["generated_at"],
        "execution_mode": "cpu_only_secondary_analysis",
        "estimand_contract": {
            "primary_metric": PRIMARY_METRIC,
            "units": PRIMARY_UNITS,
            "metric_definition": "INCORRECT mean_logprob minus CORRECT mean_logprob",
            "raw_direct_effect": "patched minus same-pair receiver unpatched",
            "forward_direction": "clean_to_corrupt",
            "reverse_direction": "corrupt_to_clean",
            "donor_aligned_t": "0.5*(corrupt_to_clean-clean_to_corrupt)",
            "positive_t_interpretation": "donor_aligned_restoration",
            "residual_router_gap": "donor_aligned_T_residual-minus-donor_aligned_T_router",
            "processbench_role_aliases": {
                "clean": "matched_correct",
                "corrupt": "first_error",
                "clean_to_corrupt": "matched_correct_to_first_error",
                "corrupt_to_clean": "first_error_to_matched_correct",
                "counterfactual_pair_claimed": False,
            },
        },
        "bootstrap": {
            "method": "paired_cluster_percentile",
            "rng": "numpy.random.Generator(PCG64)",
            "draw_method": "paired_cluster_shared_indices",
            "seed": seed,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "t_and_gap_built_from_pair_rows": True,
            "t_interval_assembled_from_directional_intervals": False,
            "simultaneous_or_max_t_interval": False,
            "multiplicity_adjustment": None,
            "population_p_values": False,
        },
        "pair_contracts": {
            name: {
                "n_pairs": len(pair_ids),
                "pair_set_sha256": pair_ids_sha256(pair_ids),
                "allowlist_source": str(sources[name]),
                "allowlist_derivation": (
                    "sorted unique pair_id from unpatched prediction rows"
                    if name == "prmbench"
                    else "sorted unique pair_id from behavior/transfer prediction rows"
                ),
            }
            for name, pair_ids in pair_contracts.items()
        },
        "comparability_contract": {
            "modern": [
                "gemma4",
                "nemotron3_nano",
                "nemotron35_lightning",
                "qwen36_moe",
            ],
            "extended": ["qwen3", "gpt_oss_20b"],
            "legacy-compatible": ["olmoe"],
            "modern_endpoint_grid": "three registered anchors by three site families",
            "extended_endpoint_grid": "five registered anchors; common-anchor flags retained",
            "olmoe_scope": "registered layer-8 residual versus simultaneous bundle-router only",
            "cross_model_raw_row_pooling_performed": False,
            "model_population_inference_claimed": False,
            "fixed_model_set": True,
        },
        "pod_b_alpha1_control_join": {
            "status": "semantic_fix_verified",
            "scope": (
                "endpoint direct alpha=1 rows only, for internal control lookup against "
                "controls whose legacy rows do not record alpha"
            ),
            "non_alpha1_rows_normalized": False,
            "predictions_or_direct_effects_changed": False,
        },
        "model_registry": dict(sorted(state.model_registry.items())),
        "bootstrap_registry": dict(sorted(bootstrap_registry.items())),
        **tables,
        "headline_results": headlines,
        "source_files": dict(sorted(state.input_sources.items())),
        "outputs": {
            "unified_results": "unified_results.json",
            "availability": "availability.csv",
            "bootstrap_registry": "bootstrap_registry.csv",
            "headline_results": "headline_results.csv",
            "directional_effects": "directional_effects.csv",
            "control_effects": "control_effects.csv",
            "donor_aligned_t": "donor_aligned_t.csv",
            "residual_router_gaps": "residual_router_gaps.csv",
            "native_manipulation": "native_manipulation.csv",
            "behavior_metrics": "behavior_metrics.csv",
            "identity_checks": "identity_checks.csv",
            "model_registry": "model_registry.csv",
            "review_status": "review_status.csv",
            "source_files": "source_files.csv",
            "pair_estimands": "pair_estimands.csv.gz",
            "report": "tables.md",
            "provenance": "provenance.json",
            "checksums": "SHA256SUMS",
            "success_marker": "SUCCESS",
        },
        "row_counts": {
            **{name: len(rows) for name, rows in tables.items()},
            "model_registry": len(state.model_registry),
            "bootstrap_registry": len(bootstrap_registry),
            "headline_result_models": len(headlines),
            "headline_results_csv": sum(
                1 + len(set(row["all_router"]) | set(row["all_router_gaps"]))
                for row in headlines.values()
            ),
            "pair_estimands": len(state.pair_estimands),
            "source_files": len(state.input_sources),
        },
        "claim_limits": [
            "Results apply to the frozen benchmark pairs, registered sites, and fixed non-random model set analyzed here.",
            "No universal model-population effect or population p-value is reported.",
            "Qwen3.6 has directional nonzero router effects in registered router analyses; results do not support router blindness or zero mediation.",
            "OLMoE is restricted to its common residual-versus-bundle-router estimand and is not part of a pooled full-endpoint raw-row analysis.",
            "ProcessBench matched-correct/first-error pairs are natural error boundaries, not counterfactual edits.",
        ],
    }
    return results, state.pair_estimands, dict(sorted(state.input_sources.items()))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("CSV rows must be non-empty")
    preferred = [
        "row_id",
        "registry_id",
        "summary_row_id",
        "benchmark",
        "model_id",
        "display_name",
        "protocol_grade",
        "anchor_layer",
        "common_anchor",
        "site",
        "site_family",
        "topology",
        "alpha",
        "alpha_status",
        "direction",
        "direction_semantic",
        "effect_kind",
        "control_name",
        "comparison_family",
        "estimate",
        "ci_low",
        "ci_high",
        "n_pairs",
        "pair_id",
        "value",
    ]
    all_fields = {str(key) for row in rows for key in row}
    fieldnames = [name for name in preferred if name in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    return fieldnames


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fieldnames = _csv_fieldnames(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    tmp.replace(path)


def _write_csv_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty compressed CSV: {path.name}")
    fieldnames = _csv_fieldnames(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="raise",
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _csv_value(value) for key, value in row.items()})
    tmp.replace(path)


def _format_interval(payload: Mapping[str, Any]) -> str:
    estimate = payload.get("estimate")
    low = payload.get("ci_low")
    high = payload.get("ci_high")
    if estimate is None or low is None or high is None:
        return "NA"
    return f"{float(estimate):.6f} [{float(low):.6f}, {float(high):.6f}]"


def _headline_csv_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    def interval_fields(prefix: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {
                f"{prefix}_row_id": None,
                f"{prefix}_estimate": None,
                f"{prefix}_ci_low": None,
                f"{prefix}_ci_high": None,
                f"{prefix}_n_pairs": None,
            }
        return {
            f"{prefix}_row_id": payload.get("row_id"),
            f"{prefix}_estimate": payload.get("estimate"),
            f"{prefix}_ci_low": payload.get("ci_low"),
            f"{prefix}_ci_high": payload.get("ci_high"),
            f"{prefix}_n_pairs": payload.get("n_pairs"),
        }

    def component_fields(
        prefix: str, payload: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if payload is None:
            return {
                **interval_fields(f"{prefix}_clean_to_corrupt", None),
                **interval_fields(f"{prefix}_corrupt_to_clean", None),
                **interval_fields(f"{prefix}_t", None),
            }
        directional = payload["directional"]
        return {
            **interval_fields(
                f"{prefix}_clean_to_corrupt", directional["clean_to_corrupt"]
            ),
            **interval_fields(
                f"{prefix}_corrupt_to_clean", directional["corrupt_to_clean"]
            ),
            **interval_fields(f"{prefix}_t", payload["donor_aligned_t"]),
        }

    rows: list[dict[str, Any]] = []
    for registry_id, headline in results["headline_results"].items():
        common = {
            "headline_id": registry_id,
            "benchmark": headline["benchmark"],
            "model_id": headline["model_id"],
            "display_name": headline["display_name"],
            "protocol_grade": headline["protocol_grade"],
            "anchor_layer": headline["anchor_layer"],
            **component_fields("residual", headline["residual"]),
        }
        rows.append(
            {
                **common,
                "row_id": (
                    f"headline/{headline['benchmark']}/{headline['model_id']}"
                    "/same-layer-router"
                ),
                "comparison_family": "same_layer_router",
                "residual_alpha": headline["residual"]["alpha"],
                "residual_alpha_status": headline["residual"]["alpha_status"],
                "router_alpha": (
                    headline["same_layer_router"]["alpha"]
                    if headline["same_layer_router"] is not None
                    else None
                ),
                "router_alpha_status": (
                    headline["same_layer_router"]["alpha_status"]
                    if headline["same_layer_router"] is not None
                    else "legacy_not_available"
                ),
                **component_fields("router", headline["same_layer_router"]),
                **interval_fields("gap", headline["same_layer_gap"]),
            }
        )
        doses = sorted(
            set(headline["all_router"]) | set(headline["all_router_gaps"]),
            key=float,
        )
        for dose in doses:
            rows.append(
                {
                    **common,
                    "row_id": (
                        f"headline/{headline['benchmark']}/{headline['model_id']}"
                        f"/all-router/alpha-{_alpha_label(float(dose), 'explicit_dose')}"
                    ),
                    "comparison_family": "all_router",
                    "residual_alpha": headline["residual"]["alpha"],
                    "residual_alpha_status": headline["residual"]["alpha_status"],
                    "router_alpha": (
                        headline["all_router"][dose]["alpha"]
                        if dose in headline["all_router"]
                        else float(dose)
                    ),
                    "router_alpha_status": (
                        headline["all_router"][dose]["alpha_status"]
                        if dose in headline["all_router"]
                        else "legacy_not_available"
                    ),
                    **component_fields("router", headline["all_router"].get(dose)),
                    **interval_fields("gap", headline["all_router_gaps"].get(dose)),
                }
            )
    return rows


def _report_markdown(results: Mapping[str, Any]) -> str:
    headlines = results["headline_results"]
    lines = [
        "# RouteVerify unified CPU reanalysis",
        "",
        "All effects use the mean-logprob margin `INCORRECT - CORRECT`. Raw direct effects are patched minus the same-pair receiver baseline. `T = 0.5 * (corrupt_to_clean - clean_to_corrupt)`; positive values indicate donor-aligned restoration. Every T and residual-router gap interval is bootstrapped directly from pair rows (50,000 PCG64 resamples, seed 0), never assembled from marginal directional intervals.",
        "",
        "## PRMBench headline",
        "",
        "| Model | Grade | Anchor | Residual T [95% CI] | Same-layer router T [95% CI] | Same-layer gap [95% CI] | All-router alpha=1 T [95% CI] | All-router gap [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in headlines.items():
        if not key.startswith("prmbench|"):
            continue
        same = row["same_layer_router"]
        all_router = row["all_router"].get("1.0")
        all_gap = row["all_router_gaps"].get("1.0")
        lines.append(
            "| {display} | {grade} | {anchor} | {residual} | {same} | {same_gap} | {all_t} | {all_gap} |".format(
                display=row["display_name"],
                grade=row["protocol_grade"],
                anchor=row["anchor_layer"],
                residual=_format_interval(row["residual"]["donor_aligned_t"]),
                same=(
                    _format_interval(same["donor_aligned_t"]) if same is not None else "NA"
                ),
                same_gap=_format_interval(row["same_layer_gap"]),
                all_t=(
                    _format_interval(all_router["donor_aligned_t"])
                    if all_router is not None
                    else "NA"
                ),
                all_gap=_format_interval(all_gap) if all_gap is not None else "NA",
            )
        )
    lines.extend(
        [
            "",
            "## PRMBench headline directional direct effects",
            "",
            "Each cell is the raw patched-minus-unpatched direct effect with its pair-bootstrap 95% interval.",
            "",
            "| Model | Residual clean-to-corrupt | Residual corrupt-to-clean | Same-layer router clean-to-corrupt | Same-layer router corrupt-to-clean | n pairs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in headlines.items():
        if not key.startswith("prmbench|"):
            continue
        residual = row["residual"]
        same = row["same_layer_router"]
        lines.append(
            "| {display} | {residual_forward} | {residual_reverse} | {router_forward} | {router_reverse} | {n} |".format(
                display=row["display_name"],
                residual_forward=_format_interval(
                    residual["directional"]["clean_to_corrupt"]
                ),
                residual_reverse=_format_interval(
                    residual["directional"]["corrupt_to_clean"]
                ),
                router_forward=(
                    _format_interval(same["directional"]["clean_to_corrupt"])
                    if same is not None
                    else "NA"
                ),
                router_reverse=(
                    _format_interval(same["directional"]["corrupt_to_clean"])
                    if same is not None
                    else "NA"
                ),
                n=residual["donor_aligned_t"]["n_pairs"],
            )
        )
    lines.extend(
        [
            "",
            "## PRMBench simultaneous all-router dose strata",
            "",
            "All-router alpha 0.5, 1, and 2 remain separate from same-layer router interventions. Missing legacy doses are explicit NA rows.",
            "",
            "| Model | Alpha | Router clean-to-corrupt | Router corrupt-to-clean | Router T [95% CI] | Residual-router gap [95% CI] | n pairs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in headlines.items():
        if not key.startswith("prmbench|"):
            continue
        doses = sorted(
            set(row["all_router"]) | set(row["all_router_gaps"]), key=float
        )
        for dose in doses:
            router = row["all_router"].get(dose)
            gap = row["all_router_gaps"].get(dose)
            lines.append(
                "| {display} | {dose} | {router_forward} | {router_reverse} | {router_t} | {gap} | {n} |".format(
                    display=row["display_name"],
                    dose=dose,
                    router_forward=(
                        _format_interval(
                            router["directional"]["clean_to_corrupt"]
                        )
                        if router is not None
                        else "NA"
                    ),
                    router_reverse=(
                        _format_interval(
                            router["directional"]["corrupt_to_clean"]
                        )
                        if router is not None
                        else "NA"
                    ),
                    router_t=(
                        _format_interval(router["donor_aligned_t"])
                        if router is not None
                        else "NA"
                    ),
                    gap=_format_interval(gap) if gap is not None else "NA",
                    n=(router["donor_aligned_t"]["n_pairs"] if router is not None else "NA"),
                )
            )
    lines.extend(
        [
            "",
            "## ProcessBench natural-error transfer",
            "",
            "ProcessBench uses `clean=matched_correct` and `corrupt=first_error` as internal aliases; it does not claim counterfactual edits.",
            "",
            "| Model | Anchor | Residual clean-to-corrupt | Residual corrupt-to-clean | Residual T [95% CI] | Router clean-to-corrupt | Router corrupt-to-clean | Router T [95% CI] | Gap [95% CI] | n pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in headlines.items():
        if not key.startswith("processbench|"):
            continue
        lines.append(
            "| {display} | {anchor} | {residual_forward} | {residual_reverse} | {residual} | {router_forward} | {router_reverse} | {router} | {gap} | {n} |".format(
                display=row["display_name"],
                anchor=row["anchor_layer"],
                residual_forward=_format_interval(
                    row["residual"]["directional"]["clean_to_corrupt"]
                ),
                residual_reverse=_format_interval(
                    row["residual"]["directional"]["corrupt_to_clean"]
                ),
                residual=_format_interval(row["residual"]["donor_aligned_t"]),
                router_forward=_format_interval(
                    row["same_layer_router"]["directional"]["clean_to_corrupt"]
                ),
                router_reverse=_format_interval(
                    row["same_layer_router"]["directional"]["corrupt_to_clean"]
                ),
                router=_format_interval(row["same_layer_router"]["donor_aligned_t"]),
                gap=_format_interval(row["same_layer_gap"]),
                n=row["residual"]["donor_aligned_t"]["n_pairs"],
            )
        )
    lines.extend(
        [
            "",
            "## Comparability and claim boundary",
            "",
            "- Modern PRMBench: Gemma 4, Nemotron-3 Nano, Nemotron-3.5 Lightning, and Qwen3.6 share the three-anchor by three-site endpoint contract.",
            "- Extended PRMBench: Qwen3 and GPT-OSS report five anchors, with common-anchor flags retained.",
            "- Legacy-compatible: OLMoE enters only the registered layer-8 residual versus simultaneous bundle-router estimand; it is not pooled into a full endpoint grid.",
            "- Same-layer router and simultaneous all-router alpha 0.5/1/2 are distinct rows. Legacy missing alpha, reviewer, topology, or manipulation fields remain explicit NA/legacy values.",
            "- Norm-random, random-matched, and pre-error controls are retained as both raw receiver-aligned effects and direct-minus-control contrasts. Pre-error uses its exact 285-pair eligible subset.",
            "- Behavior rows use the shared 64-pair comparison set; Qwen3 additionally retains its native 580-pair archive metric as a separate `native_archive` row.",
            "- These are frozen benchmark-pair results for a fixed, non-random model set. No universal model-population conclusion or population p-value is reported.",
            "- Directional nonzero router effects, including Qwen3.6, preclude claims of router blindness or zero mediation.",
            "",
            "## Machine-readable keys",
            "",
            "`unified_results.json` stores keyed dictionaries at `directional_effects[row_id]`, `control_effects[row_id]`, `donor_aligned_t[row_id]`, `residual_router_gaps[row_id]`, `native_manipulation[row_id]`, and `headline_results[benchmark|model_id]`. The same `row_id` appears in the corresponding summary CSV and in the audit-oriented `pair_estimands.csv.gz`.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_unified_artifacts(
    *,
    results: Mapping[str, Any],
    pair_estimands: Sequence[Mapping[str, Any]],
    source_files: Mapping[str, Mapping[str, Any]],
    manifest_path: str | Path,
    output_dir: str | Path,
    entrypoint_path: str | Path,
) -> dict[str, str]:
    """Atomically write the versioned JSON/CSV/Markdown/provenance bundle."""

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite unified analysis output: {output}")
    immutable_archive_root = Path("inputs/archives").resolve(strict=False)
    if output == immutable_archive_root or immutable_archive_root in output.parents:
        raise ValueError(
            "derived output must be outside the immutable inputs/archives archive root"
        )
    for source_path in source_files:
        source = Path(source_path).resolve(strict=True)
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("derived output must be outside every immutable source path")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        atomic_write_json(staging / "unified_results.json", results)
        table_files = {
            "model_registry": "model_registry.csv",
            "directional_effects": "directional_effects.csv",
            "control_effects": "control_effects.csv",
            "donor_aligned_t": "donor_aligned_t.csv",
            "residual_router_gaps": "residual_router_gaps.csv",
            "native_manipulation": "native_manipulation.csv",
            "behavior_metrics": "behavior_metrics.csv",
            "identity_checks": "identity_checks.csv",
            "review_status": "review_status.csv",
            "availability": "availability.csv",
            "bootstrap_registry": "bootstrap_registry.csv",
        }
        for table_name, filename in table_files.items():
            table = results[table_name]
            _write_csv(staging / filename, list(table.values()))
        _write_csv(staging / "headline_results.csv", _headline_csv_rows(results))
        _write_csv_gzip(staging / "pair_estimands.csv.gz", list(pair_estimands))
        _write_csv(staging / "source_files.csv", list(source_files.values()))
        (staging / "tables.md").write_text(_report_markdown(results), encoding="utf-8")

        manifest = Path(manifest_path).resolve(strict=True)
        entrypoint = Path(entrypoint_path).resolve(strict=True)
        from routeverify.analysis import bootstrap as bootstrap_module

        analysis_sources = [
            manifest,
            Path(__file__).resolve(strict=True),
            Path(str(bootstrap_module.__file__)).resolve(strict=True),
            entrypoint,
        ]
        output_hashes_before_provenance = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        provenance = {
            "schema_version": 1,
            "analysis_id": results["analysis_id"],
            "generated_at": results["generated_at"],
            "execution_mode": "cpu_only_secondary_analysis",
            "analysis_source_files": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in analysis_sources
            ],
            "immutable_input_files": list(source_files.values()),
            "derived_output_hashes_before_provenance": output_hashes_before_provenance,
            "pair_contracts": results["pair_contracts"],
            "bootstrap": results["bootstrap"],
            "pod_b_alpha1_control_join": results["pod_b_alpha1_control_join"],
            "access_and_mutation_contract": {
                "model_weights_loaded": False,
                "gpu_used": False,
                "validation_rows_accessed": False,
                "validation_split_manifests_opened": False,
                "source_archives_written": False,
                "source_predictions_or_metrics_overwritten": False,
                "narrative_summaries_used_as_numeric_inputs": False,
            },
            "excluded_incomplete_archive": (
                "inputs/archives/runpod_gpt_oss_full_20260809T110315Z"
            ),
            "forbidden_optional_analyses_not_computed": [
                "TOST",
                "SESOI",
                "max-T",
                "audit cascade",
                "model-population p-value",
            ],
            "claim_limits": results["claim_limits"],
        }
        atomic_write_json(staging / "provenance.json", provenance)
        sums = []
        for path in sorted(staging.iterdir()):
            if path.is_file() and path.name not in {"SHA256SUMS", "SUCCESS"}:
                sums.append(f"{_sha256(path)}  {path.name}\n")
        (staging / "SHA256SUMS").write_text("".join(sums), encoding="utf-8")
        (staging / "SUCCESS").write_text("", encoding="utf-8")
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": str(output),
        "results": str(output / "unified_results.json"),
        "report": str(output / "tables.md"),
        "provenance": str(output / "provenance.json"),
        "sha256sums": str(output / "SHA256SUMS"),
        "success": str(output / "SUCCESS"),
    }
