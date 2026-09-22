"""Versioned discovery-only window interventions, independent of the legacy CLI.

Run ``python -m routeverify.window_interventions --help``.  Preflight loads only
local tokenizer files.  Scoring modes require an explicit mode, a successful
tokenizer preflight, and (for full) a passing checkpoint smoke from this protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from routeverify.config import load_model_config
from routeverify.data.prompts import STEP_BINARY_CANDIDATES, step_binary, tokenize_render
from routeverify.data.schema import TraceExample, stable_hash, trace_example_from_json
from routeverify.eval.verification import verification_margin
from routeverify.instrumentation.sites import (
    CaptureSpec,
    MultiPositionPatchSpec,
    PatchBundle,
    site_key,
)
from routeverify.models.registry import create_adapter
from routeverify.window_geometry import build_pair_windows, calibrate_window_k, common_prefix_length

PROTOCOL = "receiver_window_core_v1_20260907"
WINDOWS = ("last_token", "suffix_k", "divergence_k", "full_step")
ARMS = ("router_logits", "moe_output", "post_moe_residual", "all_router")
CONTROLS = ("paired", "self", "matched", "norm_random")
ROLES = ("clean", "corrupt")
CANDIDATES = list(STEP_BINARY_CANDIDATES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_set_default(value: Any) -> list[Any]:
    """Normalize unordered loader metadata without accepting other non-JSON types."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda item: json.dumps(
            item, sort_keys=True, allow_nan=False, default=_json_set_default))
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False,
                                    default=_json_set_default) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class WindowPair:
    pair_id: str
    clean: TraceExample
    corrupt: TraceExample
    component: str


@dataclass(frozen=True)
class PreparedTrace:
    prompt: str
    input_ids: tuple[int, ...]
    step_span: tuple[int, int]
    provenance: dict[str, Any]


def load_discovery_pairs(config: dict[str, Any], pairs_path: Path) -> list[WindowPair]:
    """Validate all 580 content hashes against the frozen discovery selection.

    The normalized input must contain discovery only.  A pilot, duplicate, extra
    pair, or mismatched content fails before any checkpoint is loaded.
    """
    selection = json.loads(Path(config["selection_manifest"]).read_text())
    expected = {row["pair_id"]: row for row in selection["pairs"]}
    if len(expected) != 290 or len(selection["pairs"]) != 290:
        raise ValueError("window core requires the frozen 290-pair discovery selection")
    split = json.loads(Path(config["split_manifest"]).read_text())
    discovery = set(split["discovery_group_ids"])
    source_rows = json.loads(Path(config["source_manifest"]).read_text())["pairs"]
    components = {row["pair_id"]: row["source_problem_id"] for row in source_rows}
    grouped: dict[str, dict[str, TraceExample]] = defaultdict(dict)
    with pairs_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            pair_id, role = row.get("pair_id"), row.get("variant")
            if pair_id not in expected or role not in ROLES:
                raise ValueError("input must contain only clean/corrupt frozen discovery examples")
            example = trace_example_from_json(row)
            if example.group_id not in discovery:
                raise ValueError(f"pair is outside discovery: {pair_id}")
            if role in grouped[pair_id]:
                raise ValueError(f"duplicate example: {pair_id}/{role}")
            if example.content_hash != expected[pair_id][f"{role}_content_hash"]:
                raise ValueError(f"frozen content hash mismatch: {pair_id}/{role}")
            grouped[pair_id][role] = example
    if set(grouped) != set(expected) or any(set(roles) != set(ROLES) for roles in grouped.values()):
        raise ValueError(f"need all 290 discovery pairs, received {len(grouped)} (pilot rejected)")
    result = []
    for pair_id in sorted(expected):
        clean, corrupt = (grouped[pair_id][role] for role in ROLES)
        boundary = corrupt.first_error
        if boundary is None or boundary >= len(clean.steps):
            raise ValueError(f"invalid paired error step: {pair_id}")
        if clean.problem != corrupt.problem or clean.steps[:boundary] != corrupt.steps[:boundary]:
            raise ValueError(f"paired causal prefixes differ before the error step: {pair_id}")
        if pair_id not in components:
            raise ValueError(f"missing source component: {pair_id}")
        result.append(WindowPair(pair_id, clean, corrupt, str(components[pair_id])))
    return result


def matched_pairs(pairs: list[WindowPair], seed: int) -> dict[str, str]:
    """Old priority rules, now explicitly excluding the source component."""
    matches = {}
    for receiver in pairs:
        eligible = [p for p in pairs if p.component != receiver.component]
        if not eligible:
            raise ValueError(f"no cross-source donor for {receiver.pair_id}")
        def rank(donor: WindowPair, receiver: WindowPair = receiver) -> tuple[int, int, str]:
            return (
                int(donor.corrupt.error_type != receiver.corrupt.error_type),
                abs(int(donor.corrupt.first_error) - int(receiver.corrupt.first_error)),
                stable_hash({"seed": seed, "current": receiver.pair_id, "candidate": donor.pair_id}),
            )
        matches[receiver.pair_id] = min(eligible, key=rank).pair_id
    return matches


def prepare_trace(adapter: Any, example: TraceExample, boundary: int) -> PreparedTrace:
    raw = step_binary(example, step_index=boundary)
    formatter = getattr(adapter, "chat_completion_render", None)
    rendered = formatter(raw) if callable(formatter) else raw
    tokenized = tokenize_render(rendered, sample_id=example.sample_id, tokenizer=adapter.tokenizer)
    span = tokenized.step_token_spans[boundary]
    ids = tuple(int(item) for item in tokenized.input_ids)
    # Check actual candidate-scoring token coordinates, including Harmony renders.
    for candidate in CANDIDATES:
        inputs, target_start, _target_end, _target_ids = adapter._encode_prompt_candidate(raw.prompt, candidate)
        actual = inputs["input_ids"][0].tolist()
        if actual[:span[1]] != list(ids[:span[1]]) or span[1] > target_start:
            raise ValueError(f"candidate scoring and step token coordinates disagree: {example.sample_id}")
    return PreparedTrace(raw.prompt, ids, span, {
        "sample_id": example.sample_id,
        "content_hash": example.content_hash,
        "input_ids_sha256": stable_hash(list(ids), length=64),
        "rendered_prompt_sha256": hashlib.sha256(rendered.prompt.encode()).hexdigest(),
        "step_span": list(span),
        "input_format": "chat" if callable(formatter) else "raw",
    })


def prepare_panel(adapter: Any, pairs: list[WindowPair]) -> dict[tuple[str, str], PreparedTrace]:
    return {
        (pair.pair_id, role): prepare_trace(adapter, getattr(pair, role), int(pair.corrupt.first_error))
        for pair in pairs for role in ROLES
    }


def pair_geometries(prepared: dict, pair_id: str, role: str, k: int) -> dict:
    other = "corrupt" if role == "clean" else "clean"
    receiver, donor = prepared[pair_id, role], prepared[pair_id, other]
    return build_pair_windows(receiver.input_ids, donor.input_ids,
                              receiver_step_span=receiver.step_span,
                              donor_step_span=donor.step_span, k=k)


def map_donor_window(receiver_positions: tuple[int, ...], donor_positions: tuple[int, ...]) -> tuple[int, ...]:
    """Endpoint-aligned monotone nearest mapping, one vector per receiver token."""
    if not receiver_positions or not donor_positions:
        raise ValueError("cannot map an empty window")
    if len(receiver_positions) == 1:
        return (donor_positions[-1],)
    numerator = np.arange(len(receiver_positions)) * (len(donor_positions) - 1)
    indices = (2 * numerator + len(receiver_positions) - 1) // (2 * (len(receiver_positions) - 1))
    return tuple(donor_positions[int(i)] for i in indices)


def matched_donor_window(prepared: dict, pair_id: str, role: str, name: str, k: int) -> tuple[int, ...]:
    """Take the actual matched role's window without changing receiver targets.

    Its counterpart defines divergence, but its unused length does not disqualify
    a sufficiently long donor.  An empty tuple explicitly means unavailable.
    """
    trace = prepared[pair_id, role]
    start, end = trace.step_span
    if name == "full_step":
        return tuple(range(start, end))
    if name == "last_token":
        return (end - 1,)
    if end - start < k:
        return ()
    if name == "suffix_k":
        return tuple(range(end - k, end))
    if name != "divergence_k":
        raise ValueError(f"unknown window: {name}")
    other = prepared[pair_id, "corrupt" if role == "clean" else "clean"]
    divergence = common_prefix_length(trace.input_ids, other.input_ids)
    if trace.input_ids == other.input_ids or not start <= divergence <= end:
        return ()
    anchor = min(divergence, end - 1)
    left = min(max(anchor - (k - 1) // 2, start), end - k)
    return tuple(range(left, left + k))


def replacement_array(receiver: np.ndarray, donor: np.ndarray, *, control: str,
                      seed: int, salt: str, alpha: float = 1.0) -> np.ndarray:
    if receiver.shape != donor.shape or receiver.ndim != 3:
        raise ValueError("window replacements require equal [batch,positions,features] shapes")
    if control == "self" or alpha == 0:
        return receiver.copy()
    delta = donor.astype(np.float64) - receiver.astype(np.float64)
    if control == "norm_random":
        rng_seed = int(stable_hash({"seed": seed, "salt": salt}, length=16), 16)
        noise = np.random.default_rng(rng_seed).standard_normal(delta.shape)
        noise_norm = np.linalg.norm(noise, axis=-1, keepdims=True)
        delta = noise * np.divide(np.linalg.norm(delta, axis=-1, keepdims=True), noise_norm,
                                  out=np.zeros_like(noise_norm), where=noise_norm > 0)
    return (receiver.astype(np.float64) + alpha * delta).astype(receiver.dtype)


def replacement_metrics(receiver: np.ndarray, replacement: np.ndarray, identity_indices: list[int]) -> dict:
    delta = replacement.astype(np.float64) - receiver.astype(np.float64)
    token_l2 = np.linalg.norm(delta, axis=-1).reshape(-1)
    return {
        "measurement_scope": "requested_replacement_before_native_dtype_conversion",
        "target_shape": list(receiver.shape), "delta_l2": float(np.linalg.norm(delta)),
        "delta_l2_per_position": token_l2.tolist(),
        "nonzero_position_count": int(np.count_nonzero(token_l2)),
        "identity_position_max_abs_delta": (
            float(np.max(np.abs(delta[:, identity_indices, :]))) if identity_indices else None
        ),
        "diagnostic_only": True,
    }


def arm_sites(adapter: Any, anchor: int) -> dict[str, tuple]:
    sites = adapter.discover_layers()
    layout = adapter.moe_block_layout()
    if anchor not in layout.router_block_indices or anchor == layout.total_block_count - 1:
        raise ValueError("anchor must be an interior routed decoder block")
    result = {kind: tuple(s for s in sites if s.kind == kind and s.layer_index == anchor)
              for kind in ARMS[:3]}
    result["all_router"] = tuple(s for s in sites if s.kind == "router_logits")
    if any(len(result[kind]) != 1 for kind in ARMS[:3]):
        raise ValueError("anchor must expose each of the three interfaces exactly once")
    if len(result["all_router"]) != layout.router_block_count:
        raise ValueError("all-router must cover every routed block")
    return result


def expected_forward_counts(n_pairs: int, *, matched_capture_pairs: int | None = None) -> dict:
    """Upper bound before geometry exclusions; two candidate forwards per scoring row."""
    capture_pairs = n_pairs if matched_capture_pairs is None else matched_capture_pairs
    treatment_rows = n_pairs * len(WINDOWS) * len(ARMS) * len(ROLES) * len(CONTROLS)
    return {"baseline_forwards": n_pairs * 8, "capture_forwards": capture_pairs * 4,
            "treatment_scoring_rows": treatment_rows, "treatment_scoring_forwards": treatment_rows * 2,
            "total_forwards": n_pairs * 8 + capture_pairs * 4 + treatment_rows * 2,
            "note": "upper bound; NA windows skipped, overlapping named windows retained; no D or all-MoE bundle"}


def tokenizer_only_adapter(model_config: Any, tokenizer_path: str | None = None) -> Any:
    adapter = create_adapter(model_config.alias)
    adapter.config = model_config
    native_loader = getattr(adapter, "load_tokenizer", None)
    if callable(native_loader):
        # Native-tokenizer adapters own their serialization and local asset
        # checks. In particular, Mistral tekken is not an HF AutoTokenizer.
        native_loader(model_config, tokenizer_path=tokenizer_path)
        adapter._input_device = lambda: "cpu"
        return adapter
    from transformers import AutoTokenizer
    adapter.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or model_config.model_id,
        revision=None if tokenizer_path else model_config.tokenizer_revision,
        cache_dir=str(model_config.cache_dir) if model_config.cache_dir else None,
        local_files_only=True, trust_remote_code=model_config.trust_remote_code,
    )
    adapter._input_device = lambda: "cpu"
    template = adapter.tokenizer.get_chat_template()
    expected_template = model_config.expected_chat_template_sha256
    if expected_template and hashlib.sha256(template.encode()).hexdigest() != expected_template:
        raise ValueError("tokenizer chat template does not match the pinned model contract")
    return adapter


def tokenizer_provenance(adapter: Any, model_config: Any, tokenizer_path: str | None) -> dict:
    """Hash the actual small local assets; never infer identity from a path label."""
    native_provenance = getattr(adapter, "tokenizer_provenance", None)
    if callable(native_provenance):
        provenance = native_provenance(tokenizer_path=tokenizer_path)
        if not isinstance(provenance, dict):
            raise TypeError("adapter tokenizer_provenance must return a dictionary")
        return provenance
    from huggingface_hub import hf_hub_download
    assets = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "tokenizer.model", "vocab.json", "merges.txt", "chat_template.jinja"):
        if tokenizer_path:
            path = Path(tokenizer_path) / name
        else:
            try:
                path = Path(hf_hub_download(model_config.model_id, name,
                    revision=model_config.tokenizer_revision,
                    cache_dir=model_config.cache_dir, local_files_only=True))
            except OSError:
                continue
        if path.is_file():
            assets[name] = sha256(path)
    return {"tokenizer_class": type(adapter.tokenizer).__name__, "asset_sha256": assets,
            "chat_template_sha256": hashlib.sha256(adapter.tokenizer.get_chat_template().encode()).hexdigest(),
            "requested_revision": model_config.tokenizer_revision}


def preflight(config: dict, pairs: list[WindowPair], prepared: dict, model_config: Any,
              sources: dict) -> dict:
    lengths = [(prepared[p.pair_id, "clean"].step_span[1] - prepared[p.pair_id, "clean"].step_span[0],
                prepared[p.pair_id, "corrupt"].step_span[1] - prepared[p.pair_id, "corrupt"].step_span[0])
               for p in pairs]
    calibration = calibrate_window_k(lengths, candidates=tuple(config["k_candidates"]),
                                     min_pair_coverage=float(config["min_pair_coverage"]))
    k = int(calibration["selected_k"])
    geometry = [{"pair_id": p.pair_id, "role": role,
                 "windows": {name: geom.to_dict() for name, geom in pair_geometries(prepared, p.pair_id, role, k).items()}}
                for p in pairs for role in ROLES]
    matching = matched_pairs(pairs, int(config["seed"]))
    treatment_rows = 0
    unavailable_matched_directions = 0
    for row in geometry:
        for name, window in row["windows"].items():
            if not window["applicable"]:
                continue
            treatment_rows += len(ARMS) * 3  # paired, self, per-token norm-random
            other = "corrupt" if row["role"] == "clean" else "clean"
            matched_available = bool(matched_donor_window(prepared, matching[row["pair_id"]], other, name, k))
            treatment_rows += len(ARMS) * int(matched_available)
            unavailable_matched_directions += int(not matched_available)
    counts = {"baseline_forwards": len(pairs) * 8, "capture_forwards": len(pairs) * 4,
              "treatment_scoring_rows": treatment_rows, "treatment_scoring_forwards": treatment_rows * 2,
              "total_forwards": len(pairs) * 12 + treatment_rows * 2,
              "matched_donor_unavailable_direction_windows": unavailable_matched_directions,
              "scope": "full discovery, alpha=1 core only; exact before effects, named overlaps retained"}
    return {"protocol": PROTOCOL, "mode": "tokenizer_cpu_preflight", "passed": True,
            "checkpoint_forward_run": False, "gpu_smoke_passed": False,
            "model": model_config.model_dump(mode="json"), "sources": sources,
            "calibration": calibration, "n_pairs": len(pairs),
            "matching": matching,
            "geometry": geometry, "forward_upper_bound": expected_forward_counts(len(pairs)),
            "planned_full_counts": counts,
            "alpha": 1.0, "arms": list(ARMS), "controls": list(CONTROLS),
            "identity_semantics": "same absolute token position with identical causal token prefix",
            "full_step_semantics": "every receiver error-step token; endpoint-aligned nearest donor mapping",
            "norm_random_semantics": "Gaussian delta with per-position L2 equal to paired donor delta",
            "no_validation_access": True}


class CaptureCache:
    """One candidate-specific union capture per pair role, persisted within this run."""
    def __init__(self, adapter: Any, prepared: dict, sites: tuple, directory: Path):
        self.adapter, self.prepared, self.sites = adapter, prepared, sites
        self.directory = directory
        self.forward_count = 0
        directory.mkdir(parents=True, exist_ok=True)

    def get(self, pair_id: str, role: str) -> dict[str, dict[str, np.ndarray]]:
        trace = self.prepared[pair_id, role]
        key = stable_hash({"pair_id": pair_id, "role": role}, length=32)
        paths = {candidate: self.directory / f"{key}_{candidate}.npz" for candidate in CANDIDATES}
        if not all(path.exists() for path in paths.values()):
            positions = tuple(range(*trace.step_span))
            captures = self.adapter.capture_candidate_activations(trace.prompt, CANDIDATES,
                CaptureSpec(sites=self.sites, token_positions=positions, dtype="float32"))
            self.forward_count += len(CANDIDATES)
            for candidate, capture in captures.items():
                arrays = {name: np.asarray(array) for name, array in capture.arrays.items()}
                for name, array in arrays.items():
                    if array.ndim != 3 or array.shape[1] != len(positions) or not np.isfinite(array).all():
                        raise ValueError(f"invalid union capture {pair_id}/{role}/{candidate}/{name}")
                np.savez(paths[candidate], **arrays)
        result = {}
        for candidate, path in paths.items():
            with np.load(path, allow_pickle=False) as archive:
                result[candidate] = dict(archive)
        return result


def diagnostic_gate(diagnostics: dict, expected_sites: tuple, *, require_dispatch: bool = True) -> list[str]:
    errors = []
    expected = {site_key(s) for s in expected_sites}
    router_keys = {site_key(s) for s in expected_sites if s.kind == "router_logits"}
    for candidate in CANDIDATES:
        diag = diagnostics.get(candidate, {})
        if set(diag.get("registered_sites", ())) != expected or not diag.get("all_registered_hooks_fired"):
            errors.append(f"{candidate}: missing or unexpected patch hooks")
        if router_keys and require_dispatch:
            dispatch = diag.get("dispatch_by_site", {})
            if set(dispatch) != router_keys or not all(dispatch[k].get("downstream_dispatch_verified") for k in router_keys):
                errors.append(f"{candidate}: native downstream router dispatch is unverified")
    return errors


def score_payload(scores: dict) -> dict:
    payload = {"margin_mean": verification_margin(scores, normalized=True),
               "margin_total": verification_margin(scores, normalized=False),
               "scores": {k: asdict(v) for k, v in scores.items()}}
    if not np.isfinite([payload["margin_mean"], payload["margin_total"]]).all():
        raise ValueError("nonfinite candidate margin")
    return payload


def score_identity_delta(left: dict, right: dict) -> float:
    """Check every candidate, preventing cancelling errors in their margin."""
    return max(abs(left["scores"][c][field] - right["scores"][c][field])
               for c in CANDIDATES for field in ("mean_logprob", "total_logprob"))


def backend_resource_metrics(adapter: Any) -> dict | None:
    """Read optional worker-owned counters without relabeling driver memory."""
    provider = getattr(adapter, "resource_metrics", None)
    if not callable(provider):
        return None
    metrics = provider()
    if not isinstance(metrics, dict) or not isinstance(metrics.get("scope"), str):
        raise TypeError("adapter resource_metrics must return a dictionary with an explicit scope")
    return metrics


def execute(adapter: Any, config: dict, pairs: list[WindowPair], prepared: dict,
            report: dict, output: Path, *, mode: str, loaded_checkpoint: bool = False) -> dict:
    """Score each named window and control with atomic bundles and shared baselines."""
    arms = arm_sites(adapter, int(config["anchor"]))
    union = {site_key(site): site for arm in arms.values() for site in arm}
    cache = CaptureCache(adapter, prepared, tuple(union.values()), output / "capture_union")
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    selected = sorted(pairs, key=lambda p: stable_hash({"seed": config["seed"], "pair_id": p.pair_id}))
    if mode == "smoke":
        selected = selected[:int(config["smoke_pairs"])]
    k, seed = int(report["calibration"]["selected_k"]), int(config["seed"])
    rows: list[dict] = []
    skipped: list[dict] = []
    errors: list[str] = []
    baseline: dict[tuple[str, str], dict] = {}
    scoring_forwards = 0
    start = time.monotonic()
    with (output / "rows.jsonl").open("w") as row_file:
        def emit(row: dict) -> None:
            rows.append(row)
            row_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            row_file.flush()
        for pair in selected:
            pair_id = pair.pair_id
            for role in ROLES:
                trace = prepared[pair_id, role]
                for treatment in ("unpatched", "unpatched_rerun"):
                    scored = score_payload(adapter.score_candidates(trace.prompt, CANDIDATES))
                    scoring_forwards += 2
                    emit({"pair_id": pair_id, "component": pair.component, "receiver_role": role,
                          "treatment": treatment, "window": None, "arm": None, **scored})
                    if treatment == "unpatched":
                        baseline[pair_id, role] = scored
                    elif score_identity_delta(scored, baseline[pair_id, role]) > float(config["identity_atol"]):
                        errors.append(f"{pair_id}/{role}: baseline repeat drift")
                receiver_captures = cache.get(pair_id, role)
                other = "corrupt" if role == "clean" else "clean"
                paired_captures = cache.get(pair_id, other)
                donor_id = report["matching"][pair_id]
                matched_captures = cache.get(donor_id, other)
                geometries = pair_geometries(prepared, pair_id, role, k)
                for name, geom in geometries.items():
                    if geom.status != "ok":
                        skipped.append({"pair_id": pair_id, "role": role, "window": name, "geometry": geom.to_dict()})
                        continue
                    target_positions = geom.receiver_positions
                    indices = [pos - trace.step_span[0] for pos in target_positions]
                    identities = [i for i, pos in enumerate(target_positions) if pos in geom.identity_positions]
                    for arm, sites in arms.items():
                        controls = list(CONTROLS)
                        if mode == "smoke":
                            controls.extend(["alpha_zero", "alpha_half"])
                            if arm in {"router_logits", "all_router"}:
                                controls.append("forced_reroute")
                        for control in controls:
                            alpha = {"alpha_zero": 0.0, "alpha_half": 0.5}.get(control, 1.0)
                            donor_positions = geom.donor_positions
                            donor_trace = prepared[pair_id, other]
                            donor_captures = paired_captures
                            actual_donor_id = pair_id
                            if control == "self":
                                donor_positions, donor_trace, donor_captures = target_positions, trace, receiver_captures
                            elif control == "matched":
                                matched_positions = matched_donor_window(prepared, donor_id, other, name, k)
                                if not matched_positions:
                                    skipped.append({"pair_id": pair_id, "role": role, "window": name,
                                                    "arm": arm, "control": control, "reason": "matched donor window unavailable",
                                                    "donor_pair_id": donor_id})
                                    continue
                                donor_positions = map_donor_window(target_positions, matched_positions)
                                donor_trace, donor_captures = prepared[donor_id, other], matched_captures
                                actual_donor_id = donor_id
                            donor_indices = [pos - donor_trace.step_span[0] for pos in donor_positions]
                            bundles, magnitudes = {}, {}
                            for candidate in CANDIDATES:
                                patches, candidate_magnitudes = [], {}
                                for site in sites:
                                    key = site_key(site)
                                    receiver = receiver_captures[candidate][key][:, indices, :]
                                    donor = donor_captures[candidate][key][:, donor_indices, :]
                                    replacement = replacement_array(receiver, donor, control=control, seed=seed, alpha=alpha,
                                        salt=f"{pair_id}/{role}/{name}/{arm}/{candidate}/{key}")
                                    patches.append(MultiPositionPatchSpec(site=site, token_positions=target_positions,
                                                                        replacement=replacement,
                                                                        metadata={"forced_reroute": control == "forced_reroute"}))
                                    candidate_magnitudes[key] = replacement_metrics(receiver, replacement,
                                        identities if control in {"paired", "norm_random"} else [])
                                bundles[candidate] = PatchBundle(tuple(patches), metadata={"protocol": PROTOCOL, "window": name})
                                magnitudes[candidate] = candidate_magnitudes
                            scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
                                trace.prompt, CANDIDATES, bundles_by_candidate=bundles)
                            scoring_forwards += 2
                            scored = score_payload(scores)
                            gate_errors = diagnostic_gate(diagnostics, sites)
                            if control == "forced_reroute":
                                for candidate in CANDIDATES:
                                    manipulation = diagnostics[candidate].get("manipulation_by_site", {})
                                    if any(manipulation.get(site_key(s), {}).get("topk_changed_rate") != 1.0 for s in sites):
                                        gate_errors.append(f"{candidate}: forced routing did not change every target membership")
                            errors.extend(f"{pair_id}/{role}/{name}/{arm}/{control}: {error}" for error in gate_errors)
                            identity_delta = score_identity_delta(scored, baseline[pair_id, role])
                            if control in {"self", "alpha_zero"} and identity_delta > float(config["identity_atol"]):
                                errors.append(f"{pair_id}/{role}/{name}/{arm}: self identity delta {identity_delta}")
                            emit({"pair_id": pair_id, "component": pair.component, "receiver_role": role,
                                  "donor_role": role if control == "self" else other,
                                  "donor_pair_id": actual_donor_id,
                                  "donor_component": pair_lookup[actual_donor_id].component,
                                  "treatment": control, "window": name, "arm": arm, "alpha": alpha,
                                  "geometry": geom.to_dict(), "actual_donor_positions": list(donor_positions),
                                  "baseline_margin_mean": baseline[pair_id, role]["margin_mean"],
                                  "replacement_diagnostics": magnitudes, "patch_diagnostics": diagnostics,
                                  "replacement_diagnostics_apply_before_forced_override": control == "forced_reroute",
                                  "input_provenance": trace.provenance, **scored})
            write_json(output / "progress.json", {"completed_pair_ids": sorted({r["pair_id"] for r in rows}),
                                                   "scoring_forwards": scoring_forwards,
                                                   "capture_forwards": cache.forward_count})
            print(f"{mode}: completed {pair_id}; scoring forwards={scoring_forwards}", flush=True)
            if errors and mode == "full":
                break
    summary = {"protocol": PROTOCOL, "mode": mode, "passed": not errors,
               "checkpoint_forward_run": loaded_checkpoint,
               "gpu_smoke_passed": mode == "smoke" and loaded_checkpoint and not errors,
               "execution_scope": "checkpoint" if loaded_checkpoint else "tiny_or_injected_adapter",
               "runtime_model_class": type(adapter.model).__name__,
               "model": report["model"], "sources": report["sources"],
               "calibration": report["calibration"], "anchor": config["anchor"],
               "n_pairs": len({r["pair_id"] for r in rows}), "selected_pair_ids": [p.pair_id for p in selected],
               "scoring_forwards": scoring_forwards, "capture_forwards": cache.forward_count,
               "elapsed_seconds": time.monotonic() - start, "errors": errors,
               "skipped_cells": skipped, "effect_direction_is_not_a_gate": True,
               "no_validation_access": True}
    if mode == "smoke":
        covered = {(r["window"], r["arm"], r["receiver_role"])
                   for r in rows if r["treatment"] == "paired"}
        required = {(w, a, role) for w in WINDOWS for a in ARMS for role in ROLES}
        if covered != required:
            errors.append("technical smoke did not exercise every window/arm/direction")
            summary["passed"] = summary["gpu_smoke_passed"] = False
    if loaded_checkpoint:
        import torch
        # Worker-process backends (for example vLLM) allocate outside this
        # process. These counters must not be presented as their model footprint.
        summary["cuda_peak_memory_scope"] = "calling_process_only; worker allocations excluded"
        summary["cuda_peak_memory"] = [{
            "device": index, "allocated_bytes": torch.cuda.max_memory_allocated(index),
            "reserved_bytes": torch.cuda.max_memory_reserved(index),
        } for index in range(torch.cuda.device_count())]
        try:
            worker_metrics = backend_resource_metrics(adapter)
        except Exception as exc:  # Optional telemetry never gates the scientific run.
            summary["backend_resource_metrics_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if worker_metrics is not None:
                summary["backend_resource_metrics"] = worker_metrics
    summary["observed_forwards_per_second_including_capture_io"] = (
        (scoring_forwards + cache.forward_count) / max(summary["elapsed_seconds"], 1e-9)
    )
    write_json(output / "execution_summary.json", summary)
    write_json(output / "analysis.json", analyze_rows(rows, seed=seed))
    return summary


def _estimate(values: list[float], rng: Any, bootstrap: int) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "ci95": None}
    draws = np.empty(bootstrap)
    for start in range(0, bootstrap, 256):
        count = min(256, bootstrap - start)
        draws[start:start + count] = a[rng.integers(0, len(a), size=(count, len(a)))].mean(axis=1)
    return {"n": len(a), "mean": float(a.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def analyze_rows(rows: list[dict], *, seed: int = 0, bootstrap: int = 5000) -> dict:
    """Paired T and direct within-pair window/gap differences; no mediation claim."""
    grouped: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        if row.get("window") and row["treatment"] in CONTROLS:
            key = (row["pair_id"], row["window"], row["arm"], row["treatment"])
            role = row["receiver_role"]
            if role in grouped[key]:
                raise ValueError(f"duplicate analysis row: {key}/{role}")
            direction = 1 if role == "clean" else -1
            grouped[key][role] = direction * (row["margin_mean"] - row["baseline_margin_mean"])
    pair_t = {key: float(np.mean(list(roles.values()))) for key, roles in grouped.items() if set(roles) == set(ROLES)}
    pair_gaps = {}
    for (pair_id, window, arm, control), value in pair_t.items():
        if arm != "post_moe_residual":
            continue
        for comparison in ("router_logits", "moe_output", "all_router"):
            key = (pair_id, window, comparison, control)
            if key in pair_t:
                pair_gaps[pair_id, window, f"residual_minus_{comparison}", control] = value - pair_t[key]
    rng = np.random.default_rng(seed)
    estimates = []
    for label, mapping in (("T", pair_t), ("gap", pair_gaps)):
        cells: dict[tuple, dict[str, float]] = defaultdict(dict)
        for (pair_id, window, arm, control), value in mapping.items():
            cells[window, arm, control][pair_id] = value
        for (window, arm, control), values in sorted(cells.items()):
            estimates.append({"estimand": label, "window": window, "arm": arm, "control": control,
                              **_estimate(list(values.values()), rng, bootstrap)})
        for arm, control in sorted({(a, c) for _w, a, c in cells}):
            for left_index, left in enumerate(WINDOWS):
                for right in WINDOWS[left_index + 1:]:
                    lvalues, rvalues = cells.get((left, arm, control), {}), cells.get((right, arm, control), {})
                    common = sorted(set(lvalues) & set(rvalues))
                    estimates.append({"estimand": f"within_pair_{label}_difference", "left": left, "right": right,
                                      "arm": arm, "control": control,
                                      **_estimate([lvalues[p] - rvalues[p] for p in common], rng, bootstrap)})
    return {"protocol": PROTOCOL, "primary_margin": "mean_logprob_INCORRECT_minus_CORRECT",
            "T_definition": "mean of clean receiver margin increase and corrupt receiver margin decrease",
            "resampling": "paired pair bootstrap on direct differences; two-sided pointwise percentile intervals",
            "bootstrap_replicates": bootstrap, "seed": seed, "estimates": estimates,
            "pair_estimands": [{"pair_id": p, "window": w, "arm": a, "control": c, "T": value}
                               for (p, w, a, c), value in sorted(pair_t.items())],
            "incomplete_direction_cells": sum(set(value) != set(ROLES) for value in grouped.values()),
            "diagnostics_interpretation": "descriptive direction, replacement size and dispatch; not serial mediation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="580 normalized discovery-only examples")
    parser.add_argument("--output", type=Path, required=True, help="new empty versioned output directory")
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--tokenizer-path", help="optional local exact-revision tokenizer directory")
    parser.add_argument("--smoke-report", type=Path, help="required passing same-protocol report for full")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("protocol") != PROTOCOL or config.get("partition") != "discovery":
        parser.error("only this versioned discovery window protocol is supported")
    if int(config["smoke_pairs"]) <= 0 or float(config["identity_atol"]) <= 0:
        parser.error("smoke_pairs and identity_atol must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("refusing non-empty output; use a fresh versioned directory")
    args.output.mkdir(parents=True, exist_ok=True)
    model_config = load_model_config(Path(config["model"]))
    sources = {"pairs_sha256": sha256(args.pairs), "config_sha256": sha256(args.config),
               "model_config_sha256": sha256(Path(config["model"])),
               **{key + "_sha256": sha256(Path(config[key])) for key in ("selection_manifest", "split_manifest", "source_manifest")}}
    pairs = load_discovery_pairs(config, args.pairs)
    tokenizer_adapter = tokenizer_only_adapter(model_config, args.tokenizer_path)
    sources["tokenizer"] = tokenizer_provenance(tokenizer_adapter, model_config, args.tokenizer_path)
    package = Path(__file__).resolve().parent
    source_files = [package / "window_interventions.py", package / "window_geometry.py",
                    package / "instrumentation/sites.py", package / "instrumentation/patching.py",
                    package / "eval/candidate_scoring.py", package / "data/prompts.py",
                    *sorted((package / "models").glob("*.py"))]
    sources["implementation_sha256"] = {str(p.relative_to(package)): sha256(p) for p in source_files}
    prepared = prepare_panel(tokenizer_adapter, pairs)
    report = preflight(config, pairs, prepared, model_config, sources)
    write_json(args.output / "preflight.json", report)
    if args.mode == "preflight":
        print(json.dumps({key: report[key] for key in ("mode", "passed", "n_pairs", "calibration", "forward_upper_bound")}))
        return
    if args.mode == "full":
        if args.smoke_report is None:
            parser.error("full requires --smoke-report from this checkpoint/window protocol")
        smoke = json.loads(args.smoke_report.read_text())
        if (smoke.get("protocol") != PROTOCOL or not smoke.get("gpu_smoke_passed")
                or smoke.get("model") != report["model"] or smoke.get("sources") != sources
                or smoke.get("calibration") != report["calibration"]):
            parser.error("smoke report does not qualify this exact model/data/config/calibration")
    import torch
    if not torch.cuda.is_available():
        parser.error("checkpoint smoke/full requires CUDA; CPU tiny checks use the test suite")
    adapter = create_adapter(model_config.alias)
    load_updates: dict[str, Any] = {"load_weights": True}
    if args.tokenizer_path and callable(getattr(adapter, "load_tokenizer", None)):
        load_updates["tokenizer_path"] = args.tokenizer_path
    adapter.load(model_config.model_copy(update=load_updates))
    runtime_prepared = prepare_panel(adapter, pairs)
    if runtime_prepared != prepared:
        raise RuntimeError("loaded model tokenizer/render differs from CPU preflight")
    write_json(args.output / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "command": sys.argv,
               "runtime_contract": getattr(adapter, "loading_info", None)})
    write_json(args.output / "model_discovery.json", adapter.discovery_report.to_json())
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    result = execute(adapter, config, pairs, prepared, report, args.output, mode=args.mode, loaded_checkpoint=True)
    if not result["passed"]:
        raise SystemExit("technical gates failed; see execution_summary.json")


if __name__ == "__main__":
    main()
