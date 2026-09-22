"""Full-step, all-routed-layer fixed-receiver external-donor-role D.

Four donor-role cells are freshly scored under one loaded adapter per receiver
pair. Completed pairs are atomic recovery units; an interrupted pair is rerun
in full. No historical T row or single-token capture can supply a D cell.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from routeverify.config import load_model_config
from routeverify.data.schema import stable_hash
from routeverify.eval.candidate_scoring import CandidateScore
from routeverify.instrumentation.sites import (
    LayerSite,
    MultiPositionPatchSpec,
    PatchBundle,
    site_key,
)
from routeverify.models.registry import create_adapter
from routeverify.window_interventions import (
    CANDIDATES,
    ROLES,
    CaptureCache,
    PreparedTrace,
    WindowPair,
    diagnostic_gate,
    load_discovery_pairs,
    map_donor_window,
    prepare_panel,
    replacement_metrics,
    score_identity_delta,
    score_payload,
    sha256,
    tokenizer_only_adapter,
    tokenizer_provenance,
    write_json,
)

PROTOCOL = "window_donor_role_v1_20260908"
SCORER = "candidate_mean_logprob_INCORRECT_minus_CORRECT_v1"
GEOMETRY = "full_receiver_step_endpoint_aligned_monotone_nearest_v1"


def router_sites(adapter: Any, expected_count: int | None = None) -> tuple[LayerSite, ...]:
    """Use the adapter's routed collection, including its final routed block."""
    sites = tuple(s for s in adapter.discover_layers() if s.kind == "router_logits")
    layout = adapter.moe_block_layout()
    indices = [s.layer_index for s in sites]
    if (len(sites) != layout.router_block_count or len(set(indices)) != len(sites)
            or set(indices) != set(layout.router_block_indices)):
        raise ValueError("all-router sites must equal every adapter-defined routed layer")
    if expected_count is not None and len(sites) != expected_count:
        raise ValueError("routed layer count differs from the scientific config")
    return tuple(sorted(sites, key=lambda s: s.layer_index))


def trace_identity(adapter: Any, trace: PreparedTrace, component: str) -> dict[str, Any]:
    """Store actual candidate-conditioned input IDs and score coordinates."""
    inputs_by_candidate = {}
    for candidate in CANDIDATES:
        inputs, start, end, targets = adapter._encode_prompt_candidate(trace.prompt, candidate)
        ids = inputs["input_ids"][0].tolist()
        targets = [int(t) for t in targets]
        if (not 0 <= trace.step_span[0] <= trace.step_span[1] <= start < end <= len(ids)
                or ids[start:end] != targets or len(targets) != end - start
                or ids[:trace.step_span[1]] != list(trace.input_ids[:trace.step_span[1]])):
            raise ValueError("candidate input/target positions disagree with receiver step")
        inputs_by_candidate[candidate] = {
            "input_ids": [int(t) for t in ids],
            "input_ids_sha256": stable_hash([ids], length=64),
            "input_shape": [1, len(ids)],
            "target_token_start": int(start), "target_token_end": int(end),
            "target_ids": targets,
        }
    return {
        **trace.provenance, "source_component": component,
        "step_span": list(trace.step_span), "candidate_inputs": inputs_by_candidate,
        "input_sha256": stable_hash(inputs_by_candidate, length=64),
    }


def validate_matching(pairs: list[WindowPair], matching: dict[str, str]) -> None:
    lookup = {p.pair_id: p for p in pairs}
    if len(lookup) != len(pairs) or set(matching) != set(lookup):
        raise ValueError("fixed matching must cover exactly the complete donor pool")
    for receiver, donor in matching.items():
        if donor not in lookup or receiver == donor or lookup[receiver].component == lookup[donor].component:
            raise ValueError(f"matching must use an external cross-source donor: {receiver}")


def read_matching(path: Path, pairs: list[WindowPair]) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        matching = {r["pair_id"]: r["donor_pair_id"] for r in payload}
        if len(matching) != len(payload):
            raise ValueError("duplicate receiver in authoritative matching")
    elif isinstance(payload, dict):
        if "edges" in payload:
            payload = payload["edges"]
        elif "matching" in payload:
            payload = payload["matching"]
        if isinstance(payload, list):
            matching = {r["pair_id"]: r["donor_pair_id"] for r in payload}
            if len(matching) != len(payload):
                raise ValueError("duplicate receiver in authoritative matching")
        else:
            matching = payload
    else:
        raise ValueError("invalid matching manifest")
    validate_matching(pairs, matching)
    return matching


def pair_eligibility(pair: WindowPair, donor: WindowPair,
                     prepared: dict[tuple[str, str], PreparedTrace]) -> dict[str, Any]:
    reasons = []
    for member in (pair, donor):
        for role in ROLES:
            trace = prepared[member.pair_id, role]
            start, end = trace.step_span
            if not 0 <= start < end <= len(trace.input_ids):
                reasons.append(f"empty_or_invalid_step:{member.pair_id}:{role}")
    return {"pair_id": pair.pair_id, "source_component": pair.component,
            "component": pair.component, "donor_pair_id": donor.pair_id,
            "donor_component": donor.component, "eligible": not reasons,
            "reason_codes": reasons}


def planned_counts(selected: list[WindowPair], matching: dict[str, str],
                   eligibility: list[dict[str, Any]], *, mode: str = "full") -> dict[str, Any]:
    eligible = {r["pair_id"] for r in eligibility if r["eligible"]}
    capture_pairs = eligible | {matching[p] for p in eligible}
    extra = 2 if mode == "smoke" else 0
    rows = (10 + extra) * len(eligible)
    return {
        "receiver_pairs": len(selected), "eligible_pairs": len(eligible),
        "source_components": len({p.component for p in selected if p.pair_id in eligible}),
        "ineligible_pairs": len(selected) - len(eligible),
        "d_primary_rows": 4 * len(eligible),
        "baseline_rerun_rows": 4 * len(eligible), "self_identity_rows": 2 * len(eligible),
        "forced_dispatch_rows": extra * len(eligible), "scoring_rows": rows,
        "scoring_forwards": 2 * rows, "capture_pairs": len(capture_pairs),
        "capture_forwards": 4 * len(capture_pairs),
        "total_forwards": 2 * rows + 4 * len(capture_pairs),
        "scope": "fresh invocation; capture union includes selected receivers and fixed donors",
    }


def scientific_contract(config: dict[str, Any], pairs: list[WindowPair],
                        prepared: dict[tuple[str, str], PreparedTrace],
                        identities: dict[tuple[str, str], dict[str, Any]],
                        report: dict[str, Any], *, target_sites: list[str]) -> dict[str, Any]:
    """Bind only claim-defining inputs; paths and wrapper bytes are not gates."""
    model = report.get("model", {})
    checkpoint_keys = ("alias", "model_id", "revision", "tokenizer_revision", "weights",
                       "require_native_mxfp4", "harmony_reasoning_effort")
    return {
        "protocol": PROTOCOL, "scorer": SCORER, "geometry": GEOMETRY,
        "estimand": "D_r=m(r<-error_j)-m(r<-clean_j); D_common=(D_clean+D_error)/2",
        "window": "full_step", "arm": "all_router", "alpha": 1.0,
        "checkpoint": {k: model[k] for k in checkpoint_keys if k in model},
        "candidate_strings": CANDIDATES, "matching": report["matching"],
        "pairs": [{"pair_id": p.pair_id, "component": p.component,
                   "roles": {role: {
                       "sample_id": identities[p.pair_id, role]["sample_id"],
                       "content_hash": identities[p.pair_id, role].get("content_hash"),
                       "input_sha256": identities[p.pair_id, role]["input_sha256"],
                       "step_span": list(prepared[p.pair_id, role].step_span),
                   } for role in ROLES}} for p in sorted(pairs, key=lambda p: p.pair_id)],
        "target_sites": target_sites, "identity_atol": float(config["identity_atol"]),
        "eligibility": "nonempty valid receiver clean/error and fixed donor clean/error step spans",
        "analysis": {"unit": "pair", "resampling": "receiver_source_component_pair_weighted",
                     "seed": int(config.get("analysis_seed", 20260908)),
                     "replicates": int(config.get("bootstrap", 50000)), "confidence_level": 0.95},
    }


def _runtime(adapter: Any) -> dict[str, Any]:
    import torch
    import transformers
    source = inspect.getsourcefile(type(adapter))
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "adapter_class": type(adapter).__name__, "model_class": type(adapter.model).__name__,
            "adapter_source_sha256": sha256(Path(source)) if source else None,
            "runner_source_sha256": sha256(Path(__file__)),
            "loading_info": getattr(adapter, "loading_info", None),
            "command": sys.argv,
            "note": "runtime provenance; semantic repairs require a new execution version/output"}


def _validated_score(scores: dict[str, CandidateScore], identity: dict[str, Any]) -> dict[str, Any]:
    if set(scores) != set(CANDIDATES):
        raise ValueError("both candidate scores are required")
    for candidate, score in scores.items():
        targets = identity["candidate_inputs"][candidate]["target_ids"]
        if (score.token_count != len(targets) or score.token_count < 1
                or not np.isfinite([score.mean_logprob, score.total_logprob]).all()
                or not np.isclose(score.total_logprob, score.mean_logprob * score.token_count)):
            raise ValueError("invalid candidate score or changed candidate tokenization")
    payload = score_payload(scores)
    payload["margin_incorrect_minus_correct_mean"] = payload["margin_mean"]
    payload["margin_incorrect_minus_correct_total"] = payload["margin_total"]
    return payload


def _validate_recovered_unit(unit: dict[str, Any], binding: dict[str, Any],
                             eligibility: dict[str, Any],
                             identities: dict[tuple[str, str], dict[str, Any]],
                             sites: tuple[LayerSite, ...], *, mode: str, tolerance: float) -> None:
    """Revalidate saved observations without executing or trusting row counts alone."""
    pair_id = eligibility["pair_id"]
    if any(unit.get(k) != binding[k] for k in ("contract_id", "execution_version")):
        raise ValueError("saved pair has a different contract or execution version")
    if unit.get("pair_id") != pair_id or unit.get("technical_passed") is not True:
        raise ValueError("saved pair is not a complete passing recovery unit")
    saved_eligibility = unit.get("eligibility", {})
    if any(saved_eligibility.get(k) != v for k, v in eligibility.items()):
        raise ValueError("saved eligibility differs from current claim inputs")
    if (not isinstance(unit.get("run_id"), str) or not unit["run_id"]
            or saved_eligibility.get("run_id") != unit["run_id"]
            or any(saved_eligibility.get(k) != binding[k] for k in ("contract_id", "execution_version"))):
        raise ValueError("saved eligibility mixes loaded run or execution identity")
    rows = unit.get("rows", [])
    if not eligibility["eligible"]:
        if rows:
            raise ValueError("ineligible saved pair contains score rows")
        return
    required = {(r, t, d) for r in ROLES for t, d in (
        ("unpatched", None), ("unpatched_rerun", None), ("self_patch", r),
        ("donor_role", "clean"), ("donor_role", "corrupt"))}
    if mode == "smoke":
        required |= {(r, "forced_reroute", r) for r in ROLES}
    indexed = {(row.get("receiver_variant"), row.get("treatment"), row.get("donor_variant")): row
               for row in rows}
    if len(indexed) != len(rows) or set(indexed) != required:
        raise ValueError("saved pair has incomplete or duplicate controls/D cells")
    keys = [site_key(s) for s in sites]
    for (role, treatment, donor_role), row in indexed.items():
        if (row.get("pair_id") != pair_id or row.get("run_id") != unit.get("run_id")
                or row.get("source_component") != eligibility["source_component"]
                or any(row.get(k) != binding[k] for k in ("contract_id", "execution_version"))):
            raise ValueError("saved score row mixes pair, loaded run, source or execution identity")
        identity = identities[pair_id, role]
        metadata = row.get("metadata", {})
        if metadata.get("receiver_identity") != identity:
            raise ValueError("saved receiver input identity differs from current candidate inputs")
        if set(row.get("scores", {})) != set(CANDIDATES):
            raise ValueError("saved row is missing candidate scores")
        for candidate in CANDIDATES:
            score = row["scores"][candidate]
            n = len(identity["candidate_inputs"][candidate]["target_ids"])
            if (score.get("token_count") != n
                    or not np.isfinite([score["mean_logprob"], score["total_logprob"]]).all()
                    or not np.isclose(score["total_logprob"], n * score["mean_logprob"])):
                raise ValueError("saved candidate scores disagree with scoring contract")
        for suffix, field in (("mean", "mean_logprob"), ("total", "total_logprob")):
            margin = row["scores"]["INCORRECT"][field] - row["scores"]["CORRECT"][field]
            if not np.isclose(row[f"margin_incorrect_minus_correct_{suffix}"], margin):
                raise ValueError("saved margin differs from raw candidate scores")
        if donor_role is None:
            if row.get("donor_pair_id") is not None or metadata.get("donor_identity") is not None:
                raise ValueError("saved baseline unexpectedly names a donor")
        else:
            donor_id = eligibility["donor_pair_id"] if treatment == "donor_role" else pair_id
            if row.get("donor_pair_id") != donor_id or metadata.get("donor_identity") != identities[donor_id, donor_role]:
                raise ValueError("saved donor differs from fixed assignment/role input")
            targets = list(range(*identity["step_span"]))
            donor_span = identities[donor_id, donor_role]["step_span"]
            geometry = metadata.get("geometry", {})
            if (geometry.get("receiver_positions") != targets
                    or geometry.get("receiver_step_span") != identity["step_span"]
                    or geometry.get("donor_step_span") != donor_span
                    or geometry.get("donor_positions") != list(map_donor_window(tuple(targets), tuple(range(*donor_span))))
                    or metadata.get("target_sites") != keys):
                raise ValueError("saved geometry does not cover the full role-specific receiver/donor step")
            diagnostics = metadata.get("patch_diagnostics", {})
            if diagnostic_gate(diagnostics, sites):
                raise ValueError("saved native dispatch/hook evidence is incomplete")
            for diagnostic in diagnostics.values():
                if diagnostic.get("token_positions_by_site") != {k: targets for k in keys}:
                    raise ValueError("saved native hook positions disagree with full step")
                if treatment == "forced_reroute" and any(
                        diagnostic.get("manipulation_by_site", {}).get(k, {}).get("topk_changed_rate") != 1.0 for k in keys):
                    raise ValueError("saved forced dispatch evidence is incomplete")
        if treatment in {"self_patch", "unpatched_rerun"}:
            if score_identity_delta(row, indexed[role, "unpatched", None]) > tolerance:
                raise ValueError("saved identity control exceeds candidate-score tolerance")


def execute(adapter: Any, config: dict[str, Any], pairs: list[WindowPair],
            prepared: dict[tuple[str, str], PreparedTrace], report: dict[str, Any], output: Path,
            *, mode: str = "full", loaded_checkpoint: bool = False,
            resume: bool = False, execution_version: str = "v1") -> dict[str, Any]:
    """Execute complete 2x2 D pairs and publish only technically complete units."""
    if mode not in {"full", "smoke"} or not execution_version.strip():
        raise ValueError("mode must be full/smoke and execution_version must be nonempty")
    if any(config.get(k, v) != v for k, v in (("window", "full_step"), ("arm", "all_router"), ("alpha", 1.0))):
        raise ValueError("this protocol only implements full_step all_router alpha=1")
    matching = report["matching"]
    validate_matching(pairs, matching)
    sites = router_sites(adapter, config.get("expected_router_sites"))
    if (config.get("expected_router_block_indices") is not None
            and [s.layer_index for s in sites] != config["expected_router_block_indices"]):
        raise ValueError("adapter routed indices differ from the scientific config")
    keys = [site_key(s) for s in sites]
    lookup = {p.pair_id: p for p in pairs}
    identities = {(p.pair_id, r): trace_identity(adapter, prepared[p.pair_id, r], p.component)
                  for p in pairs for r in ROLES}
    selected = sorted(pairs, key=lambda p: stable_hash({"seed": config.get("seed", 0), "pair_id": p.pair_id}))
    if mode == "smoke":
        selected = selected[:int(config.get("smoke_pairs", 2))]
    eligibility = [pair_eligibility(p, lookup[matching[p.pair_id]], prepared) for p in selected]
    eligible_by_id = {r["pair_id"]: r for r in eligibility}
    contract = scientific_contract(config, pairs, prepared, identities, report, target_sites=keys)
    contract_id = stable_hash(contract, length=64)
    selected_ids = [p.pair_id for p in selected]
    binding = {"contract_id": contract_id, "execution_version": execution_version,
               "mode": mode, "selected_pair_ids": selected_ids,
               "checkpoint_forward_run": loaded_checkpoint}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError("existing run requires explicit --resume")
        if any(old.get(k) != v for k, v in binding.items()):
            raise ValueError("resume contract/execution version differs; use a fresh output version")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("refusing nonempty output without a recoverable manifest")
    output.mkdir(parents=True, exist_ok=True)
    units = output / "pairs"
    units.mkdir(exist_ok=True)
    write_json(manifest_path, {**binding, "protocol": PROTOCOL, "contract": contract,
                              "expected_counts": planned_counts(selected, matching, eligibility, mode=mode)})
    run_id = str(uuid.uuid4())
    attempt = output / "attempts" / run_id
    attempt.mkdir(parents=True)
    write_json(attempt / "runtime.json", _runtime(adapter))
    cache = CaptureCache(adapter, prepared, sites, attempt / "capture_union")
    all_rows, all_eligibility, completed, recovered = [], [], [], []
    forwards = 0
    start = time.monotonic()
    errors = []
    # Atomic pair files, rather than append-only JSONL tails, are authoritative.
    for pair in selected:
        path = units / (stable_hash({"pair_id": pair.pair_id}, length=32) + ".json")
        if path.exists():
            unit = json.loads(path.read_text())
            _validate_recovered_unit(unit, binding, eligible_by_id[pair.pair_id], identities,
                                     sites, mode=mode, tolerance=float(config["identity_atol"]))
            all_rows.extend(unit["rows"])
            all_eligibility.append(unit["eligibility"])
            completed.append(pair.pair_id)
            recovered.append(pair.pair_id)
            continue
        pair_rows = []
        eligibility_row = {**eligible_by_id[pair.pair_id], "run_id": run_id,
                           "contract_id": contract_id, "execution_version": execution_version}

        def row_payload(role: str, donor_role: str | None, donor_id: str | None,
                        treatment: str, scored: dict[str, Any], metadata: dict[str, Any] | None = None,
                        *, pair: WindowPair = pair) -> dict[str, Any]:
            donor_identity = None
            if donor_id is not None:
                assert donor_role is not None  # Patched rows always name both donor coordinates.
                donor_identity = identities[donor_id, donor_role]
            return {"protocol": PROTOCOL, "run_id": run_id,
                    "contract_id": contract_id, "execution_version": execution_version,
                    "pair_id": pair.pair_id, "source_component": pair.component,
                    "receiver_variant": role, "receiver_role": role,
                    "donor_variant": donor_role, "donor_role": donor_role,
                    "donor_pair_id": donor_id, "treatment": treatment,
                    "site": "all_router" if donor_id is not None else None,
                    "arm": "all_router", "window": "full_step", "alpha": 1.0,
                    "metadata": {"panel": "window_d", "window": "full_step", "alpha": 1.0,
                                 "target_sites": keys,
                                 "receiver_identity": identities[pair.pair_id, role],
                                 "donor_identity": donor_identity,
                                 **(metadata or {})}, **scored}

        try:
            if eligibility_row["eligible"]:
                donor_id = matching[pair.pair_id]
                for role in ROLES:
                    trace = prepared[pair.pair_id, role]
                    identity = identities[pair.pair_id, role]
                    baseline = None
                    for treatment in ("unpatched", "unpatched_rerun"):
                        scored = _validated_score(adapter.score_candidates(trace.prompt, CANDIDATES), identity)
                        forwards += 2
                        pair_rows.append(row_payload(role, None, None, treatment, scored))
                        if baseline is None:
                            baseline = scored
                        elif score_identity_delta(scored, baseline) > float(config["identity_atol"]):
                            raise ValueError("baseline repeat candidate-score drift")
                    assert baseline is not None  # The fixed two-control loop has scored unpatched.
                    targets = tuple(range(*trace.step_span))
                    receiver_captures = cache.get(pair.pair_id, role)
                    treatments = [("self_patch", pair.pair_id, role),
                                  ("donor_role", donor_id, "clean"),
                                  ("donor_role", donor_id, "corrupt")]
                    if mode == "smoke":
                        treatments.append(("forced_reroute", pair.pair_id, role))
                    for treatment, actual_donor, donor_role in treatments:
                        donor_trace = prepared[actual_donor, donor_role]
                        donor_positions = map_donor_window(targets, tuple(range(*donor_trace.step_span)))
                        donor_indices = [p - donor_trace.step_span[0] for p in donor_positions]
                        donor_captures = cache.get(actual_donor, donor_role)
                        bundles, metrics = {}, {}
                        for candidate in CANDIDATES:
                            patches, candidate_metrics = [], {}
                            if set(receiver_captures[candidate]) != set(keys) or set(donor_captures[candidate]) != set(keys):
                                raise ValueError("candidate capture must contain exactly all routed sites")
                            for site in sites:
                                key = site_key(site)
                                receiver = receiver_captures[candidate][key]
                                replacement = donor_captures[candidate][key][:, donor_indices, :]
                                if receiver.shape != replacement.shape or receiver.shape[1] != len(targets):
                                    raise ValueError("donor mapping did not preserve all receiver tokens")
                                patches.append(MultiPositionPatchSpec(
                                    site, targets, replacement,
                                    metadata={"forced_reroute": treatment == "forced_reroute"}))
                                candidate_metrics[key] = replacement_metrics(receiver, replacement, [])
                            bundles[candidate] = PatchBundle(tuple(patches), metadata={"protocol": PROTOCOL})
                            metrics[candidate] = candidate_metrics
                        scores, diagnostics = adapter.score_candidates_with_candidate_patch_bundles(
                            trace.prompt, CANDIDATES, bundles_by_candidate=bundles)
                        forwards += 2
                        scored = _validated_score(scores, identity)
                        gate_errors = diagnostic_gate(diagnostics, sites)
                        for candidate in CANDIDATES:
                            diag = diagnostics.get(candidate, {})
                            if diag.get("token_positions_by_site") != {k: list(targets) for k in keys}:
                                gate_errors.append(f"{candidate}: actual patch positions differ from full receiver step")
                            if treatment == "forced_reroute":
                                manipulation = diag.get("manipulation_by_site", {})
                                if any(manipulation.get(k, {}).get("topk_changed_rate") != 1.0 for k in keys):
                                    gate_errors.append(f"{candidate}: forced dispatch did not change every routed target")
                        metadata = {
                            "geometry": {"receiver_positions": list(targets),
                                         "donor_positions": list(donor_positions),
                                         "receiver_step_span": list(trace.step_span),
                                         "donor_step_span": list(donor_trace.step_span),
                                         "receiver_full_step_covered": True,
                                         "mapping": GEOMETRY},
                            "patch_diagnostics": diagnostics, "replacement_diagnostics": metrics,
                            "identity_candidate_score_delta": score_identity_delta(scored, baseline),
                            "replacement_diagnostics_apply_before_forced_override": treatment == "forced_reroute",
                        }
                        pair_rows.append(row_payload(role, donor_role, actual_donor, treatment, scored, metadata))
                        if gate_errors:
                            raise ValueError("; ".join(gate_errors))
                        if treatment == "self_patch" and score_identity_delta(scored, baseline) > float(config["identity_atol"]):
                            raise ValueError("self identity candidate-score drift")
                from routeverify.analysis.window_donor_contrast import analyze_window_donor_rows
                # Validate this atomic unit before making it recoverable. Two
                # draws are a structural check; final intervals use the contract.
                analyze_window_donor_rows(pair_rows, eligibility_rows=[eligibility_row],
                                          expected_pair_ids=[pair.pair_id], n_resamples=2)
            unit = {"pair_id": pair.pair_id, "contract_id": contract_id,
                    "execution_version": execution_version, "run_id": run_id,
                    "technical_passed": True, "eligibility": eligibility_row, "rows": pair_rows}
            write_json(path, unit)
            all_rows.extend(pair_rows)
            all_eligibility.append(eligibility_row)
            completed.append(pair.pair_id)
            write_json(output / "progress.json", {"run_id": run_id, "completed_pair_ids": completed,
                       "selected_pair_ids": selected_ids, "scoring_forwards_this_attempt": forwards,
                       "capture_forwards_this_attempt": cache.forward_count})
            print(f"{mode}: completed {pair.pair_id}; {len(completed)}/{len(selected)} pairs", flush=True)
        except Exception as exc:
            errors.append(f"{pair.pair_id}: {type(exc).__name__}: {exc}")
            write_json(attempt / "incomplete_pair.json", {"pair_id": pair.pair_id,
                       "error": errors[-1], "eligibility": eligibility_row, "partial_rows": pair_rows,
                       "recovery": "rerun entire pair under a new loaded run; never use these partial rows"})
            break
    temporary = output / "predictions.jsonl.tmp"
    with temporary.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output / "predictions.jsonl")
    write_json(output / "eligibility.json", all_eligibility)
    complete = len(completed) == len(selected)
    passed = complete and not errors
    if passed:
        from routeverify.analysis.window_donor_contrast import analyze_window_donor_rows
        analysis = analyze_window_donor_rows(all_rows, eligibility_rows=all_eligibility,
                    expected_pair_ids=selected_ids, n_resamples=int(config.get("bootstrap", 50000)),
                    seed=int(config.get("analysis_seed", 20260908)))
        write_json(output / "analysis.json", analysis)
    smoke_coverage = all(r["eligible"] for r in all_eligibility) and bool(all_rows)
    summary = {**binding, "protocol": PROTOCOL, "run_id": run_id, "passed": passed,
               "gpu_smoke_passed": passed and mode == "smoke" and loaded_checkpoint and smoke_coverage,
               "execution_scope": "checkpoint" if loaded_checkpoint else "tiny_or_injected_adapter",
               "runtime_model_class": type(adapter.model).__name__, "n_pairs": len(completed),
               "eligible_pairs": sum(r["eligible"] for r in all_eligibility),
               "source_components": len({r["source_component"] for r in all_eligibility if r["eligible"]}),
               "completed_pair_ids": completed, "recovered_pair_ids": recovered,
               "scoring_forwards": forwards, "capture_forwards": cache.forward_count,
               "forward_count_scope": "this loaded invocation; recovered pairs add no forwards",
               "expected_counts_fresh_run": planned_counts(selected, matching, eligibility, mode=mode),
               "target_sites": keys, "errors": errors, "skipped_cells": [r for r in all_eligibility if not r["eligible"]],
               "elapsed_seconds": time.monotonic() - start, "effect_direction_is_not_a_gate": True,
               "no_validation_access": True}
    write_json(output / "execution_summary.json", summary)
    write_json(attempt / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full", "analyze"), default="preflight")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--model-cache", type=Path, help="runtime cache path; does not change checkpoint identity")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-version", default="v1")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if (config.get("protocol") != PROTOCOL or config.get("partition") != "discovery"
            or config.get("window") != "full_step" or config.get("arm") != "all_router"
            or config.get("alpha") != 1.0):
        parser.error("requires this discovery-only full_step all_router alpha1 protocol")
    if args.mode == "analyze":
        from routeverify.analysis.window_donor_contrast import analyze_window_donor_rows
        manifest = json.loads((args.output / "run_manifest.json").read_text())
        summary = json.loads((args.output / "execution_summary.json").read_text())
        if not summary["passed"]:
            parser.error("analysis requires a technically complete run")
        rows = [json.loads(line) for line in (args.output / "predictions.jsonl").read_text().splitlines()]
        eligibility_rows = json.loads((args.output / "eligibility.json").read_text())
        for row in [*rows, *eligibility_rows]:
            if (row.get("pair_id") not in manifest["selected_pair_ids"]
                    or any(row.get(k) != manifest[k] for k in ("contract_id", "execution_version"))):
                parser.error("analysis input rows differ from the saved execution manifest")
        declared_analysis = manifest["contract"]["analysis"]
        if (int(config.get("bootstrap", 50000)) != declared_analysis["replicates"]
                or int(config.get("analysis_seed", 20260908)) != declared_analysis["seed"]):
            parser.error("analysis settings differ from the saved scientific contract")
        result = analyze_window_donor_rows(rows,
                    eligibility_rows=eligibility_rows,
                    expected_pair_ids=manifest["selected_pair_ids"],
                    n_resamples=int(config.get("bootstrap", 50000)), seed=int(config.get("analysis_seed", 20260908)))
        write_json(args.output / "analysis.json", result)
        return
    pairs = load_discovery_pairs(config, args.pairs)
    if len(pairs) != int(config["expected_pairs"]) or len({p.component for p in pairs}) != int(config["expected_source_components"]):
        parser.error("discovery N/K differ from the scientific contract")
    matching = read_matching(Path(config["matching_manifest"]), pairs)
    model_config = load_model_config(Path(config["model"]))
    if args.model_cache:
        model_config = model_config.model_copy(update={"cache_dir": args.model_cache})
    tokenizer_adapter = tokenizer_only_adapter(model_config, args.tokenizer_path)
    prepared = prepare_panel(tokenizer_adapter, pairs)
    identities = {(p.pair_id, role): trace_identity(tokenizer_adapter, prepared[p.pair_id, role], p.component)
                  for p in pairs for role in ROLES}
    lookup = {p.pair_id: p for p in pairs}
    eligibility = [pair_eligibility(p, lookup[matching[p.pair_id]], prepared) for p in pairs]
    report = {"protocol": PROTOCOL, "model": model_config.model_dump(mode="json"),
              "matching": matching, "checkpoint_forward_run": False, "gpu_smoke_passed": False,
              "tokenizer": tokenizer_provenance(tokenizer_adapter, model_config, args.tokenizer_path),
              "n_pairs": len(pairs), "source_components": len({p.component for p in pairs}),
              "eligibility": eligibility, "planned_full_counts": planned_counts(pairs, matching, eligibility),
              "identities": [{"pair_id": p, "role": r, **identity} for (p, r), identity in identities.items()],
              "sources": {"pairs_sha256": sha256(args.pairs), "config_sha256": sha256(args.config),
                          "matching_manifest_sha256": sha256(Path(config["matching_manifest"]))},
              "passed": True, "no_validation_access": True}
    if args.mode == "preflight":
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("preflight uses a fresh output directory")
        write_json(args.output / "preflight.json", report)
        print(json.dumps({k: report[k] for k in ("passed", "n_pairs", "source_components", "planned_full_counts")}))
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("checkpoint execution needs a fresh output or explicit --resume")
    if args.mode == "full" and args.smoke_report is None:
        parser.error("full requires a passing checkpoint smoke report")
    import torch
    if not torch.cuda.is_available():
        parser.error("checkpoint smoke/full requires CUDA; local random tiny forwards use tests")
    adapter = create_adapter(model_config.alias)
    updates: dict[str, Any] = {"load_weights": True}
    if args.tokenizer_path and callable(getattr(adapter, "load_tokenizer", None)):
        updates["tokenizer_path"] = args.tokenizer_path
    adapter.load(model_config.model_copy(update=updates))
    runtime_prepared = prepare_panel(adapter, pairs)
    if runtime_prepared != prepared:
        raise RuntimeError("loaded checkpoint tokenizer differs from actual CPU preflight inputs")
    if args.mode == "full":
        if args.smoke_report is None:
            parser.error("full requires a passing checkpoint smoke for this scientific contract")
        smoke = json.loads(args.smoke_report.read_text())
        sites = [site_key(s) for s in router_sites(adapter, config.get("expected_router_sites"))]
        contract = scientific_contract(config, pairs, prepared, identities, report, target_sites=sites)
        if (smoke.get("protocol") != PROTOCOL or not smoke.get("gpu_smoke_passed")
                or smoke.get("contract_id") != stable_hash(contract, length=64)
                or smoke.get("execution_version") != args.execution_version):
            parser.error("smoke does not qualify this checkpoint, assignment, geometry, scorer or execution version")
    result = execute(adapter, config, pairs, runtime_prepared, report, args.output,
                     mode=args.mode, loaded_checkpoint=True, resume=args.resume,
                     execution_version=args.execution_version)
    if not result["passed"]:
        raise SystemExit("technical failure; see execution_summary.json; partial pair is not reusable")


if __name__ == "__main__":
    main()
