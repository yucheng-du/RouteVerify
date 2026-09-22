"""Worker-local hooks for pinned vLLM Mistral Small 4, without importing vLLM.

The caller owns checkpoint/kernel validation and the engine configuration. This
module additionally verifies a single complete, unpadded prefill and the actual
modular dispatch. Methods are wrapped on individual instances and restored, never
patched globally. Use ``begin`` before generation, ``finish`` afterwards, and
``cleanup`` in a caller-side finally block if generation raises.

Wire schema v1 contains request_id, input_ids, capture_sites and patches. Each site
contains layer_index, kind and token_positions (or token_position for PatchSpec).
Replacements have [positions, width] or [1, positions, width] shape; a single
vector is accepted for a legacy singleton. All captures serialize as [1, P, W].
Floating captures are FP32, including the unrounded delayed residual sum.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

VLLM_SOURCE_REVISION = "537af2c3a4ba7462ddc9bc94ec7a4ea496da6d2e"
SCHEMA_VERSION = 1
_SESSION_ATTRIBUTE = "_routeverify_mistral_hook_session"
_KINDS = {
    "pre_moe_residual", "router_input", "router_logits", "router_topk_ids",
    "router_topk_weights", "moe_output", "post_moe_residual",
}
_PATCH_KINDS = {"router_logits", "moe_output", "post_moe_residual"}


def _torch() -> Any:
    import torch

    return torch


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class _Site:
    layer: int
    kind: str
    positions: tuple[int, ...]
    key: str
    replacement: Any = None
    forced: bool = False


def _site(payload: Any, length: int, *, patch: bool) -> _Site:
    if not isinstance(payload, Mapping):
        raise ValueError("site must be an object")
    layer = _integer(payload.get("layer_index"), "layer_index")
    kind = payload.get("kind")
    if kind not in (_PATCH_KINDS if patch else _KINDS):
        raise ValueError(f"unsupported {'patch' if patch else 'capture'} kind: {kind!r}")
    positions = payload.get("token_positions")
    if positions is None and "token_position" in payload:
        positions = [payload["token_position"]]
    if not isinstance(positions, (list, tuple)) or not positions:
        raise ValueError("token_positions must be a nonempty list")
    positions = tuple(_integer(p, "token position") for p in positions)
    if len(set(positions)) != len(positions) or max(positions) >= length:
        raise ValueError("token_positions must be unique and within the complete input")
    key = f"layer_{layer}.{kind}"
    if payload.get("key", key) != key:
        raise ValueError(f"site key must equal {key}")
    replacement = payload.get("replacement")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("patch metadata must be an object")
    forced = bool(metadata.get("forced_reroute", False))
    if forced and kind != "router_logits":
        raise ValueError("forced_reroute only applies to router_logits")
    if patch and replacement is None:
        raise ValueError(f"replacement is required for {key}")
    return _Site(layer, kind, positions, key, replacement, forced)


def _layers(model: Any) -> list[Any]:
    for path in ("language_model.model.layers", "model.layers", "layers"):
        obj = model
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise ValueError("Mistral worker model has no supported decoder layer path")


class _Session:
    def __init__(self, model: Any, request: Mapping[str, Any]):
        if request.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Mistral hook request must use schema_version=1")
        if request.get("capture_dtype", "float32") not in {"float32", "fp32"}:
            raise ValueError("Mistral captures require FP32, preserving delayed residual sums")
        token_ids = request.get("input_ids")
        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            raise ValueError("input_ids must contain the complete single request")
        self.ids = tuple(_integer(i, "input token ID") for i in token_ids)
        self.length = len(self.ids)
        self.model = model
        self.request_id = str(request.get("request_id", ""))
        self.capture_sites = self._parse_sites(request.get("capture_sites", []), patch=False)
        self.patches = self._parse_sites(request.get("patches", []), patch=True)
        self.handles: list[Any] = []
        self.restorations: list[tuple[Any, str, bool, Any]] = []
        self.events: dict[str, int] = {"model_prefill": 0, "model_completed": 0}
        self.captures: dict[str, Any] = {}
        self.capture_metadata: dict[str, Any] = {}
        self.captured_positions: dict[str, list[int]] = {}
        self.patched_positions: dict[str, list[int]] = {}
        self.reference_selection_calls = 0
        self.expected_dispatch: dict[int, tuple[Any, Any]] = {}
        self.dispatch_verified: dict[int, bool] = {}
        self.receiver_router: dict[int, tuple[Any, Any]] = {}
        self.patched_logits: dict[int, Any] = {}
        self.manipulation: dict[str, dict[str, float]] = {}
        self.shared: dict[int, Any] = {}
        self.routed: dict[int, Any] = {}
        self.branch_errors: dict[int, float] = {}
        self.residual_errors: dict[str, float] = {}
        self.residual_quantization: dict[str, dict[str, Any]] = {}
        self.native_dtypes: set[str] = set()
        self.executing_devices: set[str] = set()
        self.closed = False

    def _parse_sites(self, items: Any, *, patch: bool) -> dict[str, _Site]:
        if not isinstance(items, (list, tuple)):
            raise ValueError("capture_sites and patches must be lists")
        sites = [_site(item, self.length, patch=patch) for item in items]
        if len({s.key for s in sites}) != len(sites):
            raise ValueError("duplicate site in capture_sites or patches")
        return {s.key: s for s in sites}

    def _event(self, layer: int, name: str) -> None:
        key = f"layer_{layer}.{name}"
        self.events[key] = self.events.get(key, 0) + 1
        if self.events.get("model_prefill") != 1 or self.events[key] != 1:
            raise RuntimeError(f"{key} did not execute exactly once in the complete prefill")

    def _matrix(self, tensor: Any, label: str) -> Any:
        torch = _torch()
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or tensor.shape[0] != self.length:
            raise RuntimeError(f"{label} must be [complete_input_tokens, width]; chunking/padding unsupported")
        if tensor.dtype.is_floating_point and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"{label} contains non-finite values")
        self.native_dtypes.add(str(tensor.dtype))
        self.executing_devices.add(str(tensor.device))
        return tensor

    def _positions(self, layer: int) -> list[int]:
        return sorted({p for s in (*self.capture_sites.values(), *self.patches.values())
                       if s.layer == layer for p in s.positions})

    def _capture(self, layer: int, kind: str, tensor: Any, *, semantics: str = "") -> None:
        key = f"layer_{layer}.{kind}"
        spec = self.capture_sites.get(key)
        if spec is None:
            return
        if key in self.captures:
            raise RuntimeError(f"capture repeated: {key}")
        tensor = self._matrix(tensor, key)
        selected = tensor[list(spec.positions)].detach().clone()
        selected = selected.float() if tensor.dtype.is_floating_point else selected.long()
        self.captures[key] = selected.cpu().unsqueeze(0).tolist()
        self.captured_positions[key] = list(spec.positions)
        self.capture_metadata[key] = {
            "token_positions": list(spec.positions), "native_dtype": str(tensor.dtype),
            "serialized_dtype": "float32" if tensor.dtype.is_floating_point else "int64",
            "semantics": semantics or kind,
        }

    def _replacement(self, spec: _Site, target: Any) -> Any:
        torch = _torch()
        try:
            value = torch.as_tensor(spec.replacement, dtype=torch.float32, device=target.device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.key} replacement is not a numeric array") from exc
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1 and len(spec.positions) == 1:
            value = value.unsqueeze(0)
        if tuple(value.shape) != (len(spec.positions), target.shape[-1]):
            raise ValueError(f"{spec.key} replacement shape must be [1,{len(spec.positions)},{target.shape[-1]}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{spec.key} replacement contains non-finite values")
        return value

    def _replace(self, layer: int, kind: str, tensor: Any) -> Any:
        torch = _torch()
        key = f"layer_{layer}.{kind}"
        spec = self.patches.get(key)
        if spec is None:
            return tensor
        value = self._replacement(spec, tensor).to(dtype=tensor.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} replacement overflows the native dtype")
        result = tensor
        if not torch.equal(tensor[list(spec.positions)], value):
            result = tensor.clone()
            result[list(spec.positions)] = value
        self.patched_positions[key] = list(spec.positions)
        return result

    def _wrap(self, obj: Any, name: str, callback: Any) -> None:
        original = getattr(obj, name)
        owned = name in vars(obj)
        self.restorations.append((obj, name, owned, vars(obj).get(name)))

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return callback(original, args, kwargs)

        setattr(obj, name, wrapped)

    def _guard_input(self, _model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        torch = _torch()
        self.events["model_prefill"] += 1
        if self.events["model_prefill"] != 1:
            raise RuntimeError("Mistral hook session received multiple model forwards")
        ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        if not torch.is_tensor(ids) or ids.ndim != 1:
            raise RuntimeError("Mistral hooks require one flattened, unbatched prefill input_ids")
        if ids.detach().cpu().tolist() != list(self.ids):
            raise RuntimeError("actual prefill input_ids differ from the complete hook request")
        if not torch.is_tensor(positions) or positions.ndim != 1:
            raise RuntimeError("Mistral hooks require absolute one-dimensional positions")
        if positions.detach().cpu().tolist() != list(range(self.length)):
            raise RuntimeError("actual positions omit or duplicate input tokens; prefix cache/chunking unsupported")

    def _model_completed(self, _model: Any, _args: tuple[Any, ...], _output: Any) -> None:
        # This hook does not use always_call: a failed/incomplete forward is not
        # a completed scoring pass. Generation may fail later during sampling.
        self.events["model_completed"] += 1

    def _validate_runner(self, layer: Any, index: int) -> Any:
        runner = getattr(getattr(layer, "mlp", None), "experts", None)
        required = ("gate", "router", "routed_experts", "moe_config", "_shared_experts",
                    "apply_routed_output_transform", "_maybe_apply_routed_scale_to_output")
        if runner is None or any(not hasattr(runner, key) for key in required):
            raise ValueError(f"layer {index} lacks the pinned MoERunner boundaries")
        quant = getattr(runner.routed_experts, "quant_method", None)
        if quant is None or bool(getattr(quant, "is_monolithic", True)):
            raise ValueError("Mistral research hooks require a modular expert backend")
        if not callable(getattr(runner.routed_experts, "forward_modular", None)):
            raise ValueError("missing actual forward_modular dispatch boundary")
        if not callable(getattr(runner.router, "select_experts", None)) or runner.gate is None:
            raise ValueError("missing native gate or expert selection boundary")
        if runner._shared_experts is None or bool(getattr(runner, "_fse_fuse_gate", False)):
            raise ValueError("separate receiver shared experts and unfused gate are required")
        if getattr(runner._shared_experts, "_stream", None) is not None:
            raise ValueError("set VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 before model construction")
        if bool(getattr(runner, "enable_dbo", False)):
            raise ValueError("dual batch overlap is unsupported for a hook session")
        if any(getattr(runner, key, None) is not None
               for key in ("routed_input_transform", "routed_output_transform")):
            raise ValueError("Mistral hook boundary requires untransformed native hidden states")
        cfg = runner.moe_config
        parallel = getattr(cfg, "moe_parallel_config", cfg)
        for obj in (cfg, parallel):
            if any(getattr(obj, name, 1) != 1 for name in ("tp_size", "ep_size", "dp_size")):
                raise ValueError("Mistral hooks currently require TP=EP=DP=1")
            if any(bool(getattr(obj, name, False)) for name in
                   ("is_sequence_parallel", "enable_eplb", "use_all2all_kernels")):
                raise ValueError("sequence/expert redistribution and EPLB are unsupported")
        if not hasattr(layer, "post_attention_layernorm"):
            raise ValueError("Mistral decoder lacks post_attention_layernorm")
        return runner

    def install(self) -> None:
        layers = _layers(self.model)
        indices = sorted({s.layer for s in (*self.capture_sites.values(), *self.patches.values())})
        if any(i >= len(layers) for i in indices):
            raise ValueError("requested decoder layer is out of range")
        runners = {i: self._validate_runner(layers[i], i) for i in indices}
        self.handles.append(self.model.register_forward_pre_hook(self._guard_input, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._model_completed))
        for i in indices:
            self._install_layer(layers[i], runners[i], i)

    def _install_layer(self, layer: Any, runner: Any, i: int) -> None:
        torch = _torch()
        original_select = runner.router.select_experts
        for name in ("gate_input", "gate", "selection", "dispatch", "shared", "routed",
                     "moe_total", "postres", "prenorm"):
            self.events[f"layer_{i}.{name}"] = 0

        def gate_input(_module: Any, args: tuple[Any, ...]) -> None:
            self._event(i, "gate_input")
            value = self._matrix(args[0], f"layer_{i}.router_input")
            self._capture(i, "router_input", value)

        def gate_output(_module: Any, args: tuple[Any, ...], output: Any) -> Any:
            self._event(i, "gate")
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("pinned GateLinear must return (raw_logits, bias)")
            raw = self._matrix(output[0], f"layer_{i}.router_logits")
            spec = self.patches.get(f"layer_{i}.router_logits")
            result = raw
            if spec is not None:
                # A diagnostic reference only: final dispatch is independently
                # read from the subsequent forward_modular arguments below.
                weights, ids = original_select(
                    hidden_states=args[0], router_logits=raw,
                    topk_indices_dtype=getattr(runner.routed_experts.quant_method,
                                              "topk_indices_dtype", None),
                )
                del weights
                self.reference_selection_calls += 1
                positions = list(spec.positions)
                self.receiver_router[i] = (raw[positions].detach().clone(), ids[positions].detach().clone())
                if spec.forced:
                    selected = torch.zeros_like(raw[positions], dtype=torch.bool)
                    selected.scatter_(-1, ids[positions].long(), True)
                    limits = torch.finfo(raw.dtype)
                    result = raw.clone()
                    result[positions] = torch.where(
                        selected, torch.full_like(raw[positions], limits.min),
                        torch.full_like(raw[positions], limits.max),
                    )
                    self.patched_positions[spec.key] = positions
                else:
                    result = self._replace(i, "router_logits", raw)
                self.patched_logits[i] = result[positions].detach().clone()
            self._capture(i, "router_logits", result, semantics="actual_gate_output_before_native_softmax_topk")
            return (result, output[1]) if result is not raw else output

        def select(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            self._event(i, "selection")
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("native select_experts must return (weights, ids)")
            weights, ids = result
            self._matrix(weights, f"layer_{i}.selected_weights")
            self._matrix(ids, f"layer_{i}.selected_ids")
            self.expected_dispatch[i] = (ids.detach().clone(), weights.detach().clone())
            return result

        def dispatch(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            self._event(i, "dispatch")
            ids = kwargs.get("topk_ids", args[2] if len(args) > 2 else None)
            weights = kwargs.get("topk_weights", args[1] if len(args) > 1 else None)
            self._matrix(ids, f"layer_{i}.actual_dispatch_ids")
            self._matrix(weights, f"layer_{i}.actual_dispatch_weights")
            expected = self.expected_dispatch.pop(i, None)
            verified = expected is not None and torch.equal(ids, expected[0]) and torch.equal(weights, expected[1])
            self.dispatch_verified[i] = verified
            if not verified:
                raise RuntimeError("actual modular dispatch differs from native selected IDs/weights")
            self._capture(i, "router_topk_ids", ids, semantics="actual_forward_modular_argument")
            self._capture(i, "router_topk_weights", weights, semantics="actual_forward_modular_argument")
            if i in self.receiver_router:
                from routeverify.instrumentation.patching import router_manipulation_metrics

                key = f"layer_{i}.router_logits"
                positions = list(self.patches[key].positions)
                old_logits, old_ids = self.receiver_router.pop(i)
                self.manipulation[key] = router_manipulation_metrics(
                    old_logits.float().cpu().numpy(),
                    self.patched_logits.pop(i).float().cpu().numpy(),
                    receiver_topk_ids=old_ids.cpu().numpy(),
                    patched_topk_ids=ids[positions].detach().cpu().numpy(),
                )
            return original(*args, **kwargs)

        def scaled(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            self._event(i, "shared")
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2 or result[0] is None:
                raise RuntimeError("separate finalized shared/routed tensor outputs are required")
            self.shared[i] = self._matrix(result[0], f"layer_{i}.shared_output")[self._positions(i)].detach().clone()
            return result

        def routed(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            self._event(i, "routed")
            result = self._matrix(original(*args, **kwargs), f"layer_{i}.moe_output")
            result = self._replace(i, "moe_output", result)
            self._capture(i, "moe_output", result, semantics="finalized_scaled_routed_output_before_shared_add")
            self.routed[i] = result[self._positions(i)].detach().clone()
            return result

        def moe_total(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            self._event(i, "moe_total")
            output = self._matrix(output, f"layer_{i}.moe_total")[self._positions(i)]
            if i not in self.shared or i not in self.routed:
                raise RuntimeError("shared/routed branches did not precede MoE total")
            expected = self.shared.pop(i) + self.routed.pop(i)
            error = float((output.float() - expected.float()).abs().max().item())
            self.branch_errors[i] = error
            if not torch.equal(output, expected):
                raise RuntimeError("native MoE output is not the observed shared+routed sum")

        def prenorm(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            self._event(i, "prenorm")
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("post_attention_layernorm must expose (normalized, residual)")
            self._capture(i, "pre_moe_residual", output[1], semantics="post_attention_add_residual_before_moe")

        def postres(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
            self._event(i, "postres")
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("pinned decoder must return the delayed (hidden, residual) pair")
            hidden = self._matrix(output[0], f"layer_{i}.decoder_hidden")
            residual = self._matrix(output[1], f"layer_{i}.decoder_residual")
            if hidden.dtype != residual.dtype or hidden.shape != residual.shape:
                raise RuntimeError("delayed residual components must share shape and dtype")
            total = hidden.float() + residual.float()
            key = f"layer_{i}.post_moe_residual"
            spec = self.patches.get(key)
            result = output
            if spec is not None:
                positions = list(spec.positions)
                target = self._replacement(spec, total)
                receiver = total[positions].clone()
                identity = torch.equal(receiver, target)
                bound = torch.zeros_like(target, dtype=torch.float64)
                if not identity:
                    high = target.to(hidden.dtype)
                    remainder = target - high.float()
                    low = remainder.to(residual.dtype)
                    if not bool(torch.isfinite(high).all()) or not bool(torch.isfinite(low).all()):
                        raise ValueError("residual replacement overflows native high/low components")
                    hidden, residual = hidden.clone(), residual.clone()
                    hidden[positions], residual[positions] = high, low
                    result = (hidden, residual)
                    total = hidden.float() + residual.float()
                    # Let d = target - high (exact), r = RN32(d),
                    # low = RNnative(r), and executed = RN32(high + low).
                    # The three rounding errors bound |executed - target|.
                    # high rounding itself cancels in d; its residual is what
                    # receives the second native rounding. Normal rounding uses
                    # unit roundoff eps/2. Adding tiny (rather than half the
                    # smallest subnormal) also covers flushed subnormal results.
                    native_info = torch.finfo(hidden.dtype)
                    fp32_info = torch.finfo(torch.float32)
                    exact_remainder = target.double() - high.double()
                    subtract_bound = fp32_info.eps / 2 * exact_remainder.abs() + fp32_info.tiny
                    low_bound = native_info.eps / 2 * remainder.double().abs() + native_info.tiny
                    add_bound = fp32_info.eps / 2 * (high.double() + low.double()).abs() + fp32_info.tiny
                    bound = subtract_bound + low_bound + add_bound
                executed = total[positions]
                absolute_error = (executed.double() - target.double()).abs()
                error = float(absolute_error.max().item())
                self.residual_errors[key] = error
                relative_floor = torch.finfo(torch.float32).tiny
                requested_delta = target.double() - receiver.double()
                executed_delta = executed.double() - receiver.double()
                within_bound = bool((absolute_error <= bound).all())
                self.residual_quantization[key] = {
                    "strategy": "native_high_low_split_with_quantized_executed_fp32_sum",
                    "identity_preserved_original_components": identity,
                    "native_component_dtype": str(hidden.dtype),
                    "token_positions": positions,
                    "requested_executed_max_abs_error": error,
                    "requested_executed_max_relative_error": float((
                        absolute_error / target.double().abs().clamp_min(relative_floor)
                    ).max().item()),
                    "relative_error_denominator_floor": relative_floor,
                    "requested_executed_l2_error": float(torch.linalg.vector_norm(absolute_error).item()),
                    "requested_delta_l2_norm": float(torch.linalg.vector_norm(requested_delta).item()),
                    "executed_delta_l2_norm": float(torch.linalg.vector_norm(executed_delta).item()),
                    "requested_delta_l2_norm_by_position": torch.linalg.vector_norm(requested_delta, dim=-1).cpu().tolist(),
                    "executed_delta_l2_norm_by_position": torch.linalg.vector_norm(executed_delta, dim=-1).cpu().tolist(),
                    "max_abs_error_bound": float(bound.max().item()),
                    "max_abs_error_bound_by_position": bound.max(dim=-1).values.cpu().tolist(),
                    "all_errors_within_bound": within_bound,
                    "error_bound_formula": "u32*abs(target-high)+u_native*abs(RN32(target-high))+u32*abs(high+low)+tiny_native+2*tiny32; identity bound=0",
                    "underflow_bound": "smallest_normal_covers_gradual_or_flushed_subnormals",
                }
                if not within_bound:
                    raise RuntimeError("native high/low residual split exceeded its rounding error bound")
                self.patched_positions[key] = positions
            self._capture(i, "post_moe_residual", total, semantics="fp32_sum_of_delayed_hidden_and_residual")
            return result

        self.handles.append(runner.gate.register_forward_pre_hook(gate_input))
        self.handles.append(runner.gate.register_forward_hook(gate_output))
        self._wrap(runner.router, "select_experts", select)
        self._wrap(runner.routed_experts, "forward_modular", dispatch)
        self._wrap(runner, "_maybe_apply_routed_scale_to_output", scaled)
        self._wrap(runner, "apply_routed_output_transform", routed)
        self.handles.append(runner.register_forward_hook(moe_total))
        self.handles.append(layer.post_attention_layernorm.register_forward_hook(prenorm))
        self.handles.append(layer.register_forward_hook(postres))

    def result(self) -> dict[str, Any]:
        missing_events = {k: count for k, count in self.events.items() if count != 1}
        missing_captures = set(self.capture_sites) - set(self.captures)
        missing_patches = set(self.patches) - set(self.patched_positions)
        if missing_events or missing_captures or missing_patches:
            raise RuntimeError(f"incomplete Mistral hook coverage: events={missing_events}, captures={sorted(missing_captures)}, patches={sorted(missing_patches)}")
        if self.expected_dispatch:
            raise RuntimeError("native expert selections were not consumed by modular dispatch")
        return self.report(passed=True)

    def report(self, *, passed: bool = False, errors: tuple[str, ...] = ()) -> dict[str, Any]:
        """Serialize partial state without assuming all hooks or the model returned."""
        capture_coverage = all(self.captured_positions.get(key) == list(spec.positions)
                               for key, spec in self.capture_sites.items())
        patch_coverage = all(self.patched_positions.get(key) == list(spec.positions)
                             for key, spec in self.patches.items())
        all_events = all(count == 1 for count in self.events.values())
        expected_layers = {s.layer for s in (*self.capture_sites.values(), *self.patches.values())}
        dispatch_complete = expected_layers == set(self.dispatch_verified) and all(self.dispatch_verified.values())
        diagnostics = {
            "passed": passed, "partial": not passed, "errors": list(errors),
            "request_id": self.request_id, "schema_version": SCHEMA_VERSION,
            "vllm_source_revision": VLLM_SOURCE_REVISION,
            "model_forward_count": self.events["model_completed"],
            "attempted_model_forward_count": self.events["model_prefill"],
            "complete_input_token_count": self.length,
            "complete_prefill_verified": self.events["model_prefill"] == self.events["model_completed"] == 1,
            "all_selected_positions_covered": capture_coverage and patch_coverage,
            "all_registered_hooks_fired": all_events and capture_coverage and patch_coverage,
            "registered_site_count": len(self.patches), "registered_sites": list(self.patches),
            "hook_fire_counts": {key: int(key in self.patched_positions) for key in self.patches},
            "event_counts": dict(self.events),
            "token_positions_by_site": {key: list(s.positions) for key, s in self.patches.items()},
            "captured_positions_by_site": dict(self.captured_positions),
            "patched_positions_by_site": dict(self.patched_positions),
            "atomic_position_replacement": True, "simultaneous": len(self.patches) > 1,
            "all_router_dispatch_verified": dispatch_complete if expected_layers else None,
            "dispatch_by_site": {f"layer_{i}.router_logits": {
                "downstream_dispatch_verified": value,
                "actual_topk_ids_match_native_patched_routing": value,
                "actual_topk_weights_match_native_patched_routing": value,
                "expert_dispatch_hook_fire_count": self.events[f"layer_{i}.dispatch"],
                "dispatch_after_router_patch": self.events[f"layer_{i}.gate"] == 1,
            } for i, value in self.dispatch_verified.items()},
            "router_reference_selection_calls": self.reference_selection_calls,
            "router_capture_source": "actual_forward_modular_arguments",
            "router_semantics_id": "mistral_small4_native_softmax_top4_renorm_v1",
            "router_kl_interpretation": "softmax_projection_of_router_logits",
            "manipulation_by_site": dict(self.manipulation),
            "shared_plus_routed_max_abs_errors": {str(i): e for i, e in self.branch_errors.items()},
            "residual_target_max_abs_errors": dict(self.residual_errors),
            "residual_quantization_by_site": dict(self.residual_quantization),
            "residual_patch_strategy": "native_high_low_split_with_quantized_executed_fp32_sum; identity_preserves_original_components",
            "native_dtypes": sorted(self.native_dtypes), "execution_devices": sorted(self.executing_devices),
        }
        return {"captures": dict(self.captures), "capture_metadata": dict(self.capture_metadata),
                "diagnostics": diagnostics}

    def close(self) -> None:
        if self.closed:
            return
        for handle in reversed(self.handles):
            handle.remove()
        for obj, name, owned, previous in reversed(self.restorations):
            if owned:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.handles.clear()
        self.restorations.clear()
        self.closed = True


def begin(model: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Install one serializable request on the worker model; never start inference."""

    if hasattr(model, _SESSION_ATTRIBUTE):
        raise RuntimeError("Mistral hook session already active; finish or cleanup it first")
    if not isinstance(request, Mapping):
        raise ValueError("hook request must be an object")
    session = _Session(model, request)
    setattr(model, _SESSION_ATTRIBUTE, session)
    try:
        session.install()
    except BaseException:
        cleanup(model)
        raise
    return {"installed": True, "request_id": session.request_id,
            "capture_site_count": len(session.capture_sites), "patch_site_count": len(session.patches)}


def finish(model: Any) -> dict[str, Any]:
    """Return success or failed coverage plus partial counts, then remove hooks.

    Coverage failures are represented by diagnostics.passed=False, allowing the
    bridge to retain completed forward counts before raising a user-facing error.
    """

    session = getattr(model, _SESSION_ATTRIBUTE, None)
    if session is None:
        raise RuntimeError("no active Mistral hook session")
    try:
        try:
            return session.result()
        except Exception as exc:
            return session.report(errors=(f"{type(exc).__name__}: {exc}",))
    finally:
        cleanup(model)


def snapshot(model: Any) -> dict[str, Any]:
    """Read serializable partial counters/captures without changing active hooks."""
    session = getattr(model, _SESSION_ATTRIBUTE, None)
    if session is None:
        return {"captures": {}, "capture_metadata": {}, "diagnostics": {
            "passed": False, "partial": True, "active": False, "model_forward_count": 0,
            "attempted_model_forward_count": 0, "errors": ["no active Mistral hook session"],
        }}
    return session.report()


def cleanup(model: Any) -> dict[str, Any]:
    """Idempotent recovery after generation failure, restoring instance methods."""

    session = getattr(model, _SESSION_ATTRIBUTE, None)
    if session is None:
        return {"cleaned": False}
    partial = session.report()
    try:
        session.close()
    finally:
        delattr(model, _SESSION_ATTRIBUTE)
    return {"cleaned": True, "partial_diagnostics": partial["diagnostics"],
            "partial_captures": partial["captures"], "partial_capture_metadata": partial["capture_metadata"]}


def inspect_model(model: Any) -> dict[str, Any]:
    """Inspect loaded native modules and packed tensors, without running forward.

    This deliberately does not certify a CPU reference model. The engine bridge
    separately validates source hashes, CUDA architecture and checkpoint metadata.
    """

    torch = _torch()
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    validator = _Session(model, {"schema_version": 1, "input_ids": [0]})
    try:
        layers = _layers(model)
    except ValueError as exc:
        return {"passed": False, "errors": [str(exc)], "inspection_executed_forward": False}
    if len(layers) != 36:
        errors.append(f"Mistral Small 4 requires 36 decoder blocks, found {len(layers)}")
    for i, layer in enumerate(layers):
        try:
            runner = validator._validate_runner(layer, i)
            experts = runner.routed_experts
            quant = experts.quant_method
            kernel = getattr(getattr(quant, "moe_kernel", None), "fused_experts", None)
            gate_weight = getattr(runner.gate, "weight", None)
            packed = {name: getattr(experts, name, None) for name in ("w13_weight", "w2_weight")}
            row = {
                "layer_index": i, "runner_class": type(runner).__name__,
                "quant_method_class": type(quant).__name__,
                "nvfp4_backend": str(getattr(getattr(quant, "nvfp4_backend", None), "value", getattr(quant, "nvfp4_backend", None))),
                "use_a16": getattr(quant, "use_a16", None),
                "actual_expert_kernel": type(kernel).__name__,
                "gate_shape": list(gate_weight.shape) if torch.is_tensor(gate_weight) else None,
                "gate_dtype": str(gate_weight.dtype) if torch.is_tensor(gate_weight) else None,
                "packed_expert_weights": {name: {
                    "dtype": str(value.dtype) if torch.is_tensor(value) else None,
                    "shape": list(value.shape) if torch.is_tensor(value) else None,
                } for name, value in packed.items()},
                "experts_per_token": getattr(runner.moe_config, "experts_per_token", None),
            }
            if row["runner_class"] != "MoERunner":
                errors.append(f"layer {i} is not the pinned MoERunner implementation")
            if row["quant_method_class"] != "CompressedTensorsW4A4Nvfp4MoEMethod":
                errors.append(f"layer {i} does not use the native compressed-tensors NVFP4 MoE method")
            if row["use_a16"] is not False or row["nvfp4_backend"] != "FLASHINFER_CUTEDSL":
                errors.append(f"layer {i} selected an A16 or different MoE backend")
            if row["actual_expert_kernel"] != "FlashInferCuteDSLExperts":
                errors.append(f"layer {i} did not instantiate the selected modular CuTe-DSL kernel")
            if row["gate_shape"] != [128, 4096] or row["gate_dtype"] != "torch.bfloat16":
                errors.append(f"layer {i} gate must be BF16 with shape [128,4096]")
            if row["experts_per_token"] != 4:
                errors.append(f"layer {i} must dispatch top-4 experts")
            for name, value in packed.items():
                if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 3 or value.shape[0] != 128:
                    errors.append(f"layer {i} {name} does not retain 128 packed uint8 experts")
            row["shared_linears"] = {}
            shared = getattr(runner._shared_experts, "_layer", None)
            for name in ("gate_up_proj", "down_proj"):
                linear = getattr(shared, name, None)
                scheme = getattr(linear, "scheme", None)
                weight = getattr(linear, "weight", None)
                detail = {
                    "scheme": type(scheme).__name__, "use_a16": getattr(scheme, "use_a16", None),
                    "kernel": type(getattr(scheme, "kernel", None)).__name__,
                    "weight_dtype": str(weight.dtype) if torch.is_tensor(weight) else None,
                    "has_activation_scale": torch.is_tensor(getattr(linear, "input_global_scale", None)),
                }
                row["shared_linears"][name] = detail
                if (detail["scheme"] != "CompressedTensorsW4A4Fp4" or detail["use_a16"] is not False
                        or detail["weight_dtype"] != "torch.uint8" or not detail["has_activation_scale"]):
                    errors.append(f"layer {i} shared {name} must retain native packed W4A4 NVFP4")
                # A scheme can retain the W4A4 metadata while selecting an
                # emulation or weight-only kernel. Refuse those actual paths.
                lowered = detail["kernel"].lower()
                if not lowered or any(term in lowered for term in ("a16", "marlin", "emulation", "nonetype")):
                    errors.append(f"layer {i} shared {name} selected a non-native W4A4 kernel")
            rows.append(row)
        except (ValueError, AttributeError, TypeError) as exc:
            errors.append(f"layer {i}: {exc}")
    return {
        "passed": not errors, "errors": errors, "total_blocks": len(layers),
        "router_block_indices": list(range(len(layers))), "layers": rows,
        "vllm_source_revision": VLLM_SOURCE_REVISION,
        "required_moe_backend": "FLASHINFER_CUTEDSL", "required_activation_precision": "NVFP4 W4A4",
        "moe_output_semantics": "finalized_scaled_routed_output_before_shared_add",
        "post_moe_residual_semantics": "fp32_sum_of_delayed_hidden_and_residual",
        "inspection_executed_forward": False,
    }
