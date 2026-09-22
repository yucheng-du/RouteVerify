"""OLMoE Hugging Face adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.data.prompts import (
    PromptRender,
    first_error,
    plain_trace,
    step_binary,
    tokenize_render,
)
from routeverify.data.schema import TokenizedTrace, TraceExample, stable_hash
from routeverify.eval.candidate_scoring import (
    CandidateScore,
    encode_prompt_candidate_hf,
    score_candidates_hf,
    score_token_sequence,
)
from routeverify.instrumentation.hooks import atomic_forward_hooks
from routeverify.instrumentation.patching import router_manipulation_metrics
from routeverify.instrumentation.sites import (
    CaptureSpec,
    LayerSite,
    PatchBundle,
    PatchRequest,
    PatchSpec,
    SiteKind,
    patch_specs,
    site_key,
)
from routeverify.models.base import ActivationBatch, GenerationConfig, ModelOutcome, PreparedBatch
from routeverify.models.discovery import DiscoveryReport, discover_from_model_id
from routeverify.models.moe import (
    DEFAULT_MOE_CONFIG_PATHS,
    MoEBlockLayout,
    RouterOutput,
    RouterSemantics,
    build_moe_sites,
    canonical_site_manifest,
    preflight_moe_adapter,
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by model users
        raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc
    return torch


def _torch_dtype(dtype: str) -> Any:
    torch = _require_torch()
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype: {dtype}") from exc


def _olmoe_route_from_logits(gate_module: Any, router_logits: Any) -> RouterOutput:
    """Reproduce the native OLMoE/Qwen eager gate's softmax routing contract."""

    torch = _require_torch()
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_weights, router_ids = torch.topk(router_probs, gate_module.top_k, dim=-1)
    if gate_module.norm_topk_prob:
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
    return RouterOutput(
        logits=router_logits,
        topk_weights=router_weights.to(router_logits.dtype),
        topk_ids=router_ids,
    )


class OLMoEAdapter:
    alias = "olmoe"
    model_id = "allenai/OLMoE-1B-7B-0924-Instruct"
    moe_config_paths = DEFAULT_MOE_CONFIG_PATHS
    router_semantics = RouterSemantics(
        identifier="olmoe_softmax_topk_v1",
        route_from_logits=_olmoe_route_from_logits,
    )

    def __init__(self) -> None:
        self.config: ModelConfig | None = None
        self.discovery_report: DiscoveryReport | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load(self, config: ModelConfig) -> None:
        """Load tokenizer and full model weights for behavior/capture runs."""

        ensure_external_cache()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised by model users
            raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc

        model_factory = cast(Any, AutoModelForCausalLM)
        tokenizer_factory = cast(Any, AutoTokenizer)
        tokenizer = tokenizer_factory.from_pretrained(
            config.model_id,
            revision=config.tokenizer_revision or config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "revision": config.revision,
            "cache_dir": config.cache_dir,
            "trust_remote_code": config.trust_remote_code,
            "device_map": config.device_map,
            "dtype": _torch_dtype(config.dtype),
        }
        if config.attn_implementation is not None:
            kwargs["attn_implementation"] = config.attn_implementation
        model = model_factory.from_pretrained(config.model_id, **kwargs)
        model.eval()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.discovery_report = discover_from_model_id(
            config.model_copy(update={"load_weights": False}),
            config_paths=self.moe_config_paths,
        )

    def format_prompt(self, example: TraceExample, mode: str, **kwargs: Any) -> str:
        if mode == "plain_trace":
            return plain_trace(example).prompt
        if mode == "step_binary":
            return step_binary(example, step_index=int(kwargs["step_index"])).prompt
        if mode == "first_error":
            return first_error(example).prompt
        raise ValueError(f"unsupported prompt mode: {mode}")

    def tokenize_with_spans(
        self,
        prompt: str,
        step_spans: list[tuple[int, int]],
    ) -> TokenizedTrace:
        tokenizer = self._require_tokenizer()
        render = PromptRender(
            prompt=prompt,
            step_char_spans=tuple(step_spans),
            query_char_span=None,
        )
        return tokenize_render(
            render,
            sample_id=f"prompt/{stable_hash({'prompt': prompt, 'spans': step_spans})}",
            tokenizer=tokenizer,
        )

    def discover_layers(self) -> list[LayerSite]:
        return build_moe_sites(self.moe_block_layout(), self._module_name)

    def moe_block_layout(self) -> MoEBlockLayout:
        """Return absolute routed-block indices; sparse adapters override the index hook."""

        return MoEBlockLayout(
            total_block_count=len(self._layers()),
            router_block_indices=self._router_block_indices(),
        )

    def _router_block_indices(self) -> tuple[int, ...]:
        """OLMoE and Qwen route every block; hybrid subclasses override this method."""

        return tuple(range(len(self._layers())))

    def static_preflight(self, config: ModelConfig, hf_config: Any) -> dict[str, Any]:
        """Run deterministic config/site preflight on empty or synthetic modules."""

        return preflight_moe_adapter(adapter=self, model_config=config, hf_config=hf_config)

    def score_candidates(self, prompt: str, candidates: list[str]) -> dict[str, CandidateScore]:
        return score_candidates_hf(
            model=self._require_model(),
            tokenizer=self._require_tokenizer(),
            prompt=self._chat_completion_prompt(prompt),
            candidates=candidates,
            device=str(self._input_device()),
        )

    def score_candidates_with_patch(
        self,
        prompt: str,
        candidates: list[str],
        patch: PatchSpec,
    ) -> dict[str, CandidateScore]:
        """Score candidate continuations while applying one activation patch."""

        return {
            candidate: self._score_candidate_with_patch(prompt, candidate, patch)
            for candidate in candidates
        }

    def score_candidates_with_candidate_patch_bundles(
        self,
        prompt: str,
        candidates: list[str],
        *,
        bundles_by_candidate: Mapping[str, PatchRequest],
    ) -> tuple[dict[str, CandidateScore], dict[str, dict[str, Any]]]:
        """Score each candidate under one simultaneous patch request."""

        scores: dict[str, CandidateScore] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if candidate not in bundles_by_candidate:
                raise ValueError(f"missing patch bundle for candidate {candidate!r}")
            score, candidate_diagnostics = self._score_candidate_with_patch_diagnostics(
                prompt,
                candidate,
                bundles_by_candidate[candidate],
            )
            scores[candidate] = score
            diagnostics[candidate] = candidate_diagnostics
        return scores, diagnostics

    def score_candidates_with_candidate_patches(
        self,
        prompt: str,
        candidates: list[str],
        *,
        site: LayerSite,
        token_position: int,
        replacements_by_candidate: Mapping[str, object],
    ) -> dict[str, CandidateScore]:
        """Score candidates with replacements captured under each candidate input length."""

        scores, _diagnostics = self.score_candidates_with_candidate_patches_diagnostics(
            prompt,
            candidates,
            site=site,
            token_position=token_position,
            replacements_by_candidate=replacements_by_candidate,
        )
        return scores

    def score_candidates_with_candidate_patches_diagnostics(
        self,
        prompt: str,
        candidates: list[str],
        *,
        site: LayerSite,
        token_position: int,
        replacements_by_candidate: Mapping[str, object],
    ) -> tuple[dict[str, CandidateScore], dict[str, dict[str, Any]]]:
        """Score candidate-specific single-site patches and retain native hook diagnostics."""

        scores: dict[str, CandidateScore] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if candidate not in replacements_by_candidate:
                raise ValueError(f"missing replacement for candidate {candidate!r}")
            patch = PatchSpec(
                site=site,
                token_position=token_position,
                replacement=replacements_by_candidate[candidate],
            )
            score, candidate_diagnostics = self._score_candidate_with_patch_diagnostics(
                prompt, candidate, patch
            )
            scores[candidate] = score
            diagnostics[candidate] = candidate_diagnostics
        return scores, diagnostics

    def _score_candidate_with_patch(
        self,
        prompt: str,
        candidate: str,
        patch: PatchSpec,
    ) -> CandidateScore:
        score, _diagnostics = self._score_candidate_with_patch_diagnostics(prompt, candidate, patch)
        return score

    def _score_candidate_with_patch_diagnostics(
        self,
        prompt: str,
        candidate: str,
        patch: PatchRequest,
    ) -> tuple[CandidateScore, dict[str, Any]]:
        inputs, target_start, target_end, target_ids = self._encode_prompt_candidate(
            prompt, candidate
        )
        outputs, diagnostics = self._run_with_patch_inputs_diagnostics(inputs, patch)
        start = target_start - 1
        end = target_end - 1
        selected_logits = outputs.logits[0, start:end, :].detach().float().cpu().numpy()
        return (
            score_token_sequence(
                logits=selected_logits,
                target_ids=target_ids,
                candidate=candidate,
            ),
            diagnostics,
        )

    def generate(self, prompt: str, generation: GenerationConfig) -> str:
        torch = _require_torch()
        model = self._require_model()
        tokenizer = self._require_tokenizer()
        inputs = tokenizer(self._chat_completion_prompt(prompt), return_tensors="pt").to(
            self._input_device()
        )
        kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.temperature > 0,
            "temperature": generation.temperature if generation.temperature > 0 else None,
            "top_p": generation.top_p,
            "use_cache": True,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **kwargs)
        continuation = output_ids[0, inputs["input_ids"].shape[1] :]
        return cast(
            str,
            tokenizer.decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )

    def capture(self, batch: PreparedBatch, spec: CaptureSpec) -> ActivationBatch:
        capture, _logits = self._capture_inputs(self._encode_batch(batch), spec)
        return capture

    def forward_logits(self, batch: PreparedBatch) -> Any:
        """Run a plain forward pass and return detached logits."""

        model = self._require_model()
        inputs = self._encode_batch(batch)
        torch = _require_torch()
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, output_router_logits=False)
        return outputs.logits.detach().cpu()

    def capture_with_logits(
        self, batch: PreparedBatch, spec: CaptureSpec
    ) -> tuple[ActivationBatch, Any]:
        """Capture activations and return the logits from the hooked forward pass."""

        return self._capture_inputs(self._encode_batch(batch), spec)

    def capture_candidate_activations(
        self,
        prompt: str,
        candidates: list[str],
        spec: CaptureSpec,
    ) -> dict[str, ActivationBatch]:
        """Capture prompt-token activations under the exact inputs used for candidate scoring."""

        if not candidates:
            raise ValueError("candidates must be non-empty")
        return {
            candidate: self._capture_inputs(
                self._encode_prompt_candidate(prompt, candidate)[0], spec
            )[0]
            for candidate in candidates
        }

    def _capture_inputs(
        self, inputs: dict[str, Any], spec: CaptureSpec
    ) -> tuple[ActivationBatch, Any]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        arrays: dict[str, Any] = {}
        handles: list[Any] = []

        def store(site: LayerSite, tensor: Any) -> None:
            sequence_tensor = self._as_sequence_tensor(tensor, batch_size, sequence_length)
            selected = sequence_tensor[:, list(spec.token_positions), ...].detach()
            if selected.is_floating_point():
                selected = selected.to(_torch_dtype(spec.dtype))
            arrays[self._site_key(site)] = selected.cpu()

        for site in spec.sites:
            module = self._site_module(site)
            if site.kind == "pre_moe_residual":

                def pre_moe_hook(
                    _module: Any, args: tuple[Any, ...], *, site: LayerSite = site
                ) -> None:
                    store(site, args[0])

                handles.append(module.register_forward_pre_hook(pre_moe_hook))
            elif site.kind == "router_input":

                def router_input_hook(
                    _module: Any, args: tuple[Any, ...], *, site: LayerSite = site
                ) -> None:
                    store(site, args[0])

                handles.append(module.register_forward_pre_hook(router_input_hook))
            elif site.kind in {"router_logits", "router_topk_weights", "router_topk_ids"}:

                def router_hook(
                    module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    site: LayerSite = site,
                ) -> None:
                    router_output = self._unpack_router_output(module, output)
                    if site.kind == "router_logits":
                        store(site, router_output.logits)
                    elif site.kind == "router_topk_weights":
                        store(site, router_output.topk_weights)
                    else:
                        store(site, router_output.topk_ids)

                handles.append(module.register_forward_hook(router_hook))
            elif site.kind in {"moe_output", "post_moe_residual"}:

                def output_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    site: LayerSite = site,
                ) -> None:
                    store(site, output)

                handles.append(module.register_forward_hook(output_hook))
            else:  # pragma: no cover - exhaustive for current SiteKind
                raise ValueError(f"unsupported capture site: {site.kind}")

        try:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, output_router_logits=False)
            layout = self.moe_block_layout()
            metadata = {
                "logits_shape": tuple(outputs.logits.shape),
                "moe_layout": layout.to_metadata(),
                "router_semantics_id": self.router_semantics.identifier,
                "sites": canonical_site_manifest(spec.sites, layout=layout),
            }
            logits = outputs.logits.detach().cpu()
        finally:
            for handle in handles:
                handle.remove()

        return ActivationBatch(arrays=arrays, metadata=metadata), logits

    def run_with_patch(self, batch: PreparedBatch, patch: PatchSpec) -> ModelOutcome:
        outputs, diagnostics = self._run_with_patch_inputs_diagnostics(
            self._encode_batch(batch), patch
        )
        return ModelOutcome(
            logits=outputs.logits.detach().cpu(),
            metadata={
                "patched_site": self._site_key(patch.site),
                "token_position": patch.token_position,
                "patch_diagnostics": diagnostics,
            },
        )

    def run_with_patch_bundle(self, batch: PreparedBatch, bundle: PatchBundle) -> ModelOutcome:
        """Apply every bundle member during the same forward pass."""

        outputs, diagnostics = self._run_with_patch_inputs_diagnostics(
            self._encode_batch(batch), bundle
        )
        return ModelOutcome(
            logits=outputs.logits.detach().cpu(),
            metadata={"patch_diagnostics": diagnostics},
        )

    def _run_with_patch_inputs(self, inputs: dict[str, Any], patch: PatchRequest) -> Any:
        outputs, _diagnostics = self._run_with_patch_inputs_diagnostics(inputs, patch)
        return outputs

    def _run_with_patch_inputs_diagnostics(
        self,
        inputs: dict[str, Any],
        request: PatchRequest,
    ) -> tuple[Any, dict[str, Any]]:
        torch = _require_torch()
        model = self._require_model()
        batch_size, sequence_length = inputs["input_ids"].shape
        patches = patch_specs(request)
        resolved: list[tuple[PatchSpec, Any]] = []
        for patch in patches:
            if patch.token_position >= sequence_length:
                raise IndexError(
                    f"{site_key(patch.site)} token_position {patch.token_position} "
                    f"out of range for sequence length {sequence_length}"
                )
            if patch.site.kind not in {"moe_output", "post_moe_residual", "router_logits"}:
                raise ValueError(f"patching site {patch.site.kind!r} is not implemented yet")
            resolved.append((patch, self._site_module(patch.site)))

        hook_fire_counts = {site_key(patch.site): 0 for patch in patches}
        manipulation_by_site: dict[str, dict[str, float]] = {}
        registrations: list[tuple[Any, Any]] = []
        for patch, module in resolved:
            key = site_key(patch.site)
            if patch.site.kind in {"moe_output", "post_moe_residual"}:

                def output_patch_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    patch: PatchSpec = patch,
                    key: str = key,
                ) -> Any:
                    hook_fire_counts[key] += 1
                    return self._replace_sequence_slice(output, patch, batch_size, sequence_length)

                registrations.append((module, output_patch_hook))
            else:

                def router_logits_hook(
                    gate_module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    patch: PatchSpec = patch,
                    key: str = key,
                ) -> Any:
                    router_output = self._unpack_router_output(
                        gate_module,
                        output,
                        site_key_value=key,
                    )
                    router_logits = router_output.logits
                    hook_fire_counts[key] += 1
                    receiver_sequence = self._as_sequence_tensor(
                        router_logits, batch_size, sequence_length
                    )
                    receiver_ids_sequence = self._as_sequence_tensor(
                        router_output.topk_ids, batch_size, sequence_length
                    )
                    if bool(patch.metadata.get("forced_reroute")):
                        patched_sequence = receiver_sequence.clone()
                        target = patched_sequence[:, patch.token_position, ...]
                        original_ids = receiver_ids_sequence[:, patch.token_position, ...]
                        if not target.dtype.is_floating_point:
                            raise TypeError(f"{key} forced reroute requires floating router logits")
                        limits = torch.finfo(target.dtype)
                        selected = torch.zeros_like(target, dtype=torch.bool)
                        selected.scatter_(-1, original_ids, True)
                        # Every originally selected expert receives the lowest
                        # representable value and every unselected expert the
                        # highest. This guarantees a membership change even for
                        # float16/bfloat16 ties without relying on epsilon math.
                        forced_target = torch.where(
                            selected,
                            torch.full_like(target, limits.min),
                            torch.full_like(target, limits.max),
                        )
                        patched_sequence[:, patch.token_position, ...] = forced_target
                        patched_logits = patched_sequence.reshape_as(router_logits)
                    else:
                        patched_logits = self._replace_sequence_slice(
                            router_logits,
                            patch,
                            batch_size,
                            sequence_length,
                        ).reshape_as(router_logits)
                    patched_sequence = self._as_sequence_tensor(
                        patched_logits, batch_size, sequence_length
                    )
                    patched_output = self._route_from_logits(gate_module, patched_logits)
                    patched_ids_sequence = self._as_sequence_tensor(
                        patched_output.topk_ids, batch_size, sequence_length
                    )
                    manipulation_by_site[key] = router_manipulation_metrics(
                        receiver_sequence[:, patch.token_position, ...]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        patched_sequence[:, patch.token_position, ...]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        receiver_topk_ids=receiver_ids_sequence[
                            :, patch.token_position, ...
                        ]
                        .detach()
                        .cpu()
                        .numpy(),
                        patched_topk_ids=patched_ids_sequence[
                            :, patch.token_position, ...
                        ]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    return self._restore_router_output(gate_module, output, patched_output)

                registrations.append((module, router_logits_hook))
        with atomic_forward_hooks(registrations):
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, output_router_logits=False)
        missing = sorted(key for key, count in hook_fire_counts.items() if count == 0)
        if missing:
            raise RuntimeError(f"registered patch hooks did not fire: {missing}")
        diagnostics: dict[str, Any] = {
            "simultaneous": isinstance(request, PatchBundle),
            "registered_site_count": len(patches),
            "registered_sites": [site_key(patch.site) for patch in patches],
            "hook_fire_counts": hook_fire_counts,
            "all_registered_hooks_fired": not missing,
            "manipulation_by_site": manipulation_by_site,
            "router_semantics_id": self.router_semantics.identifier,
            "router_kl_interpretation": "softmax_projection_of_router_logits",
        }
        return outputs, diagnostics

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("adapter is not loaded")
        return self.model

    def _require_tokenizer(self) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("adapter is not loaded")
        return self.tokenizer

    def _input_device(self) -> Any:
        model = self._require_model()
        return getattr(model, "device", next(model.parameters()).device)

    def _encode_batch(self, batch: PreparedBatch) -> dict[str, Any]:
        if not batch.prompts:
            raise ValueError("batch.prompts must be non-empty")
        tokenizer = self._require_tokenizer()
        encoded = tokenizer(
            list(batch.prompts),
            padding=True,
            return_tensors="pt",
        )
        device = self._input_device()
        return {key: value.to(device) for key, value in encoded.items()}

    def _encode_prompt_candidate(
        self,
        prompt: str,
        candidate: str,
    ) -> tuple[dict[str, Any], int, int, list[int]]:
        return self._encode_rendered_prompt_candidate(
            self._chat_completion_prompt(prompt), candidate
        )

    def _encode_rendered_prompt_candidate(
        self,
        prompt: str,
        candidate: str,
    ) -> tuple[dict[str, Any], int, int, list[int]]:
        tokenizer = self._require_tokenizer()
        device = self._input_device()
        encoded = encode_prompt_candidate_hf(
            tokenizer=tokenizer,
            prompt=prompt,
            candidate=candidate,
            device=device,
        )
        return (
            encoded.inputs,
            encoded.target_token_start,
            encoded.target_token_end,
            encoded.target_ids,
        )

    def chat_completion_render(self, render: PromptRender) -> PromptRender:
        """Apply the instruct chat template and preserve auditable step spans."""

        prompt = self._chat_completion_prompt(render.prompt)
        offset = prompt.find(render.prompt)
        if offset < 0:
            raise RuntimeError("OLMoE chat template did not preserve the raw prompt text")
        from routeverify.data.prompts import shift_render_into_prompt

        return shift_render_into_prompt(render, prompt=prompt, offset=offset)

    def _chat_completion_prompt(self, prompt: str) -> str:
        tokenizer = self._require_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError("OLMoE tokenizer does not expose apply_chat_template")
        messages = [{"role": "user", "content": prompt}]
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def _layers(self) -> Any:
        model = self._require_model()
        try:
            return model.model.layers
        except AttributeError as exc:
            raise RuntimeError("loaded model does not expose model.layers") from exc

    def _layer(self, layer_index: int) -> Any:
        layers = self._layers()
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"layer_index {layer_index} out of range")
        return layers[layer_index]

    def _module_name(self, layer_index: int, kind: SiteKind) -> str:
        if kind == "pre_moe_residual":
            return f"model.layers.{layer_index}.post_attention_layernorm"
        if kind == "router_input":
            return f"model.layers.{layer_index}.mlp"
        if kind in {"router_logits", "router_topk_ids", "router_topk_weights"}:
            return f"model.layers.{layer_index}.mlp.gate"
        if kind == "moe_output":
            return f"model.layers.{layer_index}.mlp"
        if kind == "post_moe_residual":
            return f"model.layers.{layer_index}"
        raise ValueError(f"unsupported site kind: {kind}")

    def _site_module(self, site: LayerSite) -> Any:
        layer = self._layer(site.layer_index)
        if site.kind == "pre_moe_residual":
            return layer.post_attention_layernorm
        if site.kind == "router_input":
            return layer.mlp
        if site.kind in {"router_logits", "router_topk_ids", "router_topk_weights"}:
            return layer.mlp.gate
        if site.kind == "moe_output":
            return layer.mlp
        if site.kind == "post_moe_residual":
            return layer
        raise ValueError(f"unsupported site kind: {site.kind}")

    def _unpack_router_output(
        self,
        router_module: Any,
        output: Any,
        *,
        site_key_value: str = "router",
    ) -> RouterOutput:
        """Normalize OLMoE's gate tuple; adapters with other outputs override this hook."""

        del router_module
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError(
                f"{site_key_value} gate output must be (logits, weights, ids); "
                f"got {type(output).__name__}"
            )
        return RouterOutput(
            logits=output[0],
            topk_weights=output[1],
            topk_ids=output[2],
        )

    def _route_from_logits(self, router_module: Any, router_logits: Any) -> RouterOutput:
        """Delegate routing; adapters may override when semantics require adapter state."""

        return self.router_semantics.route(router_module, router_logits)

    def _restore_router_output(
        self,
        router_module: Any,
        original: Any,
        routed: RouterOutput,
    ) -> Any:
        """Restore OLMoE's gate tuple after adapter-native routing recomputation."""

        del router_module
        del original
        return routed.logits, routed.topk_weights, routed.topk_ids

    def _site_key(self, site: LayerSite) -> str:
        return site_key(site)

    def _as_sequence_tensor(self, tensor: Any, batch_size: int, sequence_length: int) -> Any:
        if (
            tensor.ndim >= 3
            and tensor.shape[0] == batch_size
            and tensor.shape[1] == sequence_length
        ):
            return tensor
        if tensor.ndim >= 2 and tensor.shape[0] == batch_size * sequence_length:
            return tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])
        raise ValueError(
            "cannot align tensor to batch/sequence shape: "
            f"tensor_shape={tuple(tensor.shape)} batch={batch_size} seq={sequence_length}"
        )

    def _replacement_for_target(self, replacement: Any, target: Any, *, strict_shape: bool) -> Any:
        torch = _require_torch()
        replacement_tensor = (
            replacement if torch.is_tensor(replacement) else torch.as_tensor(replacement)
        )
        replacement_tensor = replacement_tensor.to(device=target.device, dtype=target.dtype)
        if replacement_tensor.shape == target.unsqueeze(1).shape:
            replacement_tensor = replacement_tensor[:, 0, ...]
        if strict_shape and replacement_tensor.shape != target.shape:
            raise ValueError(
                f"replacement shape {tuple(replacement_tensor.shape)} does not match "
                f"target shape {tuple(target.shape)}"
            )
        return replacement_tensor

    def _replace_sequence_slice(
        self,
        tensor: Any,
        patch: PatchSpec,
        batch_size: int,
        sequence_length: int,
    ) -> Any:
        sequence_tensor = self._as_sequence_tensor(tensor, batch_size, sequence_length)
        if patch.token_position < 0 or patch.token_position >= sequence_length:
            raise IndexError("patch token_position out of range")
        patched = sequence_tensor.clone()
        target = patched[:, patch.token_position, ...]
        replacement = self._replacement_for_target(
            patch.replacement,
            target,
            strict_shape=patch.strict_shape,
        )
        patched[:, patch.token_position, ...] = replacement
        return patched
