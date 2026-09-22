"""Offline, single-request vLLM bridge for native NVFP4 research hooks.

Importing this module never imports vLLM, provisions a device, or downloads files.
Native execution is explicit and constrained to the inspected modular SM100 path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

SOURCE_MANIFEST = Path(__file__).resolve().parents[3] / "tests/fixtures/mistral_small4_nvfp4/backend_source_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_backend_environment(config: Any) -> dict[str, Any]:
    """Return actionable import/hardware/source failures without loading weights."""
    import torch

    manifest = json.loads(SOURCE_MANIFEST.read_text())
    errors = []
    if platform.system() != "Linux":
        errors.append("native vLLM NVFP4 execution requires Linux; CPU/reference checks remain available")
    devices = []
    if not torch.cuda.is_available():
        errors.append("CUDA is unavailable; no native checkpoint load/forward attempted")
    else:
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            devices.append({"index": index, "name": torch.cuda.get_device_name(index), "capability": list(capability)})
        if devices[0]["capability"][0] != 10:
            errors.append("selected flashinfer_cutedsl native W4A4 MoE path requires SM100 family; no A16 fallback")
    found = importlib.util.find_spec("vllm")
    source_status = {}
    if found is None or not found.submodule_search_locations:
        errors.append("vLLM is not installed; install the pinned source in the user-provided CUDA environment")
    else:
        root = Path(next(iter(found.submodule_search_locations)))
        for name, expected in manifest["vllm_source_sha256"].items():
            path = root / name
            actual = _sha(path) if path.is_file() else None
            source_status[name] = {"sha256": actual, "matches": actual == expected}
        if not all(row["matches"] for row in source_status.values()):
            errors.append("installed hook-relevant vLLM source differs from the inspected pinned revision")
    if getattr(config, "tensor_parallel_size", 1) != 1:
        errors.append("this research worker currently requires TP=1; multi-rank residual/dispatch is unvalidated")
    if getattr(config, "moe_backend", None) != "flashinfer_cutedsl":
        errors.append("research config must select modular flashinfer_cutedsl; automatic backend substitution is disabled")
    return {"passed": not errors, "errors": errors, "devices": devices,
            "vllm_source_status": source_status, "expected_vllm_revision": manifest["vllm_revision"],
            "checkpoint_forward_executed": False, "weights_downloaded": False}


def inspect_checkpoint_cache(config: Any) -> dict[str, Any]:
    """Verify immutable metadata and existing shards; do not read large weights."""
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    override = getattr(config, "checkpoint_path", None)
    root = Path(override) if override else Path(config.cache_dir) / ("models--" + config.model_id.replace("/", "--")) / "snapshots" / config.revision
    errors = []
    for name, expected in (("params.json", manifest["params_sha256"]), ("consolidated.safetensors.index.json", manifest["index_sha256"])):
        if not (root / name).is_file() or _sha(root / name) != expected:
            errors.append(f"missing or mismatched pinned metadata: {name}")
    missing = [name for name in manifest["shards"] if not (root / name).is_file()]
    if missing:
        errors.append(f"{len(missing)} official consolidated shards are absent; weights are never downloaded by this runner")
    return {"passed": not errors, "errors": errors, "path": str(root), "missing_shards": missing,
            "existing_shard_bytes": sum((root / name).stat().st_size for name in manifest["shards"] if (root / name).is_file()),
            "weight_content_sha256_checked": False, "metadata_sha256_checked": True}


def _worker_inspect(model: Any) -> dict[str, Any]:
    from routeverify.models.mistral_vllm_hooks import inspect_model
    return inspect_model(model)


def _worker_begin(model: Any, request: dict[str, Any]) -> Any:
    from routeverify.models.mistral_vllm_hooks import begin
    return begin(model, request)


def _worker_finish(model: Any) -> Any:
    from routeverify.models.mistral_vllm_hooks import finish
    return finish(model)


def _worker_cleanup(model: Any) -> Any:
    from routeverify.models.mistral_vllm_hooks import cleanup
    return cleanup(model)


def _worker_memory(_model: Any, *, reset: bool = False) -> dict[str, Any]:
    import torch
    result = {"scope": "vllm_worker_pytorch_cuda_allocator", "devices": [
        {"index": i, "allocated_bytes": torch.cuda.max_memory_allocated(i),
         "reserved_bytes": torch.cuda.max_memory_reserved(i)} for i in range(torch.cuda.device_count())]}
    if reset:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
    return result


class VllmMistralBackend:
    """One native full prefill per candidate, with scoped worker hooks.

    prompt_logprobs are raw teacher-forced probabilities. max_tokens=1 requests
    one unused output token from that same prefill, never a second decode pass.
    The worker rejects any extra/chunked/padded model forward.
    """

    def __init__(self, engine: Any, *, max_model_len: int, sampling_factory: Any = None):
        self.engine = engine
        self.max_model_len = max_model_len
        self.sampling_factory = sampling_factory
        self._lock = threading.Lock()
        self.request_records: list[dict[str, Any]] = []
        self.runtime_report: dict[str, Any] = {}

    @classmethod
    def load(cls, config: Any) -> VllmMistralBackend:
        environment = inspect_backend_environment(config)
        if not environment["passed"]:
            raise RuntimeError("Mistral native backend preflight: " + "; ".join(environment["errors"]))
        cache = inspect_checkpoint_cache(config)
        if not cache["passed"]:
            raise RuntimeError("Mistral checkpoint cache: " + "; ".join(cache["errors"]))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
        from vllm import LLM, SamplingParams

        maximum = int(getattr(config, "max_model_len", 8192))
        start = time.monotonic()
        engine = LLM(
            model=cache["path"], tokenizer=cache["path"], tokenizer_mode="mistral",
            # The driver uses pinned mistral-common and supplies only token IDs.
            # Avoid a second tokenizer initialization/reinterpretation in vLLM.
            skip_tokenizer_init=True, language_model_only=True,
            config_format="mistral", load_format="mistral", trust_remote_code=False,
            dtype="bfloat16", quantization="compressed-tensors", tensor_parallel_size=1,
            pipeline_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False,
            enforce_eager=True, enable_prefix_caching=False, enable_chunked_prefill=False,
            max_num_seqs=1, max_model_len=maximum, max_num_batched_tokens=maximum,
            logprobs_mode="raw_logprobs", max_logprobs=1, seed=int(getattr(config, "seed", 0)),
            gpu_memory_utilization=float(getattr(config, "gpu_memory_utilization", .85)),
            compilation_config={"mode": 0}, kernel_config={"moe_backend": "flashinfer_cutedsl"},
        )
        backend = cls(engine, max_model_len=maximum, sampling_factory=SamplingParams)
        runtime = backend._one(engine.apply_model(_worker_inspect))
        if runtime.get("passed") is not True:
            raise RuntimeError(f"loaded Mistral research boundaries failed: {runtime}")
        backend.runtime_report = {"environment": environment, "checkpoint_cache": cache,
                                  "worker": runtime, "load_seconds": time.monotonic() - start,
                                  "research_candidate_forward_executed": False,
                                  "initialization_may_execute_profiling_forwards": True}
        try:
            backend.runtime_report["worker_memory_after_load"] = backend.resource_metrics()
        except Exception as exc:
            backend.runtime_report["worker_memory_after_load_error"] = f"{type(exc).__name__}: {exc}"
        return backend

    @staticmethod
    def _one(values: list[Any]) -> Any:
        if len(values) != 1:
            raise RuntimeError("research bridge expected exactly one vLLM worker result")
        return values[0]

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        ids = request["input_ids"]
        if len(ids) < 2 or len(ids) + 1 > self.max_model_len:
            raise ValueError("full prompt/candidate exceeds configured context (including one unused sample); no truncation")
        if self.sampling_factory is None:
            from vllm import SamplingParams
            self.sampling_factory = SamplingParams
        params = self.sampling_factory(temperature=0., max_tokens=1, prompt_logprobs=1,
                                       detokenize=False, ignore_eos=True)
        record = {"request_id": request["request_id"], "input_tokens": len(ids),
                  "stage": "begin", "passed": False, "completed_model_forwards": 0}
        start = time.monotonic()
        with self._lock:
            try:
                self._one(self.engine.apply_model(partial(_worker_begin, request=request)))
                record["stage"] = "prefill"
                outputs = self.engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
                record["stage"] = "collect"
                hooks = self._one(self.engine.apply_model(_worker_finish))
                record["completed_model_forwards"] = hooks["diagnostics"].get("model_forward_count", 0)
                record["attempted_model_forwards"] = hooks["diagnostics"].get("attempted_model_forward_count", 0)
                record["worker_diagnostics"] = hooks["diagnostics"]
                record["partial_captures"] = hooks.get("captures", {}) if not hooks["diagnostics"].get("passed") else {}
                if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != ids:
                    raise RuntimeError("vLLM changed or batched the supplied teacher-forced prompt IDs")
                logs = outputs[0].prompt_logprobs
                if logs is None or len(logs) != len(ids):
                    raise RuntimeError("vLLM did not return one raw prompt-logprob entry per input token")
                probabilities: list[float | None] = [None]
                for position, token_id in enumerate(ids[1:], start=1):
                    entry = logs[position]
                    if entry is None or token_id not in entry:
                        raise RuntimeError(f"actual prompt token logprob absent at {position}")
                    probabilities.append(float(entry[token_id].logprob))
                if hooks["diagnostics"].get("passed") is not True:
                    raise RuntimeError(f"worker capture/patch verification failed: {hooks['diagnostics']}")
                record.update(stage="complete", passed=True)
                return {**hooks, "prompt_token_logprobs": probabilities, "input_ids": ids}
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    recovery = self._one(self.engine.apply_model(_worker_cleanup))
                    partial_diagnostics = recovery.get("partial_diagnostics")
                    if partial_diagnostics is not None:
                        record["partial_worker_diagnostics"] = partial_diagnostics
                        record["completed_model_forwards"] = partial_diagnostics.get("model_forward_count", 0)
                        record["attempted_model_forwards"] = partial_diagnostics.get("attempted_model_forward_count", 0)
                        record["partial_captures"] = recovery.get("partial_captures", {})
                    record["hooks_cleaned"] = True
                except BaseException as exc:
                    record["hooks_cleaned"] = False
                    record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    # Preserve the primary inference/validation error. A cleanup
                    # failure after an otherwise successful call is itself fatal.
                    if "error" not in record:
                        record["passed"] = False
                        raise
                finally:
                    record["seconds"] = time.monotonic() - start
                    self.request_records.append(record)

    def resource_metrics(self) -> dict[str, Any]:
        return self._one(self.engine.apply_model(_worker_memory))
