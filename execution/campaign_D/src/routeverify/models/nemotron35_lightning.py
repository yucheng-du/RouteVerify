"""Pinned NVIDIA Nemotron 3.5 Lightning adapter.

The checkpoint uses the same native Transformers Nemotron-H execution model as
Nemotron-3 Nano, but it is kept as a separate registry arm and revision contract.
In particular, Lightning disables the latent bottleneck, so ``fc2_latent_proj``
is an identity placed after the weighted routed-expert aggregate.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from typing import Any, cast

from routeverify.config import ModelConfig, ensure_external_cache
from routeverify.models.base import ActivationBatch
from routeverify.models.moe import (
    MoEBlockLayout,
    build_moe_sites,
    preflight_moe_layout,
)
from routeverify.models.nemotron3_nano import (
    Nemotron3NanoAdapter,
    _config_value,
)


class Nemotron35LightningAdapter(Nemotron3NanoAdapter):
    """Adapter for the exact BF16 Nemotron 3.5 Lightning checkpoint.

    The common-core ``moe_output`` endpoint remains the final *routed-only*
    aggregate, matching the already accepted Gemma 4 and Nemotron-3 Nano arms.
    The receiver's shared-expert branch is therefore preserved. The complete
    MoE module output (routed plus shared) is ``layer.mixer``; it is recorded as
    an architectural boundary but is not mislabeled as the common endpoint.
    """

    alias = "nemotron35_lightning"
    model_id = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    pinned_revision = "a9904d24bcc1d289a1950fa9d2b978c47cf903b9"
    pinned_transformers_version = "5.12.1"
    capture_loading_info = True

    @classmethod
    def _validate_requested_config(cls, config: ModelConfig) -> None:
        super()._validate_requested_config(config)
        if config.trust_remote_code:
            raise ValueError(
                "Lightning uses native Transformers code; trust_remote_code must be false"
            )
        expected_template = config.expected_chat_template_sha256
        if expected_template is None or len(expected_template) != 64:
            raise ValueError("Lightning config must pin the chat template SHA-256")

    @classmethod
    def static_config_preflight(
        cls,
        model_config: ModelConfig,
        hf_config: Mapping[str, Any] | Any,
    ) -> dict[str, Any]:
        """Validate the pinned public config and site contract without weights."""

        architecture = cls.static_architecture_preflight(
            hf_config,
            expected=model_config,
        )
        block_types = tuple(str(value) for value in architecture["block_types"])
        router_indices = tuple(int(value) for value in architecture["router_block_indices"])
        layout = MoEBlockLayout(
            total_block_count=len(block_types),
            router_block_indices=router_indices,
        )
        adapter = cls()
        report = preflight_moe_layout(
            model_config=model_config,
            hf_config=hf_config,
            layout=layout,
            sites=build_moe_sites(layout, adapter._module_name),
            router_semantics_id=adapter.router_semantics.identifier,
            config_paths=adapter.moe_config_paths,
            adapter_class=f"{cls.__module__}.{cls.__qualname__}",
            adapter_alias=cls.alias,
            adapter_model_id=cls.model_id,
        )
        errors = list(report["errors"])

        hook_facts = {
            "moe_latent_size": _config_value(hf_config, "moe_latent_size"),
            "n_group": _optional_int(_config_value(hf_config, "n_group")),
            # ``n_groups`` belongs to the Mamba state-space mixer. It is not
            # Nemotron-H's routed-expert grouping parameter (``n_group``).
            "mamba_n_groups": _optional_int(_config_value(hf_config, "n_groups")),
            "topk_group": _optional_int(_config_value(hf_config, "topk_group")),
            "moe_shared_expert_overlap": _config_value(
                hf_config, "moe_shared_expert_overlap"
            ),
            "num_nextn_predict_layers": _optional_int(
                _config_value(hf_config, "num_nextn_predict_layers")
            ),
            "mtp_layers_block_type": list(
                _config_value(hf_config, "mtp_layers_block_type", ()) or ()
            ),
        }
        expected = {
            "moe_latent_projection": getattr(
                model_config, "expected_moe_latent_projection", None
            ),
            "n_group": getattr(model_config, "expected_n_group", None),
            "mamba_n_groups": getattr(model_config, "expected_mamba_n_groups", None),
            "topk_group": getattr(model_config, "expected_topk_group", None),
            "moe_shared_expert_overlap": getattr(
                model_config, "expected_moe_shared_expert_overlap", None
            ),
            "num_nextn_predict_layers": getattr(
                model_config, "expected_num_nextn_predict_layers", None
            ),
            "mtp_layers_block_type": getattr(
                model_config, "expected_mtp_layers_block_type", None
            ),
        }
        actual_latent_projection = hook_facts["moe_latent_size"] is not None
        if (
            expected["moe_latent_projection"] is not None
            and bool(expected["moe_latent_projection"]) != actual_latent_projection
        ):
            errors.append(
                "moe latent projection mismatch: "
                f"actual={actual_latent_projection} "
                f"expected={expected['moe_latent_projection']}"
            )
        for key in (
            "n_group",
            "mamba_n_groups",
            "topk_group",
            "num_nextn_predict_layers",
        ):
            expected_value = expected[key]
            if expected_value is not None and hook_facts[key] != int(expected_value):
                errors.append(
                    f"{key} mismatch: actual={hook_facts[key]} expected={expected_value}"
                )
        if (
            expected["moe_shared_expert_overlap"] is not None
            and hook_facts["moe_shared_expert_overlap"]
            != bool(expected["moe_shared_expert_overlap"])
        ):
            errors.append(
                "moe_shared_expert_overlap mismatch: "
                f"actual={hook_facts['moe_shared_expert_overlap']} "
                f"expected={expected['moe_shared_expert_overlap']}"
            )
        if (
            expected["mtp_layers_block_type"] is not None
            and hook_facts["mtp_layers_block_type"]
            != list(expected["mtp_layers_block_type"])
        ):
            errors.append(
                "MTP layer layout mismatch: "
                f"actual={hook_facts['mtp_layers_block_type']} "
                f"expected={expected['mtp_layers_block_type']}"
            )

        report.update(
            {
                "architecture_preflight": architecture,
                "hook_relevant_config": hook_facts,
                "hook_relevant_expected": expected,
                "moe_output_semantics": (
                    "weighted_routed_aggregate_after_fc2_identity_before_shared_expert_addition"
                ),
                "full_moe_output_boundary": "model.layers.<block>.mixer",
                "post_moe_residual_boundary": "model.layers.<block>",
                "mtp_in_main_decoder_sites": False,
                "errors": errors,
                "passed": not errors,
            }
        )
        return report

    def load(self, config: ModelConfig) -> None:
        """Load through the native Nemotron-H path and bind tokenizer bytes."""

        self._validate_requested_config(config)
        ensure_external_cache()
        try:
            import transformers
            from transformers import AutoConfig
        except ImportError as exc:  # pragma: no cover - exercised by model users
            raise RuntimeError("Install model dependencies with `uv sync --extra models`.") from exc
        if transformers.__version__ != self.pinned_transformers_version:
            raise RuntimeError(
                "Lightning Transformers version mismatch: "
                f"actual={transformers.__version__} expected={self.pinned_transformers_version}"
            )
        config_factory = cast(Any, AutoConfig)
        hf_config = config_factory.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )
        static_report = self.static_config_preflight(config, hf_config)
        if not static_report["passed"]:
            raise RuntimeError(f"Lightning static preflight failed: {static_report['errors']}")

        super().load(config)
        self._validate_chat_template_contract()
        runtime = self.runtime_inspect()
        if not runtime["lightning_latent_projections_are_identity"]:
            raise RuntimeError("Lightning routed latent projections must be nn.Identity")
        if self.discovery_report is not None:
            self.discovery_report.metadata["static_preflight"] = static_report
            self.discovery_report.metadata["runtime_inspect"] = runtime

    def _capture_inputs(
        self,
        inputs: dict[str, Any],
        spec: Any,
    ) -> tuple[ActivationBatch, Any]:
        capture, logits = super()._capture_inputs(inputs, spec)
        metadata = dict(capture.metadata)
        metadata.update(
            {
                "moe_output_semantics": (
                    "weighted_routed_aggregate_after_fc2_identity_before_shared_expert_addition"
                ),
                "full_moe_output_boundary": "model.layers.<block>.mixer",
                "post_moe_residual_boundary": "model.layers.<block>",
            }
        )
        return ActivationBatch(arrays=capture.arrays, metadata=metadata), logits

    def runtime_inspect(self) -> dict[str, Any]:
        report = super().runtime_inspect()
        identity = "torch.nn.modules.linear.Identity"
        rows = list(report["moe_layers"])
        latent_projection_classes = []
        for block in self._moe_blocks():
            mixer = self._moe_layer(block.absolute_block_index).mixer
            latent_projection_classes.append(
                {
                    "absolute_block_index": block.absolute_block_index,
                    "fc1": self._class_path(mixer.fc1_latent_proj),
                    "fc2": self._class_path(mixer.fc2_latent_proj),
                }
            )
        mtp_modules = [
            name
            for name, _module in self._require_model().named_modules()
            if "mtp" in name.split(".")
        ]
        loading_info = self.loading_info
        if not isinstance(loading_info, dict):
            loading_summary: dict[str, Any] = {
                "available": False,
                "missing_keys": [],
                "unexpected_keys": [],
                "non_mtp_unexpected_keys": [],
                "mtp_unexpected_key_count": None,
                "mismatched_keys": [],
                "error_messages": [],
                "only_expected_mtp_ignored": False,
            }
        else:
            missing_keys = [str(value) for value in loading_info.get("missing_keys", [])]
            unexpected_keys = [
                str(value) for value in loading_info.get("unexpected_keys", [])
            ]
            mismatched_keys = [
                str(value) for value in loading_info.get("mismatched_keys", [])
            ]
            error_messages = [str(value) for value in loading_info.get("error_msgs", [])]
            non_mtp_unexpected = [
                key for key in unexpected_keys if not key.startswith("mtp.")
            ]
            mtp_unexpected_count = len(unexpected_keys) - len(non_mtp_unexpected)
            loading_summary = {
                "available": True,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "non_mtp_unexpected_keys": non_mtp_unexpected,
                "mtp_unexpected_key_count": mtp_unexpected_count,
                "mismatched_keys": mismatched_keys,
                "error_messages": error_messages,
                "only_expected_mtp_ignored": (
                    not missing_keys
                    and not non_mtp_unexpected
                    and mtp_unexpected_count in {0, 270}
                    and not mismatched_keys
                    and not error_messages
                ),
            }

        model = self._require_model()
        ignore_patterns = list(
            getattr(type(model), "_keys_to_ignore_on_load_unexpected", ()) or ()
        )
        mtp_ignore_pattern_present = any("mtp" in str(value) for value in ignore_patterns)
        block_types = list(_config_value(model.config, "layers_block_type", ()) or ())
        mamba_rows: list[dict[str, Any]] = []
        fast_path_available_values: set[bool] = set()
        compile_active_values: set[bool] = set()
        for index, block_type in enumerate(block_types):
            if block_type != "mamba":
                continue
            mixer = self._layer(index).mixer
            modeling_module = sys.modules.get(type(mixer).__module__)
            fast_path_available = bool(
                getattr(modeling_module, "is_fast_path_available", False)
            )
            compile_probe = getattr(modeling_module, "is_torchdynamo_compiling", None)
            compile_active = bool(compile_probe()) if callable(compile_probe) else True
            device = str(mixer.in_proj.weight.device)
            device_type = str(mixer.in_proj.weight.device.type)
            use_mamba_kernels = bool(
                getattr(mixer.config, "use_mamba_kernels", False)
            )
            fast_path_available_values.add(fast_path_available)
            compile_active_values.add(compile_active)
            mamba_rows.append(
                {
                    "absolute_block_index": index,
                    "device": device,
                    "device_type": device_type,
                    "use_mamba_kernels": use_mamba_kernels,
                    "module_fast_path_available": fast_path_available,
                    "torchdynamo_compiling": compile_active,
                    "fast_path_ready": (
                        use_mamba_kernels
                        and fast_path_available
                        and device_type == "cuda"
                        and not compile_active
                    ),
                }
            )
        mamba_devices = {row["device"] for row in mamba_rows}
        mamba_fast_path_ready = bool(
            len(mamba_rows) == 23
            and len(mamba_devices) == 1
            and all(bool(row["fast_path_ready"]) for row in mamba_rows)
        )
        report.update(
            {
                "lightning_latent_projections_are_identity": all(
                    row["fc1"] == identity and row["fc2"] == identity
                    for row in latent_projection_classes
                ),
                "lightning_latent_projection_classes": latent_projection_classes,
                "moe_output_semantics": (
                    "weighted_routed_aggregate_after_fc2_identity_before_shared_expert_addition"
                ),
                "full_moe_output_boundary": "model.layers.<block>.mixer",
                "post_moe_residual_boundary": "model.layers.<block>",
                "mtp_in_main_decoder_sites": False,
                "mtp_runtime_modules": mtp_modules,
                "mtp_ignore_patterns": ignore_patterns,
                "mtp_ignore_pattern_present": mtp_ignore_pattern_present,
                "loading_info": loading_summary,
                "loading_info_ok": loading_summary["only_expected_mtp_ignored"],
                "mamba_fast_path": {
                    "ready": mamba_fast_path_ready,
                    "single_device": len(mamba_devices) == 1,
                    "devices": sorted(mamba_devices),
                    "module_fast_path_available_values": sorted(
                        fast_path_available_values
                    ),
                    "torchdynamo_compiling_values": sorted(compile_active_values),
                    "layers": mamba_rows,
                },
                "mamba_fast_path_ready": mamba_fast_path_ready,
            }
        )
        report["contract_matches"] = bool(
            report["contract_matches"]
            and report["lightning_latent_projections_are_identity"]
            and len(rows) == len(latent_projection_classes)
            and not mtp_modules
            and mtp_ignore_pattern_present
            and loading_summary["only_expected_mtp_ignored"]
            and mamba_fast_path_ready
        )
        return report

    def _validate_chat_template_contract(self) -> None:
        config = self.config
        tokenizer = self._require_tokenizer()
        template = getattr(tokenizer, "chat_template", None)
        expected = None if config is None else config.expected_chat_template_sha256
        if not isinstance(template, str) or not template:
            raise RuntimeError("Lightning tokenizer has no chat_template text")
        actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if expected is None or actual != expected:
            raise RuntimeError(
                "Lightning chat template hash mismatch: "
                f"actual={actual} expected={expected}"
            )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
