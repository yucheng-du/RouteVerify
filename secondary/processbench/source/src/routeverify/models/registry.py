"""Adapter registry."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class AdapterSpec:
    alias: str
    model_id: str
    module: str
    class_name: str


REGISTRY: dict[str, AdapterSpec] = {
    "olmoe": AdapterSpec(
        alias="olmoe",
        model_id="allenai/OLMoE-1B-7B-0924-Instruct",
        module="routeverify.models.olmoe",
        class_name="OLMoEAdapter",
    ),
    "qwen3_moe": AdapterSpec(
        alias="qwen3_moe",
        model_id="Qwen/Qwen3-30B-A3B",
        module="routeverify.models.qwen3_moe",
        class_name="Qwen3MoEAdapter",
    ),
    "gpt_oss_20b": AdapterSpec(
        alias="gpt_oss_20b",
        model_id="openai/gpt-oss-20b",
        module="routeverify.models.gpt_oss",
        class_name="GptOssAdapter",
    ),
    "nemotron3_nano": AdapterSpec(
        alias="nemotron3_nano",
        model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        module="routeverify.models.nemotron3_nano",
        class_name="Nemotron3NanoAdapter",
    ),
    "gemma4_moe": AdapterSpec(
        alias="gemma4_moe",
        model_id="google/gemma-4-26B-A4B-it",
        module="routeverify.models.gemma4",
        class_name="Gemma4MoEAdapter",
    ),
}


def get_adapter_spec(alias: str) -> AdapterSpec:
    try:
        return REGISTRY[alias]
    except KeyError as exc:
        raise KeyError(f"unknown adapter alias: {alias}") from exc


def create_adapter(alias: str) -> Any:
    spec = get_adapter_spec(alias)
    module = import_module(spec.module)
    adapter_class = getattr(module, spec.class_name)
    return adapter_class()
