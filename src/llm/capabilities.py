from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import ModelConfig


@dataclass(frozen=True)
class ModelCapabilities:
    transport: Literal["provider_native", "openai_compatible"]
    history_format: Literal["anthropic", "gemini", "openai"]
    structured_output_mode: Literal["native", "repair_wrapper"]
    reasoning_mode: Literal["none", "effort", "budget", "adaptive", "always_on"]
    cache_mode: Literal["none", "prefix", "gemini_cached_content"]
    cache_metrics_mode: Literal["none", "anthropic", "openai", "gemini"]
    supports_dimensions: bool = False
    shared_reasoning_budget: bool = False


def get_model_capabilities(config: ModelConfig) -> ModelCapabilities:
    """Return runtime capability metadata for the effective model config."""

    if config.transport == "openai_compatible":
        return ModelCapabilities(
            transport="openai_compatible",
            history_format="openai",
            structured_output_mode="repair_wrapper",
            reasoning_mode="none",
            cache_mode="none",
            cache_metrics_mode="openai",
        )

    prefix = config.model.split("/", 1)[0]

    if prefix == "anthropic":
        return ModelCapabilities(
            transport="provider_native",
            history_format="anthropic",
            structured_output_mode="repair_wrapper",
            reasoning_mode="budget",
            cache_mode="prefix",
            cache_metrics_mode="anthropic",
        )

    if prefix == "gemini":
        return ModelCapabilities(
            transport="provider_native",
            history_format="gemini",
            structured_output_mode="native",
            reasoning_mode="budget",
            cache_mode="gemini_cached_content",
            cache_metrics_mode="gemini",
            shared_reasoning_budget=True,
        )

    if prefix == "openai":
        return ModelCapabilities(
            transport="provider_native",
            history_format="openai",
            structured_output_mode="native",
            reasoning_mode="effort",
            cache_mode="prefix",
            cache_metrics_mode="openai",
        )

    if prefix == "groq":
        return ModelCapabilities(
            transport="provider_native",
            history_format="openai",
            structured_output_mode="repair_wrapper",
            reasoning_mode="none",
            cache_mode="none",
            cache_metrics_mode="openai",
        )

    if prefix in {"openrouter", "hosted_vllm"}:
        return ModelCapabilities(
            transport="provider_native",
            history_format="openai",
            structured_output_mode="repair_wrapper",
            reasoning_mode="none",
            cache_mode="none",
            cache_metrics_mode="openai",
        )

    raise ValueError(f"Unknown model prefix: {prefix}")
