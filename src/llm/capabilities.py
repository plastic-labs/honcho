from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import ModelConfig


@dataclass(frozen=True)
class ModelCapabilities:
    transport: Literal["anthropic", "openai", "gemini"]
    history_format: Literal["anthropic", "gemini", "openai"]
    structured_output_mode: Literal["native", "repair_wrapper"]
    reasoning_mode: Literal["none", "effort", "budget", "adaptive", "always_on"]
    cache_mode: Literal["none", "prefix", "gemini_cached_content"]
    cache_metrics_mode: Literal["none", "anthropic", "openai", "gemini"]
    supports_dimensions: bool = False
    shared_reasoning_budget: bool = False


def get_model_capabilities(config: ModelConfig) -> ModelCapabilities:
    """Return runtime capability metadata for the effective model config."""

    if config.transport == "anthropic":
        return ModelCapabilities(
            transport="anthropic",
            history_format="anthropic",
            structured_output_mode="repair_wrapper",
            reasoning_mode="budget",
            cache_mode="prefix",
            cache_metrics_mode="anthropic",
        )

    if config.transport == "gemini":
        return ModelCapabilities(
            transport="gemini",
            history_format="gemini",
            structured_output_mode="native",
            reasoning_mode="budget",
            cache_mode="gemini_cached_content",
            cache_metrics_mode="gemini",
            shared_reasoning_budget=True,
        )

    if config.transport == "openai":
        return ModelCapabilities(
            transport="openai",
            history_format="openai",
            structured_output_mode="native",
            reasoning_mode="effort",
            cache_mode="prefix",
            cache_metrics_mode="openai",
        )

    raise ValueError(f"Unknown transport: {config.transport}")
