"""
Provider adapter registry.

This module exposes a single `get_adapter()` function that returns the adapter
implementation for a given provider identifier. Adapters are cached since they
are stateless.
"""

from __future__ import annotations

from src.utils.types import SupportedProviders

from .base import ProviderAdapter

_ADAPTERS: dict[str, ProviderAdapter] = {}


def _create_adapter(provider: str) -> ProviderAdapter:
    """Create a new adapter instance for the given provider."""
    if provider == "anthropic":
        from src.utils.llm.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter()
    if provider == "openai":
        from src.utils.llm.adapters.openai import OpenAIAdapter

        return OpenAIAdapter()
    if provider == "openrouter":
        from src.utils.llm.adapters.openrouter import OpenRouterAdapter

        return OpenRouterAdapter()
    if provider == "vllm":
        from src.utils.llm.adapters.vllm import VLLMAdapter

        return VLLMAdapter()
    if provider == "google":
        from src.utils.llm.adapters.google import GoogleAdapter

        return GoogleAdapter()
    if provider == "groq":
        from src.utils.llm.adapters.groq import GroqAdapter

        return GroqAdapter()
    raise ValueError(f"Unsupported provider: {provider}")


def get_adapter(provider: SupportedProviders | str) -> ProviderAdapter:
    """Return a cached ProviderAdapter implementation for `provider`."""
    if provider not in _ADAPTERS:
        _ADAPTERS[provider] = _create_adapter(provider)
    return _ADAPTERS[provider]


__all__ = ["ProviderAdapter", "get_adapter"]
