from __future__ import annotations

from functools import cache, lru_cache

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from src.config import ModelConfig, settings

from .backend import CompletionResult, ProviderBackend, StreamChunk, ToolCallResult
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.openai import OpenAIBackend
from .credentials import resolve_credentials


@lru_cache(maxsize=1)
def get_anthropic_client() -> AsyncAnthropic:
    """Default Anthropic client built from settings.LLM.ANTHROPIC_API_KEY."""
    return AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """Default OpenAI client built from settings.LLM.OPENAI_API_KEY."""
    return AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Default Gemini client built from settings.LLM.GEMINI_API_KEY."""
    return genai.Client(api_key=settings.LLM.GEMINI_API_KEY)


@cache
def get_openai_override_client(
    base_url: str | None, api_key: str | None
) -> AsyncOpenAI:
    """OpenAI client for a specific (base_url, api_key) pair. Cached by key."""
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


@cache
def get_anthropic_override_client(
    base_url: str | None,
    api_key: str | None,
) -> AsyncAnthropic:
    """Anthropic client for a specific (base_url, api_key) pair. Cached by key."""
    return AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=600.0)


@cache
def get_gemini_override_client(
    base_url: str | None, api_key: str | None
) -> genai.Client:
    """Gemini client for a specific (base_url, api_key) pair. Cached by key."""
    http_options = genai_types.HttpOptions(base_url=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_options)


def get_backend(config: ModelConfig) -> ProviderBackend:
    """Resolve a backend implementation for the effective model config."""

    credentials = resolve_credentials(config)

    if config.transport == "anthropic":
        if config.api_key is not None or config.base_url is not None:
            return AnthropicBackend(
                get_anthropic_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return AnthropicBackend(get_anthropic_client())
    if config.transport == "gemini":
        if config.api_key is not None or config.base_url is not None:
            return GeminiBackend(
                get_gemini_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return GeminiBackend(get_gemini_client())
    if config.transport == "openai":
        if config.api_key is not None or config.base_url is not None:
            return OpenAIBackend(
                get_openai_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return OpenAIBackend(get_openai_client())
    raise ValueError(f"Unknown transport: {config.transport}")


__all__ = [
    "CompletionResult",
    "ProviderBackend",
    "StreamChunk",
    "ToolCallResult",
    "get_anthropic_client",
    "get_anthropic_override_client",
    "get_backend",
    "get_gemini_client",
    "get_gemini_override_client",
    "get_openai_client",
    "get_openai_override_client",
    "resolve_credentials",
]
