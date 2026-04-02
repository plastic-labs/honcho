from __future__ import annotations

from functools import cache, lru_cache

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from groq import AsyncGroq
from openai import AsyncOpenAI

from src.config import ModelConfig, settings

from .backend import CompletionResult, ProviderBackend, StreamChunk, ToolCallResult
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.groq import GroqBackend
from .backends.openai import OpenAIBackend
from .capabilities import ModelCapabilities, get_model_capabilities
from .credentials import resolve_credentials


@lru_cache(maxsize=1)
def _get_anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        base_url=settings.LLM.ANTHROPIC_BASE_URL,
        timeout=600.0,
    )


@lru_cache(maxsize=1)
def _get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
        base_url=settings.LLM.OPENAI_BASE_URL,
    )


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    http_options: genai_types.HttpOptions | None = None
    if settings.LLM.GEMINI_BASE_URL:
        http_options = genai_types.HttpOptions(base_url=settings.LLM.GEMINI_BASE_URL)
    return genai.Client(api_key=settings.LLM.GEMINI_API_KEY, http_options=http_options)


@lru_cache(maxsize=1)
def _get_groq_client() -> AsyncGroq:
    return AsyncGroq(
        api_key=settings.LLM.GROQ_API_KEY,
        base_url=settings.LLM.GROQ_BASE_URL,
    )


@cache
def _get_openai_override_client(
    base_url: str | None, api_key: str | None
) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


@cache
def _get_anthropic_override_client(
    base_url: str | None,
    api_key: str | None,
) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=600.0)


@cache
def _get_groq_override_client(base_url: str | None, api_key: str | None) -> AsyncGroq:
    return AsyncGroq(api_key=api_key, base_url=base_url)


@cache
def _get_gemini_override_client(
    base_url: str | None, api_key: str | None
) -> genai.Client:
    http_options = genai_types.HttpOptions(base_url=base_url) if base_url else None
    return genai.Client(api_key=api_key, http_options=http_options)


def get_backend(config: ModelConfig) -> ProviderBackend:
    """Resolve a backend implementation for the effective model config."""

    credentials = resolve_credentials(config)

    if config.transport == "anthropic":
        if config.api_key is not None or config.base_url is not None:
            return AnthropicBackend(
                _get_anthropic_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return AnthropicBackend(_get_anthropic_client())
    if config.transport == "gemini":
        if config.api_key is not None or config.base_url is not None:
            return GeminiBackend(
                _get_gemini_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return GeminiBackend(_get_gemini_client())
    if config.transport == "openai":
        if config.api_key is not None or config.base_url is not None:
            return OpenAIBackend(
                _get_openai_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return OpenAIBackend(_get_openai_client())
    if config.transport == "groq":
        if config.api_key is not None or config.base_url is not None:
            return GroqBackend(
                _get_groq_override_client(
                    credentials.get("api_base"),
                    credentials.get("api_key"),
                )
            )
        return GroqBackend(_get_groq_client())
    raise ValueError(f"Unknown transport: {config.transport}")


__all__ = [
    "CompletionResult",
    "ModelCapabilities",
    "ProviderBackend",
    "StreamChunk",
    "ToolCallResult",
    "get_backend",
    "get_model_capabilities",
    "resolve_credentials",
]
