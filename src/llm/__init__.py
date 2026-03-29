from __future__ import annotations

from functools import cache, lru_cache

from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI

from src.config import ModelConfig, settings

from .backend import CompletionResult, ProviderBackend, StreamChunk, ToolCallResult
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.groq import GroqBackend
from .backends.openai import OpenAIBackend
from .backends.openai_compat import OpenAICompatibleBackend
from .capabilities import ModelCapabilities, get_model_capabilities
from .credentials import resolve_credentials


@lru_cache(maxsize=1)
def _get_anthropic_client() -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )


@lru_cache(maxsize=1)
def _get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.LLM.OPENAI_API_KEY)


@lru_cache(maxsize=1)
def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=settings.LLM.GEMINI_API_KEY)


@lru_cache(maxsize=1)
def _get_groq_client() -> AsyncGroq:
    return AsyncGroq(api_key=settings.LLM.GROQ_API_KEY)


@cache
def _get_compatible_client(base_url: str | None, api_key: str | None) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def get_backend(config: ModelConfig) -> ProviderBackend:
    """Resolve a backend implementation for the effective model config."""

    if config.transport == "openai_compatible":
        credentials = resolve_credentials(config)
        return OpenAICompatibleBackend(
            _get_compatible_client(
                credentials.get("api_base"),
                credentials.get("api_key"),
            ),
            provider_name=config.compat_provider or "generic",
        )

    prefix = config.model.split("/", 1)[0]
    if prefix == "anthropic":
        return AnthropicBackend(_get_anthropic_client())
    if prefix == "gemini":
        return GeminiBackend(_get_gemini_client())
    if prefix == "openai":
        return OpenAIBackend(_get_openai_client())
    if prefix == "groq":
        return GroqBackend(_get_groq_client())
    if prefix == "openrouter":
        return OpenAICompatibleBackend(
            _get_compatible_client(
                "https://openrouter.ai/api/v1",
                settings.LLM.OPENAI_COMPATIBLE_API_KEY,
            ),
            provider_name="openrouter",
        )
    if prefix == "hosted_vllm":
        return OpenAICompatibleBackend(
            _get_compatible_client(
                settings.LLM.VLLM_BASE_URL,
                settings.LLM.VLLM_API_KEY,
            ),
            provider_name="vllm",
        )
    raise ValueError(f"Unknown model prefix: {prefix}")


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
