"""Single owner of provider runtime objects: clients, backends, history adapters.

Consolidates wiring that previously lived in both `src/llm/__init__.py` and
`src/utils/clients.py`. Everything that touches provider SDKs at runtime
(default client construction, override client caching, backend selection,
history adapter selection) lives here now.
"""

from __future__ import annotations

from functools import cache, lru_cache
from typing import assert_never

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from src.config import ModelConfig, ModelTransport, settings

from .backend import ProviderBackend
from .backends.anthropic import AnthropicBackend
from .backends.gemini import GeminiBackend
from .backends.openai import OpenAIBackend
from .credentials import default_transport_api_key, resolve_credentials
from .history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    HistoryAdapter,
    OpenAIHistoryAdapter,
)
from .types import ProviderClient


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


# Module-level default-client registry, populated at import time. Tests patch
# this dict via `patch.dict(CLIENTS, {...})` to inject mock provider clients.
CLIENTS: dict[ModelTransport, ProviderClient] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    CLIENTS["anthropic"] = AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )

if settings.LLM.OPENAI_API_KEY:
    CLIENTS["openai"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )

if settings.LLM.GEMINI_API_KEY:
    CLIENTS["gemini"] = genai.client.Client(
        api_key=settings.LLM.GEMINI_API_KEY,
    )


def client_for_model_config(
    provider: ModelTransport,
    model_config: ModelConfig,
) -> ProviderClient:
    """Resolve the provider client for a ModelConfig.

    Fast path: no overrides → reuse the module-level default client from
    CLIENTS (the test-mockable seam). Otherwise route through the cached
    override factories.
    """
    if model_config.api_key is None and model_config.base_url is None:
        existing_client = CLIENTS.get(provider)
        if existing_client is not None:
            return existing_client

    api_key = model_config.api_key or default_transport_api_key(provider)
    base_url = model_config.base_url
    if not api_key:
        raise ValueError(f"Missing API key for {provider} model config")

    if provider == "anthropic":
        return get_anthropic_override_client(base_url, api_key)
    if provider == "openai":
        return get_openai_override_client(base_url, api_key)
    if provider == "gemini":
        return get_gemini_override_client(base_url, api_key)
    assert_never(provider)


def backend_for_provider(
    provider: ModelTransport,
    client: ProviderClient,
) -> ProviderBackend:
    """Wrap a raw provider SDK client in the matching ProviderBackend adapter."""
    if provider == "anthropic":
        return AnthropicBackend(client)
    if provider == "openai":
        return OpenAIBackend(client)
    if provider == "gemini":
        return GeminiBackend(client)
    assert_never(provider)


def history_adapter_for_provider(provider: ModelTransport) -> HistoryAdapter:
    """Provider-appropriate HistoryAdapter for assistant/tool message formatting."""
    if provider == "anthropic":
        return AnthropicHistoryAdapter()
    if provider == "gemini":
        return GeminiHistoryAdapter()
    return OpenAIHistoryAdapter()


def get_backend(config: ModelConfig) -> ProviderBackend:
    """High-level one-shot backend factory: ModelConfig → ProviderBackend."""
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
    assert_never(config.transport)


__all__ = [
    "CLIENTS",
    "backend_for_provider",
    "client_for_model_config",
    "get_anthropic_client",
    "get_anthropic_override_client",
    "get_backend",
    "get_gemini_client",
    "get_gemini_override_client",
    "get_openai_client",
    "get_openai_override_client",
    "history_adapter_for_provider",
]
