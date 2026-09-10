"""Single owner of provider runtime objects: clients, backends, history adapters.

Consolidates wiring that previously lived in both `src/llm/__init__.py` and
`src/utils/clients.py`. Everything that touches provider SDKs at runtime
(default client construction, override client caching, backend selection,
history adapter selection) lives here now.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, assert_never

from src.config import ModelConfig, ModelTransport, settings
from src.exceptions import ValidationException

from .backend import ProviderBackend
from .credentials import default_transport_api_key
from .history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    HistoryAdapter,
    OpenAIHistoryAdapter,
)
from .types import ProviderClient

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from google import genai
    from google.genai import types as genai_types
    from openai import AsyncOpenAI

# Provider SDKs are imported lazily inside the client factories below so a
# process only pays the import-time memory cost of the providers it uses.

# Default client-level HTTP timeouts. Anthropic accepts seconds (float);
# google-genai's HttpOptions.timeout is an int in milliseconds, so the Gemini
# value is kept separately. Both default to 10 minutes to match the existing
# Anthropic behavior — long enough for slow streamed responses, short enough
# that a stalled socket can no longer wedge the deriver worker (see #785).
_ANTHROPIC_TIMEOUT_S = 600.0
_GEMINI_TIMEOUT_MS = 600_000

# Client-level ``default_headers`` applied to OpenAI-compatible clients, keyed by
# base-URL prefix. Currently only OpenRouter, which uses them for app attribution
# (https://openrouter.ai/docs/app-attribution); add a prefix here to tag another
# provider. Other OpenAI-compatible backends ignore unrecognized headers.
_DEFAULT_HEADERS_BY_BASE_URL: dict[str, dict[str, str]] = {
    "https://openrouter.ai": {
        "HTTP-Referer": "https://honcho.dev",
        "X-Openrouter-Title": "Honcho",
    },
}


def _default_headers_for(base_url: str | None) -> dict[str, str]:
    """Default headers for ``base_url`` (prefix match); these merge under any
    per-request ``extra_headers`` passthrough, which wins on key collision."""
    if base_url:
        for prefix, headers in _DEFAULT_HEADERS_BY_BASE_URL.items():
            if base_url.startswith(prefix):
                return headers
    return {}


def _build_gemini_http_options(base_url: str | None) -> genai_types.HttpOptions:
    """Build Gemini ``HttpOptions`` carrying a default HTTP timeout.

    google-genai's ``HttpOptions.timeout`` is an int in milliseconds. A stalled
    Gemini socket without this value wedges the entire deriver process because
    all deriver workers share one uvloop event loop (see #785). Keep the
    timeout even when no ``base_url`` is configured — that's the path the
    default ``get_gemini_client`` takes and it's the one that was hanging.
    """
    from google.genai import types as genai_types

    return genai_types.HttpOptions(
        base_url=base_url,
        timeout=_GEMINI_TIMEOUT_MS,
    )


@lru_cache(maxsize=1)
def get_anthropic_client() -> AsyncAnthropic:
    """Default Anthropic client built from settings.LLM.ANTHROPIC_API_KEY."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        base_url=settings.LLM.ANTHROPIC_BASE_URL,
        timeout=_ANTHROPIC_TIMEOUT_S,
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """Default OpenAI client built from settings.LLM.OPENAI_API_KEY."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
        base_url=settings.LLM.OPENAI_BASE_URL,
        default_headers=_default_headers_for(settings.LLM.OPENAI_BASE_URL),
    )


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Default Gemini client built from settings.LLM.GEMINI_API_KEY."""
    from google import genai

    return genai.Client(
        api_key=settings.LLM.GEMINI_API_KEY,
        http_options=_build_gemini_http_options(settings.LLM.GEMINI_BASE_URL),
    )


# Bounded cache — in practice the (base_url, api_key) key space is small
# and process-scoped, but maxsize=128 keeps worst-case memory predictable.
@lru_cache(maxsize=128)
def get_openai_override_client(
    base_url: str | None, api_key: str | None
) -> AsyncOpenAI:
    """OpenAI client for a specific (base_url, api_key) pair. Cached by key."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=_default_headers_for(base_url),
    )


@lru_cache(maxsize=128)
def get_anthropic_override_client(
    base_url: str | None,
    api_key: str | None,
) -> AsyncAnthropic:
    """Anthropic client for a specific (base_url, api_key) pair. Cached by key."""
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(
        api_key=api_key, base_url=base_url, timeout=_ANTHROPIC_TIMEOUT_S
    )


@lru_cache(maxsize=128)
def get_gemini_override_client(
    base_url: str | None, api_key: str | None
) -> genai.Client:
    """Gemini client for a specific (base_url, api_key) pair. Cached by key."""
    from google import genai

    return genai.Client(
        api_key=api_key,
        http_options=_build_gemini_http_options(base_url),
    )


# Module-level default-client registry, populated lazily on first use so a
# provider's SDK is only imported when that provider is actually called. Tests
# patch this dict via `patch.dict(CLIENTS, {...})` to inject mock provider
# clients; a patched entry always wins because `default_client` checks the
# dict before constructing anything.
CLIENTS: dict[ModelTransport, ProviderClient] = {}


def default_client(provider: ModelTransport) -> ProviderClient | None:
    """Default client for ``provider``, built on first use.

    Returns None when no API key is configured for the provider.
    """
    existing = CLIENTS.get(provider)
    if existing is not None:
        return existing

    if provider == "anthropic":
        if not settings.LLM.ANTHROPIC_API_KEY:
            return None
        client: ProviderClient = get_anthropic_client()
    elif provider == "openai":
        if not settings.LLM.OPENAI_API_KEY:
            return None
        client = get_openai_client()
    elif provider == "gemini":
        if not settings.LLM.GEMINI_API_KEY:
            return None
        client = get_gemini_client()
    else:
        assert_never(provider)

    CLIENTS[provider] = client
    return client


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
        existing_client = default_client(provider)
        if existing_client is not None:
            return existing_client

    api_key = model_config.api_key or default_transport_api_key(provider)
    base_url = model_config.base_url
    if not api_key:
        raise ValidationException(f"Missing API key for {provider} model config")

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
        from .backends.anthropic import AnthropicBackend

        return AnthropicBackend(client)
    if provider == "openai":
        from .backends.openai import OpenAIBackend

        return OpenAIBackend(client)
    if provider == "gemini":
        from .backends.gemini import GeminiBackend

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
    """High-level one-shot backend factory: ModelConfig → ProviderBackend.

    Delegates client resolution to ``client_for_model_config``, which owns
    the CLIENTS fast-path and the missing-API-key validation. Both the
    production path (via ``honcho_llm_call_inner``) and the live-test path
    (via this function) now construct clients through the same helper, so
    validation behavior stays consistent.
    """
    client = client_for_model_config(config.transport, config)
    return backend_for_provider(config.transport, client)


__all__ = [
    "CLIENTS",
    "backend_for_provider",
    "client_for_model_config",
    "default_client",
    "get_anthropic_client",
    "get_anthropic_override_client",
    "get_backend",
    "get_gemini_client",
    "get_gemini_override_client",
    "get_openai_client",
    "get_openai_override_client",
    "history_adapter_for_provider",
]
