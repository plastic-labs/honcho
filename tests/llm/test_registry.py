"""Tests for the provider-client registry in src/llm/registry.py.

Locks the HTTP-timeout behavior added for the Gemini transport (#785) and
pins the existing 600s timeout on the Anthropic clients so regressions on
either side are caught at unit-test time.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from google.genai import types as genai_types

from src import config as app_config
from src.llm import registry as registry_module

# Gemini's HttpOptions.timeout is an int in milliseconds; keep it in lockstep
# with the Anthropic client's 600s timeout to match the rest of the registry.
_GEMINI_TIMEOUT_MS = 600_000
_ANTHROPIC_TIMEOUT_S = 600.0


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default the LLM settings so the registry reads valid values."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    yield


@pytest.fixture
def _fresh_lru_caches() -> Iterator[None]:
    """Drop lru_cache state so each test exercises a fresh client build."""
    registry_module.get_anthropic_client.cache_clear()
    registry_module.get_gemini_client.cache_clear()
    registry_module.get_anthropic_override_client.cache_clear()
    registry_module.get_gemini_override_client.cache_clear()
    yield
    registry_module.get_anthropic_client.cache_clear()
    registry_module.get_gemini_client.cache_clear()
    registry_module.get_anthropic_override_client.cache_clear()
    registry_module.get_gemini_override_client.cache_clear()


def test_get_gemini_client_sets_http_timeout(
    _fresh_lru_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default Gemini client must carry an HttpOptions timeout, not None."""
    monkeypatch.setattr(app_config.settings.LLM, "GEMINI_BASE_URL", None)

    with patch("src.llm.registry.genai.Client") as mock_client:
        registry_module.get_gemini_client()

    assert mock_client.call_count == 1
    http_options = mock_client.call_args.kwargs["http_options"]
    assert isinstance(http_options, genai_types.HttpOptions)
    assert http_options.timeout == _GEMINI_TIMEOUT_MS


def test_get_gemini_client_preserves_custom_base_url(
    _fresh_lru_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base URL and timeout must coexist on the default Gemini client."""
    monkeypatch.setattr(
        app_config.settings.LLM, "GEMINI_BASE_URL", "https://gemini-proxy.example.com"
    )

    with patch("src.llm.registry.genai.Client") as mock_client:
        registry_module.get_gemini_client()

    http_options = mock_client.call_args.kwargs["http_options"]
    assert isinstance(http_options, genai_types.HttpOptions)
    assert http_options.base_url == "https://gemini-proxy.example.com"
    assert http_options.timeout == _GEMINI_TIMEOUT_MS


def test_get_gemini_override_client_sets_http_timeout(
    _fresh_lru_caches: None,
) -> None:
    """Override Gemini client must also carry a timeout."""
    with patch("src.llm.registry.genai.Client") as mock_client:
        registry_module.get_gemini_override_client(
            "https://gemini-proxy.example.com", "sk-override"
        )

    http_options = mock_client.call_args.kwargs["http_options"]
    assert isinstance(http_options, genai_types.HttpOptions)
    assert http_options.base_url == "https://gemini-proxy.example.com"
    assert http_options.timeout == _GEMINI_TIMEOUT_MS


def test_get_gemini_override_client_handles_missing_base_url(
    _fresh_lru_caches: None,
) -> None:
    """Override Gemini client with no base URL still carries a timeout."""
    with patch("src.llm.registry.genai.Client") as mock_client:
        registry_module.get_gemini_override_client(None, "sk-override")

    http_options = mock_client.call_args.kwargs["http_options"]
    assert isinstance(http_options, genai_types.HttpOptions)
    assert http_options.timeout == _GEMINI_TIMEOUT_MS


def test_get_anthropic_client_keeps_600s_timeout(
    _fresh_lru_caches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anthropic timeout is the established behavior — lock it."""
    monkeypatch.setattr(app_config.settings.LLM, "ANTHROPIC_BASE_URL", None)

    with patch("src.llm.registry.AsyncAnthropic") as mock_anthropic:
        registry_module.get_anthropic_client()

    assert mock_anthropic.call_args.kwargs["timeout"] == _ANTHROPIC_TIMEOUT_S


def test_get_anthropic_override_client_keeps_600s_timeout(
    _fresh_lru_caches: None,
) -> None:
    """Override Anthropic client also keeps the 600s timeout."""
    with patch("src.llm.registry.AsyncAnthropic") as mock_anthropic:
        registry_module.get_anthropic_override_client(None, "sk-override")

    assert mock_anthropic.call_args.kwargs["timeout"] == _ANTHROPIC_TIMEOUT_S


def test_gemini_http_options_builder_applies_timeout() -> None:
    """The shared helper must always set a timeout, even with no base_url."""
    options = registry_module._build_gemini_http_options(None)
    assert isinstance(options, genai_types.HttpOptions)
    assert options.timeout == _GEMINI_TIMEOUT_MS
    assert options.base_url is None

    options = registry_module._build_gemini_http_options("https://example.com")
    assert isinstance(options, genai_types.HttpOptions)
    assert options.timeout == _GEMINI_TIMEOUT_MS
    assert options.base_url == "https://example.com"
