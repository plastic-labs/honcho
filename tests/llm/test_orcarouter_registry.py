"""Tests for the named OrcaRouter transport in the provider registry.

OrcaRouter (https://www.orcarouter.ai) is an OpenAI-compatible gateway, so the
orcarouter transport reuses the AsyncOpenAI SDK pointed at the OrcaRouter base
URL and the OpenAI backend. These tests pin that wiring: the default client,
the override-client default base URL, and the backend selection.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from src import config as app_config
from src.config import ModelConfig
from src.llm import registry as registry_module


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Default the LLM settings so the registry reads valid values."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr(
        app_config.settings.LLM, "ORCAROUTER_API_KEY", "test-orcarouter-key"
    )
    yield


@pytest.fixture
def fresh_lru_caches() -> Iterator[None]:
    """Drop lru_cache state so each test exercises a fresh client build."""
    registry_module.get_orcarouter_client.cache_clear()
    registry_module.get_openai_override_client.cache_clear()
    yield
    registry_module.get_orcarouter_client.cache_clear()
    registry_module.get_openai_override_client.cache_clear()


@pytest.mark.usefixtures("fresh_lru_caches")
def test_get_orcarouter_client_uses_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default OrcaRouter client points at api.orcarouter.ai/v1."""
    monkeypatch.setattr(app_config.settings.LLM, "ORCAROUTER_BASE_URL", None)

    with patch("openai.AsyncOpenAI") as mock_openai:
        registry_module.get_orcarouter_client()

    assert mock_openai.call_count == 1
    assert mock_openai.call_args.kwargs["base_url"] == ("https://api.orcarouter.ai/v1")
    assert mock_openai.call_args.kwargs["api_key"] == "test-orcarouter-key"


@pytest.mark.usefixtures("fresh_lru_caches")
def test_get_orcarouter_client_respects_custom_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator-set ORCAROUTER_BASE_URL wins over the default."""
    monkeypatch.setattr(
        app_config.settings.LLM,
        "ORCAROUTER_BASE_URL",
        "https://orcarouter.example.com/v1",
    )

    with patch("openai.AsyncOpenAI") as mock_openai:
        registry_module.get_orcarouter_client()

    assert mock_openai.call_args.kwargs["base_url"] == (
        "https://orcarouter.example.com/v1"
    )


@pytest.mark.usefixtures("fresh_lru_caches")
def test_orcarouter_backend_routes_through_openai() -> None:
    """backend_for_provider(orcarouter) returns the OpenAI backend."""
    from unittest.mock import MagicMock

    from src.llm.backends.openai import OpenAIBackend

    backend = registry_module.backend_for_provider("orcarouter", MagicMock())
    assert isinstance(backend, OpenAIBackend)


def test_client_for_model_config_defaults_base_url_for_orcarouter() -> None:
    """Per-model overrides default to the OrcaRouter endpoint when unset."""
    config = ModelConfig(
        model="auto",
        transport="orcarouter",
        api_key="per-model-key",
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        registry_module.client_for_model_config("orcarouter", config)

    assert mock_openai.call_args.kwargs["base_url"] == ("https://api.orcarouter.ai/v1")
    assert mock_openai.call_args.kwargs["api_key"] == "per-model-key"


def test_client_for_model_config_respects_orcarouter_base_url() -> None:
    """An explicit per-model base_url is forwarded untouched."""
    config = ModelConfig(
        model="auto",
        transport="orcarouter",
        api_key="per-model-key",
        base_url="https://orcarouter.example.com/v1",
    )
    with patch("openai.AsyncOpenAI") as mock_openai:
        registry_module.client_for_model_config("orcarouter", config)

    assert mock_openai.call_args.kwargs["base_url"] == (
        "https://orcarouter.example.com/v1"
    )
