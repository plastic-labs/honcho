from typing import Any

import pytest

from src.config import ModelConfig
from src.llm import registry
from src.llm.backends.openai import OpenAIBackend


def test_client_for_model_config_builds_azure_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``azure_openai`` transport must instantiate AsyncAzureOpenAI with
    azure_endpoint + api_version from the ModelConfig overrides."""
    registry.get_azure_openai_override_client.cache_clear()
    captured: dict[str, Any] = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        api_key="az-test-key",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
    )

    try:
        client = registry.client_for_model_config("azure_openai", config)

        assert isinstance(client, FakeAsyncAzureOpenAI)
        assert captured == {
            "api_key": "az-test-key",
            "azure_endpoint": "https://gateway.example/azure-openai",
            "api_version": "2024-10-21",
        }
    finally:
        # Evict the FakeAsyncAzureOpenAI instance so later tests that hit
        # the same (endpoint, key, version) cache key get the real class
        # back (restored by monkeypatch teardown).
        registry.get_azure_openai_override_client.cache_clear()


def test_backend_for_provider_wraps_azure_client_in_openai_backend() -> None:
    """Same backend serves both ``openai`` and ``azure_openai``."""
    sentinel_client = object()

    backend = registry.backend_for_provider("azure_openai", sentinel_client)

    assert isinstance(backend, OpenAIBackend)
    assert backend._client is sentinel_client  # pyright: ignore[reportPrivateUsage]
