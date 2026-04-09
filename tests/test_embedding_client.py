from types import SimpleNamespace

import pytest

from src import embedding_client as embedding_module


def test_openrouter_embedding_client_prefers_dedicated_embedding_endpoint(monkeypatch):
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_API_KEY",
        "chat-key",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_BASE_URL",
        "https://chat.example/v1",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "EMBEDDING_API_KEY",
        "embed-key",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "EMBEDDING_BASE_URL",
        "https://embed.example/v1",
    )

    captured: dict[str, str] = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(embedding_module, "AsyncOpenAI", FakeAsyncOpenAI)

    client = embedding_module._EmbeddingClient(provider="openrouter")

    assert captured == {
        "api_key": "embed-key",
        "base_url": "https://embed.example/v1",
    }
    assert client.provider == "openrouter"
    assert client.model == "openai/text-embedding-3-small"


def test_openrouter_embedding_client_falls_back_to_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_API_KEY",
        "chat-key",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_BASE_URL",
        "https://chat.example/v1",
    )
    monkeypatch.setattr(embedding_module.settings.LLM, "EMBEDDING_API_KEY", None)
    monkeypatch.setattr(embedding_module.settings.LLM, "EMBEDDING_BASE_URL", None)

    captured: dict[str, str] = {}

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(embedding_module, "AsyncOpenAI", FakeAsyncOpenAI)

    embedding_module._EmbeddingClient(provider="openrouter")

    assert captured == {
        "api_key": "chat-key",
        "base_url": "https://chat.example/v1",
    }


def test_openrouter_embedding_client_error_mentions_both_supported_keys(monkeypatch):
    monkeypatch.setattr(embedding_module.settings.LLM, "EMBEDDING_API_KEY", None)
    monkeypatch.setattr(embedding_module.settings.LLM, "OPENAI_COMPATIBLE_API_KEY", None)

    with pytest.raises(ValueError) as excinfo:
        embedding_module._EmbeddingClient(provider="openrouter")

    assert "LLM_EMBEDDING_API_KEY" in str(excinfo.value)
    assert "LLM_OPENAI_COMPATIBLE_API_KEY" in str(excinfo.value)


def test_embedding_client_wrapper_prefers_dedicated_embedding_endpoint(monkeypatch):
    monkeypatch.setattr(embedding_module.settings.LLM, "EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_API_KEY",
        "chat-key",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "OPENAI_COMPATIBLE_BASE_URL",
        "https://chat.example/v1",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "EMBEDDING_API_KEY",
        "embed-key",
    )
    monkeypatch.setattr(
        embedding_module.settings.LLM,
        "EMBEDDING_BASE_URL",
        "https://embed.example/v1",
    )

    captured: dict[str, str] = {}
    original_instance = embedding_module.EmbeddingClient._instance
    original_wrapper_instance = embedding_module.EmbeddingClient._wrapper_instance
    embedding_module.EmbeddingClient._instance = None
    embedding_module.EmbeddingClient._wrapper_instance = None

    class FakeEmbeddingClient:
        def __init__(
            self,
            *,
            api_key: str | None = None,
            provider: str | None = None,
            base_url: str | None = None,
        ):
            captured["api_key"] = api_key or ""
            captured["provider"] = provider or ""
            captured["base_url"] = base_url or ""
            self.provider = provider or ""
            self.model = "fake-model"
            self.max_embedding_tokens = 1
            self.encoding = SimpleNamespace()

    monkeypatch.setattr(embedding_module, "_EmbeddingClient", FakeEmbeddingClient)

    try:
        wrapper = embedding_module.EmbeddingClient()
        wrapper._get_client()
    finally:
        embedding_module.EmbeddingClient._instance = original_instance
        embedding_module.EmbeddingClient._wrapper_instance = original_wrapper_instance

    assert captured == {
        "api_key": "embed-key",
        "provider": "openrouter",
        "base_url": "https://embed.example/v1",
    }
