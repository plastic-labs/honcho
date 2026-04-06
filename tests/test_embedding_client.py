from __future__ import annotations

import pytest

from src.config import settings
from src.embedding_client import _EmbeddingClient
import src.embedding_client as embedding_client_module


class _DummyAsyncOpenAI:
    def __init__(self, api_key: str, base_url: str | None = None):
        self.api_key = api_key
        self.base_url = base_url


@pytest.fixture(autouse=True)
def _reset_embedding_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.LLM, "EMBEDDING_MODEL", None)
    monkeypatch.setattr(settings.LLM, "OPENAI_COMPATIBLE_API_KEY", None)
    monkeypatch.setattr(settings.LLM, "OPENAI_COMPATIBLE_BASE_URL", None)
    monkeypatch.setattr(settings.LLM, "VLLM_API_KEY", None)
    monkeypatch.setattr(settings.LLM, "VLLM_BASE_URL", None)


def test_openai_compatible_embedding_provider_uses_compatible_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_client_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(settings.LLM, "OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setattr(
        settings.LLM,
        "OPENAI_COMPATIBLE_BASE_URL",
        "http://localhost:8080/v1",
    )
    monkeypatch.setattr(settings.LLM, "EMBEDDING_MODEL", "custom-embedding-model")

    client = _EmbeddingClient(provider="openai-compatible")

    assert isinstance(client.client, _DummyAsyncOpenAI)
    assert client.client.api_key == "test-key"
    assert client.client.base_url == "http://localhost:8080/v1"
    assert client.model == "custom-embedding-model"


def test_vllm_embedding_provider_uses_vllm_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_client_module, "AsyncOpenAI", _DummyAsyncOpenAI)
    monkeypatch.setattr(settings.LLM, "VLLM_BASE_URL", "http://vllm.local/v1")

    client = _EmbeddingClient(provider="vllm")

    assert isinstance(client.client, _DummyAsyncOpenAI)
    assert client.client.base_url == "http://vllm.local/v1"
    assert client.client.api_key == "sk-no-key-required"
    assert client.model == "openai/text-embedding-3-small"


def test_vllm_embedding_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="LLM_VLLM_BASE_URL"):
        _EmbeddingClient(provider="vllm")


def test_openai_compatible_provider_requires_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.LLM, "OPENAI_COMPATIBLE_API_KEY", "test-key")

    with pytest.raises(ValueError, match="LLM_OPENAI_COMPATIBLE_BASE_URL"):
        _EmbeddingClient(provider="openai-compatible")
