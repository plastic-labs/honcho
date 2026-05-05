from types import SimpleNamespace
from typing import Any

import pytest

from src.config import EmbeddingModelConfig
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding: list[float] = embedding
        self.calls: list[dict[str, Any]] = []

    async def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input})
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=self.embedding) for _ in input]
        else:
            data = [SimpleNamespace(embedding=self.embedding)]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_openai_embedding_client_uses_configured_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.1] * 8
    assert fake_embeddings.calls == [
        {"model": "text-embedding-3-small", "input": ["hello world"]}
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 7)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await client.embed("hello world")


@pytest.mark.asyncio
async def test_gemini_embedding_client_uses_output_dimensionality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: str | list[str],
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12)],
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("src.embedding_client.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            base_url="https://gemini-proxy.example/v1beta",
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.2] * 12
    assert calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "hello world",
            "config": {"output_dimensionality": 12},
        }
    ]


@pytest.mark.asyncio
async def test_azure_openai_embedding_client_uses_azure_endpoint_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.3] * 8)
    captured: dict[str, Any] = {}

    class FakeAzureClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            azure_endpoint: str | None,
            api_version: str | None,
        ) -> None:
            captured["api_key"] = api_key
            captured["azure_endpoint"] = azure_endpoint
            captured["api_version"] = api_version
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncAzureOpenAI", FakeAzureClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="azure_openai",
            model="text-embedding-3-small",
            api_key="az-test-key",
            base_url="https://gateway.example/azure-openai",
            api_version="2024-10-21",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.3] * 8
    assert captured == {
        "api_key": "az-test-key",
        "azure_endpoint": "https://gateway.example/azure-openai",
        "api_version": "2024-10-21",
    }
    assert fake_embeddings.calls == [
        {"model": "text-embedding-3-small", "input": ["hello world"]}
    ]


def test_azure_openai_embedding_client_requires_api_version() -> None:
    with pytest.raises(ValueError, match="api_version is required"):
        _EmbeddingClient(
            EmbeddingModelConfig(
                transport="azure_openai",
                model="text-embedding-3-small",
                api_key="az-test-key",
                base_url="https://gateway.example/azure-openai",
            ),
            vector_dimensions=8,
            max_input_tokens=8192,
            max_tokens_per_request=300_000,
        )


@pytest.mark.asyncio
async def test_azure_openai_embedding_client_with_entra_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_entra_id=True`` must use ``azure_ad_token_provider`` instead
    of ``api_key`` when constructing the Azure embedding client."""
    import sys
    from unittest.mock import MagicMock

    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.4] * 8)
    captured: dict[str, Any] = {}

    class FakeAzureClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncAzureOpenAI", FakeAzureClient)

    mock_credential = MagicMock()
    mock_token_provider = MagicMock(return_value="fake-token")

    mock_azure_identity = MagicMock(
        DefaultAzureCredential=MagicMock(return_value=mock_credential),
        get_bearer_token_provider=MagicMock(return_value=mock_token_provider),
    )
    monkeypatch.setitem(sys.modules, "azure.identity", mock_azure_identity)

    try:
        client = _EmbeddingClient(
            EmbeddingModelConfig(
                transport="azure_openai",
                model="text-embedding-3-small",
                base_url="https://gateway.example/azure-openai",
                api_version="2024-10-21",
                use_entra_id=True,
            ),
            vector_dimensions=8,
            max_input_tokens=8192,
            max_tokens_per_request=300_000,
        )

        embedding = await client.embed("hello world")

        assert embedding == [0.4] * 8
        assert captured["azure_ad_token_provider"] is mock_token_provider
        assert captured["azure_endpoint"] == "https://gateway.example/azure-openai"
        assert captured["api_version"] == "2024-10-21"
        assert "api_key" not in captured
    finally:
        sys.modules.pop("azure.identity", None)


def test_azure_openai_embedding_without_azure_identity_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_entra_id=True without the azure extra installed must raise with
    a friendly ``pip install 'honcho[azure]'`` hint, not a bare ImportError."""
    import builtins

    class FakeAzureClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr("src.embedding_client.AsyncAzureOpenAI", FakeAzureClient)

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "azure.identity" or name.startswith("azure.identity."):
            raise ImportError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match=r"honcho\[azure\]"):
        _EmbeddingClient(
            EmbeddingModelConfig(
                transport="azure_openai",
                model="text-embedding-3-small",
                base_url="https://gateway.example/azure-openai",
                api_version="2024-10-21",
                use_entra_id=True,
            ),
            vector_dimensions=8,
            max_input_tokens=8192,
            max_tokens_per_request=300_000,
        )
