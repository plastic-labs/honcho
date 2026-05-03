from types import SimpleNamespace
from typing import Any

import pytest

from src.config import (
    ConfiguredEmbeddingModelSettings,
    EmbeddingModelConfig,
    resolve_embedding_model_config,
)
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding: list[float] = embedding
        self.calls: list[dict[str, Any]] = []

    async def create(
        self, *, model: str, input: str | list[str], **kwargs: Any
    ) -> SimpleNamespace:
        self.calls.append({"model": model, "input": input, **kwargs})
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


def test_resolve_embedding_model_config_preserves_provider_params() -> None:
    configured = ConfiguredEmbeddingModelSettings(
        model="text-embedding-v4",
        transport="openai",
        overrides={
            "base_url": "https://example.test/v1",
            "api_key": "test-key",
            "provider_params": {"dimensions": 1536},
        },
    )

    resolved = resolve_embedding_model_config(configured)

    assert resolved.model == "text-embedding-v4"
    assert resolved.transport == "openai"
    assert resolved.base_url == "https://example.test/v1"
    assert resolved.api_key == "test-key"
    assert resolved.provider_params == {"dimensions": 1536}


@pytest.mark.asyncio
async def test_openai_embedding_client_passes_provider_params_to_single_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding=[0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-v4",
            api_key="test-key",
            base_url="https://example.test/v1",
            provider_params={"dimensions": 8},
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    result = await client.embed("hello world")

    assert result == [0.1] * 8
    assert fake_embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["hello world"],
            "dimensions": 8,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_passes_provider_params_to_simple_batch_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding=[0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-v4",
            api_key="test-key",
            base_url="https://example.test/v1",
            provider_params={"dimensions": 8},
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    result = await client.simple_batch_embed(["hello", "world"])

    assert result == [[0.1] * 8, [0.1] * 8]
    assert fake_embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["hello", "world"],
            "dimensions": 8,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_passes_provider_params_to_chunked_batch_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding=[0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-v4",
            api_key="test-key",
            base_url="https://example.test/v1",
            provider_params={"dimensions": 8},
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    text = "hello world"
    result = await client.batch_embed({"doc-1": (text, client.encoding.encode(text))})

    assert result == {"doc-1": [[0.1] * 8]}
    assert fake_embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["hello world"],
            "dimensions": 8,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_omits_provider_params_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding=[0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="https://example.test/v1",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
    )

    result = await client.embed("hello world")

    assert result == [0.1] * 8
    assert fake_embeddings.calls == [
        {"model": "text-embedding-3-small", "input": ["hello world"]}
    ]
