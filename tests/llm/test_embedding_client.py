from types import SimpleNamespace
from typing import Any

import pytest
from src.config import EmbeddingModelConfig
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding: list[float] = embedding
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> SimpleNamespace:
        call: dict[str, Any] = {"model": model, "input": input}
        call.update(kwargs)
        self.calls.append(call)
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
        send_dimensions=False,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.1] * 8
    assert fake_embeddings.calls == [{"model": "text-embedding-3-small", "input": ["hello world"]}]


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
        send_dimensions=False,
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
            calls.append({
                "model": model,
                "contents": contents,
                "config": config,
            })
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
        send_dimensions=False,
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


def _build_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding: list[float],
    model: str,
    send_dimensions: bool,
    vector_dimensions: int,
) -> tuple[_EmbeddingClient, FakeOpenAIEmbeddingsAPI]:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model=model,
            api_key="test-key",
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=send_dimensions,
    )
    return client, fake_embeddings


@pytest.mark.asyncio
async def test_openai_embed_forwards_dimensions_when_send_dimensions_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.embed("hello")

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "dimensions": 768,
        }
    ]


@pytest.mark.asyncio
async def test_openai_embed_omits_dimensions_when_send_dimensions_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
    )

    await client.embed("hello")

    assert fake.calls == [{"model": "text-embedding-3-small", "input": ["hello"]}]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.simple_batch_embed(["a", "b"])

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768
    assert fake.calls[0]["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768


def _build_embedding_settings(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Construct a fresh EmbeddingSettings from the given env, isolated from os.environ."""
    from src.config import EmbeddingSettings

    for key in (
        "EMBEDDING_VECTOR_DIMENSIONS",
        "EMBEDDING_MODEL_CONFIG__MODEL",
        "EMBEDDING_MODEL_CONFIG__TRANSPORT",
        "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return EmbeddingSettings()


def test_resolve_send_dimensions_auto_default_dim_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({}, monkeypatch)
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_auto_explicit_dim_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({"EMBEDDING_VECTOR_DIMENSIONS": "768"}, monkeypatch)
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_auto_ada_002_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_VECTOR_DIMENSIONS": "1536",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_always_returns_true_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always"},
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_always_overrides_ada_rejecting_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_never_returns_false_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "never",
            "EMBEDDING_VECTOR_DIMENSIONS": "768",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


@pytest.mark.asyncio
async def test_simple_batch_embed_respects_token_budget_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """simple_batch_embed must split inputs across requests so per-request token cap holds."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.5] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    # max_input_tokens=100 per single input; max_tokens_per_request=120 total,
    # so two ~80-token inputs must end up in *separate* requests.
    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=100,
        max_tokens_per_request=120,
        send_dimensions=False,
    )

    # "word " * 80 produces ~80 tokens with cl100k_base/the model encoding.
    long_a = ("alpha " * 80).strip()
    long_b = ("beta " * 80).strip()

    out = await client.simple_batch_embed([long_a, long_b])
    assert len(out) == 2
    # Per-request token cap forces two separate requests.
    assert len(fake_embeddings.calls) == 2


@pytest.mark.asyncio
async def test_simple_batch_embed_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inputs that exceed max_embedding_tokens must raise ValueError immediately."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    too_long = ("word " * 50).strip()
    with pytest.raises(ValueError, match="maximum token limit"):
        await client.simple_batch_embed([too_long])


def test_prepare_chunks_returns_ordered_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_chunks must split oversized inputs using the same rules as batch_embed."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url=None,
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    short_text = "hello"
    long_text = ("word " * 50).strip()

    out = client.prepare_chunks({"short": short_text, "long": long_text})

    assert out["short"] == [short_text]
    assert len(out["long"]) > 1
    # Order preserved
    assert isinstance(out["long"][0], str)
