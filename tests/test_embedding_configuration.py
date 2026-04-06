from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _make_settings(
    *,
    provider: str,
    embedding_model: str | None = None,
    openai_api_key: str | None = None,
    openai_compatible_api_key: str | None = None,
    openai_compatible_base_url: str | None = None,
    gemini_api_key: str | None = None,
    max_embedding_tokens: int = 8192,
    max_embedding_tokens_per_request: int = 300000,
) -> SimpleNamespace:
    return SimpleNamespace(
        LLM=SimpleNamespace(
            EMBEDDING_PROVIDER=provider,
            EMBEDDING_MODEL=embedding_model,
            OPENAI_API_KEY=openai_api_key,
            OPENAI_COMPATIBLE_API_KEY=openai_compatible_api_key,
            OPENAI_COMPATIBLE_BASE_URL=openai_compatible_base_url,
            GEMINI_API_KEY=gemini_api_key,
        ),
        MAX_EMBEDDING_TOKENS=max_embedding_tokens,
        MAX_EMBEDDING_TOKENS_PER_REQUEST=max_embedding_tokens_per_request,
    )


@pytest.fixture(autouse=True)
def _reset_embedding_singleton() -> None:
    import src.embedding_client as ec

    ec.EmbeddingClient._instance = None
    ec.EmbeddingClient._wrapper_instance = None
    yield
    ec.EmbeddingClient._instance = None
    ec.EmbeddingClient._wrapper_instance = None


def test_custom_provider_uses_openai_compatible_settings() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="custom",
        embedding_model="qwen3-embedding:4b",
        openai_compatible_api_key="ok",
        openai_compatible_base_url="http://localhost:11434/v1",
    )

    with (
        patch.object(ec, "settings", fake_settings),
        patch.object(ec, "AsyncOpenAI") as mock_openai,
    ):
        client = ec._EmbeddingClient(provider="custom")

    assert client.provider == "custom"
    assert client.model == "qwen3-embedding:4b"
    mock_openai.assert_called_once_with(
        api_key="ok",
        base_url="http://localhost:11434/v1",
    )


def test_custom_provider_requires_base_url() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="custom",
        embedding_model=None,
        openai_compatible_api_key="ok",
        openai_compatible_base_url=None,
    )

    with (
        patch.object(ec, "settings", fake_settings),
        pytest.raises(
            ValueError,
            match="LLM_OPENAI_COMPATIBLE_BASE_URL",
        ),
    ):
        ec._EmbeddingClient(provider="custom")


def test_openrouter_provider_still_supported_as_alias() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="openrouter",
        embedding_model=None,
        openai_compatible_api_key="ok",
        openai_compatible_base_url=None,
    )

    with (
        patch.object(ec, "settings", fake_settings),
        patch.object(ec, "AsyncOpenAI") as mock_openai,
    ):
        client = ec._EmbeddingClient(provider="openrouter")

    assert client.provider == "openrouter"
    assert client.model == "openai/text-embedding-3-small"
    mock_openai.assert_called_once_with(
        api_key="ok",
        base_url="https://openrouter.ai/api/v1",
    )


def test_openai_provider_uses_openai_api_key() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="openai",
        embedding_model=None,
        openai_api_key="ok",
    )

    with (
        patch.object(ec, "settings", fake_settings),
        patch.object(ec, "AsyncOpenAI") as mock_openai,
    ):
        client = ec._EmbeddingClient(provider="openai")

    assert client.provider == "openai"
    assert client.model == "text-embedding-3-small"
    mock_openai.assert_called_once_with(api_key="ok")


def test_embedding_client_rejects_invalid_provider() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="invalid",
        openai_api_key="x",
        openai_compatible_api_key="x",
        openai_compatible_base_url="http://localhost:11434/v1",
        gemini_api_key="x",
    )

    with (
        patch.object(ec, "settings", fake_settings),
        pytest.raises(ValueError, match="Unsupported embedding provider"),
    ):
        ec._EmbeddingClient(provider="invalid")


@pytest.mark.asyncio
async def test_gemini_embed_calls_do_not_force_output_dimensionality_regression() -> (
    None
):
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="gemini",
        embedding_model="gemini-embedding-001",
        gemini_api_key="ok",
    )

    fake_response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
    )
    embed_content_mock = AsyncMock(
        side_effect=[
            fake_response,  # embed
            fake_response,  # simple_batch_embed
            fake_response,  # _process_batch
        ]
    )

    class FakeGenAIClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    embed_content=embed_content_mock,
                )
            )

    with (
        patch.object(ec, "settings", fake_settings),
        patch.object(ec.genai, "Client", FakeGenAIClient),
    ):
        client = ec._EmbeddingClient(provider="gemini")
        assert await client.embed("hello") == [0.1, 0.2, 0.3]
        assert await client.simple_batch_embed(["hello"]) == [[0.1, 0.2, 0.3]]
        await client._process_batch([ec.BatchItem("hello", "id-1", 0)])

    # Regression guard: no call should pass hardcoded output_dimensionality.
    for call in embed_content_mock.await_args_list:
        assert "config" not in call.kwargs


def test_wrapper_resolves_custom_provider_api_key_path() -> None:
    import src.embedding_client as ec

    fake_settings = _make_settings(
        provider="custom",
        openai_compatible_api_key="ok",
        openai_compatible_base_url="http://localhost:11434/v1",
    )

    with (
        patch.object(ec, "settings", fake_settings),
        patch.object(ec, "_EmbeddingClient") as mock_inner,
    ):
        wrapper = ec.EmbeddingClient()
        wrapper._get_client()

    mock_inner.assert_called_once_with(api_key="ok", provider="custom")


def test_models_vector_dimension_follows_settings_regression() -> None:
    import src.models as models

    assert (
        models.Document.__table__.c.embedding.type.dim
        == models.settings.VECTOR_STORE.DIMENSIONS
    )
    assert (
        models.MessageEmbedding.__table__.c.embedding.type.dim
        == models.settings.VECTOR_STORE.DIMENSIONS
    )
