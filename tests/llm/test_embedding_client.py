from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.config import EmbeddingModelConfig, settings
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]


def _fake_httpx_response(
    json_data: dict[str, Any],
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://localhost:8000/v1/embeddings"),
    )


def _build_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding: list[float],
    model: str,
    send_dimensions: bool,
    vector_dimensions: int,
    base_url: str = "http://localhost:8000/v1",
) -> tuple[_EmbeddingClient, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    async def fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        json_body: Any = kwargs.get("json")
        headers: Any = kwargs.get("headers")
        calls.append({"url": url, "json": json_body, "headers": headers})
        input_data = json_body if isinstance(json_body, dict) and "input" in json_body else {}
        inputs: list[str] = input_data.get("input", []) if isinstance(input_data.get("input"), list) else []
        return _fake_httpx_response(
            {"data": [{"embedding": embedding} for _ in (inputs or [""])]}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model=model,
            api_key="test-key",
            base_url=base_url,
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=send_dimensions,
    )
    return client, calls


@pytest.mark.asyncio
async def test_openai_embedding_client_uses_configured_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.1] * 8
    assert len(calls) == 1
    assert calls[0]["json"] == {"model": "text-embedding-3-small", "input": ["hello world"]}
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert calls[0]["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_openai_embedding_client_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 7,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
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
async def test_openai_embed_forwards_dimensions_when_send_dimensions_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.embed("hello")

    assert calls[0]["json"] == {
        "model": "text-embedding-3-small",
        "input": ["hello"],
        "dimensions": 768,
    }


@pytest.mark.asyncio
async def test_openai_embed_omits_dimensions_when_send_dimensions_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * settings.EMBEDDING.VECTOR_DIMENSIONS,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
    )

    await client.embed("hello")

    assert calls[0]["json"] == {"model": "text-embedding-3-small", "input": ["hello"]}


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.simple_batch_embed(["a", "b"])

    assert len(calls) == 1
    assert calls[0]["json"]["dimensions"] == 768
    assert calls[0]["json"]["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, calls = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(calls) == 1
    assert calls[0]["json"]["dimensions"] == 768


def _build_embedding_settings(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
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
    calls: list[dict[str, Any]] = []

    async def fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        json_body: Any = kwargs.get("json")
        input_data = json_body if isinstance(json_body, dict) else {}
        inputs: list[str] = input_data.get("input", []) if isinstance(input_data.get("input"), list) else []
        calls.append({"json": json_body})
        return _fake_httpx_response(
            {"data": [{"embedding": [0.5] * 4} for _ in inputs]}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
        ),
        vector_dimensions=4,
        max_input_tokens=100,
        max_tokens_per_request=120,
        send_dimensions=False,
    )

    long_a = ("alpha " * 80).strip()
    long_b = ("beta " * 80).strip()

    out = await client.simple_batch_embed([long_a, long_b])
    assert len(out) == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_simple_batch_embed_rejects_oversized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return _fake_httpx_response({"data": [{"embedding": [0.1] * 4}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
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
    async def fake_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return _fake_httpx_response({"data": [{"embedding": [0.1] * 4}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
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
    assert isinstance(out["long"][0], str)
