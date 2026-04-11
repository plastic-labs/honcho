import pytest

from src.embedding_client import _EmbeddingClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, calls: list[dict], **kwargs):
        self.calls = calls
        self.kwargs = kwargs

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, *, json: dict, headers: dict[str, str]):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if url.endswith("/embed"):
            return _FakeResponse({"embedding": [0.0, 0.5]})
        if url.endswith("/embed_batch"):
            texts = json["texts"]
            return _FakeResponse(
                {
                    "embeddings": [
                        [float(i), float(i) + 0.5] for i, _ in enumerate(texts)
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.mark.asyncio
async def test_local_provider_uses_local_embedding_endpoint(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_async_client(**kwargs):
        return _FakeAsyncClient(calls, **kwargs)

    monkeypatch.setattr("src.embedding_client.httpx.AsyncClient", fake_async_client)
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_BASE_URL",
        "http://127.0.0.1:8100",
    )
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_MODEL",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_API_KEY", "local")

    client = _EmbeddingClient(provider="local")

    embedding = await client.embed("안녕하세요")
    batch = await client.simple_batch_embed(["하나", "둘"])

    assert client.model == "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    assert client.base_url == "http://127.0.0.1:8100"
    assert embedding == [0.0, 0.5]
    assert batch == [[0.0, 0.5], [1.0, 1.5]]
    assert calls == [
        {
            "url": "http://127.0.0.1:8100/embed",
            "json": {
                "text": "안녕하세요",
                "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            },
            "headers": {"Authorization": "Bearer local"},
        },
        {
            "url": "http://127.0.0.1:8100/embed_batch",
            "json": {
                "texts": ["하나", "둘"],
                "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            },
            "headers": {"Authorization": "Bearer local"},
        },
    ]


@pytest.mark.asyncio
async def test_local_provider_omits_auth_header_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict] = []

    def fake_async_client(**kwargs):
        return _FakeAsyncClient(calls, **kwargs)

    monkeypatch.setattr("src.embedding_client.httpx.AsyncClient", fake_async_client)
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_BASE_URL",
        "http://127.0.0.1:8100",
    )
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_MODEL",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_API_KEY", None)

    client = _EmbeddingClient(provider="local")

    embedding = await client.embed("안녕하세요")

    assert embedding == [0.0, 0.5]
    assert calls == [
        {
            "url": "http://127.0.0.1:8100/embed",
            "json": {
                "text": "안녕하세요",
                "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            },
            "headers": {},
        }
    ]


@pytest.mark.asyncio
async def test_local_provider_batch_raises_on_mismatched_embedding_count(
    monkeypatch: pytest.MonkeyPatch,
):
    class _MismatchedBatchClient(_FakeAsyncClient):
        async def post(self, url: str, *, json: dict, headers: dict[str, str]):
            self.calls.append({"url": url, "json": json, "headers": headers})
            if url.endswith("/embed_batch"):
                return _FakeResponse({"embeddings": [[0.0, 0.5]]})
            return await super().post(url, json=json, headers=headers)

    def fake_async_client(**kwargs):
        return _MismatchedBatchClient([], **kwargs)

    monkeypatch.setattr("src.embedding_client.httpx.AsyncClient", fake_async_client)
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_BASE_URL",
        "http://127.0.0.1:8100",
    )
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_MODEL",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    )
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_API_KEY", None)

    client = _EmbeddingClient(provider="local")

    with pytest.raises(ValueError, match="expected 2, got 1"):
        await client.simple_batch_embed(["하나", "둘"])


def test_local_provider_requires_base_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_BASE_URL", None)

    with pytest.raises(ValueError, match="LLM_EMBEDDING_BASE_URL is required"):
        _EmbeddingClient(provider="local")


def test_openrouter_uses_openai_compatible_base_url(monkeypatch: pytest.MonkeyPatch):
    created: dict = {}

    class _FakeEmbeddingsAPI:
        async def create(self, *, model: str, input_text: str | list[str]):
            _ = (model, input_text)
            return type(
                "EmbeddingResponse",
                (),
                {"data": [type("EmbeddingItem", (), {"embedding": [0.0, 0.5]})]},
            )

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            created["api_key"] = api_key
            created["base_url"] = base_url
            self.embeddings = _FakeEmbeddingsAPI()

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr("src.embedding_client.settings.LLM.EMBEDDING_PROVIDER", "openrouter")
    monkeypatch.setattr("src.embedding_client.settings.LLM.OPENAI_COMPATIBLE_API_KEY", "router-key")
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.OPENAI_COMPATIBLE_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_BASE_URL",
        "http://127.0.0.1:8100",
    )
    monkeypatch.setattr(
        "src.embedding_client.settings.LLM.EMBEDDING_MODEL",
        "custom-embed-model",
    )

    client = _EmbeddingClient(provider="openrouter")

    assert created["base_url"] == "https://openrouter.ai/api/v1"
    assert client.model == "custom-embed-model"
