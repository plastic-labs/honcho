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


class FakeGeminiModelsBatch:
    """Fake Gemini `models` API allowing distinct behavior for batch vs.
    single-item calls, so tests can exercise the batch->per-item fallback."""

    def __init__(
        self,
        *,
        batch_embeddings_fn: Any = None,
        single_embeddings_fn: Any = None,
    ) -> None:
        # Deterministic function of the batch contents -> embeddings list,
        # so retries (which re-invoke the whole batch call) are idempotent.
        self.batch_embeddings_fn = batch_embeddings_fn
        self.single_embeddings_fn = single_embeddings_fn
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    async def embed_content(
        self,
        *,
        model: str,
        contents: str | list[str],
        config: dict[str, Any],
    ) -> SimpleNamespace:
        if isinstance(contents, list):
            self.batch_calls.append(contents)
            embeddings = self.batch_embeddings_fn(contents)
            return SimpleNamespace(embeddings=embeddings)
        else:
            self.single_calls.append(contents)
            embeddings = self.single_embeddings_fn(contents)
            return SimpleNamespace(embeddings=embeddings)


def _build_gemini_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_models: FakeGeminiModelsBatch,
    *,
    vector_dimensions: int = 4,
) -> _EmbeddingClient:
    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=fake_models)

    monkeypatch.setattr("src.embedding_client.genai.Client", FakeGeminiClient)

    return _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )


@pytest.mark.asyncio
async def test_gemini_batch_embed_single_item_normal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) A single-item batch where Gemini returns exactly one embedding."""
    fake_models = FakeGeminiModelsBatch(
        batch_embeddings_fn=lambda contents: [SimpleNamespace(values=[0.1] * 4)],
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    result = await client.batch_embed({"only": "hello world"})

    assert result == {"only": [[0.1] * 4]}
    assert fake_models.batch_calls == [["hello world"]]
    assert fake_models.single_calls == []


@pytest.mark.asyncio
async def test_gemini_batch_embed_multi_item_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A multi-item batch where Gemini returns one embedding per item,
    in order -- verify each text_id maps to the correct embedding."""
    fake_models = FakeGeminiModelsBatch(
        batch_embeddings_fn=lambda contents: [
            SimpleNamespace(values=[0.1] * 4),
            SimpleNamespace(values=[0.2] * 4),
            SimpleNamespace(values=[0.3] * 4),
        ],
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    result = await client.batch_embed({"first": "aaa", "second": "bbb", "third": "ccc"})

    assert result == {
        "first": [[0.1] * 4],
        "second": [[0.2] * 4],
        "third": [[0.3] * 4],
    }
    assert fake_models.batch_calls == [["aaa", "bbb", "ccc"]]
    assert fake_models.single_calls == []


@pytest.mark.asyncio
async def test_gemini_batch_embed_mismatched_response_falls_back_per_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) Gemini returns fewer embeddings than input items for a batch
    call. Must NOT crash (previously: zip(strict=True) ValueError) and
    must NOT silently misalign -- instead falls back to per-item calls
    and correctly maps every item to its own embedding."""

    def single_embeddings_fn(text: str) -> list[Any]:
        mapping = {
            "aaa": [0.1] * 4,
            "bbb": [0.2] * 4,
            "ccc": [0.3] * 4,
        }
        return [SimpleNamespace(values=mapping[text])]

    fake_models = FakeGeminiModelsBatch(
        # Batch call returns only 2 embeddings for 3 input items.
        batch_embeddings_fn=lambda contents: [
            SimpleNamespace(values=[0.9] * 4),
            SimpleNamespace(values=[0.9] * 4),
        ],
        single_embeddings_fn=single_embeddings_fn,
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    result = await client.batch_embed({"first": "aaa", "second": "bbb", "third": "ccc"})

    assert result == {
        "first": [[0.1] * 4],
        "second": [[0.2] * 4],
        "third": [[0.3] * 4],
    }
    # One initial batch call, then one fallback call per item.
    assert fake_models.batch_calls == [["aaa", "bbb", "ccc"]]
    assert fake_models.single_calls == ["aaa", "bbb", "ccc"]


@pytest.mark.asyncio
async def test_gemini_batch_embed_fallback_call_wrong_count_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) During the per-item fallback, if an individual call itself
    returns zero or multiple embeddings, a controlled ValueError is
    raised (not a crash / silent misalignment)."""

    def single_embeddings_fn(text: str) -> list[Any]:
        if text == "bbb":
            return []  # zero embeddings for this item -- should raise
        return [SimpleNamespace(values=[0.1] * 4)]

    fake_models = FakeGeminiModelsBatch(
        batch_embeddings_fn=lambda contents: [SimpleNamespace(values=[0.9] * 4)],
        single_embeddings_fn=single_embeddings_fn,
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    with pytest.raises(ValueError, match="Gemini API returned"):
        await client.batch_embed({"first": "aaa", "second": "bbb"})


@pytest.mark.asyncio
async def test_gemini_batch_embed_fallback_call_multiple_embeddings_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) During the per-item fallback, if an individual call returns
    MULTIPLE embeddings for a single item, a controlled ValueError is
    raised (not a silent pick-one / misalignment)."""

    def single_embeddings_fn(text: str) -> list[Any]:
        if text == "bbb":
            return [
                SimpleNamespace(values=[0.1] * 4),
                SimpleNamespace(values=[0.2] * 4),
            ]  # two embeddings for this item -- should raise
        return [SimpleNamespace(values=[0.1] * 4)]

    fake_models = FakeGeminiModelsBatch(
        batch_embeddings_fn=lambda contents: [SimpleNamespace(values=[0.9] * 4)],
        single_embeddings_fn=single_embeddings_fn,
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    with pytest.raises(ValueError, match="Gemini API returned"):
        await client.batch_embed({"first": "aaa", "second": "bbb"})


@pytest.mark.asyncio
async def test_gemini_batch_embed_empty_values_in_matched_batch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(f) In the cardinality-matched batch (zip) path, if one returned
    embedding has falsy `values` (empty list), it must raise a
    ValueError instead of silently omitting that item from the result."""
    fake_models = FakeGeminiModelsBatch(
        batch_embeddings_fn=lambda contents: [
            SimpleNamespace(values=[0.1] * 4),
            SimpleNamespace(values=[]),  # falsy values -- should raise
            SimpleNamespace(values=[0.3] * 4),
        ],
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    with pytest.raises(ValueError, match="no values"):
        await client.batch_embed({"first": "aaa", "second": "bbb", "third": "ccc"})


@pytest.mark.asyncio
async def test_gemini_batch_embed_fallback_empty_values_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(g) In the per-item fallback path, if the single embedding
    returned has falsy `values`, it must raise a ValueError instead of
    silently omitting that item from the result."""

    def single_embeddings_fn(text: str) -> list[Any]:
        if text == "bbb":
            return [SimpleNamespace(values=[])]  # falsy values -- should raise
        return [SimpleNamespace(values=[0.1] * 4)]

    fake_models = FakeGeminiModelsBatch(
        # Force cardinality mismatch to enter the fallback path.
        batch_embeddings_fn=lambda contents: [SimpleNamespace(values=[0.9] * 4)],
        single_embeddings_fn=single_embeddings_fn,
    )
    client = _build_gemini_client(monkeypatch, fake_models)

    with pytest.raises(ValueError, match="no values"):
        await client.batch_embed({"first": "aaa", "second": "bbb"})
