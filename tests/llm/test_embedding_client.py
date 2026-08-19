import array
import base64
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import types as genai_types

from src.config import (
    EmbeddingEncodingFormat,
    EmbeddingModelConfig,
    resolve_embedding_model_config,
)
from src.embedding_client import (
    BatchItem,
    _EmbeddingClient,  # pyright: ignore[reportPrivateUsage]
)


def gemini_call_texts(contents: Any) -> list[str]:
    """Unwrap a recorded Gemini `contents` argument back to plain texts."""
    return [content.parts[0].text for content in contents]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding: list[float] = embedding
        self.calls: list[dict[str, Any]] = []
        # Simulate a provider answering 200 with missing embeddings.
        self.returns_no_data: bool = False
        self.truncate_data_to: int | None = None

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
        # Mirror the SDK: a named encoding_format skips its base64 decode, so the
        # response carries the raw string instead of floats.
        payload: Any = self.embedding
        if kwargs.get("encoding_format") == "base64":
            payload = base64.b64encode(
                array.array("f", self.embedding).tobytes()
            ).decode()
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=payload) for _ in input]
        else:
            data = [SimpleNamespace(embedding=payload)]
        if self.returns_no_data:
            data = []
        elif self.truncate_data_to is not None:
            data = data[: self.truncate_data_to]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_openai_embedding_client_uses_configured_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 8)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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
        {
            "model": "text-embedding-3-small",
            "input": ["hello world"],
            "encoding_format": "float",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 7)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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
            contents: Any,
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

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

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
    # 10-minute HTTP timeout, in lockstep with the LLM registry's Gemini client
    # (see #785). Without this, a stalled Gemini embedding socket wedges the
    # deriver worker — the same failure mode the LLM fix addresses.
    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.base_url == "https://gemini-proxy.example/v1beta"
    assert gemini_client.http_options.timeout == 600_000
    assert calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "hello world",
            "config": {"output_dimensionality": 12},
        }
    ]


@pytest.mark.asyncio
async def test_gemini_embedding_client_keeps_timeout_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-base-url Gemini embedding client must still carry an HTTP timeout."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.base_url is None
    assert gemini_client.http_options.timeout == 600_000


@pytest.mark.asyncio
async def test_openai_embedding_client_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured embedding timeout reaches the OpenAI-compatible client."""

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.timeout: float | None = timeout
            self.embeddings: FakeOpenAIEmbeddingsAPI = FakeOpenAIEmbeddingsAPI(
                [0.1] * 8
            )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            timeout=45,
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    openai_client = cast(Any, client.client)
    assert openai_client.timeout == 45.0


@pytest.mark.asyncio
async def test_openai_embedding_client_omits_timeout_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset timeout omits the kwarg so the OpenAI SDK keeps its default."""

    missing = object()

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: object = missing,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.timeout: object = timeout
            self.embeddings: FakeOpenAIEmbeddingsAPI = FakeOpenAIEmbeddingsAPI(
                [0.1] * 8
            )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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

    openai_client = cast(Any, client.client)
    assert openai_client.timeout is missing


@pytest.mark.asyncio
async def test_gemini_embedding_client_forwards_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured embedding timeout reaches Gemini as milliseconds."""

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=SimpleNamespace())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            timeout=45,
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    gemini_client = cast(Any, client.client)
    assert gemini_client.http_options.timeout == 45_000


def _build_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding: list[float],
    model: str,
    send_dimensions: bool,
    vector_dimensions: int,
    max_batch_size: int | None = None,
    encoding_format: EmbeddingEncodingFormat = "float",
) -> tuple[_EmbeddingClient, FakeOpenAIEmbeddingsAPI]:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding)

    class FakeOpenAIClient:
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model=model,
            api_key="test-key",
            max_batch_size=max_batch_size,
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=send_dimensions,
        encoding_format=encoding_format,
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
            "encoding_format": "float",
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

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "encoding_format": "float",
        }
    ]


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
async def test_openai_simple_batch_embed_respects_configured_max_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
        max_batch_size=2,
    )

    await client.simple_batch_embed(["a", "b", "c"])

    assert [call["input"] for call in fake.calls] == [["a", "b"], ["c"]]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_defaults_to_2048_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset max_batch_size must keep the OpenAI default: one request."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
    )
    assert client.max_batch_size == 2048

    await client.simple_batch_embed(["a", "b", "c"])

    assert [call["input"] for call in fake.calls] == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_gemini_simple_batch_embed_respects_configured_max_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini transport must split batches at the configured limit too."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            n = len(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12) for _ in range(n)]
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            max_batch_size=2,
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    await client.simple_batch_embed(["a", "b", "c"])

    assert [gemini_call_texts(call["contents"]) for call in calls] == [
        ["a", "b"],
        ["c"],
    ]


@pytest.mark.asyncio
async def test_gemini_simple_batch_embed_defaults_to_100_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset max_batch_size must keep the Gemini conservative default."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            n = len(contents)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12) for _ in range(n)]
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )
    assert client.max_batch_size == 100

    await client.simple_batch_embed(["a", "b", "c"])

    assert [gemini_call_texts(call["contents"]) for call in calls] == [["a", "b", "c"]]


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


@pytest.mark.asyncio
async def test_openai_embed_requests_float_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single-query path must request float embeddings explicitly.

    Without an explicit encoding_format, the openai SDK defaults to base64,
    which OpenAI-compatible providers such as OpenRouter answer with empty
    embedding data for models that don't support base64 encoding.
    """
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )

    await client.embed("hello")

    assert fake.calls[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_openai_batch_embed_requests_float_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch path must request float embeddings explicitly, like embed()."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(fake.calls) == 1
    assert fake.calls[0]["encoding_format"] == "float"


@pytest.mark.asyncio
async def test_openai_embed_reports_missing_embedding_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit encoding_format turns off the SDK's own empty-data check, so
    a provider answering 200 with no embeddings must still fail legibly."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )
    fake.returns_no_data = True

    with pytest.raises(ValueError, match="Embedding count mismatch"):
        await client.embed("hello")


@pytest.mark.asyncio
async def test_openai_batch_embed_reports_short_embedding_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch answered with fewer embeddings than inputs must name the counts
    rather than surface a bare zip() error."""
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
    )
    fake.truncate_data_to = 1

    with pytest.raises(ValueError, match="Expected 2, got 1"):
        await client.batch_embed({"a": "hello", "b": "world"})


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
        "EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE",
        "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL",
        "EMBEDDING_MODEL_CONFIG__MAX_BATCH_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return EmbeddingSettings()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # No base_url means real OpenAI, which serves base64 at ~1/3.6 the bytes.
        ({}, "base64"),
        (
            {
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://api.openai.com/v1"
            },
            "base64",
        ),
        (
            {
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://openrouter.ai/api/v1"
            },
            "float",
        ),
        (
            {"EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "http://localhost:8000/v1"},
            "float",
        ),
        ({"EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE": "float"}, "float"),
        (
            {
                "EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT_MODE": "base64",
                "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL": "https://openrouter.ai/api/v1",
            },
            "base64",
        ),
    ],
)
def test_resolve_encoding_format(
    env: dict[str, str], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _build_embedding_settings(env, monkeypatch)
    assert s.resolve_encoding_format() == expected


@pytest.mark.asyncio
async def test_openai_base64_mode_omits_encoding_format_and_returns_floats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """base64 mode must request by omission on both paths.

    Naming `base64` explicitly makes the SDK skip its own decode and hand back
    the raw string, which then fails the dimension check.
    """
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 8,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=8,
        encoding_format="base64",
    )

    embedding = await client.embed("hello")
    batched = await client.batch_embed({"a": "hello", "b": "world"})

    assert all("encoding_format" not in call for call in fake.calls)
    assert len(embedding) == 8
    assert [len(vectors[0]) for vectors in batched.values()] == [8, 8]


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
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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
        def __init__(
            self,
            *,
            api_key: str | None,
            base_url: str | None,
            timeout: float | None = None,
        ) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAIClient)

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


def test_embedding_model_config_parses_max_batch_size_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__MAX_BATCH_SIZE": "10"},
        monkeypatch,
    )

    assert s.MODEL_CONFIG.max_batch_size == 10

    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)
    assert resolved.max_batch_size == 10


def test_embedding_model_config_parses_timeout_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__TIMEOUT": "90.0"},
        monkeypatch,
    )

    assert s.MODEL_CONFIG.timeout == 90.0

    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)
    assert resolved.timeout == 90.0


def test_embedding_model_config_rejects_invalid_timeout() -> None:
    with pytest.raises(
        ValueError, match=r"provider_params\.timeout must be a positive number"
    ):
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            timeout=-1,
        )


@pytest.mark.asyncio
async def test_gemini_process_batch_wraps_contents_as_content_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each batch item must be its own Content so gemini-embedding-2* returns
    one embedding per item instead of merging them into one document."""
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: Any,
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append({"model": model, "contents": contents, "config": config})
            embeddings = [SimpleNamespace(values=[0.3] * 8) for _ in contents]
            return SimpleNamespace(embeddings=embeddings)

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("google.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-2",
            api_key="gemini-key",
            base_url=None,
        ),
        vector_dimensions=8,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    batch = [
        BatchItem("hello", "id1", 0, 1),
        BatchItem("world", "id2", 0, 1),
    ]
    result = await client._process_batch(batch)  # pyright: ignore[reportPrivateUsage]

    assert result["id1"][0] == [0.3] * 8
    assert result["id2"][0] == [0.3] * 8

    assert len(calls) == 1
    contents = calls[0]["contents"]
    assert len(contents) == 2
    assert all(isinstance(c, genai_types.Content) for c in contents)
    assert contents[0].parts[0].text == "hello"
    assert contents[1].parts[0].text == "world"
