from types import SimpleNamespace
from typing import Any

import pytest

from src.config import EmbeddingModelConfig
from src.embedding_client import (
    _EmbeddingClient,  # pyright: ignore[reportPrivateUsage]
    _resolve_tokenizer,  # pyright: ignore[reportPrivateUsage]
)
from src.exceptions import ValidationException


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


class FakeEncoding:
    """Deterministic tokenizer double: every text maps to `token_count` tokens."""

    def __init__(self, token_count: int) -> None:
        self.token_count: int = token_count

    def encode(self, _text: str) -> list[int]:
        return list(range(self.token_count))

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


def test_resolve_tokenizer_returns_model_encoding_for_known_models() -> None:
    tokenizer = _resolve_tokenizer("text-embedding-3-small", None)
    assert tokenizer.encode("hello") is not None


def test_resolve_tokenizer_warns_and_falls_back_for_unknown_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="src.embedding_client"):
        tokenizer = _resolve_tokenizer("baai/bge-m3", None)

    assert tokenizer.encode("hello") is not None
    assert any(
        "falling back to cl100k_base" in record.message for record in caplog.records
    )


def test_resolve_tokenizer_blank_spec_uses_default() -> None:
    default = _resolve_tokenizer("text-embedding-3-small", None)
    blank = _resolve_tokenizer("text-embedding-3-small", "   ")
    assert type(blank) is type(default)


def test_resolve_tokenizer_tiktoken_spec_uses_named_encoding() -> None:
    import tiktoken

    tokenizer = _resolve_tokenizer("text-embedding-3-small", "tiktoken:o200k_base")
    assert tokenizer is tiktoken.get_encoding("o200k_base")


def test_resolve_tokenizer_rejects_unknown_tiktoken_encoding() -> None:
    with pytest.raises(ValidationException, match="unknown tiktoken encoding"):
        _resolve_tokenizer("text-embedding-3-small", "tiktoken:not-a-real-encoding")


def test_resolve_tokenizer_rejects_empty_tiktoken_spec() -> None:
    with pytest.raises(ValidationException, match="requires an encoding name"):
        _resolve_tokenizer("text-embedding-3-small", "tiktoken:")


def test_resolve_tokenizer_rejects_unknown_prefix() -> None:
    with pytest.raises(ValidationException, match="tiktoken:, hf:, file:"):
        _resolve_tokenizer("text-embedding-3-small", "sentencepiece:foo")


def test_resolve_tokenizer_rejects_empty_hf_and_file_specs() -> None:
    with pytest.raises(ValidationException, match="requires a model name"):
        _resolve_tokenizer("text-embedding-3-small", "hf:")
    with pytest.raises(ValidationException, match="requires a tokenizer path"):
        _resolve_tokenizer("text-embedding-3-small", "file:")


def test_resolve_tokenizer_hf_requires_tokenizers_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr("src.embedding_client.import_module", fake_import_module)

    with pytest.raises(ValidationException, match=r"honcho\[tokenizers\]"):
        _resolve_tokenizer("text-embedding-3-small", "hf:BAAI/bge-m3")


def test_resolve_tokenizer_hf_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeTokenizerCls:
        @staticmethod
        def from_pretrained(model_name: str) -> Any:
            calls.append(model_name)

            def _encode(_text: str, add_special_tokens: bool = True) -> Any:
                _ = add_special_tokens
                return SimpleNamespace(ids=[1, 2, 3])

            def _decode(_ids: list[int], skip_special_tokens: bool = True) -> str:
                _ = skip_special_tokens
                return "decoded"

            return SimpleNamespace(encode=_encode, decode=_decode)

    def _fake_import_module(_name: str) -> Any:
        return SimpleNamespace(Tokenizer=FakeTokenizerCls)

    monkeypatch.setattr("src.embedding_client.import_module", _fake_import_module)

    tokenizer = _resolve_tokenizer("text-embedding-3-small", "hf:BAAI/bge-m3")

    assert calls == ["BAAI/bge-m3"]
    assert tokenizer.encode("anything") == [1, 2, 3]
    assert tokenizer.decode([1, 2, 3]) == "decoded"


def test_resolve_tokenizer_loads_from_file(tmp_path: Any) -> None:
    tokenizers = pytest.importorskip("tokenizers")

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"hello": 0, "world": 1})
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    loaded = _resolve_tokenizer("text-embedding-3-small", f"file:{path}")

    assert loaded.encode("hello world") == [0, 1]
    assert loaded.decode([0, 1]) == "hello world"


def test_embedding_client_uses_configured_tokenizer_for_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk-size decisions must follow the configured tokenizer, not the
    tiktoken fallback — the core of issue #827."""
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    def _fake_get_encoding(name: str) -> FakeEncoding | None:
        return FakeEncoding(token_count=100) if name == "fake-enc" else None

    monkeypatch.setattr(
        "src.embedding_client.tiktoken.get_encoding", _fake_get_encoding
    )

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            tokenizer="tiktoken:fake-enc",
        ),
        vector_dimensions=4,
        max_input_tokens=10,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    # cl100k_base would count "hi" as 1 token and not chunk; the configured
    # tokenizer counts 100 tokens, so it must be split into <=10-token chunks.
    chunks = client.prepare_chunks({"msg": "hi"})["msg"]

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 10


def test_hf_tokenizer_special_tokens_reserved_from_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BERT-family providers count [CLS]/[SEP] toward the input limit, so the
    client's effective budget must shrink by the special-token overhead."""

    def _encode(text: str, add_special_tokens: bool = True) -> Any:
        ids = [10 + i for i, _ in enumerate(text.split())]
        return SimpleNamespace(ids=([0, *ids, 1] if add_special_tokens else ids))

    def _decode(ids: list[int], **_kwargs: Any) -> str:
        return " ".join("w" for _ in ids)

    class FakeTokenizerCls:
        @staticmethod
        def from_file(_path: str) -> Any:
            return SimpleNamespace(encode=_encode, decode=_decode)

    def _fake_import_module(_name: str) -> Any:
        return SimpleNamespace(Tokenizer=FakeTokenizerCls)

    monkeypatch.setattr("src.embedding_client.import_module", _fake_import_module)

    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 4)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="baai/bge-m3",
            api_key="test-key",
            tokenizer="file:/tmp/whatever.json",
        ),
        vector_dimensions=4,
        max_input_tokens=100,
        max_tokens_per_request=1000,
        send_dimensions=False,
    )

    # 2 specials ([CLS]/[SEP]) reserved: budget is 98, not 100.
    assert client.max_embedding_tokens == 98


def test_settings_signature_tracks_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config import settings
    from src.embedding_client import EmbeddingClient

    wrapper = EmbeddingClient()
    before = wrapper._get_settings_signature()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(
        settings.EMBEDDING.MODEL_CONFIG, "tokenizer", "tiktoken:o200k_base"
    )
    after = wrapper._get_settings_signature()  # pyright: ignore[reportPrivateUsage]

    assert before != after
