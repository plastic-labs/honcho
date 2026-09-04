"""Drive Honcho's real provider clients against the mock over ASGI.

The unit tests assert the mock's own output. These assert the hop that actually
matters: ``OpenAIBackend`` and ``EmbeddingClient`` — the production classes,
unpatched — talking to the mock through the genuine OpenAI SDK, including the
``strict: true`` json_schema transform that ``chat.completions.parse()`` applies
on the way out and the Pydantic validation it applies on the way back.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from src.config import EmbeddingModelConfig
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]
from src.llm.backends.openai import OpenAIBackend
from src.mock_provider.embeddings import content_to_embedding
from src.mock_provider.main import app
from src.utils.representation import PromptRepresentation

MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "I switched the service from pip to uv last week."}
]


@pytest.fixture
def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="sandbox",
        base_url="http://mock-provider.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
    )


@pytest.mark.asyncio
async def test_backend_parses_the_deriver_response_model(
    openai_client: AsyncOpenAI,
) -> None:
    """The production path: parse() with a Pydantic response_format."""
    backend = OpenAIBackend(openai_client)

    result = await backend.complete(
        model="mock-model",
        messages=MESSAGES,
        max_tokens=512,
        response_format=PromptRepresentation,
    )

    assert isinstance(result.content, PromptRepresentation)
    # An empty explicit list is what a prose-answering mock silently produces,
    # so it is the specific thing worth asserting against.
    assert result.content.explicit
    assert result.output_tokens > 0


@pytest.mark.asyncio
async def test_backend_json_object_mode_recovers_the_schema(
    openai_client: AsyncOpenAI,
) -> None:
    """json_object mode carries the schema in the prompt, not response_format."""
    backend = OpenAIBackend(openai_client)

    result = await backend.complete(
        model="mock-model",
        messages=MESSAGES,
        max_tokens=512,
        response_format=PromptRepresentation,
        extra_params={"structured_output_mode": "json_object"},
    )

    assert isinstance(result.content, PromptRepresentation)
    assert result.content.explicit


@pytest.mark.asyncio
async def test_backend_with_tools_uses_json_schema_and_still_parses(
    openai_client: AsyncOpenAI,
) -> None:
    """Non-strict tools force create() + explicit json_schema instead of parse()."""
    backend = OpenAIBackend(openai_client)

    result = await backend.complete(
        model="mock-model",
        messages=MESSAGES,
        max_tokens=512,
        response_format=PromptRepresentation,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": "Search memory",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert isinstance(result.content, PromptRepresentation)
    assert result.content.explicit


@pytest.mark.asyncio
async def test_backend_plain_completion(openai_client: AsyncOpenAI) -> None:
    backend = OpenAIBackend(openai_client)

    result = await backend.complete(
        model="mock-model", messages=MESSAGES, max_tokens=128
    )

    assert isinstance(result.content, str)
    assert "[mock]" in result.content
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_backend_stream_yields_content_then_a_usage_terminator(
    openai_client: AsyncOpenAI,
) -> None:
    backend = OpenAIBackend(openai_client)

    chunks = [
        chunk
        async for chunk in backend.stream(
            model="mock-model", messages=MESSAGES, max_tokens=128
        )
    ]

    assert "[mock]" in "".join(chunk.content or "" for chunk in chunks)

    terminator = chunks[-1]
    assert terminator.is_done
    assert terminator.finish_reason == "stop"
    # None here means the stream ended without a usage chunk, which is the
    # failure mode when stream_options.include_usage goes unanswered.
    assert terminator.output_tokens is not None
    assert terminator.output_tokens > 0


def _embedding_client(dimensions: int, encoding_format: str) -> _EmbeddingClient:
    # The public EmbeddingClient is a settings-driven singleton wrapper; the
    # transport behaviour under test lives on the implementation it wraps.
    return _EmbeddingClient(
        EmbeddingModelConfig(
            model="text-embedding-3-small",
            transport="openai",
            api_key="sandbox",
            base_url="http://mock-provider.invalid/v1",
        ),
        vector_dimensions=dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300000,
        send_dimensions=True,
        encoding_format=encoding_format,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture(autouse=True)
def _route_embedding_client_over_asgi(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give the embedding client's AsyncOpenAI an ASGI transport.

    EmbeddingClient builds its own client internally, so the transport has to be
    injected at construction rather than passed in.
    """
    original = AsyncOpenAI.__init__

    def patched(self: AsyncOpenAI, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault(
            "http_client",
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
        )
        original(self, *args, **kwargs)

    monkeypatch.setattr(AsyncOpenAI, "__init__", patched)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
async def test_embedding_client_round_trip(encoding_format: str) -> None:
    """Covers both wire encodings; base64 is what the SDK uses by default."""
    client = _embedding_client(1536, encoding_format)

    vector = await client.embed("I switched the service from pip to uv.")

    # _validate_embedding_dimensions raises on a width mismatch, so reaching
    # here already proves the width is right; assert the values too.
    assert len(vector) == 1536
    assert vector == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        content_to_embedding("I switched the service from pip to uv.", 1536),
        abs=1e-6,
    )


@pytest.mark.asyncio
async def test_embedding_client_honours_a_non_default_dimension() -> None:
    """send_dimensions=True forwards `dimensions`; the mock must obey it."""
    client = _embedding_client(256, "float")

    assert len(await client.embed("hello")) == 256


@pytest.mark.asyncio
async def test_embedding_client_batches() -> None:
    """_validate_embedding_count rejects a mismatched count."""
    client = _embedding_client(1536, "float")
    texts = [f"observation number {index}" for index in range(12)]

    vectors = await client.simple_batch_embed(texts)

    assert len(vectors) == len(texts)
    assert all(len(vector) == 1536 for vector in vectors)
    assert len({tuple(vector) for vector in vectors}) == len(texts)
