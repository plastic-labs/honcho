from __future__ import annotations

import time
from typing import Any, cast

import httpx
import openai
import pytest
from openai import AsyncOpenAI

from src.config import EmbeddingTransport

from .conftest import cosine_similarity, make_embedding_client
from .embedding_matrix import LiveEmbeddingSpec, get_live_embedding_specs

pytestmark = pytest.mark.live_llm

# Deliberately unrelated topics so a mix-up between them is visible in cosine
# similarity rather than lost in noise.
BATCH_TEXTS: list[str] = [
    "The mitochondria generates ATP through oxidative phosphorylation.",
    "Barcelona won the treble in the 2014-15 football season.",
    "Sourdough starter needs regular feeding with flour and water.",
    "Rust's borrow checker enforces ownership rules at compile time.",
]

ALL_SPECS = get_live_embedding_specs()
GEMINI_SPECS = get_live_embedding_specs(transport="gemini")
OPENAI_NATIVE_SPECS = tuple(
    spec for spec in ALL_SPECS if spec.family == "openai_embedding"
)

GENEROUS_TIMEOUT_SECONDS = 120.0
TIGHT_TIMEOUT_SECONDS = 0.01
# Well under the client defaults; generous enough to absorb SDK retries.
TIGHT_TIMEOUT_WALL_CLOCK_LIMIT_SECONDS = 30

EMBEDDING_TIMEOUT_EXCEPTIONS: dict[
    EmbeddingTransport, tuple[type[BaseException], ...]
] = {
    "openai": (openai.APITimeoutError,),
    # google-genai raises httpx or aiohttp timeouts depending on its transport;
    # aiohttp surfaces as asyncio.TimeoutError (== builtins.TimeoutError).
    "gemini": (httpx.TimeoutException, TimeoutError),
}

TRANSPORT_MARKS = {
    "openai": pytest.mark.requires_openai,
    "gemini": pytest.mark.requires_gemini,
}


def representative_embedding_specs() -> list[Any]:
    """One spec per transport — timeout plumbing is client-level, not model-level."""
    params: list[Any] = []
    for transport in ("openai", "gemini"):
        specs = get_live_embedding_specs(transport=transport)
        if not specs:
            continue
        # Prefer the native family over openai-compatible proxies.
        family = f"{transport}_embedding"
        native = next((s for s in specs if s.family == family), specs[0])
        params.append(
            pytest.param(native, marks=TRANSPORT_MARKS[transport], id=native.id)
        )
    return params


def assert_embedding_timeout_on_client(
    client: Any, transport: EmbeddingTransport, timeout_seconds: float
) -> None:
    if transport == "gemini":
        http_options = client.client._api_client._http_options
        assert http_options.timeout == int(timeout_seconds * 1000)
        return
    openai_client = cast(AsyncOpenAI, client.client)
    assert openai_client.timeout == timeout_seconds


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.id)
async def test_live_embed_single_returns_configured_dimensions(
    spec: LiveEmbeddingSpec,
) -> None:
    client = make_embedding_client(spec)

    embedding = await client.embed(BATCH_TEXTS[0])

    assert len(embedding) == spec.dimensions
    assert any(value != 0.0 for value in embedding)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.id)
async def test_live_batch_embed_returns_one_vector_per_input(
    spec: LiveEmbeddingSpec,
) -> None:
    """Regression guard for #745.

    `gemini-embedding-2*` treats a list of bare strings as parts of one
    document and returns a single embedding, which trips the strict zip in
    `_process_batch`. Each input must come back with its own distinct vector.
    """
    client = make_embedding_client(spec)

    embeddings = await client.simple_batch_embed(BATCH_TEXTS)

    assert len(embeddings) == len(BATCH_TEXTS)
    assert all(len(embedding) == spec.dimensions for embedding in embeddings)
    # A collapsed batch would hand the same vector back for every input.
    assert len({tuple(embedding) for embedding in embeddings}) == len(BATCH_TEXTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.id)
async def test_live_batch_embed_aligns_with_single_embed(
    spec: LiveEmbeddingSpec,
) -> None:
    """Batched vectors must match the one-at-a-time vectors, position for
    position. Catches both a collapsed batch and a silently reordered one."""
    client = make_embedding_client(spec)

    batched = await client.simple_batch_embed(BATCH_TEXTS)
    singles = [await client.embed(text) for text in BATCH_TEXTS]

    for index, (batched_vector, single_vector) in enumerate(
        zip(batched, singles, strict=True)
    ):
        self_similarity = cosine_similarity(batched_vector, single_vector)
        assert self_similarity > 0.95, (
            f"{spec.id}: batched vector {index} does not match its own "
            f"single embedding (cosine={self_similarity:.3f})"
        )
        for other_index, other_single in enumerate(singles):
            if other_index == index:
                continue
            assert self_similarity > cosine_similarity(batched_vector, other_single), (
                f"{spec.id}: batched vector {index} is closer to text "
                f"{other_index} than to its own"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.id)
async def test_live_batch_embed_maps_chunks_to_their_ids(
    spec: LiveEmbeddingSpec,
) -> None:
    """`batch_embed` splits oversized inputs, so one request carries chunks
    belonging to several ids. Every id must get back exactly its own chunks."""
    client = make_embedding_client(spec)
    long_text = " ".join(
        f"paragraph {index} about photosynthesis" for index in range(900)
    )
    expected_chunks = {
        text_id: len(chunks)
        for text_id, chunks in client.prepare_chunks(
            {"long": long_text, "short": BATCH_TEXTS[1]}
        ).items()
    }
    assert expected_chunks["long"] > 1, "test input must exceed the token limit"

    result = await client.batch_embed({"long": long_text, "short": BATCH_TEXTS[1]})

    assert {text_id: len(vectors) for text_id, vectors in result.items()} == (
        expected_chunks
    )
    assert all(
        len(vector) == spec.dimensions
        for vectors in result.values()
        for vector in vectors
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", OPENAI_NATIVE_SPECS, ids=lambda spec: spec.id)
async def test_live_openai_float_encoding_matches_base64(
    spec: LiveEmbeddingSpec,
) -> None:
    """Guard for #938, which switched the openai paths to `encoding_format="float"`.

    Requesting floats must return the same vectors the SDK's base64 default
    decoded to, so existing stored embeddings stay comparable.
    """
    client = make_embedding_client(spec)
    openai_client = cast(AsyncOpenAI, client.client)
    base64_kwargs: dict[str, Any] = {"model": spec.model, "input": [BATCH_TEXTS[0]]}
    if spec.send_dimensions:
        base64_kwargs["dimensions"] = spec.dimensions

    float_vector = await client.embed(BATCH_TEXTS[0])
    # No encoding_format → SDK sends base64 and decodes it, the pre-#938 path.
    base64_response = await openai_client.embeddings.create(**base64_kwargs)

    similarity = cosine_similarity(float_vector, base64_response.data[0].embedding)
    assert (
        similarity > 0.99999
    ), f"{spec.id}: float encoding diverges from base64 (cosine={similarity:.8f})"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.id)
async def test_live_batch_embed_truncates_oversize_instead_of_dropping_batch(
    spec: LiveEmbeddingSpec,
) -> None:
    """on_oversize='truncate' keeps one vector per input when an item exceeds the cap."""
    # Tiny cap so the oversize input stays cheap to tokenize and send.
    client = make_embedding_client(spec, max_input_tokens=32)
    oversize = " ".join(f"oversize-token-{index}" for index in range(200))
    assert len(client.encoding.encode(oversize)) > client.max_embedding_tokens

    texts = [BATCH_TEXTS[0], oversize, BATCH_TEXTS[1]]
    embeddings = await client.simple_batch_embed(texts, on_oversize="truncate")

    assert len(embeddings) == len(texts)
    assert all(len(embedding) == spec.dimensions for embedding in embeddings)
    # A collapsed or dropped batch would reuse a vector or return fewer.
    assert len({tuple(embedding) for embedding in embeddings}) == len(texts)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", GEMINI_SPECS, ids=lambda spec: spec.id)
async def test_live_gemini_batch_embed_survives_batch_split(
    spec: LiveEmbeddingSpec,
) -> None:
    """Same fix across the batch boundary: with max_batch_size=2 the four
    inputs go out as two separate Gemini requests."""
    client = make_embedding_client(spec)
    client.max_batch_size = 2

    embeddings = await client.simple_batch_embed(BATCH_TEXTS)

    assert len(embeddings) == len(BATCH_TEXTS)
    assert len({tuple(embedding) for embedding in embeddings}) == len(BATCH_TEXTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", representative_embedding_specs())
async def test_live_embedding_timeout_reaches_the_client(
    spec: LiveEmbeddingSpec,
) -> None:
    """Configured embedding timeout lands on the provider SDK client."""
    client = make_embedding_client(spec, timeout=GENEROUS_TIMEOUT_SECONDS)

    embedding = await client.embed(BATCH_TEXTS[0])

    assert len(embedding) == spec.dimensions
    assert_embedding_timeout_on_client(client, spec.transport, GENEROUS_TIMEOUT_SECONDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", representative_embedding_specs())
async def test_live_tight_embedding_timeout_aborts_request(
    spec: LiveEmbeddingSpec,
) -> None:
    """A near-zero embedding timeout aborts before the provider can answer."""
    client = make_embedding_client(spec, timeout=TIGHT_TIMEOUT_SECONDS)

    started = time.monotonic()
    with pytest.raises(EMBEDDING_TIMEOUT_EXCEPTIONS[spec.transport]):
        await client.embed(BATCH_TEXTS[0])
    elapsed = time.monotonic() - started

    assert elapsed < TIGHT_TIMEOUT_WALL_CLOCK_LIMIT_SECONDS, (
        f"tight embedding timeout took {elapsed:.1f}s — client timeout "
        f"likely not applied"
    )
