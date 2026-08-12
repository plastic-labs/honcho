from __future__ import annotations

import pytest

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
