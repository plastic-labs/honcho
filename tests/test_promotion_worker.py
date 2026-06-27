"""Tests for the graph-memory promotion worker.

These tests exercise ``process_promotion()`` against a real PostgreSQL database.
Embeddings are controlled deterministically so that semantic similarity is
observable without calling a live embedding provider.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.crud.graph_memory import list_edges
from src.deriver import promotion as promotion_mod
from src.deriver.promotion import (
    MAX_PROMOTION_ATTEMPTS,
    process_promotion,
)
from src.models import Document
from tests.fixtures.graph_memory_fixtures import (  # noqa: F401
    TOPIC_OBSERVATIONS,
    clean_graph_memory_queue_tables,
    controlled_embedding_client,
    force_promote,
    graph_memory_setup,
    topic_vector,
)


@pytest_asyncio.fixture(scope="function")
async def controlled_promotion_embedding_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[Any, None]:
    """Patch the promotion worker's embedding client with deterministic vectors."""

    class _MockClient:
        async def embed(self, text: str) -> list[float]:
            from tests.fixtures.graph_memory_fixtures import (
                TOPIC_OBSERVATIONS,
                query_vector_for_topic,
            )

            text_lower = text.lower()
            for topic, obs_list in TOPIC_OBSERVATIONS.items():
                for o in obs_list:
                    if o.lower() in text_lower or text_lower in o.lower():
                        return query_vector_for_topic(topic)
            return query_vector_for_topic("llminal")

        async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed(t) for t in texts]

    monkeypatch.setattr(
        "src.deriver.promotion.embedding_client",
        _MockClient(),
        raising=False,
    )
    yield _MockClient()


async def _edges_for_obs(
    db_session: AsyncSession,
    workspace_name: str,
    obs_id: str,
) -> list[models.Edge]:
    result = await db_session.execute(
        select(models.Edge).where(
            models.Edge.workspace_name == workspace_name,
            models.Edge.source_obs_id == obs_id,
        )
    )
    return list(result.scalars().all())


async def _load_doc(
    db_session: AsyncSession,
    workspace_name: str,
    obs_id: str,
) -> Document | None:
    result = await db_session.execute(
        select(Document).where(
            Document.workspace_name == workspace_name,
            Document.id == obs_id,
        )
    )
    doc = result.scalar_one_or_none()
    if doc is not None:
        await db_session.refresh(doc)
    return doc


@pytest.mark.asyncio
async def test_process_promotion_creates_semantic_edges_not_temporal(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
) -> None:
    """Edges link same-topic observations; cross-topic observations stay unconnected."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]

    target_doc = setup["docs_by_topic"]["llminal"][0]
    same_topic_ids = {d.id for d in setup["docs_by_topic"]["llminal"] if d.id != target_doc.id}
    cross_topic_ids = {
        d.id
        for topic, docs in setup["docs_by_topic"].items()
        if topic != "llminal"
        for d in docs
    }

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=target_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    edges = await _edges_for_obs(db_session, workspace.name, target_doc.id)
    target_ids = {e.target_obs_id for e in edges}

    assert same_topic_ids.issubset(target_ids), (
        f"Expected edges to all same-topic observations; missing "
        f"{same_topic_ids - target_ids}, got {target_ids}"
    )
    assert not (cross_topic_ids & target_ids), (
        f"Expected no cross-topic edges, but found {cross_topic_ids & target_ids}"
    )


@pytest.mark.asyncio
async def test_process_promotion_embedding_failure_isolated(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
    controlled_promotion_embedding_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One observation with a broken embedding does not block another."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]

    healthy_doc = setup["docs_by_topic"]["honcho"][0]

    # Create a sick observation without a stored embedding.
    sick_doc = models.Document(
        workspace_name=workspace.name,
        observer=observer_name,
        observed=observed_name,
        content="This observation has no embedding and will fail to vectorize.",
        level="explicit",
        times_derived=1,
        embedding=None,
        session_name=setup["session"].name,
    )
    db_session.add(sick_doc)
    await db_session.commit()
    await db_session.refresh(sick_doc)

    # Make embedding fail only for the sick observation's content.
    real_embed = controlled_promotion_embedding_client.embed

    async def _raising_embed(text: str) -> list[float]:
        if text == sick_doc.content:
            raise RuntimeError("embedding provider unavailable")
        return await real_embed(text)

    monkeypatch.setattr(
        "src.deriver.promotion.embedding_client.embed",
        _raising_embed,
    )

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=sick_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )
    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=healthy_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    refreshed_sick = await _load_doc(db_session, workspace.name, sick_doc.id)
    assert refreshed_sick is not None
    await db_session.refresh(refreshed_sick)
    assert refreshed_sick.promotion_attempts == 1
    assert refreshed_sick.promotion_failed is False

    healthy_edges = await _edges_for_obs(db_session, workspace.name, healthy_doc.id)
    assert len(healthy_edges) > 0, "Healthy observation should still be promoted"


@pytest.mark.asyncio
async def test_process_promotion_marks_failed_after_max_attempts(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After MAX_PROMOTION_ATTEMPTS failures the observation is permanently skipped."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]

    sick_doc = models.Document(
        workspace_name=workspace.name,
        observer=observer_name,
        observed=observed_name,
        content="This observation will repeatedly fail embedding.",
        level="explicit",
        times_derived=1,
        embedding=None,
        session_name=setup["session"].name,
    )
    db_session.add(sick_doc)
    await db_session.commit()
    await db_session.refresh(sick_doc)

    monkeypatch.setattr(
        "src.deriver.promotion.embedding_client.embed",
        AsyncMock(side_effect=RuntimeError("embedding provider unavailable")),
    )

    for i in range(MAX_PROMOTION_ATTEMPTS):
        await process_promotion(
            workspace_name=workspace.name,
            collection_name=collection_name,
            obs_id=sick_doc.id,
            observer=observer_name,
            observed=observed_name,
            session_name=setup["session"].name,
        )
        refreshed = await _load_doc(db_session, workspace.name, sick_doc.id)
        assert refreshed is not None
        assert refreshed.promotion_attempts == i + 1
        if i < MAX_PROMOTION_ATTEMPTS - 1:
            assert refreshed.promotion_failed is False
        else:
            assert refreshed.promotion_failed is True
            assert refreshed.promotion_error is not None
            assert "RuntimeError" in refreshed.promotion_error


@pytest.mark.asyncio
async def test_process_promotion_uses_chunking_for_oversized_observations(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
    controlled_promotion_embedding_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized observation is chunked and averaged, not silently truncated."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]

    # Sentence is only ~15 words; with a tiny per-observation token budget it
    # must be split into multiple chunks.
    content = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron."
    oversized_doc = models.Document(
        workspace_name=workspace.name,
        observer=observer_name,
        observed=observed_name,
        content=content,
        level="explicit",
        times_derived=1,
        embedding=None,
        session_name=setup["session"].name,
    )
    db_session.add(oversized_doc)
    await db_session.commit()
    await db_session.refresh(oversized_doc)

    # Force chunking by lowering the per-observation token budget.
    monkeypatch.setattr(
        promotion_mod,
        "MAX_TOKENS_PER_OBSERVATION_EMBEDDING",
        3,
    )

    batch_inputs: list[list[str]] = []
    real_batch_embed = controlled_promotion_embedding_client.simple_batch_embed

    async def _recording_batch_embed(texts: list[str]) -> list[list[float]]:
        batch_inputs.append(texts)
        return await real_batch_embed(texts)

    monkeypatch.setattr(
        "src.deriver.promotion.embedding_client.simple_batch_embed",
        _recording_batch_embed,
    )

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=oversized_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    assert batch_inputs, "simple_batch_embed should have been called for chunked observation"
    chunks = batch_inputs[0]
    assert len(chunks) > 1, f"Expected multiple chunks, got {chunks}"
    # No chunk should contain the full sentence, proving truncation did not occur.
    full_sentence = content.replace(".", "")
    assert all(full_sentence != chunk.replace(".", "") for chunk in chunks)

    # Edges should still be created using the averaged chunk representation.
    edges = await _edges_for_obs(db_session, workspace.name, oversized_doc.id)
    assert len(edges) > 0, "Chunked observation should still form promotion edges"


@pytest.mark.asyncio
async def test_process_promotion_chunking_creates_edges_to_multiple_topic_clusters(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
    controlled_promotion_embedding_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-intent oversized observation forms edges through each chunk."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]

    # Two sentences from two different topic clusters.  With a low per-observation
    # token budget each sentence becomes its own chunk, and each chunk embedding
    # points to a different topic cluster.
    content = (
        "We decided the LLMinal protocol uses L1 encoding for clarity-critical messages. "
        "The Honcho graph memory backend builds edges between semantically related observations."
    )
    multi_topic_doc = models.Document(
        workspace_name=workspace.name,
        observer=observer_name,
        observed=observed_name,
        content=content,
        level="explicit",
        times_derived=1,
        embedding=None,
        session_name=setup["session"].name,
    )
    db_session.add(multi_topic_doc)
    await db_session.commit()
    await db_session.refresh(multi_topic_doc)

    # Force sentence-level chunking by giving just enough budget for one sentence
    # but not both.
    monkeypatch.setattr(
        promotion_mod,
        "MAX_TOKENS_PER_OBSERVATION_EMBEDDING",
        12,
    )

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=multi_topic_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    edges = await _edges_for_obs(db_session, workspace.name, multi_topic_doc.id)
    target_ids = {e.target_obs_id for e in edges}

    llminal_ids = {d.id for d in setup["docs_by_topic"]["llminal"]}
    honcho_ids = {d.id for d in setup["docs_by_topic"]["honcho"]}

    assert target_ids & llminal_ids, (
        f"Expected edges to LLMinal-topic observations, got targets {target_ids}"
    )
    assert target_ids & honcho_ids, (
        f"Expected edges to Honcho-topic observations, got targets {target_ids}"
    )


@pytest.mark.asyncio
async def test_process_promotion_edges_use_correct_collection_keys(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
) -> None:
    """Created edges carry the observer/observed collection name."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]
    target_doc = setup["docs_by_topic"]["agentc_process"][0]

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=target_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    edges = await list_edges(
        db=db_session,
        workspace_name=workspace.name,
        source_obs_id=target_doc.id,
    )
    assert len(edges) > 0
    for edge in edges:
        assert edge.collection_name == collection_name, (
            f"Edge collection_name {edge.collection_name!r} != {collection_name!r}"
        )


@pytest.mark.asyncio
async def test_process_promotion_edge_weight_reflects_similarity(
    db_session: AsyncSession,
    graph_memory_setup: dict[str, Any],
    force_promote: None,
) -> None:
    """Edge weights are derived from cosine similarity, not a constant."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    observer_name = setup["observer"].name
    observed_name = setup["observed"].name
    collection_name = setup["collection_name"]
    target_doc = setup["docs_by_topic"]["user_profile"][0]

    await process_promotion(
        workspace_name=workspace.name,
        collection_name=collection_name,
        obs_id=target_doc.id,
        observer=observer_name,
        observed=observed_name,
        session_name=setup["session"].name,
    )

    edges = await _edges_for_obs(db_session, workspace.name, target_doc.id)
    assert len(edges) > 0

    weights = [float(e.edge_metadata["weight"]) for e in edges if "weight" in e.edge_metadata]
    assert weights, "Every edge should carry a weight"
    assert all(0.0 < w <= 1.0 for w in weights), (
        f"Weights should be in (0, 1], got {weights}"
    )

    # Same-topic vectors are nearly identical, so similarity should be very high.
    assert all(w > 0.95 for w in weights), (
        f"Same-topic edges should have high similarity weights, got {weights}"
    )

    # Weights should not all be identical constants.
    assert len(set(weights)) > 1, f"Weights should vary with distance, got {weights}"
