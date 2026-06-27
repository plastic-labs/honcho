"""Recall endpoint tests for graph memory.

These tests exercise the real vector search path and the collection_name
mapping (observer/observed pair) by calling the FastAPI router against a
per-test PostgreSQL database with deterministic topic-clustered embeddings.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models


def _recall(
    client: TestClient,
    workspace_name: str,
    collection_name: str,
    query: str,
    *,
    max_depth: int = 1,
    token_budget: int = 100,
    context: str | None = None,
) -> dict:
    payload: dict = {
        "collection_name": collection_name,
        "query": query,
        "max_depth": max_depth,
        "token_budget": token_budget,
    }
    if context:
        payload["context"] = context
    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json=payload,
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_recall_vector_search_returns_topic_matches(
    client: TestClient,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """A query about LLMinal should return mostly LLMinal documents."""
    setup = graph_memory_setup
    data = _recall(client, setup["workspace"].name, setup["collection_name"], "LLMinal protocol details")
    results = data["results"]
    assert results, "recall should return results"

    returned_ids = {r["obs_id"] for r in results}
    llminal_ids = set(setup["ids_by_topic"]["llminal"])
    llminal_returned = returned_ids & llminal_ids
    non_llminal_returned = returned_ids - llminal_ids
    assert len(llminal_returned) >= len(non_llminal_returned), (
        f"recall for LLMinal should return more LLMinal ids than not: "
        f"llminal={llminal_returned}, non-llminal={non_llminal_returned}"
    )


@pytest.mark.asyncio
async def test_recall_results_ranked_by_semantic_relevance(
    client: TestClient,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Results should be ordered by descending score (semantic relevance)."""
    setup = graph_memory_setup
    data = _recall(client, setup["workspace"].name, setup["collection_name"], "LLMinal protocol")
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True), f"scores not sorted descending: {scores}"


@pytest.mark.parametrize(
    "query,expected_topic",
    [
        ("LLMinal protocol", "llminal"),
        ("Honcho graph memory", "honcho"),
        ("user job search preferences", "user_profile"),
        ("AgentC adversarial review process", "agentc_process"),
    ],
)
@pytest.mark.asyncio
async def test_recall_different_queries_return_different_rankings(
    client: TestClient,
    graph_memory_setup: dict,
    db_session: AsyncSession,
    controlled_embedding_client: object,  # noqa: ARG001
    query: str,
    expected_topic: str,
) -> None:
    """A topic query should rank observations from that topic first.

    We prime the expected-topic observations with a verify and access event so
    they have non-zero score and reliably outrank unrelated observations that are
    also vector-search anchors.
    """
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name
    collection_name = setup["collection_name"]

    for obs_id in setup["ids_by_topic"][expected_topic]:
        db_session.add(
            models.AccessLogEntry(
                workspace_name=workspace_name,
                collection_name=collection_name,
                obs_id=obs_id,
                event_type="verify",
                created_by="test",
            )
        )
        db_session.add(
            models.AccessLogEntry(
                workspace_name=workspace_name,
                collection_name=collection_name,
                obs_id=obs_id,
                event_type="access",
                created_by="test",
            )
        )
    await db_session.commit()

    data = _recall(client, workspace_name, collection_name, query, token_budget=100)
    assert data["results"], f"no results for {query}"
    first_id = data["results"][0]["obs_id"]
    assert first_id in setup["ids_by_topic"][expected_topic], (
        f"query '{query}' did not return {expected_topic} first (got {first_id})"
    )


@pytest.mark.asyncio
async def test_recall_total_visited_exceeds_anchors_with_edges(
    client: TestClient,
    graph_memory_setup: dict,
    db_session: AsyncSession,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """total_visited should be > 5 when graph traversal reaches edge-connected nodes."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name
    collection_name = setup["collection_name"]
    ids = setup["ids_by_topic"]["llminal"]

    # Build a small chain of edges inside the LLMinal topic.
    for source_id, target_id in zip(ids, ids[1:]):
        db_session.add(
            models.Edge(
                workspace_name=workspace_name,
                collection_name=collection_name,
                source_obs_id=source_id,
                target_obs_id=target_id,
                edge_type="related",
                created_by="test",
            )
        )
    await db_session.commit()

    data = _recall(client, workspace_name, collection_name, "LLMinal", max_depth=2, token_budget=100)
    assert data["total_visited"] > 5, (
        f"expected graph traversal to visit more than 5 nodes, got {data['total_visited']}"
    )


@pytest.mark.asyncio
async def test_recall_confidence_positive_for_connected_observations(
    client: TestClient,
    graph_memory_setup: dict,
    db_session: AsyncSession,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Verifying a connected observation should yield confidence > 0.0 in recall."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name
    collection_name = setup["collection_name"]
    source_id = setup["ids_by_topic"]["llminal"][0]
    target_id = setup["ids_by_topic"]["llminal"][1]

    # Connect the two observations with an edge.
    db_session.add(
        models.Edge(
            workspace_name=workspace_name,
            collection_name=collection_name,
            source_obs_id=source_id,
            target_obs_id=target_id,
            edge_type="related",
            created_by="test",
        )
    )
    # Verify the target so confidence is non-zero.
    db_session.add(
        models.AccessLogEntry(
            workspace_name=workspace_name,
            collection_name=collection_name,
            obs_id=target_id,
            event_type="verify",
            created_by="test",
        )
    )
    await db_session.commit()

    data = _recall(client, workspace_name, collection_name, "LLMinal", max_depth=2, token_budget=100)
    target_result = next((r for r in data["results"] if r["obs_id"] == target_id), None)
    assert target_result is not None, "target observation was not returned by recall"
    assert target_result["confidence"] > 0.0, (
        f"expected confidence > 0.0 for verified connected observation, got {target_result['confidence']}"
    )


@pytest.mark.asyncio
async def test_recall_context_filter_returns_only_context_members(
    client: TestClient,
    graph_memory_setup: dict,
    db_session: AsyncSession,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Recall with a context filter should return only observations in that context."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name
    collection_name = setup["collection_name"]
    ctx_name = "workstream-llminal"

    # Add exactly one LLMinal observation to the context.
    member_id = setup["ids_by_topic"]["llminal"][0]
    db_session.add(
        models.ContextIndex(
            workspace_name=workspace_name,
            context_name=ctx_name,
            obs_id=member_id,
            added_by="test",
        )
    )
    await db_session.commit()

    data = _recall(
        client,
        workspace_name,
        collection_name,
        "LLMinal",
        max_depth=1,
        token_budget=100,
        context=ctx_name,
    )
    returned_ids = {r["obs_id"] for r in data["results"]}
    assert returned_ids == {member_id}, (
        f"context-scoped recall should return only the member, got {returned_ids}"
    )


@pytest.mark.asyncio
async def test_recall_no_edges_graceful_degradation(
    client: TestClient,
    graph_memory_setup: dict,
    db_session: AsyncSession,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """With no edges, recall should still return vector-search anchors."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name
    collection_name = setup["collection_name"]

    # Ensure no edges exist for this workspace.
    result = await db_session.execute(
        select(func.count()).select_from(models.Edge).where(
            models.Edge.workspace_name == workspace_name
        )
    )
    assert result.scalar() == 0, "precondition: workspace should have no edges"

    data = _recall(client, workspace_name, collection_name, "LLMinal protocol")
    assert data["results"], "recall should fall back to vector-search-only anchors"
    assert all(r["score"] >= 0.0 for r in data["results"]), "scores should be non-negative"


@pytest.mark.asyncio
async def test_recall_empty_workspace_returns_empty(
    client: TestClient,
    sample_data: tuple[models.Workspace, models.Peer],
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Recall against a workspace with no documents should be clean and empty."""
    workspace, peer_a = sample_data
    peer_b = models.Peer(name="empty-peer", workspace_name=workspace.name)
    # Note: db_session will be rolled back/truncated after the test.

    data = _recall(client, workspace.name, f"{peer_a.name}/empty-peer", "anything")
    assert data["results"] == []
    assert data["total_visited"] == 0


@pytest.mark.asyncio
async def test_recall_collection_name_requires_pair(
    client: TestClient,
    graph_memory_setup: dict,
) -> None:
    """A collection_name that is not 'observer/observed' should be rejected."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name

    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json={
            "collection_name": "not-a-pair",
            "query": "anything",
            "max_depth": 1,
            "token_budget": 100,
        },
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_recall_no_results_for_missing_collection(
    client: TestClient,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Querying a non-existent (observer, observed) pair should return empty results."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name

    data = _recall(client, workspace_name, "nobody/nothing", "LLMinal")
    assert data["results"] == []
    assert data["total_visited"] == 0
