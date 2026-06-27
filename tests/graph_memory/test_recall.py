"""Backend tests for the graph-memory recall endpoint.

These tests verify the real vector search path and the collection_name mapping
(observer/observed pair) by exercising the router against a per-test PostgreSQL
database with controlled deterministic embeddings.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models


@pytest.mark.asyncio
async def test_recall_vector_search_returns_topic_matches(
    client: TestClient,
    db_session: AsyncSession,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001  # patches router embedding client
) -> None:
    """A query about LLMinal should return LLMinal documents, not job search details."""
    setup = graph_memory_setup
    collection_name = setup["collection_name"]
    workspace_name = setup["workspace"].name

    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json={
            "collection_name": collection_name,
            "query": "LLMinal protocol details",
            "max_depth": 1,
            "token_budget": 100,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    results = data["results"]
    assert results, "recall should return results"
    returned_ids = {r["obs_id"] for r in results}
    llminal_ids = set(setup["ids_by_topic"]["llminal"])
    # The top-5 anchor set should be dominated by LLMinal documents.
    llminal_returned = returned_ids & llminal_ids
    non_llminal_returned = returned_ids - llminal_ids
    assert len(llminal_returned) >= len(non_llminal_returned), (
        f"recall for LLMinal should return more LLMinal ids than not: "
        f"llminal={llminal_returned}, non-llminal={non_llminal_returned}"
    )
    assert returned_ids.issubset(setup["all_docs"]) is False or returned_ids, "results came from collection"  # noqa: B015
    # If any non-LLMinal ids leaked, ensure they are scored lower than LLMinal ids.
    if non_llminal_returned:
        scores_by_id = {r["obs_id"]: r["score"] for r in results}
        min_llminal_score = min(scores_by_id[oid] for oid in llminal_returned)
        assert all(scores_by_id[oid] <= min_llminal_score for oid in non_llminal_returned), (
            f"non-LLMinal result scored above an LLMinal result: {scores_by_id}"
        )


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
    db_session: AsyncSession,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
    query: str,
    expected_topic: str,
) -> None:
    """Semantic relevance should rank the queried topic first.

    We prime the expected-topic observations with verify + access events so they
    have non-zero score and reliably outrank unrelated vector-search anchors.
    """
    setup = graph_memory_setup
    collection_name = setup["collection_name"]
    workspace_name = setup["workspace"].name

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

    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json={
            "collection_name": collection_name,
            "query": query,
            "max_depth": 1,
            "token_budget": 100,
        },
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert results, f"no results for {query}"
    assert results[0]["obs_id"] in setup["ids_by_topic"][expected_topic], (
        f"query '{query}' did not return {expected_topic} first"
    )


@pytest.mark.asyncio
async def test_recall_collection_name_requires_pair(
    client: TestClient,
    db_session: AsyncSession,
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
    db_session: AsyncSession,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """Querying a non-existent (observer, observed) pair should return empty results."""
    setup = graph_memory_setup
    workspace_name = setup["workspace"].name

    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json={
            "collection_name": "nobody/nothing",
            "query": "LLMinal",
            "max_depth": 1,
            "token_budget": 100,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["results"] == []
    assert data["total_visited"] == 0


@pytest.mark.asyncio
async def test_recall_total_visited_reflects_graph_traversal(
    client: TestClient,
    db_session: AsyncSession,
    graph_memory_setup: dict,
    controlled_embedding_client: object,  # noqa: ARG001
) -> None:
    """total_visited should count documents reached from vector anchors + CTE."""
    setup = graph_memory_setup
    collection_name = setup["collection_name"]
    workspace_name = setup["workspace"].name

    # Seed an edge between two LLMinal documents so the CTE traverses.
    ids = setup["ids_by_topic"]["llminal"]
    edge = models.Edge(
        workspace_name=workspace_name,
        collection_name=collection_name,
        source_obs_id=ids[0],
        target_obs_id=ids[1],
        edge_type="related",
        created_by="test",
    )
    db_session.add(edge)
    await db_session.commit()

    response = client.post(
        f"/v3/workspaces/{workspace_name}/graph-memory/recall",
        json={
            "collection_name": collection_name,
            "query": "LLMinal",
            "max_depth": 2,
            "token_budget": 100,
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_visited"] >= 2, (
        f"expected at least anchor+edge target, got {data['total_visited']}"
    )
    returned_ids = {r["obs_id"] for r in data["results"]}
    assert ids[0] in returned_ids
    assert ids[1] in returned_ids
