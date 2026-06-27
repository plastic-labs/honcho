"""CRUD tests for graph-memory endpoints.

Covers edges, contexts, thread bindings, pinning, verification, access-log
compaction, eviction, and rehydration.  These tests exercise the real FastAPI
endpoints against a per-test database and fakeredis cache.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models


def _create_edge(
    client: TestClient,
    workspace: models.Workspace,
    collection_name: str,
    source_id: str,
    target_id: str,
    edge_type: str = "related",
) -> dict[str, Any]:
    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/edges",
        json={
            "collection_name": collection_name,
            "source_obs_id": source_id,
            "target_obs_id": target_id,
            "edge_type": edge_type,
        },
    )
    assert resp.status_code == 201, f"create edge failed: {resp.status_code} {resp.text}"
    return resp.json()


@pytest.mark.asyncio
async def test_create_edge(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Creating an edge between two observations should succeed."""
    setup = graph_memory_setup
    docs = setup["docs_by_topic"]["llminal"]
    edge = _create_edge(
        client, setup["workspace"], setup["collection_name"], docs[0].id, docs[1].id
    )
    assert edge["source_obs_id"] == docs[0].id
    assert edge["target_obs_id"] == docs[1].id
    assert edge["edge_type"] == "related"


@pytest.mark.asyncio
async def test_edge_convergence_upsert_no_duplicates(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Creating the same edge twice should reinforce, not duplicate."""
    setup = graph_memory_setup
    docs = setup["docs_by_topic"]["llminal"]
    _create_edge(client, setup["workspace"], setup["collection_name"], docs[0].id, docs[1].id)
    _create_edge(client, setup["workspace"], setup["collection_name"], docs[0].id, docs[1].id)

    result = await db_session.execute(
        select(func.count()).select_from(models.Edge).where(
            models.Edge.workspace_name == setup["workspace"].name,
            models.Edge.source_obs_id == docs[0].id,
            models.Edge.target_obs_id == docs[1].id,
        )
    )
    assert result.scalar() == 1, "duplicate edges should be collapsed by convergence upsert"


@pytest.mark.asyncio
async def test_delete_edge(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Deleting an edge should remove it."""
    setup = graph_memory_setup
    docs = setup["docs_by_topic"]["llminal"]
    edge = _create_edge(client, setup["workspace"], setup["collection_name"], docs[0].id, docs[1].id)

    resp = client.delete(
        f"/v3/workspaces/{setup['workspace'].name}/graph-memory/edges/{edge['id']}"
    )
    assert resp.status_code == 204

    result = await db_session.execute(
        select(models.Edge).where(models.Edge.id == edge["id"])
    )
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_list_edges_with_filter(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Listing edges with a source filter should return only matching edges."""
    setup = graph_memory_setup
    docs = setup["docs_by_topic"]["llminal"]
    _create_edge(client, setup["workspace"], setup["collection_name"], docs[0].id, docs[1].id)
    _create_edge(client, setup["workspace"], setup["collection_name"], docs[2].id, docs[3].id)

    resp = client.post(
        f"/v3/workspaces/{setup['workspace'].name}/graph-memory/edges/list",
        json={"source_obs_id": docs[0].id},
    )
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["source_obs_id"] == docs[0].id


@pytest.mark.asyncio
async def test_context_member_lifecycle(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Create a context, add/remove members, and list members."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    ctx_name = "test-context"
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(f"/v3/workspaces/{workspace.name}/graph-memory/contexts", json={"context_name": ctx_name})
    assert resp.status_code == 201

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/contexts/{ctx_name}/members",
        json={"obs_id": obs_id},
    )
    assert resp.status_code == 201
    assert resp.json()["obs_id"] == obs_id

    resp = client.get(f"/v3/workspaces/{workspace.name}/graph-memory/contexts/{ctx_name}/members")
    assert resp.status_code == 200
    members = resp.json()
    assert any(m["obs_id"] == obs_id for m in members)

    resp = client.delete(
        f"/v3/workspaces/{workspace.name}/graph-memory/contexts/{ctx_name}/members/{obs_id}"
    )
    assert resp.status_code == 204

    resp = client.get(f"/v3/workspaces/{workspace.name}/graph-memory/contexts/{ctx_name}/members")
    assert resp.status_code == 200
    assert not any(m["obs_id"] == obs_id for m in resp.json())


@pytest.mark.asyncio
async def test_thread_binding_create_and_resolve(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Bind a thread to a context and resolve it back."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    ctx_name = "thread-test-context"
    thread_id = "123456789012345.67890"

    client.post(f"/v3/workspaces/{workspace.name}/graph-memory/contexts", json={"context_name": ctx_name})

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/thread-bindings",
        json={"thread_id": thread_id, "context_name": ctx_name},
    )
    assert resp.status_code == 201
    assert resp.json()["context_name"] == ctx_name

    resp = client.get(f"/v3/workspaces/{workspace.name}/graph-memory/thread-bindings/{thread_id}")
    assert resp.status_code == 200
    assert resp.json()["context_name"] == ctx_name


@pytest.mark.asyncio
async def test_thread_binding_rebind_denied(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Rebinding a thread to a different context should be rejected."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    thread_id = "123456789012346.67890"

    client.post(f"/v3/workspaces/{workspace.name}/graph-memory/contexts", json={"context_name": "ctx-a"})
    client.post(f"/v3/workspaces/{workspace.name}/graph-memory/contexts", json={"context_name": "ctx-b"})

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/thread-bindings",
        json={"thread_id": thread_id, "context_name": "ctx-a"},
    )
    assert resp.status_code == 201

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/thread-bindings",
        json={"thread_id": thread_id, "context_name": "ctx-b"},
    )
    assert resp.status_code == 422, f"expected 422, got {resp.status_code}"


@pytest.mark.asyncio
async def test_pin_and_unpin_observation(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Pinning and unpinning should update the document metadata."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/observations/{obs_id}/pin",
        json={"verify_cadence_days": 7},
    )
    assert resp.status_code == 200
    assert resp.json()["is_pinned"] is True

    result = await db_session.execute(select(models.Document).where(models.Document.id == obs_id))
    doc = result.scalar_one()
    assert doc.internal_metadata.get("is_pinned") is True
    assert doc.internal_metadata.get("verify_cadence_days") == 7

    resp = client.delete(
        f"/v3/workspaces/{workspace.name}/graph-memory/observations/{obs_id}/pin"
    )
    assert resp.status_code == 200
    assert resp.json()["is_pinned"] is False


@pytest.mark.asyncio
async def test_verify_observation(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Verifying an observation should append a verify event to the access log."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/observations/{obs_id}/verify"
    )
    assert resp.status_code == 200

    result = await db_session.execute(
        select(func.count()).select_from(models.AccessLogEntry).where(
            models.AccessLogEntry.workspace_name == workspace.name,
            models.AccessLogEntry.obs_id == obs_id,
            models.AccessLogEntry.event_type == "verify",
        )
    )
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_verify_due_returns_unverified_observations(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Observations that have never been verified should appear as verify-due."""
    setup = graph_memory_setup
    workspace = setup["workspace"]

    resp = client.get(f"/v3/workspaces/{workspace.name}/graph-memory/observations/verify-due")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= len(setup["all_docs"]), (
        "all unverified observations should be due for verification"
    )


@pytest.mark.asyncio
async def test_access_log_entry_creation_and_compaction(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
) -> None:
    """Create an access-log entry and compact it, receiving a report."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/access-log",
        json={
            "collection_name": setup["collection_name"],
            "obs_id": obs_id,
            "event_type": "access",
        },
    )
    assert resp.status_code == 201

    resp = client.post(f"/v3/workspaces/{workspace.name}/graph-memory/access-log/compact")
    assert resp.status_code == 200
    report = resp.json()
    for key in ("pruned_events", "retention_policy", "pre_compaction", "post_compaction", "health", "note"):
        assert key in report, f"report missing {key}"


@pytest.mark.asyncio
async def test_evict_stale_moves_to_cold_storage(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Eviction should move low-activation observations to documents_cold."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/evict-stale",
        params={"threshold": 0.12},
    )
    assert resp.status_code == 200, f"evict failed: {resp.status_code} {resp.text}"
    report = resp.json()
    assert "evicted_count" in report
    assert "skipped_pinned" in report
    assert "skipped_active" in report

    result = await db_session.execute(
        select(models.DocumentCold).where(
            models.DocumentCold.workspace_name == workspace.name,
            models.DocumentCold.id == obs_id,
        )
    )
    assert result.scalar_one_or_none() is not None, "evicted observation should exist in cold storage"


@pytest.mark.asyncio
async def test_rehydrate_cold_observation(
    client: TestClient,
    graph_memory_setup: dict[str, Any],
    db_session: AsyncSession,
) -> None:
    """Rehydrating a cold observation should restore it to active documents."""
    setup = graph_memory_setup
    workspace = setup["workspace"]
    obs_id = setup["docs_by_topic"]["llminal"][0].id

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/evict-stale",
        params={"threshold": 0.12},
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/v3/workspaces/{workspace.name}/graph-memory/rehydrate/{obs_id}"
    )
    assert resp.status_code == 200, f"rehydrate failed: {resp.status_code} {resp.text}"
    assert resp.json()["rehydrated"] is True

    result = await db_session.execute(
        select(models.Document).where(
            models.Document.workspace_name == workspace.name,
            models.Document.id == obs_id,
        )
    )
    assert result.scalar_one_or_none() is not None, "rehydrated observation should exist in active documents"
