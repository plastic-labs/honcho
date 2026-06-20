"""Tests for the queue purge endpoint (DELETE /v3/{workspace_id}/queue)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


def _make_queue_item(
    workspace_name: str,
    *,
    session_id: str | None = None,
    processed: bool = False,
) -> models.QueueItem:
    """Helper to construct an unsaved QueueItem with the minimum required fields."""
    return models.QueueItem(
        workspace_name=workspace_name,
        session_id=session_id,
        work_unit_key=f"test:wu:{session_id or 'ws'}:{processed}",
        task_type="representation",
        payload={"task_type": "representation"},
        processed=processed,
    )


class TestQueuePurge:
    """Test suite for DELETE /v3/{workspace_id}/queue"""

    @pytest.mark.asyncio
    async def test_purge_empty_workspace_returns_zero(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """No matching items → purged_count=0, message says so."""
        workspace, _ = sample_data
        response = client.delete(f"/v3/workspaces/{workspace.name}/queue")
        assert response.status_code == 200
        body = response.json()
        assert body["purged_count"] == 0
        assert "No stranded" in body["message"]

    @pytest.mark.asyncio
    async def test_purge_unprocessed_only(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """status='unprocessed' (default) deletes only unprocessed rows."""
        workspace, _ = sample_data
        # 2 unprocessed, 1 processed
        for _ in range(2):
            db_session.add(_make_queue_item(workspace.name, processed=False))
        db_session.add(_make_queue_item(workspace.name, processed=True))
        await db_session.commit()

        response = client.delete(f"/v3/workspaces/{workspace.name}/queue")
        assert response.status_code == 200
        assert response.json()["purged_count"] == 2

        # 1 processed row remains
        remaining = (
            await db_session.execute(
                models.QueueItem.__table__.select().where(
                    models.QueueItem.workspace_name == workspace.name
                )
            )
        ).fetchall()
        assert len(remaining) == 1
        assert remaining[0].processed is True

    @pytest.mark.asyncio
    async def test_purge_all_includes_processed(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """status='all' purges both processed and unprocessed rows."""
        workspace, _ = sample_data
        for _ in range(3):
            db_session.add(_make_queue_item(workspace.name, processed=False))
        for _ in range(2):
            db_session.add(_make_queue_item(workspace.name, processed=True))
        await db_session.commit()

        response = client.delete(
            f"/v3/workspaces/{workspace.name}/queue",
            params={"status": "all"},
        )
        assert response.status_code == 200
        assert response.json()["purged_count"] == 5

        remaining = (
            await db_session.execute(
                models.QueueItem.__table__.select().where(
                    models.QueueItem.workspace_name == workspace.name
                )
            )
        ).fetchall()
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_purge_filtered_by_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """session_id query param narrows the purge to one session only."""
        workspace, _ = sample_data
        sess_a = models.Session(workspace_name=workspace.name, name="sess_a")
        sess_b = models.Session(workspace_name=workspace.name, name="sess_b")
        db_session.add_all([sess_a, sess_b])
        await db_session.flush()

        for _ in range(2):
            db_session.add(_make_queue_item(workspace.name, session_id="sess_a"))
        for _ in range(3):
            db_session.add(_make_queue_item(workspace.name, session_id="sess_b"))
        await db_session.commit()

        response = client.delete(
            f"/v3/workspaces/{workspace.name}/queue",
            params={"session_id": "sess_a"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["purged_count"] == 2
        assert body["session_id"] == "sess_a"

        # sess_b rows still present
        remaining = (
            await db_session.execute(
                models.QueueItem.__table__.select().where(
                    models.QueueItem.workspace_name == workspace.name
                )
            )
        ).fetchall()
        assert len(remaining) == 3
        assert all(r.session_id == "sess_b" for r in remaining)

    @pytest.mark.asyncio
    async def test_purge_only_targets_workspace(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ) -> None:
        """Cross-workspace isolation: a purge on workspace A must not touch workspace B."""
        ws_a, _ = sample_data
        ws_b = models.Workspace(name="other-ws")
        db_session.add(ws_b)
        await db_session.flush()

        for _ in range(2):
            db_session.add(_make_queue_item(ws_a.name, processed=False))
        for _ in range(3):
            db_session.add(_make_queue_item(ws_b.name, processed=False))
        await db_session.commit()

        response = client.delete(f"/v3/workspaces/{ws_a.name}/queue")
        assert response.status_code == 200
        assert response.json()["purged_count"] == 2

        # ws_b rows still present
        remaining = (
            await db_session.execute(
                models.QueueItem.__table__.select().where(
                    models.QueueItem.workspace_name == ws_b.name
                )
            )
        ).fetchall()
        assert len(remaining) == 3
