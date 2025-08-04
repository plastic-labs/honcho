import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src import models


@pytest.mark.asyncio
class TestDeriverStatusEndpoint:
    """Test suite for the /deriver/status endpoint"""

    async def test_get_deriver_status_peer_only(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status filtered by peer only"""
        workspace, peer = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_session_only(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status filtered by session only"""
        workspace, _peer = sample_data
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"session_id": session.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_peer_and_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status filtered by both peer and session"""
        workspace, peer = sample_data
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name, "session_id": session.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_with_include_sender_true(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status with include_sender=True"""
        workspace, peer = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name, "sender_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_with_include_sender_false(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status with include_sender=False (default)"""
        workspace, peer = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_no_parameters(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status without required parameters returns 200"""
        workspace, _ = sample_data
        response = client.get(f"/v2/workspaces/{workspace.name}/deriver/status")
        assert response.status_code == 200

    async def test_get_deriver_status_nonexistent_peer(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status for nonexistent peer returns empty result"""
        workspace, _ = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": "nonexistent"},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0
        assert response.json()["completed_work_units"] == 0
        assert response.json()["in_progress_work_units"] == 0
        assert response.json()["pending_work_units"] == 0

    async def test_get_deriver_status_nonexistent_session(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status for nonexistent session returns empty result"""
        workspace, _ = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"session_id": "nonexistent"},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0
        assert response.json()["completed_work_units"] == 0
        assert response.json()["in_progress_work_units"] == 0
        assert response.json()["pending_work_units"] == 0

    async def test_get_deriver_status_nonexistent_workspace(self, client: TestClient):
        """Test getting deriver status for nonexistent workspace returns empty result"""
        response = client.get("/v2/workspaces/nonexistent/deriver/status")
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_with_queue_items(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status when there are actual queue items"""
        workspace, peer = sample_data
        session = models.Session(workspace_name=workspace.name, name="test_session")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        # Add queue items
        queue_items = [
            models.QueueItem(
                session_id=session.id,
                task_type="derive",
                payload={
                    "sender_name": peer.name,
                    "target_name": peer.name,
                    "task_type": "derive",
                },
                processed=False,
            )
            for _ in range(5)
        ]
        db_session.add_all(queue_items)
        await db_session.commit()
        # Test without parameters
        response = client.get(f"/v2/workspaces/{workspace.name}/deriver/status")
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 5
        assert response.json()["pending_work_units"] == 5
        # Test with observer_id
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 5
        assert response.json()["pending_work_units"] == 5
        # Test with sender_id (new capability)
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"sender_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 5
        assert response.json()["pending_work_units"] == 5
        # Test with both (OR filter)
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name, "sender_id": peer.name},
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 5
        assert response.json()["pending_work_units"] == 5
        # Test with different observer and sender (should be ok)
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name, "sender_id": "different"},
        )
        assert response.status_code == 200

    async def test_get_deriver_status_with_sessions_breakdown(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status shows sessions breakdown when appropriate"""
        workspace, peer = sample_data
        # Create multiple sessions
        sessions = [
            models.Session(workspace_name=workspace.name, name=f"session_{i}")
            for i in range(3)
        ]
        db_session.add_all(sessions)
        await db_session.commit()
        for s in sessions:
            await db_session.refresh(s)
        # Add queue items to different sessions
        for i, session in enumerate(sessions):
            queue_items = [
                models.QueueItem(
                    session_id=session.id,
                    task_type="derive",
                    payload={
                        "sender_name": peer.name,
                        "target_name": peer.name,
                        "task_type": "derive",
                    },
                    processed=False,
                )
                for _ in range(i + 1)  # 1,2,3 items respectively
            ]
            db_session.add_all(queue_items)
        await db_session.commit()
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={"observer_id": peer.name},
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["total_work_units"] == 6
        assert json_response["pending_work_units"] == 6
        assert "sessions" in json_response
        assert len(json_response["sessions"]) == 3
        # Check per-session counts (session names are not returned, but we can check totals)
        session_totals = sorted(
            [s["total_work_units"] for s in json_response["sessions"].values()]
        )
        assert session_totals == [1, 2, 3]

    async def test_get_deriver_status_empty_parameters(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test various edge cases with empty or invalid parameters"""
        workspace, _ = sample_data
        response = client.get(
            f"/v2/workspaces/{workspace.name}/deriver/status",
            params={
                "observer_id": "",
                "session_id": "",
            },
        )
        assert response.status_code == 200
        assert response.json()["total_work_units"] == 0

    async def test_get_deriver_status_response_consistency(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that response structure is consistent across different parameter combinations"""
        workspace, peer = sample_data
        # Add some queue items
        session = models.Session(workspace_name=workspace.name, name="test")
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)
        queue_item = models.QueueItem(
            session_id=session.id,
            task_type="derive",
            payload={
                "sender_name": peer.name,
                "target_name": peer.name,
                "task_type": "derive",
            },
            processed=False,
        )
        db_session.add(queue_item)
        await db_session.commit()
        # Get status multiple times
        responses = []
        for _ in range(3):
            response = client.get(
                f"/v2/workspaces/{workspace.name}/deriver/status",
                params={"observer_id": peer.name},
            )
            assert response.status_code == 200
            responses.append(response.json())  # pyright: ignore
        # Check consistency
        assert all(r == responses[0] for r in responses)  # pyright: ignore
