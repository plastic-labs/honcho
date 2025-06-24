import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
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
        test_workspace, test_peer = sample_data

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}"
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure matches DeriverStatus schema
        assert "peer_id" in data
        assert "total_work_units" in data
        assert "completed_work_units" in data
        assert "in_progress_work_units" in data
        assert "pending_work_units" in data
        assert data["peer_id"] == test_peer.name
        assert isinstance(data["total_work_units"], int)
        assert isinstance(data["completed_work_units"], int)
        assert isinstance(data["in_progress_work_units"], int)
        assert isinstance(data["pending_work_units"], int)

    async def test_get_deriver_status_session_only(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status filtered by session only"""
        test_workspace, _ = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.commit()

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?session_id={test_session.name}"
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "session_id" in data
        assert "total_work_units" in data
        assert "completed_work_units" in data
        assert "in_progress_work_units" in data
        assert "pending_work_units" in data
        assert data["session_id"] == test_session.name
        assert isinstance(data["total_work_units"], int)

    async def test_get_deriver_status_peer_and_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status filtered by both peer and session"""
        test_workspace, test_peer = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.commit()

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&session_id={test_session.name}"
        )
        assert response.status_code == 200
        data = response.json()

        # Should have both peer_id and session_id in response
        assert data["peer_id"] == test_peer.name
        assert data["session_id"] == test_session.name
        assert "total_work_units" in data
        assert "completed_work_units" in data
        assert "in_progress_work_units" in data
        assert "pending_work_units" in data

    async def test_get_deriver_status_with_include_sender_true(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status with include_sender=True"""
        test_workspace, test_peer = sample_data

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=true"
        )
        assert response.status_code == 200
        data = response.json()

        assert data["peer_id"] == test_peer.name
        assert "total_work_units" in data

    async def test_get_deriver_status_with_include_sender_false(
        self,
        client: TestClient,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status with include_sender=False (default)"""
        test_workspace, test_peer = sample_data

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=false"
        )
        assert response.status_code == 200
        data = response.json()

        assert data["peer_id"] == test_peer.name
        assert "total_work_units" in data

    async def test_get_deriver_status_no_parameters(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status without required parameters returns 400"""
        test_workspace, _ = sample_data

        response = client.get(f"/v2/workspaces/{test_workspace.name}/deriver/status")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert (
            "At least one of 'peer_id' or 'session_id' must be provided"
            in data["detail"]
        )

    async def test_get_deriver_status_nonexistent_peer(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status for nonexistent peer returns empty result"""
        test_workspace, _ = sample_data
        nonexistent_peer = str(generate_nanoid())

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={nonexistent_peer}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["peer_id"] == nonexistent_peer
        assert data["total_work_units"] == 0
        assert data["completed_work_units"] == 0
        assert data["in_progress_work_units"] == 0
        assert data["pending_work_units"] == 0

    async def test_get_deriver_status_nonexistent_session(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test getting deriver status for nonexistent session returns empty result"""
        test_workspace, _ = sample_data
        nonexistent_session = str(generate_nanoid())

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?session_id={nonexistent_session}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == nonexistent_session
        assert data["total_work_units"] == 0
        assert data["completed_work_units"] == 0
        assert data["in_progress_work_units"] == 0
        assert data["pending_work_units"] == 0

    async def test_get_deriver_status_nonexistent_workspace(self, client: TestClient):
        """Test getting deriver status for nonexistent workspace returns empty result"""
        nonexistent_workspace = str(generate_nanoid())
        fake_peer = str(generate_nanoid())

        response = client.get(
            f"/v2/workspaces/{nonexistent_workspace}/deriver/status?peer_id={fake_peer}"
        )
        # This should return empty result since workspace/peer doesn't exist
        assert response.status_code == 200
        data = response.json()
        assert data["peer_id"] == fake_peer
        assert data["total_work_units"] == 0
        assert data["completed_work_units"] == 0
        assert data["in_progress_work_units"] == 0
        assert data["pending_work_units"] == 0

    async def test_get_deriver_status_with_queue_items(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status when there are actual queue items"""
        test_workspace, test_peer = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.flush()

        # Create some queue items to test with
        queue_items = [
            models.QueueItem(
                session_id=test_session.id,
                processed=False,
                payload={
                    "task_type": "representation",
                    "sender_name": test_peer.name,
                    "target_name": test_peer.name,
                    "workspace_name": test_workspace.name,
                    "session_name": test_session.name,
                },
            ),
            models.QueueItem(
                session_id=test_session.id,
                processed=True,
                payload={
                    "task_type": "representation",
                    "sender_name": test_peer.name,
                    "target_name": test_peer.name,
                    "workspace_name": test_workspace.name,
                    "session_name": test_session.name,
                },
            ),
        ]
        db_session.add_all(queue_items)
        await db_session.commit()

        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&session_id={test_session.name}"
        )
        assert response.status_code == 200
        data = response.json()

        # Should have some work units
        assert data["total_work_units"] >= 2
        assert data["completed_work_units"] >= 1
        assert data["pending_work_units"] >= 1

    async def test_get_deriver_status_with_sessions_breakdown(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test getting deriver status shows sessions breakdown when appropriate"""
        test_workspace, test_peer = sample_data

        # Create multiple test sessions
        test_session1 = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        test_session2 = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add_all([test_session1, test_session2])
        await db_session.flush()

        # Create queue items for different sessions
        queue_items = [
            models.QueueItem(
                session_id=test_session1.id,
                processed=False,
                payload={
                    "task_type": "representation",
                    "sender_name": test_peer.name,
                    "target_name": test_peer.name,
                    "workspace_name": test_workspace.name,
                },
            ),
            models.QueueItem(
                session_id=test_session2.id,
                processed=True,
                payload={
                    "task_type": "representation",
                    "sender_name": test_peer.name,
                    "target_name": test_peer.name,
                    "workspace_name": test_workspace.name,
                },
            ),
        ]
        db_session.add_all(queue_items)
        await db_session.commit()

        # Get status for peer only (should include sessions breakdown)
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}"
        )
        assert response.status_code == 200
        data = response.json()

        # Should have sessions breakdown when querying by peer only
        if "sessions" in data and data["sessions"]:
            assert isinstance(data["sessions"], dict)
            # Each session should have its own status
            for _, session_data in data["sessions"].items():
                assert "total_work_units" in session_data
                assert "completed_work_units" in session_data
                assert "in_progress_work_units" in session_data
                assert "pending_work_units" in session_data

    async def test_get_deriver_status_empty_parameters(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test various edge cases with empty or invalid parameters"""
        test_workspace, _ = sample_data

        # Test with empty peer_id
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id="
        )
        assert response.status_code == 400

        # Test with empty session_id
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?session_id="
        )
        assert response.status_code == 400

    async def test_get_deriver_status_boolean_parameter_variations(
        self, client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
    ):
        """Test different boolean parameter formats for include_sender"""
        test_workspace, test_peer = sample_data

        # Test with string 'true'
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=true"
        )
        assert response.status_code == 200

        # Test with string 'false'
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=false"
        )
        assert response.status_code == 200

        # Test with boolean True
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=True"
        )
        assert response.status_code == 200

        # Test with boolean False
        response = client.get(
            f"/v2/workspaces/{test_workspace.name}/deriver/status?peer_id={test_peer.name}&include_sender=False"
        )
        assert response.status_code == 200

    async def test_get_deriver_status_response_consistency(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that response structure is consistent across different parameter combinations"""
        test_workspace, test_peer = sample_data

        # Create a test session
        test_session = models.Session(
            workspace_name=test_workspace.name, name=str(generate_nanoid())
        )
        db_session.add(test_session)
        await db_session.commit()

        # Test different parameter combinations and ensure consistent response structure
        test_cases = [
            f"?peer_id={test_peer.name}",
            f"?session_id={test_session.name}",
            f"?peer_id={test_peer.name}&session_id={test_session.name}",
            f"?peer_id={test_peer.name}&include_sender=true",
            f"?session_id={test_session.name}&include_sender=false",
        ]

        for params in test_cases:
            response = client.get(
                f"/v2/workspaces/{test_workspace.name}/deriver/status{params}"
            )
            assert response.status_code == 200
            data = response.json()

            # All responses should have these base fields
            assert "total_work_units" in data
            assert "completed_work_units" in data
            assert "in_progress_work_units" in data
            assert "pending_work_units" in data

            # Verify counts are non-negative integers
            assert data["total_work_units"] >= 0
            assert data["completed_work_units"] >= 0
            assert data["in_progress_work_units"] >= 0
            assert data["pending_work_units"] >= 0

            # Verify total equals sum of components
            assert data["total_work_units"] == (
                data["completed_work_units"]
                + data["in_progress_work_units"]
                + data["pending_work_units"]
            )
