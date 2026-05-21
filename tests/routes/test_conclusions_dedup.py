import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


class TestConclusionDedupRoutes:
    """Test suite for conclusion deduplication API"""

    @pytest.mark.asyncio
    async def test_create_conclusion_deduplication(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that creating duplicate conclusions with deduplicate=True works.

        When deduplicate=True:
        - Identical content: the new document replaces the old one (soft-delete old,
          insert new). This is the "replace" path in is_rejected_duplicate().
        - New content is more informative: same replace behavior.
        - Existing content is more informative: the new content is rejected/skipped.
        """
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        content = "User loves semantic memory"

        # Create conclusion first time
        response1 = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": content,
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                    }
                ],
                "deduplicate": True
            },
        )
        assert response1.status_code == 201
        data1 = response1.json()
        assert len(data1) == 1
        id1 = data1[0]["id"]

        # Create identical conclusion with deduplicate=True
        response2 = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": content,
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                    }
                ],
                "deduplicate": True
            },
        )
        assert response2.status_code == 201
        data2 = response2.json()

        # Identical content triggers the "replace" path: new document replaces old.
        # is_rejected_duplicate sees equal scores -> soft-deletes old, returns False
        # (don't reject new). So we get a new id, and only the active count is 1.
        assert len(data2) == 1
        id2 = data2[0]["id"]
        assert id1 != id2  # New id != old id (replacement)

        # Verify only one exists in DB (active)
        list_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer_id": test_peer.name,
                    "observed_id": test_peer2.name,
                }
            },
        )
        assert list_response.status_code == 200
        # The list endpoint only returns active (non-deleted) documents
        assert len(list_response.json()["items"]) == 1
        assert list_response.json()["items"][0]["id"] == id2

    @pytest.mark.asyncio
    async def test_create_conclusion_no_deduplication_by_default(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that conclusions are NOT deduplicated by default (backward compatibility)"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        content = "User loves maximalist storage"

        # Create conclusion twice without explicit dedup flag
        for _ in range(2):
            response = client.post(
                f"/v3/workspaces/{test_workspace.name}/conclusions",
                json={
                    "conclusions": [
                        {
                            "content": content,
                            "observer_id": test_peer.name,
                            "observed_id": test_peer2.name,
                        }
                    ]
                },
            )
            assert response.status_code == 201

        # Verify BOTH exist in DB
        list_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer_id": test_peer.name,
                    "observed_id": test_peer2.name,
                }
            },
        )
        assert list_response.status_code == 200
        assert len(list_response.json()["items"]) == 2
