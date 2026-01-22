import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


class TestConclusionRoutes:
    """Test suite for conclusion API endpoints"""

    async def _create_collection(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> models.Collection:
        """Helper to create collection for tests"""
        collection = models.Collection(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        db_session.add(collection)
        await db_session.flush()
        return collection

    @pytest.mark.asyncio
    async def test_list_conclusions_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for a session"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create test conclusions (documents)
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User prefers dark mode",
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User works late at night",
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2

        # Check conclusion structure
        conclusion = data["items"][0]
        assert "id" in conclusion
        assert "content" in conclusion
        assert "observer_id" in conclusion
        assert "observed_id" in conclusion
        assert "session_id" in conclusion
        assert "created_at" in conclusion

        # Verify content
        contents = [item["content"] for item in data["items"]]
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents

    @pytest.mark.asyncio
    async def test_list_conclusions_empty_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for a session with no conclusions"""
        test_workspace, _test_peer = sample_data

        # Create a session without any conclusions
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_list_conclusions_with_filters(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions with observer/observed filters"""
        test_workspace, test_peer = sample_data

        # Create two more peers
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_peer3 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_peer3])
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collections for both observer/observed pairs
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        await self._create_collection(
            db_session, test_workspace.name, test_peer2.name, test_peer3.name
        )

        # Create conclusions with different observer/observed pairs
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Peer1 observes Peer2",
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer2.name,
            observed=test_peer3.name,
            content="Peer2 observes Peer3",
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List conclusions filtered by observer
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {"observer": test_peer.name, "session_id": test_session.name}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["content"] == "Peer1 observes Peer2"
        assert data["items"][0]["observer_id"] == test_peer.name

    @pytest.mark.asyncio
    async def test_list_conclusions_reverse_order(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions in reverse chronological order"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create conclusions
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="First conclusion",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc1)
        await db_session.flush()

        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Second conclusion",
            embedding=[0.2] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc2)
        await db_session.commit()

        # List conclusions in reverse (oldest first)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?reverse=true",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["content"] == "First conclusion"
        assert data["items"][1]["content"] == "Second conclusion"

    @pytest.mark.asyncio
    async def test_list_conclusions_pagination(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test pagination of conclusions list"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create multiple conclusions
        for i in range(15):
            doc = models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content=f"Conclusion {i}",
                embedding=[0.1 * i] * 1536,
                session_name=test_session.name,
            )
            db_session.add(doc)
        await db_session.commit()

        # Get first page (default size)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?page=1&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] == 15

        # Get second page
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list?page=2&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
        assert data["total"] == 15

    @pytest.mark.asyncio
    async def test_query_conclusions_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with semantic search"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create test conclusions via API (ensures proper vector store integration)
        create_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "User loves pizza and pasta",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                    {
                        "content": "User dislikes vegetables",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                ]
            },
        )
        assert create_response.status_code == 201

        # Query conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "food preferences",
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1  # pyright: ignore

        # Check conclusion structure
        conclusion = data[0]  # pyright: ignore
        assert "id" in conclusion
        assert "content" in conclusion
        assert "observer_id" in conclusion
        assert "observed_id" in conclusion

    @pytest.mark.asyncio
    async def test_query_conclusions_with_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with top_k limit"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create multiple conclusions via API (ensures proper vector store integration)
        conclusions = [
            {
                "content": f"Conclusion about topic {i}",
                "observer_id": test_peer.name,
                "observed_id": test_peer2.name,
                "session_id": test_session.name,
            }
            for i in range(5)
        ]
        create_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={"conclusions": conclusions},
        )
        assert create_response.status_code == 201

        # Query with top_k=2
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "relevant topic",
                "top_k": 2,
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 2  # pyright: ignore

    @pytest.mark.asyncio
    async def test_query_conclusions_with_distance_threshold(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying conclusions with distance threshold"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create test conclusion via API (ensures proper vector store integration)
        create_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )
        assert create_response.status_code == 201

        # Query with distance threshold
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "test",
                "distance": 0.8,
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_query_conclusions_requires_observer_observed(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query conclusions requires observer and observed in filters"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Query without observer/observed filters should fail
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={"query": "test"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_conclusions_invalid_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query conclusions validates top_k range"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Query with invalid top_k (too high)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/query",
            json={
                "query": "test",
                "top_k": 101,  # Max is 100
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                },
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_delete_conclusion_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting a conclusion"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create a test conclusion
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test conclusion to delete",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        conclusion_id = doc.id

        # Delete conclusion
        response = client.delete(
            f"/v3/workspaces/{test_workspace.name}/conclusions/{conclusion_id}"
        )

        assert response.status_code == 204

        # Verify conclusion is deleted
        from sqlalchemy import select

        stmt = select(models.Document).where(models.Document.id == conclusion_id)
        result = await db_session.execute(stmt)
        doc = result.scalar_one_or_none()
        assert doc is not None
        assert doc.deleted_at is not None

    @pytest.mark.asyncio
    async def test_delete_conclusion_not_found(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting a non-existent conclusion"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to delete non-existent conclusion
        response = client.delete(
            f"/v3/workspaces/{test_workspace.name}/conclusions/nonexistent_id"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_conclusions_nonexistent_session(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing conclusions for non-existent session"""
        test_workspace, _test_peer = sample_data

        # Try to list conclusions for non-existent session
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": "nonexistent_session"}},
        )

        # Should return empty result, not error (session might exist but no conclusions)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_conclusions_field_mapping(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that conclusion fields are properly mapped from document model"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create collection
        await self._create_collection(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )

        # Create test conclusion
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test conclusion content",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        # List conclusions
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        conclusion = data["items"][0]

        # Verify field mappings
        assert conclusion["id"] == doc.id
        assert conclusion["content"] == doc.content
        assert conclusion["observer_id"] == doc.observer
        assert conclusion["observed_id"] == doc.observed
        assert conclusion["session_id"] == doc.session_name
        assert "created_at" in conclusion

        # Verify internal fields are NOT exposed
        assert "embedding" not in conclusion
        assert "internal_metadata" not in conclusion
        assert "collection" not in conclusion

    @pytest.mark.asyncio
    async def test_create_conclusion_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating a single conclusion"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusion via API
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "User prefers dark mode",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1

        conclusion = data[0]
        assert conclusion["content"] == "User prefers dark mode"
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name
        assert conclusion["session_id"] == test_session.name
        assert "id" in conclusion
        assert "created_at" in conclusion

    @pytest.mark.asyncio
    async def test_create_conclusions_batch(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating multiple conclusions in batch"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create multiple conclusions via API
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "User prefers dark mode",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                    {
                        "content": "User works late at night",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                    {
                        "content": "User enjoys programming",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 3

        contents = [obs["content"] for obs in data]
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents
        assert "User enjoys programming" in contents

    @pytest.mark.asyncio
    async def test_create_conclusion_nonexistent_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with non-existent session fails"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.commit()

        # Try to create conclusion with non-existent session
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": "nonexistent_session",
                    }
                ]
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_conclusion_nonexistent_peer(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with non-existent peer fails"""
        test_workspace, test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to create conclusion with non-existent observer
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": "nonexistent_peer",
                        "observed_id": test_peer.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_conclusion_empty_content(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusion with empty content fails validation"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to create conclusion with empty content
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_conclusion_empty_list(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusions with empty list fails validation"""
        test_workspace, _test_peer = sample_data

        # Try to create with empty conclusions list
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={"conclusions": []},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_conclusion_creates_collection(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that creating conclusion auto-creates collection if needed"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusion via API (this should auto-create collection)
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Test conclusion",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1

        # The conclusion was created successfully, which means the collection
        # was created (since documents require a collection)
        conclusion = data[0]
        assert conclusion["observer_id"] == test_peer.name
        assert conclusion["observed_id"] == test_peer2.name

    @pytest.mark.asyncio
    async def test_create_conclusion_different_observer_observed_pairs(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test creating conclusions with different observer/observed pairs in single batch"""
        test_workspace, test_peer = sample_data

        # Create two more peers
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_peer3 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_peer3])
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusions with different observer/observed pairs
        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Peer1 observes Peer2",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    },
                    {
                        "content": "Peer2 observes Peer3",
                        "observer_id": test_peer2.name,
                        "observed_id": test_peer3.name,
                        "session_id": test_session.name,
                    },
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 2

        # Verify each conclusion has correct observer/observed
        obs1 = next(o for o in data if o["content"] == "Peer1 observes Peer2")
        assert obs1["observer_id"] == test_peer.name
        assert obs1["observed_id"] == test_peer2.name

        obs2 = next(o for o in data if o["content"] == "Peer2 observes Peer3")
        assert obs2["observer_id"] == test_peer2.name
        assert obs2["observed_id"] == test_peer3.name

    @pytest.mark.asyncio
    async def test_created_conclusions_are_searchable(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that created conclusions can be found via list endpoint"""
        test_workspace, test_peer = sample_data

        # Create another peer
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        await db_session.flush()

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Create conclusion via API
        create_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "Unique test content for searchability",
                        "observer_id": test_peer.name,
                        "observed_id": test_peer2.name,
                        "session_id": test_session.name,
                    }
                ]
            },
        )

        assert create_response.status_code == 201
        created_id = create_response.json()[0]["id"]

        # List conclusions and verify the created one is there
        list_response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            json={
                "filters": {
                    "observer": test_peer.name,
                    "observed": test_peer2.name,
                    "session_id": test_session.name,
                }
            },
        )

        assert list_response.status_code == 200
        data = list_response.json()
        ids = [obs["id"] for obs in data["items"]]
        assert created_id in ids
