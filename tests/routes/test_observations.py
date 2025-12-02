import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


class TestObservationRoutes:
    """Test suite for observation API endpoints"""

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
    async def test_list_observations_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing observations for a session"""
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

        # Create test observations (documents)
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User prefers dark mode",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User works late at night",
            embedding=[0.2] * 1536,
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List observations
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2

        # Check observation structure
        observation = data["items"][0]
        assert "id" in observation
        assert "content" in observation
        assert "observer_id" in observation
        assert "observed_id" in observation
        assert "session_id" in observation
        assert "created_at" in observation

        # Verify content
        contents = [item["content"] for item in data["items"]]
        assert "User prefers dark mode" in contents
        assert "User works late at night" in contents

    @pytest.mark.asyncio
    async def test_list_observations_empty_session(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing observations for a session with no observations"""
        test_workspace, _test_peer = sample_data

        # Create a session without any observations
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # List observations
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_list_observations_with_filters(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing observations with observer/observed filters"""
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

        # Create observations with different observer/observed pairs
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Peer1 observes Peer2",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer2.name,
            observed=test_peer3.name,
            content="Peer2 observes Peer3",
            embedding=[0.2] * 1536,
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # List observations filtered by observer
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list",
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
    async def test_list_observations_reverse_order(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing observations in reverse chronological order"""
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

        # Create observations
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="First observation",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc1)
        await db_session.flush()

        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Second observation",
            embedding=[0.2] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc2)
        await db_session.commit()

        # List observations in reverse (oldest first)
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list?reverse=true",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["content"] == "First observation"
        assert data["items"][1]["content"] == "Second observation"

    @pytest.mark.asyncio
    async def test_list_observations_pagination(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test pagination of observations list"""
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

        # Create multiple observations
        for i in range(15):
            doc = models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content=f"Observation {i}",
                embedding=[0.1 * i] * 1536,
                session_name=test_session.name,
            )
            db_session.add(doc)
        await db_session.commit()

        # Get first page (default size)
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list?page=1&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["total"] == 15

        # Get second page
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list?page=2&size=10",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 5
        assert data["total"] == 15

    @pytest.mark.asyncio
    async def test_query_observations_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying observations with semantic search"""
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

        # Create test observations
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User loves pizza and pasta",
            embedding=[0.9] * 1536,
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User dislikes vegetables",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.commit()

        # Query observations
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/query",
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

        # Check observation structure
        observation = data[0]  # pyright: ignore
        assert "id" in observation
        assert "content" in observation
        assert "observer_id" in observation
        assert "observed_id" in observation

    @pytest.mark.asyncio
    async def test_query_observations_with_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying observations with top_k limit"""
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

        # Create multiple observations
        for i in range(5):
            doc = models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content=f"Observation about topic {i}",
                embedding=[0.1 * i] * 1536,
                session_name=test_session.name,
            )
            db_session.add(doc)
        await db_session.commit()

        # Query with top_k=2
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/query",
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
    async def test_query_observations_with_distance_threshold(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test querying observations with distance threshold"""
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

        # Create test observation
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        # Query with distance threshold
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/query",
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
    async def test_query_observations_requires_observer_observed(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query observations requires observer and observed in filters"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Query without observer/observed filters should fail
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/query",
            json={"query": "test"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_observations_invalid_top_k(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test query observations validates top_k range"""
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
            f"/v2/workspaces/{test_workspace.name}/observations/query",
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
    async def test_delete_observation_success(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting an observation"""
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

        # Create a test observation
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation to delete",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        observation_id = doc.id

        # Delete observation
        response = client.delete(
            f"/v2/workspaces/{test_workspace.name}/observations/{observation_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Observation deleted successfully"

        # Verify observation is deleted
        from sqlalchemy import select

        stmt = select(models.Document).where(models.Document.id == observation_id)
        result = await db_session.execute(stmt)
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_observation_not_found(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test deleting a non-existent observation"""
        test_workspace, _test_peer = sample_data

        # Create a session
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_session)
        await db_session.commit()

        # Try to delete non-existent observation
        response = client.delete(
            f"/v2/workspaces/{test_workspace.name}/observations/nonexistent_id"
        )

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_observations_nonexistent_session(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test listing observations for non-existent session"""
        test_workspace, _test_peer = sample_data

        # Try to list observations for non-existent session
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list",
            json={"filters": {"session_id": "nonexistent_session"}},
        )

        # Should return empty result, not error (session might exist but no observations)
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0

    @pytest.mark.asyncio
    async def test_observations_field_mapping(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Test that observation fields are properly mapped from document model"""
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

        # Create test observation
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation content",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.commit()

        # List observations
        response = client.post(
            f"/v2/workspaces/{test_workspace.name}/observations/list",
            json={"filters": {"session_id": test_session.name}},
        )

        assert response.status_code == 200
        data = response.json()
        observation = data["items"][0]

        # Verify field mappings
        assert observation["id"] == doc.id
        assert observation["content"] == doc.content
        assert observation["observer_id"] == doc.observer
        assert observation["observed_id"] == doc.observed
        assert observation["session_id"] == doc.session_name
        assert "created_at" in observation

        # Verify internal fields are NOT exposed
        assert "embedding" not in observation
        assert "internal_metadata" not in observation
        assert "collection" not in observation
