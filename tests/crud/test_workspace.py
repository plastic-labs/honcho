import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.exceptions import ResourceNotFoundException


class TestWorkspaceCRUD:
    """Test suite for workspace CRUD operations"""

    @pytest.mark.asyncio
    async def test_delete_workspace_not_found(self, db_session: AsyncSession):
        """Test delete_workspace with non-existent workspace raises ResourceNotFoundException"""
        with pytest.raises(ResourceNotFoundException):
            await crud.delete_workspace(db_session, "nonexistent_workspace")

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_peers(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete peers"""
        test_workspace, _test_peer = sample_data

        # Create additional peer
        peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(peer2)
        await db_session.flush()

        # Verify peers exist
        stmt = select(models.Peer).where(
            models.Peer.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        peers = result.scalars().all()
        assert len(peers) == 2

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify peers are deleted
        result = await db_session.execute(stmt)
        peers = result.scalars().all()
        assert len(peers) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_sessions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete sessions"""
        test_workspace, _test_peer = sample_data

        # Create sessions
        session1 = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session2 = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([session1, session2])
        await db_session.flush()

        # Verify sessions exist
        stmt = select(models.Session).where(
            models.Session.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        assert len(sessions) == 2

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify sessions are deleted
        result = await db_session.execute(stmt)
        sessions = result.scalars().all()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_messages(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete messages"""
        test_workspace, test_peer = sample_data

        # Create session
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        # Create messages
        message1 = models.Message(
            content="Test message 1",
            workspace_name=test_workspace.name,
            session_name=session.name,
            peer_name=test_peer.name,
        )
        message2 = models.Message(
            content="Test message 2",
            workspace_name=test_workspace.name,
            session_name=session.name,
            peer_name=test_peer.name,
        )
        db_session.add_all([message1, message2])
        await db_session.flush()

        # Verify messages exist
        stmt = select(models.Message).where(
            models.Message.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        assert len(messages) == 2

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify messages are deleted
        result = await db_session.execute(stmt)
        messages = result.scalars().all()
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_collections(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete collections"""
        test_workspace, test_peer = sample_data

        # Create collection
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
        )
        db_session.add(collection)
        await db_session.flush()

        # Verify collection exists
        stmt = select(models.Collection).where(
            models.Collection.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        collections = result.scalars().all()
        assert len(collections) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify collection is deleted
        result = await db_session.execute(stmt)
        collections = result.scalars().all()
        assert len(collections) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete documents"""
        test_workspace, test_peer = sample_data

        # Create collection
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
        )
        db_session.add(collection)
        await db_session.flush()

        # Create session for document
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        # Create document
        document = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer.name,
            session_name=session.name,
            content="Test document content",
            embedding=[0.1] * 1536,  # Mock embedding vector
        )
        db_session.add(document)
        await db_session.flush()

        # Verify document exists
        stmt = select(models.Document).where(
            models.Document.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        documents = result.scalars().all()
        assert len(documents) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify document is deleted
        result = await db_session.execute(stmt)
        documents = result.scalars().all()
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_session_peers(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete session_peers associations"""
        test_workspace, test_peer = sample_data

        # Create session
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        # Add peer to session
        from src.models import session_peers_table

        stmt = session_peers_table.insert().values(
            workspace_name=test_workspace.name,
            session_name=session.name,
            peer_name=test_peer.name,
        )
        await db_session.execute(stmt)
        await db_session.flush()

        # Verify session_peer association exists
        stmt = select(session_peers_table).where(
            session_peers_table.c.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        session_peers = result.all()
        assert len(session_peers) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify session_peer association is deleted
        result = await db_session.execute(stmt)
        session_peers = result.all()
        assert len(session_peers) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_webhooks(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete webhook endpoints"""
        test_workspace, _test_peer = sample_data

        # Create webhook endpoint
        webhook = models.WebhookEndpoint(
            workspace_name=test_workspace.name,
            url="https://example.com/webhook",
        )
        db_session.add(webhook)
        await db_session.flush()

        # Verify webhook exists
        stmt = select(models.WebhookEndpoint).where(
            models.WebhookEndpoint.workspace_name == test_workspace.name
        )
        result = await db_session.execute(stmt)
        webhooks = result.scalars().all()
        assert len(webhooks) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify webhook is deleted
        result = await db_session.execute(stmt)
        webhooks = result.scalars().all()
        assert len(webhooks) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_queue_items(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete queue items"""
        test_workspace, test_peer = sample_data

        # Create session
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        # Create queue item with work_unit_key containing workspace name
        # Format: {task_type}:{workspace_name}:{...}
        queue_item = models.QueueItem(
            work_unit_key=f"representation:{test_workspace.name}:{session.name}:{test_peer.name}:{test_peer.name}",
            task_type="representation",
            payload={"test": "data"},
        )
        db_session.add(queue_item)
        await db_session.flush()

        # Verify queue item exists
        stmt = select(models.QueueItem)
        result = await db_session.execute(stmt)
        queue_items = result.scalars().all()
        assert len(queue_items) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify queue item is deleted
        result = await db_session.execute(stmt)
        queue_items = result.scalars().all()
        assert len(queue_items) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_cascade_active_queue_sessions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that deleting a workspace cascades to delete active queue sessions"""
        test_workspace, test_peer = sample_data

        # Create session
        session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(session)
        await db_session.flush()

        # Create active queue session with work_unit_key containing workspace name
        # Format: {task_type}:{workspace_name}:{...}
        active_queue = models.ActiveQueueSession(
            work_unit_key=f"representation:{test_workspace.name}:{session.name}:{test_peer.name}:{test_peer.name}",
        )
        db_session.add(active_queue)
        await db_session.flush()

        # Verify active queue session exists
        stmt = select(models.ActiveQueueSession)
        result = await db_session.execute(stmt)
        active_queues = result.scalars().all()
        assert len(active_queues) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify active queue session is deleted
        result = await db_session.execute(stmt)
        active_queues = result.scalars().all()
        assert len(active_queues) == 0

    @pytest.mark.asyncio
    async def test_delete_workspace_returns_deleted_workspace(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test that delete_workspace returns the deleted workspace object"""
        test_workspace, _test_peer = sample_data

        # Store workspace details before deletion
        workspace_id = test_workspace.id
        workspace_name = test_workspace.name

        # Delete workspace
        deleted_workspace = await crud.delete_workspace(db_session, test_workspace.name)

        # Verify returned workspace matches the deleted workspace
        assert deleted_workspace.id == workspace_id
        assert deleted_workspace.name == workspace_name
        assert isinstance(deleted_workspace, models.Workspace)

    @pytest.mark.asyncio
    async def test_delete_workspace_complex_cascade(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test deleting a workspace with multiple related resources of different types"""
        test_workspace, test_peer = sample_data

        # Create additional peer
        peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(peer2)

        # Create sessions
        session1 = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session2 = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([session1, session2])
        await db_session.flush()

        # Create messages
        message1 = models.Message(
            content="Test message 1",
            workspace_name=test_workspace.name,
            session_name=session1.name,
            peer_name=test_peer.name,
        )
        message2 = models.Message(
            content="Test message 2",
            workspace_name=test_workspace.name,
            session_name=session2.name,
            peer_name=peer2.name,
        )
        db_session.add_all([message1, message2])

        # Create collection and document
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        document = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=peer2.name,
            session_name=session1.name,
            content="Test document",
            embedding=[0.1] * 1536,  # Mock embedding vector
        )
        db_session.add(document)

        # Create webhook
        webhook = models.WebhookEndpoint(
            workspace_name=test_workspace.name,
            url="https://example.com/webhook",
        )
        db_session.add(webhook)
        await db_session.flush()

        # Count all resources before deletion
        peer_stmt = select(models.Peer).where(
            models.Peer.workspace_name == test_workspace.name
        )
        session_stmt = select(models.Session).where(
            models.Session.workspace_name == test_workspace.name
        )
        message_stmt = select(models.Message).where(
            models.Message.workspace_name == test_workspace.name
        )
        collection_stmt = select(models.Collection).where(
            models.Collection.workspace_name == test_workspace.name
        )
        document_stmt = select(models.Document).where(
            models.Document.workspace_name == test_workspace.name
        )
        webhook_stmt = select(models.WebhookEndpoint).where(
            models.WebhookEndpoint.workspace_name == test_workspace.name
        )

        # Verify all resources exist
        assert len((await db_session.execute(peer_stmt)).scalars().all()) == 2
        assert len((await db_session.execute(session_stmt)).scalars().all()) == 2
        assert len((await db_session.execute(message_stmt)).scalars().all()) == 2
        assert len((await db_session.execute(collection_stmt)).scalars().all()) == 1
        assert len((await db_session.execute(document_stmt)).scalars().all()) == 1
        assert len((await db_session.execute(webhook_stmt)).scalars().all()) == 1

        # Delete workspace
        await crud.delete_workspace(db_session, test_workspace.name)

        # Verify all related resources are deleted
        assert len((await db_session.execute(peer_stmt)).scalars().all()) == 0
        assert len((await db_session.execute(session_stmt)).scalars().all()) == 0
        assert len((await db_session.execute(message_stmt)).scalars().all()) == 0
        assert len((await db_session.execute(collection_stmt)).scalars().all()) == 0
        assert len((await db_session.execute(document_stmt)).scalars().all()) == 0
        assert len((await db_session.execute(webhook_stmt)).scalars().all()) == 0
