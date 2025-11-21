import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.exceptions import ResourceNotFoundException


class TestDocumentCRUD:
    """Test suite for document CRUD operations"""

    async def _setup_test_data(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Collection]:
        """Helper to set up test data with collection"""
        # Create another peer to observe
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
        await db_session.flush()

        # Create collection (required for documents foreign key)
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()

        return test_peer2, test_session, collection

    @pytest.mark.asyncio
    async def test_get_all_documents_returns_query(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test get_all_documents returns a Select query for pagination"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create test documents
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation 1",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation 2",
            embedding=[0.2] * 1536,
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.flush()

        # Get documents query
        stmt = await crud.get_all_documents(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # Execute query
        result = await db_session.execute(stmt)
        documents = result.scalars().all()

        assert len(documents) == 2
        assert documents[0].content in ["Test observation 1", "Test observation 2"]

    @pytest.mark.asyncio
    async def test_query_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test query_documents with semantic search"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create test documents with different embeddings
        doc1 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User likes pizza",
            embedding=[0.9] * 1536,
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="User dislikes vegetables",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.flush()

        # Query documents
        results = await crud.query_documents(
            db_session,
            workspace_name=test_workspace.name,
            query="food preferences",
            observer=test_peer.name,
            observed=test_peer2.name,
            top_k=10,
        )

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_document_success(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test delete_document successfully deletes a document"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create a document
        doc = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation",
            embedding=[0.1] * 1536,
            session_name=test_session.name,
        )
        db_session.add(doc)
        await db_session.flush()

        doc_id = doc.id

        # Verify document exists
        stmt = select(models.Document).where(models.Document.id == doc_id)
        result = await db_session.execute(stmt)
        assert result.scalar_one_or_none() is not None

        # Delete document
        await crud.delete_document(
            db_session,
            workspace_name=test_workspace.name,
            document_id=doc_id,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # Verify document is deleted
        result = await db_session.execute(stmt)
        assert result.scalar_one_or_none() is None

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test delete_document raises exception for non-existent document"""
        test_workspace, test_peer = sample_data
        test_peer2, _, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Try to delete non-existent document
        with pytest.raises(ResourceNotFoundException):
            await crud.delete_document(
                db_session,
                workspace_name=test_workspace.name,
                document_id="nonexistent_id",
                observer=test_peer.name,
                observed=test_peer2.name,
            )

    @pytest.mark.asyncio
    async def test_create_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Test create_documents creates multiple documents"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Prepare document creation schemas
        doc_schemas = [
            schemas.DocumentCreate(
                content="Observation 1",
                session_name=test_session.name,
                embedding=[0.1] * 1536,
                level="explicit",
                metadata=schemas.DocumentMetadata(
                    message_ids=[(1, 5)],
                    message_created_at="2024-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="Observation 2",
                session_name=test_session.name,
                embedding=[0.2] * 1536,
                level="deductive",
                metadata=schemas.DocumentMetadata(
                    message_ids=[(6, 10)],
                    message_created_at="2024-01-01T00:01:00Z",
                    premises=["Premise 1", "Premise 2"],
                ),
            ),
        ]

        # Create documents
        count = await crud.create_documents(
            db_session,
            documents=doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        assert count == 2

        # Verify documents were created
        stmt = select(models.Document).where(
            models.Document.workspace_name == test_workspace.name,
            models.Document.observer == test_peer.name,
            models.Document.observed == test_peer2.name,
        )
        result = await db_session.execute(stmt)
        documents = result.scalars().all()

        assert len(documents) == 2
        assert documents[0].content in ["Observation 1", "Observation 2"]
        assert documents[1].content in ["Observation 1", "Observation 2"]
