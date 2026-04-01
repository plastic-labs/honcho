import datetime

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
            session_name=test_session.name,
        )
        doc2 = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            content="Test observation 2",
            session_name=test_session.name,
        )
        db_session.add_all([doc1, doc2])
        await db_session.flush()

        # Get documents query
        stmt = crud.get_all_documents(
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

        # Create test documents using create_documents to ensure they're in vector store
        doc_schemas = [
            schemas.DocumentCreate(
                content="User likes pizza",
                embedding=[0.9] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="User dislikes vegetables",
                embedding=[0.1] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[2],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
        ]
        await crud.create_documents(
            db_session,
            doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

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
    async def test_query_documents_excludes_soft_deleted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Query results should not include soft-deleted documents even if vectors remain"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create two documents and persist embeddings
        doc_schemas = [
            schemas.DocumentCreate(
                content="User likes pizza",
                embedding=[0.9] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="User dislikes vegetables",
                embedding=[0.1] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[2],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
        ]
        await crud.create_documents(
            db_session,
            doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # Soft-delete one document without touching vectors
        stmt = select(models.Document).where(
            models.Document.workspace_name == test_workspace.name,
            models.Document.observer == test_peer.name,
            models.Document.observed == test_peer2.name,
        )
        result = await db_session.execute(stmt)
        docs = {doc.content: doc for doc in result.scalars().all()}
        deleted_doc = docs["User likes pizza"]
        kept_doc = docs["User dislikes vegetables"]

        deleted_doc.deleted_at = datetime.datetime.now(datetime.timezone.utc)
        await db_session.commit()

        results = await crud.query_documents(
            db_session,
            workspace_name=test_workspace.name,
            query="food preferences",
            observer=test_peer.name,
            observed=test_peer2.name,
            top_k=10,
        )

        assert len(results) == 1
        assert results[0].id == kept_doc.id

    @pytest.mark.asyncio
    async def test_query_documents_applies_additional_filters(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Filters beyond vector metadata should be enforced at the DB layer"""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        doc_schemas = [
            schemas.DocumentCreate(
                content="Observation one",
                embedding=[0.5] * 1536,
                session_name=test_session.name,
                times_derived=1,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="Observation two",
                embedding=[0.5] * 1536,
                session_name=test_session.name,
                times_derived=2,
                metadata=schemas.DocumentMetadata(
                    message_ids=[2],
                    message_created_at="2025-01-01T00:00:00Z",
                ),
            ),
        ]
        await crud.create_documents(
            db_session,
            doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        result = await db_session.execute(
            select(models.Document).where(
                models.Document.workspace_name == test_workspace.name,
                models.Document.observer == test_peer.name,
                models.Document.observed == test_peer2.name,
            )
        )
        docs = result.scalars().all()
        times_derived_map = {doc.times_derived: doc.id for doc in docs}

        results = await crud.query_documents(
            db_session,
            workspace_name=test_workspace.name,
            query="any query",
            observer=test_peer.name,
            observed=test_peer2.name,
            top_k=10,
            filters={"times_derived": 2},
            embedding=[0.5] * 1536,
        )

        assert len(results) == 1
        assert results[0].id == times_derived_map[2]

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

        # Verify document is soft-deleted
        result = await db_session.execute(stmt)
        doc = result.scalar_one_or_none()
        assert doc is not None
        assert doc.deleted_at is not None

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
                embedding=[0.1] * 1536,
                session_name=test_session.name,
                level="explicit",
                metadata=schemas.DocumentMetadata(
                    message_ids=[1, 2, 3, 4, 5],
                    message_created_at="2024-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="Observation 2",
                embedding=[0.2] * 1536,
                session_name=test_session.name,
                level="deductive",
                metadata=schemas.DocumentMetadata(
                    message_ids=[6, 7, 8, 9, 10],
                    message_created_at="2024-01-01T00:01:00Z",
                    premises=["Premise 1", "Premise 2"],
                ),
            ),
        ]

        # Create documents
        created_ids = await crud.create_documents(
            db_session,
            documents=doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        assert len(created_ids) == 2
        assert all(isinstance(doc_id, str) for doc_id in created_ids)

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

    @pytest.mark.asyncio
    async def test_create_documents_returns_empty_list_when_all_deduped(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """create_documents returns [] when all docs are rejected as duplicates."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create a more informative original document
        original_schema = schemas.DocumentCreate(
            content="User loves pizza and also enjoys pasta with fresh tomato sauce",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2024-01-01T00:00:00Z",
            ),
        )
        await crud.create_documents(
            db_session,
            documents=[original_schema],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )

        # Create a less informative near-duplicate (subset of original content).
        # Dedup rejects the new doc when the existing one has more information.
        lesser_schema = schemas.DocumentCreate(
            content="User loves pizza",
            embedding=[0.5] * 1536,  # Same embedding to trigger similarity match
            session_name=test_session.name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[2],
                message_created_at="2024-01-01T00:00:00Z",
            ),
        )
        duplicate_ids = await crud.create_documents(
            db_session,
            documents=[lesser_schema],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )

        assert duplicate_ids == []

    @pytest.mark.asyncio
    async def test_dedup_sets_superseded_by(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """When a more informative doc replaces an existing one, superseded_by is set."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create original (less informative)
        original_schema = schemas.DocumentCreate(
            content="User likes pizza",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2024-01-01T00:00:00Z",
            ),
        )
        original_ids = await crud.create_documents(
            db_session,
            documents=[original_schema],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )
        assert len(original_ids) == 1
        original_id = original_ids[0]

        # Create replacement (more informative, similar embedding triggers dedup)
        replacement_schema = schemas.DocumentCreate(
            content="User likes pizza and also enjoys pasta with fresh tomato sauce",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[2],
                message_created_at="2024-01-01T00:00:00Z",
            ),
        )
        replacement_ids = await crud.create_documents(
            db_session,
            documents=[replacement_schema],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )
        assert len(replacement_ids) == 1
        replacement_id = replacement_ids[0]

        # Verify the original doc has superseded_by set
        stmt = select(models.Document).where(models.Document.id == original_id)
        old_doc = (await db_session.execute(stmt)).scalar_one()
        assert old_doc.superseded_by == replacement_id
        assert old_doc.deleted_at is not None

    @pytest.mark.asyncio
    async def test_dedup_no_supersession_when_rejected(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """When the new doc is rejected (less informative), no supersession link is created."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Create original (more informative)
        original_schema = schemas.DocumentCreate(
            content="User loves pizza and also enjoys pasta with fresh tomato sauce",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            level="explicit",
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2024-01-01T00:00:00Z",
            ),
        )
        original_ids = await crud.create_documents(
            db_session,
            documents=[original_schema],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )
        original_id = original_ids[0]

        # Create lesser duplicate — should be rejected
        await crud.create_documents(
            db_session,
            documents=[
                schemas.DocumentCreate(
                    content="User likes pizza",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    level="explicit",
                    metadata=schemas.DocumentMetadata(
                        message_ids=[2],
                        message_created_at="2024-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )

        # Original should be untouched
        stmt = select(models.Document).where(models.Document.id == original_id)
        original_doc = (await db_session.execute(stmt)).scalar_one()
        assert original_doc.superseded_by is None
        assert original_doc.deleted_at is None

    @pytest.mark.asyncio
    async def test_dedup_no_supersession_when_no_duplicate(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Unique documents have no supersession field set."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        doc_ids = await crud.create_documents(
            db_session,
            documents=[
                schemas.DocumentCreate(
                    content="Completely unique observation",
                    embedding=[0.9] * 1536,
                    session_name=test_session.name,
                    level="explicit",
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2024-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )

        stmt = select(models.Document).where(models.Document.id == doc_ids[0])
        doc = (await db_session.execute(stmt)).scalar_one()
        assert doc.superseded_by is None

    @pytest.mark.asyncio
    async def test_dedup_backward_compatible_without_flag(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """deduplicate=False skips dedup entirely, no supersession field set."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        doc_ids = await crud.create_documents(
            db_session,
            documents=[
                schemas.DocumentCreate(
                    content="Observation without dedup",
                    embedding=[0.7] * 1536,
                    session_name=test_session.name,
                    level="explicit",
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2024-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )

        stmt = select(models.Document).where(models.Document.id == doc_ids[0])
        doc = (await db_session.execute(stmt)).scalar_one()
        assert doc.superseded_by is None
