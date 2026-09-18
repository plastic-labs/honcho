import asyncio
import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.crud.document import SemanticRejectionResult, is_rejected_duplicate
from src.exceptions import ResourceNotFoundException
from src.utils.types import DocumentLevel


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

        deleted_doc.deleted_at = datetime.datetime.now(datetime.UTC)
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
    async def test_most_derived_orders_by_recency_when_reinforcement_ties(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Regression: when times_derived ties, most-derived must fall back to
        recency, not insertion order. Otherwise stale conclusions stick to the
        front of the injected representation (the mid-Jan stickiness bug)."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        # Three conclusions, all reinforced once -- the real-world steady state
        # before the fix -- inserted oldest-first.
        for i in range(3):
            db_session.add(
                models.Document(
                    workspace_name=test_workspace.name,
                    observer=test_peer.name,
                    observed=test_peer2.name,
                    content=f"tie {i}",
                    session_name=test_session.name,
                    times_derived=1,
                    created_at=base + datetime.timedelta(days=i),
                )
            )
        # A genuinely reinforced conclusion that is also the oldest of all.
        db_session.add(
            models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                content="hot",
                session_name=test_session.name,
                times_derived=5,
                created_at=base - datetime.timedelta(days=10),
            )
        )
        await db_session.flush()

        docs = await crud.query_documents_most_derived(
            db_session,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            limit=10,
        )
        contents = [d.content for d in docs]
        # Primary sort still wins: the actually-reinforced conclusion leads.
        assert contents[0] == "hot"
        # Ties break toward most-recent, not oldest-inserted.
        assert contents[1:] == ["tie 2", "tie 1", "tie 0"]

    @pytest.mark.asyncio
    async def test_duplicate_rejection_reinforces_existing(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Rejecting a new duplicate must bump the surviving doc's times_derived."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="eri loves cats and dogs and birds and snakes",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # Fewer unique tokens -> existing wins -> new doc is rejected.
        new_doc = schemas.DocumentCreate(
            content="eri loves cats",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            times_derived=1,
            metadata=schemas.DocumentMetadata(
                message_ids=[2],
                message_created_at="2026-01-02T00:00:00Z",
            ),
        )
        rejected = await is_rejected_duplicate(
            db_session,
            new_doc,
            test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        assert rejected is SemanticRejectionResult.REJECTED
        surviving = (
            await db_session.execute(
                select(models.Document).where(
                    models.Document.workspace_name == test_workspace.name,
                    models.Document.observer == test_peer.name,
                    models.Document.observed == test_peer2.name,
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()
        assert surviving.times_derived == 2

    @pytest.mark.asyncio
    async def test_duplicate_replacement_carries_count_forward(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """When a new duplicate wins, it must inherit the replaced doc's count + 1
        rather than resetting reinforcement to 1."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="eri loves cats",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=3,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # More information -> new wins -> existing is soft-deleted.
        new_doc = schemas.DocumentCreate(
            content="eri loves cats and dogs",
            embedding=[0.5] * 1536,
            session_name=test_session.name,
            times_derived=1,
            metadata=schemas.DocumentMetadata(
                message_ids=[2],
                message_created_at="2026-01-02T00:00:00Z",
            ),
        )
        rejected = await is_rejected_duplicate(
            db_session,
            new_doc,
            test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        assert rejected is SemanticRejectionResult.REPLACED_EXISTING
        # Count carried forward onto the replacement (3 -> 4), not reset to 1.
        assert new_doc.times_derived == 4
        live = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == test_workspace.name,
                        models.Document.observer == test_peer.name,
                        models.Document.observed == test_peer2.name,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        # Original is soft-deleted; replacement isn't inserted until create_documents runs.
        assert len(live) == 0

    @pytest.mark.asyncio
    async def test_exact_dedup_within_batch_drops_repeat(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Exact (case/whitespace-insensitive) duplicates within a single batch
        collapse to one document, even with semantic dedup disabled."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        # Three "exact" matches that differ only by case/surrounding whitespace.
        doc_schemas = [
            schemas.DocumentCreate(
                content="User likes coffee",
                embedding=[0.1] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1],
                    message_created_at="2026-01-01T00:00:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="user likes coffee",
                embedding=[0.2] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[2],
                    message_created_at="2026-01-01T00:01:00Z",
                ),
            ),
            schemas.DocumentCreate(
                content="  User likes coffee\n",
                embedding=[0.3] * 1536,
                session_name=test_session.name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[3],
                    message_created_at="2026-01-01T00:02:00Z",
                ),
            ),
        ]

        result = await crud.create_documents(
            db_session,
            documents=doc_schemas,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )
        accepted = result.created_documents

        assert len(accepted) == 1
        assert result.exact_dup_in_batch_count == 2
        assert result.exact_dup_existing_count == 0
        assert result.semantic_dup_rejected_count == 0
        assert result.semantic_dup_replaced_count == 0
        live = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == test_workspace.name,
                        models.Document.observer == test_peer.name,
                        models.Document.observed == test_peer2.name,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(live) == 1
        # Within-batch repeats are dropped silently, no reinforcement.
        assert live[0].times_derived == 1

    @pytest.mark.asyncio
    async def test_exact_dedup_against_existing_reinforces(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """An exact match of an existing live document is rejected and reinforces
        the existing row, even with semantic dedup disabled."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="User likes coffee",
                    embedding=[0.1] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )

        # Case/whitespace variant of the existing content -> exact match.
        result = await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="user likes coffee ",
                    embedding=[0.9] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[2],
                        message_created_at="2026-01-02T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )
        accepted = result.created_documents

        assert len(accepted) == 0
        assert result.exact_dup_existing_count == 1
        assert result.exact_dup_in_batch_count == 0
        assert result.semantic_dup_rejected_count == 0
        assert result.semantic_dup_replaced_count == 0
        surviving = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == test_workspace.name,
                        models.Document.observer == test_peer.name,
                        models.Document.observed == test_peer2.name,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(surviving) == 1
        assert surviving[0].content == "User likes coffee"
        assert surviving[0].times_derived == 2

    @pytest.mark.asyncio
    async def test_exact_dedup_honors_incoming_times_derived(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Reinforcement folds in an incoming doc that already carries
        accumulated reinforcement: the existing row becomes
        ``greatest(existing + 1, incoming)``."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        async def _live() -> list[models.Document]:
            return list(
                (
                    await db_session.execute(
                        select(models.Document).where(
                            models.Document.workspace_name == test_workspace.name,
                            models.Document.observer == test_peer.name,
                            models.Document.observed == test_peer2.name,
                            models.Document.deleted_at.is_(None),
                        )
                    )
                )
                .scalars()
                .all()
            )

        # Existing row already reinforced twice.
        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="User likes coffee",
                    embedding=[0.1] * 1536,
                    session_name=test_session.name,
                    times_derived=2,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )

        # Incoming exact match claims more accumulated reinforcement (5) than
        # existing + 1 (3) -> incoming wins.
        accepted = (
            await crud.create_documents(
                db_session,
                [
                    schemas.DocumentCreate(
                        content="user likes coffee ",
                        embedding=[0.9] * 1536,
                        session_name=test_session.name,
                        times_derived=5,
                        metadata=schemas.DocumentMetadata(
                            message_ids=[2],
                            message_created_at="2026-01-02T00:00:00Z",
                        ),
                    )
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                deduplicate=False,
            )
        ).created_documents
        assert len(accepted) == 0
        live = await _live()
        assert len(live) == 1
        assert live[0].times_derived == 5

        # A normal re-derivation (times_derived defaults to 1) now bumps by one:
        # greatest(existing + 1, 1) -> existing + 1.
        accepted = (
            await crud.create_documents(
                db_session,
                [
                    schemas.DocumentCreate(
                        content="USER LIKES COFFEE",
                        embedding=[0.4] * 1536,
                        session_name=test_session.name,
                        metadata=schemas.DocumentMetadata(
                            message_ids=[3],
                            message_created_at="2026-01-03T00:00:00Z",
                        ),
                    )
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                deduplicate=False,
            )
        ).created_documents
        assert len(accepted) == 0
        live = await _live()
        assert len(live) == 1
        assert live[0].times_derived == 6

    @pytest.mark.asyncio
    async def test_exact_dedup_flushes_before_semantic_replacement(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """An exact-match reinforcement in a batch must be visible to a later
        semantic replacement of the same existing row when autoflush is off."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="User likes coffee",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=False,
        )

        db_session.autoflush = False
        result = await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content=" user likes coffee ",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[2],
                        message_created_at="2026-01-02T00:00:00Z",
                    ),
                ),
                schemas.DocumentCreate(
                    content="User likes coffee and tea",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[3],
                        message_created_at="2026-01-03T00:00:00Z",
                    ),
                ),
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )
        accepted = result.created_documents

        assert len(accepted) == 1
        assert accepted[0].content == "User likes coffee and tea"
        assert result.exact_dup_existing_count == 1
        assert result.semantic_dup_replaced_count == 1
        assert result.exact_dup_in_batch_count == 0
        assert result.semantic_dup_rejected_count == 0

        surviving = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == test_workspace.name,
                        models.Document.observer == test_peer.name,
                        models.Document.observed == test_peer2.name,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(surviving) == 1
        assert surviving[0].content == "User likes coffee and tea"
        assert surviving[0].times_derived == 3

    @pytest.mark.asyncio
    async def test_semantic_dedup_rejected_counts(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A semantically-similar doc with less information than the existing one
        is rejected, and the rejection is counted on the result."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="eri loves cats and dogs and birds and snakes",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[1],
                        message_created_at="2026-01-01T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        # Fewer unique tokens -> existing wins -> new doc is rejected.
        result = await crud.create_documents(
            db_session,
            [
                schemas.DocumentCreate(
                    content="eri loves cats",
                    embedding=[0.5] * 1536,
                    session_name=test_session.name,
                    times_derived=1,
                    metadata=schemas.DocumentMetadata(
                        message_ids=[2],
                        message_created_at="2026-01-02T00:00:00Z",
                    ),
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
            deduplicate=True,
        )

        assert len(result.created_documents) == 0
        assert result.semantic_dup_rejected_count == 1
        assert result.exact_dup_in_batch_count == 0
        assert result.exact_dup_existing_count == 0
        assert result.semantic_dup_replaced_count == 0

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
        created_documents = (
            await crud.create_documents(
                db_session,
                documents=doc_schemas,
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        ).created_documents

        assert len(created_documents) == 2

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
    async def test_create_observations_embeds_with_truncate_on_oversize(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """API conclusion creates must opt into truncation on oversize content."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session, _ = await self._setup_test_data(
            db_session, test_workspace, test_peer
        )

        with patch(
            "src.crud.document.embedding_client.simple_batch_embed",
            new=AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536]),
        ) as mock_embed:
            created = await crud.create_observations(
                db_session,
                observations=[
                    schemas.ConclusionCreate(
                        content="short conclusion",
                        observer_id=test_peer.name,
                        observed_id=test_peer2.name,
                        session_id=test_session.name,
                    ),
                    schemas.ConclusionCreate(
                        content="another conclusion",
                        observer_id=test_peer.name,
                        observed_id=test_peer2.name,
                        session_id=test_session.name,
                    ),
                ],
                workspace_name=test_workspace.name,
            )

        assert len(created) == 2
        mock_embed.assert_awaited_once_with(
            ["short conclusion", "another conclusion"], on_oversize="truncate"
        )


class TestSessionPurityInvariant:
    """Regression tests for the explicit-document session-purity invariant.

    Explicit documents are session-pure records of what was derived from one
    session's messages (the Scopes copy-by-session model depends on this):

    - an explicit document must always carry a non-null session_name
    - dedup/merge (exact and semantic) must never cross document levels
    - dedup/merge must never cross sessions for explicit documents
    """

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Session]:
        """Create an observed peer, two sessions, and the collection."""
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add(test_peer2)
        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([session_a, session_b])
        await db_session.flush()

        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.flush()
        return test_peer2, session_a, session_b

    def _doc(
        self,
        content: str,
        *,
        session_name: str | None,
        level: str = "explicit",
        message_id: int = 1,
    ) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=[0.1] * 1536,
            session_name=session_name,
            level=level,  # pyright: ignore[reportArgumentType]
            metadata=schemas.DocumentMetadata(
                message_ids=[message_id],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    async def _live_docs(
        self,
        db_session: AsyncSession,
        workspace_name: str,
        observer: str,
        observed: str,
    ) -> list[models.Document]:
        return list(
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == observer,
                        models.Document.observed == observed,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

    @pytest.mark.asyncio
    async def test_explicit_without_session_is_refused(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """An explicit document with session_name=None must not be written;
        derived levels remain allowed without a session (dream output)."""
        test_workspace, test_peer = sample_data
        test_peer2, _, _ = await self._setup(db_session, test_workspace, test_peer)

        accepted = (
            await crud.create_documents(
                db_session,
                [
                    self._doc("Global explicit fact", session_name=None),
                    self._doc(
                        "Dream-derived conclusion",
                        session_name=None,
                        level="deductive",
                        message_id=2,
                    ),
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        ).created_documents

        assert [d.content for d in accepted] == ["Dream-derived conclusion"]
        live = await self._live_docs(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        assert len(live) == 1
        assert live[0].level == "deductive"

    @pytest.mark.asyncio
    async def test_exact_dedup_never_merges_explicit_across_sessions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """The same explicit fact stated in two sessions produces two
        session-pure documents; the other session's row is not reinforced."""
        test_workspace, test_peer = sample_data
        test_peer2, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [self._doc("User likes coffee", session_name=session_a.name)],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        accepted = (
            await crud.create_documents(
                db_session,
                [
                    self._doc(
                        "user likes coffee ", session_name=session_b.name, message_id=2
                    )
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        ).created_documents

        assert len(accepted) == 1
        live = await self._live_docs(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        assert len(live) == 2
        assert {doc.session_name for doc in live} == {session_a.name, session_b.name}
        assert all(doc.times_derived == 1 for doc in live)

    @pytest.mark.asyncio
    async def test_exact_dedup_never_merges_across_levels(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """An explicit fact must not be dropped/reinforced against a derived
        document that happens to share its content."""
        test_workspace, test_peer = sample_data
        test_peer2, session_a, _ = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                self._doc(
                    "User likes coffee", session_name=session_a.name, level="deductive"
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        accepted = (
            await crud.create_documents(
                db_session,
                [
                    self._doc(
                        "User likes coffee", session_name=session_a.name, message_id=2
                    )
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        ).created_documents

        assert len(accepted) == 1
        live = await self._live_docs(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        assert len(live) == 2
        assert {doc.level for doc in live} == {"explicit", "deductive"}
        assert all(doc.times_derived == 1 for doc in live)

    @pytest.mark.asyncio
    async def test_exact_dedup_still_merges_derived_levels_across_sessions(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Derived levels are consolidations, not session-pure records:
        cross-session exact dedup still reinforces the existing row."""
        test_workspace, test_peer = sample_data
        test_peer2, session_a, session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        await crud.create_documents(
            db_session,
            [
                self._doc(
                    "Probably a morning person",
                    session_name=session_a.name,
                    level="deductive",
                )
            ],
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        accepted = (
            await crud.create_documents(
                db_session,
                [
                    self._doc(
                        "probably a morning person",
                        session_name=session_b.name,
                        level="deductive",
                        message_id=2,
                    )
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        ).created_documents

        assert len(accepted) == 0
        live = await self._live_docs(
            db_session, test_workspace.name, test_peer.name, test_peer2.name
        )
        assert len(live) == 1
        assert live[0].times_derived == 2

    @pytest.mark.asyncio
    async def test_semantic_dedup_scoped_to_level_and_session(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """is_rejected_duplicate must constrain candidate search to the same
        level, and to the same session for explicit documents."""
        test_workspace, test_peer = sample_data
        test_peer2, session_a, _ = await self._setup(
            db_session, test_workspace, test_peer
        )

        explicit_doc = self._doc("User likes coffee", session_name=session_a.name)
        with patch(
            "src.crud.document.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            rejected = await is_rejected_duplicate(
                db_session,
                explicit_doc,
                test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        assert rejected is SemanticRejectionResult.NOT_DUPLICATE
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {
            "level": "explicit",
            "session_name": session_a.name,
        }

        deductive_doc = self._doc(
            "User likes coffee", session_name=None, level="deductive"
        )
        with patch(
            "src.crud.document.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            rejected = await is_rejected_duplicate(
                db_session,
                deductive_doc,
                test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        assert rejected is SemanticRejectionResult.NOT_DUPLICATE
        assert mock_query.await_args is not None
        assert mock_query.await_args.kwargs["filters"] == {"level": "deductive"}

    @pytest.mark.asyncio
    async def test_semantic_dedup_refuses_sessionless_explicit(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A session-less explicit document has no valid merge partner: it is
        never treated as a duplicate and no candidate search runs."""
        test_workspace, test_peer = sample_data
        test_peer2, _, _ = await self._setup(db_session, test_workspace, test_peer)

        doc = self._doc("User likes coffee", session_name=None)
        with patch(
            "src.crud.document.query_documents", new=AsyncMock(return_value=[])
        ) as mock_query:
            rejected = await is_rejected_duplicate(
                db_session,
                doc,
                test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )
        assert rejected is SemanticRejectionResult.NOT_DUPLICATE
        mock_query.assert_not_awaited()


class TestCreateDocumentsConcurrency:
    """Concurrent same-collection reinforcements lock rows in id order."""

    N_DOCS: int = 20
    N_ROUNDS: int = 5

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session]:
        """Create an observed peer, session, and collection, committed so
        they are visible to independent concurrent sessions."""
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_session])
        await db_session.flush()
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.commit()
        return test_peer2, test_session

    def _batch(self, session_name: str) -> list[schemas.DocumentCreate]:
        return [
            schemas.DocumentCreate(
                content=f"user fact number {i}",
                embedding=[0.1] * 1536,
                session_name=session_name,
                metadata=schemas.DocumentMetadata(
                    message_ids=[i],
                    message_created_at="2026-01-01T00:00:00Z",
                ),
            )
            for i in range(self.N_DOCS)
        ]

    @staticmethod
    def _chain(exc: BaseException) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        e: BaseException | None = exc
        while e is not None and id(e) not in seen:
            seen.add(id(e))
            parts.append(f"{type(e).__name__}: {e}")
            e = e.__cause__ or e.__context__
        return " <- ".join(parts)

    @pytest.mark.asyncio
    async def test_concurrent_reinforcement_does_not_deadlock(
        self,
        db_engine: "AsyncEngine",
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Opposing-order batches on one collection must not deadlock."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )

        # Seed the rows both writers will reinforce.
        await crud.create_documents(
            db_session,
            self._batch(test_session.name),
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )

        session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

        for round_num in range(self.N_ROUNDS):
            forward = self._batch(test_session.name)
            backward = list(reversed(self._batch(test_session.name)))

            async def _run(batch: list[schemas.DocumentCreate]) -> None:
                async with session_factory() as db:
                    await crud.create_documents(
                        db,
                        batch,
                        workspace_name=test_workspace.name,
                        observer=test_peer.name,
                        observed=test_peer2.name,
                    )

            results = await asyncio.gather(
                _run(forward), _run(backward), return_exceptions=True
            )
            errors = [r for r in results if isinstance(r, BaseException)]
            assert not errors, (
                f"round {round_num}: concurrent create_documents failed: "
                + "; ".join(self._chain(e) for e in errors)
            )

        # Every round reinforced the same rows: 1 seed + 2 per round.
        docs = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == test_workspace.name,
                        models.Document.observer == test_peer.name,
                        models.Document.observed == test_peer2.name,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(docs) == self.N_DOCS
        assert all(d.times_derived == 1 + 2 * self.N_ROUNDS for d in docs)


class TestCreateDocumentsErrorHandling:
    """A dead transaction aborts the batch; per-document failures skip one document."""

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session]:
        test_peer2 = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([test_peer2, test_session])
        await db_session.flush()
        collection = models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=test_peer2.name,
        )
        db_session.add(collection)
        await db_session.commit()
        return test_peer2, test_session

    def _doc(self, content: str, session_name: str) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=[0.1] * 1536,
            session_name=session_name,
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    @pytest.mark.asyncio
    async def test_db_error_on_row_update_flush_aborts_batch(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A DB error while applying row updates raises and commits nothing."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        # Plain strings: the rollback below expires ORM objects in the session.
        workspace_name = test_workspace.name
        observer = test_peer.name
        observed = test_peer2.name
        session_name = test_session.name

        await crud.create_documents(
            db_session,
            [self._doc("existing fact", session_name)],
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )

        class FakePGError(Exception):
            sqlstate: str = "40P01"

        deadlock = OperationalError("UPDATE documents", {}, FakePGError())
        with (
            patch.object(db_session, "flush", AsyncMock(side_effect=deadlock)),
            pytest.raises(OperationalError),
        ):
            await crud.create_documents(
                db_session,
                [
                    self._doc("existing fact", session_name),
                    self._doc("a brand new fact", session_name),
                ],
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )

        docs = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == observer,
                        models.Document.observed == observed,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert [d.content for d in docs] == ["existing fact"]
        assert docs[0].times_derived == 1

    @pytest.mark.asyncio
    async def test_db_error_resolving_dup_candidates_aborts_batch(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A DB error while resolving dup candidates raises and commits nothing."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        workspace_name = test_workspace.name
        observer = test_peer.name
        observed = test_peer2.name
        session_name = test_session.name

        class FakePGError(Exception):
            sqlstate: str = "40P01"

        deadlock = OperationalError("SELECT documents", {}, FakePGError())
        with (
            patch(
                "src.crud.document._pgvector_dup_candidates",
                AsyncMock(side_effect=deadlock),
            ),
            pytest.raises(OperationalError),
        ):
            await crud.create_documents(
                db_session,
                [
                    self._doc("a brand new fact", session_name),
                    self._doc("another new fact", session_name),
                ],
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                deduplicate=True,
            )

        docs = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == observer,
                        models.Document.observed == observed,
                    )
                )
            )
            .scalars()
            .all()
        )
        assert docs == []

    @pytest.mark.asyncio
    async def test_per_document_error_still_skips_only_that_document(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Non-DB per-document failures keep their skip semantics."""
        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )

        from src.crud import document as document_module

        real_dedup_key = document_module._dedup_key  # pyright: ignore[reportPrivateUsage]

        def flaky_dedup_key(
            content: str, level: str, session_name: str | None
        ) -> tuple[str, str, str | None]:
            if content == "poison":
                raise ValueError("bad content")
            return real_dedup_key(content, level, session_name)

        with patch.object(document_module, "_dedup_key", flaky_dedup_key):
            result = await crud.create_documents(
                db_session,
                [
                    self._doc("good fact one", test_session.name),
                    self._doc("poison", test_session.name),
                    self._doc("good fact two", test_session.name),
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
            )

        assert sorted(d.content for d in result.created_documents) == [
            "good fact one",
            "good fact two",
        ]

    @pytest.mark.asyncio
    async def test_empty_embedding_skips_semantic_without_embed(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Empty embeddings must not trigger embed() under an open session."""
        from src.config import settings

        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "pgvector")
        monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

        empty = self._doc("fact without vector", test_session.name)
        empty.embedding = []

        with patch(
            "src.crud.document.embedding_client.embed",
            new_callable=AsyncMock,
        ) as mock_embed:
            result = await crud.create_documents(
                db_session,
                [empty],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=test_peer2.name,
                deduplicate=True,
            )

        assert len(result.created_documents) == 1
        mock_embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_reinforce_target_falls_back_to_insert(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """If a reinforce target vanishes under lock, insert the incoming doc."""
        from src.crud import document as document_module

        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        workspace_name = test_workspace.name
        observer = test_peer.name
        observed = test_peer2.name
        session_name = test_session.name

        seeded = await crud.create_documents(
            db_session,
            [self._doc("shared fact", session_name)],
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        assert len(seeded.created_documents) == 1

        existing = (
            await db_session.execute(
                select(models.Document).where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.observer == observer,
                    models.Document.observed == observed,
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()

        real_apply = document_module._apply_document_row_updates  # pyright: ignore[reportPrivateUsage]

        async def delete_then_apply(*args: Any, **kwargs: Any) -> Any:
            existing.deleted_at = datetime.datetime.now(datetime.UTC)
            await db_session.flush()
            return await real_apply(*args, **kwargs)

        with patch.object(
            document_module,
            "_apply_document_row_updates",
            side_effect=delete_then_apply,
        ):
            result = await crud.create_documents(
                db_session,
                [self._doc("shared fact", session_name)],
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
            )

        assert len(result.created_documents) == 1
        live = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == observer,
                        models.Document.observed == observed,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(live) == 1
        assert live[0].id != existing.id
        assert live[0].content == "shared fact"

    @pytest.mark.asyncio
    async def test_same_batch_replace_then_reinforce_does_not_resurrect(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A reinforce after a same-batch replace must not insert the inferior copy."""
        from src.crud import document as document_module

        test_workspace, test_peer = sample_data
        test_peer2, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        workspace_name = test_workspace.name
        observer = test_peer.name
        observed = test_peer2.name
        session_name = test_session.name

        await crud.create_documents(
            db_session,
            [self._doc("shared fact", session_name)],
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        existing = (
            await db_session.execute(
                select(models.Document).where(
                    models.Document.workspace_name == workspace_name,
                    models.Document.observer == observer,
                    models.Document.observed == observed,
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one()

        fallback = self._doc("shared fact", session_name)
        ops = [
            document_module._DocumentRowOp("replace", existing.id),  # pyright: ignore[reportPrivateUsage]
            document_module._DocumentRowOp(  # pyright: ignore[reportPrivateUsage]
                "reinforce",
                existing.id,
                fallback_document=fallback,
            ),
        ]
        fallbacks = await document_module._apply_document_row_updates(  # pyright: ignore[reportPrivateUsage]
            db_session,
            ops,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
        )
        assert fallbacks == []
        await db_session.commit()
        live = (
            (
                await db_session.execute(
                    select(models.Document).where(
                        models.Document.workspace_name == workspace_name,
                        models.Document.observer == observer,
                        models.Document.observed == observed,
                        models.Document.deleted_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        assert live == []


class TestExternalCandidateHoist:
    """External-store dup candidates resolve before the first DB statement."""

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session]:
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed_peer, test_session])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
            )
        )
        await db_session.commit()
        return observed_peer, test_session

    def _doc(self, content: str, session_name: str) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=[0.1] * 1536,
            session_name=session_name,
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    @pytest.mark.asyncio
    async def test_external_candidates_resolved_before_db(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        from src.config import settings

        test_workspace, test_peer = sample_data
        observed_peer, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")
        monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

        events: list[str] = []
        real_execute = db_session.execute

        async def spying_execute(statement: Any, *args: Any, **kwargs: Any) -> Any:
            events.append("execute")
            return await real_execute(statement, *args, **kwargs)

        async def fake_resolve(*_args: Any, **_kwargs: Any) -> list[str]:
            events.append("resolve")
            return []

        with (
            patch.object(db_session, "execute", side_effect=spying_execute),
            patch(
                "src.crud.document.query_external_vector_document_ids",
                side_effect=fake_resolve,
            ),
            patch(
                "src.crud.document.get_external_vector_store",
                return_value=None,
            ),
        ):
            result = await crud.create_documents(
                db_session,
                [
                    self._doc("fact one", test_session.name),
                    self._doc("fact two", test_session.name),
                ],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
                deduplicate=True,
            )

        assert len(result.created_documents) == 2
        assert events[:2] == ["resolve", "resolve"]
        assert "execute" in events

    @pytest.mark.asyncio
    async def test_resolve_failure_skips_semantic_without_query_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
    ):
        from src.config import settings

        test_workspace, test_peer = sample_data
        observed_peer, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )
        monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")
        monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

        with (
            patch(
                "src.crud.document.query_external_vector_document_ids",
                side_effect=RuntimeError("store down"),
            ),
            patch(
                "src.crud.document.get_external_vector_store",
                return_value=None,
            ),
            patch(
                "src.crud.document.query_documents",
                new_callable=AsyncMock,
            ) as mock_query,
        ):
            result = await crud.create_documents(
                db_session,
                [self._doc("fact one", test_session.name)],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
                deduplicate=True,
            )

        assert len(result.created_documents) == 1
        mock_query.assert_not_awaited()


class TestCreateDocumentsQueryCount:
    """Semantic dedup costs a fixed number of queries, not one per document."""

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session]:
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        test_session = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed_peer, test_session])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
            )
        )
        db_session.add(
            models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
                session_name=test_session.name,
                content="the user drinks coffee",
                embedding=[0.1] * 1536,
                level="explicit",
                internal_metadata={},
            )
        )
        await db_session.commit()
        return observed_peer, test_session

    def _doc(self, content: str, session_name: str) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=[0.1] * 1536,
            session_name=session_name,
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    async def _count_queries(
        self,
        db_session: AsyncSession,
        *,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str,
        batch_size: int,
        tag: str,
    ) -> int:
        # Contents are tagged per call: the batches run against one database, and
        # a repeat would land in exact dedup instead of the semantic stage.
        documents = [
            self._doc(
                f"the user drinks coffee every morning, {tag} cup {i}", session_name
            )
            for i in range(batch_size)
        ]
        executed: list[Any] = []
        real_execute = db_session.execute

        async def spying_execute(statement: Any, *args: Any, **kwargs: Any) -> Any:
            executed.append(statement)
            return await real_execute(statement, *args, **kwargs)

        with patch.object(db_session, "execute", side_effect=spying_execute):
            result = await crud.create_documents(
                db_session,
                documents,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                deduplicate=True,
            )

        # Every document must have reached the semantic stage, or a flat query
        # count would only prove the batch was skipped.
        assert (
            result.semantic_dup_replaced_count + result.semantic_dup_rejected_count
            == batch_size
        )
        return len(executed)

    @pytest.mark.asyncio
    async def test_query_count_is_flat_in_batch_size(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        test_workspace, test_peer = sample_data
        observed_peer, test_session = await self._setup(
            db_session, test_workspace, test_peer
        )

        one = await self._count_queries(
            db_session,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed_peer.name,
            session_name=test_session.name,
            batch_size=1,
            tag="first",
        )
        many = await self._count_queries(
            db_session,
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed_peer.name,
            session_name=test_session.name,
            batch_size=8,
            tag="second",
        )

        assert many == one, (
            f"dedup queries scale with batch size ({one} for 1 document, "
            f"{many} for 8): the per-document candidate lookup is back"
        )


class TestPrefetchedCandidateScope:
    """Prefetched candidates are re-scoped in Python before they can merge.

    Candidates for the whole batch are fetched in one query, so the per-document
    merge scope can no longer be a SQL filter. A candidate the vector store
    returns out of scope must not merge — for explicit documents that would
    breach session purity.
    """

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Session, models.Document]:
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed_peer, session_a, session_b])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
            )
        )
        existing = models.Document(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed_peer.name,
            session_name=session_a.name,
            content="the user drinks coffee",
            embedding=[0.1] * 1536,
            level="explicit",
            internal_metadata={},
        )
        db_session.add(existing)
        await db_session.commit()
        return observed_peer, session_a, session_b, existing

    def _doc(self, content: str, session_name: str) -> schemas.DocumentCreate:
        return schemas.DocumentCreate(
            content=content,
            embedding=[0.1] * 1536,
            session_name=session_name,
            metadata=schemas.DocumentMetadata(
                message_ids=[1],
                message_created_at="2026-01-01T00:00:00Z",
            ),
        )

    @pytest.mark.parametrize(
        ("incoming_session", "expect_merge"),
        [("same", True), ("other", False)],
    )
    @pytest.mark.asyncio
    async def test_out_of_scope_candidate_does_not_merge(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        monkeypatch: pytest.MonkeyPatch,
        incoming_session: str,
        expect_merge: bool,
    ):
        from src.config import settings

        test_workspace, test_peer = sample_data
        observed_peer, session_a, session_b, existing = await self._setup(
            db_session, test_workspace, test_peer
        )
        monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")
        monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", True)

        session_name = session_a.name if incoming_session == "same" else session_b.name

        # The store hands back the session A document whatever scope was asked
        # for, so the decision rests entirely on the in-Python re-scoping.
        async def fake_resolve(*_args: Any, **_kwargs: Any) -> list[str]:
            return [existing.id]

        with (
            patch(
                "src.crud.document.query_external_vector_document_ids",
                side_effect=fake_resolve,
            ),
            patch(
                "src.crud.document.get_external_vector_store",
                return_value=None,
            ),
        ):
            result = await crud.create_documents(
                db_session,
                [self._doc("the user drinks coffee every morning", session_name)],
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
                deduplicate=True,
            )

        assert result.semantic_dup_replaced_count == (1 if expect_merge else 0)

        await db_session.refresh(existing)
        assert (existing.deleted_at is not None) is expect_merge


class TestPgvectorCandidateEquivalence:
    """The batched candidate query matches the per-document query it replaced.

    `_pgvector_dup_candidates` unions N nearest-neighbour searches into one
    statement. `_query_documents_pgvector` — still the live path for ordinary
    semantic queries — is the reference: for the same document the two must
    agree on scope, distance cutoff and which row comes back.
    """

    DIM: int = 1536

    def _vec(self, axis: int, tilt: float = 0.0) -> list[float]:
        """Unit vector on `axis`, optionally tilted toward a far-off axis.

        Cosine distance from the untilted vector is ``1 - 1/sqrt(1 + tilt**2)``:
        0.0 at tilt 0, ~0.005 at 0.1, ~0.001 at 0.05, ~0.106 at 0.5. That last
        one sits outside the 0.05 dedup cutoff.
        """
        vector = [0.0] * self.DIM
        vector[axis] = 1.0
        if tilt:
            vector[axis + 500] = tilt
        return vector

    async def _setup(
        self,
        db_session: AsyncSession,
        test_workspace: models.Workspace,
        test_peer: models.Peer,
    ) -> tuple[models.Peer, models.Session, models.Session]:
        observed_peer = models.Peer(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_a = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        session_b = models.Session(
            name=str(generate_nanoid()), workspace_name=test_workspace.name
        )
        db_session.add_all([observed_peer, session_a, session_b])
        await db_session.flush()
        db_session.add(
            models.Collection(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
            )
        )

        def existing(
            axis: int,
            tilt: float,
            level: DocumentLevel,
            session: models.Session | None,
            *,
            deleted: bool = False,
        ) -> models.Document:
            return models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed_peer.name,
                session_name=session.name if session else None,
                content=f"existing axis={axis} tilt={tilt} level={level}",
                embedding=self._vec(axis, tilt),
                level=level,
                internal_metadata={},
                deleted_at=datetime.datetime.now(datetime.UTC) if deleted else None,
            )

        db_session.add_all(
            [
                # One in-scope near neighbour per axis, so a mis-mapped batch
                # index sends a document to the wrong row.
                existing(0, 0.1, "explicit", session_a),
                existing(1, 0.1, "explicit", session_a),
                existing(2, 0.1, "deductive", None),
                # Nearer than the axis-0 match but in another session: only the
                # scope filter keeps it from winning.
                existing(0, 0.05, "explicit", session_b),
                # Nearer than the axis-1 match but a different level.
                existing(1, 0.05, "deductive", None),
                # Axis 3's only neighbour sits outside the distance cutoff.
                existing(3, 0.5, "explicit", session_a),
                # Soft-deleted, and otherwise the closest thing to axis 4.
                existing(4, 0.1, "explicit", session_a, deleted=True),
            ]
        )
        await db_session.commit()
        return observed_peer, session_a, session_b

    @pytest.mark.parametrize("order", ["authored", "reversed"])
    @pytest.mark.asyncio
    async def test_batched_candidates_match_per_document_query(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        order: str,
    ):
        """Every document's candidates must match, whatever order they arrive in.

        Parametrising over batch *shape* is what earns its keep here. Splitting
        the cases into one document per run would not: a single-document batch
        makes the index mapping trivially correct, so a permuted or constant
        `batch_index` could never be observed.
        """
        from src.crud import document as document_module

        test_workspace, test_peer = sample_data
        observed_peer, session_a, _session_b = await self._setup(
            db_session, test_workspace, test_peer
        )

        def incoming(
            axis: int, level: DocumentLevel, session: str | None
        ) -> schemas.DocumentCreate:
            return schemas.DocumentCreate(
                content=f"incoming axis={axis} level={level} session={session}",
                embedding=self._vec(axis),
                session_name=session,
                level=level,
                metadata=schemas.DocumentMetadata(
                    message_ids=[1],
                    message_created_at="2026-01-01T00:00:00Z",
                ),
            )

        # A document with no embedding, and a session-less explicit one, are both
        # skipped when the legs are built. They are interleaved with matches on
        # purpose: leg position and document index only diverge after a skip, and
        # a `batch_index` taken from the wrong one would go unnoticed without them.
        # Both sit on axis 0, which has near neighbours, so dropping either skip
        # surfaces a spurious candidate instead of the empty result a barren axis
        # would return either way.
        no_embedding = incoming(0, "explicit", session_a.name)
        no_embedding.embedding = []
        sessionless = incoming(0, "explicit", None)

        documents = [
            incoming(0, "explicit", session_a.name),
            no_embedding,
            incoming(1, "explicit", session_a.name),
            sessionless,
            incoming(2, "deductive", None),
            incoming(3, "explicit", session_a.name),  # nearest is out of range
            incoming(4, "explicit", session_a.name),  # nearest is soft-deleted
            incoming(5, "explicit", session_a.name),  # no neighbour at all
        ]
        if order == "reversed":
            documents.reverse()

        expected: list[list[str]] = []
        skipped: list[int] = []
        for index, doc in enumerate(documents):
            filters = document_module._semantic_dup_filters(doc)  # pyright: ignore[reportPrivateUsage]
            if filters is None or not doc.embedding:
                # No merge scope or no vector: the document cannot have a
                # candidate, and contributes no leg to the batched query.
                expected.append([])
                skipped.append(index)
                continue
            reference = await document_module._query_documents_pgvector(  # pyright: ignore[reportPrivateUsage]
                db_session,
                test_workspace.name,
                test_peer.name,
                observed_peer.name,
                doc.embedding,
                filters,
                document_module._SEMANTIC_DUP_MAX_DISTANCE,  # pyright: ignore[reportPrivateUsage]
                document_module._SEMANTIC_DUP_TOP_K,  # pyright: ignore[reportPrivateUsage]
            )
            expected.append([row.id for row in reference])

        actual = await document_module._pgvector_dup_candidates(  # pyright: ignore[reportPrivateUsage]
            db_session,
            documents,
            test_workspace.name,
            observer=test_peer.name,
            observed=observed_peer.name,
        )

        assert actual == expected

        # The corpus has to keep discriminating, or two equally broken
        # implementations would agree with each other.
        matched = [index for index, ids in enumerate(expected) if ids]
        assert len(matched) == 3, (
            f"expected three documents to match, got {len(matched)}: {expected}"
        )
        assert len({expected[index][0] for index in matched}) == 3, (
            "the three matches must be three distinct rows, or a permuted "
            "batch index would go unnoticed"
        )
        assert len(expected) - len(matched) - len(skipped) == 3, (
            "expected three documents that resolve no candidate without being "
            "skipped: out of range, soft-deleted, and no neighbour at all"
        )
        assert min(skipped) < max(matched), (
            "a skipped document must sit before a matching one, or leg position "
            "and document index would never diverge"
        )
