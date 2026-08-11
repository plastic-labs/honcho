import datetime
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.crud.document import SemanticRejectionResult, is_rejected_duplicate
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

        base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
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
