"""
Tests for the immediate message-embedding fast path (src/reconciler/embed_now.py).

These exercise embed_messages_now end-to-end against the test database: it opens
its own tracked_db sessions (patched to the test engine in conftest), so each test
creates committed fixture rows and asserts on the result via the provided session.
"""

from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import models
from src.reconciler.embed_now import embed_messages_now, reset_embed_semaphore
from src.vector_store import VectorStore


async def _create_message_with_pending_chunks(
    db_session: AsyncSession,
    workspace: models.Workspace,
    peer: models.Peer,
    chunk_contents: list[str],
) -> tuple[str, list[int]]:
    """Create a message plus one pending MessageEmbedding row per chunk.

    Returns (message public_id, ordered embedding row ids).
    """
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()

    message_id = str(generate_nanoid())
    message = models.Message(
        public_id=message_id,
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=peer.name,
        content=" ".join(chunk_contents),
        seq_in_session=1,
    )
    db_session.add(message)
    await db_session.commit()

    rows = [
        models.MessageEmbedding(
            content=chunk,
            message_id=message_id,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            sync_state="pending",
            embedding=None,
        )
        for chunk in chunk_contents
    ]
    db_session.add_all(rows)
    await db_session.commit()
    for row in rows:
        await db_session.refresh(row)
    return message_id, [row.id for row in rows]


@pytest.fixture(autouse=True)
def reset_semaphore_fixture():
    """Rebuild the module semaphore per test so it binds to the active loop."""
    reset_embed_semaphore()
    yield
    reset_embed_semaphore()


@pytest.mark.asyncio
class TestEmbedMessagesNow:
    async def test_pgvector_happy_path_marks_synced(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """pgvector-only mode: rows get a vector and flip to synced immediately."""
        workspace, peer = sample_data
        message_id, emb_ids = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["hello world"]
        )

        await embed_messages_now([message_id])

        for emb_id in emb_ids:
            row = await db_session.get(models.MessageEmbedding, emb_id)
            assert row is not None
            await db_session.refresh(row)
            assert row.sync_state == "synced"
            assert row.embedding is not None
            assert row.sync_attempts == 0

    async def test_no_message_ids_is_noop(self) -> None:
        """Empty input returns without touching the DB or embedding."""
        with patch(
            "src.embedding_client.embedding_client.simple_batch_embed"
        ) as mock_embed:
            await embed_messages_now([])
        mock_embed.assert_not_called()

    async def test_already_synced_rows_not_reclaimed(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """A second run finds no pending rows and does not re-embed."""
        workspace, peer = sample_data
        message_id, _ = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["first content"]
        )
        await embed_messages_now([message_id])

        with patch(
            "src.embedding_client.embedding_client.simple_batch_embed"
        ) as mock_embed:
            await embed_messages_now([message_id])
        mock_embed.assert_not_called()

    async def test_embed_failure_leaves_rows_pending_and_leased(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> None:
        """Embedding failure must leave rows pending + leased, attempts untouched,
        so the reconciler owns retry accounting."""
        workspace, peer = sample_data
        message_id, emb_ids = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["will fail"]
        )

        with patch(
            "src.embedding_client.embedding_client.simple_batch_embed",
            new=AsyncMock(side_effect=RuntimeError("provider down")),
        ):
            await embed_messages_now([message_id])

        for emb_id in emb_ids:
            row = await db_session.get(models.MessageEmbedding, emb_id)
            assert row is not None
            await db_session.refresh(row)
            assert row.sync_state == "pending"
            assert row.embedding is None
            assert row.sync_attempts == 0  # lease only, no attempt bump
            assert row.last_sync_at is not None  # leased

    async def test_external_store_upserts_with_chunk_positioned_ids(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        mock_vector_store: VectorStore,
    ) -> None:
        """External-store mode: upsert each chunk with id {message_id}_{position}
        and mark rows synced."""
        workspace, peer = sample_data
        message_id, emb_ids = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["chunk a", "chunk b", "chunk c"]
        )

        with patch(
            "src.reconciler.embed_now.get_external_vector_store",
            return_value=mock_vector_store,
        ):
            await embed_messages_now([message_id])

        upsert_mock: AsyncMock = mock_vector_store.upsert_many  # pyright: ignore[reportAssignmentType]
        upsert_mock.assert_awaited()
        upserted_ids = {
            record.id for call in upsert_mock.await_args_list for record in call.args[1]
        }
        assert upserted_ids == {
            f"{message_id}_0",
            f"{message_id}_1",
            f"{message_id}_2",
        }

        for emb_id in emb_ids:
            row = await db_session.get(models.MessageEmbedding, emb_id)
            assert row is not None
            await db_session.refresh(row)
            assert row.sync_state == "synced"

    async def test_locked_chunk_skipped_keeps_positions_stable(
        self,
        db_session: AsyncSession,
        db_engine: AsyncEngine,
        sample_data: tuple[models.Workspace, models.Peer],
        mock_vector_store: VectorStore,
    ) -> None:
        """If a sibling chunk is locked by another txn, SKIP LOCKED skips it but
        chunk positions still come from the full sibling ordering — so the claimed
        chunks keep their {message_id}_0 / _2 ids (not _0 / _1)."""
        workspace, peer = sample_data
        message_id, emb_ids = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["chunk a", "chunk b", "chunk c"]
        )
        locked_id = emb_ids[1]  # middle chunk -> position 1

        # Hold a row lock on the middle chunk from an independent transaction.
        lock_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
        lock_session = lock_factory()
        await lock_session.execute(
            select(models.MessageEmbedding)
            .where(models.MessageEmbedding.id == locked_id)
            .with_for_update()
        )
        try:
            with patch(
                "src.reconciler.embed_now.get_external_vector_store",
                return_value=mock_vector_store,
            ):
                await embed_messages_now([message_id])
        finally:
            await lock_session.rollback()
            await lock_session.close()

        upsert_mock: AsyncMock = mock_vector_store.upsert_many  # pyright: ignore[reportAssignmentType]
        upserted_ids = {
            record.id for call in upsert_mock.await_args_list for record in call.args[1]
        }
        assert upserted_ids == {f"{message_id}_0", f"{message_id}_2"}

        # The locked chunk stays pending; the other two are synced.
        locked_row = await db_session.get(models.MessageEmbedding, locked_id)
        assert locked_row is not None
        await db_session.refresh(locked_row)
        assert locked_row.sync_state == "pending"
        for emb_id in (emb_ids[0], emb_ids[2]):
            row = await db_session.get(models.MessageEmbedding, emb_id)
            assert row is not None
            await db_session.refresh(row)
            assert row.sync_state == "synced"

    async def test_external_store_unavailable_leaves_rows_pending(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
        mock_vector_store: VectorStore,
    ) -> None:
        """External-store mode: if upsert_many raises VectorStoreError, rows must
        stay pending with no vector and untouched attempts, so the reconciler
        heals them. embed_now never bumps sync_attempts."""
        from src.exceptions import VectorStoreError

        workspace, peer = sample_data
        message_id, emb_ids = await _create_message_with_pending_chunks(
            db_session, workspace, peer, ["chunk a", "chunk b"]
        )

        upsert_mock: AsyncMock = mock_vector_store.upsert_many  # pyright: ignore[reportAssignmentType]
        upsert_mock.side_effect = VectorStoreError("vector store down")

        with patch(
            "src.reconciler.embed_now.get_external_vector_store",
            return_value=mock_vector_store,
        ):
            await embed_messages_now([message_id])

        for emb_id in emb_ids:
            row = await db_session.get(models.MessageEmbedding, emb_id)
            assert row is not None
            await db_session.refresh(row)
            assert row.sync_state == "pending"
            assert row.embedding is None
            assert row.sync_attempts == 0  # embed_now never bumps attempts
