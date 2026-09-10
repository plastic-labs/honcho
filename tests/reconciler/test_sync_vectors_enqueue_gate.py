"""The sync_vectors row is only enqueued when a reconciliation cycle has work to do."""

import datetime
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.reconciler import scheduler as scheduler_module
from src.reconciler.scheduler import RECONCILER_TASKS, ReconcilerScheduler
from src.reconciler.sync_vectors import has_pending_work


@pytest.fixture(autouse=True)
def _reset_scheduler_singleton():  # pyright: ignore[reportUnusedFunction]
    ReconcilerScheduler.reset_singleton()
    yield
    ReconcilerScheduler.reset_singleton()


async def _tenant(
    db_session: AsyncSession,
) -> tuple[models.Workspace, models.Peer, models.Session]:
    workspace = models.Workspace(name=str(generate_nanoid()))
    db_session.add(workspace)
    await db_session.commit()
    peer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(peer)
    await db_session.commit()
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(session)
    await db_session.commit()
    return workspace, peer, session


async def test_no_work_when_nothing_pending(db_session: AsyncSession) -> None:
    assert await has_pending_work(db_session) is False


async def test_pending_embedding_is_work(db_session: AsyncSession) -> None:
    workspace, peer, session = await _tenant(db_session)
    message = models.Message(
        public_id=str(generate_nanoid()),
        session_name=session.name,
        workspace_name=workspace.name,
        peer_name=peer.name,
        content="hello",
        seq_in_session=1,
    )
    db_session.add(message)
    await db_session.commit()
    db_session.add(
        models.MessageEmbedding(
            content=message.content,
            message_id=message.public_id,
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=peer.name,
            sync_state="pending",
            embedding=None,
        )
    )
    await db_session.commit()

    assert await has_pending_work(db_session) is True


async def test_soft_deleted_document_is_work_only_after_grace(
    db_session: AsyncSession,
) -> None:
    workspace, peer, session = await _tenant(db_session)
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.commit()
    doc = models.Document(
        content="gone",
        workspace_name=workspace.name,
        observer=peer.name,
        observed=peer.name,
        session_name=session.name,
        sync_state="synced",
        embedding=[0.0] * 1536,
        deleted_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db_session.add(doc)
    await db_session.commit()
    assert await has_pending_work(db_session) is False

    doc.deleted_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        minutes=10
    )
    await db_session.commit()
    assert await has_pending_work(db_session) is True


@pytest.mark.parametrize("has_work", [False, True])
async def test_enqueue_gated_on_pending_work(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch, has_work: bool
) -> None:
    @asynccontextmanager
    async def _db(_: str | None = None) -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    monkeypatch.setattr(scheduler_module, "tracked_db", _db)
    monkeypatch.setattr(
        scheduler_module, "has_pending_work", AsyncMock(return_value=has_work)
    )

    enqueued = await ReconcilerScheduler()._try_enqueue_task(  # pyright: ignore[reportPrivateUsage]
        RECONCILER_TASKS["sync_vectors"]
    )
    rows = (
        (
            await db_session.execute(
                select(models.QueueItem).where(
                    models.QueueItem.work_unit_key == "reconciler:sync_vectors"
                )
            )
        )
        .scalars()
        .all()
    )

    assert enqueued is has_work
    assert len(rows) == (1 if has_work else 0)


async def test_cleanup_queue_is_not_gated(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def _db(_: str | None = None) -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    monkeypatch.setattr(scheduler_module, "tracked_db", _db)
    gate = AsyncMock(return_value=False)
    monkeypatch.setattr(scheduler_module, "has_pending_work", gate)

    enqueued = await ReconcilerScheduler()._try_enqueue_task(  # pyright: ignore[reportPrivateUsage]
        RECONCILER_TASKS["cleanup_queue"]
    )

    assert enqueued is True
    gate.assert_not_awaited()
