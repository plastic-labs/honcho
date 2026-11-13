"""The ingest fence and the ingest rows that session and workspace deletion remove."""

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.config import settings
from src.crud import ingest_fence
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.db import tenant_context
from src.models import Peer, Workspace


@pytest_asyncio.fixture
async def other_db(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A second, independent session, to hold or contend for locks."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()


async def _make_session(db: AsyncSession, workspace: Workspace, peer: Peer) -> str:
    session_name = str(generate_nanoid())
    db.add(models.Session(name=session_name, workspace_name=workspace.name))
    await db.flush()
    db.add(
        models.SessionPeer(
            workspace_name=workspace.name,
            session_name=session_name,
            peer_name=peer.name,
        )
    )
    await db.commit()
    return session_name


def _ingest_rows(
    workspace_name: str, session_name: str, peer_name: str
) -> list[object]:
    return [
        models.IngestRecord(
            workspace_name=workspace_name,
            source="discord",
            source_instance="guild",
            identity_kind="source_id",
            identity=str(generate_nanoid()),
            session_name=session_name,
            content_sha256="0" * 64,
            derivation="pending",
        ),
        models.IngestSessionBinding(
            workspace_name=workspace_name,
            source="discord",
            source_instance="guild",
            key_kind="container",
            key_value=str(generate_nanoid()),
            session_name=session_name,
        ),
        models.IngestPeerBinding(
            workspace_name=workspace_name,
            source="discord",
            source_instance="guild",
            key_kind="external_id",
            key_value=str(generate_nanoid()),
            peer_name=peer_name,
        ),
    ]


_IngestModel = (
    type[models.IngestRecord]
    | type[models.IngestSessionBinding]
    | type[models.IngestPeerBinding]
)


async def _count(db: AsyncSession, model: _IngestModel, workspace_name: str) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(model)
        .where(model.workspace_name == workspace_name)
    )
    return int(result.scalar_one())


@pytest.mark.asyncio
async def test_session_deletion_removes_only_that_sessions_ingest_rows(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """Session deletion removes its records and session bindings, not peer bindings."""
    workspace, peer = sample_data
    doomed = await _make_session(db_session, workspace, peer)
    kept = await _make_session(db_session, workspace, peer)
    for session_name in (doomed, kept):
        db_session.add_all(_ingest_rows(workspace.name, session_name, peer.name))
    await db_session.commit()

    await crud.delete_session(db_session, workspace.name, doomed)
    await db_session.commit()

    records = (
        (
            await db_session.execute(
                select(models.IngestRecord.session_name).where(
                    models.IngestRecord.workspace_name == workspace.name
                )
            )
        )
        .scalars()
        .all()
    )
    bindings = (
        (
            await db_session.execute(
                select(models.IngestSessionBinding.session_name).where(
                    models.IngestSessionBinding.workspace_name == workspace.name
                )
            )
        )
        .scalars()
        .all()
    )
    assert records == [kept]
    assert bindings == [kept]
    # Peer bindings outlive a session: the peer is still in the workspace.
    assert await _count(db_session, models.IngestPeerBinding, workspace.name) == 2


@pytest.mark.asyncio
async def test_workspace_deletion_removes_all_its_ingest_rows(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
) -> None:
    """Workspace deletion removes the rows of all three ingest tables."""
    workspace, peer = sample_data
    session_name = await _make_session(db_session, workspace, peer)
    db_session.add_all(_ingest_rows(workspace.name, session_name, peer.name))
    await db_session.commit()

    await crud.delete_workspace(db_session, workspace.name)
    await db_session.commit()

    for model in (
        models.IngestRecord,
        models.IngestSessionBinding,
        models.IngestPeerBinding,
    ):
        assert await _count(db_session, model, workspace.name) == 0


@pytest.mark.asyncio
async def test_session_deletion_waits_for_a_held_session_fence(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    """Session deletion blocks on an ingest transaction holding its session fence."""
    workspace, peer = sample_data
    session_name = await _make_session(db_session, workspace, peer)
    # Read once: a rollback expires the ORM objects.
    workspace_name = workspace.name

    # An ingest transaction in flight on this session.
    await take_workspace_fence(other_db, workspace_name, mode="shared")
    await take_session_fences(other_db, workspace_name, [session_name])

    await db_session.execute(text("SET LOCAL lock_timeout = '200ms'"))
    with pytest.raises(DBAPIError, match="lock timeout"):
        await crud.delete_session(db_session, workspace_name, session_name)
    await db_session.rollback()

    # Once the ingest transaction ends, deletion goes through.
    await other_db.rollback()
    await crud.delete_session(db_session, workspace_name, session_name)
    await db_session.commit()


@pytest.mark.asyncio
async def test_workspace_deletion_waits_for_a_shared_workspace_fence(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    """Workspace deletion blocks on any shared hold of the workspace fence."""
    workspace_name = sample_data[0].name
    await take_workspace_fence(other_db, workspace_name, mode="shared")

    await db_session.execute(text("SET LOCAL lock_timeout = '200ms'"))
    with pytest.raises(DBAPIError, match="lock timeout"):
        await crud.delete_workspace(db_session, workspace_name)
    await db_session.rollback()

    await other_db.rollback()
    await crud.delete_workspace(db_session, workspace_name)
    await db_session.commit()


@pytest.mark.asyncio
async def test_try_locks_report_a_held_fence_instead_of_waiting(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    workspace_name = sample_data[0].name
    await take_workspace_fence(other_db, workspace_name, mode="exclusive")
    assert not await take_workspace_fence(
        db_session, workspace_name, mode="shared", try_lock=True
    )
    await db_session.rollback()
    await other_db.rollback()

    await take_session_fences(other_db, workspace_name, ["s1"])
    assert await take_workspace_fence(
        db_session, workspace_name, mode="shared", try_lock=True
    )
    assert not await take_session_fences(
        db_session, workspace_name, ["s0", "s1"], try_lock=True
    )
    await db_session.rollback()
    await other_db.rollback()


@pytest.mark.asyncio
async def test_shared_workspace_fences_do_not_block_each_other(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    workspace, _ = sample_data
    await take_workspace_fence(other_db, workspace.name, mode="shared")
    assert await take_workspace_fence(
        db_session, workspace.name, mode="shared", try_lock=True
    )
    await db_session.rollback()
    await other_db.rollback()


@pytest.mark.asyncio
async def test_the_fence_does_not_block_the_message_routes_session_lock(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    """Deletion holds the fence, not the message route's lock, so messages still land."""
    workspace, peer = sample_data
    session_name = await _make_session(db_session, workspace, peer)

    await take_workspace_fence(other_db, workspace.name, mode="exclusive")
    await take_session_fences(other_db, workspace.name, [session_name])

    await db_session.execute(text("SET LOCAL lock_timeout = '200ms'"))
    created = await crud.create_messages(
        db_session,
        messages=[schemas.MessageCreate(content="hello", peer_id=peer.name)],
        workspace_name=workspace.name,
        session_name=session_name,
    )
    assert len(created) == 1
    await other_db.rollback()


@pytest.mark.asyncio
async def test_rows_an_ingest_writer_commits_while_deletion_waits_are_deleted(
    db_session: AsyncSession,
    other_db: AsyncSession,
    sample_data: tuple[Workspace, Peer],
) -> None:
    """The fence is what makes deletion's predicate delete see the writer's rows."""
    workspace, peer = sample_data
    session_name = await _make_session(db_session, workspace, peer)
    workspace_name, peer_name = workspace.name, peer.name

    # The writer holds the fence and has written rows it hasn't committed yet.
    await take_workspace_fence(other_db, workspace_name, mode="shared")
    await take_session_fences(other_db, workspace_name, [session_name])
    other_db.add_all(_ingest_rows(workspace_name, session_name, peer_name))
    await other_db.flush()

    deletion = asyncio.create_task(
        crud.delete_session(db_session, workspace_name, session_name)
    )
    await asyncio.sleep(0.3)
    assert not deletion.done()  # waiting on the fence

    await other_db.commit()
    await deletion
    await db_session.commit()

    assert await _count(db_session, models.IngestRecord, workspace_name) == 0
    assert await _count(db_session, models.IngestSessionBinding, workspace_name) == 0


@pytest.mark.asyncio
async def test_tenants_with_the_same_names_have_separate_fences(
    db_session: AsyncSession,
    other_db: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advisory locks are database-wide, so flag-on the keys carry the tenant."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    token = tenant_context.set("tenant-a")
    try:
        await take_workspace_fence(other_db, "default", mode="exclusive")
        await take_session_fences(other_db, "default", ["chat"])
    finally:
        tenant_context.reset(token)

    token = tenant_context.set("tenant-b")
    try:
        assert await take_workspace_fence(
            db_session, "default", mode="exclusive", try_lock=True
        )
        assert await take_session_fences(db_session, "default", ["chat"], try_lock=True)
    finally:
        tenant_context.reset(token)
        await db_session.rollback()
        await other_db.rollback()


def test_flag_off_keys_ignore_a_stray_tenant_context() -> None:
    """Flag-off the keys carry no tenant, matching the message route's lock."""
    assert not settings.MULTI_TENANT
    plain = (
        ingest_fence._workspace_key("ws"),  # pyright: ignore[reportPrivateUsage]
        ingest_fence._session_key("ws", "s"),  # pyright: ignore[reportPrivateUsage]
    )
    token = tenant_context.set("stray")
    try:
        assert (
            ingest_fence._workspace_key("ws"),  # pyright: ignore[reportPrivateUsage]
            ingest_fence._session_key("ws", "s"),  # pyright: ignore[reportPrivateUsage]
        ) == plain
    finally:
        tenant_context.reset(token)


def test_session_keys_cannot_be_confused_by_separators_in_names() -> None:
    """("a:b", "c") and ("a", "b:c") are different sessions with different keys."""
    key = ingest_fence._session_key  # pyright: ignore[reportPrivateUsage]
    assert key("a:b", "c") != key("a", "b:c")
