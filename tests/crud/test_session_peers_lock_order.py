"""Concurrent membership replaces (and removes) of one session both succeed.

A replace ends every ordinary membership it does not name, then upserts the ones
it does. Each phase takes row locks on `session_peers`, so two replaces with
disjoint peer sets lock in opposite orders: on a session with members a, b, c
and d, replacing with {a, b} ends c and d and then wants a and b, while replacing
with {c, d} ends a and b and then wants c and d. Each holds what the other wants
next, and Postgres aborts one with a deadlock that no caller retries. A remove of
rows the replace both ends and upserts cycles the same way. Sorting either phase
cannot help, since the two phases cover different rows.

Replaces and removes of one session therefore serialize on a per-session lock
taken before either writes a row. The window between the phases is too short for
a plain `asyncio.gather` to hit reliably, so a test-only BEFORE INSERT OR UPDATE
trigger on `session_peers` sleeps before each row of this test's workspace. The
sleeps hold each writer's row locks open long enough that both lock their first
rows before either reaches the other's, the interleaving that deadlocks without
that lock.
"""

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.config import settings
from src.crud.session import (
    _session_membership_lock_key,  # pyright: ignore[reportPrivateUsage]
)
from src.db import tenant_context
from src.exceptions import ResourceNotFoundException
from src.utils.scopes import scope_peer_name

_PACE_SECONDS = 0.3
_TRIGGER = "test_pace_session_peer_write"
_MEMBERS = ["a", "b", "c", "d"]


@pytest.fixture
async def paced_session(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
) -> AsyncIterator[tuple[str, str]]:
    """A session with members a-d, then a sleep before every membership write."""
    workspace, _ = sample_data
    # The name goes into the trigger's SQL as a literal, so it must need no escaping.
    assert re.fullmatch(r"[A-Za-z0-9_-]+", workspace.name), workspace.name
    session_name = str(generate_nanoid())
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        result = await crud.get_or_create_session(
            db,
            schemas.SessionCreate(
                name=session_name,
                peers={name: schemas.SessionPeerConfigInput() for name in _MEMBERS},
            ),
            workspace.name,
        )
        await db.commit()
    await result.post_commit()

    # Created after the members are seeded, so seeding is not paced.
    async with db_engine.begin() as conn:
        await conn.exec_driver_sql(
            f"""
            CREATE FUNCTION {_TRIGGER}() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                PERFORM pg_sleep({_PACE_SECONDS});
                RETURN NEW;
            END
            $$
            """
        )
        await conn.exec_driver_sql(
            f"""
            CREATE TRIGGER {_TRIGGER} BEFORE INSERT OR UPDATE ON session_peers
            FOR EACH ROW WHEN (NEW.workspace_name = '{workspace.name}')
            EXECUTE FUNCTION {_TRIGGER}()
            """
        )
    try:
        yield workspace.name, session_name
    finally:
        async with db_engine.begin() as conn:
            await conn.exec_driver_sql(f"DROP TRIGGER {_TRIGGER} ON session_peers")
            await conn.exec_driver_sql(f"DROP FUNCTION {_TRIGGER}()")


Write = Callable[[AsyncSession, str, str], Awaitable[object]]


def _replace(names: list[str]) -> Write:
    async def write(db: AsyncSession, workspace_name: str, session_name: str):
        return await crud.set_peers_for_session(
            db,
            workspace_name,
            session_name,
            {name: schemas.SessionPeerConfig() for name in names},
        )

    return write


def _remove(names: list[str]) -> Write:
    async def write(db: AsyncSession, workspace_name: str, session_name: str):
        return await crud.remove_peers_from_session(
            db, workspace_name, session_name, set(names)
        )

    return write


@pytest.mark.parametrize(
    ("first", "second", "outcomes"),
    [
        pytest.param(
            _replace(["a", "b"]),
            _replace(["c", "d"]),
            # Whichever replace commits last decides the membership.
            [{"a", "b"}, {"c", "d"}],
            id="disjoint-replaces",
        ),
        pytest.param(
            _replace(["a", "b"]),
            _remove(_MEMBERS),
            # Replace then remove leaves no one; remove then replace leaves a, b.
            [set[str](), {"a", "b"}],
            id="replace-and-remove",
        ),
    ],
)
async def test_concurrent_membership_writes_do_not_deadlock(
    db_engine: AsyncEngine,
    paced_session: tuple[str, str],
    first: Write,
    second: Write,
    outcomes: list[set[str]],
):
    workspace_name, session_name = paced_session
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def run(write: Write) -> None:
        async with factory() as db:
            await write(db, workspace_name, session_name)

    results = await asyncio.gather(run(first), run(second), return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"concurrent membership writes failed: {errors!r}"

    # A fresh session, closed before the fixture drops the trigger: an open
    # transaction that has read `session_peers` would block that DROP.
    async with factory() as db:
        active = set(
            (
                await db.scalars(
                    select(models.SessionPeer.peer_name)
                    .where(models.SessionPeer.workspace_name == workspace_name)
                    .where(models.SessionPeer.session_name == session_name)
                    .where(models.SessionPeer.left_at.is_(None))
                )
            ).all()
        )
    # The result of running the two writes one after the other, in some order.
    assert active in outcomes


async def _session_with_members(
    db_engine: AsyncEngine, workspace_name: str
) -> tuple[async_sessionmaker[AsyncSession], str]:
    session_name = str(generate_nanoid())
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        result = await crud.get_or_create_session(
            db,
            schemas.SessionCreate(
                name=session_name,
                peers={name: schemas.SessionPeerConfigInput() for name in _MEMBERS},
            ),
            workspace_name,
        )
        await db.commit()
    await result.post_commit()
    return factory, session_name


async def _mark_for_deletion(
    factory: async_sessionmaker[AsyncSession],
    workspace_name: str,
    session_name: str,
    *,
    tombstone: bool = False,
) -> None:
    async with factory() as db:
        session = await db.scalar(
            select(models.Session).where(
                models.Session.workspace_name == workspace_name,
                models.Session.name == session_name,
            )
        )
        assert session is not None
        if tombstone:
            session.deleted_at = func.now()
        else:
            session.is_active = False
        await db.commit()


async def _active(db: AsyncSession, workspace_name: str, session_name: str) -> set[str]:
    rows = await db.scalars(
        select(models.SessionPeer.peer_name).where(
            models.SessionPeer.workspace_name == workspace_name,
            models.SessionPeer.session_name == session_name,
            models.SessionPeer.left_at.is_(None),
        )
    )
    return set(rows)


async def _members(
    factory: async_sessionmaker[AsyncSession], workspace_name: str, session_name: str
) -> set[str]:
    async with factory() as db:
        return await _active(db, workspace_name, session_name)


_ENDING_WRITES = [
    pytest.param(_replace(["x"]), id="replace"),
    pytest.param(_remove(["a"]), id="remove"),
]


@pytest.mark.parametrize("tombstone", [False, True], ids=["marked", "tombstoned"])
@pytest.mark.parametrize("write", _ENDING_WRITES)
async def test_a_write_to_a_session_marked_for_deletion_writes_nothing(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
    write: Write,
    tombstone: bool,
) -> None:
    workspace_name = sample_data[0].name
    factory, session_name = await _session_with_members(db_engine, workspace_name)
    await _mark_for_deletion(factory, workspace_name, session_name, tombstone=tombstone)

    async with factory() as db:
        with pytest.raises(ResourceNotFoundException):
            await write(db, workspace_name, session_name)
        # Read in the writer's own transaction, before its rollback could hide
        # a change it made before raising.
        assert await _active(db, workspace_name, session_name) == set(_MEMBERS)
        await db.rollback()

    assert await _members(factory, workspace_name, session_name) == set(_MEMBERS)


@pytest.mark.parametrize("write", _ENDING_WRITES)
async def test_a_write_that_waited_on_the_lock_sees_a_deletion_made_meanwhile(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
    write: Write,
) -> None:
    """The session is read after the lock, so a deletion committed while the
    write waited is seen, and the write changes nothing."""
    workspace_name = sample_data[0].name
    factory, session_name = await _session_with_members(db_engine, workspace_name)
    key = _session_membership_lock_key(workspace_name, session_name)

    async with factory() as holder:
        await holder.execute(select(func.pg_advisory_xact_lock(key)))

        pids: list[int] = []

        async def blocked_write() -> set[str]:
            async with factory() as db:
                pids.append(int(await db.scalar(select(func.pg_backend_pid())) or 0))
                try:
                    with pytest.raises(ResourceNotFoundException):
                        await write(db, workspace_name, session_name)
                    # In the writer's own transaction, before the rollback.
                    return await _active(db, workspace_name, session_name)
                finally:
                    await db.rollback()

        waiting = asyncio.create_task(blocked_write())
        try:
            # Until the write's backend waits on the advisory lock.
            for _ in range(200):
                assert not waiting.done(), "the write didn't wait on the lock"
                if pids and await holder.scalar(
                    select(func.count())
                    .select_from(text("pg_locks"))
                    .where(
                        text("pid = :pid AND locktype = 'advisory' AND NOT granted")
                    ),
                    {"pid": pids[0]},
                ):
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("the write never reached the lock")
            await _mark_for_deletion(factory, workspace_name, session_name)
        except BaseException:
            # Stop the write while the lock still holds it, so it can't go on
            # to change the session, which is still active.
            waiting.cancel()
            with contextlib.suppress(BaseException):
                await waiting
            raise
        await holder.rollback()  # releases the lock

    seen_by_writer = await asyncio.wait_for(waiting, timeout=10)
    assert seen_by_writer == set(_MEMBERS)
    assert await _members(factory, workspace_name, session_name) == set(_MEMBERS)


async def _membership_times(
    factory: async_sessionmaker[AsyncSession],
    workspace_name: str,
    session_name: str,
    peer_name: str,
) -> tuple[object, object]:
    async with factory() as db:
        row = (
            await db.execute(
                select(models.SessionPeer.joined_at, models.SessionPeer.left_at).where(
                    models.SessionPeer.workspace_name == workspace_name,
                    models.SessionPeer.session_name == session_name,
                    models.SessionPeer.peer_name == peer_name,
                )
            )
        ).one()
        return row.joined_at, row.left_at


@pytest.mark.parametrize(
    "end_a",
    [
        pytest.param(_remove(["a"]), id="remove"),
        pytest.param(_replace(["b", "c", "d"]), id="replace"),
    ],
)
async def test_a_membership_never_ends_before_it_starts(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
    end_a: Write,
) -> None:
    """A write that ends a membership can begin its transaction before a rejoin
    of that member commits; the end is the time of the write, not of the
    transaction's start, so it's never before the rejoin."""
    workspace_name = sample_data[0].name
    factory, session_name = await _session_with_members(db_engine, workspace_name)
    async with factory() as db:
        await _remove(["a"])(db, workspace_name, session_name)

    async with factory() as ender:
        began = await ender.scalar(select(func.now()))  # begins its transaction
        async with factory() as db:
            await _replace(_MEMBERS)(db, workspace_name, session_name)  # rejoins a
        joined_at, left_at = await _membership_times(
            factory, workspace_name, session_name, "a"
        )
        assert left_at is None, "a wasn't rejoined"
        assert joined_at > began  # pyright: ignore[reportOperatorIssue]
        await end_a(ender, workspace_name, session_name)

    joined_at, left_at = await _membership_times(
        factory, workspace_name, session_name, "a"
    )
    assert left_at is not None
    assert left_at >= joined_at  # pyright: ignore[reportOperatorIssue]


async def test_a_scope_membership_never_ends_before_it_starts(
    db_engine: AsyncEngine, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace_name = sample_data[0].name
    factory, session_name = await _session_with_members(db_engine, workspace_name)
    scope_name = f"s{generate_nanoid(alphabet='0123456789abcdef', size=12)}"
    async with factory() as db:
        result = await crud.get_or_create_scopes(
            db, workspace_name, [schemas.ScopeCreate(name=scope_name)]
        )
        await db.commit()
    await result.post_commit()
    async with factory() as db:
        await crud.add_sessions_to_scope(db, workspace_name, scope_name, [session_name])
    async with factory() as db:
        await crud.remove_sessions_from_scope(
            db, workspace_name, scope_name, [session_name]
        )

    async with factory() as ender:
        began = await ender.scalar(select(func.now()))  # begins its transaction
        async with factory() as db:
            await crud.add_sessions_to_scope(
                db, workspace_name, scope_name, [session_name]
            )
        joined_at, left_at = await _membership_times(
            factory, workspace_name, session_name, scope_peer_name(scope_name)
        )
        assert left_at is None, "the session wasn't added back"
        assert joined_at > began  # pyright: ignore[reportOperatorIssue]
        await crud.remove_sessions_from_scope(
            ender, workspace_name, scope_name, [session_name]
        )

    joined_at, left_at = await _membership_times(
        factory, workspace_name, session_name, scope_peer_name(scope_name)
    )
    assert left_at is not None
    assert left_at >= joined_at  # pyright: ignore[reportOperatorIssue]


def test_lock_key_is_per_session_and_per_tenant(monkeypatch: pytest.MonkeyPatch):
    """Advisory locks are database-wide, so the key carries the tenant."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    keys: list[int] = []
    for tenant, session_name in (("t-a", "s1"), ("t-b", "s1"), ("t-a", "s2")):
        token = tenant_context.set(tenant)
        try:
            keys.append(_session_membership_lock_key("default", session_name))
        finally:
            tenant_context.reset(token)
    assert len(set(keys)) == 3
    assert all(-(2**63) <= key < 2**63 for key in keys)


def test_lock_key_without_tenant_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="no tenant bound"):
            _session_membership_lock_key("default", "s1")
    finally:
        tenant_context.reset(token)
