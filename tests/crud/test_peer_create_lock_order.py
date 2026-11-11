"""Concurrent creates of the same new peers, named in opposite orders, both succeed.

Each create inserts its new peers in one flush, and every inserted row takes the
lock on its `(tenant_id, name, workspace_name)` unique key until the transaction
ends. If the inserts followed request order, a create listing alice then bob and
one listing bob then alice could each hold their first name while waiting on the
other's: Postgres aborts one with a deadlock, which no caller retries.

The window between two rows of one INSERT is too short for a plain
`asyncio.gather` to hit reliably, so a test-only BEFORE INSERT trigger on `peers`
sleeps before each row of this test's workspace. Both creates then reach their
first row together and their second row together, which is exactly the
interleaving that deadlocks when the two orders differ. With the inserts in name
order, the second create waits on the first one's row, gets a unique violation
once it commits, and takes the existing-peer path on its retry.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import crud, models, schemas
from src.utils.scopes import scope_peer_name
from src.utils.types import GetOrCreateResult

_PACE_SECONDS = 0.3
_TRIGGER = "test_pace_peer_insert"


@pytest.fixture
async def paced_peer_inserts(
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
) -> AsyncIterator[str]:
    """Sleep before every peer row inserted into the test workspace."""
    workspace, _ = sample_data
    # Nanoid names are drawn from [A-Za-z0-9_-], so the literal needs no escaping.
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
            CREATE TRIGGER {_TRIGGER} BEFORE INSERT ON peers FOR EACH ROW
            WHEN (NEW.workspace_name = '{workspace.name}')
            EXECUTE FUNCTION {_TRIGGER}()
            """
        )
    try:
        yield workspace.name
    finally:
        async with db_engine.begin() as conn:
            await conn.exec_driver_sql(f"DROP TRIGGER {_TRIGGER} ON peers")
            await conn.exec_driver_sql(f"DROP FUNCTION {_TRIGGER}()")


Create = Callable[
    [AsyncSession, str, list[str]], Awaitable[GetOrCreateResult[list[models.Peer]]]
]


async def _create_peers(
    db: AsyncSession, workspace_name: str, names: list[str]
) -> GetOrCreateResult[list[models.Peer]]:
    return await crud.get_or_create_peers(
        db, workspace_name, [schemas.PeerSpec(name=name) for name in names]
    )


async def _create_scopes(
    db: AsyncSession, workspace_name: str, names: list[str]
) -> GetOrCreateResult[list[models.Peer]]:
    return await crud.get_or_create_scopes(
        db, workspace_name, [schemas.ScopeCreate(name=name) for name in names]
    )


def _plain_peer_name(name: str) -> str:
    return name


@pytest.mark.parametrize(
    ("create", "stored_name"),
    [
        pytest.param(_create_peers, _plain_peer_name, id="peers"),
        pytest.param(_create_scopes, scope_peer_name, id="scopes"),
    ],
)
async def test_opposite_order_creates_do_not_deadlock(
    db_engine: AsyncEngine,
    paced_peer_inserts: str,
    create: Create,
    stored_name: Callable[[str], str],
):
    workspace_name = paced_peer_inserts
    forward = ["alice", "bob", "carol"]
    backward = list(reversed(forward))
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def run(names: list[str]) -> tuple[bool, list[str]]:
        async with factory() as db:
            result = await create(db, workspace_name, names)
            returned = [peer.name for peer in result.resource]
            await db.commit()
        await result.post_commit()
        return result.created, returned

    outcomes = await asyncio.gather(run(forward), run(backward), return_exceptions=True)
    errors = [o for o in outcomes if isinstance(o, BaseException)]
    assert not errors, f"concurrent creates failed: {errors!r}"

    # One create inserted every peer; the other found them all on its retry.
    results = [o for o in outcomes if not isinstance(o, BaseException)]
    assert sorted(created for created, _ in results) == [False, True]
    # The create that inserted them returns them in request order.
    for (created, returned), names in zip(results, (forward, backward), strict=True):
        if created:
            assert returned == [stored_name(name) for name in names]

    # A fresh session, closed before the fixture drops the trigger: an open
    # transaction that has read `peers` would block that DROP.
    async with factory() as db:
        rows = await db.execute(
            select(models.Peer.name, func.count())
            .where(models.Peer.workspace_name == workspace_name)
            .where(models.Peer.name.in_([stored_name(n) for n in forward]))
            .group_by(models.Peer.name)
        )
        counts = {name: count for name, count in rows.tuples()}
    assert counts == {stored_name(name): 1 for name in forward}


@pytest.mark.parametrize(
    ("create", "stored_name"),
    [
        pytest.param(_create_peers, _plain_peer_name, id="peers"),
        pytest.param(_create_scopes, scope_peer_name, id="scopes"),
    ],
)
async def test_new_peers_are_inserted_in_name_order(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    create: Create,
    stored_name: Callable[[str], str],
):
    """The order the lock-order guarantee rests on, independent of any timing.

    The concurrent test above can only fail when both creates overlap; this one
    records the rows the flush inserts and checks their order directly.
    """
    workspace, _ = sample_data
    inserted: list[str] = []

    def record(_mapper: Any, _connection: Any, target: models.Peer) -> None:
        if target.workspace_name == workspace.name:
            inserted.append(target.name)

    event.listen(models.Peer, "before_insert", record)
    try:
        await create(db_session, workspace.name, ["carol", "alice", "bob"])
    finally:
        event.remove(models.Peer, "before_insert", record)

    assert inserted == [stored_name(name) for name in ("alice", "bob", "carol")]
