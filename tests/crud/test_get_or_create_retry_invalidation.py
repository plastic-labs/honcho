"""Regression tests for cache invalidation across the get_or_create retry path.

`get_or_create_peers` / `get_or_create_scopes` mutate existing rows, then insert
new ones inside `db.begin_nested()`. A concurrent writer that creates one of those
rows first makes the insert raise `IntegrityError`, and the function retries.

The subtlety: `begin_nested()` autoflushes the pending mutations *before* opening
the savepoint, so the rollback neither undoes them nor expires the ORM state. A
retry that recomputed "what changed" from that state would see no change and skip
the cache purge — while the row change still commits anyway, leaving the cache
stale until TTL. These tests pin the purge.

The race is real (a second session committing a real row, producing a real
IntegrityError from the database); only its *timing* is made deterministic, by
hooking the one point that sits between the SELECT and the INSERT.
"""

from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
)

from src import crud, models, schemas
from src.crud.peer import peer_cache_key
from src.crud.scope import SCOPE_PEER_CONFIGURATION, SCOPE_PEER_INTERNAL_METADATA
from src.utils.scopes import scope_peer_name


class _RaceOnBeginNested:
    """Commit a racing row on entry to `begin_nested()`, then delegate.

    That entry point is after the function's SELECT and metadata mutation but
    before its INSERT flushes — precisely the window a real concurrent writer
    has to slip through to trigger the IntegrityError retry.
    """

    _db: AsyncSession
    _engine: AsyncEngine
    _rows: list[models.Peer]
    _real: AsyncSessionTransaction | None
    fired: bool

    def __init__(self, db: AsyncSession, engine: AsyncEngine, rows: list[models.Peer]):
        self._db = db
        self._engine = engine
        self._rows = rows
        self._real = None
        self.fired = False

    def __call__(self):
        return self

    async def __aenter__(self):
        if self._rows:
            Session = async_sessionmaker(bind=self._engine, expire_on_commit=False)
            async with Session() as other:
                other.add_all(self._rows)
                await other.commit()
            self._rows = []  # race only once; the retry must succeed
            self.fired = True
        self._real = AsyncSession.begin_nested(self._db)
        return await self._real.__aenter__()

    async def __aexit__(self, *exc_info: object):
        assert self._real is not None
        return await self._real.__aexit__(*exc_info)


@pytest.mark.asyncio
async def test_peer_retry_still_invalidates_mutated_peer(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """A peer mutated before a losing race still gets its cache key purged."""
    test_workspace, existing_peer = sample_data
    racer_name = str(generate_nanoid())

    # Give the existing peer metadata we will then change, so it is a real update.
    existing_peer.h_metadata = {"v": "old"}
    await db_session.commit()

    race = _RaceOnBeginNested(
        db_session,
        db_engine,
        [models.Peer(name=racer_name, workspace_name=test_workspace.name)],
    )

    with (
        patch("src.crud.peer.safe_cache_delete", new=AsyncMock()) as mock_delete,
        patch.object(db_session, "begin_nested", race),
    ):
        result = await crud.get_or_create_peers(
            db_session,
            test_workspace.name,
            [
                schemas.PeerCreate(name=existing_peer.name, metadata={"v": "new"}),
                schemas.PeerCreate(name=racer_name),
            ],
        )
        await db_session.commit()
        await result.post_commit()

    assert race.fired, "the race must actually have fired"

    purged = {call.args[0] for call in mock_delete.await_args_list}
    assert (
        peer_cache_key(test_workspace.name, existing_peer.name) in purged
    ), "the mutated peer's cache key must still be purged after the retry"

    # The mutation really did land — which is what makes a missed purge stale.
    await db_session.refresh(existing_peer)
    assert existing_peer.h_metadata == {"v": "new"}


@pytest.mark.asyncio
async def test_scope_retry_still_invalidates_mutated_scope(
    db_session: AsyncSession,
    db_engine: AsyncEngine,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Same guarantee for the scopes facade, which mirrors get_or_create_peers."""
    test_workspace, _ = sample_data
    kept_scope, racing_scope = str(generate_nanoid()), str(generate_nanoid())

    seeded = await crud.get_or_create_scopes(
        db_session,
        test_workspace.name,
        [schemas.ScopeCreate(name=kept_scope, metadata={"v": "old"})],
    )
    await db_session.commit()
    await seeded.post_commit()

    # The racer creates the second scope's backing peer — as a *valid* scope peer,
    # so the flow reaches the insert rather than tripping the legacy-collision 409.
    race = _RaceOnBeginNested(
        db_session,
        db_engine,
        [
            models.Peer(
                name=scope_peer_name(racing_scope),
                workspace_name=test_workspace.name,
                internal_metadata=dict(SCOPE_PEER_INTERNAL_METADATA),
                configuration=dict(SCOPE_PEER_CONFIGURATION),
            )
        ],
    )

    with (
        patch("src.crud.scope.safe_cache_delete", new=AsyncMock()) as mock_delete,
        patch.object(db_session, "begin_nested", race),
    ):
        result = await crud.get_or_create_scopes(
            db_session,
            test_workspace.name,
            [
                schemas.ScopeCreate(name=kept_scope, metadata={"v": "new"}),
                schemas.ScopeCreate(name=racing_scope),
            ],
        )
        await db_session.commit()
        await result.post_commit()

    assert race.fired, "the race must actually have fired"

    purged = {call.args[0] for call in mock_delete.await_args_list}
    assert (
        peer_cache_key(test_workspace.name, scope_peer_name(kept_scope)) in purged
    ), "the mutated scope peer's cache key must still be purged after the retry"


@pytest.mark.asyncio
async def test_peer_no_race_does_not_invalidate_unchanged_peer(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    """Baseline: with no race, an unchanged peer is not purged."""
    test_workspace, existing_peer = sample_data
    existing_peer.h_metadata = {"v": "same"}
    await db_session.commit()

    with patch("src.crud.peer.safe_cache_delete", new=AsyncMock()) as mock_delete:
        result = await crud.get_or_create_peers(
            db_session,
            test_workspace.name,
            [schemas.PeerCreate(name=existing_peer.name, metadata={"v": "same"})],
        )
        await db_session.commit()
        await result.post_commit()

    assert mock_delete.await_count == 0, "an unchanged peer must not be purged"
