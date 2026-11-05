"""A Redis error in a cached fetcher runs the fetcher without the cache."""

import logging
from typing import Any

import pytest
from cashews.picklers import PicklerType
from redis.exceptions import MaxConnectionsError
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.cache.client import cache
from src.config import settings


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_command", ["get", "set_lock"])
async def test_cached_fetch_survives_a_cache_that_raises(
    failing_command: str,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace, _peer = sample_data
    session = models.Session(workspace_name=workspace.name, name="cache-down")
    db_session.add(session)
    await db_session.commit()

    async def no_free_connection(*_args: Any, **_kwargs: Any) -> Any:
        raise MaxConnectionsError()

    backend = cache._get_backend("")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(backend, failing_command, no_free_connection)

    with caplog.at_level(logging.WARNING, logger="src.cache.client"):
        found = await crud.get_session(db_session, "cache-down", workspace.name)

    assert found.id == session.id
    assert any("without the cache" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_cached_fetch_reads_the_db_when_cluster_init_fails(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cluster client whose initialization fails must not fail the fetch."""
    workspace, _peer = sample_data
    session = models.Session(workspace_name=workspace.name, name="cluster-down")
    db_session.add(session)
    await db_session.commit()

    reads = 0
    real_scalar = db_session.scalar

    async def counting_scalar(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        return await real_scalar(*args, **kwargs)

    monkeypatch.setattr(db_session, "scalar", counting_scalar)

    # Nothing listens on port 1, so every startup node refuses the connection.
    cache.setup(
        "redis://127.0.0.1:1?suppress=true",
        cluster=True,
        pickle_type=PicklerType.SQLALCHEMY,
        socket_connect_timeout=0.5,
    )
    try:
        found = await crud.get_session(db_session, "cluster-down", workspace.name)
    finally:
        await cache.close()
        cache.setup(settings.CACHE.URL, pickle_type=PicklerType.SQLALCHEMY)

    assert found.id == session.id
    assert reads >= 1, "the fetch fell back to the database"
