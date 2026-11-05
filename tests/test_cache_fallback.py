"""A Redis error in a cached fetcher runs the fetcher without the cache."""

import logging
from typing import Any

import pytest
from redis.exceptions import MaxConnectionsError
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.cache.client import cache


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
