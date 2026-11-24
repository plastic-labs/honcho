"""The background full count behind `total` on lists past PAGINATION_COUNT_LIMIT."""

import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, final

import pytest
from sqlalchemy import select

from src import models
from src.cache.client import cache
from src.config import settings
from src.utils import pagination

KEY = "test:pagination:estimate"
LOCK = f"{KEY}:lock"
STMT = select(models.Workspace)


@final
class _FakeSession:
    """Stands in for the refresh's DB session; counts how often it is asked to count."""

    def __init__(self, during_count: Callable[[], Awaitable[None]] | None = None):
        self.counts = 0
        self._during_count = during_count

    async def execute(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def scalar(self, *_args: Any, **_kwargs: Any) -> int:
        self.counts += 1
        if self._during_count is not None:
            await self._during_count()
        return 7


def _use_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> None:
    @asynccontextmanager
    async def fake_tracked_db(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any]:
        yield session

    monkeypatch.setattr(pagination, "tracked_db", fake_tracked_db)


async def _lock_is_free() -> bool:
    taken = await cache.set_lock(LOCK, "probe", expire=1)
    if taken:
        await cache.unlock(LOCK, "probe")
    return bool(taken)


async def test_a_refresh_stores_the_count_and_releases_its_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    session = _FakeSession()
    _use_session(monkeypatch, session)

    await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]

    assert session.counts == 1
    count, _ = await pagination._read_estimate(KEY)  # pyright: ignore[reportPrivateUsage]
    assert count == 7
    assert await _lock_is_free()


async def test_each_refresh_releases_only_its_own_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    """Each refresh locks and unlocks under its own token.

    Redis deletes a lock on unlock only when the token matches, so a refresh
    whose lock lapsed cannot release the lock another refresh took since. The
    in-memory test backend ignores the token, so this checks the tokens sent.
    """
    taken: list[str] = []
    released: list[str] = []
    real_set_lock, real_unlock = cache.set_lock, cache.unlock

    async def spy_set_lock(key: str, value: str, expire: float) -> bool:
        taken.append(value)
        return await real_set_lock(key, value, expire=expire)

    async def spy_unlock(key: str, value: str) -> bool:
        released.append(value)
        return await real_unlock(key, value)

    monkeypatch.setattr(cache, "set_lock", spy_set_lock)
    monkeypatch.setattr(cache, "unlock", spy_unlock)
    monkeypatch.setattr(settings.CACHE, "PAGINATION_ESTIMATE_REFRESH_SECONDS", 0)
    _use_session(monkeypatch, _FakeSession())

    await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]
    await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]

    assert released == taken
    assert len(set(taken)) == 2


async def test_a_refresh_skips_the_count_when_a_fresh_one_is_stored(
    monkeypatch: pytest.MonkeyPatch,
):
    """Another task stored a count between this request's read and its lock."""
    await cache.set(KEY, {"count": 5, "at": time.time()}, expire=60)
    session = _FakeSession()
    _use_session(monkeypatch, session)

    await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]

    assert session.counts == 0
    count, _ = await pagination._read_estimate(KEY)  # pyright: ignore[reportPrivateUsage]
    assert count == 5
    assert await _lock_is_free()


async def test_a_held_lock_skips_the_count(monkeypatch: pytest.MonkeyPatch):
    assert await cache.set_lock(LOCK, "other", expire=60)
    session = _FakeSession()
    _use_session(monkeypatch, session)

    await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]

    assert session.counts == 0


async def test_a_cache_failure_skips_the_refresh_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    async def broken_set_lock(*_args: Any, **_kwargs: Any) -> bool:
        raise ConnectionError("cache down")

    monkeypatch.setattr(cache, "set_lock", broken_set_lock)
    session = _FakeSession()
    _use_session(monkeypatch, session)

    with caplog.at_level(logging.WARNING, logger=pagination.__name__):
        await pagination._refresh_estimate(STMT, KEY)  # pyright: ignore[reportPrivateUsage]

    assert session.counts == 0
    assert not caplog.records
