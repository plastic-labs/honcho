import asyncio
from typing import Any, cast

import pytest
from cashews import Cache
from cashews.picklers import PicklerType
from redis.asyncio.retry import Retry

from src.cache import client
from src.config import settings

BLACKHOLE_URL = "redis://10.255.255.1:6379/0?suppress=true"


class _Captured(Exception):
    pass


async def _capture_setup_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_setup(url: str, **kwargs: Any) -> None:
        if url.startswith("mem://"):
            return
        captured.update(kwargs)
        raise _Captured()

    original_url = settings.CACHE.URL
    real_setup = client.cache.setup
    monkeypatch.setattr(settings.CACHE, "ENABLED", True)
    monkeypatch.setattr(settings.CACHE, "URL", BLACKHOLE_URL)
    monkeypatch.setattr(settings.SENTRY, "ENABLED", False)
    monkeypatch.setattr(client.cache, "setup", fake_setup)
    try:
        await client.init_cache()
    finally:
        monkeypatch.undo()
        real_setup(original_url, pickle_type=PicklerType.SQLALCHEMY)
    return captured


@pytest.mark.asyncio
async def test_init_cache_passes_connect_timeout_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.CACHE, "CONNECT_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(settings.CACHE, "CONNECT_RETRIES", 4)
    kwargs = await _capture_setup_kwargs(monkeypatch)
    assert kwargs["socket_connect_timeout"] == 7.0
    assert cast(Retry, kwargs["retry"]).get_retries() == 4


@pytest.mark.asyncio
async def test_connect_failures_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings.CACHE, "CONNECT_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(settings.CACHE, "CONNECT_RETRIES", 2)
    kwargs = await _capture_setup_kwargs(monkeypatch)
    kwargs.pop("pickle_type")
    kwargs.pop("cluster")
    attempts = 0
    real_open = asyncio.open_connection

    async def counting_open(*args: Any, **kw: Any):
        nonlocal attempts
        attempts += 1
        return await real_open(*args, **kw)

    monkeypatch.setattr(asyncio, "open_connection", counting_open)
    c = Cache()
    c.setup(BLACKHOLE_URL, **kwargs)
    try:
        assert await c.get("k") is None
    finally:
        await c.close()
    assert attempts == 3
