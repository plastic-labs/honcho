"""
Focused pytest tests for the three SQLite JSON operator fixes.

Runs with a pure in-memory SQLite database — no PostgreSQL, no Redis needed.

Tests:
  1. filter.py  — column[field].as_string()     (metadata filter queries)
  2. session.py — config["observe_others"].as_boolean() (peer observer count)
  3. deriver.py — payload["observer"].as_string()       (queue status)
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from cashews.backends.interface import ControlMixin
from cashews.picklers import PicklerType
from fakeredis import FakeAsyncRedis
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src import models  # noqa: F401 — ensures all models are registered
from src.cache.client import cache
from src.config import settings
from src.db import Base
from src.dependencies import get_db
from src.exceptions import HonchoException
from src.main import app

# ---------------------------------------------------------------------------
# SQLite engine / session fixtures (no PostgreSQL, no CREATE EXTENSION)
# ---------------------------------------------------------------------------

SQLITE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="module")
async def sqlite_engine():
    """Create an async SQLite engine with all tables."""
    # Save originals before any mutations so teardown always restores cleanly.
    original_uri = settings.DB.CONNECTION_URI
    original_schema = Base.metadata.schema
    original_table_schemas = {
        name: t.schema for name, t in Base.metadata.tables.items()
    }

    settings.DB.CONNECTION_URI = SQLITE_URL
    Base.metadata.schema = None
    for table in Base.metadata.tables.values():
        table.schema = None

    engine = create_async_engine(SQLITE_URL, echo=False)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
        Base.metadata.schema = original_schema
        for name, table in Base.metadata.tables.items():
            table.schema = original_table_schemas[name]
        settings.DB.CONNECTION_URI = original_uri


@pytest_asyncio.fixture(scope="function")
async def sqlite_session(sqlite_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a rollback-isolated session per test."""
    Session = async_sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    async with Session() as session:
        try:
            yield session
        finally:
            await session.rollback()
    # Clear all rows between tests
    async with sqlite_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())


# ---------------------------------------------------------------------------
# Cache fixture (fakeredis, same pattern as main conftest)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def sqlite_cache():
    original_enabled = settings.CACHE.ENABLED
    original_url = settings.CACHE.URL
    original_disable = ControlMixin._disable  # pyright: ignore[reportPrivateUsage]

    fake_redis = FakeAsyncRedis(decode_responses=True)

    def fake_from_url(*_a: Any, **_kw: Any):
        return fake_redis

    @property  # type: ignore
    def patched_disable(self):  # pyright: ignore
        try:
            return original_disable.fget(self)  # pyright: ignore[reportOptionalCall]
        except LookupError:
            return set()  # pyright: ignore

    redis_patch = patch("redis.asyncio.from_url", fake_from_url)
    redis_patch.start()
    try:
        ControlMixin._disable = patched_disable  # pyright: ignore
        settings.CACHE.ENABLED = True
        settings.CACHE.URL = "redis://fake:6379/0"
        cache.setup("redis://fake:6379/0", pickle_type=PicklerType.SQLALCHEMY, enable=True)
        yield fake_redis
    finally:
        redis_patch.stop()
        ControlMixin._disable = original_disable  # pyright: ignore
        settings.CACHE.ENABLED = original_enabled
        settings.CACHE.URL = original_url


# ---------------------------------------------------------------------------
# TestClient fixture wired to SQLite
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def sqlite_client(
    sqlite_session: AsyncSession,
    sqlite_cache: FakeAsyncRedis,  # noqa: ARG001
) -> TestClient:
    previous_handler = app.exception_handlers.get(HonchoException)

    @app.exception_handler(HonchoException)
    async def _handler(_: Request, exc: HonchoException) -> JSONResponse:  # pyright: ignore
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield sqlite_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)
    if previous_handler is None:
        app.exception_handlers.pop(HonchoException, None)
    else:
        app.exception_handlers[HonchoException] = previous_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_workspace(client: TestClient, name: str = "ws-sqlite-test") -> str:
    r = client.post("/v3/workspaces", json={"name": name})
    assert r.status_code in (200, 201), f"workspace create: {r.status_code} {r.text}"
    return r.json()["name"]


def make_peer(client: TestClient, ws: str, name: str) -> str:
    r = client.post(f"/v3/workspaces/{ws}/peers", json={"name": name})
    assert r.status_code in (200, 201), f"peer create: {r.status_code} {r.text}"
    return r.json()["name"]


def make_session(client: TestClient, ws: str, name: str = "sess") -> str:
    r = client.post(f"/v3/workspaces/{ws}/sessions", json={"name": name})
    assert r.status_code in (200, 201), f"session create: {r.status_code} {r.text}"
    return r.json()["name"]


# ---------------------------------------------------------------------------
# Fix 1 — filter.py: column[field].as_string()
# ---------------------------------------------------------------------------

def test_metadata_filter_as_string(sqlite_client: TestClient):
    """
    List messages with a metadata equality filter.
    Before the fix: as_string() didn't exist on JSONBCompat comparator,
    or astext generated wrong SQL on SQLite.
    """
    ws = make_workspace(sqlite_client, "ws-filter")
    peer = make_peer(sqlite_client, ws, "p1")
    sess = make_session(sqlite_client, ws, "s1")

    # Add peer to session
    r = sqlite_client.put(
        f"/v3/workspaces/{ws}/sessions/{sess}/peers",
        json=[{"peer_name": peer, "observe_others": False}],
    )
    assert r.status_code in (200, 201, 204)

    # Create a message with metadata
    r = sqlite_client.post(
        f"/v3/workspaces/{ws}/sessions/{sess}/messages",
        json=[{"content": "hello", "peer_name": peer, "metadata": {"topic": "sqlite"}}],
    )
    assert r.status_code in (200, 201), f"create message: {r.text}"

    # List with metadata filter — exercises as_string() in filter.py
    r = sqlite_client.post(
        f"/v3/workspaces/{ws}/sessions/{sess}/messages/list",
        json={"filters": {"topic": {"$eq": "sqlite"}}},
    )
    assert r.status_code == 200, f"filter query failed: {r.status_code} {r.text}"
    items = r.json().get("items", [])
    assert len(items) == 1, f"expected 1 result, got {len(items)}"
    assert items[0]["metadata"]["topic"] == "sqlite"


# ---------------------------------------------------------------------------
# Fix 2 — session.py: config["observe_others"].as_boolean()
# ---------------------------------------------------------------------------

def test_observe_others_as_boolean(sqlite_client: TestClient):
    """
    Update a peer's observe_others config inside a session.
    The observer-count check in session.py uses as_boolean() on a JSON field —
    before the fix this generated wrong SQL on SQLite.
    """
    ws = make_workspace(sqlite_client, "ws-obs")
    observer = make_peer(sqlite_client, ws, "observer")
    target = make_peer(sqlite_client, ws, "target")
    sess = make_session(sqlite_client, ws, "s2")

    # Add both peers; observer watches others
    r = sqlite_client.put(
        f"/v3/workspaces/{ws}/sessions/{sess}/peers",
        json=[
            {"peer_name": observer, "observe_others": True},
            {"peer_name": target, "observe_others": False},
        ],
    )
    assert r.status_code in (200, 201, 204), f"add peers: {r.text}"

    # Update target's config — this triggers the observer-count query (as_boolean path)
    r = sqlite_client.put(
        f"/v3/workspaces/{ws}/sessions/{sess}/peers/{target}/config",
        json={"observe_others": True},
    )
    assert r.status_code in (200, 204), f"set_peer_config failed: {r.status_code} {r.text}"


# ---------------------------------------------------------------------------
# Fix 3 — deriver.py: payload["observer"].as_string()
# ---------------------------------------------------------------------------

def test_queue_status_as_string(sqlite_client: TestClient):
    """
    Fetch the workspace queue status (plain + filtered by observer).
    _build_queue_status_query uses as_string() on QueueItem.payload JSON fields —
    before the fix astext generated wrong SQL on SQLite.
    """
    ws = make_workspace(sqlite_client, "ws-queue")
    peer = make_peer(sqlite_client, ws, "qpeer")
    sess = make_session(sqlite_client, ws, "s3")

    r = sqlite_client.put(
        f"/v3/workspaces/{ws}/sessions/{sess}/peers",
        json=[{"peer_name": peer, "observe_others": True}],
    )
    assert r.status_code in (200, 201, 204)

    # Create a message to enqueue a deriver task
    r = sqlite_client.post(
        f"/v3/workspaces/{ws}/sessions/{sess}/messages",
        json=[{"content": "queue trigger", "peer_name": peer}],
    )
    assert r.status_code in (200, 201), f"create message: {r.text}"

    # Plain queue status — exercises as_string() in SELECT expressions
    r = sqlite_client.get(f"/v3/workspaces/{ws}/queue/status")
    assert r.status_code == 200, f"queue/status failed: {r.status_code} {r.text}"

    # Filtered queue status — exercises as_string() in WHERE clause too
    r = sqlite_client.get(
        f"/v3/workspaces/{ws}/queue/status",
        params={"observer_id": peer},
    )
    assert r.status_code == 200, f"queue/status?observer_id failed: {r.status_code} {r.text}"
