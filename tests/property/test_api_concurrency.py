"""Concurrent requests against the same resources never produce a 5xx.

Each example fires a batch of operations at the app at the same moment
(`asyncio.gather` over httpx's ASGI transport), every request on its own DB
session and connection as in production. The property: no response is a 5xx,
and get-or-create calls leave exactly one row.
"""

import asyncio
import os
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.cache.client import cache
from src.dependencies import get_db, get_read_db
from src.main import app
from tests.conftest import _clear_all_tables  # pyright: ignore[reportPrivateUsage]

MAX_EXAMPLES = int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "15"))

# Small pools so concurrent requests collide on the same names.
NAMES = st.sampled_from(["a", "b"])

Request = tuple[str, str, Any]


def _ops(ws: str, peer: str, other: str, session: str, scope: str) -> list[Request]:
    base = f"/v3/workspaces/{ws}"
    sess = f"{base}/sessions/{session}"
    msg = {"messages": [{"content": "hi", "peer_id": peer}]}
    return [
        ("POST", "/v3/workspaces", {"id": ws}),
        ("POST", f"{base}/peers", {"id": peer}),
        ("POST", f"{base}/peers", {"id": other}),
        ("POST", f"{base}/sessions", {"id": session}),
        ("POST", f"{base}/sessions", {"id": session, "peers": {peer: {}}}),
        ("POST", f"{sess}/messages", msg),
        ("POST", f"{sess}/messages", {"messages": msg["messages"] * 3}),
        ("POST", f"{sess}/peers", {peer: {}, other: {}}),
        ("PUT", f"{sess}/peers", {other: {}}),
        ("DELETE", f"{sess}/peers", [peer]),
        ("PUT", f"{sess}", {"metadata": {"k": peer}}),
        ("PUT", f"{base}/peers/{peer}", {"metadata": {"k": session}}),
        ("PUT", f"{base}", {"metadata": {"k": session}}),
        ("DELETE", f"{sess}", None),
        ("POST", f"{sess}/clone", None),
        ("POST", f"{base}/scopes", {"id": scope}),
        ("POST", f"{base}/scopes/{scope}/sessions", {"session_ids": [session]}),
        ("DELETE", f"{base}/scopes/{scope}/sessions/{session}", None),
        (
            "POST",
            f"{base}/conclusions",
            {
                "conclusions": [
                    {
                        "content": "x",
                        "observer_id": peer,
                        "observed_id": other,
                        "session_id": session,
                    }
                ]
            },
        ),
        ("PUT", f"{base}/peers/{peer}/card", {"peer_card": ["x"]}),
        ("GET", f"{sess}/context", None),
        ("GET", f"{sess}/peers", None),
        ("DELETE", f"{base}", None),
    ]


class JitterSession(AsyncSession):
    """Yields to the event loop after every statement so requests interleave.

    Without it a request often finishes its whole get-or-create before the next
    one starts, and the unique-violation paths never run.
    """

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().execute(*args, **kwargs)
        await asyncio.sleep(random.random() * 0.005)
        return result

    async def scalar(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().scalar(*args, **kwargs)
        await asyncio.sleep(random.random() * 0.005)
        return result

    async def scalars(self, *args: Any, **kwargs: Any) -> Any:
        result = await super().scalars(*args, **kwargs)
        await asyncio.sleep(random.random() * 0.005)
        return result


class Harness:
    """Runs request batches concurrently and tracks real DB-session overlap."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.factory: async_sessionmaker[JitterSession] = async_sessionmaker(
            bind=engine, expire_on_commit=False, class_=JitterSession
        )
        self.in_flight: int = 0
        self.max_in_flight: int = 0

    async def _db(self) -> AsyncGenerator[AsyncSession, Any]:
        async with self.factory() as session:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                self.in_flight -= 1
                await session.rollback()

    async def run(self, batch: list[Request]) -> list[httpx.Response]:
        app.dependency_overrides[get_db] = self._db
        app.dependency_overrides[get_read_db] = self._db
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        start = asyncio.Event()
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:

            async def one(method: str, path: str, body: Any) -> httpx.Response:
                await start.wait()
                return await client.request(method, path, json=body)

            tasks = [asyncio.ensure_future(one(*r)) for r in batch]
            await asyncio.sleep(0)
            start.set()
            return await asyncio.gather(*tasks)


async def _reset(engine: AsyncEngine) -> None:
    await _clear_all_tables(engine)
    await cache.clear()  # rows are gone, so cached copies of them must go too


async def _scenario(
    engine: AsyncEngine,
    picks: list[int],
    names: tuple[str, str, str, str],
    seed: str = "none",
) -> tuple[list[httpx.Response], list[Request], Harness]:
    await _reset(engine)
    ws, peer, other, session = names
    ops = _ops(ws, peer, other, session, "sc")
    harness = Harness(engine)
    # Pre-create parents in some examples so batches also hit existing-parent paths.
    if seed != "none":
        await harness.run([ops[0]])
    if seed == "all":
        await harness.run([ops[1], ops[2], ops[3], ops[15]])
    batch = [ops[i % len(ops)] for i in picks]
    responses = await harness.run(batch)
    return responses, batch, harness


@pytest.fixture
def fuzz_engine(db_engine: AsyncEngine) -> str:
    return db_engine.url.render_as_string(hide_password=False)


@settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    database=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)
@given(
    picks=st.lists(st.integers(0, 100), min_size=2, max_size=30),
    ws=NAMES,
    peer=NAMES,
    session=NAMES,
    seed=st.sampled_from(["none", "workspace", "all"]),
)
def test_concurrent_requests_do_not_5xx(
    picks: list[int], ws: str, peer: str, session: str, seed: str, fuzz_engine: str
):
    async def go() -> None:
        engine = create_async_engine(fuzz_engine, pool_size=20)
        try:
            names = (f"ws-{ws}", f"p-{peer}", "p-other", f"s-{session}")
            responses, batch, harness = await _scenario(engine, picks, names, seed)
            bad = [
                (req[0], req[1], r.status_code, r.text[:300])
                for req, r in zip(batch, responses, strict=True)
                if r.status_code >= 500
            ]
            assert not bad, bad
            # Get-or-create is idempotent, so losing a race must not surface as 409.
            conflicts = [
                (req[0], req[1], r.text[:200])
                for req, r in zip(batch, responses, strict=True)
                if r.status_code == 409 and req[0] in ("POST", "PUT")
            ]
            assert not conflicts, (conflicts, seed, [(r[0], r[1]) for r in batch])
            del harness
        finally:
            await engine.dispose()

    asyncio.run(go())


def test_harness_overlaps_transactions(fuzz_engine: str):
    """Guard against a harness that quietly serializes requests."""

    async def go() -> int:
        engine = create_async_engine(fuzz_engine, pool_size=20)
        try:
            ops = _ops("ws-x", "p-x", "p-y", "s-x", "sc")
            _, _, harness = await _scenario(
                engine, [0] * 6, ("ws-x", "p-x", "p-y", "s-x")
            )
            del ops
            return harness.max_in_flight
        finally:
            await engine.dispose()

    assert asyncio.run(go()) >= 4


@pytest.mark.parametrize(
    ("index", "expected_rows"),
    [
        (0, {"workspaces": 1, "sessions": 0, "peers": 0}),
        (1, {"workspaces": 1, "sessions": 0, "peers": 1}),
        (3, {"workspaces": 1, "sessions": 1, "peers": 0}),
        (4, {"workspaces": 1, "sessions": 1, "peers": 1}),
        (5, {"workspaces": 1, "sessions": 1, "peers": 1}),
        # A scope is backed by a peer
        (15, {"workspaces": 1, "sessions": 0, "peers": 1}),
    ],
)
def test_identical_get_or_create_race(
    index: int, expected_rows: dict[str, int], fuzz_engine: str
):
    """25 identical requests at once all succeed and leave one row each."""

    async def go() -> None:
        engine = create_async_engine(fuzz_engine, pool_size=40, max_overflow=0)
        try:
            await _reset(engine)
            ops = _ops("ws-r", "p-r", "p-o", "s-r", "sc")
            harness = Harness(engine)
            if index > 0:
                await harness.run([ops[0]])
            responses = await harness.run([ops[index]] * 25)
            statuses = {r.status_code for r in responses}
            assert statuses <= {200, 201}, [r.text[:200] for r in responses]
            async with engine.connect() as conn:
                rows = {
                    table: (
                        await conn.exec_driver_sql(f"SELECT count(*) FROM {table}")  # noqa: S608
                    ).scalar_one()
                    for table in expected_rows
                }
            assert rows == expected_rows
        finally:
            await engine.dispose()

    asyncio.run(go())
