import uuid
from typing import Any

import pytest

import src.dependencies as dependencies_module
from src.config import settings
from src.db import request_context
from src.dependencies import get_db as real_get_db
from src.dependencies import tracked_db as real_tracked_db


class FakeSession:
    def __init__(self, *, in_transaction: bool = False):
        self._in_transaction: bool = in_transaction
        self.execute_calls: list[tuple[Any, ...]] = []
        self.rollback_calls: int = 0
        self.close_calls: int = 0
        self.connection_calls: int = 0

    async def connection(self) -> None:
        # Tracks checkout attempts so tests can assert get_db/tracked_db stay
        # lazy (they should never force a checkout themselves).
        self.connection_calls += 1

    async def execute(self, statement: Any, params: Any = None) -> None:
        self.execute_calls.append((statement, params))

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    def in_transaction(self) -> bool:
        return self._in_transaction


@pytest.mark.asyncio
async def test_get_db_yields_lazily_without_checkout_or_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # get_db must NOT touch the connection or run set_config itself — checkout
    # happens lazily inside the AsyncSession on first DB use, so a handler doing
    # non-DB work before its first query never pins a connection.
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", True)  # still no set_config here

    dep_gen = real_get_db()
    try:
        db = await anext(dep_gen)
        assert db is fake_db
        assert fake_db.connection_calls == 0  # no eager checkout
        assert fake_db.execute_calls == []  # no set_config in get_db
    finally:
        await dep_gen.aclose()

    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_get_db_rolls_back_and_closes_when_consumer_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    dep_gen = real_get_db()
    await anext(dep_gen)

    with pytest.raises(RuntimeError, match="boom"):
        await dep_gen.athrow(RuntimeError("boom"))

    assert fake_db.rollback_calls == 2  # once in except, once in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_creates_and_resets_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )

    clear_token = request_context.set(None)
    try:
        async with real_tracked_db("cleanup_job"):
            # tracked_db sets the task context so the lazy session can read it.
            assert request_context.get() == "task:cleanup_job:12345678"
    finally:
        request_context.reset(clear_token)

    assert request_context.get() is None
    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_preserves_existing_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)

    context_token = request_context.set("request:existing")
    try:
        async with real_tracked_db("ignored_op"):
            assert request_context.get() == "request:existing"
    finally:
        request_context.reset(context_token)

    assert fake_db.rollback_calls == 1  # unconditional rollback in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_rolls_back_on_error_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    with pytest.raises(ValueError, match="failed operation"):
        async with real_tracked_db("operation"):
            raise ValueError("failed operation")

    assert fake_db.rollback_calls == 2  # once in except, once in finally
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_rolls_back_open_transaction_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession(in_transaction=True)
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", False)

    async with real_tracked_db("operation"):
        pass

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1
