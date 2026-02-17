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
        self.execute_calls: list[Any] = []
        self.rollback_calls: int = 0
        self.close_calls: int = 0

    async def execute(self, statement: Any) -> None:
        self.execute_calls.append(statement)

    async def rollback(self) -> None:
        self.rollback_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    def in_transaction(self) -> bool:
        return self._in_transaction


@pytest.mark.asyncio
async def test_get_db_sets_application_name_when_tracing_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", True)

    context_token = request_context.set("request:test-ctx")
    dep_gen = real_get_db()

    try:
        db = await anext(dep_gen)
        assert db is fake_db
        assert len(fake_db.execute_calls) == 1
        assert "SET application_name = 'request:test-ctx'" in str(
            fake_db.execute_calls[0]
        )
    finally:
        await dep_gen.aclose()
        request_context.reset(context_token)

    assert fake_db.rollback_calls == 0
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

    assert fake_db.rollback_calls == 1
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_creates_and_resets_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", True)
    monkeypatch.setattr(
        uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )

    clear_token = request_context.set(None)
    try:
        async with real_tracked_db("cleanup_job"):
            assert request_context.get() == "task:cleanup_job:12345678"
    finally:
        request_context.reset(clear_token)

    assert request_context.get() is None
    assert len(fake_db.execute_calls) == 1
    assert "SET application_name = 'task:cleanup_job:12345678'" in str(
        fake_db.execute_calls[0]
    )
    assert fake_db.rollback_calls == 0
    assert fake_db.close_calls == 1


@pytest.mark.asyncio
async def test_tracked_db_preserves_existing_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeSession()
    monkeypatch.setattr(dependencies_module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(settings.DB, "TRACING", True)

    context_token = request_context.set("request:existing")
    try:
        async with real_tracked_db("ignored_op"):
            assert request_context.get() == "request:existing"
    finally:
        request_context.reset(context_token)

    assert len(fake_db.execute_calls) == 1
    assert "SET application_name = 'request:existing'" in str(fake_db.execute_calls[0])
    assert fake_db.rollback_calls == 0
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

    assert fake_db.rollback_calls == 1
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
