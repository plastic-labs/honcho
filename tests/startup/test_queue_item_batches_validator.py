"""Startup validator for the queue_item_batches trigger set."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.startup.embedding_validator import StartupValidationError
from src.startup.queue_item_batches_validator import (
    _REQUIRED_TRIGGERS,  # pyright: ignore[reportPrivateUsage]
    _assert_triggers_live,  # pyright: ignore[reportPrivateUsage]
    _TriggerState,  # pyright: ignore[reportPrivateUsage]
    validate_queue_item_batches,
)


def _live_triggers() -> dict[str, str]:
    return dict.fromkeys(_REQUIRED_TRIGGERS, "O")


# ---------------------------------------------------------------------------
# Pure-function unit tests for the assertion
# ---------------------------------------------------------------------------


def test_passes_when_table_exists_and_every_trigger_is_live() -> None:
    _assert_triggers_live(_TriggerState(True, _live_triggers()), schema="public")


def test_always_enabled_trigger_counts_as_live() -> None:
    triggers = _live_triggers()
    triggers["trg_queue_item_batches_insert"] = "A"
    _assert_triggers_live(_TriggerState(True, triggers), schema="public")


def test_raises_when_the_aggregate_table_is_absent() -> None:
    with pytest.raises(StartupValidationError, match="queue_item_batches is absent"):
        _assert_triggers_live(_TriggerState(False, {}), schema="public")


def test_lists_every_missing_trigger_and_points_at_the_migration() -> None:
    triggers = _live_triggers()
    del triggers["trg_queue_item_batches_update"]
    del triggers["trg_queue_item_batches_delete"]
    with pytest.raises(StartupValidationError) as excinfo:
        _assert_triggers_live(_TriggerState(True, triggers), schema="public")
    message = str(excinfo.value)
    assert "trg_queue_item_batches_update" in message
    assert "trg_queue_item_batches_delete" in message
    assert "trg_queue_item_batches_insert" not in message
    assert "alembic upgrade head" in message


def test_raises_on_a_disabled_trigger() -> None:
    triggers = _live_triggers()
    triggers["trg_queue_item_batches_delete"] = "D"
    with pytest.raises(StartupValidationError, match="not live") as excinfo:
        _assert_triggers_live(_TriggerState(True, triggers), schema="public")
    assert "trg_queue_item_batches_delete (tgenabled='D')" in str(excinfo.value)
    assert "ENABLE TRIGGER" in str(excinfo.value)


def test_raises_on_a_replica_only_trigger() -> None:
    """'R' fires only under replica mode — dead on a primary, so it must fail."""
    triggers = _live_triggers()
    triggers["trg_queue_item_batches_insert"] = "R"
    with pytest.raises(StartupValidationError, match="tgenabled='R'"):
        _assert_triggers_live(_TriggerState(True, triggers), schema="public")


def test_messages_name_the_configured_schema() -> None:
    with pytest.raises(StartupValidationError, match=r"tenant_x\.queue_item_batches"):
        _assert_triggers_live(_TriggerState(False, {}), schema="tenant_x")
    with pytest.raises(StartupValidationError, match=r"on tenant_x\.queue"):
        _assert_triggers_live(_TriggerState(True, {}), schema="tenant_x")


# ---------------------------------------------------------------------------
# Retry / fail-closed behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fails_closed_when_introspection_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the retry budget exhausts, the validator crashes — uncertainty
    is not a green light to serve traffic."""
    call_count = 0

    async def always_raise(_engine: AsyncEngine, _schema: str) -> _TriggerState:
        nonlocal call_count
        call_count += 1
        raise OperationalError("SELECT 1", {}, Exception("DB unreachable"))

    monkeypatch.setattr(
        "src.startup.queue_item_batches_validator._introspect_triggers_once",
        always_raise,
    )
    monkeypatch.setattr(
        "src.startup.queue_item_batches_validator._RETRY_BACKOFF_SECONDS", 0.0
    )

    with pytest.raises(StartupValidationError, match="could not validate"):
        await validate_queue_item_batches(AsyncMock())

    assert call_count == 3, "should exhaust the retry budget before failing"


# ---------------------------------------------------------------------------
# Integration: real test DB (migrated, so the triggers are installed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_passes_against_test_database(
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # conftest provisions the test tables in `public`; pin the validator to it
    # so a developer's local .env DB_SCHEMA can't point it at another schema.
    monkeypatch.setattr(settings.DB, "SCHEMA", "public")
    await validate_queue_item_batches(db_engine)


@pytest.mark.asyncio
async def test_validator_raises_while_a_trigger_is_disabled(
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DISABLE one trigger the way an operator might, confirm boot would be
    refused with the trigger named, then restore it."""
    monkeypatch.setattr(settings.DB, "SCHEMA", "public")
    async with db_engine.begin() as conn:
        await conn.execute(
            text("ALTER TABLE queue DISABLE TRIGGER trg_queue_item_batches_update")
        )
    try:
        with pytest.raises(
            StartupValidationError,
            match="trg_queue_item_batches_update \\(tgenabled='D'\\)",
        ):
            await validate_queue_item_batches(db_engine)
    finally:
        async with db_engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE queue ENABLE TRIGGER trg_queue_item_batches_update")
            )
    # Restored: the validator accepts the DB again.
    await validate_queue_item_batches(db_engine)
