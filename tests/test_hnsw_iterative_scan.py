"""Tests for the pgvector HNSW iterative scan connection setting.

Verifies that:
- ``DBSettings.HNSW_ITERATIVE_SCAN`` accepts valid enum values and ``None``
- An invalid value is rejected at config-load time (fail-closed)
- The ``connect`` event listener is registered when the setting is enabled
- The ``connect`` event listener is NOT registered when the setting is ``None``
"""

import pytest

from src.config import DBSettings


def test_hnsw_iterative_scan_defaults_to_strict_order() -> None:
    settings = DBSettings()
    assert settings.HNSW_ITERATIVE_SCAN == "strict_order"


def test_hnsw_iterative_scan_accepts_valid_values() -> None:
    for value in ("off", "strict_order", "relaxed_order"):
        settings = DBSettings(HNSW_ITERATIVE_SCAN=value)
        assert value == settings.HNSW_ITERATIVE_SCAN


def test_hnsw_iterative_scan_accepts_none() -> None:
    settings = DBSettings(HNSW_ITERATIVE_SCAN=None)
    assert settings.HNSW_ITERATIVE_SCAN is None


def test_hnsw_iterative_scan_rejects_invalid_value() -> None:
    with pytest.raises((ValueError, TypeError)):
        DBSettings(
            HNSW_ITERATIVE_SCAN="on"  # pyright: ignore[reportArgumentType]
        )


def test_hnsw_iterative_scan_rejects_arbitrary_string() -> None:
    with pytest.raises((ValueError, TypeError)):
        DBSettings(
            HNSW_ITERATIVE_SCAN="DROP TABLE users; --"  # pyright: ignore[reportArgumentType]
        )


def test_connect_listener_registered_when_enabled() -> None:
    """The connect event listener is attached to the engine when
    HNSW_ITERATIVE_SCAN is set.

    Uses ``sqlalchemy.event.contains`` to verify the listener is actually
    registered with the engine's event system, not just that the function
    exists and is callable.
    """

    from sqlalchemy import event

    from src import db as db_module

    # The listener is registered at import time when the setting is
    # truthy (the default is "strict_order").  We verify registration via
    # the SQLAlchemy event registry rather than just checking callability.
    assert event.contains(
        db_module.engine.sync_engine,
        "connect",
        # pyright: ignore[reportPrivateUsage]
        db_module._set_hnsw_iterative_scan_on_connect,
    ), "HNSW iterative scan connect listener should be registered on the engine"


def test_connect_listener_not_registered_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When HNSW_ITERATIVE_SCAN is None, the connect function
    short-circuits early without executing any SQL.

    This tests the early-return guard in
    ``_set_hnsw_iterative_scan_on_connect`` rather than event
    registration, because the listener is registered at import time
    based on the initial config value and is not dynamically
    removed when the setting changes at runtime.
    """

    from src.config import settings

    monkeypatch.setattr(settings.DB, "HNSW_ITERATIVE_SCAN", None)
    assert settings.DB.HNSW_ITERATIVE_SCAN is None

    import types

    from src import db as db_module

    execute_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class AssertingCursor:
        def execute(self, *args: object, **kwargs: object) -> None:
            execute_calls.append((args, kwargs))
            raise AssertionError(
                "execute should not be called when HNSW_ITERATIVE_SCAN is None"
            )

        def close(self) -> None:
            pass

    dummy_conn = types.SimpleNamespace(
        autocommit=False, cursor=lambda: AssertingCursor()
    )
    # The function reads settings.DB.HNSW_ITERATIVE_SCAN at call time,
    # so with monkeypatch it should return early without executing SQL.
    db_module._set_hnsw_iterative_scan_on_connect(dummy_conn, None)  # type: ignore[attr-defined]
    assert execute_calls == [], "No SQL should execute when HNSW_ITERATIVE_SCAN is None"


def test_version_gate_skipped_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pgvector version check should NOT run when
    HNSW_ITERATIVE_SCAN is 'off'. Only 'strict_order' and
    'relaxed_order' require pgvector >= 0.8.0.

    Uses a fake engine/connection to assert that the pg_extension
    query is never executed when the setting is 'off'.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.config import settings
    from src import db as db_module

    monkeypatch.setattr(settings.DB, "HNSW_ITERATIVE_SCAN", "off")
    assert settings.DB.HNSW_ITERATIVE_SCAN == "off"

    execute_calls: list[str] = []

    class FakeResult:
        def fetchone(self) -> None:
            return None

    async def fake_execute(stmt: object, *args: object, **kwargs: object) -> FakeResult:
        execute_calls.append(str(stmt))
        return FakeResult()

    fake_conn = MagicMock()
    fake_conn.execute = fake_execute
    fake_conn.commit = AsyncMock()

    class FakeAsyncCM:
        async def __aenter__(self) -> MagicMock:
            return fake_conn

        async def __aexit__(self, *args: object) -> None:
            pass

    fake_engine = MagicMock()
    fake_engine.connect = MagicMock(return_value=FakeAsyncCM())

    with patch.object(db_module, "engine", fake_engine), \
         patch("alembic.command.upgrade"), \
         patch("alembic.config.Config"):
        import asyncio
        asyncio.run(db_module.init_db())

    pg_extension_calls = [c for c in execute_calls if "pg_extension" in c]
    assert pg_extension_calls == [], (
        "pg_extension version query should not execute when HNSW_ITERATIVE_SCAN is 'off'"
    )
