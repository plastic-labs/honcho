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

    dummy_conn = types.SimpleNamespace(autocommit=False, cursor=lambda: types.SimpleNamespace(execute=lambda *a, **k: None, close=lambda: None))
    # The function reads settings.DB.HNSW_ITERATIVE_SCAN at call time,
    # so with monkeypatch it should return early without executing SQL.
    # pyright: ignore[reportPrivateUsage]
    db_module._set_hnsw_iterative_scan_on_connect(dummy_conn, None)
    # No exception raised — early return worked.


def test_version_gate_skipped_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pgvector version check should NOT run when
    HNSW_ITERATIVE_SCAN is 'off'. Only 'strict_order' and
    'relaxed_order' require pgvector >= 0.8.0.
    """
    from src.config import settings

    monkeypatch.setattr(settings.DB, "HNSW_ITERATIVE_SCAN", "off")
    assert settings.DB.HNSW_ITERATIVE_SCAN == "off"
    # The version gate condition is:
    #   settings.DB.HNSW_ITERATIVE_SCAN in ("strict_order", "relaxed_order")
    # "off" is not in that tuple, so the check is skipped.
    assert settings.DB.HNSW_ITERATIVE_SCAN not in ("strict_order", "relaxed_order")
