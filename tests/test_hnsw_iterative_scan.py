"""Tests for the pgvector HNSW iterative scan connection setting.

Verifies that:
- ``DBSettings.HNSW_ITERATIVE_SCAN`` accepts valid enum values and ``None``
- An invalid value is rejected at config-load time (fail-closed)
- The ``connect`` event listener is registered when the setting is enabled
- The ``connect`` event listener is NOT registered when the setting is ``None``
- ``_validate_pgvector_version`` raises for pgvector < 0.8.0 and passes for >= 0.8.0
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
    bad: str = "DROP TABLE users; --"
    with pytest.raises((ValueError, TypeError)):
        DBSettings(HNSW_ITERATIVE_SCAN=bad)  # pyright: ignore[reportArgumentType]


def test_connect_listener_registered_when_enabled() -> None:
    """The connect event listener is attached to the engine when
    HNSW_ITERATIVE_SCAN is set.

    Uses ``sqlalchemy.event.contains`` to verify the listener is actually
    registered with the engine's event system, not just that the function
    exists and is callable.
    """

    from sqlalchemy import event

    from src import db as db_module

    _listener = db_module._set_hnsw_iterative_scan_on_connect  # pyright: ignore[reportPrivateUsage]

    # The listener is registered at import time when the setting is
    # truthy (the default is "strict_order").  We verify registration via
    # the SQLAlchemy event registry rather than just checking callability.
    assert event.contains(
        db_module.engine.sync_engine,
        "connect",
        _listener,
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

    _listener = db_module._set_hnsw_iterative_scan_on_connect  # pyright: ignore[reportPrivateUsage]

    execute_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class AssertingCursor:
        def execute(self, *args: object, **_kwargs: object) -> None:
            execute_calls.append((args, _kwargs))
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
    _listener(dummy_conn, None)
    assert execute_calls == [], "No SQL should execute when HNSW_ITERATIVE_SCAN is None"


def test_validate_pgvector_version_rejects_old_versions() -> None:
    """_validate_pgvector_version raises RuntimeError for pgvector < 0.8.0."""
    from src.db import _validate_pgvector_version  # pyright: ignore[reportPrivateUsage]

    for old_version in ("0.7.0", "0.6.1", "0.5.0"):
        with pytest.raises(RuntimeError, match=r"requires pgvector >= 0\.8\.0"):
            _validate_pgvector_version(old_version)


def test_validate_pgvector_version_accepts_new_versions() -> None:
    """_validate_pgvector_version passes silently for pgvector >= 0.8.0."""
    from src.db import _validate_pgvector_version  # pyright: ignore[reportPrivateUsage]

    for new_version in ("0.8.0", "0.8.1", "0.9.0", "1.0.0"):
        _validate_pgvector_version(new_version)  # should not raise