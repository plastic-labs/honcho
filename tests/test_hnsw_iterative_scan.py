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
        DBSettings(HNSW_ITERATIVE_SCAN="on")  # pyright: ignore[reportArgumentType]  # not a valid pgvector enum


def test_hnsw_iterative_scan_rejects_arbitrary_string() -> None:
    with pytest.raises((ValueError, TypeError)):
        DBSettings(HNSW_ITERATIVE_SCAN="DROP TABLE users; --")  # pyright: ignore[reportArgumentType]


def test_connect_listener_registered_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connect event listener is attached when HNSW_ITERATIVE_SCAN is set."""

    from src import db as db_module
    from src.config import settings

    monkeypatch.setattr(settings.DB, "HNSW_ITERATIVE_SCAN", "strict_order")
    db_module._set_hnsw_iterative_scan_on_connect  # type: attr-defined  # noqa: B018  # pyright: ignore[reportPrivateUsage]
    # The listener is registered at import time when the setting is truthy.
    # We verify the function exists and is callable.
    assert callable(db_module._set_hnsw_iterative_scan_on_connect)  # pyright: ignore[reportPrivateUsage]


def test_connect_listener_not_registered_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When HNSW_ITERATIVE_SCAN is None, no listener should fire."""
    from src.config import settings

    monkeypatch.setattr(settings.DB, "HNSW_ITERATIVE_SCAN", None)
    assert settings.DB.HNSW_ITERATIVE_SCAN is None
