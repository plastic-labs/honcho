"""DB-free unit tests for src/utils/retryable_errors.py."""

import asyncio
from typing import cast

import httpx
import pytest
from sqlalchemy.exc import DBAPIError, OperationalError

from src.utils.retryable_errors import is_retryable_db_error, is_retryable_error


class FakePGError(Exception):
    """Stands in for a driver exception carrying a SQLSTATE."""

    sqlstate: str | None

    def __init__(self, sqlstate: str | None) -> None:
        super().__init__(f"fake pg error ({sqlstate})")
        self.sqlstate = sqlstate


def _dbapi_error(
    sqlstate: str | None,
    *,
    orig: BaseException | None = None,
    connection_invalidated: bool = False,
) -> DBAPIError:
    if orig is None and sqlstate is not None:
        orig = FakePGError(sqlstate)
    return OperationalError(
        "SELECT 1",
        {},
        cast(BaseException, orig),
        connection_invalidated=connection_invalidated,
    )


@pytest.mark.parametrize(
    ("sqlstate", "expected"),
    [
        ("40P01", True),  # deadlock_detected
        ("40001", True),  # serialization_failure
        ("55P03", True),  # lock_not_available
        ("57014", True),  # query_canceled
        ("08006", True),  # connection_failure
        ("23505", False),  # unique_violation
        ("42P01", False),  # undefined_table
        ("22P02", False),  # invalid_text_representation
    ],
)
def test_sqlstate_classification(sqlstate: str, expected: bool):
    exc = _dbapi_error(sqlstate)
    assert is_retryable_db_error(exc) is expected
    assert is_retryable_error(exc) is expected


def test_orig_none_is_terminal():
    assert not is_retryable_db_error(_dbapi_error(None))


def test_sqlstate_on_orig_cause():
    """SQLSTATE found by walking orig.__cause__ when orig itself has none."""
    wrapper = Exception("driver wrapper")
    wrapper.__cause__ = FakePGError("40P01")
    assert is_retryable_db_error(_dbapi_error(None, orig=wrapper))


def test_connection_invalidated_is_retryable():
    exc = _dbapi_error(None, connection_invalidated=True)
    assert is_retryable_db_error(exc)


def test_dbapi_error_nested_in_cause_chain():
    outer = RuntimeError("save failed")
    outer.__cause__ = _dbapi_error("40P01")
    assert is_retryable_db_error(outer)
    assert is_retryable_error(outer)


def test_non_db_exceptions_are_not_db_retryable():
    assert not is_retryable_db_error(ValueError("bad input"))
    assert not is_retryable_db_error(httpx.ConnectTimeout("timed out"))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectTimeout("timed out"), True),
        (httpx.ReadTimeout("timed out"), True),
        (httpx.ConnectError("connection refused"), True),
        (ConnectionResetError("reset"), True),
        (asyncio.TimeoutError(), True),
        (TimeoutError(), True),
        (ValueError("bad input"), False),
        (httpx.HTTPStatusError("401", request=None, response=None), False),  # pyright: ignore[reportArgumentType]
    ],
)
def test_transport_classification(exc: BaseException, expected: bool):
    assert is_retryable_error(exc) is expected
    assert not is_retryable_db_error(exc)


def test_transport_error_nested_in_cause_chain():
    """SDK wrappers (e.g. APIConnectionError) chain to httpx via __cause__."""
    wrapper = RuntimeError("provider call failed")
    wrapper.__cause__ = httpx.ConnectError("connection refused")
    assert is_retryable_error(wrapper)
    assert not is_retryable_db_error(wrapper)


def test_cause_cycle_terminates():
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert not is_retryable_error(a)
