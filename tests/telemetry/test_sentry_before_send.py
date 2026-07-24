"""Tests for the shared Sentry before_send filter.

default_before_send runs in every entrypoint (API + deriver). It drops known
non-actionable exceptions and collapses DB connection-pool checkout timeouts
into a single warning-level issue so they stop spawning a fresh error issue per
transaction (fleet-wide saturation symptom, tracked in DEV-1852).
"""

from typing import TYPE_CHECKING, cast

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from src.exceptions import ResourceNotFoundException
from src.telemetry.sentry import default_before_send

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint


def _hint(exc: BaseException) -> "Hint":
    return cast("Hint", {"exc_info": (type(exc), exc, None)})


def _event(**kwargs: object) -> "Event":
    return cast("Event", cast(object, dict(kwargs)))


def test_connection_timeout_is_consolidated_and_downgraded() -> None:
    exc = OperationalError("SELECT 1", {}, Exception("connection timeout expired"))
    out = default_before_send({}, _hint(exc))
    assert out == {
        "fingerprint": ["honcho-db-connection-timeout"],
        "level": "warning",
    }


def test_unrelated_operational_error_passes_through() -> None:
    exc = OperationalError("SELECT 1", {}, Exception("some other db failure"))
    event = _event(level="error")
    assert default_before_send(event, _hint(exc)) == {"level": "error"}


def test_honcho_and_validation_errors_are_dropped() -> None:
    assert default_before_send({}, _hint(ResourceNotFoundException("nope"))) is None
    assert (
        default_before_send({}, _hint(ValidationError.from_exception_data("x", [])))
        is None
    )
    assert default_before_send({}, _hint(RequestValidationError([]))) is None


def test_events_without_exc_info_pass_through() -> None:
    event = _event(release="1.0")
    assert default_before_send(event, None) == {"release": "1.0"}
    assert default_before_send(event, cast("Hint", {})) == {"release": "1.0"}
