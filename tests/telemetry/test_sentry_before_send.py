"""Tests for the shared Sentry before_send filter.

default_before_send runs in every entrypoint (API + deriver). It drops known
non-actionable exceptions and collapses two fleet-wide symptoms -- DB
connection-pool checkout timeouts and upstream model provider outages -- into a
single issue each, so they stop spawning a fresh one per transaction.
"""

from typing import TYPE_CHECKING, cast, final

import pytest
import sentry_sdk
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from src.exceptions import (
    HonchoException,
    ResourceNotFoundException,
    SentryPolicy,
    UpstreamLLMError,
)
from src.telemetry.sentry import default_before_send, initialize_sentry

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


def test_upstream_llm_error_is_consolidated_at_error_level() -> None:
    """The one HonchoException that survives the drop.

    Provider outages are the deriver's only signal that the upstream is down --
    it has no status-code metric -- so they stay in Sentry. They stay at error
    level too: the retry budget and fallback chain are spent by the time one
    escapes. The fingerprint, not the level, is what keeps the volume down.
    """
    out = default_before_send({}, _hint(UpstreamLLMError("the provider is down")))
    assert out == {
        "fingerprint": ["honcho-upstream-llm-unavailable"],
        "level": "error",
    }


def test_policy_is_read_from_the_exception_not_matched_on_by_type() -> None:
    """A subclass opts into visibility without being named in the filter.

    This is the property that makes the blanket drop safe to keep last: the
    filter never has to be edited, so there is no branch ordering to get wrong.
    """

    @final
    class KeptException(HonchoException):
        sentry_policy = SentryPolicy(fingerprint="honcho-kept", level="error")

    assert default_before_send({}, _hint(KeptException("kept"))) == {
        "fingerprint": ["honcho-kept"],
        "level": "error",
    }


def test_policy_without_a_fingerprint_keeps_sentrys_own_grouping() -> None:
    """Levelling and regrouping are independent knobs."""

    @final
    class UngroupedException(HonchoException):
        sentry_policy = SentryPolicy(level="info")

    assert default_before_send(_event(release="1.0"), _hint(UngroupedException())) == {
        "release": "1.0",
        "level": "info",
    }


def test_a_new_exception_defaults_to_being_dropped() -> None:
    """Declaring nothing inherits the base policy: client-facing, so not a bug report."""

    @final
    class PlainException(HonchoException):
        pass

    assert PlainException.sentry_policy is None
    assert default_before_send({}, _hint(PlainException("plain"))) is None


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


def _captured_before_send(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> object:
    captured: dict[str, object] = {}

    def fake_init(**init_kwargs: object) -> None:
        captured.update(init_kwargs)

    monkeypatch.setattr(sentry_sdk, "init", fake_init)
    initialize_sentry(integrations=[], **kwargs)  # pyright: ignore[reportArgumentType]
    return captured["before_send"]


def test_initialize_sentry_defaults_to_shared_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _captured_before_send(monkeypatch) is default_before_send


def test_initialize_sentry_explicit_none_bypasses_shared_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _captured_before_send(monkeypatch, before_send=None) is None


def _llm_integration_event(mechanism_type: str) -> "Event":
    return _event(
        level="error",
        exception={
            "values": [
                {
                    "type": "ServerError",
                    "value": "The service is currently unavailable.",
                    "mechanism": {"type": mechanism_type, "handled": False},
                }
            ]
        },
    )


@pytest.mark.parametrize("mechanism_type", ["google_genai", "openai", "anthropic"])
def test_llm_sdk_integration_per_attempt_errors_are_dropped(
    mechanism_type: str,
) -> None:
    exc = RuntimeError("The service is currently unavailable.")
    event = _llm_integration_event(mechanism_type)
    assert default_before_send(event, _hint(exc)) is None


def test_other_mechanisms_pass_through() -> None:
    exc = RuntimeError("boom")
    event = _llm_integration_event("starlette")
    assert default_before_send(event, _hint(exc)) is event
