"""Sentry initialization and configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import sentry_sdk
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.config import settings
from src.exceptions import HonchoException

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint
    from sentry_sdk.integrations import Integration


def _filter_sentry_event(event: Event, hint: Hint | None) -> Event | None:
    """Filter out events raised from known non-actionable exceptions before Sentry sees them."""
    if not hint:
        return event

    exc_info = hint.get("exc_info")
    if not exc_info:
        return event

    _, exc_value, _ = exc_info
    if isinstance(
        exc_value, HonchoException | ValidationError | RequestValidationError
    ):  # Filters out HonchoExceptions and ValidationErrors (typically coming from Pydantic)
        return None

    return event


# Sentry SDK's default behavior:
# - Captures INFO+ level logs as breadcrumbs
# - Captures ERROR+ level logs as Sentry events
#
# For custom log levels, use the LoggingIntegration class:
# sentry_sdk.init(..., integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)])
def initialize_sentry(
    *,
    integrations: Sequence[Integration],
) -> None:
    """Initialize Sentry SDK with project settings.

    Args:
        integrations: Sentry SDK integrations to enable (e.g., Starlette, FastAPI).
    """
    sentry_sdk.init(
        dsn=settings.SENTRY.DSN,
        enable_tracing=True,
        release=settings.SENTRY.RELEASE,
        environment=settings.SENTRY.ENVIRONMENT,
        traces_sample_rate=settings.SENTRY.TRACES_SAMPLE_RATE,
        profiles_sample_rate=settings.SENTRY.PROFILES_SAMPLE_RATE,
        before_send=_filter_sentry_event,
        integrations=integrations,
    )
