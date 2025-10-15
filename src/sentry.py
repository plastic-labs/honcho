"""Sentry initialization and configuration."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import sentry_sdk

from src.config import settings

if TYPE_CHECKING:
    from sentry_sdk._types import EventProcessor
    from sentry_sdk.integrations import Integration

logger = logging.getLogger(__name__)


# Sentry SDK's default behavior:
# - Captures INFO+ level logs as breadcrumbs
# - Captures ERROR+ level logs as Sentry events
#
# For custom log levels, use the LoggingIntegration class:
# sentry_sdk.init(..., integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)])
def initialize_sentry(
    *,
    integrations: Sequence[Integration],
    before_send: EventProcessor | None = None,
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
        before_send=before_send,
        integrations=integrations,
    )
