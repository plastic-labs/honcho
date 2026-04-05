"""Sentry initialization and configuration."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import sentry_sdk

from src.config import settings

P = ParamSpec("P")
T = TypeVar("T")

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
        before_send: Optional event filter callback to suppress specific exceptions.
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


def with_sentry_transaction(
    name: str, op: str
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to wrap a function in a Sentry transaction.

    Args:
        name: The name of the transaction.
        op: The operation type (e.g., "deriver", "http.server").

    Returns:
        A decorator that wraps the target function in a Sentry transaction.

    Example:
        @with_sentry_transaction("process_task", op="deriver")
        def process_task(data):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with sentry_sdk.start_transaction(name=name, op=op):
                    return await func(*args, **kwargs)

            return cast(Callable[P, T], async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with sentry_sdk.start_transaction(name=name, op=op):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
