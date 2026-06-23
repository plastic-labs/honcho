"""Sentry initialization and configuration."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import sentry_sdk

from src.config import settings

P = ParamSpec("P")
T = TypeVar("T")

if TYPE_CHECKING:
    from sentry_sdk._types import EventProcessor
    from sentry_sdk.integrations import Integration

logger = logging.getLogger(__name__)


# Paths whose transactions carry no debugging value but are hit constantly
# (health checks, Prometheus scrapes, OpenAPI schema, docs). Tracing them at the
# same rate as real traffic drowns the signal and burns tracing/profiling quota.
# Note: /docs and /redoc are disabled in production but are listed for safety.
_UNSAMPLED_PATHS = frozenset(
    {"/metrics", "/health", "/openapi.json", "/docs", "/redoc"}
)


def _is_unsampled_transaction_name(name: str | None) -> bool:
    """Match infra/scrape transactions by name.

    Fallback for transactions that don't expose an ASGI scope path (e.g. the
    deriver's metrics server) or whose endpoint-style name encodes the route.
    """
    if not name:
        return False
    return (
        name.endswith("openapi")
        or name.endswith("metrics_endpoint")
        or "prometheus.metrics" in name
    )


def traces_sampler(sampling_context: dict[str, Any]) -> float:
    """Drop infra/scrape transactions; sample everything else at the default rate.

    Using a sampler (rather than ``before_send_transaction``) means dropped
    transactions are never recorded or profiled, and the decision propagates to
    child spans. ``SENTRY.TRACES_SAMPLE_RATE`` remains the rate for real traffic.
    """
    asgi_scope = cast("dict[str, Any] | None", sampling_context.get("asgi_scope"))
    if asgi_scope is not None and asgi_scope.get("path") in _UNSAMPLED_PATHS:
        return 0.0

    transaction_context = cast(
        "dict[str, Any] | None", sampling_context.get("transaction_context")
    )
    name = transaction_context.get("name") if transaction_context else None
    if _is_unsampled_transaction_name(name if isinstance(name, str) else None):
        return 0.0

    # Respect an upstream sampling decision when continuing a distributed trace.
    parent_sampled = sampling_context.get("parent_sampled")
    if parent_sampled is not None:
        return float(parent_sampled)

    return settings.SENTRY.TRACES_SAMPLE_RATE


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
        # traces_sampler supersedes traces_sample_rate; it returns the configured
        # rate for real traffic and 0.0 for infra/scrape endpoints (see above).
        traces_sampler=traces_sampler,
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
