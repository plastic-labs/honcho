import asyncio
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, cast

import sentry_sdk

P = ParamSpec("P")
T = TypeVar("T")


def with_sentry_transaction(name: str, op: str):
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
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
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
