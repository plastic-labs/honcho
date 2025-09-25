from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import sentry_sdk

P = ParamSpec("P")
T = TypeVar("T")


def with_sentry_transaction(name: str, op: str):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with sentry_sdk.start_transaction(name=name, op=op):
                return func(*args, **kwargs)

        return wrapper

    return decorator
