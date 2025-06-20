from collections.abc import Callable
from typing import ParamSpec, TypeVar

from sentry_sdk.ai.monitoring import ai_track

R = TypeVar("R")
P = ParamSpec("P")


def track(description: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        def _inner(*args: P.args, **kwargs: P.kwargs) -> R:
            result: R = ai_track(description)(f)(*args, **kwargs)  # pyright: ignore
            return result

        return _inner

    return decorator
