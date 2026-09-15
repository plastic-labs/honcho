"""Startup-validator tests and their shared helpers."""

from typing import cast

from sqlalchemy.ext.asyncio import AsyncEngine


class _UntouchableEngine:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"validator touched the engine ({name!r}) when it should have been a no-op"
        )


def untouchable_engine() -> AsyncEngine:
    """An engine stand-in for validators that must not reach the database.

    Any attribute access — ``connect``, ``begin``, anything — fails the test, so a
    validator that is supposed to be a no-op is proven not to open a connection.
    Typed as ``AsyncEngine`` so call sites read like the real thing; the object
    shares no structure with it, hence the cast through ``object``.
    """
    return cast(AsyncEngine, cast(object, _UntouchableEngine()))
