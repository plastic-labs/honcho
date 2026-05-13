"""Single source of truth for the Honcho service version.

Used by the telemetry emitter to tag every event body with `honcho_version`,
and available for any other component (e.g. /info endpoints, log lines)
that needs the version at runtime.
"""

from functools import cache
from importlib.metadata import PackageNotFoundError, version


@cache
def honcho_version() -> str | None:
    """Resolve the installed Honcho package version, or None if unknown.

    Cached for the process lifetime. The version doesn't change at runtime,
    so the first call pays the lookup cost and every later call is free.
    """
    try:
        return version("honcho")
    except PackageNotFoundError:
        return None
