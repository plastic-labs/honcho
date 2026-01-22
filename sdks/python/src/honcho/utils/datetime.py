"""DateTime utilities for the Honcho Python SDK."""

from datetime import datetime


def datetime_to_iso(value: datetime | str | None) -> str | None:
    """
    Convert a datetime value to an ISO 8601 formatted string.

    Args:
        value: A datetime object, an ISO 8601 string, or None

    Returns:
        An ISO 8601 formatted string, or None if input is None
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def parse_datetime(value: datetime | str | None) -> datetime | None:
    """
    Parse an ISO 8601 datetime string into a `datetime` instance.

    This accepts timestamps in the forms commonly produced by APIs and JS runtimes,
    including a trailing "Z" UTC designator (e.g. "2024-01-15T10:30:00Z") and
    offsets (e.g. "2024-01-15T10:30:00+00:00").

    Args:
        value: A datetime object, an ISO 8601 string, or None.

    Returns:
        A `datetime` instance if provided, otherwise None.

    Raises:
        ValueError: If the string cannot be parsed as ISO 8601.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime: {value!r}") from e
