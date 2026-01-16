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
