"""
Shared formatting utility functions for both dialectic and deriver modules.

This module contains helper functions for processing observations, formatting context,
and handling temporal metadata for the reasoning system.
"""

from datetime import datetime, timezone


def format_datetime_utc(dt: datetime) -> str:
    """
    Format datetime to ISO 8601 string with Z suffix for UTC timezone.

    This ensures consistent datetime formatting across the entire backend,
    using the Z format which is the ISO 8601 standard for UTC and matches
    Pydantic's JSON serialization behavior.

    Args:
        dt: datetime object (should be timezone-aware)

    Returns:
        ISO 8601 formatted string with Z suffix for UTC

    Example:
        >>> dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> format_datetime_utc(dt)
        '2023-01-01T12:00:00Z'
    """
    if dt.tzinfo is None:
        # If no timezone info, assume UTC
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to UTC if not already
    if dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    # Remove subsecond precision
    dt = dt.replace(microsecond=0)

    # Format and replace +00:00 with Z
    return dt.isoformat().replace("+00:00", "Z")


def utc_now_iso() -> str:
    """
    Get current UTC time as ISO 8601 string with Z suffix.
    Removes subsecond precision.

    Returns:
        Current UTC time in ISO 8601 format with Z suffix

    Example:
        >>> utc_now_iso()
        '2023-01-01T12:34:56Z'
    """
    return format_datetime_utc(datetime.now(timezone.utc))


def parse_datetime_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 datetime string, handling various timezone formats.

    This function properly handles Z suffix, timezone offsets, and naive timestamps.
    It validates input and always returns a timezone-aware datetime object.

    Args:
        iso_string: ISO 8601 formatted datetime string

    Returns:
        datetime object with timezone information

    Raises:
        ValueError: If the input string is invalid or contains suspicious content

    Example:
        >>> parse_datetime_iso('2023-01-01T12:00:00Z')
        datetime.datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
        >>> parse_datetime_iso('2023-01-01T12:00:00+05:00')
        datetime.datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=18000)))
    """
    # Input validation - ensure type and reject suspicious content
    if not iso_string:
        raise ValueError("Invalid input: must be a non-empty string")

    iso_string = str(iso_string)

    # Security check - reject strings with null bytes or suspicious characters
    if "\x00" in iso_string or "\r" in iso_string or "\n" in iso_string:
        raise ValueError("Invalid input: contains null bytes or line breaks")

    # Check for non-printable unicode characters that could be used for attacks
    if any(ord(c) < 32 and c not in "\t" for c in iso_string):
        raise ValueError("Invalid input: contains non-printable characters")

    # Strip whitespace
    iso_string = iso_string.strip()
    if not iso_string:
        raise ValueError("Invalid input: empty after stripping whitespace")

    # Handle Z suffix (convert to +00:00)
    if iso_string.endswith(("Z", "z")):
        iso_string = iso_string[:-1] + "+00:00"

    try:
        # Try parsing with timezone info first
        result = datetime.fromisoformat(iso_string)

        # If no timezone info, assume UTC
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)

        return result
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime format: {e}") from e


def format_new_turn_with_timestamp(
    new_turn: str, current_time: datetime, speaker: str
) -> str:
    """
    Format new turn message with optional timestamp.

    Args:
        new_turn: The message content
        current_time: Message timestamp
        speaker: The speaker's name

    Returns:
        Formatted string like "2023-05-08 13:56:00 speaker: hello"
    """
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{current_time_str} {speaker}: {new_turn}"
